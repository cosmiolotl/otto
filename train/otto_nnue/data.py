"""`sim datagen` data sets (crates/selfplay/src/datagen/format.rs) and a chunked, shuffled batch loader."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .layout import Layout, read_header

META = np.dtype(
    [
        ("offset", "<u8"),
        ("n_me", "u1"),
        ("n_opp", "u1"),
        ("bucket_me", "u1"),
        ("bucket_opp", "u1"),
        ("result", "i1"),
        ("mode", "u1"),
        ("ply", "<u2"),
        ("to_end", "<u2"),
        ("reserved", "<u2"),
        ("search_score", "<f4"),
        ("static_score", "<f4"),
        ("game", "<u4"),
    ]
)
assert META.itemsize == 32

# Every `VAL_EVERY`-th game is validation data, so positions of one game never straddle the split.
VAL_EVERY = 20


def is_val(meta: np.ndarray, val_every: int = VAL_EVERY) -> np.ndarray:
    return (meta["game"] % val_every) == 0


class DataSet:
    def __init__(self, stem: str | Path):
        self.stem = str(stem)
        self.header, self.layout = read_header(stem)
        self.meta = np.memmap(f"{stem}.meta", dtype=META, mode="r")
        self.feat = np.memmap(f"{stem}.feat", dtype="<u2", mode="r")
        assert len(self.meta) == self.header["samples"], "meta/header sample count mismatch"

    def __len__(self) -> int:
        return len(self.meta)

    def sample(self, i: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Sample `i`: (my features, the opponent's features, my bucket)."""
        m = self.meta[i]
        off, a, b = int(m["offset"]), int(m["n_me"]), int(m["n_opp"])
        return np.array(self.feat[off : off + a]), np.array(self.feat[off + a : off + a + b]), int(m["bucket_me"])


@dataclass
class Batch:
    me_idx: np.ndarray  # int64 flat feature indices
    me_off: np.ndarray  # int64 bag offsets
    opp_idx: np.ndarray
    opp_off: np.ndarray
    bucket: np.ndarray  # int64
    result: np.ndarray  # float32 in {0, 0.5, 1}
    static_score: np.ndarray  # float32
    search_score: np.ndarray

    def to(self, dev):
        """The same batch as torch tensors on `dev` (indices int64, the rest float32)."""
        import torch

        def t(name, a):
            dt = torch.long if name in ("me_idx", "me_off", "opp_idx", "opp_off", "bucket") else torch.float32
            return torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt, non_blocking=True)

        return Batch(**{f.name: t(f.name, getattr(self, f.name)) for f in fields(self)})


def _gather(feat: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lens = lens.astype(np.int64)
    off = np.zeros(len(lens), dtype=np.int64)
    np.cumsum(lens[:-1], out=off[1:])
    pos = np.repeat(starts.astype(np.int64) - off, lens) + np.arange(int(lens.sum()), dtype=np.int64)
    return feat[pos].astype(np.int64), off


def make_batch(feat: np.ndarray, m: np.ndarray, swap: np.ndarray | None = None) -> Batch:
    """Batch from meta records `m` whose `offset` indexes `feat`. `swap[i]` exchanges the perspectives."""
    me_start = m["offset"].astype(np.int64)
    opp_start = me_start + m["n_me"]
    me_len, opp_len = m["n_me"], m["n_opp"]
    bucket = m["bucket_me"].astype(np.int64)
    result = (m["result"].astype(np.float32) + 1.0) / 2.0
    if swap is not None and swap.any():
        me_start, opp_start = np.where(swap, opp_start, me_start), np.where(swap, me_start, opp_start)
        me_len, opp_len = np.where(swap, opp_len, me_len), np.where(swap, me_len, opp_len)
        bucket = np.where(swap, m["bucket_opp"], bucket).astype(np.int64)
        result = np.where(swap, 1.0 - result, result).astype(np.float32)
    me_idx, me_off = _gather(feat, me_start, me_len)
    opp_idx, opp_off = _gather(feat, opp_start, opp_len)
    return Batch(
        me_idx, me_off, opp_idx, opp_off, bucket, result,
        m["static_score"].astype(np.float32), m["search_score"].astype(np.float32),
    )


class Loader:
    """Iterates batches of the train or validation split ([`is_val`]) of several data sets.

    Each epoch visits contiguous chunks of `chunk` samples in random order (file-friendly), shuffles inside
    a buffer of `buffer_chunks` chunks, and yields batches.
    """

    def __init__(self, sets: list[DataSet], batch: int, split: str, chunk: int = 1 << 18, buffer_chunks: int = 8,
                 swap_prob: float = 0.0, seed: int = 0):
        self.sets, self.batch, self.split = sets, batch, split
        self.chunk, self.buffer_chunks = chunk, buffer_chunks
        self.swap_prob = swap_prob if split == "train" else 0.0
        self.rng = np.random.default_rng(seed)
        layouts = {s.layout.raw for s in sets}
        if len(layouts) != 1:
            raise ValueError("data sets have different feature layouts")
        self.layout: Layout = sets[0].layout

    def _keep(self, m: np.ndarray) -> np.ndarray:
        val = is_val(m)
        return val if self.split == "val" else ~val

    def chunks(self):
        out = [(si, a) for si, s in enumerate(self.sets) for a in range(0, len(s), self.chunk)]
        self.rng.shuffle(out)
        return out

    def __iter__(self):
        order = self.chunks()
        for i in range(0, len(order), self.buffer_chunks):
            parts = []
            for si, a in order[i : i + self.buffer_chunks]:
                s = self.sets[si]
                m = np.array(s.meta[a : a + self.chunk])
                m = m[self._keep(m)]
                if len(m) == 0:
                    continue
                lo = int(m["offset"][0])
                hi = int(m["offset"][-1]) + int(m["n_me"][-1]) + int(m["n_opp"][-1])
                feat = np.array(s.feat[lo:hi])
                m["offset"] -= lo
                parts.append((feat, m))
            if not parts:
                continue
            # Merge the buffer into one feature array.
            feats, metas, base = [], [], 0
            for feat, m in parts:
                m = m.copy()
                m["offset"] += base
                base += len(feat)
                feats.append(feat)
                metas.append(m)
            feat = np.concatenate(feats)
            meta = np.concatenate(metas)
            if self.split == "train":
                meta = meta[self.rng.permutation(len(meta))]
            for b in range(0, len(meta), self.batch):
                mb = meta[b : b + self.batch]
                if self.split == "train" and len(mb) < self.batch // 2:
                    continue
                swap = self.rng.random(len(mb)) < self.swap_prob if self.swap_prob > 0 else None
                yield make_batch(feat, mb, swap)

    def count(self) -> int:
        return int(sum(self._keep(np.array(s.meta)).sum() for s in self.sets))
