"""`sim datagen` data sets (crates/selfplay/src/datagen/format.rs) and a chunked, shuffled batch loader.

A data set holds decision records and, in format 2, sibling records: one-ply children of a decision that
directly follow it (`sib` = 1..K, bit 15 set on the child the bot played). The loader keeps each decision
and its siblings together (a *group*) in one batch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .layout import read_header

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
        ("sib", "<u2"),
        ("search_score", "<f4"),
        ("static_score", "<f4"),
        ("game", "<u4"),
    ]
)
assert META.itemsize == 32
DATA_FORMATS = (1, 2)  # format 1 has no siblings (its `sib` field is always 0)
CHOSEN = 1 << 15

# Every `VAL_EVERY`-th game is validation data, so positions of one game never straddle the split.
VAL_EVERY = 20


def is_val(meta: np.ndarray) -> np.ndarray:
    return (meta["game"] % VAL_EVERY) == 0


def is_root(meta: np.ndarray) -> np.ndarray:
    """Decision records (the rest are siblings)."""
    return meta["sib"] == 0


def outcome(meta: np.ndarray) -> np.ndarray:
    """The side to move's result as a probability target: 1 win, 0.5 draw, 0 loss (float32)."""
    return (meta["result"].astype(np.float32) + 1.0) / 2.0


class DataSet:
    def __init__(self, stem: str | Path):
        self.header, self.layout = read_header(stem)
        h = self.header
        if h["format"] not in DATA_FORMATS or h["meta_bytes"] != META.itemsize:
            raise ValueError(f"{stem}: data format {h['format']} with {h['meta_bytes']}-byte records, "
                             f"trainer reads formats {DATA_FORMATS} with {META.itemsize}-byte records")
        self.siblings = int(h.get("settings", {}).get("siblings", 0))
        self.meta = np.memmap(f"{stem}.meta", dtype=META, mode="r")
        self.feat = np.memmap(f"{stem}.feat", dtype="<u2", mode="r")
        if (len(self.meta), len(self.feat)) != (h["samples"], h["features"]):
            raise ValueError(f"{stem}: files hold {len(self.meta)} samples / {len(self.feat)} features, "
                             f"header says {h['samples']} / {h['features']}")

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
    root: np.ndarray  # int64 positions of the decision records (value loss)
    grp: np.ndarray  # int64 per sibling: its group, 0..G-1
    pos: np.ndarray  # int64 per sibling: its position in the group, 0..K-1
    sib_idx: np.ndarray  # int64 per sibling: its position in the batch
    chosen: np.ndarray  # float32 per sibling: 1 for the bot's move

    def to(self, dev):
        """The same batch as torch tensors on `dev` (indices int64, the rest float32)."""
        import torch

        def t(name, a):
            dt = torch.float32 if a.dtype.kind == "f" else torch.long
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
    result = outcome(m)
    root = is_root(m)
    if swap is not None:
        swap = swap & root  # a sibling's perspective is the search's: never swapped
    if swap is not None and swap.any():
        me_start, opp_start = np.where(swap, opp_start, me_start), np.where(swap, me_start, opp_start)
        me_len, opp_len = np.where(swap, opp_len, me_len), np.where(swap, me_len, opp_len)
        bucket = np.where(swap, m["bucket_opp"], bucket).astype(np.int64)
        result = np.where(swap, 1.0 - result, result).astype(np.float32)
    me_idx, me_off = _gather(feat, me_start, me_len)
    opp_idx, opp_off = _gather(feat, opp_start, opp_len)
    sib = ~root
    gid = np.cumsum(root) - 1
    if sib.any() and gid[np.flatnonzero(sib)[0]] < 0:
        raise ValueError("a batch starts with a sibling record (groups must stay whole)")
    _, grp = np.unique(gid[sib], return_inverse=True)
    return Batch(
        me_idx, me_off, opp_idx, opp_off, bucket, result,
        m["static_score"].astype(np.float32), m["search_score"].astype(np.float32),
        np.flatnonzero(root).astype(np.int64), grp.astype(np.int64).reshape(-1),
        ((m["sib"][sib] & ~np.uint16(CHOSEN)).astype(np.int64) - 1), np.flatnonzero(sib).astype(np.int64),
        ((m["sib"][sib] & CHOSEN) != 0).astype(np.float32),
    )


class Loader:
    """Iterates batches of the train or validation split ([`is_val`]) of several data sets.

    Each epoch visits contiguous chunks of about `chunk` samples in random order (file-friendly), shuffles
    the groups inside a buffer of `buffer_chunks` chunks, and yields batches of about `batch` records cut at
    group boundaries. Chunks start at a decision record and end before one.
    """

    GROUP_MAX = 1024  # more records than any group

    def __init__(self, sets: list[DataSet], batch: int, train: bool, chunk: int = 1 << 18, buffer_chunks: int = 8,
                 swap_prob: float = 0.0, seed: int = 0):
        self.sets, self.batch, self.train = sets, batch, train
        self.chunk, self.buffer_chunks = chunk, buffer_chunks
        self.swap_prob = swap_prob if train else 0.0
        self.rng = np.random.default_rng(seed)
        if len({s.layout.raw for s in sets}) != 1:
            raise ValueError("data sets have different feature layouts")

    def _keep(self, m: np.ndarray) -> np.ndarray:
        val = is_val(m)
        return ~val if self.train else val

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
                m = np.array(s.meta[a : a + self.chunk + self.GROUP_MAX])
                r = np.flatnonzero(is_root(m))
                if len(r) == 0:
                    continue
                end = r[r >= self.chunk][0] if (r >= self.chunk).any() else (
                    len(m) if a + len(m) >= len(s) else r[-1])
                m = m[r[0] : end]
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
            feats, metas, base = [], [], 0
            for feat, m in parts:
                m = m.copy()
                m["offset"] += base
                base += len(feat)
                feats.append(feat)
                metas.append(m)
            feat = np.concatenate(feats)
            meta = np.concatenate(metas)
            root = is_root(meta)
            if self.train:
                gid = np.cumsum(root) - 1
                meta = meta[np.argsort(self.rng.permutation(int(root.sum()))[gid], kind="stable")]
                root = is_root(meta)
            starts = np.flatnonzero(root)
            b = 0
            while b < len(meta):
                k = np.searchsorted(starts, b + self.batch)
                e = int(starts[k]) if k < len(starts) else len(meta)
                mb = meta[b:e]
                b = e
                if self.train and len(mb) < self.batch // 2:
                    continue
                swap = self.rng.random(len(mb)) < self.swap_prob if self.swap_prob > 0 else None
                yield make_batch(feat, mb, swap)

    def count(self) -> int:
        return int(sum(self._keep(np.array(s.meta)).sum() for s in self.sets))
