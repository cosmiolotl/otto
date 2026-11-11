"""Quantisation, the `.nnue` file format (crates/bot/src/nnue/file.rs), and an exact integer reference pass."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .layout import (
    A_Q, DENSE_W_MAX, FORMAT_VERSION, FT_B_MAX, FT_Q, FT_W_MAX, L2, L3, L3_IN, N_BUCKETS, W_Q, Layout, supported_l1,
)

MAGIC = b"OTTONNUE"


@dataclass
class QuantNet:
    layout_hash: int
    n_features: int
    desc: str
    ft_b: np.ndarray  # i16[L1]
    ft_w: np.ndarray  # i16[N, L1]
    l2_w: np.ndarray  # i8[B, L2, L1]
    l2_b: np.ndarray  # i32[B, L2]
    l3_w: np.ndarray  # i8[B, L3, L3_IN]
    l3_b: np.ndarray  # i32[B, L3]
    out_w: np.ndarray  # i8[B, L3]
    out_b: np.ndarray  # i32[B]

    @property
    def l1(self) -> int:
        return len(self.ft_b)


def _q(x: np.ndarray, scale: float, dtype, lo: float, hi: float) -> np.ndarray:
    info = np.iinfo(dtype)
    return np.clip(np.round(np.clip(x, lo, hi) * scale), info.min, info.max).astype(dtype)


def quantize(model, layout: Layout, desc: str = "") -> QuantNet:
    import torch

    with torch.no_grad():
        ft_w = model.ft.folded().cpu().numpy()
        ft_b = model.ft.bias.cpu().numpy()
        l2w = model.l2.weight.cpu().numpy().reshape(N_BUCKETS, L2, model.l1)
        l2b = model.l2.bias.cpu().numpy().reshape(N_BUCKETS, L2)
        l3w = model.l3.weight.cpu().numpy().reshape(N_BUCKETS, L3, L3_IN)
        l3b = model.l3.bias.cpu().numpy().reshape(N_BUCKETS, L3)
        ow = model.out.weight.cpu().numpy()
        ob = model.out.bias.cpu().numpy()
    big = 1e9
    return QuantNet(
        layout.hash, layout.n_features, desc,
        _q(ft_b, FT_Q, np.int16, -FT_B_MAX, FT_B_MAX),
        _q(ft_w, FT_Q, np.int16, -FT_W_MAX, FT_W_MAX),
        _q(l2w, W_Q, np.int8, -DENSE_W_MAX, DENSE_W_MAX), _q(l2b, A_Q * W_Q, np.int32, -big, big),
        _q(l3w, W_Q, np.int8, -DENSE_W_MAX, DENSE_W_MAX), _q(l3b, A_Q * W_Q, np.int32, -big, big),
        _q(ow, W_Q, np.int8, -DENSE_W_MAX, DENSE_W_MAX), _q(ob, A_Q * W_Q, np.int32, -big, big),
    )


def write_nnue(q: QuantNet, path: str | Path) -> None:
    desc = q.desc.encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8I", FORMAT_VERSION, q.layout_hash, q.n_features, q.l1, L2, L3, N_BUCKETS, len(desc)))
        f.write(desc)
        f.write(q.ft_b.astype("<i2").tobytes())
        f.write(q.ft_w.astype("<i2").tobytes())
        for b in range(N_BUCKETS):
            f.write(q.l2_w[b].astype("i1").tobytes())
            f.write(q.l2_b[b].astype("<i4").tobytes())
            f.write(q.l3_w[b].astype("i1").tobytes())
            f.write(q.l3_b[b].astype("<i4").tobytes())
            f.write(q.out_w[b].astype("i1").tobytes())
            f.write(np.array([q.out_b[b]], dtype="<i4").tobytes())


def read_nnue(path: str | Path) -> QuantNet:
    data = Path(path).read_bytes()
    assert data[:8] == MAGIC, "not an otto network file"
    version, h, nf, l1, l2, l3, nb, dl = struct.unpack_from("<8I", data, 8)
    assert (version, l2, l3, nb) == (FORMAT_VERSION, L2, L3, N_BUCKETS) and supported_l1(l1), "architecture mismatch"
    pos = 40
    desc = data[pos : pos + dl].decode(errors="replace")
    pos += dl

    def take(dtype, count):
        nonlocal pos
        a = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
        pos += a.nbytes
        return a

    ft_b = take("<i2", l1)
    ft_w = take("<i2", nf * l1).reshape(nf, l1)
    l2_w, l2_b, l3_w, l3_b, out_w, out_b = [], [], [], [], [], []
    for _ in range(nb):
        l2_w.append(take("i1", L2 * l1).reshape(L2, l1))
        l2_b.append(take("<i4", L2))
        l3_w.append(take("i1", L3 * L3_IN).reshape(L3, L3_IN))
        l3_b.append(take("<i4", L3))
        out_w.append(take("i1", L3))
        out_b.append(take("<i4", 1)[0])
    assert pos == len(data), "trailing bytes"
    return QuantNet(h, nf, desc, ft_b, ft_w, np.stack(l2_w), np.stack(l2_b), np.stack(l3_w), np.stack(l3_b),
                    np.stack(out_w), np.array(out_b))


def _wrap16(x: np.ndarray) -> np.ndarray:
    return ((x + 32768) % 65536) - 32768


def reference_eval(q: QuantNet, me: np.ndarray, opp: np.ndarray, bucket: int) -> int:
    """Exact integer forward pass (same roundings and wrapping as net.rs). Returns the raw output."""

    half = q.l1 // 2

    def side(feats):
        acc = _wrap16(q.ft_b.astype(np.int64) + q.ft_w[np.asarray(feats, dtype=np.int64)].astype(np.int64).sum(0))
        a = np.clip(acc[:half], 0, FT_Q)
        b = np.clip(acc[half:], 0, FT_Q)
        return (a * b) >> 9

    x = np.concatenate([side(me), side(opp)])
    v2 = (q.l2_b[bucket].astype(np.int64) + q.l2_w[bucket].astype(np.int64) @ x) >> 6
    c = np.clip(v2, 0, A_Q)
    h2 = np.concatenate([c, (c * c) >> 7])
    v3 = (q.l3_b[bucket].astype(np.int64) + q.l3_w[bucket].astype(np.int64) @ h2) >> 6
    h3 = np.clip(v3, 0, A_Q)
    return int(q.out_b[bucket] + q.out_w[bucket].astype(np.int64) @ h3)

