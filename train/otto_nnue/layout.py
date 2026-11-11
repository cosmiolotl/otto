"""Feature layout and architecture constants. Must match `crates/bot/src/nnue/{features,net,file}.rs`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FEATURE_SET = "rows-v1"

# Architecture (net.rs). L1, the accumulator width, is per network: L1 is the default (v1, v2), and any
# multiple of 128 up to L1_MAX is supported (`supported_l1`).
L1 = 256
L1_MAX = 512
L2 = 16
L3_IN = 2 * L2
L3 = 32
N_BUCKETS = 8
FT_Q = 255
A_Q = 127
W_Q = 64
FORMAT_VERSION = 1

# Clamp ranges the integer path relies on (net.rs module docs).
FT_W_MAX = 1.0
FT_B_MAX = 8.0
DENSE_W_MAX = 127.0 / W_Q


def supported_l1(l1: int) -> bool:
    return 0 < l1 <= L1_MAX and l1 % 128 == 0


def fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


@dataclass(frozen=True)
class Block:
    name: str
    offset: int
    dims: tuple[int, ...]

    @property
    def size(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass(frozen=True)
class Layout:
    raw: str  # compact JSON, exactly as Rust writes it (its FNV-1a is the layout hash in .nnue files)
    n_features: int
    max_active: int
    blocks: dict[str, Block]  # in offset order

    @property
    def hash(self) -> int:
        return fnv1a(self.raw.encode())

    @staticmethod
    def from_json(j: dict) -> "Layout":
        if j["feature_set"] != FEATURE_SET:
            raise ValueError(f"feature set {j['feature_set']!r}, trainer expects {FEATURE_SET!r}")
        blocks = {b["name"]: Block(b["name"], b["offset"], tuple(b["dims"])) for b in j["blocks"]}
        off = 0
        for b in blocks.values():
            if b.offset != off:
                raise ValueError(f"block {b.name} at {b.offset}, expected {off}")
            off += b.size
        if off != j["n_features"]:
            raise ValueError(f"blocks cover {off} features, layout says {j['n_features']}")
        # The int16 accumulator must not overflow: every active feature at the weight clamp plus the bias.
        if (j["max_active"] * FT_W_MAX + FT_B_MAX) * FT_Q > 32767:
            raise ValueError(f"max_active {j['max_active']} can overflow the int16 accumulator at these clamps")
        # Rust's `layout_json` is compact with keys in this order; re-serialising reproduces it.
        raw = json.dumps(j, separators=(",", ":"))
        return Layout(raw, j["n_features"], j["max_active"], blocks)


def read_header(stem: str | Path) -> tuple[dict, Layout]:
    """`STEM.json` → (header, layout)."""
    header = json.loads(Path(f"{stem}.json").read_text(encoding="utf8"))
    return header, Layout.from_json(header["layout"])
