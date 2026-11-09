"""Float model, quantisation-aware: its forward pass is the integer pass of net.rs with the roundings removed.

Feature transformer (FT). Each block's effective weight matrix is built from "virtual" factors that share
statistics across rare features (Stockfish's factorisers), and folded into one `N_FEATURES × L1` matrix.
The factoriser of each block ([`FACTORISERS`]):

  pattern  row / covered (y, p):  W[y,p] + P[p] + Σ_{x∈p} C[y,x] + F[y, popcount p]
  height   height (x, h):         W[x,h] + Σ_{k≤h} T[x,k] + Σ_{k≤h} S[k]          (thermometer)
  shared   step2 (x, d), step3 (x, d1, d2), next (i, p):  W[x, ...] + D[...]    (shared over x / i)
  thermo   b2b/combo/pending/ready/clock/speed (k):  W[k] + Σ_{j≤k} T[j]        (thermometer)
  plain    current, hold, last_clear

Because the FT is linear, folding is exact; export writes only the folded matrix. The folded weights are
clamped to ±FT_W_MAX in the forward pass, as the int16 accumulator requires.

Dense layers: 8 buckets (my stack height / 3); all buckets are computed and the sample's bucket is selected.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Batch
from .layout import DENSE_W_MAX, FT_B_MAX, FT_W_MAX, HALF, L1, L2, L2_IN, L3, L3_IN, N_BUCKETS, Layout

FACTORISERS = {
    "row": "pattern",
    "covered": "pattern",
    "height": "height",
    "step2": "shared",
    "step3": "shared",
    "current": "plain",
    "hold": "plain",
    "next": "shared",
    "b2b": "thermo",
    "combo": "thermo",
    "last_clear": "plain",
    "pending": "thermo",
    "ready": "thermo",
    "clock": "thermo",
    "speed": "thermo",
}


def _bits(width: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.arange(1 << width)
    bits = ((p[:, None] >> torch.arange(width)[None, :]) & 1).float()
    return bits, bits.sum(1).long()


class FeatureTransformer(nn.Module):
    def __init__(self, layout: Layout, init_scale: float = 0.02):
        super().__init__()
        unknown = set(layout.blocks) ^ set(FACTORISERS)
        if unknown:
            raise ValueError(f"blocks without a factoriser, or factorisers without a block: {sorted(unknown)}")
        self.layout = layout

        def z(*shape):
            return nn.Parameter(torch.zeros(*shape, L1))

        def r(*shape):
            return nn.Parameter(torch.randn(*shape, L1) * init_scale)

        p = {}
        for name, block in layout.blocks.items():
            kind, dims = FACTORISERS[name], block.dims
            if kind == "pattern":
                rows, pats = dims
                width = (pats - 1).bit_length()
                bits, pop = _bits(width)
                self.register_buffer(f"{name}_bits", bits, persistent=False)
                self.register_buffer(f"{name}_pop", pop, persistent=False)
                p[f"{name}_w"] = z(rows, pats)
                p[f"{name}_pat"] = r(pats)
                p[f"{name}_cell"] = r(rows, width)
                p[f"{name}_fill"] = r(rows, width + 1)
            elif kind == "height":
                nx, nh = dims
                p[f"{name}_w"] = z(nx, nh)
                p[f"{name}_t"] = r(nx, nh)
                p[f"{name}_s"] = r(nh)
            elif kind == "shared":
                p[f"{name}_w"] = z(*dims)
                p[f"{name}_d"] = r(*dims[1:])
            elif kind == "thermo":
                p[f"{name}_w"] = z(*dims)
                p[f"{name}_t"] = r(*dims)
            else:
                p[name] = r(*dims)
        self.p = nn.ParameterDict(p)
        self.bias = nn.Parameter(torch.full((L1,), 0.5))

    def _block(self, name: str) -> torch.Tensor:
        p = self.p
        kind = FACTORISERS[name]
        if kind == "pattern":
            bits, pop = getattr(self, f"{name}_bits"), getattr(self, f"{name}_pop")
            return (p[f"{name}_w"] + p[f"{name}_pat"][None]
                    + torch.einsum("px,yxl->ypl", bits, p[f"{name}_cell"])
                    + p[f"{name}_fill"][:, pop])
        if kind == "height":
            return p[f"{name}_w"] + torch.cumsum(p[f"{name}_t"], 1) + torch.cumsum(p[f"{name}_s"], 0)[None]
        if kind == "shared":
            return p[f"{name}_w"] + p[f"{name}_d"][None]
        if kind == "thermo":
            return p[f"{name}_w"] + torch.cumsum(p[f"{name}_t"], 0)
        return p[name]

    def folded(self) -> torch.Tensor:
        """The effective `N_FEATURES × L1` matrix (unclamped), blocks in layout order."""
        w = torch.cat([self._block(name).reshape(-1, L1) for name in self.layout.blocks], 0)
        assert w.shape[0] == self.layout.n_features, (w.shape, self.layout.n_features)
        return w

    def weight(self) -> torch.Tensor:
        return self.folded().clamp(-FT_W_MAX, FT_W_MAX)

    def side(self, w: torch.Tensor, idx: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        acc = F.embedding_bag(idx, w, off, mode="sum") + self.bias.clamp(-FT_B_MAX, FT_B_MAX)
        a, b = acc[:, :HALF].clamp(0, 1), acc[:, HALF:].clamp(0, 1)
        return a * b


class ValueNet(nn.Module):
    def __init__(self, layout: Layout):
        super().__init__()
        self.ft = FeatureTransformer(layout)
        self.l2 = nn.Linear(L2_IN, L2 * N_BUCKETS)
        self.l3 = nn.Linear(L3_IN, L3 * N_BUCKETS)
        self.out = nn.Linear(L3, N_BUCKETS)
        with torch.no_grad():
            for lin in (self.l2, self.l3, self.out):
                lin.weight.mul_(0.5)
                lin.bias.zero_()

    def forward(self, b: Batch) -> torch.Tensor:
        """Logits for a batch of tensors."""
        w = self.ft.weight()
        x = torch.cat([self.ft.side(w, b.me_idx, b.me_off), self.ft.side(w, b.opp_idx, b.opp_off)], 1)
        n = x.shape[0]
        ar = torch.arange(n, device=x.device)
        z2 = self.l2(x).view(n, N_BUCKETS, L2)[ar, b.bucket]
        c = z2.clamp(0, 1)
        h2 = torch.cat([c, c * c * (127.0 / 128.0)], 1)
        z3 = self.l3(h2).view(n, N_BUCKETS, L3)[ar, b.bucket]
        h3 = z3.clamp(0, 1)
        # Output bucket b uses row b of `out`.
        return (h3 * self.out.weight[b.bucket]).sum(1) + self.out.bias[b.bucket]

    @torch.no_grad()
    def clip_(self):
        for lin in (self.l2, self.l3, self.out):
            lin.weight.clamp_(-DENSE_W_MAX, DENSE_W_MAX)
