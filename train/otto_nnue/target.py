"""The training target: per sample (side to move),

    t = lam * sigmoid(a * teacher + c) + (1 - lam) * z,   z = 1 win / 0.5 draw / 0 loss

`teacher` is the hand-written static evaluation or the search score recorded by `sim datagen`; (a, c) are
fitted by logistic regression of z on the teacher score, so the teacher is expressed as a win probability
(Stockfish's "eval → WDL" scaling). `lam = 0` trains on outcomes only. The fit is done regardless: `a` also scales the sibling ranking target
(train.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch, DataSet, is_root, is_val, outcome

TEACHERS = ("static", "search")
SCORE_CLAMP = 1e4


@dataclass(frozen=True)
class Target:
    lam: float
    teacher: str
    a: float = 0.0
    c: float = 0.0

    @staticmethod
    def fit(sets: list[DataSet], lam: float, teacher: str, n: int = 1_000_000, seed: int = 0) -> "Target":
        """Fit (a, c) on up to `n` decisive training decisions (siblings are not positions of the game)."""
        if teacher not in TEACHERS:
            raise ValueError(f"teacher {teacher!r} not in {TEACHERS}")
        parts = []
        for s in sets:
            m = np.array(s.meta)
            parts.append(m[(m["result"] != 0) & ~is_val(m) & is_root(m)])
        m = np.concatenate(parts)
        if len(m) == 0:
            raise ValueError("no decisive training samples to fit the teacher on (use lam = 0)")
        m = m[np.random.default_rng(seed).permutation(len(m))[:n]]
        x = torch.from_numpy(m[f"{teacher}_score"].astype(np.float64)).clamp(-SCORE_CLAMP, SCORE_CLAMP)
        z = torch.from_numpy(outcome(m).astype(np.float64))
        s = x.std().clamp_min(1e-6)
        p = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([p], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(p[0] * x / s + p[1], z)
            loss.backward()
            return loss

        opt.step(closure)
        return Target(lam, teacher, float(p[0].detach() / s), float(p[1].detach()))

    def __call__(self, b: Batch) -> torch.Tensor:
        """Value targets for a batch of tensors (meaningful for its decision records)."""
        if self.lam <= 0:
            return b.result
        score = getattr(b, f"{self.teacher}_score")
        tp = torch.sigmoid(self.a * score.clamp(-SCORE_CLAMP, SCORE_CLAMP) + self.c)
        return self.lam * tp + (1 - self.lam) * b.result
