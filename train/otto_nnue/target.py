"""The training target: per sample (side to move),

    t = lam * sigmoid(a * teacher + c) + (1 - lam) * z,   z = 1 win / 0.5 draw / 0 loss

`teacher` is the static evaluation or the search score recorded by `sim datagen` (the playing bot's own
search score, or with `--sibling-search D,W` the teacher's search at that depth and width); (a, c) are fitted
by logistic regression of z on the teacher score of decisions, so the teacher is expressed as a win
probability (Stockfish's "eval → WDL" scaling). `lam = 0` trains on outcomes only. The fit is done
regardless: `a` of the `--rank-score` field also scales the sibling ranking target (train.py).

**Per-set calibration.** A hand-written teacher scores in points (hundreds), a network teacher in logits
(about ±10), and two teachers or two searches of one kind need not agree either. So (a, c) are fitted
separately for each *calibration group*: the data sets with equal `DataSet.calib_key(field)` (same teacher
kind and spec, and for the search field the same `--sibling-search`). Every record uses its own set's
(a, c), so hand-teacher and network-teacher shards can be trained together. Data of one group (every
older run) gets exactly the single pooled fit of before.

**Clamp.** Scores are clamped to ±`SCORE_CLAMP[kind]` before the fit and the target: lines that top out
inside the search score `-1e6 + 1000·ply` whatever the evaluator, which must neither dominate the fit nor
overflow. 1e4 points for a hand teacher (as before), 64 logits for a network (beyond any real eval).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch, DataSet, is_root, is_val, outcome

TEACHERS = ("static", "search")
SCORE_CLAMP = {"hand": 1e4, "nnue": 64.0}


@dataclass(frozen=True)
class Target:
    lam: float
    teacher: str
    # Per data set (indexed by `Batch.set_id`, the order of the `sets` given to `fit`).
    a: tuple[float, ...] = (0.0,)
    c: tuple[float, ...] = (0.0,)
    clamp: tuple[float, ...] = (SCORE_CLAMP["hand"],)
    # Per calibration group: (key, set indices, a, c, decisions fitted on).
    groups: tuple = ()

    @staticmethod
    def fit(sets: list[DataSet], lam: float, teacher: str, n: int = 1_000_000, seed: int = 0) -> "Target":
        """Fit (a, c) per calibration group on up to `n` decisive training decisions of the group (siblings are
        not positions of the game)."""
        if teacher not in TEACHERS:
            raise ValueError(f"teacher {teacher!r} not in {TEACHERS}")
        keys: dict[tuple, list[int]] = {}
        for i, s in enumerate(sets):
            keys.setdefault(s.calib_key(teacher), []).append(i)
        a, c = [0.0] * len(sets), [0.0] * len(sets)
        clamp = [SCORE_CLAMP[s.teacher_kind] for s in sets]
        groups = []
        for key, idx in keys.items():
            ga, gc, used = _fit_group([sets[i] for i in idx], teacher, SCORE_CLAMP[sets[idx[0]].teacher_kind], n, seed)
            for i in idx:
                a[i], c[i] = ga, gc
            groups.append((key, tuple(idx), ga, gc, used))
        return Target(lam, teacher, tuple(a), tuple(c), tuple(clamp), tuple(groups))

    def describe(self) -> str:
        if len(self.groups) == 1:
            return f"a={self.groups[0][2]:.4f} c={self.groups[0][3]:.4f}"
        return "; ".join(f"[{','.join(map(str, idx))}] {key[0]} {key[1] or ''} a={ga:.4f} c={gc:.4f} ({used} decisions)"
                         for key, idx, ga, gc, used in self.groups)

    def _per_set(self, b: Batch):
        dev = b.set_id.device
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)[b.set_id]
        return t(self.a), t(self.c), t(self.clamp)

    def scaled(self, b: Batch, field: str | None = None) -> torch.Tensor:
        """`a · clamp(<field>_score)` per record of `b` (default field: this target's teacher), each with its
        set's calibration: the teacher's logit up to its set's constant c."""
        a, _, cl = self._per_set(b)
        score = getattr(b, f"{field or self.teacher}_score")
        return a * torch.maximum(torch.minimum(score, cl), -cl)

    def __call__(self, b: Batch) -> torch.Tensor:
        """Value targets for a batch of tensors (meaningful for its decision records)."""
        if self.lam <= 0:
            return b.result
        _, c, _ = self._per_set(b)
        tp = torch.sigmoid(self.scaled(b) + c)
        return self.lam * tp + (1 - self.lam) * b.result


def _fit_group(sets: list[DataSet], teacher: str, clamp: float, n: int, seed: int) -> tuple[float, float, int]:
    parts = []
    for s in sets:
        m = np.array(s.meta)
        parts.append(m[(m["result"] != 0) & ~is_val(m) & is_root(m)])
    m = np.concatenate(parts)
    if len(m) == 0:
        raise ValueError(f"no decisive training samples to fit the teacher on in {[s.stem for s in sets]} "
                         "(use lam = 0)")
    m = m[np.random.default_rng(seed).permutation(len(m))[:n]]
    x = torch.from_numpy(m[f"{teacher}_score"].astype(np.float64)).clamp(-clamp, clamp)
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
    return float(p[0].detach() / s), float(p[1].detach()), len(m)
