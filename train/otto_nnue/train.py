"""Train the value network on `sim datagen` data.

    uv run --extra cu128 python -m otto_nnue.train --data ../runs/nnue/m5-a ../runs/nnue/m5-b --out ../runs/nnue/v1

Value loss: BCE with logits against `target.Target` (outcome blended with a teacher score), on decision
records. Ranking loss (data with siblings, `sim datagen --siblings K`): per decision, the KL divergence from
softmax(a * teacher / T) over its siblings to softmax(net / T), where `a` is the teacher's fitted
logit-per-point scale, so the net's logit differences between siblings learn to match the teacher's
calibrated ones. `--rank-score` picks the siblings' teacher field: `static` (the teacher's static eval of
the child) or `search` (its search value, data from `sim datagen --sibling-search D,W`); `a` is fitted on
the decisions' score of the same field. Both calibrations are fitted per data set group (target.py), so
hand-teacher (points) and network-teacher (logits) data can be mixed. Loss = value + rank_weight * rank.
Every epoch writes `OUT/epoch-NNN.nnue`, `OUT/last.nnue` and `OUT/last.pt` (resume with `--resume`).
`--init RUN/last.pt` warm-starts the weights from another run (same layout and L1; fresh optimizer and LR
schedule).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import Batch, DataSet, Loader
from .export import quantize, write_nnue
from .model import ValueNet
from .target import TEACHERS, Target


def sibling_logits(y: torch.Tensor, b: Batch, rank: Target, temp: float):
    """Net and teacher logits of the siblings as dense [G, K] matrices (padding -1e9), the mask, and the
    bot's move (1.0) per entry. The teacher logits are `rank.scaled(b)`: `a * <rank.teacher>_score` with each
    record's set calibration. None if the batch has no siblings."""
    if len(b.grp) == 0:
        return None
    g, k = int(b.grp.max()) + 1, int(b.pos.max()) + 1
    at = (b.grp, b.pos)
    fill = torch.full((g, k), -1e9, device=y.device)
    net = fill.index_put(at, y[b.sib_idx] / temp)
    teach = fill.index_put(at, rank.scaled(b)[b.sib_idx] / temp)
    mask = torch.zeros((g, k), dtype=torch.bool, device=y.device).index_put(at, torch.ones_like(b.chosen, dtype=torch.bool))
    chosen = torch.zeros((g, k), device=y.device).index_put(at, b.chosen)
    return net, teach, mask, chosen


def rank_kl(net: torch.Tensor, teach: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per group: KL(softmax(teach) || softmax(net)) over the masked entries."""
    lq, lp = torch.log_softmax(teach, 1), torch.log_softmax(net, 1)
    return torch.where(mask, lq.exp() * (lq - lp), torch.zeros_like(lq)).sum(1)


def losses(model: ValueNet, b: Batch, target: Target, rank: Target, temp: float):
    """(value BCE summed over decisions, rank KL summed over groups, #decisions, #groups, logits, dense
    sibling matrices or None)."""
    y = model(b)
    yr = y[b.root]
    value = F.binary_cross_entropy_with_logits(yr, target(b)[b.root], reduction="sum")
    d = sibling_logits(y, b, rank, temp)
    if d is None:
        return value, y.new_zeros(()), len(yr), 0, y, None
    rank = rank_kl(d[0], d[1], d[2]).sum()
    return value, rank, len(yr), d[0].shape[0], y, d


def evaluate(model: ValueNet, loader: Loader, dev, target: Target, rank: Target, temp: float) -> dict:
    model.eval()
    tot = {"n": 0, "bce_z": 0.0, "bce_t": 0.0, "acc": 0.0, "dec": 0,
           "groups": 0, "kl": 0.0, "top1": 0.0, "pairs": 0.0, "pair_ok": 0.0, "chosen_net": 0.0,
           "chosen_teacher": 0.0, "with_chosen": 0}
    with torch.no_grad():
        for b in loader:
            b = b.to(dev)
            _, kl, nd, ng, y, d = losses(model, b, target, rank, temp)
            y = y[b.root]
            z = b.result[b.root]
            tot["n"] += nd
            tot["bce_z"] += F.binary_cross_entropy_with_logits(y, z, reduction="sum").item()
            tot["bce_t"] += F.binary_cross_entropy_with_logits(y, target(b)[b.root], reduction="sum").item()
            dec = z != 0.5
            tot["acc"] += ((y[dec] > 0) == (z[dec] > 0.5)).sum().item()
            tot["dec"] += dec.sum().item()
            if d is None:
                continue
            net, teach, mask, chosen = d
            tot["groups"] += ng
            tot["kl"] += kl.item()
            bn, bt = net.argmax(1), teach.argmax(1)
            tot["top1"] += (bn == bt).sum().item()
            # Pairs the teacher orders strictly: does the net order them the same way?
            pm = mask[:, :, None] & mask[:, None, :] & (teach[:, :, None] > teach[:, None, :])
            tot["pairs"] += pm.sum().item()
            tot["pair_ok"] += (pm & (net[:, :, None] > net[:, None, :])).sum().item()
            has = chosen.sum(1) > 0
            tot["with_chosen"] += has.sum().item()
            ar = torch.arange(len(bn), device=bn.device)
            tot["chosen_net"] += chosen[ar, bn][has].sum().item()
            tot["chosen_teacher"] += chosen[ar, bt][has].sum().item()
    model.train()
    n, g = max(tot["n"], 1), max(tot["groups"], 1)
    out = {"val_bce_outcome": tot["bce_z"] / n, "val_bce_target": tot["bce_t"] / n,
           "val_acc_decisive": tot["acc"] / max(tot["dec"], 1), "val_samples": tot["n"]}
    if tot["groups"]:
        wc = max(tot["with_chosen"], 1)
        out.update(val_rank_kl=tot["kl"] / g, val_top1=tot["top1"] / g,
                   val_pair_acc=tot["pair_ok"] / max(tot["pairs"], 1),
                   val_net_pick_played=tot["chosen_net"] / wc, val_teacher_pick_played=tot["chosen_teacher"] / wc,
                   val_groups=tot["groups"])
    return out


def calib(*per_set: tuple[float, ...]) -> list:
    """The logged calibration: scalars when every set shares one (as before per-set fits), else lists per set."""
    if all(len(set(v)) == 1 for v in per_set):
        return [v[0] for v in per_set]
    return [list(v) for v in per_set]


def init_from(model: ValueNet, path: str, layout, l1: int, dev) -> None:
    """Load another run's model weights (`--init`): the feature layout and L1 must match."""
    ck = torch.load(path, map_location=dev)
    ck_l1 = ck.get("args", {}).get("l1", 256)
    if ck_l1 != l1:
        raise SystemExit(f"--init {path}: L1 {ck_l1}, this run uses --l1 {l1}")
    if "layout" in ck and ck["layout"] != layout.raw:
        raise SystemExit(f"--init {path}: trained on another feature layout than this data's "
                         f"({json.loads(ck['layout'])['feature_set']} vs {layout.feature_set})")
    have, want = ck["model"], model.state_dict()
    bad = sorted(set(have) ^ set(want)) + sorted(k for k in set(have) & set(want) if have[k].shape != want[k].shape)
    if bad:
        # Checkpoints from before the layout was stored: the parameter shapes still tell the layouts apart.
        raise SystemExit(f"--init {path}: its model does not fit this data's layout {layout.feature_set} with L1 {l1} "
                         f"(mismatched parameters: {', '.join(bad[:6])}{' ...' if len(bad) > 6 else ''})")
    model.load_state_dict(have)
    print(f"initialised from {path} (epoch {ck.get('epoch')}, step {ck.get('step')}); optimizer and LR schedule fresh")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="data stems written by `sim datagen`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2.5e-4, help="peak learning rate (1e-3 often ends in a worse regime)")
    ap.add_argument("--lr-end", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.5, help="teacher weight in the target (0 = outcomes only)")
    ap.add_argument("--teacher", choices=TEACHERS, default="static")
    ap.add_argument("--swap", type=float, default=0.0, help="probability of exchanging the perspectives")
    ap.add_argument("--l1", type=int, default=256, help="accumulator width (a multiple of 128, at most 512)")
    ap.add_argument("--rank-weight", type=float, default=1.0, help="weight of the sibling ranking loss")
    ap.add_argument("--rank-temp", type=float, default=1.0, help="softmax temperature of the ranking loss")
    ap.add_argument("--rank-score", choices=TEACHERS, default="static",
                    help="siblings' teacher score for the ranking loss and its metrics (search: data from "
                         "`sim datagen --sibling-search`)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="continue OUT/last.pt (model, optimizer, schedule)")
    ap.add_argument("--init", metavar="CKPT", help="warm start: the model weights of another run's last.pt (same "
                    "layout and --l1); optimizer and LR schedule start fresh. Ignored when --resume resumes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.swap > 0 and args.lam > 0:
        # Teacher scores belong to the recorded side to move; a swapped sample has none for the new one.
        ap.error("--swap needs --lam 0")

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    sets = [DataSet(s) for s in args.data]
    if args.rank_score == "search":
        plain = [s.stem for s in sets if s.siblings and s.sibling_search is None]
        if plain:
            # Without it, a sibling's search_score is its static score and a decision's is the playing bot's.
            ap.error(f"--rank-score search needs data from `sim datagen --sibling-search`: {', '.join(plain)}")
    layout = sets[0].layout
    if args.swap > 0 and "plan_mode" in layout.blocks:
        # Plan features belong to the side to move only; a swap would put them in the opponent's half.
        ap.error("--swap can't be used with plan-feature data (rows-v1-plan)")
    train = Loader(sets, args.batch, train=True, swap_prob=args.swap, seed=args.seed)
    val = Loader(sets, args.batch, train=False, seed=args.seed + 1)
    n_train = train.count()
    steps_per_epoch = max(1, math.ceil(n_train / args.batch))
    total_steps = steps_per_epoch * args.epochs
    # Calibrations per group of data sets with the same teacher (and search): target.py.
    target = Target.fit(sets, args.lam, args.teacher, seed=args.seed)
    # The ranking scale is fitted on the decisions' score of the same field as the siblings' ranking score.
    rank = target if args.teacher == args.rank_score else Target.fit(sets, 1.0, args.rank_score, seed=args.seed)
    siblings = max(s.siblings for s in sets)
    kinds = sorted({s.teacher_kind for s in sets})
    print(f"data: {sum(len(s) for s in sets)} samples ({n_train} train) from {len(sets)} set(s), up to {siblings} "
          f"siblings per decision, teacher kind {'+'.join(kinds)}; {steps_per_epoch} steps/epoch; device {dev}; "
          f"teacher {args.teacher} fit {target.describe()}; rank {args.rank_score} scale {rank.describe()}, "
          f"weight {args.rank_weight}, temp {args.rank_temp}")

    model = ValueNet(layout, args.l1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gamma = (args.lr_end / args.lr) ** (1 / max(total_steps, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    epoch0, step = 0, 0
    resuming = args.resume and (out / "last.pt").exists()
    if args.init and not resuming:
        init_from(model, args.init, layout, args.l1, dev)
    elif args.init:
        print(f"--init {args.init} ignored: resuming {out / 'last.pt'}")
    if resuming:
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        epoch0, step = ck["epoch"], ck["step"]
        print(f"resumed at epoch {epoch0}, step {step}")

    log = open(out / "log.jsonl", "a", encoding="utf8")
    for epoch in range(epoch0, args.epochs):
        t0, run_loss, seen = time.time(), 0.0, 0
        run_v, run_r, run_nd, run_ng = 0.0, 0.0, 0, 0
        for b in train:
            b = b.to(dev)
            value, kl, nd, ng, _, _ = losses(model, b, target, rank, args.rank_temp)
            loss = value / max(nd, 1) + args.rank_weight * kl / max(ng, 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            model.clip_()
            step += 1
            run_loss += loss.item() * len(b.result)
            seen += len(b.result)
            run_v, run_r, run_nd, run_ng = run_v + value.item(), run_r + kl.item(), run_nd + nd, run_ng + ng
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step} value {run_v / max(run_nd, 1):.5f} "
                      f"rank {run_r / max(run_ng, 1):.5f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {seen / (time.time() - t0):.0f} samples/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        stats = evaluate(model, val, dev, target, rank, args.rank_temp)
        stats.update(epoch=epoch + 1, step=step, train_loss=run_loss / max(seen, 1),
                     train_value=run_v / max(run_nd, 1), train_rank=run_r / max(run_ng, 1),
                     secs=time.time() - t0, teacher=calib(target.a, target.c), lam=args.lam,
                     rank_scale=calib(rank.a)[0], rank_score=args.rank_score)
        print(json.dumps(stats), flush=True)
        log.write(json.dumps(stats) + "\n")
        log.flush()
        desc = json.dumps({"epoch": epoch + 1, "data": args.data, "l1": args.l1, "lam": args.lam, "teacher": args.teacher,
                           "rank_weight": args.rank_weight, "rank_score": args.rank_score, "val_bce_outcome": stats["val_bce_outcome"],
                           "val_pair_acc": stats.get("val_pair_acc")})
        q = quantize(model, layout, desc)
        write_nnue(q, out / f"epoch-{epoch + 1:03d}.nnue")
        write_nnue(q, out / "last.nnue")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "step": step, "args": vars(args), "layout": layout.raw}, out / "last.pt")
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
