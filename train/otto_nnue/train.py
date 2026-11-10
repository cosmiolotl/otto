"""Train the value network on `sim datagen` data.

    uv run --extra cu128 python -m otto_nnue.train --data ../runs/nnue/m5-a ../runs/nnue/m5-b --out ../runs/nnue/v1

Value loss: BCE with logits against `target.Target` (outcome blended with a teacher score), on decision
records. Ranking loss (data with siblings, `sim datagen --siblings K`): per decision, the KL divergence from
softmax(a * teacher / T) over its siblings to softmax(net / T), where `a` is the teacher's fitted
logit-per-point scale, so the net's logit differences between siblings learn to match the teacher's
calibrated ones. Loss = value + rank_weight * rank.
Every epoch writes `OUT/epoch-NNN.nnue`, `OUT/last.nnue` and `OUT/last.pt` (resume with `--resume`).
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
from .target import SCORE_CLAMP, TEACHERS, Target


def sibling_logits(y: torch.Tensor, b: Batch, a: float, temp: float):
    """Net and teacher logits of the siblings as dense [G, K] matrices (padding -1e9), the mask, and the
    bot's move (1.0) per entry. None if the batch has no siblings."""
    if len(b.grp) == 0:
        return None
    g, k = int(b.grp.max()) + 1, int(b.pos.max()) + 1
    at = (b.grp, b.pos)
    fill = torch.full((g, k), -1e9, device=y.device)
    net = fill.index_put(at, y[b.sib_idx] / temp)
    teach = fill.index_put(at, a * b.static_score[b.sib_idx].clamp(-SCORE_CLAMP, SCORE_CLAMP) / temp)
    mask = torch.zeros((g, k), dtype=torch.bool, device=y.device).index_put(at, torch.ones_like(b.chosen, dtype=torch.bool))
    chosen = torch.zeros((g, k), device=y.device).index_put(at, b.chosen)
    return net, teach, mask, chosen


def rank_kl(net: torch.Tensor, teach: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per group: KL(softmax(teach) || softmax(net)) over the masked entries."""
    lq, lp = torch.log_softmax(teach, 1), torch.log_softmax(net, 1)
    return torch.where(mask, lq.exp() * (lq - lp), torch.zeros_like(lq)).sum(1)


def losses(model: ValueNet, b: Batch, target: Target, rank_a: float, temp: float):
    """(value BCE summed over decisions, rank KL summed over groups, #decisions, #groups, logits, dense
    sibling matrices or None)."""
    y = model(b)
    yr = y[b.root]
    value = F.binary_cross_entropy_with_logits(yr, target(b)[b.root], reduction="sum")
    d = sibling_logits(y, b, rank_a, temp)
    if d is None:
        return value, y.new_zeros(()), len(yr), 0, y, None
    rank = rank_kl(d[0], d[1], d[2]).sum()
    return value, rank, len(yr), d[0].shape[0], y, d


def evaluate(model: ValueNet, loader: Loader, dev, target: Target, rank_a: float, temp: float) -> dict:
    model.eval()
    tot = {"n": 0, "bce_z": 0.0, "bce_t": 0.0, "acc": 0.0, "dec": 0,
           "groups": 0, "kl": 0.0, "top1": 0.0, "pairs": 0.0, "pair_ok": 0.0, "chosen_net": 0.0,
           "chosen_teacher": 0.0, "with_chosen": 0}
    with torch.no_grad():
        for b in loader:
            b = b.to(dev)
            _, rank, nd, ng, y, d = losses(model, b, target, rank_a, temp)
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
            tot["kl"] += rank.item()
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="data stems written by `sim datagen`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.5, help="teacher weight in the target (0 = outcomes only)")
    ap.add_argument("--teacher", choices=TEACHERS, default="static")
    ap.add_argument("--swap", type=float, default=0.0, help="probability of exchanging the perspectives")
    ap.add_argument("--rank-weight", type=float, default=1.0, help="weight of the sibling ranking loss")
    ap.add_argument("--rank-temp", type=float, default=1.0, help="softmax temperature of the ranking loss")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
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
    layout = sets[0].layout
    train = Loader(sets, args.batch, train=True, swap_prob=args.swap, seed=args.seed)
    val = Loader(sets, args.batch, train=False, seed=args.seed + 1)
    n_train = train.count()
    steps_per_epoch = max(1, math.ceil(n_train / args.batch))
    total_steps = steps_per_epoch * args.epochs
    target = Target.fit(sets, args.lam, args.teacher, seed=args.seed)
    # The ranking target is the static teacher (siblings carry no search score of their own).
    rank_a = target.a if args.teacher == "static" else Target.fit(sets, 1.0, "static", seed=args.seed).a
    siblings = max(s.siblings for s in sets)
    print(f"data: {sum(len(s) for s in sets)} samples ({n_train} train) from {len(sets)} set(s), up to {siblings} "
          f"siblings per decision; {steps_per_epoch} steps/epoch; device {dev}; teacher {args.teacher} fit "
          f"a={target.a:.4f} c={target.c:.4f}; rank scale {rank_a:.4f}, weight {args.rank_weight}, "
          f"temp {args.rank_temp}")

    model = ValueNet(layout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gamma = (args.lr_end / args.lr) ** (1 / max(total_steps, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    epoch0, step = 0, 0
    if args.resume and (out / "last.pt").exists():
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
            value, rank, nd, ng, _, _ = losses(model, b, target, rank_a, args.rank_temp)
            loss = value / max(nd, 1) + args.rank_weight * rank / max(ng, 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            model.clip_()
            step += 1
            run_loss += loss.item() * len(b.result)
            seen += len(b.result)
            run_v, run_r, run_nd, run_ng = run_v + value.item(), run_r + rank.item(), run_nd + nd, run_ng + ng
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step} value {run_v / max(run_nd, 1):.5f} "
                      f"rank {run_r / max(run_ng, 1):.5f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {seen / (time.time() - t0):.0f} samples/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        stats = evaluate(model, val, dev, target, rank_a, args.rank_temp)
        stats.update(epoch=epoch + 1, step=step, train_loss=run_loss / max(seen, 1),
                     train_value=run_v / max(run_nd, 1), train_rank=run_r / max(run_ng, 1),
                     secs=time.time() - t0, teacher=[target.a, target.c], lam=args.lam, rank_scale=rank_a)
        print(json.dumps(stats), flush=True)
        log.write(json.dumps(stats) + "\n")
        log.flush()
        desc = json.dumps({"epoch": epoch + 1, "data": args.data, "lam": args.lam, "teacher": args.teacher,
                           "rank_weight": args.rank_weight, "val_bce_outcome": stats["val_bce_outcome"],
                           "val_pair_acc": stats.get("val_pair_acc")})
        q = quantize(model, layout, desc)
        write_nnue(q, out / f"epoch-{epoch + 1:03d}.nnue")
        write_nnue(q, out / "last.nnue")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "step": step, "args": vars(args)}, out / "last.pt")
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
