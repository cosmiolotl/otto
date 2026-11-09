"""Train the value network on `sim datagen` data.

    uv run --extra cu128 python -m otto_nnue.train --data ../runs/nnue/m5-a ../runs/nnue/m5-b --out ../runs/nnue/v1

The target is `target.Target` (outcome blended with a teacher score); the loss is BCE with logits.
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

from .data import DataSet, Loader
from .export import quantize, write_nnue
from .model import ValueNet
from .target import TEACHERS, Target


def evaluate(model: ValueNet, loader: Loader, dev, target: Target) -> dict:
    model.eval()
    tot = {"n": 0, "bce_z": 0.0, "bce_t": 0.0, "acc": 0.0, "dec": 0}
    with torch.no_grad():
        for b in loader:
            b = b.to(dev)
            y = model(b)
            z = b.result
            tot["n"] += len(z)
            tot["bce_z"] += F.binary_cross_entropy_with_logits(y, z, reduction="sum").item()
            tot["bce_t"] += F.binary_cross_entropy_with_logits(y, target(b), reduction="sum").item()
            dec = z != 0.5
            tot["acc"] += ((y[dec] > 0) == (z[dec] > 0.5)).sum().item()
            tot["dec"] += dec.sum().item()
    model.train()
    n = max(tot["n"], 1)
    return {"val_bce_outcome": tot["bce_z"] / n, "val_bce_target": tot["bce_t"] / n,
            "val_acc_decisive": tot["acc"] / max(tot["dec"], 1), "val_samples": tot["n"]}


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
    print(f"data: {sum(len(s) for s in sets)} samples ({n_train} train) from {len(sets)} set(s); "
          f"{steps_per_epoch} steps/epoch; device {dev}; teacher {args.teacher} fit a={target.a:.4f} c={target.c:.4f}")

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
        for b in train:
            b = b.to(dev)
            loss = F.binary_cross_entropy_with_logits(model(b), target(b))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            model.clip_()
            step += 1
            run_loss += loss.item() * len(b.result)
            seen += len(b.result)
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step} loss {run_loss / seen:.5f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {seen / (time.time() - t0):.0f} samples/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        stats = evaluate(model, val, dev, target)
        stats.update(epoch=epoch + 1, step=step, train_loss=run_loss / max(seen, 1),
                     secs=time.time() - t0, teacher=[target.a, target.c], lam=args.lam)
        print(json.dumps(stats), flush=True)
        log.write(json.dumps(stats) + "\n")
        log.flush()
        desc = json.dumps({"epoch": epoch + 1, "data": args.data, "lam": args.lam, "teacher": args.teacher,
                           "val_bce_outcome": stats["val_bce_outcome"]})
        q = quantize(model, layout, desc)
        write_nnue(q, out / f"epoch-{epoch + 1:03d}.nnue")
        write_nnue(q, out / "last.nnue")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "step": step, "args": vars(args)}, out / "last.pt")
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
