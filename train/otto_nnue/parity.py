"""Check that Rust inference equals the integer reference on a data set, bit for bit.

    uv run --extra cpu python -m otto_nnue.parity --net ../runs/nnue/v1/last.nnue --data ../runs/nnue/smoke \
        --sim ../target/release/sim --n 500

Runs `sim nnue-eval` (Rust loader + AVX2/scalar kernels) and `export.reference_eval` (this package's reader
and an independent numpy implementation) on the same samples and compares the raw integer outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .data import DataSet
from .export import read_nnue, reference_eval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sim", default="../target/release/sim")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args(argv)

    q = read_nnue(args.net)
    ds = DataSet(args.data)
    if q.layout_hash != ds.layout.hash:
        print(f"layout hash mismatch: net {q.layout_hash:#x}, data {ds.layout.hash:#x}")
        return 1
    out = subprocess.run([args.sim, "nnue-eval", "--net", args.net, "--data", args.data, "--n", str(args.n)],
                         check=True, capture_output=True, text=True).stdout.split("\n")
    rows = [line.split() for line in out if line.strip()]
    rust = {int(c[0]): int(c[3]) for c in rows}  # `index bucket result raw logit`
    if sorted(rust) != list(range(min(args.n, len(ds)))):
        print(f"sim nnue-eval returned samples {sorted(rust)[:5]}..., expected 0..{min(args.n, len(ds))}")
        return 1
    bad = 0
    for i, r in rust.items():
        want = reference_eval(q, *ds.sample(i))
        if want != r:
            bad += 1
            if bad <= 5:
                print(f"sample {i}: rust {r} reference {want}")
    print(f"parity: {len(rust) - bad}/{len(rust)} samples identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
