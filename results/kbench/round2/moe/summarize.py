#!/usr/bin/env python3
"""Per-step MoE GEMM estimate from bench_moe2.py JSON: for each (bs, mode), the shipped reference vs the best
bit-identical config per GEMM (w13, w2), scaled to a verify step: 44 NVFP4 visits + 4 MXFP8 visits.

  python3 summarize.py results/run1.json [more.json ...]
"""

import json
import statistics
import sys

VISITS = {"nvfp4": 44, "mxfp8": 4}


def main(paths):
    rs = []
    for p in paths:
        rs += [r for r in json.load(open(p)) if "us" in r]
    keys = sorted({(r["bs"], r["mode"]) for r in rs})
    best_cfg = {}
    print(f"{'bs':>3} {'mode':<6} {'kind':<6} {'gemm':<4} {'union':>5} {'ref us':>8} {'best us':>8} {'gain':>6}  best config")
    tot = {}
    for bs, mode in keys:
        for kind in ("nvfp4", "mxfp8"):
            for gemm in ("w13", "w2"):
                sel = [r for r in rs if (r["bs"], r["mode"], r["kind"], r["gemm"]) == (bs, mode, kind, gemm)]
                if not sel:
                    continue
                steps = sorted({r["step"] for r in sel})
                by = {}
                for r in sel:
                    by.setdefault(r["cfg"], {})[r["step"]] = r
                ref = [c for c, d in by.items() if all(x["is_ref"] for x in d.values())]
                if not ref:
                    continue
                ref_us = statistics.fmean(by[ref[0]][s]["us"] for s in steps)
                cands = []
                for c, d in by.items():
                    if len(d) != len(steps):
                        continue
                    if any(x["mismatch"] or not x["finite"] or not x["ref_det"] for x in d.values()):
                        continue
                    cands.append((statistics.fmean(d[s]["us"] for s in steps), c))
                b_us, b_c = min(cands)
                u = statistics.fmean(r["union"] for r in sel if r["cfg"] == ref[0])
                print(f"{bs:3d} {mode:<6} {kind:<6} {gemm:<4} {u:5.1f} {ref_us:8.1f} {b_us:8.1f} {100 * (1 - b_us / ref_us):5.1f}%  {b_c}")
                t = tot.setdefault((bs, mode), [0.0, 0.0])
                t[0] += VISITS[kind] * ref_us / 1000
                t[1] += VISITS[kind] * b_us / 1000
                best_cfg.setdefault((kind, gemm, bs), []).append((mode, b_c))
    print("\nper verify step (44 NVFP4 + 4 MXFP8 visits, w13 + w2), ms:")
    for (bs, mode), (a, b) in sorted(tot.items()):
        print(f"  bs {bs:2d} {mode:<6}: shipped {a:7.2f}  best {b:7.2f}  saves {a - b:5.2f} ms ({100 * (1 - b / a):4.1f} %)")


if __name__ == "__main__":
    main(sys.argv[1:])
