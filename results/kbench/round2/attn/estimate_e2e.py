#!/usr/bin/env python3
"""End-to-end prefill estimate from check_attn.py's bench records (results.json).

Per 8192-token chunk with prefix P: 40 SWA visits + 8 global visits. Saving vs a baseline variant =
40 * (base_swa(P) - ours_swa(P)) + 8 * (base_global(P) - ours_global(P)); per-visit times are linearly
interpolated in P between the measured prefixes. Baseline TTFTs: the shipped recipe with all inkling_prefill
levers (README / docs/experiments.md: 1K 2,363, 8K 4,253, 32K 3,958, 128K 2,749 tok/s), which serves SWA and
global attention through bias_layout (Triton sheared bias + FA4 presheared); and stock (compose.txt measured
TTFT 2.40 / 10.1 / 54.0 s) against the stock einsum + FA4 path.

    python3 -I estimate_e2e.py results.json
"""
import json
import sys

TTFT_RECIPE = {8192: 8192 / 4253, 32768: 32768 / 3958, 131072: 131072 / 2749}
TTFT_STOCK = {8192: 2.40, 32768: 10.10, 131072: 54.0}
CHUNK = 8192


def interp(points, x):
    pts = sorted(points.items())
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return y1 + (y1 - y0) * (x - x1) / (x1 - x0)


def main(path):
    recs = [r for r in json.load(open(path)) if r.get("section") == "bench" and r.get("chunk") == CHUNK]
    t = {}
    for r in recs:
        t.setdefault((r["variant"], r["kind"]), {})[r["prefix"]] = r["med_us"] * 1e-6
    for base, ttft in (("bias_layout", TTFT_RECIPE), ("stock", TTFT_STOCK)):
        if (base, "swa") not in t or ("ours", "swa") not in t:
            continue
        print(f"baseline {base}:")
        for n, tt in ttft.items():
            save = 0.0
            save_swa = 0.0
            for k in range(n // CHUNK):
                p = k * CHUNK
                ds = interp(t[(base, "swa")], p) - interp(t[("ours", "swa")], p)
                dg = interp(t[(base, "global")], p) - interp(t[("ours", "global")], p)
                save_swa += 40 * ds
                save += 40 * ds + 8 * dg
            print(f"  {n // 1024:>4}K: TTFT {tt:6.2f} s -> {tt - save:6.2f} s (SWA only: {tt - save_swa:6.2f} s); "
                  f"saves {save:5.2f} s = {100 * save / tt:4.1f} %, {n / tt:5.0f} -> {n / (tt - save):5.0f} tok/s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.json")
