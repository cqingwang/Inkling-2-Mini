#!/usr/bin/env python3
"""Per-step MoE GEMM ms: shipped choice vs the inkling_moe2 default table, from bench_moe2.py JSON."""
import json, statistics, sys
rs = [r for p in sys.argv[1:] for r in json.load(open(p)) if "us" in r]
def table(kind, gemm, tokens):
    rows = tokens * 8
    if kind == "nvfp4":
        if 9 <= tokens <= 64 and gemm == "w13": return "swap_ab=1,tile_n=32,tile_k=256"
        if 65 <= tokens <= 128: return "swap_ab=1,tile_n=32,tile_k=256"
        return None
    if gemm == "w2" and 512 <= rows <= 1536: return "tile_m=32,tile_n=128"
    if gemm == "w13" and 1024 <= rows <= 1536: return "tile_m=32,tile_n=128"
    return None
V = {"nvfp4": 44, "mxfp8": 4}
out = {}
for (bs, mode) in sorted({(r["bs"], r["mode"]) for r in rs}):
    a = b = 0.0; detail = []
    for kind in ("nvfp4", "mxfp8"):
        for gemm in ("w13", "w2"):
            sel = [r for r in rs if (r["bs"], r["mode"], r["kind"], r["gemm"]) == (bs, mode, kind, gemm)]
            ref = statistics.fmean(r["us"] for r in sel if r["is_ref"])
            cfg = table(kind, gemm, sel[0]["tokens"])
            new = ref if cfg is None else statistics.fmean(r["us"] for r in sel if r["cfg"] == cfg)
            assert cfg is None or all(r["mismatch"] == 0 for r in sel if r["cfg"] == cfg)
            a += V[kind] * ref / 1e3; b += V[kind] * new / 1e3
            detail.append(f"{kind[:2]}.{gemm} {100*(1-new/ref):+.1f}%")
    print(f"bs {bs:2d} {mode:<5}: shipped {a:6.1f} ms  table {b:6.1f} ms  saves {a-b:4.2f} ms ({100*(1-b/a):.1f} %)   " + "  ".join(detail))
