#!/usr/bin/env python3
"""End-to-end TTFT estimate of inkling_pfuse from the GPU-check microbench (stock vs fused at 8192 rows).
Per 8K chunk the fused path serves (layer 0 stays stock):
  resnorm  attention input: 19 local (Q=1) + 4 global (Q=3) visits; mlp input: 23 visits
  loopmix  attention side: 20 local + 4 global visits; mlp side: 24 visits
  sconvres 47 visits
Shipped TTFT (track H, prefill.txt): 8K 1.92 s (1 chunk), 32K 8.31 s (4), 128K 46.86 s (16)."""
import json, sys
r = {x["name"]: x for x in json.load(open(sys.argv[1]))["results"] if x["name"].startswith("time")}
extra = json.load(open(sys.argv[2]))["results"] if len(sys.argv) > 2 else []
r.update({x["name"]: x for x in extra if x["name"].startswith("time")})
loop_variant = sys.argv[3] if len(sys.argv) > 3 else "fused_us"
def save(name, key="fused_us"):
    x = r.get(name)
    if not x:
        return 0.0
    return (x["stock_us"] - x.get(key, x["fused_us"])) / 1e3
items = [
    ("resnorm attn local", 19, save("time resnorm attn local N=3")),
    ("resnorm attn global", 4, save("time resnorm attn global N=3 Q=3")),
    ("resnorm mlp", 23, save("time resnorm mlp N=4 +addend")),
    ("loopmix attn local", 20, save("time loopmix attn local L7 N=3", loop_variant)),
    ("loopmix attn global", 4, save("time loopmix attn global L11 N=3", loop_variant)),
    ("loopmix mlp", 24, save("time loopmix mlp L7 N=3 +addend")),
    ("sconvres", 47, (save("time sconvres partial=True") * 40 + save("time sconvres partial=False") * 7) / 47),
]
tot = 0
for n, c, s in items:
    print(f"{n:22s} x{c:3d}  {s:7.3f} ms/call  {s*c:7.1f} ms/chunk")
    tot += s * c
print(f"total saved per 8K chunk: {tot:.1f} ms")
for label, ttft, chunks, tokens in (("8K", 1.92, 1, 8192), ("32K", 8.31, 4, 32768), ("128K", 46.86, 16, 131072)):
    new = ttft - chunks * tot / 1e3
    print(f"{label}: {ttft:.2f} s -> {new:.2f} s ({tokens/ttft:,.0f} -> {tokens/new:,.0f} tok/s, {ttft/new:.3f}x)")
