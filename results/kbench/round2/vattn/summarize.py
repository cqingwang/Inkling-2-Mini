#!/usr/bin/env python3
"""Tables from vattn_bench.py results: python3 summarize.py r1.json"""
import json
import sys

res = json.load(open(sys.argv[1]))
corr = [r for k, r in res if k == "correct"]
bench = [r for k, r in res if k == "bench"]
if corr:
    print("| bs | ctx | layer | stock out | stock lse | compact out | compact lse | ink32 out | ink32 lse | ink64 out | final st / cmp / ink32 |")
    print("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in corr:
        print(f"| {r['bs']} | {r['ctx']} | {'swa' if r['local'] else 'global'} | {r['stock_out']:.2e} | {r['stock_lse']:.2e} | "
              f"{r['compact_out']:.2e} | {r['compact_lse']:.2e} | {r['ink32_out']:.2e} | {r['ink32_lse']:.2e} | "
              f"{r['ink64_out']:.2e} | {r['stock_final']:.1e} / {r['compact_final']:.1e} / {r['ink32_final']:.1e} |")
if bench:
    print()
    print("| bs | ctx | layer | KV MB | stock FA4+gs us (GB/s) | compact stock-cfg us (GB/s) | compact best us (GB/s) cfg | in-kernel best us (GB/s) cfg |")
    print("|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in bench:
        def f(k):
            return f"{r[k]:.0f} ({r[k + '_GBs']:.0f})" if k in r else "-"
        print(f"| {r['bs']} | {r['ctx']} | {'swa' if r['local'] else 'global'} | {r['kv_MB']} | {f('stock_total')} | "
              f"{f('compact_total')} {tuple(r['compact_cfg'])} | {f('best_total')} {tuple(r.get('best_cfg', ()))} | "
              f"{f('ink_total')} {tuple(r.get('ink_cfg', ()))} |")
