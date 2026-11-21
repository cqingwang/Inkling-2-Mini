#!/usr/bin/env python3
"""Rebuild the cold-prefill TTFT of Inkling-2-Mini from bench_tc.py measurements (stdlib only).

  python3 -I compose.py bench.json

Per 8192-token chunk it sums the measured per-visit kernel times of the serving path:
48 MoE layers (whole layer incl. router, quantize, GEMMs, swiglu-quant, combine), the dense GEMMs
(20 single + 28 stacked qkvr, wo, dyn-conv, rel-bias einsum) and FA4 attention with rel-bias
(SWA x40, global x8 at the chunk's prefix; global time interpolated linearly in the prefix between
the measured prefixes). What is not benchmarked (attn-res, norms, clips, convs, KV store, gate_sink,
loop controllers, DSpark inject, host overhead) is the 'remainder' against the measured TTFT.
"""
import json
import sys

MEASURED = {1024: 0.46, 8192: 2.40, 32768: 10.1, 131072: 54.0}


def main():
    d = json.load(open(sys.argv[1]))
    r = d['records']
    g = {(x['name'], x['m']): x['med_us'] for x in r if x['section'] == 'gemm' and 'med_us' in x}
    moe = {(x['precision'], x['tokens']): x for x in r if x['section'] == 'moe' and 'total_us' in x}
    att = {(x['kind'], x['chunk'], x['prefix'], x['rel_bias']): x['med_us']
           for x in r if x['section'] == 'attn' and 'med_us' in x}

    def glob(prefix, bias=True):
        pts = sorted((p, t) for (k, c, p, b), t in att.items() if k == 'global' and c == 8192 and b == bias)
        for (p0, t0), (p1, t1) in zip(pts, pts[1:]):
            if p0 <= prefix <= p1:
                return t0 + (t1 - t0) * (prefix - p0) / (p1 - p0)
        (p0, t0), (p1, t1) = pts[-2], pts[-1]
        return t1 + (t1 - t0) * (prefix - p1) / (p1 - p0)

    def chunk(c, prefix):
        m = 8192 if c > 1024 else 1024
        rows = {}
        rows['MoE NVFP4 x44 (whole layer)'] = 44 * moe[('nvfp4', m)]['total_us']
        rows['  of which grouped GEMMs'] = 44 * (moe[('nvfp4', m)]['stages_us']['GEMM1 w13']
                                                 + moe[('nvfp4', m)]['stages_us']['GEMM2 w2'])
        rows['MoE MXFP8 x4 (whole layer)'] = 4 * moe[('mxfp8', m)]['total_us']
        rows['dense qkvr 20 single'] = 20 * g[('qkvr local (1 source)', m)]
        rows['dense qkvr 28 stacked (3M rows)'] = 28 * g[('qkvr stacked 3 sources (serving: 28 of 48 visits)', 3 * m)]
        rows['dense wo x48'] = 48 * g[('wo', m)]
        rows['dense dyn-conv proj 2x8'] = 16 * g[('global dyn-conv weight_proj (32 x 2304)', m)]
        rows['rel-bias einsum x48'] = 48 * g[('rel-bias einsum [t,24,16]x[16,1024]', m)]
        rows['attention SWA x40 (FA4 + rel-bias shear)'] = 40 * att[('swa', m, 0, True)]
        rows['attention global x8 (FA4 + rel-bias shear)'] = 8 * (att[('global', m, 0, True)] if m == 1024
                                                                   else glob(prefix))
        rows['  attention without rel-bias (SWA+global)'] = 40 * att[('swa', m, 0, False)] + 8 * (
            att[('global', m, 0, False)] if m == 1024 else glob(prefix, False))
        rows['DSpark ctx K/V'] = g[('DSpark draft ctx K/V (2048 x 2304)', m)]
        return {k: v * c / m for k, v in rows.items()}

    for n in (1024, 8192, 32768, 131072):
        c = min(n, 8192)
        tot = {}
        for i in range(-(-n // 8192)):
            for k, v in chunk(min(c, n - i * 8192), i * 8192).items():
                tot[k] = tot.get(k, 0.0) + v
        s = sum(v for k, v in tot.items() if not k.startswith('  '))
        meas = MEASURED[n] * 1e6
        print(f'## N={n}: measured TTFT {MEASURED[n]:.2f} s, benchmarked kernels {s/1e6:.2f} s '
              f'({s/meas*100:.0f} %), remainder {(meas-s)/1e6:.2f} s')
        for k, v in tot.items():
            print(f'  {k:<46} {v/1e3:9.0f} ms {v/meas*100:5.1f} %  {v/n:6.1f} us/tok')
        print()


if __name__ == '__main__':
    main()
