#!/usr/bin/env python3
"""Achieved tensor-core throughput on GB10 for the Inkling-2-Mini PREFILL kernels.

Runs inside lmsysorg/sglang:inkling-2-mini with the model server STOPPED (it allocates ~6 GB of
unified memory). It never touches clocks or power limits; it only reads them via nvidia-smi.

Sections (each runs in its own subprocess, so a CUDA fault in one cannot poison the others):
  env     device, clocks (read-only), library versions
  bw      achievable DRAM bandwidth (device copy), the rate the activation-traffic model uses
  gemm    cuBLAS BF16 (F.linear, the serving path for M > 16 on sm12x) at prefill shapes:
          qkvr 8600x2304 at M=1024/8192, the 3M-row stacked qkvr (M=24576) vs the exact column
          split (q-source 6552 + k 1024 + v 1024 columns, bitwise compared), wo 2304x3072,
          router 128x2304 FP32-out, global dyn-conv proj 32x2304, draft ctx K/V 2048x2304,
          rel-bias einsum, an 8192^3 peak probe, and an FP8 _scaled_mm probe (LOSSY reference only)
  nvfp4   FlashInfer group_gemm_nvfp4_nt_groupwise (CUTLASS SM120, W4A4) on expert-sorted rows,
          T = 1024/4096/8192/16384 tokens x top-8 over 128 experts (uniform and skewed routing),
          w13 (N 2304, K 2304) and w2 (N 2304, K 1152); the stock _tiles() choice plus a sweep of
          (swap_ab, tile_n, tile_k[, tile_m]), one subprocess per config; outputs bitwise vs stock
  mxfp8   FlashInfer moe_gemm_mxfp8_nt_groupwise (CuTe SM120, W8A8) at the same shapes (no tile knobs)
  moe     one full MoE layer as serving runs it at prefill (router GEMM, route_topk, sort-route,
          routed quantize, GEMM1, swiglu-quantize, GEMM2, routed_partial, bf16 cast), per stage
  attn    the serving FA4 SM120 kernel (flash_attn_with_kvcache, paged KV page 128, FP32 out, LSE,
          rel_bias [t,24,1024] bf16 + shear) for chunk 8192 at prefix 0/8K/24K/120K, SWA (1023,0)
          and global causal, with and without rel_bias; chunk 1024 at prefix 0

  python3 bench_tc.py                          # everything (~10-20 min incl. JIT compiles)
  python3 bench_tc.py --sections gemm,attn     # a subset
  python3 bench_tc.py --json /opt/x/bench.json # also write all records + a "rates" block
  python3 bench_tc.py --full-sweep             # NVFP4 sweep also over tile_m 64/128/256

The "rates" block feeds roofline.py --rates. TFLOP/s counts useful FLOPs only (2*M*N*K for GEMMs,
4*heads*head_dim per (query, unmasked key) for attention).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

H, INTER, E, TOPK = 2304, 1152, 128, 8
HEADS, KVH, HD, D_REL, REL_EXT, WINDOW = 24, 8, 128, 16, 1024, 1024
QKVR_SPLITS = [3072, 1024, 1024, 384, 3072, 24]   # q, k, v, rel, gate, sinks (sum 8600)
QKVR_N = sum(QKVR_SPLITS)
MARK = '@@BENCH_TC@@'
SECTIONS = ('env', 'bw', 'gemm', 'nvfp4', 'mxfp8', 'moe', 'attn')
GEMMS = {'w13': (2 * INTER, H), 'w2': (H, INTER)}  # (N, K)


def log(*a):
    print(*a, flush=True)


# ===================================================================== timing


def bench(torch, fn, iters=20, warmup=5, flush=None):
    """Median / min / p90 us of fn() with CUDA events; optional L2 flush before each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush is not None:
            flush[0].copy_(flush[1].sum(dtype=torch.float32))
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    us = sorted(a.elapsed_time(b) * 1e3 for a, b in zip(st, en))
    return dict(med_us=statistics.median(us), min_us=us[0], p90_us=us[int(0.9 * (len(us) - 1))],
                max_us=us[-1])


def make_flush(torch, dev, mb=96):
    return (torch.empty((), device=dev), torch.ones(mb * (1 << 20) // 4, device=dev))


def bitwise(torch, a, b):
    if a.shape != b.shape:
        return f'shape {tuple(a.shape)} vs {tuple(b.shape)}'
    ai, bi = a.contiguous().view(torch.int16 if a.element_size() == 2 else torch.int32), \
        b.contiguous().view(torch.int16 if b.element_size() == 2 else torch.int32)
    if torch.equal(ai, bi):
        return 'bit-equal'
    d = (a.float() - b.float()).abs()
    return f'DIFF {int((ai != bi).sum())} el, max_abs {float(d.max()):.3g}'


# ===================================================================== sections


def sec_env(torch, dev, a):
    import flashinfer
    p = torch.cuda.get_device_properties(dev)
    r = dict(device=p.name, cc=f'{p.major}.{p.minor}', sms=p.multi_processor_count,
             l2_mib=getattr(p, 'L2_cache_size', 0) / 2**20, mem_gib=p.total_memory / 2**30,
             torch=torch.__version__, cuda=torch.version.cuda,
             flashinfer=getattr(flashinfer, '__version__', '?'))
    try:
        import sglang
        r['sglang'] = getattr(sglang, '__version__', '?')
    except Exception as exc:  # noqa: BLE001
        r['sglang'] = f'import failed: {exc}'
    try:
        q = ('clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,'
             'clocks_throttle_reasons.active')
        out = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv'],
                             capture_output=True, text=True, timeout=20)
        r['nvidia_smi'] = out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        r['nvidia_smi'] = f'unavailable: {exc}'
    return [dict(section='env', **r)]


def sec_bw(torch, dev, a):
    out = []
    for mb in (256, 1024):
        n = mb * (1 << 20) // 2
        x = torch.empty(n, dtype=torch.bfloat16, device=dev).normal_()
        y = torch.empty_like(x)
        t = bench(torch, lambda: y.copy_(x), iters=20, warmup=3)
        out.append(dict(section='bw', name=f'device copy {mb} MiB bf16', **t,
                        gbps=2 * n * 2 / (t['med_us'] * 1e3)))
        # a typical prefill elementwise op: fp32 in, bf16 out (gate_sink-like)
        if mb == 256:
            f = torch.empty(n // 2, dtype=torch.float32, device=dev).normal_()
            g = torch.empty(n // 2, dtype=torch.bfloat16, device=dev).normal_()
            o = torch.empty(n // 2, dtype=torch.bfloat16, device=dev)
            t = bench(torch, lambda: torch.mul(f, g, out=o), iters=20, warmup=3)
            out.append(dict(section='bw', name='elementwise fp32*bf16->bf16 (256 MiB-class)', **t,
                            gbps=(n // 2) * (4 + 2 + 2) / (t['med_us'] * 1e3)))
        del x, y
    return out


def sec_gemm(torch, dev, a):
    import torch.nn.functional as F
    out = []

    def rec(name, m, n, k, t, **kw):
        fl = 2.0 * m * n * k
        out.append(dict(section='gemm', name=name, m=m, n=n, k=k, **t,
                        tflops=fl / (t['med_us'] * 1e-6) / 1e12, **kw))
        log(f'  gemm {name:<52} M={m:<6} N={n:<5} K={k:<5} {t["med_us"]:9.1f} us '
            f'{out[-1]["tflops"]:6.1f} TFLOP/s {kw.get("check", "")}')

    g = torch.Generator(device=dev).manual_seed(0)

    def rnd(*s, scale=1.0):
        return (torch.randn(*s, device=dev, generator=g) * scale).to(torch.bfloat16)

    # --- peak probe
    big = 8192
    xa, wb = rnd(big, big, scale=0.05), rnd(big, big, scale=0.05)
    rec('peak probe 8192^3 (F.linear)', big, big, big, bench(torch, lambda: F.linear(xa, wb), 10, 3))
    del xa, wb

    # --- qkvr, wo, router, small
    wq = rnd(QKVR_N, H, scale=0.02)
    wo = rnd(H, 3072, scale=0.02)
    wr = rnd(E, H, scale=0.02)
    wdyn = rnd(KVH * 4, H, scale=0.02)
    wctx = rnd(2048, H, scale=0.02)
    cols = torch.cumsum(torch.tensor([0] + QKVR_SPLITS), 0).tolist()   # q|k|v|rel|gate|sinks
    q_rows = torch.cat([torch.arange(cols[0], cols[1]), torch.arange(cols[3], cols[6])]).to(dev)
    wq_q = wq.index_select(0, q_rows).contiguous()                       # 6552 x 2304
    wq_k = wq[cols[1]:cols[2]].contiguous()
    wq_v = wq[cols[2]:cols[3]].contiguous()
    for m in (1024, 8192, 16384):
        x = rnd(m, H)
        rec('qkvr local (1 source)', m, QKVR_N, H, bench(torch, lambda: F.linear(x, wq)))
        x3 = rnd(3 * m, H)
        y3 = F.linear(x3, wq)
        t3 = bench(torch, lambda: F.linear(x3, wq))
        rec('qkvr stacked 3 sources (serving: 28 of 48 visits)', 3 * m, QKVR_N, H, t3)
        xs = x3.view(3, m, H)

        def split():
            return F.linear(xs[0], wq_q), F.linear(xs[1], wq_k), F.linear(xs[2], wq_v)

        sq, sk, sv = split()
        ref = y3.view(3, m, QKVR_N)
        ref_q = torch.cat([ref[0, :, cols[0]:cols[1]], ref[0, :, cols[3]:cols[6]]], 1)
        chk = ' | '.join([f'q:{bitwise(torch, sq, ref_q)}',
                          f'k:{bitwise(torch, sk, ref[1, :, cols[1]:cols[2]])}',
                          f'v:{bitwise(torch, sv, ref[2, :, cols[2]:cols[3]])}'])
        ts = bench(torch, split)
        out.append(dict(section='gemm', name='qkvr column split (3 GEMMs, same outputs)', m=m,
                        n=QKVR_N, k=H, **ts, tflops=2.0 * m * QKVR_N * H / (ts['med_us'] * 1e-6) / 1e12,
                        check=chk, stacked_us=t3['med_us']))
        log(f'  gemm qkvr column split M={m}: {ts["med_us"]:.1f} us vs stacked {t3["med_us"]:.1f} us '
            f'({out[-1]["tflops"]:.1f} useful TFLOP/s) bitwise {chk}')
        xo = rnd(m, 3072)
        rec('wo', m, H, 3072, bench(torch, lambda: F.linear(xo, wo)))
        rec('router (torch.mm FP32 out)', m, E, H,
            bench(torch, lambda: torch.mm(x, wr.T, out_dtype=torch.float32)))
        rec('global dyn-conv weight_proj (32 x 2304)', m, KVH * 4, H,
            bench(torch, lambda: F.linear(x, wdyn)))
        rec('DSpark draft ctx K/V (2048 x 2304)', m, 2048, H, bench(torch, lambda: F.linear(x, wctx)))
        # rel-bias einsum thd,de->the (bandwidth-bound: writes m*24*1024 bf16)
        r = rnd(m, HEADS, D_REL)
        proj = rnd(D_REL, REL_EXT)
        t = bench(torch, lambda: torch.einsum('thd,de->the', r, proj))
        out.append(dict(section='gemm', name='rel-bias einsum [t,24,16]x[16,1024]', m=m, **t,
                        gbps=m * HEADS * (REL_EXT + D_REL) * 2 / (t['med_us'] * 1e3)))
        log(f'  gemm rel-bias einsum M={m}: {t["med_us"]:.1f} us, {out[-1]["gbps"]:.0f} GB/s')
        del x, x3, y3, xs, sq, sk, sv, ref, ref_q, xo, r
        torch.cuda.empty_cache()

    # --- does the chunk size change dense numerics? rows of an M=16384 GEMM vs the same rows at M=8192
    for nm, w, k in (('qkvr', wq, H), ('wo', wo, 3072)):
        x8 = rnd(8192, k)
        x16 = torch.cat([x8, rnd(8192, k)])
        chk = bitwise(torch, F.linear(x16, w)[:8192], F.linear(x8, w))
        x24 = torch.cat([x8, rnd(16384, k)])
        chk3 = bitwise(torch, F.linear(x24, w)[:8192], F.linear(x8, w))
        out.append(dict(section='gemm', name=f'{nm}: rows at M=16384 / 24576 vs M=8192', m=8192,
                        check=f'M16384 {chk} | M24576 {chk3}'))
        log(f'  gemm {nm} chunk-size numerics: M=16384 {chk}; M=24576 {chk3}')
        del x8, x16, x24

    # --- FP8 dense probe (LOSSY reference: what a quantized dense path could reach)
    try:
        for m, n, k, nm in ((8192, 8608, H, 'qkvr (N padded to 8608)'), (8192, H, 3072, 'wo'), (8192, 8192, 8192, 'peak')):
            xa = rnd(m, k).to(torch.float8_e4m3fn)
            wb = rnd(n, k).to(torch.float8_e4m3fn)
            one = torch.ones((), device=dev, dtype=torch.float32)
            fn = lambda: torch._scaled_mm(xa, wb.t(), scale_a=one, scale_b=one,  # noqa: E731
                                          out_dtype=torch.bfloat16)
            fn()
            rec(f'LOSSY FP8 _scaled_mm {nm} (reference only)', m, n, k, bench(torch, fn, 10, 3))
    except Exception as exc:  # noqa: BLE001
        out.append(dict(section='gemm', name='LOSSY FP8 _scaled_mm', error=str(exc)[:300]))
        log(f'  FP8 _scaled_mm unavailable: {exc}')
    return out


# ------------------------------------------------------------------ MoE helpers


def swizzle_blockscale(torch, scale):
    """[B, M, K] block scales -> CUTLASS 128x4 atom layout (M % 128 == 0, K % 4 == 0)."""
    b, m, k = scale.shape
    s = scale.reshape(b, m // 128, 4, 32, k // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return s.contiguous().reshape(b, m, k)


def make_routing(torch, dev, t, kind, seed):
    """ids [t, 8] int32 (distinct per token), weights [t, 8] fp32."""
    g = torch.Generator(device=dev).manual_seed(seed * 7919 + t)
    logits = torch.rand((t, E), device=dev, generator=g)
    if kind == 'skew':
        bias = torch.randn((E,), device=dev, generator=g) * 0.5
        logits = torch.randn((t, E), device=dev, generator=g) + bias
    w, ids = torch.topk(logits, TOPK, dim=1)
    w = torch.softmax(w.float(), dim=1)
    return ids.to(torch.int32), w


def sort_route(torch, ids, weights):
    """Inkling2Experts._sm120_route for > ROUTE_MAX_IDS ids (every prefill chunk >= 512 tokens)."""
    from sglang.kernels.ops.moe.inkling_moe import compute_grouped_gemm_metadata, get_src2dst
    flat = ids.flatten()
    sorted_ids, dst2src = torch.sort(flat.to(torch.int16), stable=True)
    dst2src = dst2src.int()
    counts, offsets, _, _ = compute_grouped_gemm_metadata(sorted_ids, E, block_size_m=128)
    return get_src2dst(dst2src), dst2src, weights.flatten().float()[dst2src], offsets, counts


class Nvfp4W:
    def __init__(self, torch, dev, n, k, seed):
        g = torch.Generator(device=dev).manual_seed(seed + n + k)
        self.n, self.k = n, k
        self.weight = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g)
        raw = (torch.rand((E, n, k // 16), device=dev, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)
        self.scale = swizzle_blockscale(torch, raw).flatten(1)          # as the model stores it
        self.scale_arg = self.scale.view(torch.uint8).view(E, n, k // 16)
        self.amax = torch.ones(E, device=dev, dtype=torch.float32)
        self.den = torch.ones(1, device=dev, dtype=torch.float32)
        self.bytes_per_expert = n * k // 2 + n * k // 16


class Mxfp8W:
    def __init__(self, torch, dev, n, k, seed):
        from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import mxfp8_weight_scales
        g = torch.Generator(device=dev).manual_seed(seed + 3 * n + k)
        self.n, self.k = n, k
        self.weight = (torch.randn((E, n, k), device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
        ue8m0 = torch.randint(122, 130, (E, n, k // 32), dtype=torch.uint8, device=dev, generator=g)
        self.scale = mxfp8_weight_scales(ue8m0)
        self.bytes_per_expert = n * k + n * k // 32


def moe_inputs(torch, dev, t, kind, seed):
    """Expert-sorted quantized inputs of GEMM1 (w13) and GEMM2 (w2), as serving builds them."""
    from sglang.kernels.ops.moe.inkling2_experts_sm120 import nvfp4_routed_quantize
    ids, w = make_routing(torch, dev, t, kind, seed)
    src2dst, dst2src, gammas, offsets, counts = sort_route(torch, ids, w)
    g = torch.Generator(device=dev).manual_seed(seed + t)
    x = torch.randn((t, H), device=dev, generator=g).to(torch.bfloat16)
    rows = dst2src.numel()
    v13, s13, _ = nvfp4_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
    h = torch.randn((rows, INTER), device=dev, generator=g).to(torch.bfloat16)
    ident = torch.arange(rows, dtype=torch.int32, device=dev)
    v2, s2, _ = nvfp4_routed_quantize(x=h, dst2src=ident, top_k=1, offsets=offsets)
    c = counts.float()
    stats = dict(rows=rows, experts=int((counts > 0).sum()), rows_min=int(c.min()),
                 rows_max=int(c.max()), rows_mean=float(c.mean()))
    return dict(w13=(v13, s13), w2=(v2, s2), offsets=offsets, rows=rows, x=x, h=h, dst2src=dst2src,
                ident=ident, stats=stats)


TOKENS = (1024, 4096, 8192, 16384)
ROUTINGS = ('uniform', 'skew')


def nvfp4_configs(full):
    cfgs = []
    for tm in ((64, 128, 256) if full else (None,)):
        for sw in (1, 0):
            for tn in (32, 64, 128):
                for tk in (128, 256):
                    c = dict(swap_ab=bool(sw), tile_n=tn, tile_k=tk)
                    if tm is not None:
                        c['tile_m'] = tm
                    cfgs.append(c)
    return cfgs


def tag(c):
    return ','.join(f'{k}={int(v) if isinstance(v, bool) else v}' for k, v in c.items())


def sec_nvfp4(torch, dev, a):
    """Child mode: a.only = one config tag, or 'stock'. Returns one record per (gemm, T, routing)."""
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise as fn
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped
    stock_tiles = getattr(grouped._tiles, '__wrapped__', grouped._tiles)
    ws = {gname: Nvfp4W(torch, dev, n, k, a.seed) for gname, (n, k) in GEMMS.items()}
    out = []
    cfg = None if a.only == 'stock' else json.loads(a.only)
    tokens = [int(v) for v in a.tokens.split(',')]
    for t in tokens:
        for kind in ROUTINGS:
            inp = moe_inputs(torch, dev, t, kind, a.seed)
            for gname, w in ws.items():
                vals, scl = inp[gname]
                sw, tn, tk = stock_tiles(tokens=t, width=w.k)
                stock = dict(swap_ab=bool(sw), tile_n=tn, tile_k=tk)
                use = stock if cfg is None else dict(cfg, swap_ab=bool(cfg['swap_ab']))
                o = torch.empty((inp['rows'], w.n), dtype=torch.bfloat16, device=dev)
                ref = torch.empty_like(o)

                def call(c, dst):
                    fn(vals, w.weight, scl, w.scale_arg, inp['offsets'], out=dst, **c)

                rec = dict(section='nvfp4', gemm=gname, n=w.n, k=w.k, tokens=t, routing=kind,
                           cfg=tag(use), is_stock=(use == stock), stock=tag(stock), **inp['stats'])
                try:
                    call(stock, ref)
                    call(use, o)
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001
                    rec['error'] = f'{type(exc).__name__}: {str(exc).splitlines()[0][:200]}'
                    out.append(rec)
                    return out          # a rejected config: skip its other shapes
                rec['bitwise_vs_stock'] = bitwise(torch, o, ref)
                tm = bench(torch, lambda: call(use, o), iters=a.iters, warmup=5)
                fl = 2.0 * inp['rows'] * w.n * w.k
                rec.update(tm, tflops=fl / (tm['med_us'] * 1e-6) / 1e12,
                           w_gbps=inp['stats']['experts'] * w.bytes_per_expert / (tm['med_us'] * 1e3))
                out.append(rec)
            del inp
            torch.cuda.empty_cache()
    return out


def sec_mxfp8(torch, dev, a):
    from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise as fn
    from sglang.kernels.ops.moe.inkling2_experts_sm120 import mxfp8_routed_quantize
    ws = {gname: Mxfp8W(torch, dev, n, k, a.seed) for gname, (n, k) in GEMMS.items()}
    out = []
    for t in [int(v) for v in a.tokens.split(',')]:
        for kind in ROUTINGS:
            ids, wts = make_routing(torch, dev, t, kind, a.seed)
            src2dst, dst2src, gammas, offsets, counts = sort_route(torch, ids, wts)
            rows = dst2src.numel()
            g = torch.Generator(device=dev).manual_seed(a.seed + t)
            x = torch.randn((t, H), device=dev, generator=g).to(torch.bfloat16)
            h = torch.randn((rows, INTER), device=dev, generator=g).to(torch.bfloat16)
            ident = torch.arange(rows, dtype=torch.int32, device=dev)
            q = {'w13': mxfp8_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets),
                 'w2': mxfp8_routed_quantize(x=h, dst2src=ident, top_k=1, offsets=offsets)}
            ne = int((counts > 0).sum())
            for gname, w in ws.items():
                vals, scl = q[gname]
                o = torch.empty((rows, w.n), dtype=torch.bfloat16, device=dev)
                rec = dict(section='mxfp8', gemm=gname, n=w.n, k=w.k, tokens=t, routing=kind,
                           rows=rows, experts=ne, cfg='default (no tile args)')
                try:
                    f = lambda: fn(vals, w.weight, scl, w.scale, offsets,  # noqa: E731
                                   scale_granularity_mnk=(1, 1, 32), out=o)
                    f()
                    torch.cuda.synchronize()
                    tm = bench(torch, f, iters=a.iters, warmup=5)
                    rec.update(tm, tflops=2.0 * rows * w.n * w.k / (tm['med_us'] * 1e-6) / 1e12,
                               w_gbps=ne * w.bytes_per_expert / (tm['med_us'] * 1e3))
                except Exception as exc:  # noqa: BLE001
                    rec['error'] = f'{type(exc).__name__}: {str(exc).splitlines()[0][:200]}'
                out.append(rec)
            del x, h, q
            torch.cuda.empty_cache()
    return out


def sec_moe(torch, dev, a):
    """One full MoE layer, stock serving path at prefill, stages timed with events."""
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped
    from sglang.kernels.ops.moe import inkling2_experts_sm120 as ex
    from sglang.kernels.ops.moe.inkling2_moe import routed_partial
    from sglang.kernels.ops.moe.inkling2_route import route_topk
    out = []
    g = torch.Generator(device=dev).manual_seed(a.seed)
    wg = (torch.randn((E, H), device=dev, generator=g) * 0.02).to(torch.bfloat16)
    bias = torch.zeros(E, device=dev, dtype=torch.float32)
    one = torch.ones((), device=dev, dtype=torch.float32)
    nv = {gname: Nvfp4W(torch, dev, n, k, a.seed) for gname, (n, k) in GEMMS.items()}
    mx = {gname: Mxfp8W(torch, dev, n, k, a.seed) for gname, (n, k) in GEMMS.items()}

    def layer(x, prec, ev):
        ev[0].record()
        partials = torch.mm(x, wg.T, out_dtype=torch.float32)[None]
        _, ids, weights = route_topk(partials, bias, TOPK, 8.0, one)
        ev[1].record()
        src2dst, dst2src, gammas, offsets, _ = sort_route(torch, ids, weights)
        ev[2].record()
        t = x.shape[0]
        if prec == 'nvfp4':
            v, s, amax = ex.nvfp4_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
            ev[3].record()
            gu = grouped.nvfp4_expert_matmul(v, s, nv['w13'].weight, nv['w13'].scale, offsets, t)
            ev[4].record()
            v, s, amax2 = ex.nvfp4_routed_swiglu_quantize(gu, amax, nv['w13'].amax, nv['w13'].den,
                                                          gammas, offsets, 4.0, 25.0)
            ev[5].record()
            down = grouped.nvfp4_expert_matmul(v, s, nv['w2'].weight, nv['w2'].scale, offsets, t)
            ev[6].record()
            o = routed_partial(down, offsets, src2dst, TOPK,
                               nvfp4_scales=(amax2, nv['w2'].amax, nv['w2'].den)).to(x.dtype)
        else:
            v, s = ex.mxfp8_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
            ev[3].record()
            gu = grouped.mxfp8_expert_matmul(v, s, mx['w13'].weight, mx['w13'].scale, offsets)
            ev[4].record()
            v, s = ex.mxfp8_routed_swiglu_quantize(gu, gammas, offsets, 4.0, 25.0)
            ev[5].record()
            down = grouped.mxfp8_expert_matmul(v, s, mx['w2'].weight, mx['w2'].scale, offsets)
            ev[6].record()
            o = routed_partial(down, offsets, src2dst, TOPK).to(x.dtype)
        ev[7].record()
        return o

    stages = ('router GEMM + route_topk', 'sort-route metadata', 'routed quantize', 'GEMM1 w13',
              'swiglu-quantize', 'GEMM2 w2', 'routed_partial combine + cast')
    for prec in ('nvfp4', 'mxfp8'):
        for t in (1024, 8192, 16384):
            x = (torch.randn((t, H), device=dev, generator=g)).to(torch.bfloat16)
            rec = dict(section='moe', precision=prec, tokens=t)
            try:
                per = {s: [] for s in stages}
                tot = []
                for i in range(a.iters + 3):
                    ev = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
                    layer(x, prec, ev)
                    torch.cuda.synchronize()
                    if i < 3:
                        continue
                    for j, s in enumerate(stages):
                        per[s].append(ev[j].elapsed_time(ev[j + 1]) * 1e3)
                    tot.append(ev[0].elapsed_time(ev[7]) * 1e3)
                rec['stages_us'] = {s: statistics.median(v) for s, v in per.items()}
                rec['total_us'] = statistics.median(tot)
                fl = t * TOPK * (2 * H * 2 * INTER + 2 * INTER * H)
                rec['tflops_layer'] = fl / (rec['total_us'] * 1e-6) / 1e12
                rec['us_per_token'] = rec['total_us'] / t
            except Exception as exc:  # noqa: BLE001
                rec['error'] = f'{type(exc).__name__}: {str(exc).splitlines()[0][:300]}'
            out.append(rec)
            del x
            torch.cuda.empty_cache()
    return out


def sec_attn(torch, dev, a):
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache
    out = []
    page = 128
    max_ctx = 131072
    pages = max_ctx // page
    g = torch.Generator(device=dev).manual_seed(a.seed)
    kc = (torch.randn((pages, page, KVH, HD), device=dev, generator=g) * 0.5).to(torch.bfloat16)
    vc = (torch.randn((pages, page, KVH, HD), device=dev, generator=g) * 0.5).to(torch.bfloat16)
    ptab = torch.arange(pages, dtype=torch.int32, device=dev)[None]
    scale = HD ** -0.5
    cases = [(1024, 0)] + [(8192, p) for p in (0, 8192, 24576, 122880)]
    for c, p in cases:
        q = (torch.randn((c, HEADS, HD), device=dev, generator=g)).to(torch.bfloat16)
        rb = (torch.randn((c, HEADS, REL_EXT), device=dev, generator=g) * 0.5).to(torch.bfloat16)
        o = torch.empty((c, HEADS, HD), device=dev, dtype=torch.float32)
        seql = torch.tensor([p + c], dtype=torch.int32, device=dev)
        cuq = torch.tensor([0, c], dtype=torch.int32, device=dev)
        for kind in ('swa', 'global'):
            ws = (WINDOW - 1, 0) if kind == 'swa' else (-1, -1)
            keys = sum(min(p + i + 1, WINDOW) if kind == 'swa' else p + i + 1 for i in range(c))
            fl = 4.0 * HEADS * HD * keys
            for with_bias in (True, False):
                rec = dict(section='attn', kind=kind, chunk=c, prefix=p, ctx=p + c, rel_bias=with_bias,
                           useful_gflop=fl / 1e9)

                def f():
                    return flash_attn_with_kvcache(
                        q=q, k_cache=kc, v_cache=vc, page_table=ptab, cache_seqlens=seql,
                        cu_seqlens_q=cuq, cu_seqlens_k_new=None, max_seqlen_q=c,
                        softmax_scale=scale, causal=True, window_size=ws, softcap=0.0,
                        return_softmax_lse=True, num_splits=1, pack_gqa=False, out=o,
                        rel_bias=rb if with_bias else None,
                        rel_bias_prep_cache={} if with_bias else None, ver=4)
                try:
                    t0 = time.time()
                    f()
                    torch.cuda.synchronize()
                    rec['first_call_s'] = round(time.time() - t0, 1)
                    tm = bench(torch, f, iters=max(5, a.iters // 2), warmup=2)
                    rec.update(tm, tflops=fl / (tm['med_us'] * 1e-6) / 1e12)
                    log(f'  attn {kind:<6} chunk {c:<5} prefix {p:<6} bias={int(with_bias)} '
                        f'{tm["med_us"]/1e3:8.2f} ms {rec["tflops"]:6.1f} TFLOP/s')
                except Exception as exc:  # noqa: BLE001
                    rec['error'] = f'{type(exc).__name__}: {str(exc).splitlines()[0][:300]}'
                    log(f'  attn {kind} {c} {p} bias={with_bias}: {rec["error"]}')
                out.append(rec)
        del q, rb, o
        torch.cuda.empty_cache()
    return out


FUNCS = dict(env=sec_env, bw=sec_bw, gemm=sec_gemm, nvfp4=sec_nvfp4, mxfp8=sec_mxfp8, moe=sec_moe,
             attn=sec_attn)


# ===================================================================== driver


def run_child(a, section, only=''):
    cmd = [sys.executable, os.path.abspath(__file__), '--child', section, '--only', only,
           '--iters', str(a.iters), '--seed', str(a.seed), '--tokens', a.tokens]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        if not ln.startswith(MARK) and ln.strip():
            log(ln)
    got = [ln for ln in p.stdout.splitlines() if ln.startswith(MARK)]
    if p.returncode != 0 or not got:
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:] or ['?']
        log(f'  [{section} {only}] child exit {p.returncode} after {time.time()-t0:.0f}s: '
            + ' / '.join(x[:200] for x in tail))
        return [dict(section=section, only=only, error=f'child exit {p.returncode}: {tail[-1][:300]}')]
    return json.loads(got[-1][len(MARK):])


def report(recs):
    by = lambda s: [r for r in recs if r.get('section') == s]  # noqa: E731
    log('\n================ REPORT ================')
    for r in by('env'):
        log(f"device {r.get('device')} cc {r.get('cc')} SMs {r.get('sms')} L2 {r.get('l2_mib', 0):.0f} MiB "
            f"torch {r.get('torch')} cuda {r.get('cuda')} flashinfer {r.get('flashinfer')} sglang {r.get('sglang')}")
        log(f"nvidia-smi: {r.get('nvidia_smi')}")
    log('\n-- bandwidth')
    for r in by('bw'):
        log(f"  {r['name']:<48} {r.get('gbps', 0):7.1f} GB/s")
    log('\n-- cuBLAS BF16 / misc GEMM')
    for r in by('gemm'):
        if 'error' in r:
            log(f"  {r['name']}: ERROR {r['error']}")
        elif 'tflops' in r:
            log(f"  {r['name']:<56} M={r['m']:<6} {r['med_us']:9.1f} us {r['tflops']:6.1f} TFLOP/s "
                f"{r.get('check', '')}")
        elif 'med_us' not in r:
            log(f"  {r['name']:<56} {r.get('check', '')}")
        else:
            log(f"  {r['name']:<56} M={r['m']:<6} {r['med_us']:9.1f} us {r.get('gbps', 0):6.0f} GB/s")
    nv = [r for r in by('nvfp4') if 'error' not in r and 'tflops' in r]
    bad = {r.get('cfg', r.get('only')): r['error'] for r in by('nvfp4') if 'error' in r}
    if bad:
        log('\n-- NVFP4 configs rejected / failed')
        for k, v in bad.items():
            log(f'  {k}: {v}')
    log('\n-- NVFP4 grouped GEMM (useful TFLOP/s; w GB/s = touched expert weights / time)')
    keys = sorted({(r['gemm'], r['tokens'], r['routing']) for r in nv},
                  key=lambda x: (x[0] != 'w13', x[1], x[2]))
    best = {}
    for key in keys:
        rows = sorted([r for r in nv if (r['gemm'], r['tokens'], r['routing']) == key],
                      key=lambda r: r['med_us'])
        st = next((r for r in rows if r['is_stock']), None)
        r0 = rows[0]
        log(f"  {key[0]} N={r0['n']} K={r0['k']} T={key[1]} {key[2]}: rows {r0['rows']} experts {r0['experts']} "
            f"rows/expert min {r0['rows_min']} mean {r0['rows_mean']:.0f} max {r0['rows_max']}")
        seen = set()
        for r in rows[:6] + ([st] if st and st not in rows[:6] else []):
            if id(r) in seen:
                continue
            seen.add(id(r))
            sp = st['med_us'] / r['med_us'] if st else float('nan')
            log(f"    {r['cfg']:<40} {r['med_us']:9.1f} us {r['tflops']:6.1f} TFLOP/s {r['w_gbps']:6.0f} wGB/s "
                f"x{sp:5.3f} vs stock  {r['bitwise_vs_stock']}{'  <- stock' if r['is_stock'] else ''}")
        exact = [r for r in rows if r['bitwise_vs_stock'] == 'bit-equal']
        best[key] = (st, exact[0] if exact else None)
    log('\n-- MXFP8 grouped GEMM')
    for r in by('mxfp8'):
        if 'error' in r:
            log(f"  {r['gemm']} T={r['tokens']} {r['routing']}: ERROR {r['error']}")
        else:
            log(f"  {r['gemm']} N={r['n']} K={r['k']} T={r['tokens']:<6} {r['routing']:<8} {r['med_us']:9.1f} us "
                f"{r['tflops']:6.1f} TFLOP/s {r['w_gbps']:6.0f} wGB/s")
    log('\n-- full MoE layer (stock serving path), per stage us')
    for r in by('moe'):
        if 'error' in r:
            log(f"  {r['precision']} T={r['tokens']}: ERROR {r['error']}")
            continue
        st = ', '.join(f'{k} {v:.0f}' for k, v in r['stages_us'].items())
        log(f"  {r['precision']} T={r['tokens']:<6} total {r['total_us']:9.1f} us = {r['us_per_token']:.3f} us/token "
            f"({r['tflops_layer']:.1f} TFLOP/s whole layer): {st}")
    log('\n-- FA4 SM120 attention (useful TFLOP/s)')
    for r in by('attn'):
        if 'error' in r:
            log(f"  {r['kind']} chunk {r['chunk']} prefix {r['prefix']} bias={int(r['rel_bias'])}: ERROR {r['error']}")
        elif 'tflops' in r:
            log(f"  {r['kind']:<6} chunk {r['chunk']:<5} prefix {r['prefix']:<6} bias={int(r['rel_bias'])} "
                f"{r['med_us']/1e3:8.2f} ms {r['tflops']:6.1f} TFLOP/s (first call {r.get('first_call_s')} s)")

    # ---- rates for roofline.py
    rates = {}
    try:
        rates['bw'] = max(r['gbps'] for r in by('bw') if 'copy' in r['name'])
    except ValueError:
        pass
    g = {(r['name'], r['m']): r for r in by('gemm') if 'tflops' in r}
    pick = [g.get(('qkvr local (1 source)', 8192)), g.get(('wo', 8192))]
    pick = [p for p in pick if p]
    if pick:
        fl = sum(2.0 * p['m'] * p['n'] * p['k'] for p in pick)
        rates['bf16_gemm'] = fl / sum(p['med_us'] * 1e-6 for p in pick) / 1e12
    for prec, sec in (('nvfp4_group', 'nvfp4'), ('mxfp8_group', 'mxfp8')):
        rs = [r for r in by(sec) if 'tflops' in r and r['tokens'] == 8192 and r['routing'] == 'uniform'
              and (sec == 'mxfp8' or r['is_stock'])]
        fastest = {}
        for r in rs:                     # stock appears twice (stock child + equal sweep config)
            if r['gemm'] not in fastest or r['med_us'] < fastest[r['gemm']]['med_us']:
                fastest[r['gemm']] = r
        rs = list(fastest.values())
        if len(rs) == 2:
            fl = sum(2.0 * r['rows'] * r['n'] * r['k'] for r in rs)
            rates[prec] = fl / sum(r['med_us'] * 1e-6 for r in rs) / 1e12
    for kind in ('swa', 'global'):
        rs = [r for r in by('attn') if 'tflops' in r and r['kind'] == kind and r['rel_bias']
              and r['chunk'] == 8192 and r['prefix'] == 24576]
        if rs:
            rates[f'attn_{kind}'] = rs[0]['tflops']
    log('\nrates (for roofline.py --rates): ' + json.dumps({k: round(v, 1) for k, v in rates.items()}))
    best_exact = {}
    for key, (st, bx) in best.items():
        if st and bx:
            best_exact['/'.join(map(str, key))] = dict(stock=st['cfg'], stock_us=st['med_us'],
                                                       best=bx['cfg'], best_us=bx['med_us'])
    if best_exact:
        log('\nNVFP4 best bit-exact config vs stock per (gemm, T, routing):')
        for k, v in best_exact.items():
            log(f"  {k:<22} stock {v['stock']:<34} {v['stock_us']:9.1f} us | best {v['best']:<34} "
                f"{v['best_us']:9.1f} us ({(1 - v['best_us'] / v['stock_us']) * 100:5.1f} % less)")
    return rates, best_exact


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sections', default=','.join(SECTIONS))
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tokens', default=','.join(map(str, TOKENS)), help='MoE token counts')
    ap.add_argument('--full-sweep', action='store_true', help='NVFP4 sweep over tile_m 64/128/256 too')
    ap.add_argument('--no-sweep', action='store_true', help='NVFP4: stock tiles only')
    ap.add_argument('--json', default='')
    ap.add_argument('--child', default='', help=argparse.SUPPRESS)
    ap.add_argument('--only', default='', help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.child:
        import torch
        dev = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(dev)
        res = FUNCS[a.child](torch, dev, a)
        print(MARK + json.dumps(res, default=str), flush=True)
        return

    recs = []
    t0 = time.time()
    for s in [x for x in a.sections.split(',') if x]:
        log(f'== {time.strftime("%H:%M:%S")} section {s}')
        if s == 'nvfp4':
            cfgs = ['stock'] + ([] if a.no_sweep else [json.dumps(c) for c in nvfp4_configs(a.full_sweep)])
            for c in cfgs:
                ts = time.time()
                got = run_child(a, s, c)
                recs.extend(got)
                ok = all('error' not in r for r in got)
                log(f'  nvfp4 {c}: {"ok" if ok else "FAILED/REJECTED"} ({time.time() - ts:.0f}s)')
        else:
            recs.extend(run_child(a, s))
    rates, best_exact = report(recs)
    log(f'\ntotal {time.time() - t0:.0f} s')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dict(records=recs, rates=rates, nvfp4_best_exact=best_exact), f, indent=1,
                      default=str)
        log(f'wrote {a.json}')


if __name__ == '__main__':
    main()
