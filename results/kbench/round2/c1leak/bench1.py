#!/usr/bin/env python3
"""Track c1leak, run 1 (measure only, no new kernels): cold-L2 microbenchmarks of
  A. pico12's stacked qkvr path at M rows per source (today's code): q rows over the full coded matrix, kv rows
     over the kv pack, the copy; plus the ideal (a coded copy of only the q-side rows) for the byte-once floor;
  B. inkling_draftbytes' _w8_mm_kernel on MXFP8 copies of the draft layer shapes at 1-16 rows: default tile and a
     tile sweep (bit identity of each tile against the default kernel is reported; FP32 order is tile-independent).
Run in /work (overlay copy at /work/overlay), inside the SGLang image."""

import argparse
import glob
import itertools
import os
import statistics
import sys
import time

OV = "/work/overlay"
sys.path.insert(0, OV)

import torch  # noqa: E402

PROJ = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")
DRAFT_SHAPES = {"qkv": (5120, 2304), "o": (2304, 3072), "gate_up": (18432, 2304), "down": (2304, 9216),
                "ctxkv": (2048, 2304)}


def log(*a):
    print(*a, flush=True)


p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=30)
p.add_argument("--layers", default="0,6")
p.add_argument("--ms", default="4,8,16")
p.add_argument("--no-pico", action="store_true")
p.add_argument("--no-draft", action="store_true")
p.add_argument("--sweep-ms", default="8,16")
args = p.parse_args()

dev = torch.device("cuda")
flush = torch.empty(256 << 18, device=dev, dtype=torch.float32).fill_(1.0)  # 256 MiB, read to evict L2
sink = torch.empty((), device=dev)


def timeit(fn, iters=None):
    iters = iters or args.iters
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        sink.copy_(flush.sum())
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1000.0 for s, e in zip(st, en))


def load_qkvr(layer):
    from safetensors import safe_open

    where = {}
    for path in sorted(glob.glob("/model/*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as h:
            for key in h.keys():
                where.setdefault(key, path)
    parts = []
    for pr in PROJ:
        name = f"model.llm.layers.{layer}.attn.{pr}.weight"
        with safe_open(where[name], framework="pt", device="cpu") as h:
            parts.append(h.get_tensor(name))
    return torch.cat(parts).to(dev), [t.shape[0] for t in parts]


def pico_part():
    from inkling_pico12 import build, ops

    t0 = time.monotonic()
    ext = build.load("/root/.cache/c1leak-pico12")
    log(f"pico ext loaded in {time.monotonic() - t0:.1f} s")
    ops.EXT = ext
    ops.SET.wk, ops.SET.pdl = 16, True
    ops.SET.max_m = ops.SET.stacked_max_m = 16
    for layer in [int(v) for v in args.layers.split(",")]:
        w, splits = load_qkvr(layer)
        n, k = w.shape
        lo, hi = ops.kv_bounds(splits)
        full = ops.encode(w, f"L{layer}.qkvr")
        kv = ops.encode(w[lo:hi], f"L{layer}.kv")
        qonly = ops.encode(torch.cat([w[:lo], w[hi:]]).contiguous(), f"L{layer}.qonly")
        plan = ops.QkvrPlan(full, kv, lo, hi)
        log(f"pico L{layer} splits={splits} kv=[{lo}:{hi}] MB full={full.nbytes / 1e6:.2f} kv={kv.nbytes / 1e6:.2f} "
            f"qonly={qonly.nbytes / 1e6:.2f} (bf16 full {n * k * 2 / 1e6:.2f}) raw_tiles full={full.raw_tiles}/{full.tiles}")
        for m in [int(v) for v in args.ms.split(",")]:
            g = torch.Generator(device=dev).manual_seed(100 + m)
            x = (torch.randn(3 * m, k, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            o_full = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
            o_q = torch.empty((m, qonly.n), device=dev, dtype=torch.bfloat16)
            tmp = torch.empty((2 * m, kv.n), device=dev, dtype=torch.bfloat16)
            out3 = torch.empty((3 * m, n), device=dev, dtype=torch.bfloat16)

            def kvcalls():
                for r0 in range(0, 2 * m, 16):
                    r1 = min(2 * m, r0 + 16)
                    ops.gemv(x[m + r0 : m + r1], kv, tmp[r0:r1])

            t_full = timeit(lambda: ops.gemv(x[:m], full, o_full))
            t_q = timeit(lambda: ops.gemv(x[:m], qonly, o_q))
            t_kv = timeit(kvcalls)
            t_cp = timeit(lambda: out3[m:, lo:hi].copy_(tmp))
            t_stk = timeit(lambda: ops.stacked_project(x, m, plan))
            t_ideal = timeit(lambda: (ops.gemv(x[:m], qonly, o_q), kvcalls()))
            log(f"pico L{layer} M={m} full_us={t_full:.1f} ({full.nbytes / t_full / 1e3:.0f} GB/s) "
                f"qonly_us={t_q:.1f} ({qonly.nbytes / t_q / 1e3:.0f} GB/s) kv_us={t_kv:.1f} "
                f"({kv.nbytes * ((2 * m + 15) // 16) / t_kv / 1e3:.0f} GB/s) copy_us={t_cp:.1f} "
                f"stacked_today_us={t_stk:.1f} ideal_qonly+kv_us={t_ideal:.1f} leak_us={t_stk - t_ideal:.1f}")
        del full, kv, qonly, w
        torch.cuda.empty_cache()


def draft_part():
    from inkling_draftbytes import _triton as tk
    from inkling_draftbytes import quant

    packs = {}
    for name, (n, k) in DRAFT_SHAPES.items():
        g = torch.Generator(device=dev).manual_seed(hash(name) & 0xFFFF)
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        packs[name] = quant.quantize(w, "mx")
        del w
    xs = {}

    def xin(m, k):
        key = (m, k)
        if key not in xs:
            g = torch.Generator(device=dev).manual_seed(7 * m + k)
            xs[key] = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
        return xs[key]

    def run(pack, x, bn, bk, warps, stages):
        out = torch.empty((x.shape[0], pack.n), device=dev, dtype=torch.bfloat16)
        kw = {} if stages is None else {"num_stages": stages}
        # rows=0: the default launch; num_stages only takes effect through _stages for rmode != 0, so call the
        # kernel directly when a stage count is asked for
        if stages is None:
            tk.w8_mm(x, pack, None, out, bn=bn, bk=bk, num_warps=warps)
        else:
            w, w_fp8 = tk._w_arg(pack)
            tk._w8_mm_kernel[(triton_cdiv(pack.n, bn), 1)](
                x, w, pack.scale, x, out, x.shape[0], pack.n, x.stride(0), out.stride(0), K=pack.k, G=pack.group,
                BK=tk._bk(pack, bk), MP=tk._mp(x.shape[0]), BN=bn, SCALE=pack.scale_kind, HAS_BIAS=False,
                W_FP8=w_fp8, ROWS=0, NRB=1, num_warps=warps, **kw)
        return out

    def nbytes(pack):
        return pack.nbytes()

    log("draft default tile (BN 32, BK 256, 4 warps, Triton default stages)")
    tot = {}
    for m in (1, 4, 8, 16):
        line = []
        tsum = bsum = 0.0
        for name, pack in packs.items():
            x = xin(m, pack.k)
            t = timeit(lambda: run(pack, x, 32, 256, 4, None))
            line.append(f"{name}={t:.1f}us/{nbytes(pack) / t / 1e3:.0f}GB/s")
            if name != "ctxkv":
                tsum += t
                bsum += nbytes(pack)
        tot[m] = tsum
        log(f"draft default M={m} " + " ".join(line) + f" | 4 layer linears {tsum:.1f} us {bsum / tsum / 1e3:.0f} GB/s "
            f"x3 layers = {3 * tsum / 1e3:.3f} ms")
    best = {}
    for m in [int(v) for v in args.sweep_ms.split(",") if v]:
        for name, pack in packs.items():
            x = xin(m, pack.k)
            ref = run(pack, x, 32, 256, 4, None)
            res = []
            for bn, bk, warps, stages in itertools.product((16, 32, 64), (128, 256, 512), (2, 4, 8),
                                                           (None, 3)):
                try:
                    o = run(pack, x, bn, bk, warps, stages)
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001
                    log(f"draft sweep M={m} {name} bn={bn} bk={bk} w={warps} s={stages} ERROR {type(exc).__name__}")
                    continue
                same = torch.equal(o.view(torch.int16), ref.view(torch.int16))
                t = timeit(lambda: run(pack, x, bn, bk, warps, stages), iters=15)
                res.append((t, bn, bk, warps, stages, same))
            res.sort(key=lambda r: r[0])
            for t, bn, bk, warps, stages, same in res[:5]:
                log(f"draft sweep M={m} {name} bn={bn} bk={bk} warps={warps} stages={stages} {t:.1f} us "
                    f"{nbytes(pack) / t / 1e3:.0f} GB/s bits={'SAME' if same else 'DIFF'}")
            nd = sum(1 for r in res if not r[5])
            log(f"draft sweep M={m} {name}: {len(res)} configs, {nd} not bit-identical to the default")
            best[(m, name)] = res[0]


def triton_cdiv(a, b):
    return (a + b - 1) // b


log(f"bench1 device={torch.cuda.get_device_name()} torch={torch.__version__}")
if not args.no_pico:
    pico_part()
if not args.no_draft:
    draft_part()
log("bench1 done")
