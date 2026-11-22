#!/usr/bin/env python3
"""Track c1leak, run 2: GPU check + cold-L2 microbenchmarks of the two new kernels.
  A. inkling_pico12 `stack1` (csrc/pico12_range.cu): the stacked qkvr call reading the coded qkvr once.
     Gated: kept elements == today's stacked_project (kv pack) == per-source gemv, bit for bit; no write outside
     the kept columns (sentinel); CUDA-graph replay == eager. Timed: today vs stack1 (two launches, kv_mt 1/2) vs
     stack1 fused (one launch).
  B. inkling_draftbytes `w8gemv` (csrc/w8gemv.cu) on MXFP8 packs of the draft shapes at 1-16 rows.
     Gated: == Triton _w8_mm_kernel (default tile) bit for bit, every variant, with / without bias, strided x,
     wide scale range; graph replay. Timed: every variant vs Triton default and Triton BN 16.
Last line: c1leak bench2 result PASS|FAIL."""

import argparse
import glob
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
FAILS = []


def log(*a):
    print(*a, flush=True)


def gate(name, ok, detail=""):
    log(f"{name} {detail} {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(name)


p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=30)
p.add_argument("--layers", default="0,6")
p.add_argument("--no-pico", action="store_true")
p.add_argument("--no-draft", action="store_true")
p.add_argument("--no-time", action="store_true")
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


def bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def graph_of(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


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
    rext = build.load_range("/root/.cache/c1leak-pico12")
    log(f"pico ext+range loaded in {time.monotonic() - t0:.1f} s ({build.INFO.get('range_build_dir')})")
    ops.EXT, ops.RANGE_EXT = ext, rext
    ops.SET.wk, ops.SET.pdl = 16, True
    ops.SET.max_m = ops.SET.stacked_max_m = 16
    SENT = -12345.0
    for layer in [int(v) for v in args.layers.split(",")]:
        w, splits = load_qkvr(layer)
        n, k = w.shape
        lo, hi = ops.kv_bounds(splits)
        full = ops.encode(w, f"L{layer}.qkvr")
        kv = ops.encode(w[lo:hi], f"L{layer}.kv")
        plan_old = ops.QkvrPlan(full, kv, lo, hi)
        plan_new = ops.QkvrPlan(full, None, lo, hi)
        cols_q = torch.tensor([c for c in range(n) if not lo <= c < hi], device=dev)
        klo, khi = lo, lo + splits[1]

        def kept(o3, m):
            return torch.cat([o3[:m].index_select(1, cols_q).flatten(), o3[m : 2 * m, klo:khi].flatten(),
                              o3[2 * m :, khi:hi].flatten()])

        def written_mask_ok(o3, m):
            """outside the columns each row block owns, the sentinel is untouched"""
            a = o3[:m, lo:hi]
            b = torch.cat([o3[m:, :lo], o3[m:, hi:]], dim=1)
            return bool((a == SENT).all()) and bool((b == SENT).all())

        for m in (6, 8, 12, 16):
            g = torch.Generator(device=dev).manual_seed(300 + m)
            x = (torch.randn(3 * m, k, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            per = torch.empty((3 * m, n), device=dev, dtype=torch.bfloat16)
            for s in range(3):
                ops.gemv(x[s * m : (s + 1) * m], full, per[s * m : (s + 1) * m])
            ops.SET.stack1 = False
            old = ops.stacked_project(x, m, plan_old)
            torch.cuda.synchronize()
            ok_old = bits_equal(kept(old, m), kept(per, m))
            variants = []
            for fuse in (True, False):
                for kv_mt in (1, 2):
                    if fuse and (m > 8 or kv_mt == 2):
                        continue
                    variants.append((fuse, kv_mt))
            ops.SET.stack1 = True
            for fuse, kv_mt in variants:
                ops.SET.stack1_fuse, ops.SET.stack1_kv_mt = fuse, kv_mt
                new = ops.stacked_project(x, m, plan_new)
                # sentinel run: the same launches into a pre-filled buffer
                sent = torch.full((3 * m, n), SENT, device=dev, dtype=torch.bfloat16)
                if fuse and m <= 8:
                    rext.gemv_stacked(x, full.buf, n, k, sent, full.raw_tiles, m, 16, True, lo, hi)
                else:
                    rext.gemv_range(x[:m], full.buf, n, k, sent[:m], full.raw_tiles, ops._range_mt(full.mt, lo, hi), 16, True, lo, hi,
                                    False)
                    for r0 in range(m, 3 * m, 16):
                        r1 = min(3 * m, r0 + 16)
                        rext.gemv_range(x[r0:r1], full.buf, n, k, sent[r0:r1], full.raw_tiles, ops._range_mt(kv_mt, lo, hi), 16, True,
                                        lo, hi, True)
                gph, og = graph_of(lambda: ops.stacked_project(x, m, plan_new))
                og.fill_(float("nan"))
                gph.replay()
                torch.cuda.synchronize()
                same = bits_equal(kept(new, m), kept(per, m)) and bits_equal(kept(new, m), kept(old, m))
                same_s = bits_equal(kept(sent, m), kept(per, m))
                untouched = written_mask_ok(sent, m)
                same_g = bits_equal(kept(og, m), kept(new, m))
                gate(f"pico stack1 L{layer} 3x{m} fuse={int(fuse)} kv_mt={kv_mt}",
                     ok_old and same and same_s and untouched and same_g,
                     f"old_vs_per={ok_old} new_vs_per_and_old={same} sentinel_kept={same_s} "
                     f"sentinel_untouched={untouched} graph={same_g}")
                del gph, og
            if not args.no_time:
                ops.SET.stack1 = False
                t_old = timeit(lambda: ops.stacked_project(x, m, plan_old))
                o_full = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
                t_full = timeit(lambda: ops.gemv(x[:m], full, o_full))
                ops.SET.stack1 = True
                line = []
                for fuse, kv_mt in variants:
                    ops.SET.stack1_fuse, ops.SET.stack1_kv_mt = fuse, kv_mt
                    t = timeit(lambda: ops.stacked_project(x, m, plan_new))
                    line.append(f"fuse{int(fuse)}/kvmt{kv_mt}={t:.1f}us({full.nbytes * (1 if 2 * m <= 16 else 1) / t / 1e3:.0f}GB/s)")
                log(f"pico time L{layer} 3x{m} today={t_old:.1f}us one_full_call={t_full:.1f}us " + " ".join(line)
                    + f"  [coded bytes once = {full.nbytes / 1e6:.2f} MB; today reads {(full.nbytes + kv.nbytes * ((2 * m + 15) // 16)) / 1e6:.2f} MB]")
        del full, kv, w
        torch.cuda.empty_cache()


def draft_part():
    from inkling_draftbytes import _triton as tk
    from inkling_draftbytes import quant, w8gemv

    t0 = time.monotonic()
    w8gemv.load("/root/.cache/c1leak-w8gemv")
    log(f"w8gemv ext loaded in {time.monotonic() - t0:.1f} s ({w8gemv.INFO.get('build_dir')})")
    packs = {}
    for name, (n, k) in DRAFT_SHAPES.items():
        g = torch.Generator(device=dev).manual_seed(hash(name) & 0xFFFF)
        w = torch.randn(n, k, device=dev, generator=g) * 0.02
        if name == "o":  # a wide scale range: rows scaled by 2^-40 .. 2^+20, one all-zero 32-block per row
            w = w * torch.exp2(torch.randint(-40, 21, (n, 1), device=dev, generator=g).float())
            w[:, 64:96] = 0
        packs[name] = quant.quantize(w.to(torch.bfloat16), "mx")
        del w
    log("w8gemv scale byte range per pack: " + " ".join(
        f"{nm}=[{int(pk.scale.min())},{int(pk.scale.max())}]" for nm, pk in packs.items()))

    def triton(pack, x, bias=None, bn=32, bk=256, warps=4):
        out = torch.empty((x.shape[0], pack.n), device=dev, dtype=torch.bfloat16)
        tk.w8_mm(x, pack, bias, out, bn=bn, bk=bk, num_warps=warps)
        return out

    def ours(pack, x, bias, cfg):
        out = torch.empty((x.shape[0], pack.n), device=dev, dtype=torch.bfloat16)
        w8gemv._EXT.w8gemv(x, pack.q, pack.scale, bias, out, cfg)
        return out

    for name, pack in packs.items():
        bad = []
        for m in (1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16):
            g = torch.Generator(device=dev).manual_seed(11 * m)
            xbig = torch.randn(m, pack.k + 64, device=dev, generator=g).to(torch.bfloat16)
            for strided in (False, True):
                x = xbig[:, : pack.k] if strided else xbig[:, : pack.k].contiguous()
                for use_bias in (False, True):
                    bias = (torch.randn(pack.n, device=dev, generator=g) * 0.1).to(torch.bfloat16) if use_bias else None
                    ref = triton(pack, x, bias)
                    cfgs = w8gemv.VARIANTS if m in (1, 8, 9, 16) or not strided else (w8gemv.DEFAULT_CFG,)
                    for cfg in cfgs:
                        o = ours(pack, x, bias, cfg)
                        torch.cuda.synchronize()
                        if not bits_equal(o, ref):
                            nd = int((o.view(torch.int16) != ref.view(torch.int16)).sum())
                            bad.append(f"M={m} strided={strided} bias={use_bias} cfg={cfg} ndiff={nd} "
                                       f"maxabs={float((o.float() - ref.float()).abs().max()):.3e}")
        gate(f"w8gemv bits {name} [{pack.n}x{pack.k}]", not bad, f"vs_triton {'IDENTICAL' if not bad else bad[:4]}")
        # graph replay
        x = torch.randn(8, pack.k, device=dev).to(torch.bfloat16)
        gph, og = graph_of(lambda: ours(pack, x, None, w8gemv.DEFAULT_CFG))
        og.fill_(float("nan"))
        gph.replay()
        torch.cuda.synchronize()
        gate(f"w8gemv graph {name}", bits_equal(og, triton(pack, x)), "")
        del gph, og

    if args.no_time:
        return
    tot = {}
    for m in (1, 8, 16):
        for name, pack in packs.items():
            x = torch.randn(m, pack.k, device=dev).to(torch.bfloat16)
            nb = pack.nbytes()
            t_def = timeit(lambda: triton(pack, x))
            t_16 = timeit(lambda: triton(pack, x, bn=16, bk=128 if name in ("o", "down", "ctxkv") else 256))
            res = []
            for cfg in w8gemv.VARIANTS:
                res.append((timeit(lambda: ours(pack, x, None, cfg)), cfg))
            res.sort()
            tot[(m, name)] = (t_def, t_16, res[0])
            log(f"w8gemv time M={m} {name} [{pack.n}x{pack.k}] {nb / 1e6:.2f} MB: triton_default={t_def:.1f}us "
                f"({nb / t_def / 1e3:.0f} GB/s) triton_bn16={t_16:.1f}us ({nb / t_16 / 1e3:.0f} GB/s) | "
                + " ".join(f"{c}={t:.1f}us/{nb / t / 1e3:.0f}" for t, c in res[:6])
                + " | all: " + " ".join(f"{c}:{t:.0f}" for t, c in sorted(res, key=lambda r: r[1])))
    for m in (24, 32, 64):  # rows the Triton kernel keeps: default tile vs BN 16 (bit identity and time)
        line = []
        for name, pack in packs.items():
            x = torch.randn(m, pack.k, device=dev).to(torch.bfloat16)
            bk16 = 128 if name in ("o", "down", "ctxkv") else 256
            same = bits_equal(triton(pack, x), triton(pack, x, bn=16, bk=bk16))
            t_def = timeit(lambda: triton(pack, x))
            t_16 = timeit(lambda: triton(pack, x, bn=16, bk=bk16))
            line.append(f"{name}={t_def:.1f}->{t_16:.1f}us({'same' if same else 'DIFF'})")
        log(f"triton M={m} default->bn16: " + " ".join(line))
    for m in (1, 8, 16):
        names = ("qkv", "o", "gate_up", "down")
        a = sum(tot[(m, nm)][0] for nm in names)
        b = sum(tot[(m, nm)][1] for nm in names)
        c = sum(tot[(m, nm)][2][0] for nm in names)
        nb = sum(packs[nm].nbytes() for nm in names)
        log(f"w8gemv layer sum M={m}: triton_default={a:.1f}us ({nb / a / 1e3:.0f} GB/s) triton_bn16={b:.1f}us "
            f"({nb / b / 1e3:.0f} GB/s) w8gemv_best={c:.1f}us ({nb / c / 1e3:.0f} GB/s); x3 layers saves "
            f"{3 * (a - c) / 1e3:.3f} ms vs default")


import triton  # noqa: E402
log(f"bench2 device={torch.cuda.get_device_name()} torch={torch.__version__} triton={triton.__version__}")
if not args.no_pico:
    pico_part()
if not args.no_draft:
    draft_part()
log(f"c1leak bench2 result {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failed: {FAILS[:6]})")
