#!/usr/bin/env python3
"""Inkling-2-Mini routed-expert grouped GEMMs on GB10 with real routing: time + bit-identity per config.

Runs inside the SGLang image (kbrun.sh). Verify-step routings come from steps.npz (make_steps.py: the
CPU oracle's real top-8 ids of 8 consecutive greedy positions per request, B requests per step, at
one MoE visit). For each step the inputs are built with SGLang's own route_metadata and routed
quantize kernels, and the GEMMs are called exactly as sglang/kernels/ops/gemm/inkling2_grouped_sm120.py
does:
  NVFP4 (layers 0-19): flashinfer group_gemm_nvfp4_nt_groupwise, every (swap_ab, tile_n, tile_k) the
        API accepts; reference = the SHIPPED tiles (stock _tiles + the shipped INKLING_MOE_TILES rules).
  MXFP8 (layers 20-23): flashinfer moe_gemm_mxfp8_nt_groupwise as SGLang calls it (its C++ tile
        heuristic) = reference, and each explicit tactic (tile_m, tile_n) of the module's
        moe_gemm_mxfp8_nt_groupwise_tuned entry.
Every config's output is compared bitwise (int16 view) with the reference on the same inputs.
Timing: CUDA events around one call, a READ of a >= 128 MiB buffer before every call (cold L2, as the
weights are in serving), median over --iters. GB/s = union x weight bytes per expert / time.

  python3 bench_moe2.py --steps steps.npz [--real /model] [--kinds nvfp4,mxfp8] [--json out.json]
"""

import argparse
import json
import os
import statistics
import sys
import time

E, TOPK, HIDDEN, INTER = 128, 8, 2304, 1152
GEMMS = {"w13": (2 * INTER, HIDDEN), "w2": (HIDDEN, INTER)}  # name -> (N, K)
# Shipped INKLING_MOE_TILES (scripts/launch.sh MOE_TILES default), applied over stock _tiles.
SHIPPED_RULES = ("1@2304:swap_ab=1,tile_n=32,tile_k=256;1@1152:swap_ab=1,tile_n=32,tile_k=256;"
                 "8@2304:swap_ab=1,tile_n=32,tile_k=256;8@1152:swap_ab=1,tile_n=32,tile_k=256;"
                 "64@2304:swap_ab=0,tile_n=64,tile_k=256;64@1152:swap_ab=1,tile_n=32,tile_k=256")
MX_TACTICS = ((32, 128), (64, 64), (64, 128), (128, 128), (128, 8))


def log(*a):
    print(*a, flush=True)


def stock_tiles(tokens, width):  # sglang inkling2_grouped_sm120._tiles (checked against the image)
    if tokens <= 128:
        return True, 32, 128
    if tokens <= 1536:
        return True, 32, 256
    if tokens <= 3072:
        return True, 64, 256
    return (False, 128, 256) if width >= 2048 else (True, 128, 128)


def parse_rules(spec):
    rules = []
    for raw in spec.split(";"):
        head, body = raw.split(":")
        tok, width = (head.split("@") + [None])[:2]
        vals = dict(kv.split("=") for kv in body.split(","))
        rules.append((int(tok), int(width) if width else None, {k: int(v) for k, v in vals.items()}))
    rules.sort(key=lambda r: (r[0], r[1] is None))
    return rules


def shipped_tiles(rules, tokens, width):
    sw, tn, tk = stock_tiles(tokens, width)
    for tok, w, v in rules:
        if tokens <= tok and (w is None or w == width):
            return bool(v.get("swap_ab", sw)), v.get("tile_n", tn), v.get("tile_k", tk)
    return sw, tn, tk


def swizzle_blockscale(scale):
    """[B, M, K] -> CUTLASS 128x4 atom layout (sglang.srt.layers.quantization.utils.swizzle_blockscale)."""
    b, m, k = scale.shape
    s = scale.reshape(b, m // 128, 4, 32, k // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return s.contiguous().reshape(b, m, k)


def load_real(path, names, torch, device):
    from safetensors import safe_open

    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors")]
    out = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as h:
            keys = set(h.keys())
            for n in names:
                if n in keys:
                    out[n] = h.get_tensor(n).to(device)
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"missing tensors {missing}")
    return out


class Nvfp4:
    kind = "nvfp4"

    def __init__(self, torch, device, real, layer, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        self.w = {}
        if real:
            pre = f"model.llm.layers.{layer}.mlp.experts."
            t = load_real(real, [pre + n for n in ("w13_weight", "w13_weight_scale", "w2_weight",
                                                   "w2_weight_scale")], torch, device)
        for name, (n, k) in GEMMS.items():
            if real:
                wt = t[pre + name + "_weight"]
                sc = t[pre + name + "_weight_scale"].view(torch.float8_e4m3fn)
            else:
                wt = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=device, generator=g)
                sc = (torch.rand((E, n, k // 16), device=device, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)
            assert tuple(wt.shape) == (E, n, k // 2), wt.shape
            sw = swizzle_blockscale(sc.view(torch.uint8)).flatten(1)
            self.w[name] = (wt.contiguous(), sw.view(torch.uint8).view(E, n, k // 16), n * (k // 2) + n * (k // 16))
        from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
        self.fn = group_gemm_nvfp4_nt_groupwise

    def configs(self):
        return [dict(swap_ab=sw, tile_n=tn, tile_k=tk) for sw in (True, False) for tn in (32, 64, 128)
                for tk in (128, 256)]

    def inputs(self, torch, ids, device, seed):
        from sglang.kernels.ops.moe.inkling2_experts_sm120 import nvfp4_routed_quantize
        from sglang.kernels.ops.moe.inkling2_route import route_metadata

        t = ids.shape[0]
        g = torch.Generator(device=device).manual_seed(seed)
        wts = torch.rand((t, TOPK), device=device, generator=g)
        _, dst2src, _, counts, offsets, *_ = route_metadata(ids.flatten(), wts.flatten(), 0, E, E)
        rows = dst2src.numel()
        x = torch.randn((t, HIDDEN), device=device, generator=g).to(torch.bfloat16)
        v13, s13, _ = nvfp4_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
        h = torch.randn((rows, INTER), device=device, generator=g).to(torch.bfloat16)
        ident = torch.arange(rows, dtype=torch.int32, device=device)
        v2, s2, _ = nvfp4_routed_quantize(x=h, dst2src=ident, top_k=1, offsets=offsets)
        return dict(tokens=t, rows=rows, offsets=offsets, union=int((counts > 0).sum()),
                    w13=(v13, s13), w2=(v2, s2))

    def run(self, inp, gemm, cfg, out):
        v, s = inp[gemm]
        wt, ws, _ = self.w[gemm]
        self.fn(v, wt, s, ws, inp["offsets"], tile_n=cfg["tile_n"], tile_k=cfg["tile_k"],
                swap_ab=cfg["swap_ab"], out=out)


class Mxfp8:
    kind = "mxfp8"

    def __init__(self, torch, device, real, layer, seed):
        from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import mxfp8_weight_scales

        g = torch.Generator(device=device).manual_seed(seed + 1)
        self.w = {}
        if real:
            pre = f"model.llm.layers.{layer}.mlp.experts."
            t = load_real(real, [pre + n for n in ("w13_weight", "w13_weight_scale", "w2_weight",
                                                   "w2_weight_scale")], torch, device)
        for name, (n, k) in GEMMS.items():
            if real:
                wt = t[pre + name + "_weight"].view(torch.float8_e4m3fn)
                ue = t[pre + name + "_weight_scale"].view(torch.uint8)
            else:
                wt = (torch.randn((E, n, k), device=device, generator=g) * 0.5).to(torch.float8_e4m3fn)
                ue = torch.randint(122, 130, (E, n, k // 32), dtype=torch.uint8, device=device, generator=g)
            assert tuple(wt.shape) == (E, n, k), wt.shape
            self.w[name] = (wt.contiguous(), mxfp8_weight_scales(ue.contiguous()), n * k + n * (k // 32))
        from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise
        from flashinfer.grouped_mm.cute_sm120_mxfp8_groupwise.core import get_gemm_sm120_module_cute_mxfp8
        self.fn = moe_gemm_mxfp8_nt_groupwise
        self.mod = get_gemm_sm120_module_cute_mxfp8()

    def configs(self):
        return [dict(tactic="default")] + [dict(tile_m=m, tile_n=n) for m, n in MX_TACTICS]

    def inputs(self, torch, ids, device, seed):
        from sglang.kernels.ops.moe.inkling2_experts_sm120 import mxfp8_routed_quantize
        from sglang.kernels.ops.moe.inkling2_route import route_metadata

        t = ids.shape[0]
        g = torch.Generator(device=device).manual_seed(seed)
        wts = torch.rand((t, TOPK), device=device, generator=g)
        _, dst2src, _, counts, offsets, *_ = route_metadata(ids.flatten(), wts.flatten(), 0, E, E)
        rows = dst2src.numel()
        x = torch.randn((t, HIDDEN), device=device, generator=g).to(torch.bfloat16)
        a13 = mxfp8_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
        h = torch.randn((rows, INTER), device=device, generator=g).to(torch.bfloat16)
        ident = torch.arange(rows, dtype=torch.int32, device=device)
        a2 = mxfp8_routed_quantize(x=h, dst2src=ident, top_k=1, offsets=offsets)
        return dict(tokens=t, rows=rows, offsets=offsets, union=int((counts > 0).sum()), w13=a13, w2=a2)

    def run(self, inp, gemm, cfg, out):
        v, s = inp[gemm]
        wt, ws, _ = self.w[gemm]
        if cfg.get("tactic") == "default":  # the C++ heuristic (what serving ran: no autotune cache entry)
            self.mod.moe_gemm_mxfp8_nt_groupwise(v, wt, s, ws, inp["offsets"], out, "MN", 1, 1, 32, False)
        else:
            self.mod.moe_gemm_mxfp8_nt_groupwise_tuned(v, wt, s, ws, inp["offsets"], out, "MN", 1, 1, 32,
                                                       False, cfg["tile_m"], cfg["tile_n"])


def quant_bench(torch, np, z, a, dev):
    """Routed NVFP4 input quantize at decode token counts: stock nvfp4_routed_quantize vs the shipped
    inkling_prefill quant_once (quantize each token once, gather to its routed rows; shipped gated at
    >= 256 tokens by INKLING_PREFILL_QUANT_MIN_T). Bitwise: values, amax, and the w13 GEMM output
    (reads every scale byte the GEMM uses). Warm timing (x was just written in serving)."""
    sys.path.insert(0, a.overlay)
    from inkling_prefill import quant_once as qo
    from sglang.kernels.ops.moe import inkling2_experts_sm120 as ex
    from sglang.kernels.ops.moe.inkling2_route import route_metadata
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

    stock = ex.nvfp4_routed_quantize
    g = torch.Generator(device=dev).manual_seed(7)
    n, k = GEMMS["w13"]
    wt = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    sc = (torch.rand((E, n, k // 16), device=dev, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)
    ws = swizzle_blockscale(sc.view(torch.uint8)).flatten(1).view(E, n, k // 16)
    log("\n=== routed NVFP4 quantize: stock vs quant_once (warm, median us)")
    for bs in [int(b) for b in a.bs.split(",")]:
        key = f"ids_nvfp4_{bs}_full"
        if key not in z.files:
            continue
        tot = {"stock": 0.0, "once": 0.0}
        ok = True
        for si, ids_np in enumerate(z[key][:3]):
            ids = torch.from_numpy(ids_np).to(dev)
            t = ids.shape[0]
            wts = torch.rand((t, TOPK), device=dev, generator=g)
            _, dst2src, _, counts, offsets, *_ = route_metadata(ids.flatten(), wts.flatten(), 0, E, E)
            x = torch.randn((t, HIDDEN), device=dev, generator=g).to(torch.bfloat16)
            fns = {"stock": lambda: stock(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets),
                   "once": lambda: qo.quant_once(stock, ex._nvfp4_outputs, x, dst2src, TOPK, offsets)}
            outs = {}
            for name, f in fns.items():
                v, s, am = f()
                o = torch.empty((dst2src.numel(), n), dtype=torch.bfloat16, device=dev)
                group_gemm_nvfp4_nt_groupwise(v, wt, s, ws, offsets, tile_n=32, tile_k=256, swap_ab=True, out=o)
                outs[name] = (v, am, o)
                for _ in range(5):
                    f()
                st = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters)]
                en = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters)]
                for i in range(a.iters):
                    st[i].record()
                    f()
                    en[i].record()
                torch.cuda.synchronize()
                tot[name] += statistics.median(sorted(s_.elapsed_time(e_) * 1e3 for s_, e_ in zip(st, en)))
            same = (torch.equal(outs["stock"][0], outs["once"][0])
                    and torch.equal(outs["stock"][1].view(torch.int32), outs["once"][1].view(torch.int32))
                    and torch.equal(outs["stock"][2].view(torch.int16), outs["once"][2].view(torch.int16)))
            ok = ok and same
        log(f"  bs {bs:2d} ({bs * 8} tokens): stock {tot['stock'] / 3:6.1f} us  quant_once {tot['once'] / 3:6.1f} us "
            f"-> x{tot['stock'] / tot['once']:.2f}  bit-identical (values, amax, w13 out): {ok}")


def swiglu_bench(torch, z, a, dev):
    """inkling_moe2's multi-row swiglu quantize (csrc/swiglu_quant_mr.cuh) vs the stock JIT kernel on the same
    pre-filled output buffers (every byte compared, written or not). Warm timing (acc was just written)."""
    import pathlib
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
    from sglang.kernels.ops.moe import inkling2_experts_sm120 as ex
    from sglang.kernels.ops.moe.inkling2_route import route_metadata

    src = str((pathlib.Path(a.overlay) / "inkling_moe2" / "csrc" / "swiglu_quant_mr.cuh").resolve())
    mods = {}

    def mr(r):
        if r not in mods:
            t0 = time.time()
            mods[r] = load_jit("inkling_moe2_swiglu_mr", str(INTER), str(r), cuda_files=[src],
                               cuda_wrappers=[("run", f"sglang::inkling_moe2::nvfp4_swiglu_quantize_mr<{INTER}, {r}>")],
                               extra_include_paths=[str(KERNEL_PATH / "csrc")])
            log(f"  built R={r} in {time.time() - t0:.0f}s")
        return mods[r]

    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    stock_mod = ex._module(INTER)
    g = torch.Generator(device=dev).manual_seed(11)
    w_amax = torch.rand(E, device=dev, generator=g) + 0.5
    w_den = torch.tensor([2.5], device=dev)
    log("\n=== NVFP4 swiglu quantize: stock vs multi-row (warm, median us; caps 4/25 as config.json)")
    variants = [(r, c) for r in (2, 4, 8) for c in (4 * sms, 8 * sms)]
    # Edge cases first (review): offsets[0] > 0, empty experts, experts of 127/128/129 rows, a partial last
    # group, one CTA (parity reuse over many iterations), and caps off; every byte of all three outputs compared.
    edge_fail = 0
    for offs, ctas, caps in (([3, 3, 130, 258, 387, 388], 1, (4.0, 25.0)), ([3, 3, 130, 258, 387, 388], 3, (0.0, 0.0)),
                             ([0, 1, 1, 129, 257, 300], 2, (4.0, 25.0)), ([0, 0, 0, 0, 0, 5], 1, (4.0, 25.0)),
                             ([0, 127, 255, 384, 512, 513], 7, (4.0, 25.0))):
        ne = len(offs) - 1
        rows = offs[-1]
        offsets = torch.tensor(offs, dtype=torch.int32, device=dev)
        acc = (torch.randn((rows, 2 * INTER), device=dev, generator=g) * 6).to(torch.bfloat16)
        acc_amax = torch.rand(rows, device=dev, generator=g) + 0.5
        gam = torch.rand(rows, device=dev, generator=g)
        wam = torch.rand(ne, device=dev, generator=g) + 0.5
        srows = -(-(rows + 127 * ne) // 128) * 128
        outs = []
        for r in (None, 2, 4, 8):
            o = (torch.full((rows, INTER // 2), 0xAB, dtype=torch.uint8, device=dev),
                 torch.full((srows, INTER // 16), 0xCD, dtype=torch.uint8, device=dev),
                 torch.full((rows,), float("nan"), device=dev))
            if r is None:
                stock_mod.nvfp4_swiglu_quantize(acc, acc_amax, wam, w_den, gam, *o, offsets, *caps, sms)
            else:
                mr(r).run(acc, acc_amax, wam, w_den, gam, *o, offsets, *caps, ctas)
            torch.cuda.synchronize()
            outs.append(o)
        for r, o in zip((2, 4, 8), outs[1:]):
            if not all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(o, outs[0])):
                edge_fail += 1
                log(f"  EDGE MISMATCH offsets={offs} ctas={ctas} caps={caps} R={r}")
    log(f"  edge cases: {'ALL BIT-IDENTICAL' if not edge_fail else f'{edge_fail} FAILED'}")
    for bs in [int(b) for b in a.bs.split(",")]:
        key = f"ids_nvfp4_{bs}_full"
        if key not in z.files:
            continue
        tot = {}
        bad = {}
        for si, ids_np in enumerate(z[key][:3]):
            ids = torch.from_numpy(ids_np).to(dev)
            t = ids.shape[0]
            wts = torch.rand((t, TOPK), device=dev, generator=g)
            _, dst2src, gam, counts, offsets, *_ = route_metadata(ids.flatten(), wts.flatten(), 0, E, E)
            rows = dst2src.numel()
            acc = (torch.randn((rows, 2 * INTER), device=dev, generator=g) * 6).to(torch.bfloat16)
            acc_amax = torch.rand(rows, device=dev, generator=g) + 0.5
            srows = -(-(rows + 127 * E) // 128) * 128

            def bufs():
                return (torch.full((rows, INTER // 2), 0xAB, dtype=torch.uint8, device=dev),
                        torch.full((srows, INTER // 16), 0xCD, dtype=torch.uint8, device=dev),
                        torch.full((rows,), float("nan"), device=dev))

            runs = {"stock": lambda o: stock_mod.nvfp4_swiglu_quantize(acc, acc_amax, w_amax, w_den, gam, *o, offsets,
                                                                        4.0, 25.0, sms)}
            for r, c in variants:
                runs[f"R{r}/ctas{c}"] = (lambda r_, c_: lambda o: mr(r_).run(acc, acc_amax, w_amax, w_den, gam, *o,
                                                                             offsets, 4.0, 25.0, c_))(r, c)
            ref = None
            for name, f in runs.items():
                o = bufs()
                f(o)
                torch.cuda.synchronize()
                if ref is None:
                    ref = o
                elif not all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(o, ref)):
                    bad[name] = bad.get(name, 0) + 1
                for _ in range(5):
                    f(o)
                st = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters)]
                en = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters)]
                for i in range(a.iters):
                    st[i].record()
                    f(o)
                    en[i].record()
                torch.cuda.synchronize()
                tot[name] = tot.get(name, 0.0) + statistics.median(sorted(s_.elapsed_time(e_) * 1e3
                                                                         for s_, e_ in zip(st, en)))
        line = "  ".join(f"{k} {v / 3:.1f}{'' if k not in bad else ' MISMATCH!'}" for k, v in tot.items())
        log(f"  bs {bs:2d} ({bs * 64} rows): {line}")


def tag(cfg):
    return ",".join(f"{k}={int(v) if isinstance(v, bool) else v}" for k, v in cfg.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="steps.npz")
    ap.add_argument("--kinds", default="nvfp4,mxfp8")
    ap.add_argument("--bs", default="4,8,16,24")
    ap.add_argument("--modes", default="full,cap")
    ap.add_argument("--nsteps", type=int, default=6)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--flush-mb", type=int, default=128)
    ap.add_argument("--real", default="", help="checkpoint dir: real layer weights (0 / 20)")
    ap.add_argument("--configs", default="", help="only these tags, ';'-separated")
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant", action="store_true", help="also bench the routed quantize (stock vs quant_once)")
    ap.add_argument("--swiglu", action="store_true", help="also bench the multi-row swiglu quantize vs stock")
    ap.add_argument("--overlay", default="/work/overlay", help="repo overlay dir (for inkling_prefill)")
    a = ap.parse_args()

    import numpy as np
    import torch
    import flashinfer

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    p = torch.cuda.get_device_properties(dev)
    log(f"device {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count} torch {torch.__version__} "
        f"flashinfer {flashinfer.__version__}")
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped
    import inspect
    src = inspect.getsource(grouped._tiles)
    for t in (1, 8, 32, 64, 128, 129, 192, 1536, 1537, 4000):
        for w in (1152, 2304):
            if grouped._tiles(t, w) != stock_tiles(t, w):
                raise SystemExit(f"stock _tiles changed at {t},{w}: {grouped._tiles(t, w)}\n{src}")
    rules = parse_rules(SHIPPED_RULES)
    z = np.load(a.steps)
    flush = torch.ones(a.flush_mb * (1 << 20) // 4, device=dev)
    sink = torch.empty((), device=dev)
    want = set(a.configs.split(";")) if a.configs else None
    results = []
    if a.quant:
        quant_bench(torch, np, z, a, dev)
    if a.swiglu:
        swiglu_bench(torch, z, a, dev)
    for kind in [k for k in a.kinds.split(",") if k]:
        t0 = time.time()
        k = (Nvfp4 if kind == "nvfp4" else Mxfp8)(torch, dev, a.real, 0 if kind == "nvfp4" else 20, a.seed)
        log(f"\n=== {kind}: weights ready in {time.time() - t0:.0f}s (real={bool(a.real)})")
        for bs in [int(b) for b in a.bs.split(",")]:
            for mode in a.modes.split(","):
                key = f"ids_{kind}_{bs}_{mode}"
                if key not in z.files:
                    continue
                allids = z[key][: a.nsteps]
                for si, ids_np in enumerate(allids):
                    assert ids_np.dtype == np.int32 and ids_np.min() >= 0 and ids_np.max() < E and ids_np.shape[1] == TOPK
                    assert all(len(set(r)) == TOPK for r in ids_np.tolist()), "duplicate expert in a row"
                    ids = torch.from_numpy(ids_np).to(dev)
                    inp = k.inputs(torch, ids, dev, a.seed + 1000 * bs + si)
                    for gemm, (n, kk) in GEMMS.items():
                        if kind == "nvfp4":
                            sw, tn, tkk = shipped_tiles(rules, inp["tokens"], kk)
                            ref_cfg = dict(swap_ab=sw, tile_n=tn, tile_k=tkk)
                            sw, tn, tkk = stock_tiles(inp["tokens"], kk)
                            stock_cfg = dict(swap_ab=sw, tile_n=tn, tile_k=tkk)
                        else:
                            ref_cfg = stock_cfg = dict(tactic="default")
                        ref = torch.empty((inp["rows"], n), dtype=torch.bfloat16, device=dev)
                        k.run(inp, gemm, ref_cfg, ref)
                        again = torch.empty_like(ref)
                        k.run(inp, gemm, ref_cfg, again)
                        torch.cuda.synchronize()
                        det = torch.equal(ref.view(torch.int16), again.view(torch.int16))
                        wbytes = inp["union"] * k.w[gemm][2]
                        for cfg in k.configs():
                            if want and tag(cfg) not in want:
                                continue
                            rec = dict(kind=kind, bs=bs, mode=mode, step=si, gemm=gemm, tokens=inp["tokens"],
                                       rows=inp["rows"], union=inp["union"], cfg=tag(cfg),
                                       is_ref=cfg == ref_cfg, is_stock=cfg == stock_cfg, ref_det=det)
                            out = torch.empty_like(ref)
                            try:
                                c0 = time.time()
                                k.run(inp, gemm, cfg, out)
                                torch.cuda.synchronize()
                                rec["first_s"] = round(time.time() - c0, 2)
                            except Exception as exc:  # noqa: BLE001
                                rec["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                                results.append(rec)
                                continue
                            ne = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
                            rec["mismatch"] = ne
                            rec["finite"] = bool(torch.isfinite(out.float()).all())
                            for _ in range(a.warmup):
                                k.run(inp, gemm, cfg, out)
                            st = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters)]
                            en = [torch.cuda.Event(enable_timing=True) for _ in range(a.iters)]
                            for i in range(a.iters):
                                sink.copy_(flush.sum(dim=0))
                                st[i].record()
                                k.run(inp, gemm, cfg, out)
                                en[i].record()
                            torch.cuda.synchronize()
                            us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(st, en))
                            rec["us"] = statistics.median(us)
                            rec["p10"] = us[len(us) // 10]
                            rec["p90"] = us[(9 * len(us)) // 10]
                            rec["gbps"] = wbytes / (rec["us"] * 1e3)
                            results.append(rec)
                log(f"  {kind} bs {bs} {mode}: {len(allids)} steps done ({time.time() - t0:.0f}s)")
        summarize(results, kind)
        del k
        torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f)
        log(f"wrote {a.json}")


def summarize(results, kind):
    rs = [r for r in results if r["kind"] == kind]
    bad = {}
    for r in rs:
        if "error" in r:
            bad.setdefault(r["cfg"], r["error"])
    for c, e in bad.items():
        log(f"  REJECTED {kind} {c}: {e}")
    keys = sorted({(r["bs"], r["mode"], r["gemm"]) for r in rs})
    for bs, mode, gemm in keys:
        sel = [r for r in rs if (r["bs"], r["mode"], r["gemm"]) == (bs, mode, gemm) and "us" in r]
        if not sel:
            continue
        nst = len({r["step"] for r in sel})
        u = statistics.fmean(r["union"] for r in sel)
        ref_us = sum(r["us"] for r in sel if r["is_ref"])
        log(f"\n{kind} {gemm} bs {bs} ({sel[0]['tokens']} tokens) {mode}: {nst} steps, mean union {u:.1f}, "
            f"ref det {all(r['ref_det'] for r in sel)}")
        log(f"  {'config':<34} {'sum us':>9} {'us/step':>8} {'GB/s':>6} {'vs ref':>7}  exact")
        by = {}
        for r in sel:
            by.setdefault(r["cfg"], []).append(r)
        for c, lst in sorted(by.items(), key=lambda kv: sum(x["us"] for x in kv[1])):
            if len(lst) != nst:
                continue
            tot = sum(x["us"] for x in lst)
            gb = statistics.fmean(x["gbps"] for x in lst)
            mm = sum(x["mismatch"] for x in lst)
            fin = all(x["finite"] for x in lst) and all(x["ref_det"] for x in lst)
            mark = (" *ref" if lst[0]["is_ref"] else "") + (" *stock" if lst[0]["is_stock"] else "")
            ex = "yes" if mm == 0 else f"NO ({mm} el)"
            log(f"  {c:<34} {tot:9.1f} {tot / nst:8.1f} {gb:6.1f} {ref_us / tot if tot else 0:7.3f}  "
                f"{ex}{'' if fin else ' INVALID(nonfinite or ref not repeatable)'}{mark}")


if __name__ == "__main__":
    main()
