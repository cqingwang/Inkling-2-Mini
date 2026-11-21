#!/usr/bin/env python3
"""GPU check of the inkling_moe2 overlay as installed (inside the SGLang image, no server):

1. install() from the env (INKLING_KERNELS=moe_tiles + INKLING_MOE_TILES as shipped, INKLING_MOE2=tiles,quant),
   then import SGLang's grouped-GEMM and routed-quant modules: the patches must be applied through the hooks.
2. For each token count of the bench (32/64/128/192 tokens x top-8, real routing from steps.npz): the patched
   path (_tiles table -> nvfp4_expert_matmul, mxfp8_expert_matmul, nvfp4_routed_swiglu_quantize) against the
   unpatched functions (__wrapped__ chains) on identical inputs, eagerly AND replayed from a CUDA graph captured
   with the patched functions; outputs must be bit-identical.

  python3 gpu_check_moe2.py --steps steps.npz --overlay /work/overlay
"""

import argparse
import os
import sys

E, TOPK, HIDDEN, INTER = 128, 8, 2304, 1152


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="steps.npz")
    ap.add_argument("--overlay", default="/work/overlay")
    ap.add_argument("--bs", default="4,8,16,24")
    a = ap.parse_args()
    sys.path.insert(0, a.overlay)
    os.environ.setdefault("INKLING_MOE2", "tiles")
    os.environ.setdefault("INKLING_MOE2_VERBOSE", "1")
    import inkling_kernels
    import inkling_moe2

    if os.environ.get("INKLING_KERNELS"):
        inkling_kernels.install()
    inkling_moe2.install()

    import numpy as np
    import torch
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped
    from sglang.kernels.ops.moe import inkling2_experts_sm120 as ex
    from sglang.kernels.ops.moe.inkling2_route import route_metadata
    from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import mxfp8_weight_scales

    print("applied:", inkling_moe2.STATE["applied"], "kernels:", inkling_kernels.STATE["applied"], flush=True)
    dev = torch.device("cuda", 0)
    g = torch.Generator(device=dev).manual_seed(5)
    z = np.load(a.steps)
    fails = []

    def swz(sc):
        b, m, k = sc.shape
        return sc.reshape(b, m // 128, 4, 32, k // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(b, m, k)

    # NVFP4 weights (random; bytes and layout as SGLang's after _prepare_sm120_scales)
    w13 = torch.randint(0, 256, (E, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device=dev, generator=g)
    s13 = swz((torch.rand((E, 2 * INTER, HIDDEN // 16), device=dev, generator=g) + 0.25).to(torch.float8_e4m3fn)
              .view(torch.uint8)).flatten(1).view(torch.float8_e4m3fn)
    w2 = torch.randint(0, 256, (E, HIDDEN, INTER // 2), dtype=torch.uint8, device=dev, generator=g)
    s2 = swz((torch.rand((E, HIDDEN, INTER // 16), device=dev, generator=g) + 0.25).to(torch.float8_e4m3fn)
             .view(torch.uint8)).flatten(1).view(torch.float8_e4m3fn)
    m13 = (torch.randn((E, 2 * INTER, HIDDEN), device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    m13s = mxfp8_weight_scales(torch.randint(122, 130, (E, 2 * INTER, HIDDEN // 32), dtype=torch.uint8, device=dev,
                                             generator=g))
    m2 = (torch.randn((E, HIDDEN, INTER), device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    m2s = mxfp8_weight_scales(torch.randint(122, 130, (E, HIDDEN, INTER // 32), dtype=torch.uint8, device=dev,
                                            generator=g))
    w_amax = torch.rand(E, device=dev, generator=g) + 0.5
    w_den = torch.tensor([2.5], device=dev)

    def unwrap(fn):
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        return fn

    stock_tiles = unwrap(grouped._tiles)
    stock_swiglu = unwrap(ex.nvfp4_routed_swiglu_quantize)
    stock_mx = unwrap(grouped.mxfp8_expert_matmul)

    for bs in [int(b) for b in a.bs.split(",")]:
        ids = torch.from_numpy(z[f"ids_nvfp4_{bs}_full"][0]).to(dev)
        t = ids.shape[0]
        wts = torch.rand((t, TOPK), device=dev, generator=g)
        _, dst2src, gam, counts, offsets, *_ = route_metadata(ids.flatten(), wts.flatten(), 0, E, E)
        x = torch.randn((t, HIDDEN), device=dev, generator=g).to(torch.bfloat16)

        def moe_nvfp4(tiles_fn, swiglu_fn):
            saved = grouped._tiles
            grouped._tiles = tiles_fn
            try:
                v, s, am = ex.nvfp4_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
                acc = grouped.nvfp4_expert_matmul(values=v, scales=s, weight=w13, weight_scales=s13, offsets=offsets,
                                                  tokens=t)
                v2, s2_, am2 = swiglu_fn(acc=acc, acc_amax=am, w_amax=w_amax, w_denominator=w_den, gammas=gam,
                                         offsets=offsets, gate_cap=4.0, up_cap=25.0)
                down = grouped.nvfp4_expert_matmul(values=v2, scales=s2_, weight=w2, weight_scales=s2,
                                                   offsets=offsets, tokens=t)
                return [acc, v2, am2, down]
            finally:
                grouped._tiles = saved

        h = torch.randn((dst2src.numel(), INTER), device=dev, generator=g).to(torch.bfloat16)
        ident = torch.arange(dst2src.numel(), dtype=torch.int32, device=dev)

        def moe_mx(fn):
            v, s = ex.mxfp8_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
            v2, s2 = ex.mxfp8_routed_quantize(x=h, dst2src=ident, top_k=1, offsets=offsets)
            return [fn(values=v, scales=s, weight=m13, weight_scales=m13s, offsets=offsets),
                    fn(values=v2, scales=s2, weight=m2, weight_scales=m2s, offsets=offsets)]

        patched_tiles, patched_swiglu, patched_mx = grouped._tiles, ex.nvfp4_routed_swiglu_quantize, \
            grouped.mxfp8_expert_matmul
        ref = moe_nvfp4(stock_tiles, stock_swiglu) + moe_mx(stock_mx)
        eager = moe_nvfp4(patched_tiles, patched_swiglu) + moe_mx(patched_mx)  # first eager call: self-checks
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            moe_nvfp4(patched_tiles, patched_swiglu)
            moe_mx(patched_mx)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_g = moe_nvfp4(patched_tiles, patched_swiglu) + moe_mx(patched_mx)
        graph.replay()
        torch.cuda.synchronize()
        names = ["w13", "swiglu values", "swiglu amax", "w2", "mxfp8 w13", "mxfp8 w2"]
        for name, r_, e_, g_ in zip(names, ref, eager, out_g):
            ok_e = torch.equal(r_.view(torch.uint8), e_.view(torch.uint8))
            ok_g = torch.equal(r_.view(torch.uint8), g_.view(torch.uint8))
            if not (ok_e and ok_g):
                fails.append(f"bs {bs} {name}: eager {ok_e} graph {ok_g}")
        print(f"bs {bs} ({t} tokens): tiles w13 {patched_tiles(t, HIDDEN)} w2 {patched_tiles(t, INTER)} "
              f"(stock {stock_tiles(t, HIDDEN)} / {stock_tiles(t, INTER)}); "
              f"{'all bit-identical (eager + graph)' if not any(f.startswith(f'bs {bs} ') for f in fails) else 'FAIL'}",
              flush=True)
    print("FAILURES: " + "; ".join(fails) if fails else "ALL CHECKS PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
