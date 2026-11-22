#!/usr/bin/env python3
"""inkling_scodes integration check in the SGLang image (kbrun.sh): the installed hooks, not the raw kernel.

- installs inkling_kernels (moe_tiles, shipped INKLING_MOE_TILES), inkling_moe2 (tiles) and inkling_scodes like the
  combined loader does, imports the SGLang modules, and checks the wrappers are in place;
- runs the patched Inkling2Experts.swizzle_nvfp4_scales on a stand-in object holding real layer-0 tensors and
  checks the codes were attached;
- calls the patched grouped_sm120.nvfp4_expert_matmul with real routing at 8/64/192 tokens: output must equal
  the stock function's (eager), the self-check counters must move, and a CUDA graph captured after the
  self-checks must replay bit-identical;
- a corrupted code table must trip the self-check (SystemExit 78) on a fresh configuration;
- a reload through the patched _load_shard must re-encode into the same buffers.
"""

import os
import sys
import types

sys.path.insert(0, "/work/overlay")
os.environ.setdefault("INKLING_KERNELS", "moe_tiles")
os.environ.setdefault(
    "INKLING_MOE_TILES",
    "1@2304:swap_ab=1,tile_n=32,tile_k=256;1@1152:swap_ab=1,tile_n=32,tile_k=256;8@2304:swap_ab=1,tile_n=32,"
    "tile_k=256;8@1152:swap_ab=1,tile_n=32,tile_k=256;64@2304:swap_ab=0,tile_n=64,tile_k=256;64@1152:swap_ab=1,"
    "tile_n=32,tile_k=256")
os.environ.setdefault("INKLING_MOE2", "tiles")
os.environ["INKLING_SCODES"] = "1"

E, TOPK, HIDDEN, INTER = 128, 8, 2304, 1152
fails = 0


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    fails += 0 if cond else 1


def main():
    import inkling_kernels
    import inkling_moe2
    import inkling_scodes

    inkling_kernels.install()
    inkling_moe2.install()
    inkling_scodes.install()

    import numpy as np
    import torch
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped
    from sglang.kernels.ops.moe.inkling2_experts_sm120 import nvfp4_routed_quantize
    from sglang.kernels.ops.moe.inkling2_route import route_metadata
    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.models.inkling_common import inkling2_moe

    sys.path.insert(0, "/work/tools")
    from gpu_scodes import load_layer, step_ids

    check(getattr(grouped.nvfp4_expert_matmul, inkling_scodes.MARK, False), "nvfp4_expert_matmul wrapped")
    check(getattr(inkling2_moe.Inkling2Experts.swizzle_nvfp4_scales, inkling_scodes.MARK, False),
          "swizzle_nvfp4_scales wrapped")
    check(hasattr(grouped._tiles, "__wrapped__"), "_tiles wrapped by moe_tiles/moe2")
    stock_mm = grouped.nvfp4_expert_matmul.__wrapped__

    dev = torch.device("cuda")
    raw = load_layer(torch, "/model", 0, dev)
    fake = types.SimpleNamespace(native_sm120=True, _routed_kernels=torch.uint8, _nvfp4_scale_shapes={})
    for g in ("w13", "w2"):
        wt, sc = raw[g]
        p = torch.nn.Parameter(sc.view(torch.float8_e4m3fn).clone(), requires_grad=False)
        fake._nvfp4_scale_shapes[g] = tuple(p.shape)
        setattr(fake, g + "_weight", wt)
        setattr(fake, g + "_scale", p)

    def prepare(self):  # the stock _prepare_sm120_scales effect on the scales
        for g in ("w13", "w2"):
            p = getattr(self, g + "_scale")
            p.data = swizzle_blockscale(p.data).flatten(1)

    orig = inkling2_moe.Inkling2Experts.swizzle_nvfp4_scales.__wrapped__
    # Run the wrapper with the stock body replaced by `prepare` (the stand-in has no full module state).
    wrapper = inkling2_moe.Inkling2Experts.swizzle_nvfp4_scales
    cell_swz = [c for c in wrapper.__closure__ if c.cell_contents is orig][0]
    cell_swz.cell_contents = prepare
    try:
        wrapper(fake)
    finally:
        cell_swz.cell_contents = orig
    for g in ("w13", "w2"):
        check(getattr(getattr(fake, g + "_scale"), inkling_scodes.ATTR, None) is not None, f"{g} codes attached")

    z = np.load("/work/steps.npz")
    moe2_tiles = grouped._tiles
    for phase, tiles_fn in (("moe2", moe2_tiles), ("moe_tiles only (non-swap w13 at 9-64)", moe2_tiles.__wrapped__)):
        grouped._tiles = tiles_fn
        print("phase:", phase, flush=True)
        for bs in (1, 8, 24):
            ids = torch.from_numpy(step_ids(z, bs)[0].astype(np.int32)).to(dev)
            t = ids.shape[0]
            gen = torch.Generator(device=dev).manual_seed(bs)
            wts = torch.rand((t, TOPK), device=dev, generator=gen)
            _, dst2src, _, counts, offsets, *_ = route_metadata(ids.flatten(), wts.flatten(), 0, E, E)
            x = torch.randn((t, HIDDEN), device=dev, generator=gen).to(torch.bfloat16)
            v, s, _ = nvfp4_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
            rows = dst2src.numel()
            h = torch.randn((rows, INTER), device=dev, generator=gen).to(torch.bfloat16)
            v2, s2, _ = nvfp4_routed_quantize(x=h, dst2src=torch.arange(rows, dtype=torch.int32, device=dev),
                                              top_k=1, offsets=offsets)
            for g, (vv, ss) in (("w13", (v, s)), ("w2", (v2, s2))):
                kw = dict(values=vv, scales=ss, weight=getattr(fake, g + "_weight"),
                          weight_scales=getattr(fake, g + "_scale"), offsets=offsets, tokens=t)
                calls0 = inkling_scodes.STATE["calls"]
                for _ in range(inkling_scodes.CHECKS_PER_CONFIG + 1):
                    got = grouped.nvfp4_expert_matmul(**kw)
                ref = stock_mm(**kw)
                torch.cuda.synchronize()
                check(inkling_scodes.STATE["calls"] > calls0, f"[{phase}] bs{bs} {g}: scodes kernel used "
                      f"(tiles {grouped._tiles(tokens=t, width=getattr(fake, g + '_weight').shape[2] * 2)})")
                check(torch.equal(got.view(torch.int16), ref.view(torch.int16)), f"[{phase}] bs{bs} {g}: eager == stock")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    grouped.nvfp4_expert_matmul(**kw)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    gout = grouped.nvfp4_expert_matmul(**kw)
                graph.replay()
                torch.cuda.synchronize()
                check(torch.equal(gout.view(torch.int16), ref.view(torch.int16)), f"[{phase}] bs{bs} {g}: graph replay == stock")
    grouped._tiles = moe2_tiles
    print("self-check counters", inkling_scodes.STATE["checked"], "calls", inkling_scodes.STATE["calls"],
          "stock calls", inkling_scodes.STATE["stock_calls"], flush=True)

    # Corrupted codes on a fresh configuration must exit 78.
    p = fake.w2_scale
    codes, bases, stride, n, k = getattr(p, inkling_scodes.ATTR)
    bad_codes = codes.clone()
    bad_codes[:-640] ^= 0x01
    setattr(p, inkling_scodes.ATTR, (bad_codes, bases, stride, n, k))
    inkling_scodes.STATE["checked"].clear()
    try:
        grouped.nvfp4_expert_matmul(values=v2, scales=s2, weight=fake.w2_weight, weight_scales=p, offsets=offsets,
                                    tokens=t)
        torch.cuda.synchronize()
        check(False, "corrupted codes tripped the self-check")
    except SystemExit as exc:
        check(exc.code == 78, f"corrupted codes tripped the self-check (exit {exc.code})")
    setattr(p, inkling_scodes.ATTR, (codes, bases, stride, n, k))

    # A reload through _load_shard re-encodes in place (same buffers; the stock body is stubbed on the stand-in).
    load = inkling2_moe.Inkling2Experts._load_shard
    cell = [c for c in load.__closure__ if c.cell_contents is load.__wrapped__][0]
    keep = cell.cell_contents
    cell.cell_contents = lambda self, param, *a, **k: None
    codes, bases, stride, n, k = getattr(p, inkling_scodes.ATTR)
    want = codes.clone()
    codes.zero_()
    try:
        load(fake, p, None, None)
    finally:
        cell.cell_contents = keep
    after = getattr(p, inkling_scodes.ATTR)
    check(after[0] is codes and torch.equal(codes, want), "reload re-encoded in place (same buffer)")

    print("RESULT", "PASS" if fails == 0 else f"FAIL ({fails})", flush=True)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
