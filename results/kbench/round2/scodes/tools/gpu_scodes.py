#!/usr/bin/env python3
"""inkling_scodes GPU check + microbench (runs inside the SGLang image via kbrun.sh, /work = this directory).

For each requested NVFP4 layer (real /model weights):
  1. swizzle the weight scales as SGLang does, encode them (overlay/inkling_scodes/encode.py), and require
     decode(encode(x)) == x for every byte (format check, kernel arithmetic in torch);
  2. real routing (steps.npz: the CPU oracle's top-8 ids of verify steps, bs = requests x 8 rows): inputs built
     with SGLang's own route_metadata + nvfp4_routed_quantize; for every tile the scodes kernel supports, the
     stock FlashInfer group_gemm_nvfp4_nt_groupwise (swap_ab=True, same tile) is the reference and the scodes
     output must be bit-identical (int16 view), every output element written (NaN-pattern poison first);
  3. one CUDA-graph capture + replay of the scodes call must equal the eager output;
  4. timing (--iters > 0): stock vs scodes at the shipped tiles, cold L2 (READ of a 128 MiB buffer before every
     call), CUDA events, median / p10 / p90 over --iters, GB/s of bytes actually read (union x per-expert bytes).
Exit code 1 on any mismatch.
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/work/overlay")

E, TOPK, HIDDEN, INTER = 128, 8, 2304, 1152
GEMMS = {"w13": (2 * INTER, HIDDEN), "w2": (HIDDEN, INTER)}  # name -> (N, K)


def log(*a):
    print(*a, flush=True)


def load_layer(torch, path, layer, device):
    from safetensors import safe_open

    pre = f"model.llm.layers.{layer}.mlp.experts."
    names = [pre + n for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")]
    out = {}
    for f in sorted(os.listdir(path)):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(path, f), framework="pt", device="cpu") as h:
            keys = set(h.keys())
            for n in names:
                if n in keys:
                    out[n] = h.get_tensor(n).to(device)
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"missing {missing}")
    return {g: (out[pre + g + "_weight"].contiguous(), out[pre + g + "_weight_scale"]) for g in GEMMS}


def step_ids(z, bs):
    """[S, bs*8, 8] real verify-step routings; bs 1 = the first request (8 rows) of the bs-4 steps."""
    key = f"ids_nvfp4_{bs}_full"
    if key in z.files:
        return z[key]
    if bs == 1 and "ids_nvfp4_4_full" in z.files:
        return z["ids_nvfp4_4_full"][:, :8, :]
    return None


def shipped_tiles(tokens, width):
    """Tiles serving uses (stock _tiles + shipped INKLING_MOE_TILES + INKLING_MOE2 defaults), swap_ab only."""
    if tokens <= 8:
        return 32, 256
    if tokens <= 128:
        return 32, 256
    if tokens <= 1536:
        return 32, 256
    if tokens <= 3072:
        return 64, 256
    return 128, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--layers", default="0,6,17,19,0x")  # "0x": layer 0 with forced escape words
    ap.add_argument("--steps", default="/work/steps.npz")
    ap.add_argument("--bs", default="1,8,24")
    ap.add_argument("--nsteps", type=int, default=2)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--flush-mb", type=int, default=128)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    import numpy as np
    import torch
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
    from sglang.kernels.ops.moe.inkling2_experts_sm120 import nvfp4_routed_quantize
    from sglang.kernels.ops.moe.inkling2_route import route_metadata
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    from inkling_scodes import encode, kernel

    dev = torch.device("cuda")
    t0 = time.time()
    kernel.module()
    log(f"scodes module {kernel.source_hash()} ready in {time.time() - t0:.0f}s; tiles {kernel.TILES}")
    z = np.load(a.steps)
    flush = torch.ones(a.flush_mb * (1 << 20) // 4, device=dev)
    sink = torch.empty((), device=dev)
    bad = 0
    results = []

    for layer_spec in a.layers.split(","):
        force = layer_spec.endswith("x")
        layer = int(layer_spec.rstrip("x"))
        raw = load_layer(torch, a.model, layer, dev)
        if force:
            # Every 5th row of every expert gets a far outlier in its first and last scale (first and last valid
            # chunk; escape words at K-tile boundaries, incl. w2's half-valid last K256 stage). Both kernels see the
            # same modified scales, so bit identity still has to hold.
            for g in GEMMS:
                wt, sc = raw[g]
                sc = sc.view(torch.uint8).clone()
                sc[:, ::5, 0] = 0x20
                sc[:, ::5, -1] = 0x21
                raw[g] = (wt, sc)
        w = {}
        for g, (n, k) in GEMMS.items():
            wt, sc = raw[g]
            assert tuple(wt.shape) == (E, n, k // 2) and tuple(sc.shape) == (E, n, k // 16), (wt.shape, sc.shape)
            sw = swizzle_blockscale(sc.view(torch.float8_e4m3fn)).flatten(1)       # what SGLang stores
            codes, bases, stride = encode.encode(torch, sw, n, k)
            back = encode.decode(torch, codes, bases, n, k, stock=sw)
            same = torch.equal(back, sw.view(torch.uint8))
            st = encode.stats(torch, sw, n, k)
            log(f"layer {layer_spec} {g}: roundtrip {'OK' if same else 'MISMATCH'}; scale bytes {st['raw']} -> "
                f"{st['coded']} ({st['coded'] / st['raw']:.4f}); range max {st['range_max']}, escape rows "
                f"{st['escape_rows']}")
            if not same:
                bad += 1
            per_expert_stock = n * (k // 2) + n * (k // 16)
            per_expert_codes = n * (k // 2) + stride + n
            w[g] = dict(wt=wt, sw=sw, sw_view=sw.view(torch.uint8).view(E, n, k // 16), codes=codes,
                        bases=bases, stride=stride, n=n, k=k, b_stock=per_expert_stock, b_codes=per_expert_codes)
        del raw

        def make_inputs(ids, seed):
            t = ids.shape[0]
            g_ = torch.Generator(device=dev).manual_seed(seed)
            wts = torch.rand((t, TOPK), device=dev, generator=g_)
            _, dst2src, _, counts, offsets, *_ = route_metadata(ids.flatten(), wts.flatten(), 0, E, E)
            rows = dst2src.numel()
            x = torch.randn((t, HIDDEN), device=dev, generator=g_).to(torch.bfloat16)
            v13, s13, _ = nvfp4_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
            h = torch.randn((rows, INTER), device=dev, generator=g_).to(torch.bfloat16)
            ident = torch.arange(rows, dtype=torch.int32, device=dev)
            v2, s2, _ = nvfp4_routed_quantize(x=h, dst2src=ident, top_k=1, offsets=offsets)
            return dict(tokens=t, rows=rows, union=int((counts > 0).sum()), maxrows=int(counts.max()),
                        offsets=offsets, w13=(v13, s13), w2=(v2, s2))

        def stock(ww, inp, g, tn, tk, out, swap=True):
            v, s_ = inp[g]
            group_gemm_nvfp4_nt_groupwise(v, ww["wt"], s_, ww["sw_view"], inp["offsets"], tile_n=tn, tile_k=tk,
                                          swap_ab=swap, out=out)

        def scodes(ww, inp, g, tn, tk, out):
            v, s_ = inp[g]
            kernel.grouped_gemm(torch, v, s_, ww["wt"], ww["sw"], inp["offsets"], ww["codes"], ww["bases"],
                                ww["stride"], tn, tk, out)

        def poison(rows, n):
            return torch.full((rows, n), -1, dtype=torch.int16, device=dev).view(torch.bfloat16)

        cases = []
        for bs in [int(x) for x in a.bs.split(",")]:
            steps = step_ids(z, bs)
            if steps is None:
                log(f"  no routing for bs {bs} in the steps file")
                continue
            for si, ids_np in enumerate(steps[: max(a.nsteps, 2)]):
                cases.append((f"bs{bs}s{si}", bs, si, torch.from_numpy(ids_np.astype(np.int32)).to(dev)))
        # Synthetic: 160 tokens all on 8 experts with gaps -> 160 rows per expert (> 128: several N tiles).
        sel = torch.tensor([5, 9, 40, 41, 77, 100, 101, 127], dtype=torch.int32, device=dev)
        cases.append(("synth160", 0, 0, sel.repeat(160, 1)))

        built = {}
        for name, bs, si, ids in cases:
            inp = make_inputs(ids, 1000 * bs + si + 7 * layer + (99 if bs == 0 else 0))
            built[name] = inp
            t, rows = inp["tokens"], inp["rows"]
            for g, (n, k) in GEMMS.items():
                ww = w[g]
                for tn, tk in kernel.TILES:
                    ref, got = poison(rows, n), poison(rows, n)
                    stock(ww, inp, g, tn, tk, ref)
                    scodes(ww, inp, g, tn, tk, got)
                    torch.cuda.synchronize()
                    ne = int((ref.view(torch.int16) != got.view(torch.int16)).sum())
                    unwritten = int((got.view(torch.int16) == -1).sum())
                    finite = bool(torch.isfinite(ref.float()).all())
                    nz = int((ref.float() != 0).sum())
                    ok = ne == 0 and finite
                    bad += 0 if ok else 1
                    log(f"  L{layer_spec} {name} {g} tile {tn}x{tk}: rows {rows} union {inp['union']} max/expert "
                        f"{inp['maxrows']} mismatch {ne} unwritten(-1) {unwritten} finite {finite} nonzero "
                        f"{nz}/{ref.numel()}{'' if ok else '  <-- FAIL'}")
                if 9 <= t <= 64 and g == "w13":  # serving without INKLING_MOE2: stock non-swap 64x256
                    ref, got = poison(rows, n), poison(rows, n)
                    stock(ww, inp, g, 64, 256, ref, swap=False)
                    scodes(ww, inp, g, 32, 256, got)
                    torch.cuda.synchronize()
                    ne = int((ref.view(torch.int16) != got.view(torch.int16)).sum())
                    bad += 0 if ne == 0 else 1
                    log(f"  L{layer_spec} {name} {g} stock noswap 64x256 vs scodes 32x256: mismatch {ne}"
                        f"{'' if ne == 0 else '  <-- FAIL'}")

        # CUDA graph: capture on step 0's inputs (static buffers), replay; then write step 1's routing into the
        # same buffers, poison the output, replay again; both must equal stock on the respective inputs.
        for bs in [int(x) for x in a.bs.split(",")]:
            i0, i1 = built.get(f"bs{bs}s0"), built.get(f"bs{bs}s1")
            if i0 is None or i1 is None:
                continue
            for g, (n, k) in GEMMS.items():
                ww = w[g]
                v0, s0_ = i0[g]
                v1, s1_ = i1[g]
                same = v0.shape == v1.shape and s0_.shape == s1_.shape
                static = dict(offsets=i0["offsets"].clone(), **{g: (v0.clone(), s0_.clone())})
                tn, tk = shipped_tiles(i0["tokens"], k)
                gout = poison(i0["rows"], n)
                sg = torch.cuda.Stream()
                sg.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(sg):
                    scodes(ww, static, g, tn, tk, gout)
                torch.cuda.current_stream().wait_stream(sg)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=sg):
                    scodes(ww, static, g, tn, tk, gout)
                for label, inp in (("routing A", i0), ("routing B", i1)):
                    if label == "routing B":
                        if not same:
                            log(f"  L{layer_spec} bs{bs} {g} graph: routing B has other buffer shapes, skipped")
                            break
                        static["offsets"].copy_(i1["offsets"])
                        static[g][0].copy_(v1)
                        static[g][1].copy_(s1_)
                    gout.view(torch.int16).fill_(-1)
                    graph.replay()
                    ref = poison(inp["rows"], n)
                    stock(ww, inp, g, tn, tk, ref)
                    torch.cuda.synchronize()
                    geq = torch.equal(gout.view(torch.int16), ref.view(torch.int16))
                    bad += 0 if geq else 1
                    log(f"  L{layer_spec} bs{bs} {g} graph replay {tn}x{tk} {label}: "
                        f"{'== stock' if geq else 'DIFFERENT  <-- FAIL'}")
                del graph

        if a.iters <= 0:
            continue
        # Timing, cold L2: stock vs scodes at the same swap tile (shipped tile and 32x128), plus the serving
        # reference without INKLING_MOE2 for w13 at 9-64 tokens (stock non-swap 64x256).
        for bs in [int(x) for x in a.bs.split(",")]:
            inp = built.get(f"bs{bs}s0")
            if inp is None:
                continue
            t, rows, union = inp["tokens"], inp["rows"], inp["union"]
            for g, (n, k) in GEMMS.items():
                ww = w[g]
                runs = []
                for tn, tk in sorted({shipped_tiles(t, k), (32, 128)}):
                    runs.append((f"stock_swap_{tn}x{tk}", lambda o, tn=tn, tk=tk: stock(ww, inp, g, tn, tk, o),
                                 ww["b_stock"]))
                    runs.append((f"scodes_{tn}x{tk}", lambda o, tn=tn, tk=tk: scodes(ww, inp, g, tn, tk, o),
                                 ww["b_codes"]))
                if 9 <= t <= 64 and g == "w13":
                    runs.append(("stock_noswap_64x256", lambda o: stock(ww, inp, g, 64, 256, o, swap=False),
                                 ww["b_stock"]))
                rec = dict(layer=layer_spec, bs=bs, tokens=t, rows=rows, union=union, gemm=g)
                for label, fn, per in runs:
                    out = torch.empty((rows, n), dtype=torch.bfloat16, device=dev)
                    for _ in range(3):
                        fn(out)
                    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                          for _ in range(a.iters)]
                    for e0, e1 in ev:
                        sink.copy_(flush.sum())
                        e0.record()
                        fn(out)
                        e1.record()
                    torch.cuda.synchronize()
                    us = sorted(e0.elapsed_time(e1) * 1e3 for e0, e1 in ev)
                    med = statistics.median(us)
                    rec[label] = dict(us=round(med, 2), p10=round(us[len(us) // 10], 2),
                                      p90=round(us[(9 * len(us)) // 10], 2), logical_mb=round(union * per / 1e6, 3),
                                      gbps=round(union * per / (med * 1e3), 1))
                results.append(rec)
                log("  TIME " + json.dumps(rec))
        del w
        torch.cuda.empty_cache()

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    log(f"RESULT {'PASS' if bad == 0 else 'FAIL'} ({bad} failures)")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
