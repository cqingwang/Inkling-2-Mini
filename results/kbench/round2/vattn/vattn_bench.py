#!/usr/bin/env python3
"""DSpark target-verify attention on GB10: stock padded FA4 path vs the compact split-KV rel-bias kernel.

Run in /work inside the SGLang image (kbrun.sh):  python3 vattn_bench.py --sections correct,bench,sweep [--quick]

Stock verify (> 32 tokens, SM120): q and rel bias padded to 4 heads per KV group (32 heads),
  flash_attn_with_kvcache(pack_gqa=True, rel_bias=padded) (FA4 shear kernel + FA4 mainloop) -> gate_sink.
Compact verify (<= 32 tokens today): decode_partials (Triton, one CTA per (request, kv head, split), all 8
  verify rows x 3 q heads in one 32-row tile, compact bias gathered by distance) -> combine_gate_sink.

correct  out / LSE of both paths vs an FP32 reference (no TF32) on every row; final gated output too.
bench    cold (L2 flushed by a 256 MiB READ before each rep) median us per visit, GB/s of KV bytes read
         (keys actually attended x 4 KiB), CUDA-graph replays of each path.
sweep    the compact kernel at other (splits, BLOCK_N, num_stages) launch configs (same kernel, same math).
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

T0 = time.time()
BUDGET_S = float(os.environ.get("VATTN_BUDGET_S", 720))

HEADS, KVH, HD, DREL, EXT, PAGE, QLEN = 24, 8, 128, 16, 1024, 128, 8
GROUP, PGROUP = 3, 4
WINDOW_LEFT = 1023  # RadixAttention sliding_window_size = 1024 - 1
SCALE = HD ** -0.5
RESULTS = []


def log(*a):
    print(*a, flush=True)


def load_proj(torch, dev, layer):
    try:
        from safetensors import safe_open

        with safe_open("/model/model-00001-of-00001.safetensors", framework="pt") as f:
            t = f.get_tensor(f"model.llm.layers.{layer}.attn.rel_logits_proj.proj")
        return t.to(dev, torch.bfloat16).contiguous(), "real"
    except Exception as exc:  # noqa: BLE001
        g = torch.Generator(device=dev).manual_seed(layer)
        return torch.randn((DREL, EXT), device=dev, generator=g).to(torch.bfloat16), f"random ({exc!r})"[:80]


class Case:
    """bs requests, each with ctx_b cached tokens INCLUDING its 8 verify tokens (stock writes KV first)."""

    def __init__(self, torch, dev, bs, ctx, proj, seed=0, jitter=True):
        g = torch.Generator(device=dev).manual_seed(seed + bs * 7 + ctx)
        self.bs, self.ctx = bs, ctx
        sks = [max(QLEN + 1, ctx - ((b * 37) % 97 if jitter else 0)) for b in range(bs)]
        pages_per = [-(-s // PAGE) for s in sks]
        total = sum(pages_per) + 2
        perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
        pt = torch.zeros((bs, max(pages_per)), dtype=torch.int32, device=dev)
        c = 0
        for i, n in enumerate(pages_per):
            pt[i, :n] = perm[c:c + n]
            c += n
        self.pt = pt
        self.kc = (torch.randn((total, PAGE, KVH, HD), device=dev, generator=g) * 1.5).to(torch.bfloat16)
        self.vc = torch.randn((total, PAGE, KVH, HD), device=dev, generator=g).to(torch.bfloat16)
        T = bs * QLEN
        self.T = T
        self.q = (torch.randn((T, HEADS, HD), device=dev, generator=g) * 1.5).to(torch.bfloat16)
        r = (torch.randn((T, HEADS, DREL), device=dev, generator=g) * 2.0).to(torch.bfloat16)
        self.rb = torch.einsum("thd,de->the", r, proj).contiguous()  # compact BF16 bias (T, 24, 1024)
        # inkling_vattn bias lever: the prologue run with proj = [I | 0] (16, 128) writes r into [..., :16]
        self.r128 = torch.zeros((T, HEADS, 128), dtype=torch.bfloat16, device=dev)
        self.r128[..., :DREL] = r
        self.proj = proj
        self.gate = torch.randn((T, HEADS * HD), device=dev, generator=g).to(torch.bfloat16)
        self.sinks = torch.randn((T, HEADS), device=dev, generator=g).to(torch.bfloat16)
        self.seqk = torch.tensor(sks, dtype=torch.int32, device=dev)
        self.cu = torch.arange(0, T + 1, QLEN, dtype=torch.int32, device=dev)
        self.sks = sks

    def kv_bytes(self, local):
        keys = sum(min(s, WINDOW_LEFT + QLEN) if local else s for s in self.sks)
        return keys * KVH * HD * 2 * 2


def wave_fill_splits(tiles, sms, max_splits=128, min_fill=0.9):
    best, best_fill = 1, 0.0
    for splits in range(1, max_splits + 1):
        ctas = tiles * splits
        fill = ctas / (-(-ctas // sms) * sms)
        if fill >= min_fill:
            return splits
        if fill > best_fill:
            best, best_fill = splits, fill
    return best


def stock_padded(torch, case, local, sms):
    """Inkling2Attention._forward padded verify branch (attn_padded, SM120) + gate_sink."""
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache
    from sglang.kernels.ops.attention.inkling2_attn import gate_sink, pad_head_groups

    q32 = pad_head_groups(case.q, GROUP, PGROUP)
    rb32 = pad_head_groups(case.rb, GROUP, PGROUP)
    splits = 1
    if not local or case.bs == 1:
        splits = wave_fill_splits(case.bs * KVH * -(-case.T * PGROUP // (128 * case.bs)), sms,
                                  -(-1024 // 128) + 1 if local else 128)

    def attn():
        out = torch.empty((case.T, KVH * PGROUP, HD), dtype=torch.float32, device=case.q.device)
        res = flash_attn_with_kvcache(
            q=q32, k_cache=case.kc, v_cache=case.vc, page_table=case.pt, cache_seqlens=case.seqk,
            cu_seqlens_q=case.cu, cu_seqlens_k_new=None, max_seqlen_q=QLEN, softmax_scale=SCALE, causal=True,
            window_size=(WINDOW_LEFT, 0) if local else (-1, -1), softcap=0.0, return_softmax_lse=True,
            num_splits=splits, pack_gqa=True, out=out, rel_bias=rb32, rel_bias_prep_cache={})
        o, lse = res[0], res[1]
        lse = lse.T  # FA4 varlen LSE is (heads, total_q); the backend passes .T (strided)
        return o, lse

    def full():
        o, lse = attn()
        return gate_sink(o.reshape(case.T, -1), case.gate, lse, case.sinks, output_dtype=torch.bfloat16,
                         head_groups=(GROUP, PGROUP))

    return attn, full, splits


def launch_partials(torch, case, local, splits, block_n, stages, warps=4):
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import _decode_partials

    T, H, D = case.q.shape
    o = torch.empty((splits, T, 1, H, D), dtype=torch.float32, device=case.q.device)
    m = torch.empty((splits, T, 1, H), dtype=torch.float32, device=case.q.device)
    l = torch.empty_like(m)
    _decode_partials[(case.bs * KVH, splits)](
        case.q, case.kc, case.vc, case.pt, case.seqk, case.rb, o, m, l, SCALE, case.bs,
        case.q.stride(0), case.q.stride(1), case.kc.stride(0), case.kc.stride(1), case.kc.stride(2),
        case.pt.stride(0), case.rb.stride(0), case.rb.stride(1), KVH, QLEN, GROUP, 32, D, PAGE, EXT,
        WINDOW_LEFT if local else -1, block_n, num_warps=warps, num_stages=stages, launch_pdl=True)
    return o, m, l


def inkernel(torch, case, local, cfg=None):
    from inkling_vattn import ops
    from sglang.kernels.ops.attention.inkling2_attn import combine_gate_sink

    def parts():
        return ops.inkernel_bias_partials(case.q, case.kc, case.vc, case.pt, case.seqk, case.r128, case.proj, SCALE,
                                          window_size_left=WINDOW_LEFT if local else None, config=cfg)

    def full():
        o, m, l = parts()
        return combine_gate_sink(o, m, l, gate=case.gate, sinks=case.sinks)

    return parts, full


def compact(torch, case, local, cfg=None):
    """cfg None: stock decode_partials heuristics; else (splits, block_n, stages)."""
    from sglang.kernels.ops.attention.inkling2_attn import combine_gate_sink
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    def parts():
        if cfg is None:
            return decode_partials(case.q, case.kc, case.vc, case.pt, case.seqk, case.rb, SCALE,
                                   window_size_left=WINDOW_LEFT if local else None)
        return launch_partials(torch, case, local, *cfg)

    def full():
        o, m, l = parts()
        return combine_gate_sink(o, m, l, gate=case.gate, sinks=case.sinks)

    return parts, full


def stock_cfg(case, local, sms):
    span = WINDOW_LEFT + QLEN if local else case.pt.shape[1] * PAGE
    max_splits = max(1, min(2 * sms // max(1, case.bs * KVH), 64))
    block_n = 32
    return min(-(-span // block_n), max_splits), block_n, 3


def reduce_partials(torch, o, m, l):
    M = m.max(0).values
    Mr = torch.where(torch.isinf(M), torch.zeros_like(M), M)
    w = torch.where(torch.isinf(m), torch.zeros_like(m), torch.exp(m - Mr))
    tot = (w * l).sum(0)
    out = (w[..., None] * o).sum(0) / tot[..., None]
    return out[:, 0], (Mr + torch.log(tot))[:, 0]  # (T, H, D), (T, H)


def reference(torch, case, local):
    """FP32 reference out (T, 24, 128) and lse (T, 24)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    outs, lses = [], []
    for b in range(case.bs):
        sk = case.sks[b]
        j = torch.arange(sk, device=case.q.device)
        pg = case.pt[b, j // PAGE].long()
        K = case.kc[pg, j % PAGE].float()  # (sk, 8, 128)
        V = case.vc[pg, j % PAGE].float()
        pos = sk - QLEN + torch.arange(QLEN, device=K.device)
        d = pos[:, None] - j[None, :]
        vis = d >= 0
        if local:
            vis &= d <= WINDOW_LEFT
        rows = slice(b * QLEN, (b + 1) * QLEN)
        q = case.q[rows].float().view(QLEN, KVH, GROUP, HD)
        s = torch.einsum("tkgd,skd->kgts", q, K) * SCALE  # (8, 3, 8, sk)
        rb = case.rb[rows].float().view(QLEN, KVH, GROUP, EXT).permute(1, 2, 0, 3)  # (8,3,8,1024)
        dc = d.clamp(0, EXT - 1)
        bias = torch.gather(rb, 3, dc.expand(KVH, GROUP, QLEN, sk))
        bias = torch.where(d < EXT, bias, torch.zeros_like(bias))
        s = torch.where(vis, s + bias, torch.full_like(s, float("-inf")))
        mx = s.max(-1, keepdim=True).values
        p = torch.exp(s - mx)
        den = p.sum(-1, keepdim=True)
        o = torch.einsum("kgts,skd->tkgd", p, V) / den.permute(2, 0, 1, 3)
        outs.append(o.reshape(QLEN, HEADS, HD))
        lses.append((mx + torch.log(den)).squeeze(-1).permute(2, 0, 1).reshape(QLEN, HEADS))
    return torch.cat(outs), torch.cat(lses)


def final_ref(torch, case, out, lse):
    g = torch.sigmoid(case.gate.float().view(case.T, HEADS, HD))
    s = torch.sigmoid(lse - case.sinks.float())
    return (out * g * s[..., None]).reshape(case.T, -1)


def unpad(torch, o32):
    return o32.view(o32.shape[0], KVH, PGROUP, -1)[:, :, :GROUP].reshape(o32.shape[0], HEADS, -1)


def section_correct(torch, dev, proj, sms, quick):
    log("== correct: max |err| vs FP32 reference (out = normalized attention output, lse natural log)")
    shapes = [(bs, ctx) for bs in (4, 8, 24, 48) for ctx in (1024, 4096, 32768)]
    shapes += [(1, 32768), (1, 131072)]
    if quick:
        shapes = [(8, 1024), (48, 4096)]
    for bs, ctx in shapes:
        case = Case(torch, dev, bs, ctx, proj)
        for local in (True, False):
            ro, rl = reference(torch, case, local)
            attn, full_st, _ = stock_padded(torch, case, local, sms)
            o32, l32 = attn()
            so, sl = unpad(torch, o32), unpad(torch, l32.unsqueeze(-1)).squeeze(-1)
            res = {"bs": bs, "ctx": ctx, "local": local}
            res["stock_out"] = (so - ro).abs().max().item()
            res["stock_lse"] = (sl - rl).abs().max().item()
            fr = final_ref(torch, case, ro, rl)
            res["stock_final"] = (full_st().float() - fr).abs().max().item()
            sp = min(4, max(1, 96 // (bs * 8)))
            for name, mk, cfg in (("compact", compact, None), ("c64s2", compact, (sp, 64, 2)),
                                  ("ink32", inkernel, (sp, 32, 3)), ("ink64", inkernel, (sp, 64, 2))):
                parts, full_c = mk(torch, case, local, cfg)
                co, cl = reduce_partials(torch, *parts())
                res[name + "_out"] = (co - ro).abs().max().item()
                res[name + "_lse"] = (cl - rl).abs().max().item()
                res[name + "_final"] = (full_c().float() - fr).abs().max().item()
                if name == "compact":
                    res["compact_vs_stock_out"] = (co - so).abs().max().item()
            bad = any(not math.isfinite(v) for k, v in res.items() if isinstance(v, float))
            res["finite"] = not bad
            RESULTS.append(("correct", res))
            log(json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in res.items()}))
        del case
        torch.cuda.empty_cache()


class Timer:
    def __init__(self, torch, dev):
        self.torch = torch
        self.flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev).normal_()  # 256 MiB

    def __call__(self, fn, reps=25, graph=True):
        torch = self.torch
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        run = fn
        if graph:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
                torch.cuda.synchronize()
                with torch.cuda.graph(g, stream=s):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            run = g.replay
        ts = []
        for _ in range(reps):
            self.flush.sum()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            run()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        return statistics.median(ts)


def sweep_cfgs(case, local, quick):
    span = WINDOW_LEFT + QLEN if local else max(case.sks)
    cfgs = []
    for bn in (32, 64):
        for st in (2, 3):
            for sp in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48):
                if sp > -(-span // bn) or case.bs * KVH * sp > 48 * 24:
                    continue
                cfgs.append((sp, bn, st))
    if quick:
        cfgs = cfgs[::3]
    return cfgs


def section_bench(torch, dev, proj, sms, quick, sweep):
    log("== bench: cold median us per visit; GB/s = KV bytes attended / time")
    timer = Timer(torch, dev)
    shapes = []
    for bs in (1, 4, 8, 16, 24, 48):
        shapes += [(bs, 1024, True), (bs, 4096, True), (bs, 1024, False), (bs, 32768, False)]
    shapes += [(1, 131072, False), (4, 131072, False), (2, 32768, False)]
    shapes.sort(key=lambda x: (x[1] >= 32768, x))  # short contexts first (time budget)
    if quick:
        shapes = [(8, 1024, True), (24, 1024, True), (1, 32768, False)]
    cache = {}
    for bs, ctx, local in shapes:
        if time.time() - T0 > BUDGET_S:
            log(f"time budget reached, skipping {bs},{ctx},{local}")
            continue
        key = (bs, ctx)
        if key not in cache:
            cache.clear()
            torch.cuda.empty_cache()
            cache[key] = Case(torch, dev, bs, ctx, proj)
        case = cache[key]
        nbytes = case.kv_bytes(local)
        attn, full_st, st_splits = stock_padded(torch, case, local, sms)
        res = {"bs": bs, "ctx": ctx, "local": local, "kv_MB": round(nbytes / 1e6, 2), "fa4_splits": st_splits}
        res["stock_fa4"] = timer(attn)
        res["stock_total"] = timer(full_st)
        parts, full_c = compact(torch, case, local)
        res["compact_cfg"] = stock_cfg(case, local, sms)
        res["compact_parts"] = timer(parts)
        res["compact_total"] = timer(full_c)
        if sweep:
            best = None
            for cfg in sweep_cfgs(case, local, quick):
                p, _ = compact(torch, case, local, cfg)
                t = timer(p, reps=9, graph=False)
                if best is None or t < best[0]:
                    best = (t, cfg)
            p, f = compact(torch, case, local, best[1])
            res["best_cfg"] = best[1]
            res["best_parts"] = timer(p)
            res["best_total"] = timer(f)
            best = None
            for cfg in sweep_cfgs(case, local, quick):
                p, _ = inkernel(torch, case, local, cfg)
                t = timer(p, reps=9, graph=False)
                if best is None or t < best[0]:
                    best = (t, cfg)
            p, f = inkernel(torch, case, local, best[1])
            res["ink_cfg"] = best[1]
            res["ink_parts"] = timer(p)
            res["ink_total"] = timer(f)
        for k in ("stock_fa4", "stock_total", "compact_parts", "compact_total", "best_parts", "best_total",
                  "ink_parts", "ink_total"):
            if k in res:
                res[k + "_GBs"] = round(nbytes / res[k] / 1e3, 1)
                res[k] = round(res[k], 1)
        RESULTS.append(("bench", res))
        log(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="correct,bench,sweep")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="vattn_results.json")
    args = ap.parse_args()
    import torch
    import triton

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    log(f"torch {torch.__version__} triton {triton.__version__}")
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    proj, src = load_proj(torch, dev, 0)
    log(f"device {torch.cuda.get_device_name()} sms={sms} proj={src}")
    secs = args.sections.split(",")
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "RUN_VISIT")):  # follow-up run (the queued kbrun command is fixed)
        for script in ("gpu_check_patch.py", "compile_check.py", "bench_visit.py"):
            log(f"== {script}")
            subprocess.run([sys.executable, os.path.join(here, script)], timeout=900, check=False)
        return
    if "correct" in secs:  # the patch-level GPU check and the resource check ride along in the same lock window
        here = os.path.dirname(os.path.abspath(__file__))
        for script in ("gpu_check_patch.py", "compile_check.py"):
            log(f"== {script}")
            subprocess.run([sys.executable, os.path.join(here, script)], timeout=300, check=False)
    if "correct" in secs:
        section_correct(torch, dev, proj, sms, args.quick)
    if "bench" in secs:
        section_bench(torch, dev, proj, sms, args.quick, "sweep" in secs)
    with open(args.out, "w") as f:
        json.dump(RESULTS, f, indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
