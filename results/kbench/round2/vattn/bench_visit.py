#!/usr/bin/env python3
"""Whole verify-attention visit, cold, CUDA-graph replays: prologue -> attention -> gate/sink epilogue.

stock   (> 32 rows today) prologue padded to 4 heads/group (rel [t, 32, 1024]) -> FA4 SM120 (shear + mainloop,
        pack_gqa, the model's wave-fill splits) -> gate_sink
compact (<= 32 rows today) prologue (rel [t, 24, 1024]) -> stock decode_partials heuristics -> combine_gate_sink
cap     inkling_vattn cap: compact prologue -> stock kernel with ops.launch_config -> combine_gate_sink
bias    inkling_vattn cap+bias: prologue with [I|0] (rel [t, 24, 128]) -> in-kernel bias kernel -> combine
Also the prologue alone for stock / compact / bias. Reports median us per visit and the KV GB/s.
"""
import json
import os
import statistics
import sys

os.environ.setdefault("SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from gpu_check_patch import HEADS, KVH, QLEN, Inputs, load_proj  # noqa: E402

PG = 4


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


class Timer:
    def __init__(self, dev):
        self.flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev).normal_()

    def __call__(self, fn, reps=25):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        ts = []
        for _ in range(reps):
            self.flush.sum()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        return statistics.median(ts)


def main():
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    import inkling_vattn

    inkling_vattn.install(levers=("cap", "bias"))
    import sglang.kernels.ops.attention.inkling2_attn as ops_mod
    import sglang.srt.models.inkling_common.inkling2_attn as model_mod
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache
    from sglang.kernels.ops.attention.inkling2_attn_prologue import attention_prologue as stock_prologue
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    proj = load_proj(dev)
    scale = 128 ** -0.5
    timer = Timer(dev)
    shapes = []
    for bs in (1, 4, 8, 16, 24, 48):
        for ctx in (700, 1500):
            shapes += [(bs, ctx, True), (bs, ctx, False)]
    shapes += [(1, 32768, False), (1, 131072, False), (4, 32768, False), (4, 131072, False), (8, 32768, False)]
    out = []
    for bs, ctx, local in shapes:
        x = Inputs(dev, bs, ctx, proj, seed=bs + ctx)
        wl = 1023 if local else None
        win = x.windows()
        cu32 = torch.arange(0, x.T + 1, QLEN, dtype=torch.int32, device=dev)
        splits = 1
        if not local or bs == 1:
            splits = wave_fill_splits(bs * KVH * -(-x.T * PG // (128 * bs)), sms, 9 if local else 128)

        def pro(fn, pg):
            return fn(x.q_in, x.k_in, x.v_in, x.r_in, HEADS, KVH, 16, pg, x.k_w, x.v_w, x.k_cache, x.v_cache,
                      x.meta, x.q_scale, x.r_scale, x.proj, x.seq_lens, True, 1.0, 8.0, 1e-6, 1e-6,
                      k_dynamic=None, v_dynamic=None, positions=None, tau_floor=None, tau_alpha=None,
                      verify_windows=win, shear=None)

        def stock():
            q, k, v, rel = pro(stock_prologue, PG)
            o = torch.empty((x.T, KVH * PG, 128), dtype=torch.float32, device=dev)
            res = flash_attn_with_kvcache(
                q=q, k_cache=x.kc, v_cache=x.vc, page_table=x.pt, cache_seqlens=x.seq_lens, cu_seqlens_q=cu32,
                cu_seqlens_k_new=None, max_seqlen_q=QLEN, softmax_scale=scale, causal=True,
                window_size=(1023, 0) if local else (-1, -1), softcap=0.0, return_softmax_lse=True,
                num_splits=splits, pack_gqa=True, out=o, rel_bias=rel, rel_bias_prep_cache={})
            return ops_mod.gate_sink(res[0].reshape(x.T, -1), x.gate, res[1].T, x.sinks,
                                     output_dtype=torch.bfloat16, head_groups=(3, PG))

        def compact():
            q, k, v, rel = pro(stock_prologue, 3)
            return ops_mod.combine_gate_sink(*decode_partials(q, x.kc, x.vc, x.pt, x.seq_lens, rel, scale,
                                                              window_size_left=wl), gate=x.gate, sinks=x.sinks)

        def cap():
            q, k, v, rel = pro(stock_prologue, 3)
            return ops_mod.combine_gate_sink(*ops_mod.rel_bias_partials(q, x.kc, x.vc, x.pt, x.seq_lens, rel, scale,
                                                                        None, window_size_left=wl, verify=True),
                                             gate=x.gate, sinks=x.sinks)

        def bias():
            q, k, v, rel = pro(model_mod.attention_prologue, 3)
            return ops_mod.combine_gate_sink(*ops_mod.rel_bias_partials(q, x.kc, x.vc, x.pt, x.seq_lens, rel, scale,
                                                                        None, window_size_left=wl, verify=True),
                                             gate=x.gate, sinks=x.sinks)

        keys = sum(min(s, 1031) if local else s for s in x.seq_lens.tolist())
        nbytes = keys * 4096
        r = {"bs": bs, "ctx": ctx, "local": local, "kv_MB": round(nbytes / 1e6, 2)}
        for name, fn in (("stock", stock), ("compact", compact), ("cap", cap), ("bias", bias),
                         ("pro_stock", lambda: pro(stock_prologue, PG)), ("pro_compact", lambda: pro(stock_prologue, 3)),
                         ("pro_bias", lambda: pro(model_mod.attention_prologue, 3))):
            if name == "pro_bias":
                from inkling_vattn import patch
            t = timer(fn)
            if name == "pro_bias":
                patch._PENDING.clear()  # the prologue-only timing leaves its record unconsumed
            r[name] = round(t, 1)
            if not name.startswith("pro"):
                r[name + "_GBs"] = round(nbytes / t / 1e3, 1)
        ref = stock().float()
        r["max_diff_bias_vs_stock"] = (bias().float() - ref).abs().max().item()
        r["max_diff_cap_vs_stock"] = (cap().float() - ref).abs().max().item()
        out.append(r)
        print(json.dumps(r), flush=True)
        del x
        torch.cuda.empty_cache()
    with open("visit.json", "w") as f:
        json.dump(out, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
