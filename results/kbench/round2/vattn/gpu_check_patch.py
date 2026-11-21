#!/usr/bin/env python3
"""GPU check of the inkling_vattn patches on the real sglang modules (run in /work inside the image).

1. install(cap,bias); importing the model attention module applies the cap (384) and wraps attention_prologue;
   importing the kernels module wraps rel_bias_partials.
2. Real stock attention_prologue (compact verify, real layer-0 proj, synthetic projections / conv state) vs the
   wrapped one: q, k, v and the conv windows bit-identical; the wrapped rel holds r in [..., :16] and
   BF16(dot(r, proj)) equals the stock rel bias bit for bit.
3. Attention: stock decode_partials + combine_gate_sink on the stock bias vs the wrapped rel_bias_partials
   (in-kernel bias) + combine_gate_sink, and the cap path (stock kernel, tuned launch) — max |diff|.
4. CUDA graph: the wrapped prologue + partials + combine captured at bs 8, replayed with new inputs / seqlens
   equals eager.
"""
import os
import sys

os.environ.setdefault("SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

HEADS, KVH, HD, DREL, EXT, QLEN, W = 24, 8, 128, 16, 1024, 8, 4
FAIL = []


def log(*a):
    print(*a, flush=True)


def check(name, ok, detail=""):
    log(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAIL.append(name)


def load_proj(dev):
    try:
        from safetensors import safe_open

        with safe_open("/model/model-00001-of-00001.safetensors", framework="pt") as f:
            return f.get_tensor("model.llm.layers.0.attn.rel_logits_proj.proj").to(dev, torch.bfloat16).contiguous()
    except Exception as exc:  # noqa: BLE001
        log(f"random proj ({exc!r})"[:120])
        return torch.randn((DREL, EXT), device=dev).to(torch.bfloat16)


class Inputs:
    def __init__(self, dev, bs, ctx, proj, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        T = bs * QLEN
        self.bs, self.T = bs, T
        width = 8600
        X = (torch.randn((T, width), device=dev, generator=g) * 0.5).to(torch.bfloat16)
        self.X = X
        o = 0
        self.q_in = X[:, o:o + HEADS * HD]; o += HEADS * HD
        self.k_in = X[:, o:o + KVH * HD]; o += KVH * HD
        self.v_in = X[:, o:o + KVH * HD]; o += KVH * HD
        self.r_in = X[:, o:o + HEADS * DREL]
        kvw = KVH * HD
        self.k_w = (torch.randn((kvw, W), device=dev, generator=g) * 0.3).to(torch.bfloat16)
        self.v_w = (torch.randn((kvw, W), device=dev, generator=g) * 0.3).to(torch.bfloat16)
        slots = bs + 2
        self.k_cache = (torch.randn((slots, W - 1, kvw), device=dev, generator=g) * 0.5).to(torch.bfloat16)
        self.v_cache = (torch.randn((slots, W - 1, kvw), device=dev, generator=g) * 0.5).to(torch.bfloat16)
        self.meta = {
            "safe_idx": torch.arange(1, bs + 1, device=dev, dtype=torch.int64),
            "cache_mask": torch.ones(bs, device=dev, dtype=torch.bool),
            "cu": torch.arange(0, T + 1, QLEN, device=dev, dtype=torch.int64),
            "si": torch.arange(bs, device=dev, dtype=torch.int32).repeat_interleave(QLEN),
        }
        self.q_scale = (1 + 0.1 * torch.randn((HEADS, HD), device=dev, generator=g)).to(torch.bfloat16)
        self.r_scale = (1 + 0.1 * torch.randn((HEADS, DREL), device=dev, generator=g)).to(torch.bfloat16)
        self.proj = proj
        sks = [max(QLEN + 1, ctx - (b * 53) % 211) for b in range(bs)]
        self.seq_lens = torch.tensor(sks, device=dev, dtype=torch.int32)
        self.windows = lambda: (
            torch.zeros((bs, QLEN, W - 1, kvw), device=dev, dtype=torch.bfloat16),
            torch.zeros((bs, QLEN, W - 1, kvw), device=dev, dtype=torch.bfloat16),
        )
        pages = [-(-s // 128) for s in sks]
        total = sum(pages) + 1
        perm = torch.randperm(total, device=dev, generator=g).to(torch.int32)
        pt = torch.zeros((bs, max(pages)), device=dev, dtype=torch.int32)
        c = 0
        for i, n in enumerate(pages):
            pt[i, :n] = perm[c:c + n]
            c += n
        self.pt = pt
        self.kc = (torch.randn((total, 128, KVH, HD), device=dev, generator=g) * 0.1).to(torch.bfloat16)
        self.vc = torch.randn((total, 128, KVH, HD), device=dev, generator=g).to(torch.bfloat16)
        self.gate = torch.randn((T, HEADS * HD), device=dev, generator=g).to(torch.bfloat16)
        self.sinks = torch.randn((T, HEADS), device=dev, generator=g).to(torch.bfloat16)

    def prologue(self, fn, windows):
        return fn(self.q_in, self.k_in, self.v_in, self.r_in, HEADS, KVH, DREL, HEADS // KVH,
                  self.k_w, self.v_w, self.k_cache, self.v_cache, self.meta, self.q_scale, self.r_scale,
                  self.proj, self.seq_lens, True, 1.0, 8.0, 1e-6, 1e-6,
                  k_dynamic=None, v_dynamic=None, positions=None, tau_floor=None, tau_alpha=None,
                  verify_windows=windows, shear=None)


def main():
    dev = torch.device("cuda")
    import inkling_vattn

    levers = inkling_vattn.install(levers=("cap", "bias"))
    check("install", levers == ("cap", "bias"), str(levers))
    import sglang.kernels.ops.attention.inkling2_attn as ops_mod
    import sglang.srt.models.inkling_common.inkling2_attn as model_mod
    from inkling_vattn import ops, patch
    from sglang.kernels.ops.attention.inkling2_attn_prologue import attention_prologue as stock_prologue
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    check("cap", model_mod.VERIFY_REL_BIAS_MAX_TOKENS == 384, str(model_mod.VERIFY_REL_BIAS_MAX_TOKENS))
    check("prologue wrapped", getattr(model_mod.attention_prologue, "_inkling_vattn", False))
    check("partials wrapped", getattr(ops_mod.rel_bias_partials, "_inkling_vattn", False))
    proj = load_proj(dev)
    scale = HD ** -0.5
    for bs, ctx in ((1, 300), (4, 1500), (8, 1024), (24, 700), (48, 2000), (2, 40000)):
        for local in (True, False):
            wl = 1023 if local else None
            x = Inputs(dev, bs, ctx, proj, seed=bs * 13 + ctx)
            ws, wn = x.windows(), x.windows()
            q0, k0, v0, rel0 = x.prologue(stock_prologue, ws)
            q1, k1, v1, rel1 = x.prologue(model_mod.attention_prologue, wn)
            same = all(torch.equal(a, b) for a, b in ((q0, q1), (k0, k1), (v0, v1), (ws[0], wn[0]), (ws[1], wn[1])))
            check(f"prologue q/k/v/windows bs{bs}", same)
            expect_r = x.T > 32  # bias lever only above the stock cap
            if not expect_r:
                check(f"passthrough <=32 rows bs{bs}", torch.equal(rel1, rel0) and not patch._PENDING)
            else:
                check(f"rel shape bs{bs}", tuple(rel1.shape) == (x.T, HEADS, 128), str(tuple(rel1.shape)))
                rb = torch.einsum("thd,de->the", rel1[..., :DREL], proj)  # the prologue's dot matches this GEMM
                nz = rel1[..., DREL:].abs().max().item()
                check(f"rel = BF16(r.proj) bitwise bs{bs}", torch.equal(rb, rel0),
                      f"maxdiff {(rb.float() - rel0.float()).abs().max().item():.3g} pad {nz}")
                check(f"pending bs{bs}", rel1.data_ptr() in patch._PENDING)
            kc, vc = x.kc, x.vc
            for b in range(bs):  # write the verify K/V into the cache at the last QLEN positions like the backend
                s = int(x.seq_lens[b])
                for i in range(QLEN):
                    p = s - QLEN + i
                    kc[x.pt[b, p // 128], p % 128] = k0[b * QLEN + i]
                    vc[x.pt[b, p // 128], p % 128] = v0[b * QLEN + i]
            st = ops_mod.combine_gate_sink(*decode_partials(q0, kc, vc, x.pt, x.seq_lens, rel0, scale, window_size_left=wl),
                                           gate=x.gate, sinks=x.sinks)
            pw = ops_mod.rel_bias_partials(q1, kc, vc, x.pt, x.seq_lens, rel1, scale, None, window_size_left=wl, verify=True)
            ours = ops_mod.combine_gate_sink(*pw, gate=x.gate, sinks=x.sinks)
            check(f"pending consumed bs{bs}", not patch._PENDING)
            d1 = (ours.float() - st.float()).abs().max().item()
            fin = bool(torch.isfinite(ours.float()).all())
            log(f"  bitwise ours==stock: {torch.equal(ours, st)}")
            check(f"attn bs{bs} ctx{ctx} {'swa' if local else 'global'}", fin and d1 < 2e-2,
                  f"ours-vs-stock {d1:.3g} (BF16 outputs, |out|max {st.float().abs().max().item():.2f})")
            if expect_r:
                try:
                    ops_mod.rel_bias_partials(q1, kc, vc, x.pt, x.seq_lens, rel1, scale, None, window_size_left=wl,
                                              verify=True)
                    check("orphan marker raises", False)
                except RuntimeError:
                    check("orphan marker raises", True)
            del x
    # interleaved layers: prologue A, prologue B, attention A, attention B
    xa, xb = Inputs(dev, 8, 900, proj, seed=21), Inputs(dev, 8, 1300, proj, seed=22)
    qa, _, _, ra = xa.prologue(model_mod.attention_prologue, xa.windows())
    qb, _, _, rb_ = xb.prologue(model_mod.attention_prologue, xb.windows())
    oa = ops_mod.combine_gate_sink(*ops_mod.rel_bias_partials(qa, xa.kc, xa.vc, xa.pt, xa.seq_lens, ra, scale, None,
                                                              window_size_left=1023, verify=True), gate=xa.gate, sinks=xa.sinks)
    ob = ops_mod.combine_gate_sink(*ops_mod.rel_bias_partials(qb, xb.kc, xb.vc, xb.pt, xb.seq_lens, rb_, scale, None,
                                                              window_size_left=None, verify=True), gate=xb.gate, sinks=xb.sinks)
    qa0, _, _, ra0 = xa.prologue(stock_prologue, xa.windows())
    qb0, _, _, rb0 = xb.prologue(stock_prologue, xb.windows())
    sa = ops_mod.combine_gate_sink(*decode_partials(qa0, xa.kc, xa.vc, xa.pt, xa.seq_lens, ra0, scale, window_size_left=1023),
                                   gate=xa.gate, sinks=xa.sinks)
    sb = ops_mod.combine_gate_sink(*decode_partials(qb0, xb.kc, xb.vc, xb.pt, xb.seq_lens, rb0, scale), gate=xb.gate,
                                   sinks=xb.sinks)
    da, db = (oa.float() - sa.float()).abs().max().item(), (ob.float() - sb.float()).abs().max().item()
    check("interleaved A,B,A,B", da < 2e-2 and db < 2e-2 and not patch._PENDING, f"{da:.3g} {db:.3g}")
    del xa, xb
    # CUDA graph: capture at bs 8, replay with new inputs and seqlens
    x = Inputs(dev, 8, 1200, proj, seed=5)
    wnd = x.windows()

    def step():
        q, k, v, rel = x.prologue(model_mod.attention_prologue, wnd)
        o, m, l = ops_mod.rel_bias_partials(q, x.kc, x.vc, x.pt, x.seq_lens, rel, scale, None,
                                            window_size_left=1023, verify=True)
        return ops_mod.combine_gate_sink(o, m, l, gate=x.gate, sinks=x.sinks)

    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            gout = step()
    torch.cuda.current_stream().wait_stream(s)
    for it in range(3):
        x.X.copy_((torch.randn_like(x.X.float()) * 0.5).to(torch.bfloat16))
        x.seq_lens.copy_(torch.randint(QLEN + 1, 1200, (8,), device=dev, dtype=torch.int32))
        x.seq_lens[0] = 1  # a CUDA-graph padding request (seqlen < QLEN)
        g.replay()
        ref = step()
        torch.cuda.synchronize()
        check(f"graph replay {it}", torch.equal(gout, ref), f"maxdiff {(gout.float() - ref.float()).abs().max().item():.3g}")
    # first verify visit inside a capture (no eager warm-up): must fall back to the stock bias, still correct
    patch._EYE.clear()
    x2 = Inputs(dev, 6, 900, proj, seed=9)
    w2 = x2.windows()

    def step2():
        q, k, v, rel = x2.prologue(model_mod.attention_prologue, w2)
        o, m, l = ops_mod.rel_bias_partials(q, x2.kc, x2.vc, x2.pt, x2.seq_lens, rel, scale, None,
                                            window_size_left=None, verify=True)
        return ops_mod.combine_gate_sink(o, m, l, gate=x2.gate, sinks=x2.sinks)

    g2 = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g2, stream=s):
            gout2 = step2()
    torch.cuda.current_stream().wait_stream(s)
    g2.replay()
    q0, k0, v0, rel0 = x2.prologue(stock_prologue, x2.windows())
    ref2 = ops_mod.combine_gate_sink(*decode_partials(q0, x2.kc, x2.vc, x2.pt, x2.seq_lens, rel0, scale),
                                     gate=x2.gate, sinks=x2.sinks)
    torch.cuda.synchronize()
    check("capture-first fallback", patch.STATS["eye_in_capture"] >= 1 and
          (gout2.float() - ref2.float()).abs().max().item() < 2e-2,
          f"eye_in_capture={patch.STATS['eye_in_capture']} maxdiff {(gout2.float() - ref2.float()).abs().max().item():.3g}")
    log(f"STATS {patch.STATS}")
    log("RESULT", "FAIL " + ",".join(FAIL) if FAIL else "ALL PASS")


if __name__ == "__main__":
    main()
