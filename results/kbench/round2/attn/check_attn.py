#!/usr/bin/env python3
"""GPU check + microbenchmark of inkling_attn (run in /work inside the SGLang image via kbrun.sh).

  python3 check_attn.py --sections correct,graph,patch,bench [--quick]

correct  ours vs stock FA4+bias (einsum bias, FA4 shear) vs an FP32 reference (no TF32) on sampled rows:
         chunk 8192 at prefix 0 / 8K / 32K and a mixed multi-request batch with shuffled pages, SWA and global.
         Reports max abs error of out and LSE vs the reference for ours and for stock.
graph    CUDA-graph capture of ours, replayed with other cu_seqlens / seqused_k values: equal to eager.
patch    the inkling_attn varlen wrapper inside use_visit(): equal to the direct call; fallback path equal to stock.
bench    cold (L2 flushed by a 128 MiB read before each rep) median us and useful TFLOP/s, t = 2K / 8K,
         prefix 0 / 8K / 32K, SWA and global: stock (einsum + FA4 incl. shear), bias_layout (Triton sheared bias +
         FA4 presheared), ours.
Inputs: q, k ~ N(0, 1.5^2) (scores of a few units, like L2-clipped q/k), v ~ N(0, 1), r ~ N(0, 2^2) (rel clip at
norm 8), proj = the real layer-0 (SWA) / layer-5 (global) rel_logits_proj from /model when present, else N(0,1).
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HEADS, KVH, HD, DREL, EXT, WINDOW, PAGE = 24, 8, 128, 16, 1024, 1024, 128
SCALE = HD ** -0.5
RESULTS = []


def log(*a):
    print(*a, flush=True)


def load_proj(torch, dev, layer):
    path = "/model/model-00001-of-00001.safetensors"
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            t = f.get_tensor(f"model.llm.layers.{layer}.attn.rel_logits_proj.proj")
        log(f"proj: real layer {layer} {tuple(t.shape)} absmax {t.float().abs().max():.3f}")
        return t.to(dev, torch.bfloat16).contiguous()
    except Exception as exc:  # noqa: BLE001
        log(f"proj: random ({exc!r})")
        g = torch.Generator(device=dev).manual_seed(layer)
        return torch.randn((DREL, EXT), device=dev, generator=g).to(torch.bfloat16)


class Case:
    """A paged KV cache + one batch of requests (sq_b new tokens after prefix_b cached tokens)."""

    def __init__(self, torch, dev, reqs, seed=0, shuffle=True, slots=None):
        self.torch = torch
        self.reqs = reqs  # [(sq, prefix)]
        g = torch.Generator(device=dev).manual_seed(seed)
        n = len(reqs) if slots is None else slots
        sqs = [sq for sq, _ in reqs] + [0] * (n - len(reqs))
        sks = [sq + p for sq, p in reqs] + [0] * (n - len(reqs))
        self.T = sum(sqs)
        pages_per = [-(-sk // PAGE) for sk in sks]
        total_pages = sum(pages_per) + 4
        perm = torch.randperm(total_pages, generator=g, device=dev) if shuffle else torch.arange(total_pages, device=dev)
        maxp = max(max(pages_per), 1)
        pt = torch.zeros((n, maxp), dtype=torch.int32, device=dev)
        c = 0
        for i, npg in enumerate(pages_per):
            pt[i, :npg] = perm[c:c + npg].to(torch.int32)
            c += npg
        self.page_table = pt
        self.kc = (torch.randn((total_pages, PAGE, KVH, HD), device=dev, generator=g) * 1.5).to(torch.bfloat16)
        self.vc = torch.randn((total_pages, PAGE, KVH, HD), device=dev, generator=g).to(torch.bfloat16)
        self.q = (torch.randn((max(self.T, 1), HEADS, HD), device=dev, generator=g) * 1.5).to(torch.bfloat16)
        self.r = (torch.randn((max(self.T, 1), HEADS, DREL), device=dev, generator=g) * 2.0).to(torch.bfloat16)
        cu = [0]
        for sq in sqs:
            cu.append(cu[-1] + sq)
        self.cu = torch.tensor(cu, dtype=torch.int32, device=dev)
        self.seqk = torch.tensor(sks, dtype=torch.int32, device=dev)
        self.sqs, self.sks, self.cu_host = sqs, sks, cu

    def keys(self, b):
        """[sk, KVH, HD] K and V of request b gathered from the pages."""
        torch = self.torch
        sk = self.sks[b]
        j = torch.arange(sk, device=self.kc.device)
        pg = self.page_table[b, j // PAGE].long()
        return self.kc[pg, j % PAGE], self.vc[pg, j % PAGE]


def useful_flops(case, local):
    keys = 0
    for sq, sk in zip(case.sqs, case.sks):
        off = sk - sq
        for i in range(sq):
            p = i + off
            keys += min(p + 1, WINDOW) if local else p + 1
    return 4.0 * HEADS * HD * keys


def run_ours(case, proj, local, out=None, packed=None, warps=None):
    from inkling_attn import ops

    return ops.prefill_attention(case.q, case.kc, case.vc, case.page_table, case.cu, case.seqk, case.r, proj, SCALE,
                                 WINDOW - 1 if local else None, out=out, packed=packed, warps=warps)


def stock_bias(torch, case, proj):
    return torch.einsum("thd,de->the", case.r, proj)


def run_stock(torch, case, proj, local, out, rb=None):
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache

    if rb is None:
        rb = stock_bias(torch, case, proj)
    return flash_attn_with_kvcache(
        q=case.q, k_cache=case.kc, v_cache=case.vc, page_table=case.page_table, cache_seqlens=case.seqk,
        cu_seqlens_q=case.cu, cu_seqlens_k_new=None, max_seqlen_q=max(case.sqs), softmax_scale=SCALE, causal=True,
        window_size=(WINDOW - 1, 0) if local else (-1, -1), softcap=0.0, return_softmax_lse=True, num_splits=1,
        pack_gqa=False, out=out, rel_bias=rb, rel_bias_prep_cache={})


def run_bias_layout(torch, case, proj, local, out):
    import importlib

    from inkling_prefill import bias_layout
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache

    bias_layout.patch_sm120(importlib.import_module(bias_layout.SM120_MODULE))  # idempotent

    sheared = bias_layout.sheared_rel_bias(case.r, proj, case.cu, case.seqk, local, WINDOW - 1)
    with bias_layout.use_presheared(sheared):
        return flash_attn_with_kvcache(
            q=case.q, k_cache=case.kc, v_cache=case.vc, page_table=case.page_table, cache_seqlens=case.seqk,
            cu_seqlens_q=case.cu, cu_seqlens_k_new=None, max_seqlen_q=max(case.sqs), softmax_scale=SCALE,
            causal=True, window_size=(WINDOW - 1, 0) if local else (-1, -1), softcap=0.0, return_softmax_lse=True,
            num_splits=1, pack_gqa=False, out=out, rel_bias=sheared, rel_bias_prep_cache={})


def sample_rows(torch, sq, n_rand, g):
    rows = set(range(min(sq, 80))) | set(range(max(0, sq - 80), sq))
    for m in range(64, sq, 64 * 17):  # some query-block boundaries
        rows |= {m - 1, m, m + 15, m + 16}
    if sq > 0:
        rows |= set(torch.randint(0, sq, (n_rand,), generator=g).tolist())
    return sorted(r for r in rows if 0 <= r < sq)


def reference(torch, case, proj, local, b, rows):
    """FP32 (no TF32) reference out [n, H, HD] and lse [H, n] for request b's rows (request-relative)."""
    sq, sk = case.sqs[b], case.sks[b]
    off = sk - sq
    K, V = case.keys(b)
    K, V = K.float(), V.float()
    projf = proj.float()
    rows_t = torch.tensor(rows, device=K.device)
    t = case.cu_host[b] + rows_t
    pos = rows_t + off
    j = torch.arange(sk, device=K.device)
    d = pos[:, None] - j[None, :]
    vis = d >= 0
    if local:
        vis &= d <= WINDOW - 1
    outs, lses = [], []
    for h in range(HEADS):
        kvh = h // (HEADS // KVH)
        qh = case.q[t, h].float()
        s = (qh @ K[:, kvh].T) * SCALE
        rb = (case.r[t, h].float() @ projf).to(torch.bfloat16).float()  # [n, EXT], the stock BF16 bias
        dc = d.clamp(0, EXT - 1)
        bias = torch.gather(rb, 1, dc)
        bias = torch.where(d < EXT, bias, torch.zeros_like(bias))
        s = torch.where(vis, s + bias, torch.full_like(s, float("-inf")))
        m = s.max(dim=1, keepdim=True).values
        pexp = torch.exp(s - m)
        l = pexp.sum(dim=1, keepdim=True)
        outs.append((pexp @ V[:, kvh]) / l)
        lses.append((m + torch.log(l)).squeeze(1))
    return torch.stack(outs, 1), torch.stack(lses, 0)


def compare(torch, case, proj, local, name, g, n_rand=256):
    o_ours = torch.full((case.q.shape[0], HEADS, HD), float("nan"), device=case.q.device)
    o_ours, l_ours = run_ours(case, proj, local, out=o_ours)
    n_cov = case.cu_host[-1]
    packed_same = True
    for kw in (dict(packed=False), dict(warps=8)):  # same per-row math -> bit-identical
        o_up, l_up = run_ours(case, proj, local, **kw)
        packed_same &= bool(torch.equal(o_up[:n_cov], o_ours[:n_cov])) and bool(torch.equal(l_up[:, :n_cov], l_ours[:, :n_cov]))
        del o_up, l_up
    o_st = torch.full_like(o_ours, float("nan"))
    res = run_stock(torch, case, proj, local, o_st)
    o_st, l_st = res[0], res[1]
    torch.cuda.synchronize()
    err = dict(ours_out=0.0, stock_out=0.0, ours_lse=0.0, stock_lse=0.0, ours_vs_stock_out=0.0,
               ours_vs_stock_lse=0.0, rows=0)
    nan_ours = nan_stock = 0
    for b, sq in enumerate(case.sqs):
        if sq == 0:
            continue
        rows = sample_rows(torch, sq, n_rand, g)
        ro, rl = reference(torch, case, proj, local, b, rows)
        t = case.cu_host[b] + torch.tensor(rows, device=ro.device)
        for tag, o, l in (("ours", o_ours, l_ours), ("stock", o_st, l_st)):
            oo, ll = o[t].float(), l[:, t].float()
            if tag == "ours":
                nan_ours += int(torch.isnan(oo).sum()) + int(torch.isnan(ll).sum())
            else:
                nan_stock += int(torch.isnan(oo).sum()) + int(torch.isnan(ll).sum())
            err[f"{tag}_out"] = max(err[f"{tag}_out"], float((oo - ro).abs().nan_to_num(1e9).max()))
            err[f"{tag}_lse"] = max(err[f"{tag}_lse"], float((ll - rl).abs().nan_to_num(1e9).max()))
        err["ours_vs_stock_out"] = max(err["ours_vs_stock_out"], float((o_ours[t] - o_st[t]).abs().max()))
        err["ours_vs_stock_lse"] = max(err["ours_vs_stock_lse"], float((l_ours[:, t] - l_st[:, t]).abs().max()))
        err["rows"] += len(rows)
    # every covered row of ours is written (no NaN sentinel left)
    covered = case.cu_host[-1]
    unwritten = int(torch.isnan(o_ours[:covered]).any(dim=(1, 2)).sum())
    # PASS: no worse than stock within 2 % (both errors are dominated by the BF16 rounding of P)
    ok = (packed_same and nan_ours == 0 and unwritten == 0 and err["ours_out"] <= max(err["stock_out"], 1e-6) * 1.02
          and err["ours_lse"] <= max(err["stock_lse"], 1e-6) * 1.02)
    verdict = "PASS" if ok else ("CLOSE" if err["ours_out"] <= 2 * err["stock_out"] and nan_ours == 0 else "FAIL")
    log(f"  {name:<34} {'swa' if local else 'global':<6} rows {err['rows']:5d}  out err ours {err['ours_out']:.3e} "
        f"stock {err['stock_out']:.3e} | lse ours {err['ours_lse']:.3e} stock {err['stock_lse']:.3e} | "
        f"ours-stock out {err['ours_vs_stock_out']:.3e} lse {err['ours_vs_stock_lse']:.3e} | nan {nan_ours}/"
        f"{nan_stock} unwritten {unwritten} variants bit-equal {packed_same}  {verdict}")
    RESULTS.append(dict(section="correct", case=name, kind="swa" if local else "global", verdict=verdict, **err,
                        nan_ours=nan_ours, nan_stock=nan_stock, unwritten=unwritten))
    return verdict


def sec_correct(torch, dev, a):
    g = torch.Generator().manual_seed(1)
    projs = {True: load_proj(torch, dev, 0), False: load_proj(torch, dev, 5)}
    cases = [
        ("chunk8192 prefix0", [(8192, 0)]),
        ("chunk8192 prefix8192", [(8192, 8192)]),
        ("chunk8192 prefix32768", [(8192, 32768)]),
        ("chunk2048 prefix0", [(2048, 0)]),
        ("mixed 5 req + empty slots", [(1000, 3000), (0, 0), (4500, 0), (77, 1500), (613, 129)]),
        ("tiny rows", [(1, 0), (3, 5000), (65, 63), (17, 1)]),
    ]
    if a.quick:
        cases = cases[:1] + cases[4:]
    verdicts = []
    for name, reqs in cases:
        slots = len(reqs) + (3 if "empty" in name else 0)
        case = Case(torch, dev, reqs, seed=len(name), slots=slots)
        for local in (True, False):
            verdicts.append(compare(torch, case, projs[local], local, name, g))
        del case
        torch.cuda.empty_cache()
    return verdicts


def sec_graph(torch, dev, a):
    """Capture ours once (static q / r / out / metadata buffers), replay with other request splits."""
    proj = load_proj(torch, dev, 0)
    ok = True
    for local in (True, False):
        # slots sized for the largest seqused_k of every split below (page_table rows must cover them)
        big = Case(torch, dev, [(3000, 7000), (2000, 9000), (1000, 9000)], seed=3, slots=4)
        out = torch.zeros((big.q.shape[0], HEADS, HD), device=dev)
        lse_holder = {}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run_ours(big, proj, local, out=out)  # warm-up (build / first launch) outside capture
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, lse_holder["lse"] = run_ours(big, proj, local, out=out)
        for split in ([(3000, 2000), (2000, 100), (1000, 7000)], [(5999, 1), (1, 9000), (0, 0)],
                      [(6000, 0)], [(2500, 300), (0, 0), (900, 4000)]):
            cu = [0]
            for sq, _ in split:
                cu.append(cu[-1] + sq)
            cu += [cu[-1]] * (5 - len(cu))
            sk = [sq + p for sq, p in split] + [0] * (4 - len(split))
            big.cu.copy_(torch.tensor(cu, dtype=torch.int32, device=dev))
            big.seqk.copy_(torch.tensor(sk, dtype=torch.int32, device=dev))
            out.fill_(float("nan"))
            lse_holder["lse"].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            o_graph, l_graph = out.clone(), lse_holder["lse"].clone()
            tail_ok = bool(torch.isnan(o_graph[cu[-1]:]).all()) and bool(torch.isnan(l_graph[:, cu[-1]:]).all())
            head_ok = not bool(torch.isnan(o_graph[:cu[-1]]).any()) and not bool(torch.isnan(l_graph[:, :cu[-1]]).any())
            eager_case = big
            eager_case.sqs = [sq for sq, _ in split] + [0] * (4 - len(split))
            eager_case.sks = sk
            eager_case.cu_host = cu
            o_e, l_e = run_ours(eager_case, proj, local)
            torch.cuda.synchronize()
            n = cu[-1]
            same = bool(torch.equal(o_graph[:n], o_e[:n])) and bool(torch.equal(l_graph[:, :n], l_e[:, :n]))
            log(f"  graph {'swa' if local else 'global':<6} split {split}: replay == eager {same}, rows past "
                f"cu[-1] untouched {tail_ok}, covered rows written {head_ok}")
            ok &= same and tail_ok and head_ok
        del graph, big
        torch.cuda.empty_cache()
    RESULTS.append(dict(section="graph", ok=ok))
    return ok


def sec_patch(torch, dev, a):
    """The varlen wrapper route (as SGLang's backend reaches it) and the fallback route."""
    import importlib

    from inkling_attn import patch

    sm120 = importlib.import_module(patch.SM120_MODULE)
    patch.patch_sm120(sm120)
    proj = load_proj(torch, dev, 0)
    ok = True
    for local in (True, False):
        case = Case(torch, dev, [(1500, 700), (900, 0)], seed=5, slots=3)
        n = case.cu_host[-1]
        o_direct, l_direct = run_ours(case, proj, local)
        o_p = torch.empty_like(o_direct)
        with patch.use_visit(case.r, proj, WINDOW - 1 if local else None):
            res = run_stock(torch, case, proj, local, o_p, rb=case.r[:n])
        same = bool(torch.equal(res[0][:n], o_direct[:n])) and bool(torch.equal(res[1][:, :n], l_direct[:, :n]))
        # fallback: a call outside the kernel's coverage (softcap set) must equal stock with the einsum bias
        kfb = patch.STATS["fallback"]
        o_s = torch.empty_like(o_direct)
        stock = run_stock(torch, case, proj, local, o_s)
        log(f"  patch {'swa' if local else 'global':<6} wrapper == direct {same}  kernel calls {patch.STATS['kernel']}")
        try:
            with patch.use_visit(case.r, proj, WINDOW - 1 if local else None):
                patch._CTX.get().local = not local  # force a window mismatch -> fallback
                patch._CTX.get().window_left = (WINDOW - 1) if not local else None
                o_f = torch.empty_like(o_direct)
                res_f = run_stock(torch, case, proj, local, o_f, rb=case.r[:n])
            fb_same = bool(torch.equal(res_f[0][:n], stock[0][:n])) and bool(torch.equal(res_f[1][:, :n], stock[1][:, :n]))
            log(f"  patch fallback == stock {fb_same} (fallbacks {patch.STATS['fallback'] - kfb})")
            ok &= fb_same
        except Exception as exc:  # noqa: BLE001
            log(f"  patch fallback raised {exc!r}")
            ok = False
        try:
            run_stock(torch, case, proj, local, o_p, rb=case.r[:n])
            log("  patch: marker outside visit did NOT raise")
            ok = False
        except RuntimeError as exc:
            log(f"  patch: marker outside visit raised as intended ({str(exc)[:60]}...)")
        ok &= same
        del case
    ok &= rel_dispatch_check(torch, dev, proj)
    RESULTS.append(dict(section="patch", ok=ok))
    return ok


def rel_dispatch_check(torch, dev, proj):
    """Real RelLogitsProj with bias_layout's forward wrapper and this package's __call__ dispatch, in both
    install orders: inside a visit it returns r; outside, the stock einsum."""
    import types

    from inkling_attn import patch
    from inkling_prefill import bias_layout
    from sglang.srt.models.inkling_common import attn as cattn

    ok = True
    for order in ("bias_layout first", "inkling_attn first"):
        cls = cattn.RelLogitsProj
        saved = (cls.forward, cls.__dict__.get("__call__"))
        try:
            if order == "bias_layout first":
                bias_layout.patch_common_attn(cattn)
                patch.patch_common_attn(cattn)
            else:
                patch.patch_common_attn(cattn)
                bias_layout.patch_common_attn(cattn)
            m = cls(16, 1024).to(device=dev, dtype=torch.bfloat16)
            m.proj.data.copy_(proj)
            r = (torch.randn((300, 24, 16), device=dev) * 2).to(torch.bfloat16)
            owner = types.SimpleNamespace(rel_logits_proj=m, num_tp_heads=24)
            visit = patch._Visit(owner, True, 1023, m.proj)
            tok = patch._CTX.set(visit)
            try:
                got = m(r)
            finally:
                patch._CTX.reset(tok)
            inside = got is r and visit.r is r
            outside = bool(torch.equal(m(r), torch.einsum("thd,de->the", r, m.proj)))
            log(f"  RelLogitsProj dispatch ({order}): visit -> r {inside}, no visit -> einsum {outside}")
            ok &= inside and outside
        finally:
            cls.forward = saved[0]
            if saved[1] is None:
                if "__call__" in cls.__dict__:
                    del cls.__call__
            else:
                cls.__call__ = saved[1]
    return ok


def bench(torch, fn, flush, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush[0].copy_(flush[1].sum())  # 128 MiB read: evicts L2 (24 MiB)
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    us = sorted(x.elapsed_time(y) * 1e3 for x, y in zip(st, en))
    return statistics.median(us), us[0], us[int(0.9 * (len(us) - 1))]


def sec_bench(torch, dev, a):
    projs = {True: load_proj(torch, dev, 0), False: load_proj(torch, dev, 5)}
    flush = (torch.empty((), device=dev), torch.ones(32 << 20, device=dev))
    shapes = [(2048, 0), (2048, 8192), (2048, 32768), (8192, 0), (8192, 8192), (8192, 32768), (8192, 122880)]
    if a.quick:
        shapes = [(8192, 0), (8192, 32768)]
    variants = a.variants.split(",")
    for c, pfx in shapes:
        case = Case(torch, dev, [(c, pfx)], seed=11, shuffle=False)
        out = torch.empty((c, HEADS, HD), device=dev)
        for local in (True, False):
            proj = projs[local]
            fl = useful_flops(case, local)
            row = {}
            fns = {
                "stock": lambda: run_stock(torch, case, proj, local, out),
                "stock_fa4_only": None,
                "bias_layout": lambda: run_bias_layout(torch, case, proj, local, out),
                "ours": lambda: run_ours(case, proj, local, out=out),
                "ours_unpacked": lambda: run_ours(case, proj, local, out=out, packed=False),
                "ours_w8": lambda: run_ours(case, proj, local, out=out, warps=8),
            }
            rb = stock_bias(torch, case, proj)
            fns["stock_fa4_only"] = lambda: run_stock(torch, case, proj, local, out, rb=rb)
            fns["einsum_only"] = lambda: stock_bias(torch, case, proj)
            for v in variants:
                try:
                    med, mn, p90 = bench(torch, fns[v], flush, a.iters)
                    row[v] = med
                    tf = fl / (med * 1e-6) / 1e12 if v not in ("einsum_only",) else 0.0
                    RESULTS.append(dict(section="bench", chunk=c, prefix=pfx, kind="swa" if local else "global",
                                        variant=v, med_us=med, min_us=mn, p90_us=p90, tflops=tf))
                except Exception as exc:  # noqa: BLE001
                    log(f"  bench {v} chunk {c} prefix {pfx}: {type(exc).__name__}: {str(exc)[:200]}")
            del rb
            parts = "  ".join(f"{v} {row[v]:9.1f} us ({fl / (row[v] * 1e-6) / 1e12:5.1f} TF)" for v in row
                              if v != "einsum_only")
            sp = ""
            if "ours" in row and "stock" in row:
                sp += f"  ours/stock x{row['stock'] / row['ours']:.2f}"
            if "ours" in row and "bias_layout" in row:
                sp += f"  ours/bias_layout x{row['bias_layout'] / row['ours']:.2f}"
            if "einsum_only" in row:
                parts += f"  einsum {row['einsum_only']:.1f} us"
            log(f"  {'swa' if local else 'global':<6} chunk {c:<5} prefix {pfx:<6} {parts}{sp}")
        del case, out
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sections", default="correct,graph,patch,bench")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--variants", default="stock,bias_layout,ours,ours_unpacked,ours_w8,einsum_only")
    p.add_argument("--json", default="/work/results.json")
    a = p.parse_args()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    log(f"device {torch.cuda.get_device_name()} cap {torch.cuda.get_device_capability()} torch {torch.__version__}")
    t0 = time.time()
    from inkling_attn import build

    build.load(verbose=False)
    log(f"extension {'reused' if build.INFO['cached'] else 'built'} in {build.INFO['seconds']:.1f} s")
    failed = []
    for sec in a.sections.split(","):
        log(f"== {sec}")
        ts = time.time()
        res = globals()[f"sec_{sec}"](torch, dev, a)
        if res is False or (isinstance(res, list) and "FAIL" in res):
            failed.append(sec)
        log(f"== {sec} done in {time.time() - ts:.0f} s")
    with open(a.json, "w") as f:
        json.dump(RESULTS, f, indent=1)
    log(f"total {time.time() - t0:.0f} s; failed sections: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
