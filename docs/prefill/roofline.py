#!/usr/bin/env python3
"""Prefill roofline for Inkling-2-Mini (+ DSpark) on one GB10, per prompt token.

Stdlib only. FLOPs and bytes come from config.json and the SGLang serving source
(sglang 0.0.0.dev1+g03fde9cc3.d20261120); see PREFILL.md section 1 for the derivation.

  python3 -I roofline.py                         # assumed rates (placeholders, not measured)
  python3 -I roofline.py --rates bench.json      # rates measured by bench_tc.py --json bench.json
  python3 -I roofline.py --chunk 16384           # what a larger chunk changes

Time model per chunk of C tokens (cold prompt of N tokens, chunks run back to back):
  every kernel class costs max(FLOPs / class rate, bytes / BW); classes are summed (one stream,
  no overlap between kernels); MoE weight bytes are paid once per layer visit per chunk.
"""
import argparse
import json
import math

# ------------------------------------------------------------------ model constants (config.json)
H = 2304                 # hidden
I = 1152                 # expert intermediate
E, TOPK = 128, 8
HEADS, KVH, HD = 24, 8, 128
D_REL, REL_EXT, WINDOW = 16, 1024, 1024
VOCAB = 201024
QKVR_N = 3072 + 1024 + 1024 + 384 + 3072 + 24        # 8600: q, k, v, rel, gate, sinks
QSRC_N = 3072 + 384 + 3072 + 24                       # 6552 columns needed from the q source
WO_K = HEADS * HD                                     # 3072
GLOBAL = {5, 11, 17, 23}
CORE = range(6, 18)
PASSES = 3


def visits():
    """48 layer applications: (stored layer, pass index)."""
    out = [(l, 0) for l in range(6)]
    for p in range(PASSES):
        out += [(l, p) for l in CORE]
    out += [(l, 0) for l in range(18, 24)]
    return out


VISITS = visits()
assert len(VISITS) == 48


def stacked(layer, pass_idx):
    """qkvr runs on 3 stacked sources (3M rows): global layers always, local core layers in passes 2-3."""
    return layer in GLOBAL or (layer in CORE and pass_idx > 0)


N_STACKED = sum(stacked(*v) for v in VISITS)               # 28
N_GLOBAL = sum(v[0] in GLOBAL for v in VISITS)             # 8
N_SWA = 48 - N_GLOBAL                                       # 40
N_NVFP4 = sum(v[0] < 20 for v in VISITS)                    # 44
N_MXFP8 = 48 - N_NVFP4                                      # 4
N_CTRL = sum(v[0] in CORE and v[1] > 0 for v in VISITS)    # 24 visits with loop messages
assert (N_STACKED, N_GLOBAL, N_NVFP4, N_CTRL) == (28, 8, 44, 24)

# ------------------------------------------------------------------ FLOPs per token per visit
F_QKVR = 2 * H * QKVR_N                  # 39.6 MFLOP
F_QKVR_SPLIT_EXTRA = 0                    # column split computes exactly 8600 columns once
F_WO = 2 * WO_K * H                       # 14.2 MFLOP
F_ROUTER = 2 * H * E                      # 0.59 MFLOP (FP32 out)
F_DYNCONV = 2 * (2 * H * KVH * 4)         # global layers: k and v dynamic conv weight_proj
F_RELPROJ = 2 * HEADS * D_REL * REL_EXT   # einsum thd,de->the (bandwidth-bound, 48 KB out)
F_CTRL = 4 * (2 * 2 * H * 64 + 2 * 2 * H * 64)   # 4 controllers: down (2H->64) + gain + forget (64->H)
F_MOE = TOPK * (2 * H * 2 * I + 2 * I * H)       # 127.4 MFLOP: w13 2304x2304 + w2 2304x1152, 8 experts
F_ATTN_PER_KEY = 4 * HEADS * HD           # QK^T + PV per (query, key)
F_DRAFT_CTX = 3 * 2 * H * (2 * 1024)      # DSpark: kv_proj_ctx of 3 draft layers on every prompt token
F_LMHEAD = 2 * H * VOCAB                  # once per request (last position only)

# ------------------------------------------------------------------ weight bytes per visit
NVFP4_EXPERT = H * 2 * I // 2 + H * 2 * I // 16 + H * I // 2 + H * I // 16    # 4.479 MB
MXFP8_EXPERT = H * 2 * I + H * 2 * I // 32 + H * I + H * I // 32              # 8.211 MB
DENSE_VISIT = (QKVR_N * H + H * WO_K + E * H) * 2                              # BF16 qkvr+wo+router

# ------------------------------------------------------------------ activation bytes per token per visit
# Estimated from the eager prefill path (models/inkling2.py, inkling2_attn.py, inkling2_moe.py,
# inkling2_sconv.py, flash_attn/cute/interface.py rel-bias shear). Read + write, bf16 unless noted.
ACT = {
    'attn-res input mix (~3 blocks + partial in, 1 out)': 25e3,
    'attn_norm': 9.2e3,
    'k/v short conv (+state update)': 8.2e3,
    'q/k/rel L2 clip + v_norm': 22e3,
    'rel-bias einsum out [t,24,1024] bf16': 50e3,
    'rel-bias shear copy (48 KB in, 61 KB out)': 110e3,
    'KV cache store': 8.2e3,
    'FA4: q + sheared bias + out fp32 (+K/V window once)': 6.1e3 + 61e3 + 12.3e3 + 37e3,
    'gate_sink (fp32 out in, gate in, bf16 out)': 24.6e3,
    'attn sconv': 9.2e3,
    'attn-res mlp mix': 25e3,
    'mlp_norm': 9.2e3,
    'MoE routed quantize (8 gathered rows in, NVFP4 out)': 37e3 + 9.8e3,
    'MoE gate_up BF16 rows out (GEMM1 epilogue)': 36.9e3,
    'MoE swiglu-quantize (36.9 KB in, 4.9 KB out)': 41.8e3,
    'MoE down rows BF16 out (GEMM2 epilogue)': 36.9e3,
    'MoE routed_partial combine (36.9 KB in, fp32 out) + to(bf16)': 46e3 + 13.8e3,
    'mlp sconv + residual finalize': 27.6e3,
}
ACT_STACKED_EXTRA = 2 * QKVR_N * 2       # the 2 unused thirds of the stacked qkvr output (written by GEMM)
ACT_GLOBAL_EXTRA = 2 * 9.2e3 + 4.6e3     # 3 input norms instead of 1, dynamic conv input
ACT_CTRL_EXTRA = 110e3                   # loop message mix + controllers (passes 2-3, core layers)
ACT_DRAFT = 4 * H * 2 * 2 + 3 * (4 * H * 2 + 2 * 1024 * 2 * 2)   # aux capture + 3 ctx mixes + K/V store

ACT_PER_VISIT = sum(ACT.values())


def attn_keys(n_tokens, chunk):
    """Mean keys per query over a cold prompt: (SWA, global)."""
    swa = sum(min(p + 1, WINDOW) for p in range(0, n_tokens, 64)) * 64 / n_tokens
    glob = (n_tokens + 1) / 2
    return swa, glob


def per_token_flops(n_tokens, column_split=False):
    swa_keys, glob_keys = attn_keys(n_tokens, None)
    qkvr = (48 - N_STACKED) * F_QKVR + N_STACKED * F_QKVR * (1 if column_split else 3)
    return {
        'MoE NVFP4 (44 visits)': N_NVFP4 * F_MOE,
        'MoE MXFP8 (4 visits)': N_MXFP8 * F_MOE,
        'dense BF16 qkvr' + (' (column split)' if column_split else ' (28 stacked visits x3)'): qkvr,
        'dense BF16 wo': 48 * F_WO,
        'dense BF16 router (FP32 out)': 48 * F_ROUTER,
        'dense BF16 small (dyn-conv proj, loop controllers)': N_GLOBAL * F_DYNCONV + N_CTRL * F_CTRL,
        'rel-bias einsum (bandwidth-bound)': 48 * F_RELPROJ,
        'attention SWA (40 visits)': N_SWA * F_ATTN_PER_KEY * swa_keys,
        'attention global (8 visits)': N_GLOBAL * F_ATTN_PER_KEY * glob_keys,
        'DSpark draft ctx K/V (3 layers)': F_DRAFT_CTX,
        'LM head (last token only)': F_LMHEAD / n_tokens,
    }


def act_bytes_per_token(column_split=False):
    b = 48 * ACT_PER_VISIT + N_GLOBAL * ACT_GLOBAL_EXTRA + N_CTRL * ACT_CTRL_EXTRA + ACT_DRAFT
    if not column_split:
        b += N_STACKED * ACT_STACKED_EXTRA
    return b


DEFAULT_RATES = {
    # PLACEHOLDERS until bench_tc.py has run (TFLOP/s and GB/s). Rough GB10 guesses only.
    'bf16_gemm': 90.0,
    'nvfp4_group': 300.0,
    'mxfp8_group': 150.0,
    'attn_swa': 55.0,
    'attn_global': 60.0,
    'bw': 220.0,
}


def rate_for(name, rates):
    if name.startswith('MoE NVFP4'):
        return rates['nvfp4_group']
    if name.startswith('MoE MXFP8'):
        return rates['mxfp8_group']
    if name.startswith('attention SWA'):
        return rates['attn_swa']
    if name.startswith('attention global'):
        return rates['attn_global']
    if name.startswith('rel-bias'):
        return None
    return rates['bf16_gemm']


def chunk_time_per_token(n_tokens, chunk, rates, column_split=False, with_act=True):
    """us per prompt token, and the per-class breakdown."""
    fl = per_token_flops(n_tokens, column_split)
    out = {}
    c = min(chunk, n_tokens)
    for k, f in fl.items():
        r = rate_for(k, rates)
        if r is None:
            out[k] = 0.0  # its bytes are in the activation term
            continue
        t = f / (r * 1e12) * 1e6
        if k.startswith('MoE NVFP4'):
            wt = N_NVFP4 * E * NVFP4_EXPERT / c / (rates['bw'] * 1e9) * 1e6
            t = max(t, wt)
        elif k.startswith('MoE MXFP8'):
            wt = N_MXFP8 * E * MXFP8_EXPERT / c / (rates['bw'] * 1e9) * 1e6
            t = max(t, wt)
        elif k.startswith('dense BF16 qkvr'):
            wt = 48 * QKVR_N * H * 2 / c / (rates['bw'] * 1e9) * 1e6
            t = max(t, wt)
        out[k] = t
    if with_act:
        out['activation traffic (elementwise, norms, quant, shear, combine)'] = (
            act_bytes_per_token(column_split) / (rates['bw'] * 1e9) * 1e6)
    return out


MEASURED = {1024: 0.46, 8192: 2.40, 32768: 10.1, 131072: 54.0}   # cold TTFT s, speedbench longctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', help='bench_tc.py --json output (uses its "rates" block)')
    ap.add_argument('--chunk', type=int, default=8192)
    a = ap.parse_args()
    rates = dict(DEFAULT_RATES)
    src = 'ASSUMED placeholder rates'
    if a.rates:
        with open(a.rates) as f:
            got = json.load(f).get('rates', {})
        rates.update({k: v for k, v in got.items() if v})
        src = f'rates from {a.rates}'
    print(f'# Inkling-2-Mini prefill roofline per prompt token ({src}; chunk {a.chunk})')
    print('rates: ' + ', '.join(f'{k}={v:.0f}' for k, v in rates.items()))
    print(f'visits: {N_SWA} SWA + {N_GLOBAL} global; {N_NVFP4} NVFP4 + {N_MXFP8} MXFP8 MoE; '
          f'{N_STACKED} stacked-qkvr visits; {N_CTRL} loop-message visits')
    print(f'activation bytes/token: {act_bytes_per_token()/1e6:.1f} MB '
          f'(column split: {act_bytes_per_token(True)/1e6:.1f} MB); per visit {ACT_PER_VISIT/1e3:.0f} KB')
    print()
    for n in (1024, 8192, 32768, 131072):
        fl = per_token_flops(n)
        tot = sum(fl.values())
        print(f'## N = {n} ({tot/1e9:.2f} GFLOP/token as served; '
              f'{sum(per_token_flops(n, True).values())/1e9:.2f} with qkvr column split)')
        t = chunk_time_per_token(n, a.chunk, rates)
        ts = chunk_time_per_token(n, a.chunk, rates, column_split=True)
        tg = {k: v for k, v in t.items() if not k.startswith('activation')}
        print(f'  {"class":<62} {"MFLOP/tok":>10} {"us/tok":>8}')
        for k, v in fl.items():
            print(f'  {k:<62} {v/1e6:10.1f} {t.get(k, 0.0):8.2f}')
        ak = [k for k in t if k.startswith('activation')][0]
        print(f'  {ak:<62} {"":>10} {t[ak]:8.2f}')
        tt, tts, tgg = sum(t.values()), sum(ts.values()), sum(tg.values())
        meas = MEASURED.get(n)
        mtok = n / meas if meas else float('nan')
        print(f'  ceiling: compute-only {1e6/tgg:7.0f} tok/s | + activation traffic {1e6/tt:7.0f} tok/s'
              f' | + qkvr column split {1e6/tts:7.0f} tok/s')
        print(f'  measured {mtok:7.0f} tok/s ({1e6/mtok:.0f} us/tok) = {mtok*tt/1e6*100:.0f} % of the '
              f'with-traffic ceiling, {mtok*tgg/1e6*100:.0f} % of compute-only')
        print()


if __name__ == '__main__':
    main()
