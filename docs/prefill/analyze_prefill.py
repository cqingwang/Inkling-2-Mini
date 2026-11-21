#!/usr/bin/env python3
"""Prefill analysis of a SGLang torch-profiler trace of Inkling-2-Mini (+DSpark) on one GB10.

  python3 -I analyze_prefill.py TRACE_OR_DIR --tokens 32768 [--chunk 8192] [--json out.json]
                                [--rates bench.json] [--names]

Stdlib only. Forwards are cut at the embedding gather (index_select / embedding kernel) that starts
every model forward; a forward is a PREFILL forward if it holds >= 40 FA4 SM120 forward kernels
(the per-visit prefill attention) and >= 40 MoE grouped GEMMs. Inside a prefill forward, layer
visits end at `_routed_partial` (48 per forward; the visit's own mlp-sconv / residual kernels are
attributed to the next visit, which does not change any class total). Dense GEMMs in a visit are
named by position: before the attention kernel [qkvr, (dyn-conv k, dyn-conv v on global layers),
rel-bias einsum], after it [wo, router]. Kernels after the 48th combine are the forward tail (final
norm, aux capture, LM head, DSpark context K/V inject).

Busy = union of kernel intervals; per-class 'excl' = exclusive timeline occupancy (sums to busy).
TFLOP/s uses the exact per-chunk FLOPs (prefix = chunk index x chunk); GB/s uses the activation-byte
model of roofline.py (an estimate) for the traffic classes.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import statistics
import sys

H, I, E, TOPK = 2304, 1152, 128, 8
HEADS, HD, WINDOW = 24, 128, 1024
QKVR_N = 8600
GLOBAL = {5, 11, 17, 23}


def stored_layer(app):
    if app < 6:
        return app, 1
    if app < 42:
        return 6 + (app - 6) % 12, (app - 6) // 12 + 1
    return app - 24, 1


def region(app):
    if app < 6:
        return 'prelude L0-5'
    if app < 42:
        return f'core pass {(app - 6) // 12 + 1} (L6-17)'
    return 'coda L18-23'


def stacked(app):
    layer, p = stored_layer(app)
    return layer in GLOBAL or (6 <= layer < 18 and p > 1)


GEMM_RE = re.compile(r'nvjet|wmma|gemvx|Kernel2<|cutlass_80|sm90_xmma|gemm_bf16|cublas|'
                     r'gemmSN|gemmk1|splitKreduce|b12xgemm|_gemv_kernel', re.I)


def tag(n):
    if 'GemmUniversal' in n and ('GroupProblemShape' in n or 'BlockScaled' in n):
        return 'moe_gemm_nvfp4'
    if 'cute_sm12x_gemm' in n or 'SM120BlockScaledMoe' in n or ('sm120_blockscaled' in n and 'Moe' in n):
        return 'moe_gemm_mxfp8'
    if 'group_gemm_args' in n:
        return 'moe_args'
    if 'swiglu_quantize_routed' in n:
        return 'moe_swiglu_quant'
    if 'quantize_routed' in n:
        return 'moe_in_quant'
    if '_routed_partial' in n:
        return 'moe_combine'
    if '_route_topk' in n:
        return 'moe_route'
    if ('RadixSort' in n or 'radixSort' in n or 'sort' in n.lower() and 'cub' in n.lower()
            or '_compute_src2dst' in n or 'expert_offs' in n or 'num_tokens_per_expert' in n
            or 'expert_block' in n or '_route_metadata' in n):
        return 'moe_route'
    if 'FlashAttentionForward' in n or 'flash_fwd' in n:
        return 'fa4'
    if 'hear' in n or 'CuSeqlensToBlocks' in n:          # shear bias + its block-map prep
        return 'relbias_shear'
    if 'gate_sink' in n:
        return 'gate_sink'
    if 'store_kvcache' in n or 'set_kv_buffer' in n:
        return 'kv_store'
    if 'l2_clip' in n or 'L2Clip' in n or 'l2clip' in n.lower():
        return 'qk_clip'
    if 'sconv' in n:
        return 'sconv'
    if 'AttnRes' in n or 'attn_res' in n:
        return 'attn_res'
    if '_controller_' in n or 'message_controllers' in n:
        return 'loop_ctrl'
    if 'mix_rmsnorm' in n or 'rope' in n.lower():
        return 'draft_inject'
    if 'RMSNorm' in n or 'rms_norm' in n or 'rmsnorm' in n.lower():
        return 'norm'
    if 'bfloat16_copy_kernel' in n:
        return 'cast'
    if 'indexSelect' in n or 'embedding' in n.lower():
        return 'embed'
    if GEMM_RE.search(n):
        return 'gemm'
    if 'softcap' in n or 'argmax' in n or 'sampl' in n.lower():
        return 'logits'
    return 'other'


CLASS_OF = {
    'moe_gemm_nvfp4': 'MoE grouped GEMM NVFP4', 'moe_gemm_mxfp8': 'MoE grouped GEMM MXFP8',
    'moe_args': 'MoE route/args/sort', 'moe_route': 'MoE route/args/sort',
    'moe_in_quant': 'MoE routed quantize', 'moe_swiglu_quant': 'MoE swiglu-quantize',
    'moe_combine': 'MoE combine (+cast)', 'cast': 'MoE combine (+cast)',
    'fa4': 'attention FA4 kernel', 'relbias_shear': 'rel-bias shear (+prep)',
    'gate_sink': 'gate_sink', 'kv_store': 'KV store', 'qk_clip': 'q/k/rel L2 clip',
    'sconv': 'short convs', 'attn_res': 'attn-res mixes', 'loop_ctrl': 'loop controllers',
    'norm': 'RMSNorms', 'draft_inject': 'DSpark ctx inject (norm/rope)', 'embed': 'embedding',
    'logits': 'logits/sampling', 'other': 'other elementwise/glue',
}


def load(path):
    if os.path.isdir(path):
        c = sorted(glob.glob(os.path.join(path, '*.trace.json*')))
        if not c:
            sys.exit(f'no trace in {path}')
        path = c[0]
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        d = json.load(f)
    return path, (d['traceEvents'] if isinstance(d, dict) else d)


def union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ((ce - cs) if cs is not None else 0.0)


def exclusive(items):
    """[(cls, ts, dur)] -> {cls: exclusive us}."""
    out = collections.Counter()
    run = -1e30
    for cls, s, d in sorted(items, key=lambda x: x[1]):
        e = s + d
        out[cls] += max(0.0, e - max(s, run))
        run = max(run, e)
    return out


def split_forwards(ks):
    """Cut the kernel list at embedding kernels; keep prefill forwards."""
    segs, cur = [], []
    for k in ks:
        if k['t'] == 'embed' and cur and any(x['t'] == 'moe_combine' for x in cur):
            segs.append(cur)
            cur = []
        cur.append(k)
    if cur:
        segs.append(cur)
    pre = []
    for s in segs:
        fa = [x for x in s if x['t'] == 'fa4']
        moe = [x for x in s if x['t'] in ('moe_gemm_nvfp4', 'moe_gemm_mxfp8')]
        if len(fa) >= 40 and len(moe) >= 40 and statistics.median(x['dur'] for x in fa) > 100:
            pre.append(s)
    return pre, len(segs)


def classify_forward(seg):
    """Attach visit index + refined class to each kernel of one prefill forward."""
    visit, combines = 0, 0
    phase = 'pre'          # pre: before the visit's FA4 kernel; post: after it
    pre_gemms = []
    post_gemms = 0
    out = []
    for k in seg:
        t = k['t']
        if combines >= 48:
            cls = {'gemm': 'tail GEMM (LM head / DSpark ctx K/V)'}.get(t, 'tail (final norm, aux, inject)')
            if t == 'draft_inject':
                cls = 'DSpark ctx inject (norm/rope)'
            out.append((k, None, cls))
            continue
        app = visit
        if t == 'fa4':
            phase = 'post'
        if t == 'gemm':
            if phase == 'pre':
                pre_gemms.append(k)
                cls = 'dense qkvr' if len(pre_gemms) == 1 else 'dense pre-attn small (rel einsum / dyn-conv)'
            else:
                post_gemms += 1
                cls = 'dense wo' if post_gemms == 1 else 'dense router (FP32 out)'
            out.append((k, app, cls))
            continue
        out.append((k, app, CLASS_OF.get(t, t)))
        if t == 'moe_combine':
            combines += 1
            visit += 1
            phase, pre_gemms, post_gemms = 'pre', [], 0
    # the last pre-attn GEMM of each visit is the rel-bias einsum; the others (global) are dyn-conv
    by_visit = collections.defaultdict(list)
    for i, (k, app, cls) in enumerate(out):
        if cls == 'dense pre-attn small (rel einsum / dyn-conv)':
            by_visit[app].append(i)
    for app, idx in by_visit.items():
        for j, i in enumerate(idx):
            k, a, _ = out[i]
            out[i] = (k, a, 'rel-bias einsum' if j == len(idx) - 1 else 'dense dyn-conv proj (global)')
    return out, combines


def chunk_flops(tokens, prefix):
    """Exact useful FLOPs of one chunk, by class."""
    swa_keys = sum(min(prefix + i + 1, WINDOW) for i in range(tokens))
    glob_keys = sum(prefix + i + 1 for i in range(tokens))
    f_moe = TOPK * (2 * H * 2 * I + 2 * I * H) * tokens
    return {
        'MoE grouped GEMM NVFP4': 44 * f_moe,
        'MoE grouped GEMM MXFP8': 4 * f_moe,
        'dense qkvr': (20 + 3 * 28) * 2 * H * QKVR_N * tokens,
        'dense wo': 48 * 2 * 3072 * H * tokens,
        'dense router (FP32 out)': 48 * 2 * H * E * tokens,
        'attention FA4 kernel': 4 * HEADS * HD * (40 * swa_keys + 8 * glob_keys),
        'attn_swa': 4 * HEADS * HD * swa_keys,
        'attn_global': 4 * HEADS * HD * glob_keys,
    }


# activation bytes per token per visit (roofline.py estimate) for GB/s of traffic classes
ACT_KB = {
    'rel-bias einsum': 50, 'rel-bias shear (+prep)': 110, 'gate_sink': 24.6, 'KV store': 8.2,
    'q/k/rel L2 clip': 22, 'short convs': 8.2 + 9.2 + 9.2, 'attn-res mixes': 50, 'RMSNorms': 18.4,
    'MoE routed quantize': 46.8, 'MoE swiglu-quantize': 41.8, 'MoE combine (+cast)': 59.8,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--tokens', type=int, default=0, help='prompt length (default: chunks x chunk)')
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--json', default='')
    ap.add_argument('--names', action='store_true', help='list the top unclassified kernel names')
    a = ap.parse_args()
    path, ev = load(a.trace)
    ks = []
    for e in ev:
        if e.get('cat') != 'kernel' or e.get('ph') != 'X':
            continue
        ks.append(dict(name=e['name'], ts=float(e['ts']), dur=float(e['dur']), t=tag(e['name']),
                       graph=(e.get('args') or {}).get('graph id', 0) or 0,
                       stream=(e.get('args') or {}).get('stream')))
    ks.sort(key=lambda k: k['ts'])
    rt = [e for e in ev if e.get('cat') == 'cuda_runtime' and e.get('name') == 'cudaGraphLaunch']
    pre, nseg = split_forwards(ks)
    print(f'# {path}\n{len(ks)} kernels, {nseg} forwards, {len(pre)} prefill forwards, '
          f'{len(rt)} cudaGraphLaunch')
    if not pre:
        sys.exit('no prefill forward found (need >= 40 FA4 + 40 MoE GEMM kernels per forward)')
    n_tok = a.tokens or len(pre) * a.chunk
    report = dict(trace=path, forwards=[])
    cls_tot = collections.Counter()
    raw_tot = collections.Counter()
    flops_tot = collections.Counter()
    lt = collections.defaultdict(lambda: collections.Counter())   # layer-type table
    unk = collections.Counter()
    print(f'\n## per forward (chunk {a.chunk}, prompt {n_tok} tokens)')
    print(f'{"chunk":>5} {"tokens":>6} {"prefix":>7} {"wall ms":>9} {"busy ms":>9} {"idle ms":>8} '
          f'{"graph":>6} {"kernels":>7} {"combines":>8} {"attn glob ms":>12} {"glob TFLOP/s":>12}')
    for ci, seg in enumerate(pre):
        prefix = ci * a.chunk
        tokens = max(0, min(a.chunk, n_tok - prefix))
        out, ncomb = classify_forward(seg)
        wall = seg[-1]['ts'] + seg[-1]['dur'] - seg[0]['ts']
        busy = union([(k['ts'], k['ts'] + k['dur']) for k in seg])
        ex = exclusive([(cls, k['ts'], k['dur']) for k, _, cls in out])
        for cls, v in ex.items():
            cls_tot[cls] += v
        for k, app, cls in out:
            raw_tot[cls] += k['dur']
            if cls == 'other elementwise/glue':
                unk[k['name'][:110]] += k['dur']
            if app is None:
                continue
            layer, p = stored_layer(app)
            kind = 'global' if layer in GLOBAL else 'SWA'
            if cls in ('attention FA4 kernel', 'rel-bias shear (+prep)', 'gate_sink', 'KV store',
                       'q/k/rel L2 clip', 'rel-bias einsum'):
                lt[f'attention block, {kind}'][cls] += k['dur']
                lt[f'attention block, {kind}']['_visits'] = 40 if kind == 'SWA' else 8
            if cls.startswith('MoE'):
                prec = 'NVFP4 L0-19' if layer < 20 else 'MXFP8 L20-23'
                lt[f'MoE {prec}'][cls] += k['dur']
                lt[f'MoE {prec}']['_visits'] = 44 if layer < 20 else 4
            if cls == 'dense qkvr':
                key = 'dense qkvr stacked 3M (28)' if stacked(app) else 'dense qkvr single (20)'
                lt[key]['us'] += k['dur']
                lt[key]['_visits'] = 28 if stacked(app) else 20
            lt[f'region {region(app)}']['us'] += k['dur']
        fl = chunk_flops(tokens, prefix)
        for c, v in fl.items():
            flops_tot[c] += v
        g_ms = sum(k['dur'] for k, app, cls in out
                   if cls == 'attention FA4 kernel' and app is not None and stored_layer(app)[0] in GLOBAL) / 1e3
        g_tf = fl['attn_global'] * 8 / (g_ms * 1e-3) / 1e12 if g_ms else float('nan')
        graph = sum(1 for k in seg if k['graph']) / len(seg)
        print(f'{ci:5d} {tokens:6d} {prefix:7d} {wall/1e3:9.1f} {busy/1e3:9.1f} {(wall-busy)/1e3:8.1f} '
              f'{graph*100:5.0f}% {len(seg):7d} {ncomb:8d} {g_ms:12.1f} {g_tf:12.1f}')
        report['forwards'].append(dict(chunk=ci, tokens=tokens, prefix=prefix, wall_ms=wall / 1e3,
                                       busy_ms=busy / 1e3, graph_frac=graph, kernels=len(seg),
                                       combines=ncomb, global_attn_ms=g_ms, global_tflops=g_tf,
                                       classes_ms={c: v / 1e3 for c, v in ex.items()}))
    busy_all = sum(f['busy_ms'] for f in report['forwards'])
    wall_all = sum(f['wall_ms'] for f in report['forwards'])
    print(f'\nprefill forwards: wall {wall_all:.0f} ms, busy {busy_all:.0f} ms '
          f'({n_tok / (wall_all / 1e3):.0f} tok/s on wall; {wall_all * 1e3 / n_tok:.0f} us/token)')

    print(f'\n## classes over all prefill forwards')
    print(f'{"class":<48} {"excl ms":>9} {"%":>6} {"us/tok":>7} {"raw ms":>9} {"TFLOP/s":>8} {"GB/s(model)":>11}')
    tot = sum(cls_tot.values())
    for c, v in cls_tot.most_common():
        tf = ''
        if c in flops_tot and raw_tot[c]:
            tf = f'{flops_tot[c] / (raw_tot[c] * 1e-6) / 1e12:8.1f}'
        gb = ''
        if c in ACT_KB and raw_tot[c]:
            gb = f'{ACT_KB[c] * 1e3 * 48 * n_tok / (raw_tot[c] * 1e-6) / 1e9:11.0f}'
        print(f'{c:<48} {v/1e3:9.1f} {v/tot*100:6.1f} {v/n_tok:7.1f} {raw_tot[c]/1e3:9.1f} {tf:>8} {gb:>11}')
    print(f'{"total":<48} {tot/1e3:9.1f}')

    print('\n## per layer type (raw kernel ms summed over all prefill forwards)')
    for key in sorted(lt):
        c = lt[key]
        visits = c.pop('_visits', 0)
        s = sum(c.values())
        parts = ', '.join(f'{k} {v/1e3:.1f}' for k, v in c.most_common() if k != 'us')
        per = f' = {s / 1e3 / max(1, visits) / len(pre):.2f} ms per visit per chunk' if visits else ''
        print(f'  {key:<36} {s/1e3:9.1f} ms{per}' + (f'  [{parts}]' if parts else ''))

    if a.names or unk:
        print('\n## top unclassified kernels (raw ms)')
        for n, v in unk.most_common(15):
            print(f'  {v/1e3:8.1f}  {n}')
    report.update(classes_excl_ms={c: v / 1e3 for c, v in cls_tot.items()},
                  classes_raw_ms={c: v / 1e3 for c, v in raw_tot.items()},
                  flops={c: v for c, v in flops_tot.items()}, tokens=n_tok)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'wrote {a.json}')


if __name__ == '__main__':
    main()
