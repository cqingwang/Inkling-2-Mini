#!/usr/bin/env python3
"""Per-step analysis of a SGLang torch-profiler trace of Inkling-2-Mini + DSpark on one GB10.

Usage:
  python3 -I analyze_trace.py TRACE_OR_DIR [--label c1] [--log server.log] [--json out.json]
                              [--per-layer] [--big-ms 5]

Stdlib only. What it does:
  * steps = consecutive gpu_user_annotation 'step[DRAFT ...]' starts (one draft graph + one target
    verify graph per step). Forwards are also counted by the embedding kernel (indexSelectSmallIndex)
    and by the per-layer attention prologue, never by MoE kernels.
  * kernels > --big-ms are dropped as startup artefacts (with their interval), and listed.
  * GPU busy time = union of kernel/memcpy/memset intervals (PDL overlap and side streams are merged).
    Per-class 'excl ms' is exclusive timeline occupancy (each instant attributed once), so the class
    column sums to busy time; 'raw ms' is the plain sum of kernel durations (used for GB/s).
  * kernels inside CUDA graphs are identified by name, grid and their position in the layer sequence
    (qkvr -> attention -> wo -> router -> MoE -> next qkvr; last GEMM = LM head; after argmax = draft
    context K/V; draft: 4 GEMMs per draft layer, then draft LM head, then 7 x (rnn_proj, Markov fold)).
"""
import argparse, bisect, collections, glob, gzip, json, math, os, re, statistics, sys

MB = 1e6
# ---------------------------------------------------------------- weight bytes (safetensors headers)
H = 2304
VOCAB = 201024
B_QKVR = 8600 * H * 2            # gate_attn + wq_du + wk_dv + wv_dv + wr_du (+sinks) stacked
B_WO = H * 3072 * 2
B_ROUTER = 128 * H * 2
B_LMHEAD = VOCAB * H * 2
B_CTRL_VISIT = 4 * (64 * 4608 + 2 * H * 64) * 2   # loop controllers, per core visit in passes 2-3
B_D_QKV = (3072 + 1024 + 1024) * H * 2
B_D_O = H * 3072 * 2
B_D_GU = 2 * 9216 * H * 2
B_D_DOWN = H * 9216 * 2
B_D_CTX = 2 * 1024 * H * 2       # k_proj_ctx + v_proj_ctx
B_RNN = 768 * 5120 * 2
B_FOLD = VOCAB * 256 * 2
NVFP4_W13 = H * 1152 + H * 144   # packed e2m1 + e4m3 scale per 16
NVFP4_W2 = H * 576 + H * 72
MXFP8_W13 = H * 2304 + H * 72    # e4m3 + e8m0 scale per 32
MXFP8_W2 = H * 1152 + H * 36
N_EXPERTS, TOPK = 128, 8
KV_BYTES_PER_TOKEN_VISIT = 4096  # K+V bf16 per token per layer visit (one KV head group of 128 x 8)
GLOBAL_LAYERS = {5, 11, 17, 23}

GEMM_RE = re.compile(r'_gemv_kernel|nvjet|wmma|gemvx|b12xgemm|cutlass_80|sm90_xmma|gemm_bf16|Kernel2<')


def stored_layer(app):
    """layer application index (0..47) -> (stored layer, pass)."""
    if app < 6:
        return app, 1
    if app < 42:
        return 6 + (app - 6) % 12, (app - 6) // 12 + 1
    return app - 24, 1


def load(path):
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, '*.trace.json*')))
        if not cands:
            sys.exit(f'no trace in {path}')
        path = cands[0]
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        return path, json.load(f)


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def union_len(iv):
    return sum(e - s for s, e in merge(iv))


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    return dict(n=len(xs), mean=statistics.fmean(xs) if xs else float('nan'), min=min(xs) if xs else float('nan'),
                p10=pct(xs, .1), p50=pct(xs, .5), p90=pct(xs, .9), max=max(xs) if xs else float('nan'))


# ---------------------------------------------------------------- name tags
def base_tag(n):
    if 'GemmUniversal' in n and 'BlockScaled' in n:
        return 'moe_nvfp4'
    if 'cute_sm12x_gemm' in n and 'MoeGemm' in n:
        return 'moe_mxfp8'
    if 'group_gemm_args' in n:
        return 'moe_args'
    if 'quantize_routed' in n:
        return 'moe_quant'
    if '_route_topk' in n or '_route_metadata' in n:
        return 'moe_route'
    if '_routed_partial' in n or 'bfloat16_copy_kernel' in n:
        return 'moe_combine'
    if 'splitKreduce' in n:
        return 'splitk'
    if GEMM_RE.search(n):
        return 'gemm'
    if '_attention_prologue' in n:
        return 'attn_prologue'
    if 'store_kvcache' in n or 'set_kv_buffer' in n:
        return 'attn_kvstore'
    if '_decode_partials' in n or '_combine_gate_sink' in n:
        return 'attn_relbias'
    if 'shearing' in n:
        return 'attn_relbias'
    if 'fa4_sm120' in n or 'FlashAttentionForwardCombine' in n or 'flash_fwd' in n or 'gate_sink_kernel' in n:
        return 'attn_fa4'
    if 'qk_norm_rope' in n or 'fused_rope' in n:
        return 'attn_rope'
    if 'sconv_kernel' in n:
        return 'sconv'
    if 'AttnRes' in n:
        return 'attn_res'
    if '_controller_' in n:
        return 'loop_ctrl'
    if 'rms_norm' in n or 'rmsnorm' in n.lower() or 'RMSNorm' in n:
        return 'norm'
    if 'softcap' in n:
        return 'softcap'
    if 'argmax' in n:
        return 'argmax'
    if 'indexSelectSmallIndex' in n or 'indexSelectLargeIndex' in n or 'vectorized_gather_kernel' in n:
        return 'embed'
    if '_markov_sample' in n:
        return 'markov_sample'
    if '_rnn_inputs' in n or '_rnn_gates' in n:
        return 'markov_rnn_misc'
    if 'act_and_mul' in n:
        return 'act'
    return 'glue'


class Classified:
    __slots__ = ('ev', 'cls', 'app', 'bytes', 'role')

    def __init__(self, ev, cls, app=None, nbytes=0.0, role=None):
        self.ev, self.cls, self.app, self.bytes, self.role = ev, cls, app, nbytes, role


def attn_cls(app):
    L, _ = stored_layer(app)
    return 'attention global layers (8 visits)' if L in GLOBAL_LAYERS else 'attention sliding-window (40 visits)'


def classify_verify(ks):
    """state machine over one verify window; returns list[Classified] and per-visit MoE records."""
    out = []
    n_pro = 0                 # attention prologues seen
    state = 'pre'             # pre: next gemm = qkvr of app n_pro ; attn: next = wo ; post_wo: next = router
    moe_seen = 0
    after_last = False
    after_argmax = False
    visits = []               # (app, kind, [gemm durations])
    cur_visit = None
    qkvr_done = False
    last_gemm = None
    for e in ks:
        n = e['name']
        t = base_tag(n)
        app = n_pro - 1 if n_pro else None
        if t == 'attn_prologue':
            n_pro += 1
            state = 'attn'
            qkvr_done = False
            out.append(Classified(e, attn_cls(n_pro - 1), n_pro - 1))
            continue
        if t in ('moe_nvfp4', 'moe_mxfp8'):
            kind = 'nvfp4' if t == 'moe_nvfp4' else 'mxfp8'
            if cur_visit is None or cur_visit[0] != app or len(cur_visit[2]) >= 2:
                cur_visit = [app, kind, []]
                visits.append(cur_visit)
            cur_visit[2].append(e['dur'])
            w = (NVFP4_W13, NVFP4_W2) if kind == 'nvfp4' else (MXFP8_W13, MXFP8_W2)
            role = 'w13' if len(cur_visit[2]) == 1 else 'w2'
            out.append(Classified(e, f'MoE grouped GEMM {kind.upper()} ' + ('(layers 0-19)' if kind == 'nvfp4' else '(layers 20-23)'),
                                  app, 0.0, role))
            continue
        if t == 'moe_combine' and app is not None:
            state = 'pre'
            if n_pro == 48 and '_routed_partial' in n:
                after_last = True
        if t == 'argmax':
            after_argmax = True
        if t == 'splitk':
            cls = last_gemm.cls if last_gemm else 'dense GEMM other'
            out.append(Classified(e, cls, app))
            continue
        if t == 'gemm':
            if after_argmax:
                c = Classified(e, 'draft ctx K/V proj (in verify)', None, B_D_CTX)
            elif after_last and not after_argmax:
                c = Classified(e, 'target LM head', None, B_LMHEAD)
            elif state == 'pre':
                if not qkvr_done:
                    nxt = n_pro
                    c = Classified(e, 'dense qkvr ' + ('GEMV(sm12x)' if '_gemv_kernel' in n else 'cuBLAS'), nxt, B_QKVR)
                    qkvr_done = True
                else:
                    c = Classified(e, 'global-layer conv weight_proj', n_pro, 0.0)
            elif state == 'attn':
                c = Classified(e, 'dense wo ' + ('GEMV(sm12x)' if '_gemv_kernel' in n else 'cuBLAS'), app, B_WO)
                state = 'post_wo'
            elif state == 'post_wo':
                c = Classified(e, 'MoE router gate', app, B_ROUTER)
                state = 'moe'
            else:
                c = Classified(e, 'dense GEMM other', app)
            out.append(c)
            last_gemm = c
            continue
        if t.startswith('attn_') and t != 'attn_res' and app is not None:
            out.append(Classified(e, attn_cls(app), app))
            continue
        cls = {
            'moe_args': 'MoE args/route/quant/combine', 'moe_quant': 'MoE args/route/quant/combine',
            'moe_route': 'MoE args/route/quant/combine', 'moe_combine': 'MoE args/route/quant/combine',
            'attn_kvstore': 'attention (prologue/kvstore/core)', 'attn_relbias': 'attention (prologue/kvstore/core)',
            'attn_fa4': 'attention (prologue/kvstore/core)', 'attn_rope': 'attention (prologue/kvstore/core)',
            'sconv': 'short conv', 'attn_res': 'attn-res mix', 'loop_ctrl': 'loop controllers',
            'norm': 'norms', 'softcap': 'logits softcap+argmax', 'argmax': 'logits softcap+argmax',
            'embed': 'embed + verify glue', 'act': 'dense GEMM other',
        }.get(t, 'embed + verify glue' if not after_argmax else 'accept/commit glue (in verify)')
        nb = 0.0
        if t == 'loop_ctrl' and ('_controller_mix' in n or '_controller_down' in n):
            nb = B_CTRL_VISIT / 4.0        # 2 mix + 2 down kernels per core visit in passes 2-3 (class-level GB/s only)
        out.append(Classified(e, cls, app, nb))
    return out, visits, n_pro


def classify_draft(ks):
    out = []
    gemm_i = 0
    lm_done = False
    rnn_phase = 0
    for e in ks:
        n = e['name']
        t = base_tag(n)
        if t == 'splitk':
            out.append(Classified(e, out[-1].cls if out else 'draft glue'))
            continue
        if t == 'gemm':
            if not lm_done:
                if e['dur'] > 1000 and gemm_i >= 12:
                    out.append(Classified(e, 'draft LM head (base logits)', None, B_LMHEAD))
                    lm_done = True
                else:
                    role = gemm_i % 4
                    nb = (B_D_QKV, B_D_O, B_D_GU, B_D_DOWN)[role]
                    out.append(Classified(e, 'draft layers (3 x BF16 dense)', gemm_i // 4, nb,
                                          ('qkv', 'o', 'gate_up', 'down')[role]))
                    gemm_i += 1
            else:
                if rnn_phase == 0:
                    out.append(Classified(e, 'draft Markov rnn_proj (cuBLAS gemv)', None, B_RNN))
                    rnn_phase = 1
                else:
                    out.append(Classified(e, 'draft Markov fold addmm (x7)', None, B_FOLD))
                    rnn_phase = 0
            continue
        if t in ('markov_sample', 'markov_rnn_misc', 'argmax', 'softcap'):
            out.append(Classified(e, 'draft sampling/argmax + rnn glue'))
            continue
        if not lm_done and t in ('attn_fa4', 'attn_rope', 'attn_kvstore', 'norm', 'act', 'embed'):
            out.append(Classified(e, 'draft layers (3 x BF16 dense)'))
            continue
        out.append(Classified(e, 'draft glue (prep/metadata)'))
    return out


def exclusive(items):
    """exclusive timeline occupancy per item (sorted by start), clipped by running max end."""
    items = sorted(items, key=lambda c: c.ev['ts'])
    run = -1e30
    res = []
    for c in items:
        s, e = c.ev['ts'], c.ev['ts'] + c.ev['dur']
        ex = max(0.0, e - max(s, run))
        run = max(run, e)
        res.append((c, ex))
    return res


def parse_log(path):
    acc, tput, run = [], [], []
    rx = re.compile(r'Decode batch, #running-req: (\d+).*?accept len: ([\d.]+), accept rate: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)')
    for line in open(path, errors='replace'):
        m = rx.search(line)
        if m:
            run.append(int(m.group(1)))
            acc.append(float(m.group(2)))
            tput.append(float(m.group(4)))
    return run, acc, tput


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--label', default='')
    ap.add_argument('--log')
    ap.add_argument('--json')
    ap.add_argument('--big-ms', type=float, default=5.0)
    ap.add_argument('--per-layer', action='store_true')
    ap.add_argument('--ctx-tokens', type=float, default=None,
                    help='total KV tokens across the batch (for the KV-read estimate)')
    a = ap.parse_args()
    path, d = load(a.trace)
    ev = d['traceEvents']
    gpu = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    big = [e for e in gpu if e.get('cat') == 'kernel' and e['dur'] > a.big_ms * 1000]
    bigset = set(id(e) for e in big)
    gpu = sorted([e for e in gpu if id(e) not in bigset], key=lambda e: e['ts'])
    K = [e for e in gpu if e['cat'] == 'kernel']
    ann = sorted([e for e in ev if e.get('cat') == 'gpu_user_annotation' and e['name'].startswith('step[')],
                 key=lambda e: e['ts'])
    D = [g for g in ann if 'DRAFT' in g['name']]
    V = [g for g in ann if 'VERIFY' in g['name']]
    bs = re.search(r'bs=(\d+)', D[0]['name']).group(1) if D else '?'
    starts = [e['ts'] for e in gpu]
    kst = [e['ts'] for e in K]

    def window(lo, hi, src=K, st=kst):
        i = bisect.bisect_left(st, lo)
        j = bisect.bisect_left(st, hi)
        return src[i:j]

    lines = []
    P = lines.append
    P(f'# {a.label or os.path.basename(path)}  (bs={bs})  trace={os.path.basename(path)}')
    dev = d.get('deviceProperties', [{}])[0]
    P(f'device: {dev.get("name")} sm_{dev.get("computeMajor")}{dev.get("computeMinor")} SMs={dev.get("numSms")}')
    P(f'kernels={len(K)} dropped >{a.big_ms} ms: {len(big)} ' + ' '.join(f'{e["name"][:40]}({e["dur"]/1e3:.1f}ms)' for e in big))

    # ------------------------------------------------ steps
    steps = []
    for i, dg in enumerate(D):
        vg = next((v for v in V if v['ts'] >= dg['ts']), None)
        if vg is None:
            continue
        end = D[i + 1]['ts'] if i + 1 < len(D) else None
        steps.append((dg, vg, end))
    full = [s for s in steps if s[2] is not None]
    rec = []
    for dg, vg, end in full:
        lo = dg['ts']
        acts = window(lo, end, gpu, starts)
        busy = union_len([(max(e['ts'], lo), min(e['ts'] + e['dur'], end)) for e in acts])
        period = end - lo
        d_end = dg['ts'] + dg['dur']
        v_end = vg['ts'] + vg['dur']
        dk = window(dg['ts'], d_end)
        vk = window(vg['ts'], v_end)
        d_busy = union_len([(e['ts'], min(e['ts'] + e['dur'], d_end)) for e in dk])
        v_busy = union_len([(e['ts'], min(e['ts'] + e['dur'], v_end)) for e in vk])
        moe_t = sum(e['dur'] for e in vk if base_tag(e['name']) in ('moe_nvfp4', 'moe_mxfp8'))
        rec.append(dict(moe=moe_t, nonmoe_busy=busy - moe_t, period=period, busy=busy, idle=period - busy, draft=dg['dur'], verify=vg['dur'],
                        d_idle=dg['dur'] - d_busy, v_idle=vg['dur'] - v_busy,
                        gap_dv=vg['ts'] - d_end, gap_vd=end - v_end,
                        n_emb_d=sum(1 for e in dk if base_tag(e['name']) == 'embed'),
                        n_emb_v=sum(1 for e in vk if base_tag(e['name']) == 'embed'),
                        n_pro=sum(1 for e in vk if '_attention_prologue' in e['name']),
                        eager=sum(1 for e in window(lo, end) if e['args'].get('graph id', 0) == 0)))
    P(f'\n## 1. Steps (draft graph + verify graph)  captured: {len(D)} draft / {len(V)} verify windows, '
      f'{len(full)} complete periods')
    def row(name, xs, scale=1e3):
        ds = dist([x / scale for x in xs])
        return f'| {name} | {ds["mean"]:.3f} | {ds["min"]:.3f} | {ds["p10"]:.3f} | {ds["p50"]:.3f} | {ds["p90"]:.3f} | {ds["max"]:.3f} |'
    P('| ms per step | mean | min | p10 | p50 | p90 | max |')
    P('|---|---:|---:|---:|---:|---:|---:|')
    for k, nm in (('period', 'wall period (draft start -> next draft start)'), ('busy', 'GPU busy (union)'),
                  ('idle', 'GPU idle in period'), ('draft', 'draft graph window'), ('verify', 'verify graph window'),
                  ('d_idle', '  idle inside draft window'), ('v_idle', '  idle inside verify window'),
                  ('moe', '  MoE grouped GEMMs in verify'), ('nonmoe_busy', '  busy minus MoE GEMMs'),
                  ('gap_dv', 'gap draft end -> verify start'), ('gap_vd', 'gap verify end -> next draft start')):
        P(row(nm, [r[k] for r in rec]))
    if rec:
        P(f'forwards per step: embedding kernels draft={statistics.fmean(r["n_emb_d"] for r in rec):.2f} '
          f'verify={statistics.fmean(r["n_emb_v"] for r in rec):.2f}; attention prologues per verify='
          f'{statistics.fmean(r["n_pro"] for r in rec):.2f} (48 = one 48-visit target forward); '
          f'eager (non-graph) kernels per step={statistics.fmean(r["eager"] for r in rec):.1f}')
        if len(rec) > 2:
            xs = [r['moe'] for r in rec]
            ys = [r['period'] for r in rec]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            vx = sum((x - mx) ** 2 for x in xs)
            P(f'step-time spread explained by MoE: slope d(period)/d(MoE) = {cov/vx:.2f}, '
              f'stdev period {statistics.pstdev(ys)/1e3:.2f} ms, stdev MoE {statistics.pstdev(xs)/1e3:.2f} ms')

    # ------------------------------------------------ CPU side
    rt = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for e in ev:
        if e.get('cat') in ('cuda_runtime', 'cuda_driver'):
            x = rt[e['name']]
            x[0] += 1
            x[1] += e['dur']
            x[2] = max(x[2], e['dur'])
    n = max(1, len(D))
    P('\nCPU runtime calls per step: ' + ', '.join(
        f'{k} {v[0]/n:.1f}x avg {v[1]/v[0]/1e3:.2f} ms' for k, v in sorted(rt.items(), key=lambda x: -x[1][1])[:5]))
    mem = collections.Counter((e['name'], e['args'].get('stream')) for e in ev if e.get('cat') == 'gpu_memcpy')
    P('memcpys per step: ' + ', '.join(f'{k[0]} (stream {k[1]}) {v/n:.1f}' for k, v in mem.items()))

    # ------------------------------------------------ classification
    allc = []
    visits_all = []
    for dg, vg, end in steps:
        dk = window(dg['ts'], dg['ts'] + dg['dur'])
        vk = window(vg['ts'], vg['ts'] + vg['dur'])
        allc += classify_draft(dk)
        vc, visits, _ = classify_verify(vk)
        allc += vc
        visits_all.append(visits)
    inside = set(id(c.ev) for c in allc)
    lo_all = steps[0][0]['ts'] if steps else 0
    hi_all = steps[-1][1]['ts'] + steps[-1][1]['dur'] if steps else 0
    for e in K:
        if id(e) not in inside and lo_all <= e['ts'] <= hi_all:
            allc.append(Classified(e, 'eager glue between graphs (accept/commit/prep)'))
    nsteps = len(steps)
    excl = collections.defaultdict(float)
    for c, x in exclusive(allc):
        excl[c.cls] += x
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for c in allc:
        g = agg[c.cls]
        g[0] += 1
        g[1] += c.ev['dur']
        g[2] += c.bytes
    mean_period = statistics.fmean(r['period'] for r in rec) if rec else float('nan')

    # MoE union back-solve
    moe = {'nvfp4': [0.0, 0.0, 0], 'mxfp8': [0.0, 0.0, 0]}
    per_app = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for visits in visits_all:
        for app, kind, durs in visits:
            if len(durs) != 2:
                continue
            m = moe[kind]
            m[0] += durs[0]
            m[1] += durs[1]
            m[2] += 1
            pa = per_app[app]
            pa[0] += durs[0]
            pa[1] += durs[1]
            pa[2] += 1
    rows = int(bs) * 8 if bs != '?' else 8
    u_rand = N_EXPERTS * (1 - (1 - TOPK / N_EXPERTS) ** rows)

    def u_of(t_us, nbytes, bw):
        return t_us * 1e-6 * bw * 1e9 / nbytes

    P(f'\n## 2. Kernel classes per step (mean over {nsteps} steps; mean wall period {mean_period/1e3:.2f} ms)')
    P('| class | n/step | raw ms | excl ms | % of period | MB/step | GB/s (bytes/raw) |')
    P('|---|---:|---:|---:|---:|---:|---:|')
    tot_raw = tot_ex = tot_b = 0.0
    order = sorted(agg.items(), key=lambda x: -excl[x[0]])
    for cls, (cnt, raw, nb) in order:
        ex = excl[cls]
        tot_raw += raw
        tot_ex += ex
        tot_b += nb
        gbs = f'{nb/raw/1e3:.0f}' if nb > 0 and raw > 0 else '-'
        P(f'| {cls} | {cnt/nsteps:.1f} | {raw/nsteps/1e3:.3f} | {ex/nsteps/1e3:.3f} | {100*ex/nsteps/mean_period:.1f} | '
          f'{nb/nsteps/MB:.0f} | {gbs} |')
    P(f'| **total kernels** | | {tot_raw/nsteps/1e3:.3f} | {tot_ex/nsteps/1e3:.3f} | {100*tot_ex/nsteps/mean_period:.1f} | '
      f'{tot_b/nsteps/MB:.0f} (no MoE, no KV) | |')

    # per-kernel detail for GEMM roles
    det = collections.defaultdict(lambda: [0, 0.0, 0.0, set()])
    for c in allc:
        if (c.bytes > 0 or c.role) and c.cls != 'loop controllers':
            k = (c.cls, c.role or '', tuple(c.ev['args'].get('grid', [])))
            x = det[k]
            x[0] += 1
            x[1] += c.ev['dur']
            x[2] += c.bytes
            x[3].add(re.sub(r'^void ', '', c.ev['name'])[:48])
    P('\nGEMM detail (per kernel shape): n/step, us/call, MB/call, GB/s, kernel')
    P('| class | role | grid | n/step | us/call | MB/call | GB/s | kernel |')
    P('|---|---|---|---:|---:|---:|---:|---|')
    for (cls, role, grid), (cnt, raw, nb, names) in sorted(det.items(), key=lambda x: -x[1][1]):
        gbs = f'{nb/raw/1e3:.0f}' if nb else '-'
        P(f'| {cls} | {role} | {list(grid)} | {cnt/nsteps:.1f} | {raw/cnt:.1f} | {nb/cnt/MB:.2f} | {gbs} | {sorted(names)[0]} |')

    P(f'\n### MoE expert union back-solve (rows per verify = {rows}; random union of {rows} independent top-8 '
      f'draws from 128 = {u_rand:.1f})')
    P('| kind | visits/step | us/visit w13 | us/visit w2 | w13/w2 time | u @220 | u @240 | u max @273 (hard bound) |')
    P('|---|---:|---:|---:|---:|---:|---:|---:|')
    ures = {}
    for kind, (t1, t2, cnt) in moe.items():
        if not cnt:
            continue
        w13, w2 = (NVFP4_W13, NVFP4_W2) if kind == 'nvfp4' else (MXFP8_W13, MXFP8_W2)
        tv = (t1 + t2) / cnt
        u220, u240, u273 = (u_of(tv, w13 + w2, b) for b in (220, 240, 273))
        ures[kind] = dict(us_visit=tv, u220=u220, u240=u240, u273=u273, visits=cnt / nsteps)
        P(f'| {kind} | {cnt/nsteps:.1f} | {t1/cnt:.1f} | {t2/cnt:.1f} | {t1/t2:.2f} (bytes {w13/w2:.2f}) | '
          f'{u220:.1f} | {u240:.1f} | {u273:.1f} |')
    if 'nvfp4' in ures and 'mxfp8' in ures:
        r = ures['mxfp8']['us_visit'] / ures['nvfp4']['us_visit']
        P(f'MXFP8 visit / NVFP4 visit time = {r:.2f}x (bytes per expert {(MXFP8_W13+MXFP8_W2)/(NVFP4_W13+NVFP4_W2):.2f}x)')
    moe_ms = sum((m[0] + m[1]) for m in moe.values()) / nsteps / 1e3
    P(f'MoE GEMM raw ms/step = {moe_ms:.3f}')
    if a.per_layer:
        P('\nPer layer application (mean over steps): app, stored layer, pass, w13 us, w2 us, implied u @240 (w13+w2)')
        P('| app | layer | pass | global | w13 us | w2 us | u@240 |')
        P('|---:|---:|---:|---|---:|---:|---:|')
        for app in sorted(per_app):
            t1, t2, cnt = per_app[app]
            L, ps = stored_layer(app)
            w = (NVFP4_W13 + NVFP4_W2) if L < 20 else (MXFP8_W13 + MXFP8_W2)
            P(f'| {app} | {L} | {ps} | {"G" if L in GLOBAL_LAYERS else ""} | {t1/cnt:.0f} | {t2/cnt:.0f} | '
              f'{u_of((t1+t2)/cnt, w, 240):.1f} |')
        # per stored layer
        byL = collections.defaultdict(list)
        for app in per_app:
            t1, t2, cnt = per_app[app]
            L, _ = stored_layer(app)
            w = (NVFP4_W13 + NVFP4_W2) if L < 20 else (MXFP8_W13 + MXFP8_W2)
            byL[L].append(u_of((t1 + t2) / cnt, w, 240))
        P('implied u@240 per stored layer: ' + ', '.join(f'L{L}:{statistics.fmean(v):.0f}' for L, v in sorted(byL.items())))

    # ------------------------------------------------ roofline
    known = tot_b / nsteps
    ctx = a.ctx_tokens
    kv = 48 * ctx * KV_BYTES_PER_TOKEN_VISIT if ctx else 0.0
    nv_bytes_per_u = 44 * (NVFP4_W13 + NVFP4_W2) + 4 * (MXFP8_W13 + MXFP8_W2)
    P('\n## 4. Bytes-per-step roofline')
    P(f'known weight bytes/step (dense, LM heads, draft, controllers) = {known/1e9:.3f} GB; KV read estimate '
      f'(48 visits x {ctx or 0:.0f} tokens x 4 KiB) = {kv/1e9:.3f} GB; MoE = u x {nv_bytes_per_u/MB:.1f} MB')
    P('| MoE union assumption | u | total GB | floor @220 ms | floor @240 ms | measured busy ms | busy / floor@240 |')
    P('|---|---:|---:|---:|---:|---:|---:|')
    busy = statistics.fmean(r['busy'] for r in rec) / 1e3 if rec else float('nan')
    for nm, u in (('u from MoE time @240 (MoE assumed at 240 GB/s)', ures.get('nvfp4', {}).get('u240')),
                  ('u from MoE time @220', ures.get('nvfp4', {}).get('u220')),
                  ('u hard upper bound @273', ures.get('nvfp4', {}).get('u273')),
                  (f'random union ({u_rand:.1f})', u_rand)):
        if u is None:
            continue
        tb = known + kv + u * nv_bytes_per_u
        P(f'| {nm} | {u:.1f} | {tb/1e9:.2f} | {tb/220e9*1e3:.1f} | {tb/240e9*1e3:.1f} | {busy:.1f} | {busy/(tb/240e9*1e3):.2f} |')
    nonmoe_ms = (tot_ex / nsteps / 1e3) - moe_ms
    P(f'non-MoE kernel occupancy = {nonmoe_ms:.2f} ms vs known+KV bytes floor {((known+kv)/240e9)*1e3:.2f} ms @240 / '
      f'{((known+kv)/220e9)*1e3:.2f} ms @220')

    if a.log:
        run, acc, tput = parse_log(a.log)
        P('\n## 5. Server log (decode_log_interval windows; accept len = tokens per step per request incl. bonus)')
        for r in sorted(set(run)):
            xs = [x for x, y in zip(acc, run) if y == r]
            ts = [x for x, y in zip(tput, run) if y == r]
            dd = dist(xs)
            P(f'running-req={r}: windows={len(xs)} accept len mean={dd["mean"]:.2f} min={dd["min"]:.2f} '
              f'p50={dd["p50"]:.2f} max={dd["max"]:.2f}; gen tok/s median={pct(ts,.5):.1f}')
    print('\n'.join(lines))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dict(steps=rec, classes={k: dict(n=v[0] / nsteps, raw_ms=v[1] / nsteps / 1e3,
                                                         excl_ms=excl[k] / nsteps / 1e3, mb=v[2] / nsteps / MB)
                                                for k, v in agg.items()},
                           moe=ures, per_app={k: v for k, v in per_app.items()}), f, indent=1)


if __name__ == '__main__':
    main()
