#!/usr/bin/env python3
"""Per-step, per-class analysis of a torch-profiler trace of the SHIPPED Inkling-2-Mini overlay stack (1x GB10).

    python3 -I analyze2.py TRACE_OR_DIR [--label c4] [--json out.json] [--long-ms 5] [--ctx 450] [--steps-detail]

Stdlib only. Replaces sglang-survey/analyze_trace.py for the round-2 stack (pico gemv16, wide-free cuBLAS above
16 rows, FP8 draft head + Markov fold + MXFP8 draft layers in `_w8_mm_kernel`, vcap `_remap_rows_kernel`,
prefill-overlay qkvr split at >= 128 rows/source, CuTe MXFP8 MoE in `cutlass::device_kernel`).

Method
  * A step = one 'step[DRAFT bs=N]' GPU annotation start to the next one (40 profiled steps -> 39 complete
    periods). Inside a period: draft window, gap draft->verify, verify window, gap verify->next draft.
    GPU events (kernels, memcpy, memset) are assigned to a period by start time; events outside every complete
    period are counted and reported, never dropped silently. Long kernels (> --long-ms) are LISTED and KEPT.
  * busy = union of event intervals inside the period; idle = period - busy.
    'excl' = exclusive occupancy: an instant covered by several events (PDL overlap, side streams) belongs to the
    EARLIEST-started active event (with PDL the later kernel is waiting on griddepcontrol.wait). Sum of excl over
    classes == busy exactly. 'raw' = plain sum of durations (used for GB/s).
  * Classification: CUDA-graph kernels are classified by name and by their position in the target's 48-visit
    layer sequence (prologue counting; PRE -> ATTN -> wo -> POST -> router -> MoE -> EPI), and in the draft
    sequence (12 layer GEMMs -> head -> 7 x (rnn_proj, fold)). Eager kernels (graph id 0) are classified by the
    gap they run in. Every event gets a class; unknown names inside a known phase go to that phase's glue class.
  * Bytes: weight bytes per call from the safetensors shapes (BF16 2 B, pico bf16x packed 0.7738/0.7729/0.7720 of
    BF16 for qkvr/wo/head, FP8 draft copies 1 B + scales). MoE: no per-call byte model (expert union not in the
    trace); the report gives the union implied at 220 GB/s. Attention: KV floor from an ASSUMED mean context
    (--ctx), K+V BF16 4 KiB per token per visit, SWA visits read min(ctx, 1024) tokens.
"""
import argparse, bisect, collections, glob, gzip, json, math, os, re, statistics, sys

MB = 1e6
H = 2304
VOCAB = 201024
# ------------------------------------------------------------------ weight bytes (MB per call)
QKVR_BF16 = 8600 * H * 2 / MB            # 39.63
KVPACK_BF16 = 2048 * H * 2 / MB          # 9.44 (pico's coded copy of the adjacent k,v rows)
WO_BF16 = H * 3072 * 2 / MB              # 14.16
HEAD_BF16 = VOCAB * H * 2 / MB           # 926.3
PK_QKVR, PK_WO, PK_HEAD = 0.7738, 0.7729, 0.7720   # m0/offline-analysis.md (measured bf16x / BF16)
ROUTER_BF16 = 128 * H * 2 / MB           # 0.59
WPROJ_BF16 = 64 * H * 2 / MB / 2         # 0.15 per call (one 64x2304 tensor per global layer, 2 calls / visit)
CTRL_VISIT_BF16 = 113.2 / 24             # 4.72 MB per looped visit in passes 2-3 (offline-analysis.md)
# draft (FP8 e4m3 1 B/elem; draftmx8 = MX e8m0 per 32; head8/fold8 = fp32 per row)
D_W8 = {'qkv': 5120 * H * (1 + 1 / 32) / MB, 'o': H * 3072 * (1 + 1 / 32) / MB,
        'gate_up': 18432 * H * (1 + 1 / 32) / MB, 'down': H * 9216 * (1 + 1 / 32) / MB}
D_BF16 = {'qkv': 5120 * H * 2 / MB, 'o': H * 3072 * 2 / MB, 'gate_up': 18432 * H * 2 / MB, 'down': H * 9216 * 2 / MB}
D_HEAD8 = (VOCAB * H + VOCAB * 4) / MB   # 463.9
D_FOLD8 = (VOCAB * 256 + VOCAB * 4) / MB # 52.3
D_RNN = 768 * 5120 * 2 / MB              # 7.86 (rnn_proj BF16)
D_CTX_W8 = 2048 * H * (1 + 1 / 32) / MB  # 4.87
D_CTX_BF16 = 2048 * H * 2 / MB           # 9.44
MOE_UNIT_MB = 229.92                     # MB per unit of expert union over the 48 visits (NVFP4 197.1 + MXFP8 32.8)
NVFP4_EXPERT_MB = (H * 1296 + H * 648) / MB   # 4.479 per expert per visit (w13 + w2, incl. e4m3 scales)
MXFP8_EXPERT_MB = (H * 2376 + H * 1188) / MB  # 8.211
KV_TOKEN_VISIT = 8 * 128 * 2 * 2         # bytes: 8 kv heads x 128 x (K,V) x BF16
WINDOW = 1024
GLOBAL_LAYERS = {5, 11, 17, 23}


def stored_layer(app):
    if app < 6:
        return app
    if app < 42:
        return 6 + (app - 6) % 12
    return app - 24


def is_global(app):
    return stored_layer(app) in GLOBAL_LAYERS


def three_sources(app):
    """visits whose q/k/v inputs differ: global layers (own q/k/v norms) and looped visits in passes 2-3."""
    return is_global(app) or 18 <= app < 42


# ------------------------------------------------------------------ name tags
GEMM_PAT = [('k_gemv16_r16', 'pico_r16'), ('k_gemv16<', 'pico'), ('_w8_mm_kernel', 'w8'),
            ('SmallNGemvKernel', 'b12x'), ('nvjet_', 'nvjet'), ('cutlass::Kernel2<', 'k2'),
            ('internal::gemvx', 'gemvx'), ('std::enable_if<!(false), void>::type internal::g', 'gemvx'),
            ('b12xgemm', 'b12x')]


def tag(e):
    n = e['name']
    cat = e.get('cat')
    if cat == 'gpu_memcpy':
        return 'memcpy'
    if cat == 'gpu_memset':
        return 'memset'
    if 'GemmUniversal' in n and 'BlockScaled' in n:
        return 'moe_nvfp4'
    if 'cutlass::device_kernel' in n and ('MoeGemm' in n or 'cute_sm12x' in n or 'flashinfer::gemm' in n):
        return 'moe_mxfp8'
    if 'splitKreduce' in n:
        return 'splitk'
    for p, t in GEMM_PAT:
        if p in n:
            return 'gemm:' + t
    table = [
        ('group_gemm_args', 'moe_args'), ('swiglu_quantize_routed', 'moe_quant'), ('quantize_routed', 'moe_quant'),
        ('_route_topk', 'moe_route'), ('_route_metadata', 'moe_route'), ('route_metadata_kernel', 'moe_route'),
        ('_remap_rows_kernel', 'vcap_remap'), ('_routed_partial', 'moe_combine'),
        ('_attention_prologue', 'attn_prologue'), ('store_kvcache', 'attn_kvstore'),
        ('_decode_partials', 'attn_core'), ('_combine_gate_sink', 'attn_epi'), ('gate_sink_kernel', 'attn_epi'),
        ('ShearingBias', 'attn_bias'), ('shearing', 'attn_bias'), ('flash_fwd_combine', 'attn_epi'),
        ('fa4_sm120flash_fwd', 'attn_core'), ('_table_qk_norm_rope', 'attn_rope'), ('fused_rope', 'attn_rope'),
        ('set_kv_buffer', 'attn_kvstore'),
        ('sconv_kernel', 'sconv'), ('AttnRes', 'attn_res'), ('_controller_', 'ctrl'),
        ('_mix_rmsnorm', 'norm'), ('rms_norm', 'norm'), ('RMSNorm', 'norm'), ('rmsnorm', 'norm'),
        ('softcap', 'softcap'), ('argmax', 'argmax'), ('act_and_mul', 'act'),
        ('indexSelect', 'embed'), ('vectorized_gather_kernel', 'embed'),
        ('_rnn_inputs', 'rnn'), ('_rnn_gates', 'rnn'), ('_markov_sample', 'sample'),
        ('_fused_conv_window_scatter', 'conv_commit'),
        ('_fused_correct_drafts', 'accept'), ('_cap_correct_len', 'accept'), ('_gather_row_bonus', 'accept'),
        ('_finalize_accept_lens', 'accept'), ('_build_out_tokens', 'accept'), ('_fused_commit_track', 'accept'),
        ('_fused_decode_metadata', 'meta'), ('_fused_extend_metadata', 'meta'), ('_fused_replay_state', 'meta'),
        ('assign_extend_cache_locs', 'meta'), ('alloc_extend_kernel', 'meta'), ('get_last_loc', 'meta'),
        ('assign_req_to_token_pool', 'meta'),
    ]
    for p, t in table:
        if p in n:
            return t
    return 'misc'   # PyTorch elementwise / copy / scan / memcpy32_post / small_copy


def grid(e):
    return tuple(e.get('args', {}).get('grid') or (0, 0, 0))


def is_graph(e):
    return (e.get('args', {}).get('graph id') or 0) != 0


# ------------------------------------------------------------------ helpers
def load(path):
    if os.path.isdir(path):
        c = sorted(glob.glob(os.path.join(path, '*.trace.json*')))
        if not c:
            sys.exit(f'no trace in {path}')
        path = c[0]
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        return path, json.load(f)


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class K:
    __slots__ = ('e', 'ts', 'te', 'dur', 'tag', 'cls', 'mb', 'seg', 'excl', 'note')

    def __init__(self, e):
        self.e, self.ts, self.dur = e, e['ts'], e['dur']
        self.te = self.ts + self.dur
        self.tag = tag(e)
        self.cls, self.mb, self.seg, self.excl, self.note = None, 0.0, None, 0.0, ''


# ------------------------------------------------------------------ verify window classifier
def classify_verify(ks, counts):
    """ks: graph kernels of one verify window in time order. Sets k.cls / k.mb. Returns validation counters."""
    phase = 'PREP'
    nvis = 0            # prologues seen; current visit = nvis - 1
    pre_gemms = []      # GEMM K objects before the next prologue
    attach = {}         # id(pre GEMM) -> its cuBLAS splitKreduce kernels
    moe_in_visit = 0
    last_gemm = None
    akind = lambda app: 'glob' if is_global(app) else 'swa'
    for k in ks:
        t = k.tag
        app = nvis - 1
        if t == 'splitk':
            if phase == 'PRE' and pre_gemms:
                attach.setdefault(id(pre_gemms[-1]), []).append(k)   # class fixed with its GEMM at the prologue
                k.cls = 'dense.qkvr'
            else:
                k.cls = last_gemm.cls if last_gemm is not None else 'fwd.glue'
            continue
        if t == 'attn_prologue':
            # finalize the PRE gemms of visit nvis
            v = nvis
            g = [x for x in pre_gemms]
            nw = 2 if (is_global(v) and len(g) >= 3) else 0
            qk = g[:len(g) - nw] if nw else g
            for x in g[len(g) - nw:]:
                x.cls, x.mb = 'dense.conv_wproj', WPROJ_BF16
            for x in g:
                for y in attach.pop(id(x), []):
                    y.cls = x.cls if x.cls == 'dense.conv_wproj' else 'dense.qkvr'
            kinds = {x.tag for x in qk}
            if 'gemm:pico' in kinds:
                for x in qk:
                    if x.tag == 'gemm:pico':
                        x.cls, x.mb = 'dense.qkvr', QKVR_BF16 * PK_QKVR
                    elif x.tag == 'gemm:pico_r16':
                        x.cls, x.mb = 'dense.qkvr', KVPACK_BF16 * PK_QKVR
                    else:
                        x.cls, x.mb = 'dense.qkvr', 0.0
                counts['qkvr_pico_visits'] += 1
            else:
                for i, x in enumerate(qk):
                    x.cls = 'dense.qkvr'
                    x.mb = QKVR_BF16 / len(qk)   # stacked (1 call) or prefill split (4 calls): one weight read
                counts['qkvr_cublas_visits'] += 1
                counts[f'qkvr_calls_{len(qk)}'] += 1
            counts['qkvr_visits'] += 1 if qk else 0
            counts['wproj_calls'] += nw
            pre_gemms = []
            nvis += 1
            moe_in_visit = 0
            phase = 'ATTN'
            k.cls = f'attn.{akind(nvis - 1)}.prologue'
            counts['prologues'] += 1
            continue
        if phase in ('PREP', 'PRE', 'EPI'):
            if t.startswith('gemm:'):
                if phase == 'EPI' and nvis == 48:
                    phase = 'HEAD'
                    k.cls = 'lmhead'
                    k.mb = HEAD_BF16 * (PK_HEAD if t == 'gemm:pico' else 1.0)
                    counts['lmhead'] += 1
                    last_gemm = k
                    continue
                phase = 'PRE'
                k.cls = 'dense.qkvr'   # provisional, fixed at the prologue
                pre_gemms.append(k)
                last_gemm = k
                continue
            if phase == 'PREP' and t in ('meta', 'embed', 'misc', 'memcpy', 'memset'):
                k.cls = 'verify.prep'
                continue
            if phase == 'PREP':
                phase = 'PRE'
        if phase == 'ATTN':
            if t.startswith('gemm:'):
                k.cls = 'dense.wo'
                k.mb = WO_BF16 * (PK_WO if t == 'gemm:pico' else 1.0)
                counts['wo'] += 1
                phase = 'POST'
                last_gemm = k
                continue
            sub = {'attn_kvstore': 'prologue', 'attn_rope': 'prologue', 'attn_bias': 'bias',
                   'attn_core': 'core', 'attn_epi': 'epi'}.get(t, 'glue')
            k.cls = f'attn.{akind(app)}.{sub}'
            continue
        if phase == 'POST' and t.startswith('gemm:'):
            k.cls, k.mb = 'dense.router', ROUTER_BF16
            counts['router'] += 1
            phase = 'MOE'
            last_gemm = k
            continue
        if phase == 'MOE' or (phase == 'POST' and t in ('moe_route', 'vcap_remap')):
            if t in ('moe_nvfp4', 'moe_mxfp8'):
                kind = 'nvfp4' if t == 'moe_nvfp4' else 'mxfp8'
                role = 'w13' if moe_in_visit == 0 else 'w2'
                moe_in_visit += 1
                k.cls = f'moe.{kind}.{role}'
                k.note = str(app)
                counts['moe_' + kind] += 1
                continue
            if t == 'moe_combine':
                k.cls = 'moe.glue.combine'
                phase = 'EPI'
                continue
            k.cls = {'moe_route': 'moe.glue.route', 'vcap_remap': 'moe.glue.vcap_remap', 'moe_quant': 'moe.glue.quant',
                     'moe_args': 'moe.glue.args'}.get(t, 'moe.glue.quant')   # MXFP8 scale/pad elementwise
            continue
        if phase == 'HEAD':
            if t in ('softcap', 'argmax'):
                k.cls = 'logits.softcap_argmax'
                if t == 'argmax':
                    phase = 'TAIL'
                continue
            k.cls = 'logits.softcap_argmax' if t == 'misc' else 'verify.accept'
            continue
        if phase == 'TAIL':
            if t.startswith('gemm:'):
                k.cls = 'draft.ctxkv'
                k.mb = D_CTX_W8 if t == 'gemm:w8' else D_CTX_BF16
                counts['ctxkv'] += 1
                last_gemm = k
                continue
            if t in ('norm', 'attn_rope', 'attn_kvstore'):
                k.cls = 'draft.ctxkv'
                continue
            k.cls = 'verify.accept'
            continue
        # PRE / POST / EPI non-GEMM kernels of the layer stack
        k.cls = {'sconv': 'small.sconv', 'attn_res': 'small.attn_res', 'ctrl': 'small.ctrl', 'norm': 'small.norm',
                 'act': 'fwd.glue'}.get(t, 'fwd.glue')
        if t == 'ctrl' and '_controller_mix' in k.e['name']:
            k.mb = CTRL_VISIT_BF16 / 2      # mix kernels read gate/gain weights; class-level GB/s only (approx.)
    for k in pre_gemms:       # GEMMs after the last prologue that never met a prologue (should not happen)
        k.cls = 'fwd.glue'
        counts['orphan_pre_gemms'] += 1
    return nvis


# ------------------------------------------------------------------ draft window classifier
ROLES = ('qkv', 'o', 'gate_up', 'down')


def classify_draft(ks, counts):
    g = 0              # draft-layer GEMMs seen
    state = 'prep'     # prep -> layers -> head -> rnn
    pending = None     # 'rnn' after _rnn_inputs, 'fold' after _rnn_gates
    last_gemm = None
    for k in ks:
        t = k.tag
        if t == 'splitk':
            k.cls = last_gemm.cls if last_gemm is not None else 'draft.misc'
            continue
        if t.startswith('gemm:'):
            last_gemm = k
            if state in ('prep', 'layers') and g < 12:
                role = ROLES[g % 4]
                k.cls = 'draft.layers.gemm'
                k.note = role
                k.mb = D_W8[role] if t == 'gemm:w8' else D_BF16[role]
                g += 1
                state = 'layers'
                counts['draft_layer_gemms'] += 1
                continue
            if state == 'layers':
                k.cls, k.mb, state = 'draft.head', (D_HEAD8 if t == 'gemm:w8' else HEAD_BF16), 'head'
                counts['draft_head'] += 1
                continue
            if pending == 'rnn':
                k.cls, k.mb, pending = 'draft.rnn', D_RNN, None
                counts['draft_rnn_proj'] += 1
                continue
            if pending == 'fold':
                k.cls, k.mb, pending = 'draft.fold', (D_FOLD8 if t == 'gemm:w8' else 2 * D_FOLD8), None
                counts['draft_fold'] += 1
                continue
            k.cls = 'draft.misc'
            counts['draft_unplaced_gemm'] += 1
            continue
        if t == 'rnn':
            state = 'rnn'
            pending = 'rnn' if '_rnn_inputs' in k.e['name'] else 'fold'
            k.cls = 'draft.rnn'
            continue
        if t == 'sample':
            k.cls = 'draft.sample'
            continue
        if state == 'prep':
            k.cls = 'draft.prep'
            continue
        if state == 'layers':
            if t.startswith('attn_'):
                k.cls = 'draft.layers.attn'
            else:
                k.cls = 'draft.layers.misc'
            continue
        k.cls = 'draft.misc'


# ------------------------------------------------------------------ main analysis
def analyze(path, long_ms=5.0, ctx=None):
    path, tr = load(path)
    ev = tr['traceEvents']
    gpu = sorted((K(e) for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')),
                 key=lambda k: k.ts)
    ann = sorted((e for e in ev if e.get('ph') == 'X' and e.get('cat') == 'gpu_user_annotation'
                  and str(e.get('name', '')).startswith('step[')), key=lambda e: e['ts'])
    drafts = [a for a in ann if a['name'].startswith('step[DRAFT')]
    verifs = [a for a in ann if a['name'].startswith('step[VERIFY')]
    bs = int(re.search(r'bs=(\d+)', drafts[0]['name']).group(1))
    props = tr.get('deviceProperties') or [{}]
    starts = [k.ts for k in gpu]
    periods = []
    for i in range(len(drafts) - 1):
        d, dn = drafts[i], drafts[i + 1]
        v = [x for x in verifs if d['ts'] <= x['ts'] < dn['ts']]
        if len(v) != 1:
            continue
        periods.append((d, v[0], dn['ts']))
    counts_all = []
    steps = []
    assigned = set()
    for d, v, tend in periods:
        t0 = d['ts']
        i0, i1 = bisect.bisect_left(starts, t0), bisect.bisect_left(starts, tend)
        ks = gpu[i0:i1]
        dend, vs, vend = d['ts'] + d['dur'], v['ts'], v['ts'] + v['dur']
        for k in ks:
            assigned.add(id(k))
            k.seg = ('draft' if k.ts < dend else 'gap_dv' if k.ts < vs else 'verify' if k.ts < vend else 'gap_vd')
        counts = collections.Counter()
        dk = [k for k in ks if k.seg == 'draft' and is_graph(k.e) and k.tag not in ('memcpy', 'memset')]
        vk = [k for k in ks if k.seg == 'verify' and is_graph(k.e) and k.tag not in ('memcpy', 'memset')]
        classify_draft(dk, counts)
        nvis = classify_verify(vk, counts)
        for k in ks:
            if k.cls is not None:
                continue
            if k.tag in ('memcpy', 'memset'):
                k.cls = 'mem.copy_set'      # incl. in-graph memsets (c8: 48/step, FA4 split workspace)
                continue
            if k.tag == 'conv_commit':
                k.cls = 'eager.conv_state_commit'
            elif k.seg == 'gap_dv' or (k.seg == 'verify' and not is_graph(k.e)):
                k.cls = 'eager.draft_to_verify'
            elif k.seg == 'draft':
                k.cls = 'eager.draft_prep'
            else:
                k.cls = 'eager.verify_to_draft'
            if k.seg in ('draft', 'verify') and is_graph(k.e):
                counts['graph_unclassified'] += 1
        # exclusive occupancy: earliest-started active event owns each instant
        cover = t0
        for k in sorted(ks, key=lambda x: (x.ts, -x.te)):
            s, e = max(k.ts, t0), min(k.te, tend)
            k.excl = max(0.0, e - max(s, cover))
            cover = max(cover, e)
        busy = sum(e - s for s, e in merge([(max(k.ts, t0), min(k.te, tend)) for k in ks]))
        win_idle = {}
        for name, a, b in (('draft', t0, dend), ('verify', vs, vend)):
            iv = merge([(max(k.ts, a), min(k.te, b)) for k in ks if k.te > a and k.ts < b])
            win_idle[name] = (b - a) - sum(max(0, y - x) for x, y in iv)
        cls = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        for k in ks:
            c = cls[k.cls]
            c[0] += 1
            c[1] += k.dur
            c[2] += k.excl
            c[3] += k.mb
        rows = 8 * bs
        pro = [k for k in ks if k.tag == 'attn_prologue' and k.seg == 'verify']
        if pro:
            rows = grid(pro[0].e)[0]
        steps.append(dict(period=tend - t0, busy=busy, idle=(tend - t0) - busy, draft=d['dur'], verify=v['dur'],
                          gap_dv=vs - dend, gap_vd=tend - vend, idle_draft=win_idle['draft'],
                          idle_verify=win_idle['verify'], cls={c: list(x) for c, x in cls.items()},
                          counts=dict(counts), nvis=nvis, rows=rows,
                          moe_visits=[(k.cls, k.dur, k.note) for k in ks if k.cls and k.cls.startswith('moe.') and
                                      ('w13' in k.cls or 'w2' in k.cls)]))
        counts_all.append(counts)
    outside = [k for k in gpu if id(k) not in assigned]
    longk = [k for k in gpu if k.dur > long_ms * 1000]
    return dict(path=path, bs=bs, device=(props[0].get('name'), props[0].get('numSms')), steps=steps,
                outside=(len(outside), sum(k.dur for k in outside)),
                longk=[(k.e['name'][:60], k.dur, k.cls or 'outside-step', k.seg) for k in longk],
                n_events=len(gpu), n_drafts=len(drafts), n_verifs=len(verifs))


GROUPS = [  # (group, class prefixes) for the summary table; order = report order
    ('MoE grouped GEMM', ('moe.nvfp4', 'moe.mxfp8')),
    ('MoE glue (route/remap/quant/args/combine)', ('moe.glue',)),
    ('dense qkvr', ('dense.qkvr',)),
    ('dense wo', ('dense.wo',)),
    ('router + global conv wproj', ('dense.router', 'dense.conv_wproj')),
    ('target LM head', ('lmhead',)),
    ('logits softcap+argmax', ('logits.',)),
    ('attention SWA (40 visits)', ('attn.swa',)),
    ('attention global (8 visits)', ('attn.glob',)),
    ('small: norms/attn-res/sconv/controllers', ('small.',)),
    ('fwd glue (copies/act in layer stack)', ('fwd.glue',)),
    ('verify prep + accept/commit (graph)', ('verify.',)),
    ('draft layers (3)', ('draft.layers', 'draft.prep')),
    ('draft vocab: head (FP8)', ('draft.head',)),
    ('draft vocab: Markov fold (FP8) x7', ('draft.fold',)),
    ('draft rnn + sample + misc', ('draft.rnn', 'draft.sample', 'draft.misc')),
    ('draft ctx K/V (in verify graph)', ('draft.ctxkv',)),
    ('eager kernels between graphs', ('eager.',)),
    ('memcpy/memset', ('mem.',)),
]


def mean_cls(steps):
    n = len(steps)
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for s in steps:
        for c, (cnt, raw, excl, mb) in s['cls'].items():
            a = agg[c]
            a[0] += cnt / n
            a[1] += raw / n / 1e3
            a[2] += excl / n / 1e3
            a[3] += mb / n
    return agg


def kv_floor_gb(bs, ctx):
    """KV bytes read per verify step (GB): bs requests x (40 SWA x min(ctx,1024) + 8 global x ctx) tokens x 4 KiB."""
    if ctx is None:
        return None
    return bs * (40 * min(ctx, WINDOW) + 8 * ctx) * KV_TOKEN_VISIT / 1e9


def report(r, label, ctx=None, detail=False):
    st = r['steps']
    n = len(st)
    out = []
    p = out.append
    p(f"# {label}  bs={r['bs']}  rows/verify={st[0]['rows']}  trace={os.path.basename(r['path'])}")
    p(f"device {r['device'][0]} SMs={r['device'][1]}; GPU events={r['n_events']}; annotations draft={r['n_drafts']} "
      f"verify={r['n_verifs']}; complete periods={n}")
    p(f"events outside complete periods (kept out of per-step numbers, not dropped): {r['outside'][0]} "
      f"({r['outside'][1] / 1e3:.2f} ms total, i.e. the partial last step and pre-profile tail)")
    if r['longk']:
        p(f"long kernels > threshold (KEPT): {len(r['longk'])}: " + ', '.join(
            sorted({f'{nm[:24]}[{c}] {d / 1e3:.1f}ms' for nm, d, c, s in r['longk']})[:6]))
    else:
        p("long kernels > threshold: none")
    p("")
    p("## Step timeline (ms per step, 39 complete periods)")
    p("| | mean | p10 | p50 | p90 |")
    p("|---|---:|---:|---:|---:|")
    for key, name in (('period', 'wall period'), ('busy', 'GPU busy (union)'), ('idle', 'GPU idle'),
                      ('draft', 'draft window'), ('idle_draft', '  idle in draft window'),
                      ('gap_dv', 'gap draft->verify'), ('verify', 'verify window'),
                      ('idle_verify', '  idle in verify window'), ('gap_vd', 'gap verify->next draft')):
        xs = [s[key] / 1e3 for s in st]
        p(f"| {name} | {statistics.fmean(xs):.3f} | {pct(xs, .1):.3f} | {pct(xs, .5):.3f} | {pct(xs, .9):.3f} |")
    agg = mean_cls(st)
    busy = statistics.fmean(s['busy'] for s in st) / 1e3
    period = statistics.fmean(s['period'] for s in st) / 1e3
    tot_excl = sum(a[2] for a in agg.values())
    p("")
    p(f"## Classes (mean per step; excl sums to busy: {tot_excl:.3f} vs {busy:.3f} ms, diff {tot_excl - busy:+.4f})")
    p("| group | n/step | raw ms | excl ms | % period | weight MB/step | GB/s (MB/raw) |")
    p("|---|---:|---:|---:|---:|---:|---:|")
    used = set()
    gsum = {}
    for gname, prefs in GROUPS:
        cs = [c for c in agg if c.startswith(prefs) and c not in used]
        used.update(cs)
        cnt = sum(agg[c][0] for c in cs)
        raw = sum(agg[c][1] for c in cs)
        exc = sum(agg[c][2] for c in cs)
        mb = sum(agg[c][3] for c in cs)
        gsum[gname] = dict(n=cnt, raw=raw, excl=exc, mb=mb)
        gbs = f"{mb / raw:.0f}" if mb and raw else '-'
        p(f"| {gname} | {cnt:.1f} | {raw:.3f} | {exc:.3f} | {100 * exc / period:.1f} | {mb:.0f} | {gbs} |")
    left = [c for c in agg if c not in used]
    for c in left:
        p(f"| UNGROUPED {c} | {agg[c][0]:.1f} | {agg[c][1]:.3f} | {agg[c][2]:.3f} | | | |")
    p(f"| **busy** | | {sum(a[1] for a in agg.values()):.3f} | {tot_excl:.3f} | {100 * tot_excl / period:.1f} | | |")
    p(f"| idle | | | {period - busy:.3f} | {100 * (period - busy) / period:.1f} | | |")
    p("")
    p("### Class detail")
    p("| class | n/step | raw ms | excl ms | MB/step | GB/s |")
    p("|---|---:|---:|---:|---:|---:|")
    for c in sorted(agg, key=lambda c: -agg[c][2]):
        a = agg[c]
        p(f"| {c} | {a[0]:.1f} | {a[1]:.3f} | {a[2]:.3f} | {a[3]:.1f} | {(a[3] / a[1]) if a[3] and a[1] else 0:.0f} |")
    # MoE union implied
    p("")
    nv = statistics.fmean(sum(1 for c, d, a in s['moe_visits'] if 'w13' in c and 'nvfp4' in c) for s in st)
    nm = statistics.fmean(sum(1 for c, d, a in s['moe_visits'] if 'w13' in c and 'mxfp8' in c) for s in st)
    tn = sum(agg[c][1] for c in agg if c.startswith('moe.nvfp4'))
    tm = sum(agg[c][1] for c in agg if c.startswith('moe.mxfp8'))
    un = tn * 1e-3 * 220e9 / (nv * NVFP4_EXPERT_MB * 1e6) if nv else 0
    um = tm * 1e-3 * 220e9 / (nm * MXFP8_EXPERT_MB * 1e6) if nm else 0
    p(f"MoE: NVFP4 {nv:.0f} visits {tn:.2f} ms ({tn / nv * 1e3:.0f} us/visit) -> union u = {un:.1f} experts/visit at "
      f"220 GB/s; MXFP8 {nm:.0f} visits {tm:.2f} ms ({tm / max(nm, 1) * 1e3:.0f} us/visit) -> u = {um:.1f} at 220 GB/s "
      f"(MXFP8 CuTe kernel rate not measured in isolation: its u is indicative only)")
    kv = kv_floor_gb(r['bs'], ctx)
    if kv:
        at = sum(agg[c][2] for c in agg if c.startswith('attn.'))
        p(f"Attention: excl {at:.2f} ms; KV floor at ASSUMED mean ctx {ctx} tok: {kv:.2f} GB = {kv / 0.24:.2f} ms at "
          f"240 GB/s -> measured / floor = {at / (kv / 0.24):.1f}x")
    # validation
    p("")
    p("## Validation (per step)")
    cnt = collections.Counter()
    for s in st:
        for k2, v2 in s['counts'].items():
            cnt[k2] += v2 / n
    exp = dict(prologues=48, moe_nvfp4=88, moe_mxfp8=8, wo=48, router=48, lmhead=1, qkvr_visits=48,
               wproj_calls=16, draft_layer_gemms=12, draft_head=1, draft_rnn_proj=7, draft_fold=7)
    bad = []
    for k2, e2 in exp.items():
        ok = abs(cnt.get(k2, 0) - e2) < 1e-9
        if not ok:
            bad.append(f"{k2}={cnt.get(k2, 0):.2f} (expected {e2})")
    p("expected counts: " + ('ALL OK (' + ', '.join(f'{k2}={e2}' for k2, e2 in exp.items()) + ')' if not bad
                             else 'MISMATCH: ' + '; '.join(bad)))
    extra = {k2: round(v2, 2) for k2, v2 in cnt.items() if k2 not in exp}
    p(f"other counters: {extra}")
    p(f"per-step |sum(excl) - busy| max = {max(abs(sum(c[2] for c in s['cls'].values()) - s['busy']) for s in st):.4f} us")
    if detail:
        p("")
        p("## Per step: period / busy / MoE ms")
        for i, s in enumerate(st):
            moe = sum(v[1] for c, v in s['cls'].items() if c.startswith('moe.nvfp4') or c.startswith('moe.mxfp8'))
            p(f"{i:2d} {s['period'] / 1e3:7.2f} {s['busy'] / 1e3:7.2f} {moe / 1e3:7.2f}")
    return '\n'.join(out), dict(label=label, bs=r['bs'], rows=st[0]['rows'], period=period, busy=busy,
                                idle=period - busy, groups=gsum,
                                classes={c: dict(n=a[0], raw=a[1], excl=a[2], mb=a[3]) for c, a in agg.items()},
                                moe=dict(nvfp4_ms=tn, mxfp8_ms=tm, u_nvfp4_220=un, u_mxfp8_220=um),
                                timeline={k2: statistics.fmean(s[k2] for s in st) / 1e3 for k2 in
                                          ('draft', 'verify', 'gap_dv', 'gap_vd', 'idle_draft', 'idle_verify')},
                                counts=dict(cnt), ctx=ctx, kv_gb=kv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--label', default='')
    ap.add_argument('--json')
    ap.add_argument('--long-ms', type=float, default=5.0)
    ap.add_argument('--ctx', type=int, default=None, help='ASSUMED mean context per request (for the KV floor)')
    ap.add_argument('--steps-detail', action='store_true')
    a = ap.parse_args()
    r = analyze(a.trace, a.long_ms, a.ctx)
    txt, js = report(r, a.label or os.path.basename(a.trace.rstrip('/')), a.ctx, a.steps_detail)
    print(txt)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(js, f, indent=1)


if __name__ == '__main__':
    main()
