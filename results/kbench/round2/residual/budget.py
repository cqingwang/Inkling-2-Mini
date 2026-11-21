#!/usr/bin/env python3
"""Residual budget for a native layer-stack backend vs the best round-2 overlay projection (gate 1.20x).

    python3 -I budget.py [out_dir]   (reads out_dir/c{1,4,8,24}.json written by analyze2.py --json)

Every number is labelled: [M] measured in the R2P-s05 traces (analyze2.py), [A] assumption, [P] projection of an
in-flight round-2 overlay lever at its brief's stated gain. Edit ASSUME to test sensitivity.
"""
import json, math, os, sys

H, VOCAB = 2304, 201024
ASSUME = dict(
    wide_gbs=220.0,          # [P] wide packed dense + LM head (> 16 rows), GB/s of packed bytes
    attn_floor_mult=1.2,     # [P] compact verify attention: core+bias+epilogue down to 1.2 x KV-byte floor
    ctx={1: 550, 4: 450, 8: 400, 24: 400},   # [A] mean context per request during the profile (see report)
    hot_vocab=49152,         # [A] hot-vocab size (M0: 48K needed for prose); head+fold time scales with N / V
    hot_accept=0.98,         # [P] acceptance factor of hot vocab (~2 % cost), same on both sides
    policy={1: 0.005, 4: 0.02, 8: 0.05, 24: 0.06},   # [P] batch verify policy (c24 = c16 value, assumed)
    peak_gbs=240.0,          # [A] best sustained DRAM streaming on GB10 (probe 220-244)
    chain_us_per_visit=7.0,  # [M0 probe] graph chaining excess per visit above the 220 GB/s model
    moe_rate_gbs=209.0,      # [M c8] MoE effective rate: union U(8)=68.8 (policy track) x 229.9 MB / 75.8 ms
)
PK = dict(qkvr=0.7738, wo=0.7729, head=0.7720)
DENSE_PACKED_MB = 48 * (8600 * H * 2 * PK['qkvr'] + H * 3072 * 2 * PK['wo']) / 1e6   # 1997 MB
HEAD_PACKED_MB = VOCAB * H * 2 * PK['head'] / 1e6                                    # 715 MB
CTRL_MB, ROUTER_MB, WPROJ_MB, RNN_MB = 113.2, 28.3, 2.4, 55.0


def kv_gb(bs, ctx):
    return bs * (40 * min(ctx, 1024) + 8 * ctx) * 4096 / 1e9


def ms(mb, gbs):
    return mb / gbs   # MB / (GB/s) = ms


def budget(j, c, A):
    C = j['classes']
    g = lambda *pre: sum(v['excl'] for k, v in C.items() if k.startswith(pre))
    T0 = j['period']
    bs = j['bs']
    rows = j['rows']
    out = dict(c=c, T0=T0)
    # ---------------- round-2 overlay projection
    dense_head = g('dense.qkvr', 'dense.wo', 'lmhead')
    if rows > 16:
        wide_to = ms(DENSE_PACKED_MB + HEAD_PACKED_MB, A['wide_gbs'])
        s_wide = max(0.0, dense_head - wide_to)
    else:
        wide_to, s_wide = dense_head, 0.0
    attn_core = g('attn.swa.core', 'attn.swa.bias', 'attn.swa.epi', 'attn.glob.core', 'attn.glob.bias', 'attn.glob.epi')
    kv = kv_gb(bs, A['ctx'][c])
    attn_to = min(attn_core, A['attn_floor_mult'] * ms(kv * 1e3, A['peak_gbs']))
    s_attn = attn_core - attn_to
    vocab = g('draft.head', 'draft.fold')
    vocab_to = vocab * A['hot_vocab'] / VOCAB
    s_hot = vocab - vocab_to
    T_ov = T0 - s_wide - s_attn - s_hot
    out.update(dense_head=dense_head, wide_to=wide_to, s_wide=s_wide, attn_core=attn_core, attn_to=attn_to,
               kv_gb=kv, s_attn=s_attn, vocab=vocab, vocab_to=vocab_to, s_hot=s_hot, T_ov=T_ov,
               thr_ov=T0 / T_ov * (1 + A['policy'][c]) * A['hot_accept'])
    # ---------------- native-only residual, tier 1: cross-op fusion inside the 48-visit core
    t1 = {
        'norms': g('small.norm'),
        'attn-res mix': g('small.attn_res'),
        'short conv': g('small.sconv'),
        'controllers (above 113 MB weight floor)': max(0.0, g('small.ctrl') - ms(CTRL_MB, A['peak_gbs'])),
        'MoE glue (route/remap/quant/args/combine)': g('moe.glue'),
        'router + conv wproj (above weight floor)': max(0.0, g('dense.router', 'dense.conv_wproj')
                                                        - ms(ROUTER_MB + WPROJ_MB, A['peak_gbs'])),
        'attention prologue + KV store (above KV-write floor)': max(0.0, g('attn.swa.prologue', 'attn.glob.prologue')
                                                                     - ms(rows * 48 * 4096 / 1e6, A['peak_gbs'])),
        'layer-stack copies/act glue': g('fwd.glue'),
        'idle inside graphs': j['timeline']['idle_draft'] + j['timeline']['idle_verify'],
    }
    # tier 2: everything else a backend could plausibly absorb or that is not a weight/KV stream
    t2 = {
        'visit chaining (M0 probe 7 us/visit)': 48 * A['chain_us_per_visit'] / 1e3,
        'logits softcap+argmax (head epilogue)': g('logits.'),
        'verify prep + accept/commit (graph)': g('verify.'),
        'draft ctx K/V (above weight floor)': max(0.0, g('draft.ctxkv') - ms(14.6, A['peak_gbs'])),
        'draft small kernels (norms/attn/rnn/sample/prep)': g('draft.layers.misc', 'draft.layers.attn', 'draft.prep',
                                                              'draft.misc', 'draft.sample')
                                                            + max(0.0, g('draft.rnn') - ms(RNN_MB, A['peak_gbs'])),
        'eager kernels + memcpy between graphs': g('eager.', 'mem.'),
        'idle between graphs': j['idle'] - j['timeline']['idle_draft'] - j['timeline']['idle_verify'],
    }
    # tier 3 (NOT native-specific; kernel-rate ceiling): all streams at peak_gbs
    moe = g('moe.nvfp4', 'moe.mxfp8')
    t3 = {
        f"MoE grouped GEMM {A['moe_rate_gbs']:.0f} -> {A['peak_gbs']:.0f} GB/s": moe * (1 - A['moe_rate_gbs'] / A['peak_gbs']),
        'dense + head at peak instead of the wide/pico rate': (wide_to if rows > 16 else dense_head) - ms(
            DENSE_PACKED_MB + HEAD_PACKED_MB + (0.0 if rows > 16 else 28 * 9.437 * PK['qkvr']), A['peak_gbs']),
        'attention at 1.0x KV floor': attn_to - ms(kv * 1e3, A['peak_gbs']),
        'draft layers + hot vocab at peak': max(0.0, g('draft.layers.gemm') - ms(255.5 if rows <= 64 else 495.0, A['peak_gbs']))
                                            + max(0.0, vocab_to - ms((463.9 + 7 * 52.3) * A['hot_vocab'] / VOCAB, A['peak_gbs'])),
    }
    R1, R2, R3 = sum(t1.values()), sum(t2.values()), sum(t3.values())
    out.update(t1=t1, t2=t2, t3=t3, R1=R1, R2=R2, R3=R3,
               S1=T_ov / (T_ov - R1), S2=T_ov / (T_ov - R1 - R2), S3=T_ov / (T_ov - R1 - R2 - R3))
    return out


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    J = {c: json.load(open(os.path.join(d, f'c{c}.json'))) for c in (1, 4, 8, 24)}
    B = {c: budget(J[c], c, ASSUME) for c in (1, 4, 8, 24)}
    p = print
    p("## Measured per-class time (excl ms per step; analyze2.py groups) [M]")
    p("| group | c1 | c4 | c8 | c24 |")
    p("|---|---:|---:|---:|---:|")
    for gname in J[1]['groups']:
        p(f"| {gname} | " + " | ".join(f"{J[c]['groups'][gname]['excl']:.2f}" for c in J) + " |")
    p("| GPU idle | " + " | ".join(f"{J[c]['idle']:.2f}" for c in J) + " |")
    p("| **wall period** | " + " | ".join(f"**{J[c]['period']:.2f}**" for c in J) + " |")
    p("")
    p("## Round-2 overlay projection (ms per step)")
    p("| | c1 | c4 | c8 | c24 |")
    p("|---|---:|---:|---:|---:|")
    row = lambda name, key, f='{:.2f}': p(f"| {name} | " + " | ".join(f.format(B[c][key]) for c in B) + " |")
    row('measured step T0 [M]', 'T0')
    row('dense+head measured [M]', 'dense_head')
    row('  -> wide packed @220 [P] (c1: pico, unchanged)', 'wide_to')
    row('attention core+bias+epi measured [M]', 'attn_core')
    row('  KV floor GB at assumed ctx [A]', 'kv_gb', '{:.2f}')
    row('  -> compact verify attention 1.2x floor [P]', 'attn_to')
    row('draft vocab (FP8 head + fold) measured [M]', 'vocab')
    row('  -> hot vocab 48K [P]', 'vocab_to')
    row('**best round-2 overlay step T_ov**', 'T_ov')
    row('overlay throughput vs measured (incl. policy, hot-vocab acceptance)', 'thr_ov', '{:.3f}x')
    for tier, key in (('tier 1: cross-op fusion inside the 48-visit core', 't1'),
                      ('tier 2: + launches/glue outside the core (generous)', 't2'),
                      ('tier 3: + every stream at 240 GB/s (NOT native-specific)', 't3')):
        p("")
        p(f"### {tier} (ms per step removed)")
        p("| item | c1 | c4 | c8 | c24 |")
        p("|---|---:|---:|---:|---:|")
        for item in B[1][key]:
            p(f"| {item} | " + " | ".join(f"{B[c][key][item]:.2f}" for c in B) + " |")
        R = 'R' + key[1]
        p(f"| **sum** | " + " | ".join(f"**{B[c][R]:.2f}**" for c in B) + " |")
    p("")
    p("## Best-case native speedup vs the best round-2 overlay (same policy, acceptance, context)")
    p("| | c1 | c4 | c8 | c24 | geomean c4/c8/c24 |")
    p("|---|---:|---:|---:|---:|---:|")
    for name, S in (('tier 1 removed', 'S1'), ('tiers 1+2 removed (gate input)', 'S2'), ('tiers 1+2+3 (ceiling)', 'S3')):
        p(f"| {name} | " + " | ".join(f"{B[c][S]:.3f}" for c in B) + f" | {gmean([B[c][S] for c in (4, 8, 24)]):.3f} |")
    g2 = gmean([B[c]['S2'] for c in (4, 8, 24)])
    p("")
    p(f"GATE: geomean c4/c8/c24 with all boundary-specific overhead removed = {g2:.3f}x "
      f"-> {'KILL (< 1.20x)' if g2 < 1.20 else 'PASS: prototype one core block'}")
    js = os.path.join(d, 'budget.json')
    with open(js, 'w') as f:
        json.dump({c: {k: v for k, v in b.items()} for c, b in B.items()}, f, indent=1)


if __name__ == '__main__':
    main()
