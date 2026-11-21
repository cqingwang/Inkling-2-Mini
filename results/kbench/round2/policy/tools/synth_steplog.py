#!/usr/bin/env python3
"""Synthetic multi-batch-size steplog (.npz, steplog column format) for self-tests of
bsreplay.py: informative m (accept_j ~ Bernoulli(m_j^1.5)), per-request routing with
in-request correlation and a skewed expert popularity, union_k computed exactly."""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--bs", default="1,2,4,8")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--indep", action="store_true", help="m carries no information")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
os.makedirs(a.out, exist_ok=True)
NE, TOPK, VIS = 128, 8, 6
pop = rng.dirichlet(np.full(NE, 0.6), size=VIS)  # per-visit expert popularity
rows = []
step = 0
rid_ct = 0
for bs in [int(x) for x in a.bs.split(",")]:
    rids = [f"r{rid_ct + i}" for i in range(bs)]
    rid_ct += bs
    pref = [rng.dirichlet(np.full(NE, 0.3)) for _ in range(bs)]
    for s in range(a.steps):
        if s % 150 == 149:  # replace one request
            i = rng.integers(bs)
            rids[i] = f"r{rid_ct}"; rid_ct += 1
            pref[i] = rng.dirichlet(np.full(NE, 0.3))
        ms, accs = [], []
        for r in range(bs):
            hi = rng.random(7) < 0.6
            m = np.where(hi, rng.beta(12, 1.2, 7), rng.beta(2, 3, 7))
            pa = (rng.uniform(0.2, 1, 7) if a.indep else m ** 1.5)
            ok = rng.random(7) < pa
            accs.append(int(np.argmin(ok)) if not ok.all() else 7)
            ms.append(m)
        uk = np.zeros(8)
        for v in range(VIS):
            sets = []
            for r in range(bs):
                p = 0.5 * pop[v] + 0.5 * pref[r]
                sets.append([set(rng.choice(NE, TOPK, replace=False, p=p)) for _ in range(8)])
            for k in range(8):
                u = set()
                for r in range(bs):
                    for j in range(k + 1):
                        u |= sets[r][j]
                uk[k] += len(u) / VIS
        for r in range(bs):
            rows.append(dict(step=step, row=r, bs=bs, rid=rids[r], m=ms[r], acc=accs[r],
                             greedy=(r % 2 == 0), union=uk))
        step += 1
n = len(rows)
z = dict(
    step=np.asarray([x["step"] for x in rows], np.int64),
    row=np.asarray([x["row"] for x in rows], np.int16),
    bs=np.asarray([x["bs"] for x in rows], np.int16),
    rid=np.asarray([x["rid"] for x in rows]),
    m=np.asarray([x["m"] for x in rows], np.float32),
    qd=np.asarray([x["m"] for x in rows], np.float32),
    draft=np.zeros((n, 7), np.int32),
    accept=np.asarray([x["acc"] for x in rows], np.int16),
    uncapped=np.asarray([x["acc"] for x in rows], np.int16),
    live=np.full(n, 8, np.int16),
    greedy=np.asarray([x["greedy"] for x in rows]),
    argmax_ok=np.ones(n, bool),
    temp=np.asarray([0.0 if x["greedy"] else 0.8 for x in rows], np.float32),
    union_k=np.asarray([x["union"] for x in rows], np.float32),
    seq_len=np.zeros(n, np.int32), prompt_len=np.zeros(n, np.int32),
    forward_ct=np.zeros(n, np.int64))
np.savez(os.path.join(a.out, "steplog-synth-1-000000.npz"), **z)
print("rows", n)
