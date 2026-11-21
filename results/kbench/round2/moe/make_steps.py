#!/usr/bin/env python3
"""Build verify-step routings for the tile bench from the oracle's real top-8 ids (routes.npz).

A verify step of B requests has B*8 rows: request r contributes 8 consecutive positions of one
prompt's greedy text, at one MoE visit. "full" = every row live (stock, and the shipped stack
above bsmax=16); "cap" = the shipped verify cap: rows >= L of a request take row 0's ids (dead-row
remap), L drawn from LCAP. Prints union / per-expert row statistics (to compare with the steplog
unions: bs 8 U(1)/U(8) = 32.8/68.8, bs 16 51.4/91.2) and writes steps.npz:
  ids_<kind>_<bs>_<mode>  int32 [S, B*8, 8]   mode full | cap | fullc | capc ("c": the B requests come
  from only B//3 prompts, several windows each, to reach the lower serving unions of the steplog)   kind = nvfp4 (visits of layers 0-19) | mxfp8 (20-23)
"""

import argparse

import numpy as np

LCAP = (2, 3, 4, 5, 6, 7, 8)  # cap mode: L uniform over these (mean 5)


def load(path):
    z = np.load(path)
    lov = z["layer_of_visit"]
    reqs = []
    for k in z.files:
        if k.startswith("ids_"):
            pid = k[4:]
            reqs.append((pid, z[k].astype(np.int32), int(z["plen_" + pid])))
    return lov, reqs


def windows(reqs, bs, rng, min_gap=48, nprompts=None):
    """bs (prompt, start) windows of 8 completion positions; distinct prompts first (at most
    nprompts distinct prompts: a 'concentrated' batch, several windows of one text)."""
    order = rng.permutation(len(reqs))[: nprompts or len(reqs)]
    out, used = [], {}
    i = 0
    while len(out) < bs:
        p = order[i % len(order)]
        pid, arr, plen = reqs[p]
        lo, hi = plen, arr.shape[1] - 8
        for _ in range(50):
            s = int(rng.integers(lo, hi + 1))
            if all(abs(s - t) >= min_gap for t in used.get(p, [])):
                break
        else:
            i += 1
            continue
        used.setdefault(p, []).append(s)
        out.append((p, s))
        i += 1
    return out


def step(reqs, wins, visit, cap, rng):
    rows = []
    for p, s in wins:
        blk = reqs[p][1][visit, s:s + 8].copy()  # [8, 8]
        if cap:
            L = int(rng.choice(LCAP))
            blk[L:] = blk[0]
        rows.append(blk)
    return np.concatenate(rows, 0)


def stats(ids):
    flat = ids.reshape(-1)
    cnt = np.bincount(flat, minlength=128)
    used = cnt[cnt > 0]
    return len(used), int(used.max()), float(np.mean(used))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="routes.npz")
    ap.add_argument("--out", default="steps.npz")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--bs", default="4,8,16,24")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    lov, reqs = load(a.routes)
    print(f"{len(reqs)} prompts, visits {len(lov)}")
    nv_visits = [v for v in range(len(lov)) if lov[v] < 20]
    mx_visits = [v for v in range(len(lov)) if lov[v] >= 20]
    out = {}
    # steplog-comparable unions: mean over visits of U(1) and U(8), uncapped
    for bs in (8, 16):
        u1, u8 = [], []
        for _ in range(40):
            wins = windows(reqs, bs, rng)
            for v in range(len(lov)):
                ids = step(reqs, wins, v, False, rng)
                u8.append(stats(ids)[0])
                u1.append(stats(ids.reshape(bs, 8, 8)[:, 0])[0])
        print(f"oracle routing bs {bs}: U(1) {np.mean(u1):.1f}  U(8) {np.mean(u8):.1f}   (steplog: "
              f"{'32.8/68.8' if bs == 8 else '51.4/91.2'})")
    for kind, visits in (("nvfp4", nv_visits), ("mxfp8", mx_visits)):
        for bs in [int(b) for b in a.bs.split(",")]:
            for mode in ("full", "cap", "fullc", "capc"):
                arr = []
                for i in range(a.steps):
                    wins = windows(reqs, bs, rng, min_gap=(16 if mode.endswith("c") else 48), nprompts=(max(2, bs // 3) if mode.endswith("c") else None))
                    v = visits[int(rng.integers(len(visits)))]
                    arr.append(step(reqs, wins, v, mode.startswith("cap"), rng))
                arr = np.stack(arr)
                out[f"ids_{kind}_{bs}_{mode}"] = arr.astype(np.int32)
                st = np.array([stats(x) for x in arr])
                print(f"{kind} bs {bs:2d} rows {bs * 8:3d} {mode}: union {st[:, 0].mean():5.1f} "
                      f"[{st[:, 0].min()}-{st[:, 0].max()}]  max rows/expert {st[:, 1].mean():5.1f} "
                      f"[{st[:, 1].min()}-{st[:, 1].max()}]  mean rows/used expert {st[:, 2].mean():4.1f}")
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
