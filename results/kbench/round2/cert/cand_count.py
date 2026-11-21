#!/usr/bin/env python3
"""Candidate-count measurement for the certified greedy LM head (CPU, numpy).

For real LM-head input rows (hid/*.npz from dump_hidden.py) and the real unembedding, and for several screen
formats, computes the rigorous interval [s - B, s + B] of every logit, the threshold T = max lo (real vocab only),
the candidates hi >= T - delta, and the cost unit of the exact pass: pico CTA tiles of 16 vocab rows, unioned over
the 8 consecutive rows of a verify step (positions p..p+7 of one transcript; c1 shape).

Bound: B = sum_g ||R_vg|| ||x_g|| + 2 C_ACC max(||W_v||, ||D_v||) ||x||, C_ACC = (K/16) * 17 * 2^-23.
"""
import glob
import json
import os
import sys
import time

import numpy as np
from safetensors import safe_open

W_PATH = os.environ.get("INKLING_WEIGHTS", "model-00001-of-00001.safetensors")
V_REAL = 200058
K = 2304
TILE = 16
C_ACC = (K / 16) * 17 * 2.0 ** -23


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def load_w():
    with safe_open(W_PATH, "np") as f:
        try:
            w = f.get_tensor("model.llm.unembed.weight")
        except Exception:
            raise
    return w


def load_w_bf16():
    # safetensors numpy cannot hold bf16: read raw bytes
    import json as _j
    import struct
    with open(W_PATH, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = _j.loads(fh.read(n))
        meta = hdr["model.llm.unembed.weight"]
        a, b = meta["data_offsets"]
        assert meta["dtype"] == "BF16", meta["dtype"]
        fh.seek(8 + n + a)
        raw = np.frombuffer(fh.read(b - a), dtype=np.uint16).reshape(meta["shape"])
    return bf16_to_f32(raw)


# ---------------- screen formats: return D (float32, as the screen kernel would use) ----------------
def q_int(W, bits, group, scale_kind):
    """Symmetric integer quantisation, scale per (row, group). scale_kind: 'pow2' (e8m0, MX style), 'f16', 'bf16'."""
    V, Kk = W.shape
    qmax = 2 ** (bits - 1) - 1
    Wg = W.reshape(V, Kk // group, group)
    amax = np.abs(Wg).max(-1)
    if scale_kind == "pow2":
        e = np.ceil(np.log2(np.maximum(amax, 1e-30) / qmax))
        s = np.exp2(e)
    elif scale_kind == "f16":
        s = (amax / qmax).astype(np.float16).astype(np.float32)
        s = np.where(s * qmax < amax, np.nextafter(s.astype(np.float16), np.float16(np.inf)).astype(np.float32), s)
    else:
        s = (amax / qmax).astype(np.float32)
    s = np.where(amax > 0, s, 1.0).astype(np.float32)
    q = np.clip(np.rint(Wg / s[..., None]), -qmax, qmax)
    return (q * s[..., None]).reshape(V, Kk).astype(np.float32)


FORMATS = {
    # name: (fn, bytes per weight incl. scales)
    "int8_row": (lambda W: q_int(W, 8, K, "f16"), 1.0 + 2 / K),
    "mxint8_g32": (lambda W: q_int(W, 8, 32, "pow2"), 1.0 + 1 / 32),
    "int8_g128": (lambda W: q_int(W, 8, 128, "f16"), 1.0 + 2 / 128),
    "int7_g64": (lambda W: q_int(W, 7, 64, "f16"), 7 / 8 + 2 / 64),
    "int6_g32": (lambda W: q_int(W, 6, 32, "f16"), 6 / 8 + 2 / 32),
    "int5_g32": (lambda W: q_int(W, 5, 32, "f16"), 5 / 8 + 2 / 32),
    "int4_g32": (lambda W: q_int(W, 4, 32, "f16"), 0.5 + 2 / 32),
    "int6_row": (lambda W: q_int(W, 6, K, "f16"), 6 / 8 + 2 / K),
    "int6_g192": (lambda W: q_int(W, 6, 192, "f16"), 6 / 8 + 2 / 192),
    "int6_g64p2": (lambda W: q_int(W, 6, 64, "pow2"), 6 / 8 + 1 / 64),
    "int5_g192": (lambda W: q_int(W, 5, 192, "f16"), 5 / 8 + 2 / 192),
    "int5_g32": (lambda W: q_int(W, 5, 32, "f16"), 5 / 8 + 2 / 32),
    "int7_g192": (lambda W: q_int(W, 7, 192, "f16"), 7 / 8 + 2 / 192),
}


def residual_norms(W, D, G, chunk=8192):
    V = W.shape[0]
    out = np.empty((V, K // G), np.float32)
    nv = np.empty(V, np.float32)
    for a in range(0, V, chunk):
        Wc = W[a:a + chunk].astype(np.float64)
        Dc = D[a:a + chunk].astype(np.float64)
        R = Wc - Dc
        out[a:a + chunk] = np.sqrt((R.reshape(-1, K // G, G) ** 2).sum(-1)) * (1 + 2 ** -20)
        nv[a:a + chunk] = np.maximum(np.linalg.norm(Wc, axis=1), np.linalg.norm(Dc, axis=1)) * (1 + 2 ** -20)
    return out, nv


def main():
    hid_dir = sys.argv[1]
    fmts = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FORMATS)
    groups = [int(g) for g in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["32", "128"])]
    deltas = (0.0, 0.25)
    files = sorted(glob.glob(os.path.join(hid_dir, "*.npz")))
    X = []
    seg = []
    for f in files:
        z = np.load(f)
        h = bf16_to_f32(z["hidden"])
        X.append(h)
        seg.append((os.path.basename(f)[:-4], h.shape[0]))
    X = np.concatenate(X).astype(np.float32)
    M = X.shape[0]
    print(f"rows {M} from {len(files)} transcripts", flush=True)
    t0 = time.time()
    W = load_w_bf16()
    print(f"W {W.shape} loaded {time.time() - t0:.0f}s; row norm median {np.median(np.linalg.norm(W[:V_REAL], axis=1)):.3f}", flush=True)
    xn = np.linalg.norm(X.astype(np.float64), axis=1)
    print(f"||x|| median {np.median(xn):.2f} p10 {np.percentile(xn, 10):.2f} p90 {np.percentile(xn, 90):.2f}")
    # exact logits (fp32 GEMM is far below the bound resolution) and per-row max over the real vocab
    CH = 256
    L = np.empty((M, W.shape[0]), np.float32) if M * W.shape[0] * 4 < 12e9 else None
    assert L is not None, "too many rows for in-memory logits"
    for a in range(0, M, CH):
        L[a:a + CH] = X[a:a + CH] @ W.T
    lmax = L[:, :V_REAL].max(1)
    top2 = np.partition(L[:, :V_REAL], -2, axis=1)[:, -2]
    print(f"logits: max median {np.median(lmax):.2f}, top1-top2 gap median {np.median(lmax - top2):.3f}; "
          f"#(l >= max-1) median {np.median((L[:, :V_REAL] >= lmax[:, None] - 1).sum(1)):.0f}, "
          f"#(l >= max-3) median {np.median((L[:, :V_REAL] >= lmax[:, None] - 3).sum(1)):.0f}", flush=True)
    res = {}
    for fname in fmts:
        fn, bpw = FORMATS[fname]
        t0 = time.time()
        D = fn(W)
        S = np.empty_like(L)
        for a in range(0, M, CH):
            S[a:a + CH] = X[a:a + CH] @ D.T
        err = np.abs(S[:, :V_REAL] - L[:, :V_REAL])
        for G in groups:
            Rg, Nv = residual_norms(W, D, G)
            xg = np.sqrt((X.astype(np.float64).reshape(M, K // G, G) ** 2).sum(-1))
            # B in chunks (M x V)
            cand_rows = {d: np.zeros(M, np.int64) for d in deltas}
            tiles_step = {d: [] for d in deltas}
            row_tiles = {d: [] for d in deltas}
            Bmed = []
            ratio = []
            for a in range(0, M, CH):
                b = min(M, a + CH)
                B = (xg[a:b] @ Rg.T.astype(np.float64) + 2 * C_ACC * xn[a:b, None] * Nv[None, :].astype(np.float64))
                B = (B * (1 + 2 ** -18)).astype(np.float32)
                lo = S[a:b] - B
                hi = S[a:b] + B
                T = lo[:, :V_REAL].max(1)
                Bmed.append(np.median(B[:, :V_REAL], axis=1))
                ratio.append(np.median(err[a:b] / B[:, :V_REAL], axis=1))
                assert np.all(hi[np.arange(b - a), L[a:b, :V_REAL].argmax(1)] >= T), "certificate violated"
                for d in deltas:
                    c = hi >= (T - d)[:, None]
                    c[:, V_REAL:] = False
                    cand_rows[d][a:b] = c.sum(1)
                    row_tiles[d].append(c.reshape(b - a, -1, TILE).any(-1))
            for d in deltas:
                rt = np.concatenate(row_tiles[d])  # [M, ntiles]
                # steps: 8 consecutive rows within one transcript
                steps = []
                p = 0
                for _, n in seg:
                    for s0 in range(p, p + n - 7, 8):
                        steps.append(int(rt[s0:s0 + 8].any(0).sum()))
                    p += n
                steps = np.array(steps)
                # c2: two requests per step (16 rows): pair step windows of consecutive transcripts
                wins = []
                p = 0
                for _, n in seg:
                    wins.append([rt[s0:s0 + 8].any(0) for s0 in range(p, p + n - 7, 8)])
                    p += n
                c2 = []
                for a_ in range(0, len(wins) - 1, 2):
                    for w1, w2 in zip(wins[a_], wins[a_ + 1]):
                        c2.append(int((w1 | w2).sum()))
                c2 = np.array(c2) if c2 else np.zeros(1)
                cr = cand_rows[d]
                screen_mb = W.shape[0] * K * bpw / 1e6 + W.shape[0] * (K // G) * 2 / 1e6 + W.shape[0] * 4 / 1e6
                exact_mb = steps * TILE * K * 1.55 / 1e6
                key = f"{fname} G={G} delta={d}"
                res[key] = dict(
                    bpw=bpw, screen_mb=round(screen_mb, 1),
                    cand_per_row=dict(p50=float(np.median(cr)), p90=float(np.percentile(cr, 90)),
                                      p99=float(np.percentile(cr, 99)), max=int(cr.max())),
                    tiles_per_step=dict(mean=float(steps.mean()), p50=float(np.median(steps)),
                                        p90=float(np.percentile(steps, 90)), p99=float(np.percentile(steps, 99)),
                                        max=int(steps.max())),
                    exp_ms=round(screen_mb / 235.0 + float(exact_mb.mean()) / 200.0 + 0.05, 3),
                    exp_ms_p90=round(screen_mb / 235.0 + float(np.percentile(exact_mb, 90)) / 200.0 + 0.05, 3),
                    tiles_per_step_c2=dict(mean=float(c2.mean()), p90=float(np.percentile(c2, 90)),
                                           p99=float(np.percentile(c2, 99)), max=int(c2.max())),
                    B_median=float(np.median(np.concatenate(Bmed))),
                    err_over_B_median=float(np.median(np.concatenate(ratio))),
                    err_max=float(err.max()),
                )
                print(key, json.dumps(res[key]), flush=True)
        print(f"  {fname} done {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(hid_dir, "..", "cand_count.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
