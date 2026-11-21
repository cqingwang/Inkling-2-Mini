"""Triton kernels (CUDA only; imported lazily by ops.py).

w8_mm:    Y[M, N] = X[M, K] @ dequant(W)[N, K]^T (+ B), FP32 accumulate. W is FP8
          e4m3 with fp32 per-group scales or e8m0 MX scales, or plain BF16.
          FP8 -> BF16 is exact; e8m0 scales are powers of two, so the MX dequant
          in BF16 is exact too; fp32 scales multiply the FP32 partial of each
          K tile (tiles never straddle a scale group).
hot_step: one draft step over the hot vocabulary: the Markov fold on the hot
          rows, the same BF16 roundings as the stock MarkovSampleStep
          (bias -> bf16, base + bias -> bf16, softcap -> bf16, vocab limit),
          scattered to the hot columns of a full-width row buffer.

Weight-streaming layout: one CTA per BN output rows (W rows) and MP input rows;
the W tile is loaded [BK, BN] (K contiguous in memory). M <= MP is padded by
masking; larger M uses a second grid dimension.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .quant import SCALE_FP32


@triton.jit
def _acc_tile(
    X,
    W,
    S,
    row0,
    M,
    n,
    nmask,
    SX,
    K: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    W_FP8: tl.constexpr,
):
    rows = row0 + tl.arange(0, MP)
    rmask = rows < M
    NG: tl.constexpr = K // G
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < K
        w = tl.load(
            W + n[None, :].to(tl.int64) * K + ks[:, None],
            nmask[None, :] & kmask[:, None],
            other=0,
        )
        if W_FP8:
            # FP8 planes arrive as uint8 (a masked load cannot fill fp8 with
            # an int constant); reinterpret the bits after the load.
            w = w.to(tl.float8e4nv, bitcast=True)
        x = tl.load(
            X + rows[:, None].to(tl.int64) * SX + ks[None, :],
            rmask[:, None] & kmask[None, :],
            other=0,
        ).to(tl.bfloat16)
        if SCALE == 2:
            e = tl.load(
                S + n[None, :].to(tl.int64) * NG + ks[:, None] // G,
                nmask[None, :] & kmask[:, None],
                other=127,
            )
            wf = w.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)
            acc += tl.dot(x, wf.to(tl.bfloat16))
        elif SCALE == 1:
            s = tl.load(S + n.to(tl.int64) * NG + k0 // G, nmask, other=0)
            acc += tl.dot(x, w.to(tl.bfloat16)) * s[None, :]
        else:
            acc += tl.dot(x, w.to(tl.bfloat16))
    return acc, rows, rmask


@triton.jit
def _w8_mm_kernel(
    X,
    W,
    S,
    B,
    Y,
    M,
    N,
    SX,
    SY,
    K: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    W_FP8: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = n < N
    acc, rows, rmask = _acc_tile(
        X, W, S, tl.program_id(1) * MP, M, n, nmask, SX, K, G, BK, MP, BN, SCALE, W_FP8
    )
    if HAS_BIAS:
        acc += tl.load(B + n, nmask, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None].to(tl.int64) * SY + n[None, :],
        acc.to(Y.dtype.element_ty),
        rmask[:, None] & nmask[None, :],
    )


@triton.jit
def _hot_step_kernel(
    X,
    W,
    S,
    FB,
    BASE,
    IDS,
    OUT,
    M,
    N,
    SX,
    SBASE,
    SOUT,
    softcap,
    vocab_limit,
    K: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    LIMIT: tl.constexpr,
    W_FP8: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = n < N
    acc, rows, rmask = _acc_tile(
        X, W, S, tl.program_id(1) * MP, M, n, nmask, SX, K, G, BK, MP, BN, SCALE, W_FP8
    )
    acc += tl.load(FB + n, nmask, other=0).to(tl.float32)[None, :]
    # The stock fold (torch.addmm) returns a BF16 bias.
    bias = acc.to(tl.bfloat16).to(tl.float32)
    ids = tl.load(IDS + n, nmask, other=0)
    mask = rmask[:, None] & nmask[None, :]
    base = tl.load(
        BASE + rows[:, None].to(tl.int64) * SBASE + ids[None, :], mask, other=0
    ).to(tl.float32)
    value = (base + bias).to(tl.bfloat16).to(tl.float32)
    if SOFTCAP:
        value = (libdevice.tanh(value / softcap) * softcap).to(tl.bfloat16).to(
            tl.float32
        )
    if LIMIT:
        value = tl.where(ids[None, :] < vocab_limit, value, -float("inf"))
    tl.store(
        OUT + rows[:, None].to(tl.int64) * SOUT + ids[None, :],
        value.to(OUT.dtype.element_ty),
        mask,
    )


def _mp(m: int) -> int:
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    return 64


def _bk(pack, cfg_bk: int) -> int:
    if pack.scale_kind != SCALE_FP32:
        return min(cfg_bk, triton.next_power_of_2(pack.k))
    bk = cfg_bk
    while bk >= 16 and pack.group % bk:
        bk //= 2
    if bk < 16:
        raise ValueError(f"no K tile >= 16 divides the scale group {pack.group}")
    return bk


def _scale_arg(pack, like):
    return pack.scale if pack.scale is not None else like


def _w_arg(pack):
    """(weight pointer, is_fp8): FP8 planes are passed as uint8 views."""
    if pack.q.dtype == torch.float8_e4m3fn:
        return pack.q.view(torch.uint8), True
    return pack.q, False


def w8_mm(x, pack, bias, out, *, bn: int, bk: int, num_warps: int):
    m, k = x.shape
    assert k == pack.k and x.stride(1) == 1 and out.stride(1) == 1
    mp = _mp(m)
    grid = (triton.cdiv(pack.n, bn), triton.cdiv(m, mp))
    w, w_fp8 = _w_arg(pack)
    _w8_mm_kernel[grid](
        x,
        w,
        _scale_arg(pack, w),
        bias if bias is not None else x,
        out,
        m,
        pack.n,
        x.stride(0),
        out.stride(0),
        K=pack.k,
        G=pack.group,
        BK=_bk(pack, bk),
        MP=mp,
        BN=bn,
        SCALE=pack.scale_kind,
        HAS_BIAS=bias is not None,
        W_FP8=w_fp8,
        num_warps=num_warps,
    )
    return out


def hot_step(
    latent,
    pack,
    fold_bias,
    base,
    ids32,
    out,
    *,
    softcap,
    vocab_limit,
    bn: int,
    bk: int,
    num_warps: int,
):
    m, k = latent.shape
    assert k == pack.k and latent.stride(1) == 1
    assert base.stride(1) == 1 and out.stride(1) == 1
    mp = _mp(m)
    grid = (triton.cdiv(pack.n, bn), triton.cdiv(m, mp))
    w, w_fp8 = _w_arg(pack)
    _hot_step_kernel[grid](
        latent,
        w,
        _scale_arg(pack, w),
        fold_bias,
        base,
        ids32,
        out,
        m,
        pack.n,
        latent.stride(0),
        base.stride(0),
        out.stride(0),
        float(softcap or 0.0),
        int(vocab_limit or 0),
        K=pack.k,
        G=pack.group,
        BK=_bk(pack, bk),
        MP=mp,
        BN=bn,
        SCALE=pack.scale_kind,
        SOFTCAP=softcap is not None,
        LIMIT=vocab_limit is not None,
        W_FP8=w_fp8,
        num_warps=num_warps,
    )
    return out

