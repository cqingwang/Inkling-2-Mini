"""Python entry point of the inkling_attn kernel (imported on the GPU path only)."""

from . import build

D_REL = 16
HEAD_DIM = 128


import os

WARPS = int(os.environ.get("INKLING_ATTN_WARPS", "4"))  # 4: BM 64, 2 CTAs / SM; 8: BM 128, 1 CTA / SM
PACKED_PROJ = True  # repack proj per call (one tiny launch) so each lane loads its B fragment with one 8-byte load


def prefill_attention(q, k_cache, v_cache, page_table, cu_seqlens_q, seqused_k, r, proj, softmax_scale,
                      window_left, out=None, packed=None, warps=None):
    """FA4-compatible (out, lse) of the Inkling prefill attention with the relative bias computed in-kernel.

    q [T, H, 128] bf16; k_cache / v_cache [pages, page_size, H_kv, 128] bf16; page_table [B, max_pages] int32;
    cu_seqlens_q [n_req + 1] int32; seqused_k [n_req] int32 (prefix + new tokens); r [T, H, 16] bf16 (after the
    rel L2 clip, the RelLogitsProj input); proj [16, rel_extent] bf16; window_left: the layer's left window
    (sliding-window layers, rel_extent == window_left + 1) or None / negative (global causal layers).
    Returns out [T, H, 128] fp32 and lse [H, T] fp32 (FA4's varlen layout). Rows past cu_seqlens_q[-1] are not
    written.
    """
    import torch

    ext = build.load()
    t, heads, _ = q.shape
    if out is None:
        out = torch.empty((t, heads, HEAD_DIM), dtype=torch.float32, device=q.device)
    lse = torch.empty((heads, t), dtype=torch.float32, device=q.device)
    wl = -1 if window_left is None or window_left < 0 else int(window_left)
    if r.stride(-1) != 1:
        r = r.contiguous()
    packed = PACKED_PROJ if packed is None else packed
    projp = torch.empty(((proj.shape[1] + 1) * D_REL,), dtype=torch.bfloat16, device=q.device) if packed else None
    ext.fwd(q, k_cache, v_cache, page_table, cu_seqlens_q, seqused_k, r, proj, out, lse, float(softmax_scale), wl,
            projp, WARPS if warps is None else int(warps))
    return out, lse
