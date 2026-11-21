"""Device-side logic of the verify cap: pure torch, no host syncs, no sglang imports.

Verify block layout (static DSpark verify, request-major): request ``b`` owns rows
``b*S .. b*S+S-1`` where S = gamma + 1. Row 0 is the anchor (last committed token);
row r >= 1 holds draft token d_{r-1}. The target's logits at row r predict the token
after row r, so the greedy accept compares argmax(row r) with d_r for r = 0..S-2.

A live length L (1 <= L <= S) keeps rows 0..L-1 exact and lets the accept check at
most L-1 drafts; the bonus comes from row min(correct, L-1). Rows >= L are "dead".
"""

from __future__ import annotations

from typing import Optional

import torch


def draft_token_probs(
    step_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    *,
    temperatures: Optional[torch.Tensor] = None,
    greedy_mask: Optional[torch.Tensor] = None,
    use_max: bool = False,
    chunk_rows: int = 64,
) -> torch.Tensor:
    """[bs, gamma] probability the draft gave its proposed token (``use_max``: the
    max probability) at each position, from the draft's kept step logits
    [bs, gamma, V]. Sampled rows use their temperature, greedy rows temperature 1.
    Out-of-range token ids are clamped (only affects the cut, never exactness)."""
    bs, gamma, width = step_logits.shape
    rows = bs * gamma
    flat = step_logits.reshape(rows, width)
    tokens = draft_tokens.reshape(rows).to(torch.long).clamp(0, width - 1)
    temp_rows = None
    if temperatures is not None:
        temps = temperatures.reshape(-1)[:bs].to(torch.float32).clamp_min(1e-5)
        if greedy_mask is not None:
            temps = torch.where(
                greedy_mask.reshape(-1)[:bs].to(torch.bool),
                torch.ones_like(temps),
                temps,
            )
        temp_rows = temps.repeat_interleave(gamma)
    out = torch.empty(rows, dtype=torch.float32, device=step_logits.device)
    step = max(1, int(chunk_rows))
    for start in range(0, rows, step):
        end = min(rows, start + step)
        x = flat[start:end].to(torch.float32)
        if temp_rows is not None:
            x = x / temp_rows[start:end, None]
        lse = torch.logsumexp(x, dim=-1)
        if use_max:
            picked = x.amax(dim=-1)
        else:
            picked = x.gather(1, tokens[start:end, None]).squeeze(1)
        torch.exp(picked - lse, out=out[start:end])
    out = torch.nan_to_num(out, nan=0.0).clamp_(0.0, 1.0)
    return out.view(bs, gamma)


def live_lens_from_probs(
    probs: torch.Tensor, *, threshold: float, min_len: int, max_len: int
) -> torch.Tensor:
    """[bs] int64 live lengths. surv_j = prod_{i<=j} p_i estimates P(d_0..d_j all
    accepted); draft j is worth verifying (row j+1 live) while surv_j >= threshold.
    cumprod of values in [0, 1] is non-increasing, so the count of qualifying
    positions is the leading run: L = 1 + count, clamped to [min_len, max_len]."""
    surv = torch.cumprod(probs.to(torch.float32), dim=1)
    count = (surv >= float(threshold)).sum(dim=1)
    return (count + 1).clamp_(int(min_len), int(max_len)).to(torch.int64)


class RowLayout:
    """Precomputed per-row tensors for ``capacity`` requests of ``stride`` rows."""

    def __init__(self, capacity: int, stride: int, device) -> None:
        self.capacity = int(capacity)
        self.stride = int(stride)
        n = self.capacity * self.stride
        self.rows = torch.arange(n, dtype=torch.int64, device=device)
        self.req = torch.div(self.rows, self.stride, rounding_mode="floor")
        self.pos = self.rows - self.req * self.stride
        self.anchor = self.req * self.stride


def build_src_index(
    live_lens: torch.Tensor,
    layout: RowLayout,
    *,
    num_reqs: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Row -> source row for the expert-id remap. Live rows map to themselves, dead
    rows of a real request (L >= 1) to that request's anchor row, rows of padding
    requests (L == 0) to row 0. With ``num_reqs`` only the first num_reqs*stride rows
    are rebuilt (the rest must already be valid, e.g. built once with zero L)."""
    n = layout.capacity * layout.stride if num_reqs is None else num_reqs * layout.stride
    if out is None:
        out = torch.empty(layout.capacity * layout.stride, dtype=torch.int64,
                          device=layout.rows.device)
    req = layout.req[:n]
    lens_rows = live_lens.to(torch.int64).index_select(0, req)
    dead_target = torch.where(lens_rows > 0, layout.anchor[:n], torch.zeros_like(req))
    torch.where(layout.pos[:n] < lens_rows, layout.rows[:n], dead_target, out=out[:n])
    return out


def remap_rows(
    weights: torch.Tensor, ids: torch.Tensor, src_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite every row's routed (weights, ids) with its source row's."""
    m = ids.shape[0]
    idx = src_index[:m]
    return weights.index_select(0, idx), ids.index_select(0, idx)


def cap_correct_len(
    correct_len: torch.Tensor, live_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference of the engine's CapCorrectLen: (capped, trimmed)."""
    ell = live_lens.to(correct_len.dtype) - 1
    capped = torch.clamp(torch.minimum(correct_len, ell), min=0)
    return capped, correct_len - capped


def greedy_accept(
    candidates: torch.Tensor, target_predict: torch.Tensor,
    live_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference greedy accept over [bs, S] candidates and [bs, S] target argmax ids:
    (correct_len, bonus), capped at live_lens - 1 when given."""
    bs, stride = candidates.shape
    match = (candidates[:, 1:] == target_predict[:, : stride - 1]).to(torch.int64)
    correct = match.cumprod(dim=1).sum(dim=1)
    if live_lens is not None:
        correct, _ = cap_correct_len(correct, live_lens)
    bonus = target_predict[torch.arange(bs), correct]
    return correct, bonus


def distinct_experts(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """0-d float tensor: number of distinct experts in ``ids``. Graph-safe (no sync)."""
    present = torch.zeros(num_experts, dtype=torch.float32, device=ids.device)
    flat = ids.reshape(-1).to(torch.int64).clamp(0, num_experts - 1)
    present.index_fill_(0, flat, 1.0)
    return present.sum()
