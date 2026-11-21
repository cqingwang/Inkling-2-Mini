"""CPU stand-in with the patched classes' names and signatures: a toy float64 MoE."""

import torch
from torch import nn


class Inkling2Gate(nn.Module):
    def __init__(self, hidden, experts, topk, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.topk = topk
        self.weight = nn.Parameter(
            torch.randn(experts, hidden, generator=g, dtype=torch.float64),
            requires_grad=False,
        )

    def forward(self, x):
        logits = x @ self.weight.T
        top, ids = torch.topk(logits, self.topk, dim=-1)
        return torch.softmax(top, dim=-1), ids.to(torch.int32)


class _ToyExperts(nn.Module):
    def __init__(self, hidden, experts, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 1)
        self.w = nn.Parameter(
            torch.randn(experts, hidden, hidden, generator=g, dtype=torch.float64)
            / hidden**0.5,
            requires_grad=False,
        )
        self.touched = []  # experts read per call (the "union")

    def forward(self, x, weights, ids, record=False):
        if record:
            self.touched.append(sorted(set(ids.reshape(-1).tolist())))
        out = torch.zeros_like(x)
        for k in range(ids.shape[1]):
            w = self.w[ids[:, k].long()]  # [M, H, H]
            out += weights[:, k : k + 1] * torch.tanh(torch.einsum("mh,mhj->mj", x, w))
        return out


class Inkling2MoE(nn.Module):
    def __init__(self, hidden=16, experts=16, topk=2, seed=0):
        super().__init__()
        self.gate = Inkling2Gate(hidden, experts, topk, seed)
        self.experts = _ToyExperts(hidden, experts, seed)

    def forward(self, x, forward_batch=None, combine=True):
        weights, ids = self.gate(x)
        mode = getattr(forward_batch, "forward_mode", None)
        out = self.experts(x, weights, ids, record=bool(mode and mode.is_target_verify()))
        return out
