"""Stateless building blocks: the SwiGLU MLP, grouping routed rows, and one grouped GEMM."""

import torch
import torch.nn.functional as F

from .type_hints import (
    Group,
    GroupEnds,
    GroupedOut,
    GroupedRows,
    InWeight,
    OutWeight,
    Rows,
    SlotExperts,
    SlotWeights,
    StackedWeight,
    Tokens,
)


def mlp(x: Tokens, gate: InWeight, up: InWeight, down: OutWeight) -> Tokens:
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def groups(e: SlotExperts, r: SlotWeights) -> list[Group]:
    """One routing slot -> [(delta index or -1, token rows, their weights)] for the experts it uses."""
    out: list[Group] = []
    for k in e.unique().tolist():
        rows = (e == k).nonzero().squeeze(-1)
        out.append((k, rows, r[rows, None]))
    return out


def sort_rows(e: SlotExperts, n: int) -> tuple[Rows, GroupEnds]:
    """Rows with e >= 0, sorted by e, and the cumulative group sizes that grouped_mm takes as offs."""
    rows = (e >= 0).nonzero().squeeze(-1)
    rows = rows[e[rows].argsort()]
    return rows, torch.bincount(e[rows], minlength=n).cumsum(0).to(torch.int32)


def gmm(a: GroupedRows, w: StackedWeight, offs: GroupEnds) -> GroupedOut:
    """One grouped GEMM: rows offs[g-1]:offs[g] of a times w[g]^T (CUDA sm80+, or CPU; bf16/fp16/fp32)."""
    return F.grouped_mm(a, w.mT, offs=offs)
