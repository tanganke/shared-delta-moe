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
    """SwiGLU MLP: down(silu(gate x) * up x).

    Args:
        x: (T, H) tokens.
        gate: (I, H) gate weight.
        up: (I, H) up weight.
        down: (H, I) down weight.

    Returns:
        (T, H) MLP output.
    """
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def groups(e: SlotExperts, r: SlotWeights) -> list[Group]:
    """Group the tokens of one routing slot by the expert they use.

    Args:
        e: (T,) int64 delta index per token for this slot, -1 = any zero expert.
        r: (T,) routing weight per token for this slot.

    Returns:
        One (k, rows, coef) per distinct expert in e: k the delta index or -1, rows (R,) int64 token indices routed
        to it, coef (R, 1) their routing weights.
    """
    out: list[Group] = []
    for k in e.unique().tolist():
        rows = (e == k).nonzero().squeeze(-1)
        out.append((k, rows, r[rows, None]))
    return out


def sort_rows(e: SlotExperts, n: int) -> tuple[Rows, GroupEnds]:
    """Rows with e >= 0, sorted by e, and the cumulative group sizes that grouped_mm takes as offs.

    Args:
        e: (T,) int64 group index per row; rows with e < 0 are dropped.
        n: Number of groups G; e must be < n.

    Returns:
        A tuple (rows, offs): rows (R,) int64 indices of the rows with e >= 0, sorted by e; offs (G,) int32
        cumulative group sizes.
    """
    rows = (e >= 0).nonzero().squeeze(-1)
    rows = rows[e[rows].argsort()]
    return rows, torch.bincount(e[rows], minlength=n).cumsum(0).to(torch.int32)


def gmm(a: GroupedRows, w: StackedWeight, offs: GroupEnds) -> GroupedOut:
    """One grouped GEMM: rows offs[g-1]:offs[g] of a times w[g]^T (CUDA sm80+, or CPU; bf16/fp16/fp32).

    Args:
        a: (R, in) rows sorted by group.
        w: (G, out, in) one weight per group.
        offs: (G,) int32 cumulative group sizes, as from sort_rows.

    Returns:
        (R, out) row i of a times the transposed weight of its group.
    """
    return F.grouped_mm(a, w.mT, offs=offs)
