"""Stateless building blocks: the SwiGLU MLP, grouping routed rows, one grouped GEMM, and bit packing."""

import torch
import torch.nn.functional as F

from .constants import GROUPED_DTYPES, GROUPED_MIN_PAIRS
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


def pow2(n: int) -> int:
    r"""Smallest power of 2 that is at least :math:`n` (:math:`n \ge 1`)."""
    return 1 << (n - 1).bit_length()


def pack_bits(b: torch.Tensor) -> torch.Tensor:
    r"""Pack booleans along the last dimension into int32 words.

    Args:
        b: (..., C) bool.

    Returns:
        (..., ceil(C / 32)) int32: entry :math:`c` in bit :math:`c \bmod 32` of word :math:`\lfloor c / 32 \rfloor`,
        set where True; the padding bits of the last word are clear.
    """
    words = -(-b.shape[-1] // 32)
    b = F.pad(b, (0, words * 32 - b.shape[-1])).view(*b.shape[:-1], words, 32).long()
    w = (b << torch.arange(32, device=b.device)).sum(-1)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def unpack_bits(w: torch.Tensor) -> torch.Tensor:
    """Unpack ``pack_bits`` words.

    Args:
        w: (..., W) int32 words.

    Returns:
        (..., 32 W) int32 of 0 and 1, padding bits included.
    """
    return ((w[..., None] >> torch.arange(32, dtype=torch.int32, device=w.device)) & 1).flatten(-2)


def mlp(x: Tokens, gate: InWeight, up: InWeight, down: OutWeight) -> Tokens:
    r"""SwiGLU MLP: :math:`\mathrm{down}(\mathrm{silu}(\mathrm{gate}\, x) \odot \mathrm{up}\, x)`.

    Args:
        x: (T, H) tokens.
        gate: (I, H) gate weight.
        up: (I, H) up weight.
        down: (H, I) down weight.

    Returns:
        (T, H) MLP output.
    """
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def softmax_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k routing: float32 softmax (as transformers' MoE routers; bf16 can flip near-ties), top-k, renormalised.

    Args:
        logits: (T, E) router output.
        k: experts picked per token.

    Returns:
        (idx, r): idx (T, k) int64 expert indices; r (T, k) float32 weights, rows summing to 1.
    """
    r, i = logits.softmax(-1, dtype=torch.float).topk(k, dim=-1)
    return i, r / r.sum(-1, keepdim=True)


def use_grouped(x: Tokens, top_k: int) -> bool:
    """Use grouped GEMM? Needs ``F.grouped_mm`` with this dtype on this device (CPU, or CUDA sm80+), and more than
    one (token, slot) pair (with one pair the sort and count cost more than the loop saves).

    Args:
        x: (T, H) tokens.
        top_k: slots per token.

    Returns:
        True to take the grouped-GEMM forward, False for the loop.
    """
    if not hasattr(F, "grouped_mm") or x.dtype not in GROUPED_DTYPES or x.device.type not in ("cuda", "cpu"):
        return False
    if x.device.type == "cuda" and torch.cuda.get_device_capability(x.device) < (8, 0):
        return False
    return x.shape[0] * top_k >= GROUPED_MIN_PAIRS


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
    r"""Rows with :math:`e \ge 0`, sorted by :math:`e`, and the cumulative group sizes that ``grouped_mm`` takes as ``offs``.

    Args:
        e: (T,) int64 group index per row; rows with :math:`e < 0` are dropped.
        n: Number of groups G; e must be < n.

    Returns:
        A tuple (rows, offs): rows (R,) int64 indices of the rows with :math:`e \ge 0`, sorted by :math:`e`; offs (G,) int32
        cumulative group sizes.
    """
    rows = (e >= 0).nonzero().squeeze(-1)
    rows = rows[e[rows].argsort()]
    return rows, torch.bincount(e[rows], minlength=n).cumsum(0).to(torch.int32)


def gmm(a: GroupedRows, w: StackedWeight, offs: GroupEnds) -> GroupedOut:
    r"""One grouped GEMM: rows ``offs[g-1]:offs[g]`` of ``a`` times :math:`w[g]^\top` (CUDA sm80+, or CPU;
    bf16/fp16/fp32).

    Args:
        a: (R, in) rows sorted by group.
        w: (G, out, in) one weight per group.
        offs: (G,) int32 cumulative group sizes, as from ``sort_rows``.

    Returns:
        (R, out) row i of a times the transposed weight of its group.
    """
    return F.grouped_mm(a, w.mT, offs=offs)
