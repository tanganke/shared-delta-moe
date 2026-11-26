"""Model-merging methods for the parameters an upscaled MoE keeps one copy of (attention, norms, embeddings, ...),
named and parametrised after fusion_bench's algorithms. Each takes the base tensor and the fine-tuned tensors (same
shape) and returns the merged tensor in the base's dtype, computed in float32.

ponytail: per tensor. fusion_bench's TIES trims over the whole flattened task vector; here each tensor keeps its own
top threshold. Add a global pass if that difference matters.
"""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def simple_average(base: Tensor, tuned: Sequence[Tensor]) -> Tensor:
    return torch.stack([t.float() for t in tuned]).mean(0).to(base.dtype)


def task_arithmetic(base: Tensor, tuned: Sequence[Tensor], scaling_factor: float = 0.3) -> Tensor:
    """base + scaling_factor * sum_i (tuned_i - base)."""
    b = base.float()
    return (b + scaling_factor * sum(t.float() - b for t in tuned)).to(base.dtype)


def ties_merging(base: Tensor, tuned: Sequence[Tensor], scaling_factor: float = 0.3, threshold: float = 20,
                 merge_func: str = "sum") -> Tensor:  # fmt: skip
    """TIES: keep each task vector's top threshold entries by magnitude (as fusion_bench: a fraction if <= 1, else a
    percentage), elect a sign per entry (sign of the sum of
    the kept values), merge the values agreeing with it (merge_func "sum" or "mean"), base + scaling_factor * that."""
    b = base.float()
    tv = torch.stack([t.float() - b for t in tuned]).reshape(len(tuned), -1)  # N, numel (0-dim tensors too)
    k = max(1, int(tv.shape[1] * (threshold if threshold <= 1 else threshold / 100)))
    cut = tv.abs().kthvalue(tv.shape[1] - k + 1, dim=1, keepdim=True).values if k < tv.shape[1] else 0
    tv = tv * (tv.abs() >= cut)
    agree = (tv.sign() == tv.sum(0).sign()) & (tv != 0)
    merged = (tv * agree).sum(0)
    if merge_func == "mean":
        merged = merged / agree.sum(0).clamp_min(1)
    else:
        assert merge_func == "sum", merge_func
    return (b + scaling_factor * merged.view_as(b)).to(base.dtype)


MERGE: dict[str, Callable[..., Tensor]] = {
    "simple_average": simple_average,
    "task_arithmetic": task_arithmetic,
    "ties_merging": ties_merging,
}
