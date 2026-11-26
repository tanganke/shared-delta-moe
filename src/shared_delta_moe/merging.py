"""Model-merging methods for the parameters an upscaled MoE keeps one copy of (attention, norms, embeddings, ...),
named and parametrised after ``fusion_bench``'s algorithms. Each takes the base tensor and the fine-tuned tensors (same
shape) and returns the merged tensor in the base's dtype, computed in float32.

ponytail: per tensor. ``fusion_bench``'s TIES trims over the whole flattened task vector; here each tensor keeps its own
top threshold. Add a global pass if that difference matters.
"""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def simple_average(base: Tensor, tuned: Sequence[Tensor]) -> Tensor:
    r"""Mean of the fine-tuned tensors.

    Args:
        base: the base tensor; only its dtype is used.
        tuned: the fine-tuned tensors, each shaped like ``base``.

    Returns:
        :math:`\frac{1}{N} \sum_i \theta_i` over the :math:`N` fine-tuned tensors :math:`\theta_i`, shaped like
        ``base``, in ``base``'s dtype (computed in float32).
    """
    return torch.stack([t.float() for t in tuned]).mean(0).to(base.dtype)


def task_arithmetic(base: Tensor, tuned: Sequence[Tensor], scaling_factor: float = 0.3) -> Tensor:
    r"""Task arithmetic: :math:`\theta_0 + \lambda \sum_i (\theta_i - \theta_0)`, with base :math:`\theta_0`, fine-tuned
    :math:`\theta_i` and :math:`\lambda` = ``scaling_factor``.

    Args:
        base: the base tensor.
        tuned: the fine-tuned tensors, each shaped like ``base``.
        scaling_factor: scale of the summed task vectors.

    Returns:
        The merged tensor, shaped like ``base``, in base's dtype (computed in float32).
    """
    b = base.float()
    return (b + scaling_factor * sum(t.float() - b for t in tuned)).to(base.dtype)


def ties_merging(base: Tensor, tuned: Sequence[Tensor], scaling_factor: float = 0.3, threshold: float = 20,
                 merge_func: str = "sum") -> Tensor:  # fmt: skip
    r"""TIES merging: trim, elect sign, merge agreeing values.

    Keeps each task vector's (:math:`\tau_i = \theta_i - \theta_0`, fine-tuned minus base) top ``threshold`` entries
    by magnitude :math:`|\tau_i|` (at least one), elects a sign per entry (the sign of the sum of the kept values),
    merges the kept values agreeing with it into :math:`\tau`, and returns :math:`\theta_0 + \lambda \tau` with
    :math:`\lambda` = ``scaling_factor``.

    Args:
        base: the base tensor (any shape, 0-dim included).
        tuned: the fine-tuned tensors, each shaped like ``base``.
        scaling_factor: scale of the merged task vector.
        threshold: share of entries kept per task vector, as ``fusion_bench``: a fraction if :math:`\le 1`, else a
            percentage.
        merge_func: ``"sum"`` (sum of the agreeing values) or ``"mean"`` (their mean over the task vectors that agree).

    Returns:
        The merged tensor, shaped like ``base``, in base's dtype (computed in float32).
    """
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
