"""Model merging for the parameters an upscaled MoE keeps one copy of (attention, norms, embeddings, ...), done by
``fusion_bench`` (extra ``merging``). Each method follows a ``fusion_bench`` algorithm step for step (named and
parametrised as it) on state dicts: it takes the base state dict and the fine-tuned ones (same keys and shapes) and
returns the merged state dict in the base's dtypes, computed in float32.

ponytail: the converter merges tensor by tensor (it streams the checkpoints), so TIES trims each tensor at its own
threshold, where ``fusion_bench``'s ``TiesMergingAlgorithm`` trims the whole flattened model at once. Pass larger
state dicts (e.g. a whole layer) if that difference matters.
"""

import contextlib
import io
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor

StateDict = Mapping[str, Tensor]


def _f32(sd: StateDict) -> dict[str, Tensor]:
    return {k: v.float() for k, v in sd.items()}


def _like(sd: StateDict, ref: StateDict) -> dict[str, Tensor]:
    return {k: sd[k].to(ref[k].dtype) for k in ref}


def simple_average(base: StateDict, tuned: Sequence[StateDict]) -> dict[str, Tensor]:
    r"""Mean of the fine-tuned state dicts, :math:`\frac{1}{N} \sum_i \theta_i` (``fusion_bench``'s
    ``SimpleAverageAlgorithm``: ``state_dict_avg``).

    Args:
        base: the base state dict; only its dtypes are used.
        tuned: the :math:`N` fine-tuned state dicts :math:`\theta_i`, keyed and shaped like ``base``.

    Returns:
        The merged state dict, in ``base``'s dtypes.
    """
    from fusion_bench.utils.state_dict_arithmetic import state_dict_avg

    return _like(state_dict_avg([_f32(s) for s in tuned]), base)


def task_arithmetic(base: StateDict, tuned: Sequence[StateDict], scaling_factor: float = 0.3) -> dict[str, Tensor]:
    r"""Task arithmetic, :math:`\theta_0 + \lambda \sum_i (\theta_i - \theta_0)` with base :math:`\theta_0` and
    :math:`\lambda` = ``scaling_factor`` (``fusion_bench``'s ``task_arithmetic_merge``).

    Args:
        base: the base state dict :math:`\theta_0`.
        tuned: the fine-tuned state dicts :math:`\theta_i`, keyed and shaped like ``base``.
        scaling_factor: scale of the summed task vectors.

    Returns:
        The merged state dict, in ``base``'s dtypes.
    """
    from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_mul, state_dict_sub, state_dict_sum

    b = _f32(base)
    task_vector = state_dict_sum([state_dict_sub(_f32(s), b) for s in tuned])
    return _like(state_dict_add(b, state_dict_mul(task_vector, scaling_factor)), base)


def ties_merging(base: StateDict, tuned: Sequence[StateDict], scaling_factor: float = 0.3, threshold: float = 20,
                 merge_func: str = "sum") -> dict[str, Tensor]:  # fmt: skip
    r"""TIES merging (``fusion_bench``'s ``TiesMergingAlgorithm``): flattens each task vector
    :math:`\tau_i = \theta_i - \theta_0`, keeps its top ``threshold`` entries by magnitude, elects a sign per entry,
    merges the agreeing values into :math:`\tau` and returns :math:`\theta_0 + \lambda \tau`.

    Args:
        base: the base state dict :math:`\theta_0`.
        tuned: the fine-tuned state dicts, keyed and shaped like ``base``.
        scaling_factor: :math:`\lambda`, scale of the merged task vector.
        threshold: share of entries kept per task vector: a fraction if :math:`\le 1`, else a percentage. 1 (or 100)
            keeps every entry; ``fusion_bench`` 0.2.33 fails on it, fixed on its main (tanganke/fusion_bench#219).
        merge_func: ``"sum"``, ``"mean"`` or ``"max"``, as ``fusion_bench``.

    Returns:
        The merged state dict, in ``base``'s dtypes.
    """
    from fusion_bench.method.ties_merging.ties_merging_utils import (
        state_dict_to_vector,
        ties_merging,
        vector_to_state_dict,
    )

    b = _f32(base)
    flat_ptm = state_dict_to_vector(b)
    tv_flat_checks = torch.vstack([state_dict_to_vector(_f32(s)) for s in tuned]) - flat_ptm
    with contextlib.redirect_stdout(io.StringIO()):  # it prints two lines per call; the converter calls it per tensor
        merged_tv = ties_merging(tv_flat_checks, reset_thresh=threshold, merge_func=merge_func)
    return _like(vector_to_state_dict(flat_ptm + scaling_factor * merged_tv, b), base)


MERGE: dict[str, Callable[..., dict[str, Tensor]]] = {
    "simple_average": simple_average,
    "task_arithmetic": task_arithmetic,
    "ties_merging": ties_merging,
}
