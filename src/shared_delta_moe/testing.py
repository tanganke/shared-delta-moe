"""Random experts for tests and benchmarks."""

import torch

from .type_hints import MLPWeights


def random_experts(
    hidden: int, inter: int, n: int, dtype: torch.dtype, device: str
) -> list[MLPWeights]:
    """n random (gate, up, down), scaled 1/sqrt(fan_in) so activations stay O(1)."""

    def r(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=dtype, device=device) * shape[-1] ** -0.5

    return [(r(inter, hidden), r(inter, hidden), r(hidden, inter)) for _ in range(n)]
