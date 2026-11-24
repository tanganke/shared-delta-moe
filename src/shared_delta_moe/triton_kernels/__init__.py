"""Fused Triton kernels for parameter aggregation (import needs triton)."""

from .fused import Delta, fused_moe_any as fused_moe

__all__ = ["Delta", "fused_moe"]
