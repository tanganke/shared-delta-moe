"""Shape-annotated tensor types shared across the package. The string is the shape, read as:

- T: tokens
- H: hidden
- I: intermediate
- N: deltas
- Z: zero experts
- K: top-k
- R: rows of one group
- G: groups
"""

from typing import Annotated, TypeAlias

from torch import Tensor

Tokens: TypeAlias = Annotated[Tensor, "T H"]
Hidden: TypeAlias = Annotated[Tensor, "T I"]
PairHidden: TypeAlias = Annotated[Tensor, "T*K I, one row per (token, slot) pair"]
Experts: TypeAlias = Annotated[Tensor, "T K int64, delta index, -1 = any zero expert"]
RouteWeights: TypeAlias = Annotated[Tensor, "T K, rows sum to 1"]
SlotExperts: TypeAlias = Annotated[Tensor, "T int64, one routing slot of Experts"]
SlotWeights: TypeAlias = Annotated[Tensor, "T, one routing slot of RouteWeights"]
Rows: TypeAlias = Annotated[Tensor, "R int64 token indices"]
Coef: TypeAlias = Annotated[Tensor, "R 1 router weight of those rows"]
GroupEnds: TypeAlias = Annotated[
    Tensor, "G int32 cumulative group sizes (grouped_mm offs)"
]
GroupedRows: TypeAlias = Annotated[Tensor, "R in, rows sorted by group"]
GroupedOut: TypeAlias = Annotated[Tensor, "R out"]
StackedWeight: TypeAlias = Annotated[Tensor, "G out in"]
InWeight: TypeAlias = Annotated[Tensor, "I H"]
OutWeight: TypeAlias = Annotated[Tensor, "H I"]
MLPWeights: TypeAlias = tuple[InWeight, InWeight, OutWeight]
Group: TypeAlias = tuple[int, Rows, Coef]
RouterWeight: TypeAlias = Annotated[
    Tensor, "N+Z H, rows N..N+Z-1 = zero experts (base)"
]
DenseExperts: TypeAlias = list[Annotated[Tensor, "N+1 out in, index 0 = base"]]
