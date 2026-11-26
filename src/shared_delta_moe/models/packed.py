r"""Base class for DeltaMoEs whose deltas are stored packed (seeded DARE in ``dare.py``, BitDelta in ``bitdelta.py``).

A subclass registers its storage in ``_init_deltas`` and implements ``_unpack(name, e)``, the dense delta of one
projection of expert :math:`e`; it builds from a DeltaMoE with ``from_delta_moe``, which copies the shared parts with
``_copy_shared`` and packs the deltas. The rest is common: the materialised path unpacks only the routed expert
(``delta(e)``), ``d_gate`` / ``d_up`` / ``d_down`` unpack every expert on access (so the other inherited forwards work
too, as references), and the fused Triton kernels read the packed form forward-only (``_kernel_deltas``); with a
gradient needed the materialised path runs.
"""

import torch
from torch import Tensor

from ..constants import HAS_TRITON, SORT_ONE_HOT_MAX
from ..ops import pow2
from .delta_moe import DeltaMoE

PROJ = ("gate", "up", "down")


class PackedDeltaMoE(DeltaMoE):
    """DeltaMoE with packed deltas, unpacked per expert (see the module docstring). Not used on its own."""

    def _unpack(self, name: str, e: int) -> Tensor:
        """Dense delta of projection ``name`` of expert ``e``: (I, H) for gate / up, (H, I) for down."""
        raise NotImplementedError

    def delta(self, e: int) -> tuple[Tensor, Tensor, Tensor]:
        r"""Dense deltas of one expert, unpacked from the packed storage; differentiable in it.

        Args:
            e: Delta expert index, in [0, n_delta).

        Returns:
            (:math:`\Delta_\mathrm{gate}` (I, H), :math:`\Delta_\mathrm{up}` (I, H), :math:`\Delta_\mathrm{down}`
            (H, I)) of expert :math:`e`.

        Raises:
            RuntimeError: If the storage cannot be unpacked yet (DARE: mask bitmaps not built, e.g. made on meta,
                then ``to_empty``, no load).
        """
        return self._unpack("gate", e), self._unpack("up", e), self._unpack("down", e)

    def _delta(self, name: str) -> Tensor:
        G, R, C = self.shapes[name]
        if G == 0:
            return self.gate.new_zeros(G, R, C)
        return torch.stack([self._unpack(name, e) for e in range(G)])

    @property
    def d_gate(self) -> Tensor:  # type: ignore[override]
        """Dense gate deltas of every expert, unpacked on each access: (N, I, H)."""
        return self._delta("gate")

    @property
    def d_up(self) -> Tensor:  # type: ignore[override]
        """Dense up deltas of every expert, unpacked on each access: (N, I, H)."""
        return self._delta("up")

    @property
    def d_down(self) -> Tensor:  # type: ignore[override]
        """Dense down deltas of every expert, unpacked on each access: (N, H, I)."""
        return self._delta("down")

    def forward_matrix(self, x: Tensor) -> Tensor:
        r"""MoE forward by per-matrix output aggregation (``DeltaMoE.forward_matrix``).

        On an H100 / H200 with the prebuilt CUDA library (``cuda_kernels/packed_ops.py``), forward-only and up to
        ``MATRIX_MAX_PAIRS`` (token, slot) pairs: :math:`W_b` products by cuBLAS, the packed deltas' products by the
        kernels (no host sync, CUDA-graph capturable). Otherwise the inherited forward on unpacked deltas.

        Args:
            x: (T, H) tokens.

        Returns:
            (T, H) MoE output.
        """
        if self._matrix_cuda(x):
            from ..cuda_kernels.packed_ops import matrix_moe_any

            idx, r = self.topk(x)
            return matrix_moe_any(x, idx, r, self.n_delta, (self.gate, self.up, self.down), self._kernel_deltas())
        return super().forward_matrix(x)

    def _matrix_cuda(self, x: Tensor) -> bool:
        if not (HAS_TRITON and x.is_cuda and x.dim() == 2) or self._needs_grad(x):
            return False
        if pow2(self.top_k) * pow2(self.n_delta + 2) > SORT_ONE_HOT_MAX:  # the routing sort (Triton)
            return False
        from ..cuda_kernels import packed_ops

        if not packed_ops.available():
            return False
        return packed_ops.supports(x, x.shape[0] * self.top_k, (self.gate, self.up, self.down), self._kernel_deltas(),
                                   matrix=True)  # fmt: skip

    def _use_triton(self, x: Tensor) -> bool:
        # the fused kernels read packed deltas forward-only (triton_train's backward takes dense deltas): with a
        # gradient needed, the materialised path
        return not self._needs_grad(x) and super()._use_triton(x)

    @torch.no_grad()
    def _copy_shared(self, m: DeltaMoE) -> None:
        """Copy m's router and base weights, which the packed form keeps as they are.

        Raises:
            TypeError: If m is a HierDeltaMoE (a packed DeltaMoE has a flat router).
        """
        from .hier import HierDeltaMoE

        if isinstance(m, HierDeltaMoE):
            raise TypeError(f"{type(self).__name__} has a flat router; a HierDeltaMoE cannot be converted")
        self.router.weight.copy_(m.router.weight)
        for n in PROJ:
            getattr(self, n).copy_(getattr(m, n))
