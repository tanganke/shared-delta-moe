r"""BitDelta deltas (Liu et al. 2024, "BitDelta: Your Fine-Tune May Only Be Worth One Bit"): each :math:`\Delta_e` is
stored as its sign, 1 bit per entry, times a scale:

.. math::

   \Delta_e \approx \mathrm{diag}(s_e)\,\mathrm{sign}(\Delta_e).

BitDelta takes one scale per matrix, the mean of :math:`|\Delta_e|` (for a fixed sign matrix, the scalar closest to
:math:`\Delta_e` in Frobenius norm); ``scale="row"`` takes one per output row (that row's mean of :math:`|\Delta_e|`);
an int scale :math:`B` one per :math:`1 \times B` block of a row (its mean of :math:`|\Delta_e|`, as DARE's blocks:
:math:`R C / B` values per matrix). The scales are ``nn.Parameter`` objects, so they can be trained afterwards
(BitDelta's scale distillation) through the materialised forward; the signs are fixed.

Layout, projection (R, C) of delta :math:`e`, :math:`C` a multiple of 32:

- ``b_<proj>``: (N, R, C / 32) int32; entry :math:`c` of a row in bit :math:`c \bmod 32` of word
  :math:`\lfloor c / 32 \rfloor`, set = :math:`+s`, clear = :math:`-s` (a zero entry counts as :math:`+`).
- ``s_<proj>``: (N, R, ceil(C / SB)) in the model dtype; entry :math:`c` of a row is scaled by
  ``s[..., c // SB]``, where SB = :math:`B` for block scales, else ``next_power_of_2(C)`` (one per row; matrix scales
  are equal rows).

Storage per delta: 1/16 of bf16, plus the scales.
"""

from typing import Any

import torch
from torch import Tensor, nn

from ..ops import pack_bits, pow2, unpack_bits
from .delta_moe import DeltaMoE
from .packed import PROJ, PackedDeltaMoE


def pack_signs(d: Tensor) -> Tensor:
    r"""Pack the signs of the last dimension into int32 words.

    Args:
        d: (..., C) tensor, C a multiple of 32.

    Returns:
        (..., C / 32) int32: bit :math:`c \bmod 32` of word :math:`\lfloor c / 32 \rfloor` set where
        ``d[..., c] >= 0``.
    """
    assert d.shape[-1] % 32 == 0
    return pack_bits(d >= 0)


def unpack_signs(w: Tensor, dtype: torch.dtype) -> Tensor:
    """Unpack ``pack_signs`` words back to signs.

    Args:
        w: (..., C / 32) int32 sign words, as ``pack_signs`` returns.
        dtype: Dtype of the result.

    Returns:
        (..., C) of :math:`+1` (bit set) and :math:`-1` (bit clear) in dtype.
    """
    return (unpack_bits(w) * 2 - 1).to(dtype)


def scale_block(scale: "str | int", cols: int) -> int:
    """Entries per scale along a row, SB (a power of 2).

    Args:
        scale: ``"matrix"``, ``"row"``, or an int block size B (a power of 2 dividing cols).
        cols: C, columns of the delta matrix.

    Returns:
        B for an int scale, else ``next_power_of_2(cols)`` (one scale per row).
    """
    if isinstance(scale, int):
        assert scale >= 1 and scale & (scale - 1) == 0 and cols % scale == 0, (scale, cols)
        return scale
    assert scale in ("matrix", "row"), scale
    return pow2(cols)


def scales(d: Tensor, scale: "str | int") -> Tensor:
    r"""BitDelta scales of a stack of deltas: the mean of :math:`|\Delta|` over each scale's entries.

    Args:
        d: (N, R, C) deltas.
        scale: ``"matrix"`` (one per matrix, BitDelta), ``"row"`` (one per row) or an int B (one per
            :math:`1 \times B` block of a row).

    Returns:
        (N, R, ceil(C / SB)) scales in d's dtype; ``"matrix"`` and ``"row"`` give one per row ((N, R, 1)), equal
        across the rows of a matrix for ``"matrix"``.
    """
    a = d.float().abs()
    if scale == "matrix":
        s = a.mean((-2, -1))[:, None, None].expand(*d.shape[:2], 1)
    elif scale == "row":
        s = a.mean(-1, keepdim=True)
    else:
        s = a.view(*d.shape[:2], -1, scale_block(scale, d.shape[-1])).mean(-1)
    return s.to(d.dtype)


class BitDeltaMoE(PackedDeltaMoE):
    """DeltaMoE whose deltas are BitDelta (see the module docstring).

    The materialised path unpacks only the routed expert (``delta(e)``); ``d_gate`` / ``d_up`` / ``d_down`` unpack
    every expert on access, so the other inherited forwards work too (as references); the fused Triton path reads the
    sign bits directly (``triton_kernels/fused.py``, delta mode 2). Build with ``from_delta_moe``; the constructor
    leaves the buffers uninitialised (``torch.empty``).

    Args:
        hidden: H, model hidden size; a multiple of 32.
        inter: I, expert intermediate size; a multiple of 32.
        n_delta: :math:`N`, number of delta experts.
        top_k: Experts per token.
        num_zero_experts: :math:`Z`, router rows that route to the base alone.
        device: Device of the parameters and buffers.
        dtype: Dtype of the parameters and scales.
        scale: Fixes the scales' shape: an int B for block scales, else (``"matrix"`` or ``"row"``) one per row.
    """

    def __init__(self, hidden: int, inter: int, n_delta: int, top_k: int = 1, num_zero_experts: int = 1,
                 device: torch.device | str | None = None, dtype: torch.dtype | None = None,
                 scale: "str | int" = "row") -> None:  # fmt: skip
        self.scale = scale  # read by _init_deltas, which DeltaMoE.__init__ calls
        super().__init__(hidden, inter, n_delta, top_k, num_zero_experts, device, dtype)

    def _init_deltas(self, n: int, inter: int, hidden: int, factory: dict[str, Any]) -> None:
        assert inter % 32 == 0 and hidden % 32 == 0, "sign bits are packed 32 to an int32 word along each row"
        self.sblock = {name: scale_block(self.scale, C) for name, (_, _, C) in self.shapes.items()}
        for name, (G, R, C) in self.shapes.items():
            self.register_buffer(f"b_{name}", torch.empty(G, R, C // 32, dtype=torch.int32, device=factory["device"]))
            setattr(self, f"s_{name}", nn.Parameter(torch.empty(G, R, -(-C // self.sblock[name]), **factory)))

    @classmethod
    @torch.no_grad()
    def from_delta_moe(cls, m: DeltaMoE, scale: "str | int" = "matrix") -> "BitDeltaMoE":
        """BitDelta form of a DeltaMoE, with the same base, router, top-k and zero experts.

        Args:
            m: The DeltaMoE to convert.
            scale: ``"matrix"`` (BitDelta), ``"row"`` or an int block size (see ``scales``).

        Returns:
            The BitDeltaMoE, on m's device and in m's dtype, holding each delta's sign bits and its scales.

        Raises:
            TypeError: If m is a HierDeltaMoE (BitDeltaMoE has a flat router).
        """
        g = m.gate
        out = cls(g.shape[1], g.shape[0], m.n_delta, m.top_k, m.num_zero_experts, g.device, g.dtype, scale)
        out._copy_shared(m)
        for name in PROJ:
            d = getattr(m, f"d_{name}")
            getattr(out, f"b_{name}").copy_(pack_signs(d))
            getattr(out, f"s_{name}").copy_(scales(d, scale))
        return out

    def _unpack(self, name: str, e: int) -> Tensor:
        r"""Dense :math:`\mathrm{diag}(s_e)\,\mathrm{sign}(\Delta_e)` of one projection; differentiable in the
        scales."""
        s, C = getattr(self, f"s_{name}"), self.shapes[name][2]
        return s[e].repeat_interleave(self.sblock[name], 1)[:, :C] * unpack_signs(getattr(self, f"b_{name}")[e], s.dtype)

    def _kernel_deltas(self) -> "tuple[Delta, Delta, Delta]":
        from ..triton_kernels import Delta

        # block 0: one scale per row, which the kernels load once per tile rather than once per K step
        return tuple(Delta(2, getattr(self, f"b_{n}"), s := getattr(self, f"s_{n}"), block=self.sblock[n] if s.shape[-1] > 1 else 0)
                     for n in PROJ)  # fmt: skip  # type: ignore[return-value]
