r"""Group-grid deltas (``compress.quantize``, round to nearest or GPTQ): each entry of :math:`\Delta_e` is a code
:math:`q \in \{0, \dots, L - 1\}` on a grid of :math:`L` evenly spaced values per group of :math:`g` entries along a
row,

.. math::

   \hat\delta = \ell + s\,q,

with per group a lowest level :math:`\ell` and step :math:`s` (symmetric grids: :math:`\ell = -s (L - 1) / 2`, not
stored). The steps and lowest levels are ``nn.Parameter`` objects, trainable through the materialised forward as
BitDelta's scales; the codes are fixed.

Layout, projection (R, C) of delta :math:`e`, with :math:`b` code bits (the smallest of 1, 2, 4 with
:math:`2^b \ge L`) and :math:`C b` a multiple of 32:

- ``c_<proj>``: (N, R, C b / 32) int32; entry :math:`c` of a row in bits :math:`b (c \bmod 32/b)` onwards of word
  :math:`\lfloor c b / 32 \rfloor`.
- ``s_<proj>``: (N, R, C / g) steps in the model dtype.
- ``z_<proj>``: (N, R, C / g) lowest levels in the model dtype (asymmetric grids only).

Storage per entry, with 16-bit (bf16 / fp16) grids: :math:`b + 16/g` bits (symmetric) or :math:`b + 32/g`; 2 bits,
:math:`g = 128`: 2.25. The module computes :math:`\ell + s q` in the model dtype from grids rounded to it, so in bf16
its deltas differ from ``compress``'s float32 simulation by up to one bf16 ulp of the group's largest entry, far
below the quantisation error. Ternary grids
take 2 code bits here (a base-3 packing would take 1.6). ``GridHierDeltaMoE`` is the same storage under
``HierDeltaMoE``'s two-level routing.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .. import compress as C
from ..constants import GROUPED_DTYPES, HAS_TRITON
from ..ops import recompute_saved
from .delta_moe import DeltaMoE
from .hier import HierDeltaMoE
from .moe import grouped_moe
from .packed import PROJ, PackedDeltaMoE


def code_bits(levels: int) -> int:
    """Code bits for ``levels`` grid values: the smallest of 1, 2, 4 that holds them (a power of 2 divides 32)."""
    assert 2 <= levels <= 16, f"{levels} levels"
    return next(b for b in (1, 2, 4) if 2**b >= levels)


def pack_codes(q: Tensor, bits: int) -> Tensor:
    """Pack codes along the last dimension into int32 words, ``32 / bits`` per word, low bits first.

    Args:
        q: (..., C) integer codes in [0, 2**bits), C * bits a multiple of 32.
        bits: 1, 2 or 4.

    Returns:
        (..., C * bits / 32) int32.
    """
    per = 32 // bits
    assert q.shape[-1] % per == 0, f"{q.shape[-1]} codes do not fill {bits}-bit words"
    shift = torch.arange(0, 32, bits, device=q.device, dtype=torch.int64)
    w = (q.reshape(*q.shape[:-1], -1, per).to(torch.int64) << shift).sum(-1)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def unpack_codes(w: Tensor, bits: int) -> Tensor:
    """Inverse of ``pack_codes``: (..., words) int32 -> (..., words * 32 / bits) int32 codes."""
    shift = torch.arange(0, 32, bits, device=w.device, dtype=torch.int32)
    return ((w[..., None] >> shift) & (2**bits - 1)).flatten(-2)


def grid_delta(codes: Tensor, step: Tensor, low: Tensor | None, levels: int, bits: int) -> Tensor:
    r"""One dense delta from its grid storage, :math:`\ell + s q` per group; differentiable in ``step`` and ``low``.

    Args:
        codes: (R, C b / 32) int32 packed codes (``pack_codes``).
        step: (R, C / g) steps.
        low: (R, C / g) lowest levels; None = symmetric grids (:math:`\ell = -s (L - 1) / 2`).
        levels: :math:`L`.
        bits: :math:`b`, code bits.

    Returns:
        (R, C) in ``step``'s dtype.
    """
    R, n = step.shape
    s = step[..., None]
    z = -s * ((levels - 1) / 2) if low is None else low[..., None]
    q = unpack_codes(codes, bits).view(R, n, -1)
    return torch.addcmul(z, q.to(s.dtype), s).view(R, -1)  # one fma, the grids broadcast over each group


class GridDeltaMoE(PackedDeltaMoE):
    """DeltaMoE whose deltas are group-grid codes (see the module docstring).

    The materialised path unpacks only the routed expert (``delta(e)``); ``d_gate`` / ``d_up`` / ``d_down`` unpack
    every expert on access, so the other inherited forwards work too (as references). The module's forward, for
    2-bit codes: the fused Triton forward (``triton_kernels/fused.py`` mode 3, forward-only, the pair counts of
    ``DeltaMoE._use_triton``), else the materialised one. Build with ``from_delta_moe``; the constructor leaves the buffers
    uninitialised (``torch.empty``).

    Router training (a gradient needed, every parameter but the router frozen, 2-bit codes, CUDA): every expert decoded
    in one Triton pass per call (``triton_kernels/decode.py``), then cuBLAS grouped GEMMs (``_forward_decoded``); the
    decoded weights are rebuilt for the backward, not kept. Other training (trainable grids or base) takes the
    materialised path.

    Args:
        hidden: H, model hidden size.
        inter: I, expert intermediate size.
        n_delta: :math:`N`, number of delta experts.
        top_k: Experts per token.
        num_zero_experts: :math:`Z`, router rows that route to the base alone.
        device: Device of the parameters and buffers.
        dtype: Dtype of the parameters, steps and lowest levels.
        levels: :math:`L`, grid values.
        group: :math:`g`, entries per grid; divides H and I.
        symmetric: Grids centred on 0 (no lowest levels stored).
    """

    def __init__(self, hidden: int, inter: int, n_delta: int, top_k: int = 1, num_zero_experts: int = 1,
                 device: torch.device | str | None = None, dtype: torch.dtype | None = None, levels: int = 4,
                 group: int = 128, symmetric: bool = False) -> None:  # fmt: skip
        self._grid_args(levels, group, symmetric)  # read by _init_deltas, which DeltaMoE.__init__ calls
        super().__init__(hidden, inter, n_delta, top_k, num_zero_experts, device, dtype)

    def _grid_args(self, levels: int, group: int, symmetric: bool) -> None:
        self.levels, self.group, self.symmetric, self.bits = levels, group, symmetric, code_bits(levels)

    def _init_deltas(self, n: int, inter: int, hidden: int, factory: dict[str, Any]) -> None:
        for name, (G, R, Cn) in self.shapes.items():
            assert Cn % self.group == 0 and Cn * self.bits % 32 == 0, f"{name}: {Cn} columns, group {self.group}"
            self.register_buffer(f"c_{name}", torch.empty(G, R, Cn * self.bits // 32, dtype=torch.int32,
                                                          device=factory["device"]))  # fmt: skip
            setattr(self, f"s_{name}", nn.Parameter(torch.empty(G, R, Cn // self.group, **factory)))
            if not self.symmetric:
                setattr(self, f"z_{name}", nn.Parameter(torch.empty(G, R, Cn // self.group, **factory)))

    def _unpack(self, name: str, e: int) -> Tensor:
        r"""Dense :math:`\Delta_e` of one projection, :math:`\ell + s q` per group; differentiable in the steps and
        lowest levels."""
        return grid_delta(getattr(self, f"c_{name}")[e], getattr(self, f"s_{name}")[e],
                          None if self.symmetric else getattr(self, f"z_{name}")[e], self.levels, self.bits)

    def bits_per_entry(self) -> float:
        """Storage per delta entry: code bits plus the steps (and lowest levels) in the model dtype."""
        return self.bits + (1 if self.symmetric else 2) * 8 * self.s_gate.element_size() / self.group

    def kernel_ok(self) -> bool:
        """Whether the fused Triton kernels read these grids (mode 3): 2-bit codes, a power-of-2 group."""
        return self.bits == 2 and self.group & (self.group - 1) == 0

    def _router_only(self) -> bool:
        return all(not p.requires_grad for n, p in self.named_parameters() if n != "router.weight")

    def _use_triton(self, x: Tensor) -> bool:
        return self.kernel_ok() and super()._use_triton(x)

    def _use_decoded(self, x: Tensor) -> bool:
        return self._needs_grad(x) and self._decodable(x)

    def _decodable(self, x: Tensor) -> bool:
        # x in the weights' dtype, or (an fp32 residual stream) under autocast to it, which the GEMMs would cast to
        same = x.dtype == self.gate.dtype or (x.is_cuda and torch.is_autocast_enabled("cuda")
                                              and torch.get_autocast_dtype("cuda") == self.gate.dtype)  # fmt: skip
        return (self.kernel_ok() and self._router_only() and HAS_TRITON and x.is_cuda
                and self.gate.dtype in GROUPED_DTYPES and same)  # fmt: skip

    def _decoded(self) -> list[Tensor]:
        """(N + 1, R, C) full weights of every expert per projection, the base last (``decode_grid``)."""
        from ..triton_kernels.decode import decode_grid

        return [decode_grid(d, getattr(self, n).detach()) for d, n in zip(self._kernel_deltas(), PROJ)]

    def _forward_decoded(self, x: Tensor) -> Tensor:
        """Router training: the decoded experts (``_decoded``, rebuilt for the backward through ``recompute_saved``
        rather than kept) and ``moe.grouped_moe``; gradients to x and the routing weights."""
        idx, r = self.topk(x)
        ws = self._decoded()
        with recompute_saved(ws, self._decoded):  # every zero expert -> the base, n_delta
            return grouped_moe(x.to(ws[0].dtype), idx.clamp(max=self.n_delta), r, *ws).to(x.dtype)

    def forward_param(self, x: Tensor) -> Tensor:
        """``DeltaMoE.forward_param``, or for router training ``_forward_decoded`` (see the class docstring)."""
        return self._forward_decoded(x) if self._use_decoded(x) else super().forward_param(x)

    def _kernel_deltas(self) -> "tuple[Any, Any, Any]":
        from ..triton_kernels import Delta

        out = []
        for n in PROJ:
            s = getattr(self, f"s_{n}").detach()
            # ponytail: symmetric grids rebuild their lowest levels per call (1/g of the codes); cache if it shows
            z = -s * ((self.levels - 1) / 2) if self.symmetric else getattr(self, f"z_{n}").detach()
            out.append(Delta(3, getattr(self, f"c_{n}"), s, z, self.group))
        return tuple(out)  # type: ignore[return-value]

    MATRIX_FORWARD_PAIRS = 1024  # the CUDA matrix kernel (mode 3) up to packed_ops.MATRIX_MAX_PAIRS

    def _matrix_cuda(self, x: Tensor) -> bool:
        return self.kernel_ok() and super()._matrix_cuda(x)

    @classmethod
    @torch.no_grad()
    def from_delta_moe(cls, m: DeltaMoE, levels: int = 4, group: int = 128, symmetric: bool = False,
                       clip: bool = False, x: Sequence[Tensor] | None = None,
                       damp: float = 0.01) -> "GridDeltaMoE":  # fmt: skip
        """Grid form of a DeltaMoE (``GridDeltaMoE``) or HierDeltaMoE (``GridHierDeltaMoE``), same base, router and
        routing: every delta quantised by ``compress.quantize``, round to nearest, or GPTQ given calibration inputs.

        Args:
            m: The module, with dense deltas.
            levels: :math:`L`, grid values (4: 2 bits; 3: ternary; 2: one bit).
            group: Entries per grid; divides H and I.
            symmetric: Grids centred on 0.
            clip: Shrink each group's span to the factor of least rounding error (``compress.CLIP``).
            x: Per delta expert, its (T_e, H) MLP inputs, for GPTQ (``compress.calibration_grams``); None: round to
                nearest, no data.
            damp: GPTQ damping.

        Returns:
            The packed module, on m's device and in m's dtype.
        """
        g = m.gate
        if issubclass(cls, HierDeltaMoE) and not isinstance(m, HierDeltaMoE):
            raise TypeError(f"{cls.__name__} and {type(m).__name__} route differently (flat vs two-level)")
        if isinstance(m, HierDeltaMoE):
            out: GridDeltaMoE = GridHierDeltaMoE(g.shape[1], g.shape[0], m.group_sizes, m.group_zeros, m.k1,
                                                 m.group_top_k, g.device, g.dtype, levels, group, symmetric)  # fmt: skip
        else:
            out = GridDeltaMoE(g.shape[1], g.shape[0], m.n_delta, m.top_k, m.num_zero_experts, g.device, g.dtype, levels,
                      group, symmetric)  # fmt: skip
        out._copy_shared(m)
        grams = None if x is None else C.calibration_grams(m, x)
        for j, name in enumerate(PROJ):
            d = getattr(m, f"d_{name}")
            for e in range(m.n_delta):
                q, lo, step = C.quantize(d[e], None if grams is None else grams[e][j // 2], levels, group, symmetric,
                                         clip, damp)  # fmt: skip
                getattr(out, f"c_{name}")[e] = pack_codes(q, out.bits)
                getattr(out, f"s_{name}")[e] = step.to(g.dtype)
                if not symmetric:
                    getattr(out, f"z_{name}")[e] = lo.to(g.dtype)
        return out


class GridHierDeltaMoE(GridDeltaMoE, HierDeltaMoE):
    """``HierDeltaMoE`` (two-level routing) with ``GridDeltaMoE``'s storage. Build with
    ``GridDeltaMoE.from_delta_moe`` from a HierDeltaMoE.

    Args:
        hidden: H.
        inter: I.
        group_sizes: Delta experts per group (``HierDeltaMoE``).
        group_zeros: Zero experts per group.
        top_k: Experts picked per group.
        group_top_k: Groups picked per token; None = dense routing over groups.
        device: Device of the parameters and buffers.
        dtype: Dtype of the parameters.
        levels: Grid values.
        group: Entries per grid.
        symmetric: Grids centred on 0.
    """

    def __init__(self, hidden: int, inter: int, group_sizes: Sequence[int], group_zeros: Sequence[int],
                 top_k: int = 1, group_top_k: int | None = None, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None, levels: int = 4, group: int = 128,
                 symmetric: bool = False) -> None:  # fmt: skip
        self._grid_args(levels, group, symmetric)
        HierDeltaMoE.__init__(self, hidden, inter, group_sizes, group_zeros, top_k, group_top_k, device, dtype)

