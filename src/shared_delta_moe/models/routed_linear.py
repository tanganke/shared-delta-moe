r"""Token-routed deltas on a plain linear layer, for the weights outside the MoE MLP (attention, linear attention,
``lm_head``) of fine-tunes that changed every weight.

Token :math:`t` with routing slots :math:`(e_i, r_i)`, :math:`\sum_i r_i = 1`, gets

.. math::

   y_t = W_b x_t + \sum_i r_i \, \Delta_{e_i} x_t = \Big(W_b + \sum_i r_i \Delta_{e_i}\Big) x_t,

the routed experts' weights mixed per token (the layer is linear, so mixing outputs and mixing weights agree). A slot
with :math:`e_i \ge N` (a zero expert) adds nothing. The routing comes from outside (``route``): one router per
attention block, routed like the layer's MoE MLP (``DeltaMoE.topk`` with that router), set by a forward pre-hook
(``models/qwen3_5/common.py``), so the projections of a block share it.

Deltas are dense (``deltas``) or group-grid codes as in ``grid.py`` (``codes``, ``steps``, ``lows``). The forward is
:math:`W_b x` plus, without a gradient, on CUDA in bf16 / fp16 with 2-bit grids: one fused Triton launch for
:math:`\sum_i r_i \Delta_{e_i} x` (``triton_kernels/fused.py``'s down kernel, mode 3, over a zero base; no host
sync). Otherwise a loop over the routed experts with ``index_add`` (differentiable: routers, steps, dense deltas);
there frozen 2-bit grids are decoded by one Triton pass per expert (``triton_kernels/decode.py``) and rebuilt for the
backward rather than kept (``ops.recompute_saved``).
"""

import threading
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..constants import HAS_TRITON
from ..ops import recompute_saved
from .grid import code_bits, grid_delta


def set_table(sets: list[int], device: torch.device | str | None = None) -> Tensor:
    """Delta expert -> its set (``routed_deltas.sets``: consecutive expert counts), (sum(sets),) int64."""
    return torch.repeat_interleave(torch.arange(len(sets), device=device), torch.tensor(sets, device=device))


def to_sets(idx: Tensor, n_delta: int, table: Tensor, n_sets: int) -> Tensor:
    """Map routing slots from the MoE's delta experts (``>= n_delta``: zero experts / the base) to sets (``n_sets``:
    no delta); no host sync."""
    return torch.where(idx < n_delta, table[idx.clamp(min=0, max=max(n_delta - 1, 0))], n_sets)


_tls = threading.local()  # routes are per thread: forwards may run in parallel threads (one micro-batch each)


class RoutedDeltaLinear(nn.Linear):
    r"""``nn.Linear`` (no bias) holding :math:`W_b` (``weight``) plus :math:`N` per-expert deltas, applied per token
    by the routing in ``route``. Parameters and buffers are uninitialised (``torch.empty``).

    Args:
        in_features: Input width C.
        out_features: Output width R.
        n_delta: :math:`N`, delta experts.
        levels: None = dense deltas; else :math:`L` grid values (``grid.py``).
        group: :math:`g`, entries per grid along the input; divides C.
        symmetric: Grids centred on 0 (no lowest levels).
        device: Device.
        dtype: Dtype of the weight, deltas, steps and lowest levels.

    Attributes:
        route: (idx, r), both (T, K): delta index per slot (:math:`\ge N`: none) and routing weight, for the T tokens
            of the next forward in order; set before each forward, per thread.
        weight: (R, C) :math:`W_b`.
        deltas: (N, R, C) dense deltas (``levels`` None).
        codes: (N, R, C b / 32) int32 packed codes (grids).
        steps: (N, R, C / g) grid steps.
        lows: (N, R, C / g) lowest levels (asymmetric grids).
    """

    @property
    def route(self) -> tuple[Tensor, Tensor] | None:
        return getattr(_tls, "routes", {}).get(id(self))

    @route.setter
    def route(self, v: tuple[Tensor, Tensor] | None) -> None:
        _tls.__dict__.setdefault("routes", {})[id(self)] = v

    def __init__(self, in_features: int, out_features: int, n_delta: int, levels: int | None = None, group: int = 128,
                 symmetric: bool = False, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> None:  # fmt: skip
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        f: dict[str, Any] = {"device": device, "dtype": dtype}
        self.n_delta, self.levels, self.group, self.symmetric, self.route = n_delta, levels, group, symmetric, None
        self.sets: list[int] | None = None  # set by models/qwen3_5/common.py when routed deltas are shared by sets
        shape = (n_delta, out_features, in_features)
        if levels is None:
            self.deltas = nn.Parameter(torch.empty(shape, **f))
            return
        self.bits = code_bits(levels)
        assert in_features % group == 0 and in_features * self.bits % 32 == 0, f"{in_features} columns, group {group}"
        self.register_buffer("codes", torch.empty(*shape[:2], in_features * self.bits // 32, dtype=torch.int32,
                                                  device=device))  # fmt: skip
        self.steps = nn.Parameter(torch.empty(*shape[:2], in_features // group, **f))
        if not symmetric:
            self.lows = nn.Parameter(torch.empty(*shape[:2], in_features // group, **f))

    def extra_repr(self) -> str:
        fmt = "dense" if self.levels is None else f"levels={self.levels}, group={self.group}"
        return f"{super().extra_repr()}, n_delta={self.n_delta}, {fmt}"

    def delta(self, e: int) -> Tensor:
        r""":math:`\Delta_e`, (R, C), dense or unpacked from its grids."""
        if self.levels is None:
            return self.deltas[e]
        return grid_delta(self.codes[e], self.steps[e], None if self.symmetric else self.lows[e], self.levels,
                          self.bits)  # fmt: skip

    def _decodable(self, x: Tensor) -> bool:
        """Frozen 2-bit grids with a power-of-2 group on CUDA: the loop decodes them in Triton (``_decoded``)."""
        if not (HAS_TRITON and x.is_cuda and self.levels is not None and code_bits(self.levels) == 2):
            return False
        frozen = not any(p.requires_grad for p in (self.steps, getattr(self, "lows", self.steps)))
        return frozen and self.group & (self.group - 1) == 0

    def _decoded(self, e: int) -> Tensor:
        """:math:`\\Delta_e` as ``delta(e)``, decoded in one Triton pass (no gradient)."""
        from ..triton_kernels.decode import decode_grid
        from ..triton_kernels.fused import Delta

        s = self.steps[e : e + 1].detach()
        z = -s * ((self.levels - 1) / 2) if self.symmetric else self.lows[e : e + 1].detach()  # type: ignore[operator]
        return decode_grid(Delta(3, self.codes[e : e + 1], s, z, self.group))[0]

    FUSED_MAX_ENTRIES = 1 << 28  # (pairs, out) the down kernel writes before the slot sum: 512 MiB in bf16

    def _fused(self, x: Tensor, r: Tensor, pairs: int) -> bool:
        if not (HAS_TRITON and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and self.levels is not None):
            return False
        if code_bits(self.levels) != 2 or self.group & (self.group - 1) or self.n_delta == 0:
            return False
        if pairs * self.out_features > self.FUSED_MAX_ENTRIES:  # e.g. lm_head over a whole sequence: the loop
            return False
        # forward-only: a gradient for x, the routers (through r) or any parameter takes the loop
        return not (torch.is_grad_enabled() and (x.requires_grad or r.requires_grad
                                                 or any(p.requires_grad for p in self.parameters())))

    def _fused_deltas(self, x: Tensor, idx: Tensor, r: Tensor) -> Tensor:
        r""":math:`\sum_i r_i \Delta_{e_i} x` per token (``triton_kernels.fused.routed_deltas``)."""
        from ..triton_kernels.fused import Delta, routed_deltas_any

        s = self.steps.detach()
        z = -s * ((self.levels - 1) / 2) if self.symmetric else self.lows.detach()  # type: ignore[operator]
        return routed_deltas_any(x, idx, r, self.n_delta, Delta(3, self.codes, s, z, self.group))

    def forward(self, x: Tensor) -> Tensor:
        """``x @ weight.T`` plus each token's routed deltas.

        Args:
            x: (..., C), T tokens in all.

        Returns:
            (..., R).

        Raises:
            RuntimeError: If ``route`` is unset or not for T tokens.
        """
        y = F.linear(x, self.weight)
        if self.route is None or self.route[0].shape[0] != x.numel() // x.shape[-1]:
            raise RuntimeError(f"RoutedDeltaLinear: no routing for these {x.numel() // x.shape[-1]} tokens")
        idx, r = self.route
        xs, ys = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
        if self._fused(x, r, idx.numel()):
            return (ys + self._fused_deltas(xs, idx, r)).view(y.shape)
        dec = self._decodable(x)
        for e in idx.unique().tolist():
            if e >= self.n_delta:
                continue
            hit = idx == e  # one row per token, its slots on e summed (several slots share a set)
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            w = (r * hit).sum(-1)[rows, None]
            if dec:
                d = self._decoded(e)
                with recompute_saved([d], lambda e=e: [self._decoded(e)]):
                    ys = ys.index_add(0, rows, (w * F.linear(xs[rows], d)).to(ys.dtype))
                del d
            else:
                ys = ys.index_add(0, rows, (w * F.linear(xs[rows], self.delta(e))).to(ys.dtype))
        return ys.view(y.shape)
