"""BitDelta deltas (Liu et al. 2024, "BitDelta: Your Fine-Tune May Only Be Worth One Bit"): each Delta_e is stored as
its sign, 1 bit per entry, times a scale: Delta_e ~ diag(s_e) sign(Delta_e). BitDelta takes one scale per matrix,
mean |Delta_e| (for a fixed sign matrix, the scalar closest to Delta_e in Frobenius norm); scale="row" takes one per
output row (that row's mean |Delta_e|: R values per matrix, closer). The scales are nn.Parameters, so they can be
trained afterwards (BitDelta's scale distillation) through the materialised forward; the signs are fixed.

Layout, projection (R, C) of delta e, C a multiple of 32: b_<proj> (N, R, C / 32) int32, entry c of a row in bit
c % 32 of word c // 32, set = +s, clear = -s (a zero entry counts as +); s_<proj> (N, R) in the model dtype.
Storage per delta: 1/16 of bf16, plus the scales.
"""

from typing import Any

import torch
from torch import Tensor, nn

from .delta_moe import DeltaMoE

PROJ = ("gate", "up", "down")


def pack_signs(d: Tensor) -> Tensor:
    """(..., C) -> (..., C / 32) int32: bit c % 32 of word c // 32 set where d[..., c] >= 0."""
    assert d.shape[-1] % 32 == 0
    b = (d >= 0).view(*d.shape[:-1], d.shape[-1] // 32, 32).long() << torch.arange(32, device=d.device)
    w = b.sum(-1)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def unpack_signs(w: Tensor, dtype: torch.dtype) -> Tensor:
    """pack_signs back to +-1 in dtype, (..., C)."""
    bits = (w[..., None] >> torch.arange(32, dtype=torch.int32, device=w.device)) & 1
    return (bits * 2 - 1).to(dtype).flatten(-2)


def scales(d: Tensor, scale: str) -> Tensor:
    """(N, R, C) deltas -> (N, R) scales: mean |Delta| per matrix ("matrix", BitDelta) or per row ("row")."""
    assert scale in ("matrix", "row"), scale
    a = d.float().abs()
    s = a.mean(-1) if scale == "row" else a.mean((-2, -1))[:, None].expand(d.shape[:2])
    return s.to(d.dtype)


class BitDeltaMoE(DeltaMoE):
    """DeltaMoE whose deltas are BitDelta (see the module docstring). The materialised path unpacks only the routed
    expert (delta(e)); d_gate / d_up / d_down unpack every expert on access, so the other inherited forwards work
    too (as references); the fused Triton path reads the sign bits directly (triton_kernels/fused.py, delta mode 2).
    Build with from_delta_moe."""

    def _init_deltas(self, n: int, inter: int, hidden: int, factory: dict[str, Any]) -> None:
        assert inter % 32 == 0 and hidden % 32 == 0, "sign bits are packed 32 to an int32 word along each row"
        self.shapes = {"gate": (n, inter, hidden), "up": (n, inter, hidden), "down": (n, hidden, inter)}
        for name, (G, R, C) in self.shapes.items():
            self.register_buffer(f"b_{name}", torch.empty(G, R, C // 32, dtype=torch.int32, device=factory["device"]))
            setattr(self, f"s_{name}", nn.Parameter(torch.empty(G, R, **factory)))

    @classmethod
    @torch.no_grad()
    def from_delta_moe(cls, m: DeltaMoE, scale: str = "matrix") -> "BitDeltaMoE":
        g = m.gate
        out = cls(g.shape[1], g.shape[0], m.n_delta, m.top_k, m.num_zero_experts, g.device, g.dtype)
        out.router.weight.copy_(m.router.weight)
        for name in PROJ:
            getattr(out, name).copy_(getattr(m, name))
            d = getattr(m, f"d_{name}")
            getattr(out, f"b_{name}").copy_(pack_signs(d))
            getattr(out, f"s_{name}").copy_(scales(d, scale))
        return out

    def _unpack(self, name: str, e: int) -> Tensor:
        """Dense diag(s_e) sign(Delta_e) of one projection; differentiable in the scales."""
        s = getattr(self, f"s_{name}")
        return s[e, :, None] * unpack_signs(getattr(self, f"b_{name}")[e], s.dtype)

    def delta(self, e: int) -> tuple[Tensor, Tensor, Tensor]:
        return self._unpack("gate", e), self._unpack("up", e), self._unpack("down", e)

    def _delta(self, name: str) -> Tensor:
        G, R, C = self.shapes[name]
        if G == 0:
            return getattr(self, f"s_{name}").new_zeros(G, R, C)
        return torch.stack([self._unpack(name, e) for e in range(G)])

    @property
    def d_gate(self) -> Tensor:  # type: ignore[override]
        return self._delta("gate")

    @property
    def d_up(self) -> Tensor:  # type: ignore[override]
        return self._delta("up")

    @property
    def d_down(self) -> Tensor:  # type: ignore[override]
        return self._delta("down")

    def _use_triton(self, x: Tensor) -> bool:
        # the fused kernels take sign bits forward-only (triton_train's backward: dense deltas); with a gradient
        # needed, the materialised path
        return not self._needs_grad(x) and super()._use_triton(x)

    def _kernel_deltas(self) -> "tuple[Delta, Delta, Delta]":
        from ..triton_kernels import Delta

        return tuple(Delta(2, getattr(self, f"b_{n}"), getattr(self, f"s_{n}"), block=32) for n in PROJ)  # type: ignore[return-value]
