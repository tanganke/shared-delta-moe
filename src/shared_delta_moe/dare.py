"""Seeded DARE deltas: each Delta_e keeps a random fraction `keep` of its entries (or of its 1 x block blocks along
the input dimension), rescaled by 1/keep; only the kept values are stored. The mask is not stored: it is a function of
(seed, expert, projection, position) through Philox4x32-10, pure 32-bit integer arithmetic, so the packer (any
device, torch) and the fused kernel (Triton tl.randint) regenerate the same bits on any hardware.

Mask spec, for projection j (0 gate, 1 up, 2 down) of delta e, an (R, C) matrix, block size B dividing C:
  unit u = row * (C / B) + col // B,  counter = (u, 3e + j, 0, 0),  key = (seed mod 2^32, seed >> 32)
  kept  <=>  word 0 of Philox4x32-10(counter, key) < round(keep * 2^32)
Kept values are stored row-major, expert by expert, as Delta / keep in the model dtype; p_<proj> (N, R) holds where
each row starts.
"""

from typing import Any

import torch
from torch import Tensor

from .delta_moe import DeltaMoE

PROJ = ("gate", "up", "down")
M32 = 0xFFFFFFFF


def _mulhilo(a: int, b: Tensor) -> tuple[Tensor, Tensor]:
    """(a * b) >> 32 and (a * b) mod 2^32 for a < 2^32 and b in [0, 2^32) held in int64, without int64 overflow."""
    lo, mid = a * (b & 0xFFFF), a * (b >> 16)  # a * b = mid * 2^16 + lo, each < 2^48
    return (mid + (lo >> 16)) >> 16, (((mid & 0xFFFF) << 16) + lo) & M32


def philox(
    c0: Tensor, c1: Tensor, c2: Tensor, c3: Tensor, k0: int, k1: int, rounds: int = 10
) -> Tensor:
    """Word 0 of Philox4x32 (Salmon et al. 2011; the constants and round of Random123 and Triton), counters as int64
    tensors in [0, 2^32)."""
    for _ in range(rounds):
        hb, lb = _mulhilo(0xCD9E8D57, c2)
        ha, la = _mulhilo(0xD2511F53, c0)
        c0, c1, c2, c3 = hb ^ c1 ^ k0, lb, ha ^ c3 ^ k1, la
        k0, k1 = (k0 + 0x9E3779B9) & M32, (k1 + 0xBB67AE85) & M32
    return c0


def threshold(keep: float) -> int:
    return round(keep * 2**32)


def keep_mask(
    seed: int,
    stream: int,
    rows: int,
    cols: int,
    keep: float,
    block: int,
    device: Any = "cpu",
) -> Tensor:
    """(rows, cols) bool mask of one delta matrix; stream = 3 * expert + projection."""
    assert cols % block == 0 and rows * (cols // block) <= 2**32 and 0 <= seed < 2**63
    u = torch.arange(rows * (cols // block), device=device)
    z = torch.zeros_like(u)
    kept = philox(u, z + stream, z, z, seed & M32, seed >> 32) < threshold(keep)
    return kept.view(rows, -1).repeat_interleave(block, 1)


def pack(
    d: Tensor, seed: int, j: int, keep: float, block: int
) -> tuple[Tensor, Tensor]:
    """Deltas (N, R, C) of projection j -> kept values / keep (flat) and each row's start in them (N, R) int64."""
    vals, counts = [], []
    for e in range(d.shape[0]):
        m = keep_mask(seed, 3 * e + j, d.shape[1], d.shape[2], keep, block, d.device)
        vals.append((d[e].float() / keep).to(d.dtype)[m])
        counts.append(m.sum(1))
    c = torch.cat(counts)
    return torch.cat(vals), (c.cumsum(0) - c).view(d.shape[:2])


def unpack(
    v: Tensor, seed: int, j: int, shape: tuple[int, int, int], keep: float, block: int
) -> Tensor:
    d = torch.zeros(shape, dtype=v.dtype, device=v.device)
    i = 0
    for e in range(shape[0]):
        m = keep_mask(seed, 3 * e + j, shape[1], shape[2], keep, block, v.device)
        k = int(m.sum())
        d[e][m] = v[i : i + k]
        i += k
    assert i == v.numel()
    return d


class DareDeltaMoE(DeltaMoE):
    """DeltaMoE whose deltas are seeded DARE (see the module docstring). d_gate / d_up / d_down unpack to dense
    tensors on access, so every inherited forward works (as a slow reference); the fused Triton path reads the packed
    values directly (triton_kernels.dare_gmm). Build with from_delta_moe."""

    def __init__(
        self,
        hidden: int,
        inter: int,
        n_delta: int,
        keep: float,
        block: int = 1,
        seed: int = 0,
        top_k: int = 1,
        num_zero_experts: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden, inter, n_delta, top_k, num_zero_experts, device, dtype)
        self.keep, self.block, self.seed = keep, block, seed

    def _init_deltas(
        self, n: int, inter: int, hidden: int, factory: dict[str, Any]
    ) -> None:
        # ponytail: buffers start empty, so load_state_dict needs them resized first; fill with from_delta_moe
        self.shapes = {
            "gate": (n, inter, hidden),
            "up": (n, inter, hidden),
            "down": (n, hidden, inter),
        }
        for name in PROJ:
            self.register_buffer(f"v_{name}", torch.empty(0, **factory))
            self.register_buffer(
                f"p_{name}", torch.empty(0, dtype=torch.int64, device=factory["device"])
            )

    @classmethod
    @torch.no_grad()
    def from_delta_moe(
        cls, m: DeltaMoE, keep: float, block: int = 1, seed: int = 0
    ) -> "DareDeltaMoE":
        g = m.gate
        out = cls(
            g.shape[1],
            g.shape[0],
            m.n_delta,
            keep,
            block,
            seed,
            m.top_k,
            m.num_zero_experts,
            g.device,
            g.dtype,
        )
        out.router.weight.copy_(m.router.weight)
        for j, name in enumerate(PROJ):
            getattr(out, name).copy_(getattr(m, name))
            v, p = pack(getattr(m, f"d_{name}"), seed, j, keep, block)
            setattr(out, f"v_{name}", v)
            setattr(out, f"p_{name}", p)
        return out

    def _delta(self, j: int) -> Tensor:
        name = PROJ[j]
        return unpack(
            getattr(self, f"v_{name}"),
            self.seed,
            j,
            self.shapes[name],
            self.keep,
            self.block,
        )

    @property
    def d_gate(self) -> Tensor:  # type: ignore[override]
        return self._delta(0)

    @property
    def d_up(self) -> Tensor:  # type: ignore[override]
        return self._delta(1)

    @property
    def d_down(self) -> Tensor:  # type: ignore[override]
        return self._delta(2)

    def _fused_gmm(
        self, x: Tensor, j: int, starts: Tensor, ends: Tensor, max_rows: int
    ) -> Tensor:
        from .triton_kernels import dare_gmm

        name = PROJ[j]
        v, p = getattr(self, f"v_{name}"), getattr(self, f"p_{name}")
        return dare_gmm(
            x,
            getattr(self, name),
            v,
            p,
            self.seed,
            j,
            self.keep,
            self.block,
            starts,
            ends,
            max_rows,
        )
