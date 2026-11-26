"""Seeded DARE deltas: each Delta_e keeps a random fraction `keep` of its entries (or of its 1 x block blocks along
the input dimension), rescaled by 1/keep; only the kept values are stored. The mask is not stored: it is a function of
(seed, expert, projection, position) through Philox4x32-10, pure 32-bit integer arithmetic, so it regenerates
bit for bit on any device.

Mask spec, for projection j (0 gate, 1 up, 2 down) of delta e, an (R, C) matrix, block size B dividing C::

  unit u = row * (C / B) + col // B,  counter = (u, 3e + j, 0, 0),  key = (seed mod 2^32, seed >> 32)
  kept  <=>  word 0 of Philox4x32-10(counter, key) < round(keep * 2^32)

Kept values are stored row-major, expert by expert, as Delta / keep in the model dtype; p_<proj> (N, R) holds where
each row starts. The fused kernel and the materialised (training) path read the mask as a bitmap (mask_bits: 1 bit per
block, built once per model and kept as a non-persistent buffer) rather than running Philox each time: rows x C / B
bits per delta matrix, e.g. 0.7 MB per Qwen3.5-4B MoE layer at block 64. It is built in __init__ and again after load_state_dict (which also covers a
model made on the meta device), never inside a forward: there it could land in a CUDA-graph capture or a compiled
trace. seed, keep and block are fixed at construction.
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


def mask_bits(seed: int, stream: int, rows: int, cols: int, keep: float, block: int, device: Any = "cpu") -> Tensor:
    """keep_mask as a bitmap: (rows, ceil(cols / block / 32)) int32, block j of a row in bit j % 32 of word j // 32."""
    u = torch.arange(rows * (cols // block), device=device)
    z = torch.zeros_like(u)
    kept = (philox(u, z + stream, z, z, seed & M32, seed >> 32) < threshold(keep)).view(rows, -1)
    words = -(-kept.shape[1] // 32)
    kept = torch.nn.functional.pad(kept, (0, words * 32 - kept.shape[1])).view(rows, words, 32).long()
    w = (kept << torch.arange(32, device=device)).sum(-1)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def pack(
    d: Tensor, seed: int, j: int, keep: float, block: int
) -> tuple[Tensor, Tensor]:
    """Deltas (N, R, C) of projection j -> kept values / keep (flat) and each row's start in them (N, R) int64."""
    vals, counts = [d.new_empty(0)], [torch.zeros(0, dtype=torch.int64, device=d.device)]  # N = 0: empty
    for e in range(d.shape[0]):
        m = keep_mask(seed, 3 * e + j, d.shape[1], d.shape[2], keep, block, d.device)
        vals.append((d[e].float() / keep).to(d.dtype)[m])
        counts.append(m.sum(1))
    c = torch.cat(counts)
    return torch.cat(vals), (c.cumsum(0) - c).view(d.shape[:2])


class DareDeltaMoE(DeltaMoE):
    """DeltaMoE whose deltas are seeded DARE (see the module docstring). The materialised path (training) unpacks
    only the routed expert e from the mask bitmap (delta(e)); d_gate / d_up / d_down unpack every expert on access, so
    the other inherited forwards work too (as references); the fused Triton path reads the packed values directly
    (triton_kernels/fused.py). Build with from_delta_moe."""

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
        self._loaded: set[str] = set()  # projections the current load_state_dict fills (checked after it)
        self._mask_bits()
        self.register_load_state_dict_pre_hook(DareDeltaMoE._resize_packed)
        self.register_load_state_dict_post_hook(DareDeltaMoE._after_load)  # a method: the module pickles

    def _after_load(self, _: Any = None) -> None:
        """Rebuild the bitmaps and check the loaded packing against them: keep, block and seed are not in the state
        dict, so a checkpoint made with others would otherwise load and scatter its values to the wrong places."""
        self._mask_bits()
        loaded, self._loaded = self._loaded, set()
        for n in loaded:  # only what this load filled: strict=False may leave a block for later
            m, v, p = getattr(self, f"m_{n}"), getattr(self, f"v_{n}"), getattr(self, f"p_{n}")
            if m.device.type == "meta" or v.device.type == "meta":
                continue
            c = sum(((m >> b) & 1).sum(-1) for b in range(32)).flatten() * self.block  # kept per row
            if v.numel() != int(c.sum()) or not torch.equal(p.flatten(), c.cumsum(0) - c):
                raise RuntimeError(
                    f"DareDeltaMoE: packed {n} deltas do not match keep={self.keep}, block={self.block}, "
                    f"seed={self.seed} (made with other DARE settings?)"
                )

    def _resize_packed(self, state: dict[str, Tensor], prefix: str, *_: Any) -> None:
        """v_<proj> holds as many values as the mask keeps: size the packed buffers to the incoming ones."""
        for n in PROJ:
            for k in (f"v_{n}", f"p_{n}"):
                if (t := state.get(prefix + k)) is not None:
                    setattr(self, k, getattr(self, k).new_empty(t.shape))
                    self._loaded.add(n)

    def _init_deltas(
        self, n: int, inter: int, hidden: int, factory: dict[str, Any]
    ) -> None:
        # packed buffers start empty: filled by from_delta_moe, or sized by load_state_dict (_resize_packed)
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
            # derived from (seed, keep, block, shape) alone: built by _mask_bits, never saved
            self.register_buffer(f"m_{name}", torch.empty(0, dtype=torch.int32, device=factory["device"]), persistent=False)

    @classmethod
    @torch.no_grad()
    def from_delta_moe(
        cls, m: DeltaMoE, keep: float, block: int = 1, seed: int = 0
    ) -> "DareDeltaMoE":
        """DARE form of a DeltaMoE (same base, router, top-k and zero experts): each delta keeps the fraction `keep`
        of its 1 x `block` blocks chosen by `seed`, rescaled by 1 / keep. A HierDeltaMoE is refused."""
        from .hier import HierDeltaMoE

        if isinstance(m, HierDeltaMoE):
            raise TypeError("DareDeltaMoE has a flat router; a HierDeltaMoE cannot be converted")
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

    def _bits(self, name: str) -> Tensor:
        m = getattr(self, f"m_{name}")
        if m.ndim != 3:  # e.g. made on meta, then to_empty, no load
            raise RuntimeError("DareDeltaMoE mask bitmaps not built: call load_state_dict or _mask_bits() first")
        return m

    def _unpack(self, name: str, e: int) -> Tensor:
        """Dense Delta_e of one projection from the bitmap (no Philox); differentiable in v_<name>."""
        _, R, C = self.shapes[name]
        w = self._bits(name)[e]  # (R, words)
        bits = (w[..., None] >> torch.arange(32, dtype=torch.int32, device=w.device)) & 1
        keep = bits.view(R, -1)[:, : C // self.block].bool().repeat_interleave(self.block, 1)
        v = getattr(self, f"v_{name}")
        return v.new_zeros(R, C).masked_scatter(keep, v[int(getattr(self, f"p_{name}")[e, 0]) :])

    def delta(self, e: int) -> tuple[Tensor, Tensor, Tensor]:
        return self._unpack("gate", e), self._unpack("up", e), self._unpack("down", e)

    def _delta(self, j: int) -> Tensor:
        name = PROJ[j]
        G, R, C = self.shapes[name]
        if G == 0:
            return getattr(self, f"v_{name}").new_zeros(G, R, C)
        return torch.stack([self._unpack(name, e) for e in range(G)])

    @property
    def d_gate(self) -> Tensor:  # type: ignore[override]
        return self._delta(0)

    @property
    def d_up(self) -> Tensor:  # type: ignore[override]
        return self._delta(1)

    @property
    def d_down(self) -> Tensor:  # type: ignore[override]
        return self._delta(2)

    def _use_triton(self, x: Tensor) -> bool:
        # the kernel's K tile and the block must divide one another; tiles are powers of 2. Forward only: the Triton
        # backward takes dense deltas (DARE trains through the materialised path)
        return self.block & (self.block - 1) == 0 and not self._needs_grad(x) and super()._use_triton(x)

    @torch.no_grad()
    def _mask_bits(self) -> None:
        """(Re)build m_<proj> (G, R, words) with mask_bits on the buffers' device (skipped on meta: built after
        load_state_dict)."""
        for j, n in enumerate(PROJ):
            dev = getattr(self, f"v_{n}").device
            if dev.type == "meta":
                continue
            G, R, C = self.shapes[n]
            bits = [mask_bits(self.seed, 3 * e + j, R, C, self.keep, self.block, dev) for e in range(G)]
            setattr(self, f"m_{n}", torch.stack(bits) if bits else torch.zeros(0, R, 1, dtype=torch.int32, device=dev))

    def _kernel_deltas(self) -> "tuple[Delta, Delta, Delta]":
        from ..triton_kernels import Delta

        return tuple(  # type: ignore[return-value]
            Delta(1, getattr(self, f"v_{n}"), getattr(self, f"p_{n}"), self._bits(n), self.block) for n in PROJ
        )
