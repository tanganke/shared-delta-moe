r"""Seeded DARE deltas: each :math:`\Delta_e` keeps a random fraction :math:`\mathrm{keep}` of its entries (or of its
:math:`1 \times B` blocks along the input dimension), rescaled by :math:`1 / \mathrm{keep}`; only the kept values are
stored. The mask is not stored: it is a function of (seed, expert, projection, position) through Philox4x32-10, pure
32-bit integer arithmetic, so it regenerates bit for bit on any device.

Mask spec, for projection :math:`j` (0 gate, 1 up, 2 down) of delta :math:`e`, an (R, C) matrix, block size :math:`B`
dividing :math:`C`::

  unit u = row * (C / B) + col // B,  counter = (u, 3e + j, 0, 0),  key = (seed mod 2^32, seed >> 32)
  kept  <=>  word 0 of Philox4x32-10(counter, key) < round(keep * 2^32)

Kept values are stored row-major, expert by expert, as :math:`\Delta / \mathrm{keep}` in the model dtype; ``p_<proj>``
(N, R) holds where each row starts. The fused kernel and the materialised (training) path read the mask as a bitmap
(``mask_bits``: 1 bit per block, built once per model and kept as a non-persistent buffer) rather than running Philox
each time: :math:`R \cdot C / B` bits per delta matrix, e.g. 0.7 MB per Qwen3.5-4B MoE layer at block 64. It is built in
``__init__`` and again after ``load_state_dict`` (which also covers a model made on the meta device), never inside a
forward: there it could land in a CUDA-graph capture or a compiled trace. ``seed``, ``keep`` and ``block`` are fixed
at construction.
"""

from typing import Any

import torch
from torch import Tensor

from ..ops import pack_bits, unpack_bits
from .delta_moe import DeltaMoE
from .packed import PROJ, PackedDeltaMoE

M32 = 0xFFFFFFFF


def _mulhilo(a: int, b: Tensor) -> tuple[Tensor, Tensor]:
    r"""(a * b) >> 32 and :math:`(a b) \bmod 2^{32}` for :math:`a < 2^{32}` and :math:`b \in [0, 2^{32})` held in int64,
    without int64 overflow."""
    lo, mid = a * (b & 0xFFFF), a * (b >> 16)  # a * b = mid * 2^16 + lo, each < 2^48
    return (mid + (lo >> 16)) >> 16, (((mid & 0xFFFF) << 16) + lo) & M32


def philox(
    c0: Tensor, c1: Tensor, c2: Tensor, c3: Tensor, k0: int, k1: int, rounds: int = 10
) -> Tensor:
    r"""Word 0 of Philox4x32 (Salmon et al. 2011; the constants and round of Random123 and Triton).

    Pure 32-bit integer arithmetic carried in int64, so the result is bit-identical on every device.

    Args:
        c0: Counter word 0, int64 tensor with values in :math:`[0, 2^{32})`.
        c1: Counter word 1, int64, same shape as c0.
        c2: Counter word 2, int64, same shape as c0.
        c3: Counter word 3, int64, same shape as c0.
        k0: Key word 0, an int in :math:`[0, 2^{32})`.
        k1: Key word 1, an int in :math:`[0, 2^{32})`.
        rounds: Number of Philox rounds (10: Philox4x32-10).

    Returns:
        Word 0 of the output, int64 with values in :math:`[0, 2^{32})`, the shape of c0.
    """
    for _ in range(rounds):
        hb, lb = _mulhilo(0xCD9E8D57, c2)
        ha, la = _mulhilo(0xD2511F53, c0)
        c0, c1, c2, c3 = hb ^ c1 ^ k0, lb, ha ^ c3 ^ k1, la
        k0, k1 = (k0 + 0x9E3779B9) & M32, (k1 + 0xBB67AE85) & M32
    return c0


def threshold(keep: float) -> int:
    r"""Philox threshold of a keep fraction.

    Args:
        keep: Fraction of units kept, in [0, 1].

    Returns:
        :math:`\mathrm{round}(\mathrm{keep} \cdot 2^{32})`: a unit is kept where Philox word 0 is below it.
    """
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
    r"""Keep mask of one delta matrix (see the module docstring for the spec).

    Args:
        seed: DARE seed, in :math:`[0, 2^{63})`.
        stream: :math:`3 e + j` for expert :math:`e` and projection :math:`j` (0 gate, 1 up, 2 down).
        rows: R, rows of the delta matrix.
        cols: C, columns of the delta matrix; a multiple of block.
        keep: Fraction of blocks kept.
        block: B, entries per :math:`1 \times B` block along a row.
        device: Device the mask is computed on.

    Returns:
        (rows, cols) bool, True where the entry is kept (constant over each block).
    """
    assert cols % block == 0 and rows * (cols // block) <= 2**32 and 0 <= seed < 2**63
    u = torch.arange(rows * (cols // block), device=device)
    z = torch.zeros_like(u)
    kept = philox(u, z + stream, z, z, seed & M32, seed >> 32) < threshold(keep)
    return kept.view(rows, -1).repeat_interleave(block, 1)


def mask_bits(seed: int, stream: int, rows: int, cols: int, keep: float, block: int, device: Any = "cpu") -> Tensor:
    r"""``keep_mask`` as a bitmap, 1 bit per block.

    Args:
        seed: DARE seed.
        stream: :math:`3 e + j` for expert :math:`e` and projection :math:`j` (0 gate, 1 up, 2 down).
        rows: R, rows of the delta matrix.
        cols: C, columns of the delta matrix; a multiple of block.
        keep: Fraction of blocks kept.
        block: B, entries per :math:`1 \times B` block along a row.
        device: Device the bitmap is computed on.

    Returns:
        (rows, ceil(cols / block / 32)) int32: block :math:`j` of a row in bit :math:`j \bmod 32` of word
        :math:`\lfloor j / 32 \rfloor`, set where kept.
    """
    u = torch.arange(rows * (cols // block), device=device)
    z = torch.zeros_like(u)
    return pack_bits((philox(u, z + stream, z, z, seed & M32, seed >> 32) < threshold(keep)).view(rows, -1))


def pack(
    d: Tensor, seed: int, j: int, keep: float, block: int
) -> tuple[Tensor, Tensor]:
    r"""Pack the deltas of one projection into their kept values.

    Args:
        d: (N, R, C) deltas of projection j, one per delta expert.
        seed: DARE seed.
        j: Projection index (0 gate, 1 up, 2 down).
        keep: Fraction of blocks kept.
        block: B, entries per :math:`1 \times B` block along a row.

    Returns:
        A pair (v, p): v the kept values :math:`\Delta / \mathrm{keep}`, flat, row-major expert by expert, in d's
        dtype; p (N, R) int64, each row's start in v.
    """
    vals, counts = [d.new_empty(0)], [torch.zeros(0, dtype=torch.int64, device=d.device)]  # N = 0: empty
    for e in range(d.shape[0]):
        m = keep_mask(seed, 3 * e + j, d.shape[1], d.shape[2], keep, block, d.device)
        vals.append((d[e].float() / keep).to(d.dtype)[m])
        counts.append(m.sum(1))
    c = torch.cat(counts)
    return torch.cat(vals), (c.cumsum(0) - c).view(d.shape[:2])


class DareDeltaMoE(PackedDeltaMoE):
    r"""DeltaMoE whose deltas are seeded DARE (see the module docstring).

    The materialised path (training) unpacks only the routed expert :math:`e` from the mask bitmap (``delta(e)``);
    ``d_gate`` / ``d_up`` / ``d_down`` unpack every expert on access, so the other inherited forwards work too (as
    references); the fused Triton path reads the packed values directly (``triton_kernels/fused.py``). Build with
    ``from_delta_moe``; the constructor leaves the packed buffers empty (``load_state_dict`` sizes them to the
    checkpoint's).

    Args:
        hidden: H, model hidden size.
        inter: I, expert intermediate size.
        n_delta: :math:`N`, number of delta experts.
        keep: Fraction of each delta's blocks kept.
        block: B, entries per :math:`1 \times B` block along the input dimension; must divide each projection's
            columns.
        seed: DARE seed, in :math:`[0, 2^{63})`.
        top_k: Experts per token.
        num_zero_experts: :math:`Z`, router rows that route to the base alone.
        device: Device of the parameters and buffers.
        dtype: Dtype of the parameters and packed values.
    """

    # forward_matrix up to here (H200, hidden 2560, inter 9216, 5 deltas, keep 0.2 / block 64: 2-3x faster than
    # forward_param at 1-192 pairs, 1.8x slower at 512)
    MATRIX_FORWARD_PAIRS = 256

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
            c = unpack_bits(m).sum(-1).flatten() * self.block  # kept per row
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
        r"""DARE form of a DeltaMoE, with the same base, router, top-k and zero experts.

        Each delta keeps the fraction ``keep`` of its :math:`1 \times B` blocks (:math:`B` = ``block``) chosen by
        ``seed``, rescaled by :math:`1 / \mathrm{keep}`.

        Args:
            m: The DeltaMoE to convert (not a HierDeltaMoE).
            keep: Fraction of each delta's blocks kept.
            block: B, entries per :math:`1 \times B` block along the input dimension.
            seed: DARE seed, in :math:`[0, 2^{63})`.

        Returns:
            The DareDeltaMoE, on m's device and in m's dtype.

        Raises:
            TypeError: If m is a HierDeltaMoE (DareDeltaMoE has a flat router).
        """
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
        out._copy_shared(m)
        for j, name in enumerate(PROJ):
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
        r"""Dense :math:`\Delta_e` of one projection from the bitmap (no Philox); differentiable in ``v_<name>``."""
        _, R, C = self.shapes[name]
        keep = unpack_bits(self._bits(name)[e])[:, : C // self.block].bool().repeat_interleave(self.block, 1)
        v = getattr(self, f"v_{name}")
        return v.new_zeros(R, C).masked_scatter(keep, v[int(getattr(self, f"p_{name}")[e, 0]) :])

    def _use_triton(self, x: Tensor) -> bool:
        # the kernel's K tile and the block must divide one another; tiles are powers of 2
        return self.block & (self.block - 1) == 0 and super()._use_triton(x)

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
