r"""Delta compression methods, simulated: each maps a dense delta :math:`\Delta` (R, C) to its compressed-then-restored
approximation :math:`\hat\Delta`, dense, and reports the storage it would take in bits per entry (dense bf16 = 16).
They are for measuring accuracy before a format gets packed storage and kernels (as DARE and BitDelta have,
``models/dare.py``, ``models/bitdelta.py``); results are in ``docs/guide/delta-compression.md``.

A method is a callable ``(delta, gram) -> (approx, bits)``. ``delta`` is one projection's delta in float32, the input
dimension last; ``gram`` is :math:`X^\top X` (C, C) of that projection's inputs on the tokens routed to the expert
(``calibration_grams``), used by the calibrated methods (GPTQ, SparseGPT, Wanda, importance DARE) and ignored by the
others. ``compress_deltas`` applies a method to every delta of a ``DeltaMoE`` or ``HierDeltaMoE`` in place.
"""

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .models.delta_moe import DeltaMoE
from .models.packed import PackedDeltaMoE

Method = Callable[[Tensor, Tensor | None], tuple[Tensor, float]]


def _cholesky_inverse(delta: Tensor, gram: Tensor, damp: float) -> tuple[Tensor, Tensor]:
    """GPTQ / SparseGPT setup: dead inputs zeroed, the damped Gram's inverse as its upper Cholesky factor."""
    w, h = delta.clone(), gram.clone()
    dead = h.diag() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h += damp * h.diag().mean() * torch.eye(len(h), device=h.device, dtype=h.dtype)
    return w, torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)


CLIP = torch.linspace(1.0, 0.1, 19).tolist()  # range shrink factors tried by ``clip=True``


def _grid(x: Tensor, levels: int, symmetric: bool, clip: bool) -> tuple[Tensor, Tensor]:
    """A grid of ``levels`` evenly spaced values per row of the last dim: (lowest level, step). Asymmetric: spans
    [min, max]; symmetric: [-max|x|, max|x|]. With ``clip``, the span is shrunk towards 0 by the factor in ``CLIP``
    with the least squared rounding error, per group."""
    if symmetric:
        a = x.abs().amax(-1, keepdim=True)
        lo, hi = -a, a
    else:
        lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    best = None
    for f in CLIP if clip else (1.0,):
        l, step = lo * f, ((hi - lo) * f).clamp_min(1e-12) / (levels - 1)
        if not clip:
            return l, step
        err = (_snap(x, l, step, levels) - x).square().sum(-1, keepdim=True)
        if best is None:
            best = (err, l, step)
        else:
            better = err < best[0]
            best = (torch.where(better, err, best[0]), torch.where(better, l, best[1]), torch.where(better, step, best[2]))
    return best[1], best[2]


def _snap(x: Tensor, lo: Tensor, step: Tensor, levels: int) -> Tensor:
    return ((x - lo) / step).round().clamp(0, levels - 1) * step + lo


def _grid_bits(levels: int, group: int, symmetric: bool) -> float:
    """Code bits (log2 levels, e.g. ternary 1.58: 5 values per byte would be 1.6) plus fp16 scale (and offset)."""
    return math.log2(levels) + (16 if symmetric else 32) / group


def svd(rank: int) -> Method:
    r"""Truncated SVD, :math:`\hat\Delta = U_r \Sigma_r V_r^\top`, factors in bf16.

    Args:
        rank: Kept singular values.

    Returns:
        The method; bits per entry :math:`16 r (R + C) / (R C)`.
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        u, s, v = torch.linalg.svd(delta, full_matrices=False)
        r, c = delta.shape
        return (u[:, :rank] * s[:rank]) @ v[:rank], 16 * rank * (r + c) / (r * c)

    return f


def quantize(delta: Tensor, gram: Tensor | None, levels: int, group: int = 128, symmetric: bool = False,
             clip: bool = False, damp: float = 0.01, block: int = 128) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
    r"""Group grid quantisation of one delta: codes :math:`q \in \{0, \dots, L - 1\}` and per group of ``group``
    entries along the input dim a lowest level :math:`\ell` and step :math:`s`, :math:`\hat\delta = \ell + s q`.
    Round to nearest without ``gram`` (``rtn``), GPTQ with it (``gptq``).

    Args:
        delta: (R, C) delta.
        gram: (C, C) input Gram for GPTQ, or None for round to nearest.
        levels: :math:`L`, grid values.
        group: Group size; divides C.
        symmetric: Grids centred on 0, :math:`\ell = -s (L - 1) / 2`.
        clip: Shrink each group's span to the factor in ``CLIP`` with the least rounding error.
        damp: GPTQ diagonal damping, a fraction of the Gram's mean diagonal.
        block: GPTQ columns per lazy-update block.

    Returns:
        (codes (R, C) uint8, :math:`\ell` (R, C / group), :math:`s` (R, C / group)), float32 grids.
    """
    assert levels >= 2, f"a grid needs at least 2 levels, got {levels}"
    delta = delta.float()  # GPTQ's updates in float32 whatever the model dtype
    r, c = delta.shape
    assert c % group == 0, f"group {group} does not divide {c}"
    if gram is None:
        x = delta.reshape(r, c // group, group)
        lo, step = _grid(x, levels, symmetric, clip)
        q = ((x - lo) / step).round().clamp(0, levels - 1)
        return q.reshape(r, c).to(torch.uint8), lo[..., 0], step[..., 0]
    w, hinv = _cholesky_inverse(delta, gram, damp)
    q = torch.zeros(r, c, dtype=torch.uint8, device=w.device)
    los, steps = w.new_empty(r, c // group), w.new_empty(r, c // group)
    lo = step = None
    for i1 in range(0, c, block):
        i2 = min(i1 + block, c)
        w1, err, hi = w[:, i1:i2].clone(), torch.zeros(r, i2 - i1, device=w.device), hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            k = i1 + j
            if k % group == 0:  # this group's grid, from its current (error-updated) values
                lo, step = _grid(torch.cat([w1[:, j:j + group], w[:, i2:k + group]], 1), levels, symmetric, clip)
                los[:, k // group], steps[:, k // group] = lo[:, 0], step[:, 0]
            col = w1[:, j:j + 1]
            code = ((col - lo) / step).round().clamp(0, levels - 1)
            q[:, k:k + 1] = code.to(torch.uint8)
            e = (col - (code * step + lo)) / hi[j, j]
            w1[:, j:] -= e @ hi[j:j + 1, j:]
            err[:, j:j + 1] = e
        w[:, i2:] -= err @ hinv[i1:i2, i2:]
    return q, los, steps


def dequantize(codes: Tensor, lo: Tensor, step: Tensor) -> Tensor:
    """``quantize``'s approximation: :math:`\\ell + s q` per group, float32 (R, C)."""
    r, c = codes.shape
    g = c // lo.shape[1]
    return (codes.view(r, -1, g).float() * step[..., None] + lo[..., None]).view(r, c)


def rtn(bits: int = 2, group: int = 128, levels: int | None = None, symmetric: bool = False,
        clip: bool = False) -> Method:  # fmt: skip
    r"""Round to nearest on a grid of ``levels`` values per group of ``group`` entries along the input dim, its scale
    (and offset) in fp16 (``quantize`` without a Gram).

    Args:
        bits: Bits per value: ``levels`` defaults to ``2 ** bits``.
        group: Group size; divides C.
        levels: Grid values, e.g. 3 for ternary.
        symmetric: Grid symmetric about 0 (one scale per group, no offset) instead of spanning [min, max].
        clip: Shrink each group's span to the factor in ``CLIP`` with the least rounding error.

    Returns:
        The method; bits per entry :math:`\log_2` ``levels`` plus 32 / ``group`` (16 / ``group`` symmetric).
    """
    n = 2**bits if levels is None else levels
    assert n >= 2, f"a grid needs at least 2 levels, got {n}"

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        return dequantize(*quantize(delta, None, n, group, symmetric, clip)), _grid_bits(n, group, symmetric)

    return f


def gptq(bits: int = 2, group: int = 128, levels: int | None = None, symmetric: bool = False, clip: bool = False,
         damp: float = 0.01, block: int = 128) -> Method:  # fmt: skip
    r"""GPTQ (Frantar et al., 2023) on the delta: columns quantised in order on ``rtn``'s grid (each group's grid
    fixed when its first column is reached), each one's error, weighted by the inverse Gram, pushed onto the columns
    not yet quantised (``quantize`` with a Gram). Minimises :math:`\|X(\hat\Delta - \Delta)^\top\|_F` greedily.

    Args:
        bits: Bits per value: ``levels`` defaults to ``2 ** bits``.
        group: Group size; divides C.
        levels: Grid values, e.g. 3 for ternary.
        symmetric: Symmetric grid, as ``rtn``.
        clip: Clipped span, as ``rtn``.
        damp: Diagonal damping, a fraction of the Gram's mean diagonal.
        block: Columns per lazy-update block.

    Returns:
        The method (needs ``gram``); bits per entry as ``rtn``.
    """
    n = 2**bits if levels is None else levels
    assert n >= 2, f"a grid needs at least 2 levels, got {n}"

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        assert gram is not None, "gptq needs the inputs' Gram"
        return dequantize(*quantize(delta, gram, n, group, symmetric, clip, damp, block)), _grid_bits(n, group, symmetric)

    return f


def hadamard_rotation(c: int, block: int = 512, seed: int = 0, device: torch.device | str | None = None,
                      dtype: torch.dtype = torch.float32) -> Tensor:  # fmt: skip
    r"""A random orthogonal (C, C) matrix :math:`R = \mathrm{diag}(\sigma)\, \mathrm{blockdiag}(H_B, \dots, H_B)`:
    random signs :math:`\sigma` (seeded) times Sylvester-Hadamard blocks :math:`H_B / \sqrt{B}`. :math:`x R` costs
    :math:`O(C \log B)` per token with a fast Hadamard transform.

    Args:
        c: Size; a multiple of ``block``.
        block: Hadamard block size, a power of 2.
        seed: Seed of the signs.
        device: Device of the result.
        dtype: Dtype of the result.

    Returns:
        :math:`R`, with :math:`R R^\top = I`.
    """
    assert block & (block - 1) == 0 and c % block == 0, f"size {c} is not a multiple of the power of 2 {block}"
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < block:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    sign = torch.randint(0, 2, (c,), generator=torch.Generator().manual_seed(seed)).to(device=device, dtype=dtype) * 2 - 1
    return sign[:, None] * torch.block_diag(*[h / block**0.5] * (c // block))


def rotated(method: Method, block: int = 512, seed: int = 0) -> Method:
    r"""Incoherence processing (QuIP, Chee et al., 2023): compress :math:`\Delta R` with the Gram
    :math:`R^\top \mathbf{H} R` of the rotated inputs :math:`X R`, and restore :math:`\hat\Delta = \widehat{\Delta R}\,
    R^\top`. The output :math:`X \Delta^\top = (XR)(\Delta R)^\top` is unchanged; the rotation spreads each row's
    large entries over the block, so a coarse grid fits it better. A kernel stores :math:`\widehat{\Delta R}` and
    rotates the input (``hadamard_rotation``); the seed is the only extra storage.

    Args:
        method: The compression of the rotated delta.
        block: Hadamard block size; divides C.
        seed: Seed of the random signs.

    Returns:
        The method; bits per entry as ``method``.
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        r = hadamard_rotation(delta.shape[1], block, seed, delta.device, delta.dtype)
        approx, bits = method(delta @ r, None if gram is None else r.T @ gram @ r)
        return approx @ r.T, bits

    return f


def residual_sign(planes: int) -> Method:
    r"""Residual binarisation: :math:`\hat\Delta = \sum_k \mathrm{diag}(a_k)\,\mathrm{sign}(R_k)`, :math:`R_1 = \Delta`,
    :math:`R_{k+1} = R_k - \mathrm{diag}(a_k)\,\mathrm{sign}(R_k)`, :math:`a_k` the row means of :math:`|R_k|`.
    One plane is BitDelta with row scales.

    Args:
        planes: Sign planes.

    Returns:
        The method; bits per entry ``planes`` plus the row scales.
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        approx, res = torch.zeros_like(delta), delta
        for _ in range(planes):
            p = res.sign() * res.abs().mean(1, keepdim=True)
            approx, res = approx + p, res - p
        return approx, planes + 16 * planes / delta.shape[1]

    return f


def magnitude(keep: float) -> Method:
    """Keep the ``keep`` largest entries by magnitude (TIES-style trimming), no rescaling.

    Args:
        keep: Kept fraction.

    Returns:
        The method; bits per entry: values plus a bitmap, ``16 keep + 1``.
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        return delta * _top(delta.abs(), keep), 16 * keep + 1

    return f


def dare(keep: float, block: int = 1, generator: torch.Generator | None = None) -> Method:
    r"""DARE: keep each 1 x ``block`` block with probability ``keep``, rescaled by :math:`1 / \mathrm{keep}` (what
    ``DareDeltaMoE`` stores, its mask regenerated from a seed).

    Args:
        keep: Kept fraction.
        block: Block length along the input dim; divides C.
        generator: Random source of the mask.

    Returns:
        The method; bits per entry ``16 keep + 1 / block`` (values plus the decode-time bitmap).
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        r, c = delta.shape
        m = torch.rand(r, c // block, device=delta.device, generator=generator) < keep
        return delta * m.repeat_interleave(block, 1) / keep, 16 * keep + 1 / block

    return f


def _top(score: Tensor, keep: float) -> Tensor:
    """Mask of the round(keep * numel) largest entries (ties broken arbitrarily)."""
    k = round(keep * score.numel())
    m = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    m[score.flatten().topk(k).indices] = True
    return m.view(score.shape)


def _xnorm(gram: Tensor) -> Tensor:
    return gram.diag().clamp_min(0).sqrt()


def wanda(keep: float, block: int = 1) -> Method:
    r"""Wanda (Sun et al., 2024) on the delta: score :math:`|\Delta_{ij}| \, \|X_{:,j}\|`, keep the top ``keep`` of
    each row (whole 1 x ``block`` blocks, scored by their sum, when ``block`` > 1), no rescaling.

    Args:
        keep: Kept fraction per row.
        block: Block length along the input dim; divides C.

    Returns:
        The method (needs ``gram``); bits per entry ``16 keep + 1 / block``.
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        assert gram is not None, "wanda needs the inputs' Gram"
        r, c = delta.shape
        score = (delta.abs() * _xnorm(gram)).reshape(r, c // block, block).sum(-1)
        top = score.topk(max(1, int(keep * score.shape[1])), 1).indices
        m = torch.zeros_like(score, dtype=torch.bool).scatter_(1, top, True)
        return delta * m.repeat_interleave(block, 1), 16 * keep + 1 / block

    return f


def sparsegpt(keep: float, damp: float = 0.01, block: int = 128) -> Method:
    r"""SparseGPT (Frantar & Alistarh, 2023) on the delta: per block of ``block`` columns keep the ``keep`` entries
    of largest Optimal Brain Surgeon saliency :math:`\Delta_{ij}^2 / [\mathbf{H}_F^{-1}]_{jj} = \Delta_{ij}^2 / U_{jj}^2`
    (:math:`U` the upper Cholesky factor of :math:`\mathbf{H}^{-1}`), prune the rest column by column, each pruned
    column's error pushed onto the columns after it as in ``gptq``.

    Args:
        keep: Kept fraction.
        damp: Diagonal damping, a fraction of the Gram's mean diagonal.
        block: Columns per mask-selection and lazy-update block.

    Returns:
        The method (needs ``gram``); bits per entry ``16 keep + 1`` (values plus a bitmap).
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        assert gram is not None, "sparsegpt needs the inputs' Gram"
        w, hinv = _cholesky_inverse(delta, gram, damp)
        r, c = w.shape
        for i1 in range(0, c, block):
            i2 = min(i1 + block, c)
            w1, err, hi = w[:, i1:i2].clone(), torch.zeros(r, i2 - i1, device=w.device), hinv[i1:i2, i1:i2]
            m = _top(w1**2 / hi.diag() ** 2, keep)  # U_jj^2 = [H_F^-1]_jj: the OBS saliency
            for j in range(i2 - i1):
                col = w1[:, j]
                q = col * m[:, j]
                e = (col - q) / hi[j, j]
                w1[:, j:] -= e[:, None] @ hi[j:j + 1, j:]
                w1[:, j] = q
                err[:, j] = e
            w[:, i1:i2] = w1
            w[:, i2:] -= err @ hinv[i1:i2, i2:]
        return w, 16 * keep + 1

    return f


def importance_dare(keep: float, block: int = 64, gamma: float = 1.0, rank1: bool = True,
                    generator: torch.Generator | None = None) -> Method:  # fmt: skip
    r"""DARE with importance-weighted keep probabilities: block :math:`(i, b)` is kept with probability
    :math:`q_{ib} = \min(1, c\, s_{ib}^\gamma)`, :math:`s_{ib} = \|X_{:,b}\, \delta_{i,b}\| =
    (\delta_{i,b}^\top \mathbf{H}_{bb}\, \delta_{i,b})^{1/2}` (the block's contribution to output row :math:`i`;
    for blocks of 1 the Wanda score), :math:`c` set so the mean of :math:`q` is ``keep``, and rescaled by
    :math:`1 / q_{ib}`: unbiased like DARE; with :math:`\gamma = 1` and per-block :math:`q` it minimises the variance
    of :math:`X \hat\Delta^\top` for the budget.

    With ``rank1`` the scores are replaced by the outer product of their row and column sums, so a decoder needs only
    those two vectors (R + C / block values) and the seed to regenerate the mask (a heuristic, no longer optimal);
    without, :math:`q` is per block and must be stored (16 bits per block, counted).

    Args:
        keep: Mean keep probability.
        block: Block length along the input dim; divides C.
        gamma: Exponent of the score.
        rank1: Rank-1 probabilities (storable) instead of per block.
        generator: Random source of the mask.

    Returns:
        The method (needs ``gram``); bits per entry ``16 keep + 1 / block`` plus the two vectors.
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        assert gram is not None, "importance_dare needs the inputs' Gram"
        r, c = delta.shape
        n = c // block
        d = delta.reshape(r, n, block)
        hb = gram.reshape(n, block, n, block).diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # (n, block, block)
        s = torch.einsum("rnb,nbc,rnc->rn", d, hb, d).clamp_min(0).sqrt() ** gamma
        if rank1:
            s = s.sum(1, keepdim=True) * s.sum(0, keepdim=True) / s.sum().clamp_min(1e-30)
        if (s > 0).float().mean() <= keep:  # the budget covers every block that matters
            q = (s > 0).float()
        else:
            s = s / s.mean()
            lo, hi = 0.0, 1.0
            while (hi * s).clamp(max=1).mean() < keep:
                hi *= 2
            for _ in range(40):  # bisect c
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if (mid * s).clamp(max=1).mean() < keep else (lo, mid)
            q = (hi * s).clamp(max=1)
        m = torch.rand(q.shape, device=delta.device, generator=generator) < q
        scale = (m / q.clamp_min(1e-12)).repeat_interleave(block, 1)
        side = 16 * (r + n) if rank1 else 16 * r * n
        return delta * scale, 16 * keep + 1 / block + side / (r * c)

    return f


def calibration_grams(moe: DeltaMoE, x: Sequence[Tensor]) -> list[tuple[Tensor, Tensor]]:
    r"""Input Grams of every delta expert: :math:`X_e^\top X_e` for its gate / up projections and
    :math:`H_e^\top H_e`, :math:`H_e = \mathrm{silu}(X_e W_{g,e}^\top) \odot X_e W_{u,e}^\top`, for its down
    projection, in float32.

    Args:
        moe: A ``DeltaMoE`` or ``HierDeltaMoE``.
        x: Per delta expert (in ``d_gate`` order), the (T_e, H) MLP inputs it should be calibrated on, e.g. the
            tokens routed to it or its domain's data.

    Returns:
        Per delta expert, (gate / up Gram (H, H), down Gram (I, I)).
    """
    assert len(x) == moe.n_delta, f"{len(x)} input sets for {moe.n_delta} delta experts"
    g, u = moe.gate.float(), moe.up.float()
    out = []
    for e, xe in enumerate(x):
        xe = xe.float()
        h = F.silu(F.linear(xe, g + moe.d_gate[e].float())) * F.linear(xe, u + moe.d_up[e].float())
        out.append((xe.T @ xe, h.T @ h))
    return out


@torch.no_grad()
def compress_deltas(moe: DeltaMoE, method: Method, grams: Sequence[tuple[Tensor, Tensor]] | None = None) -> float:
    r"""Replace every delta of ``moe`` (a ``DeltaMoE`` or ``HierDeltaMoE``, dense deltas) with ``method``'s
    approximation, in place, in the deltas' dtype.

    Args:
        moe: The module.
        method: A compression method of this module.
        grams: ``calibration_grams(moe, x)``; needed by the calibrated methods.

    Returns:
        Bits per delta entry, averaged over the three projections by size.

    Raises:
        TypeError: If ``moe`` stores its deltas packed (DARE, BitDelta).
    """
    if isinstance(moe, PackedDeltaMoE):
        raise TypeError(f"{type(moe).__name__} stores packed deltas; compress the dense DeltaMoE it was made from")
    total = bits = 0.0
    for e in range(moe.n_delta):
        for p, name in enumerate(("d_gate", "d_up", "d_down")):
            d = getattr(moe, name)
            approx, b = method(d[e].float(), None if grams is None else grams[e][p // 2])
            d[e] = approx.to(d.dtype)
            total, bits = total + d[e].numel(), bits + b * d[e].numel()
    return bits / total


def _fake_quant(w: Tensor, lo: Tensor, step: Tensor, levels: int, group: int) -> Tensor:
    """``_snap`` per group with a straight-through round: the gradient passes the rounding unchanged to ``w`` (inside
    the grid's span) and reaches ``lo`` and ``step``."""
    r, c = w.shape
    v = (w.view(r, c // group, group) - lo) / step
    v = v + (v.round() - v).detach()
    return (v.clamp(0, levels - 1) * step + lo).view(r, c)


def qat_deltas(moe: DeltaMoE, x: Sequence[Tensor], levels: int = 2, group: int = 128, symmetric: bool = False,
               clip: bool = True, steps: int = 200, batch: int = 1024, lr: float = 1e-3, scale_lr: float = 0.1, val: float = 0.1,
               damp: float = 0.01, generator: torch.Generator | None = None) -> list[tuple[float, float]]:  # fmt: skip
    r"""Quantisation-aware training of every delta expert's MLP, block by block (EfficientQAT's Block-AP), in place.

    Each delta expert :math:`e` starts from ``gptq(levels=levels, group=group, ...)`` and is trained to reproduce its
    dense MLP on its own inputs :math:`X_e`. A delta entry is :math:`\hat\delta = \ell + s\,\mathrm{clamp}(
    \lfloor (w - \ell) / s \rceil, 0, L - 1)` with a latent weight :math:`w` and per group a lowest level
    :math:`\ell` and step :math:`s` (symmetric: :math:`\ell = -s (L - 1) / 2`); the rounding passes gradients
    straight through (STE), so :math:`w` can move entries to other levels and :math:`\ell, s` are fitted too. The
    loss is the relative error of the delta's effect on the MLP output,

    .. math::

        \frac{\| f(X; W_b + \hat\Delta) - f(X; W_b + \Delta) \|^2}{\| f(X; W_b) - f(X; W_b + \Delta) \|^2},

    with :math:`f` the SwiGLU MLP and gate, up and down trained jointly, by Adam with a cosine schedule. Adam's step
    for :math:`w` is ``lr`` times the matrix's mean grid step (``lr`` = 0.001 moves an entry by about a thousandth
    of a level per step), for :math:`\ell, s` ``scale_lr`` times that. Each expert keeps whichever of the GPTQ
    start and the trained deltas has the lower held-out loss. The result is on a grid: same storage as GPTQ,
    ``_grid_bits(levels, group, symmetric)`` bits per entry.

    Args:
        moe: A ``DeltaMoE`` or ``HierDeltaMoE`` with dense deltas; replaced by the trained quantised ones.
        x: Per delta expert, its (T_e, H) MLP inputs; a ``val`` share is held out.
        levels: :math:`L`, grid levels.
        group: Entries per grid along the input dimension; divides it.
        symmetric: Grids centred on 0 (only :math:`s` stored).
        clip: Clipped grids for the GPTQ start.
        steps: Adam steps per delta expert.
        batch: Tokens per step.
        lr: Relative step size of the latent weights (see above).
        scale_lr: Step size of the grids relative to ``lr``.
        val: Held-out share of each :math:`X_e`.
        damp: GPTQ damping.
        generator: For the held-out split and batches.

    Returns:
        Per delta expert, the held-out loss of the GPTQ start and after training (the kept one is the lower).
    """
    assert len(x) == moe.n_delta, f"{len(x)} input sets for {moe.n_delta} delta experts"
    start = gptq(levels=levels, group=group, symmetric=symmetric, clip=clip, damp=damp)
    grams = calibration_grams(moe, x)
    W = [moe.gate.detach().float(), moe.up.detach().float(), moe.down.detach().float()]

    def mlp(xb: Tensor, d: Sequence[Tensor]) -> Tensor:
        return F.linear(F.silu(F.linear(xb, W[0] + d[0])) * F.linear(xb, W[1] + d[1]), W[2] + d[2])

    out = []
    for e, xe in enumerate(x):
        xe = xe.float()
        perm = torch.randperm(len(xe), generator=generator).to(xe.device)
        nv = max(1, int(len(xe) * val))
        xv, xt = xe[perm[:nv]], xe[perm[nv:]]
        names = ("d_gate", "d_up", "d_down")
        dense = [getattr(moe, n)[e].detach().float() for n in names]
        with torch.no_grad():
            init = [start(d, grams[e][p // 2])[0] for p, d in enumerate(dense)]
            yv, ybv = mlp(xv, dense), mlp(xv, [torch.zeros_like(d) for d in dense])
        params, groups = [], []
        for a in init:
            r, c = a.shape
            lo, step = _grid(a.view(r, c // group, group), levels, symmetric, False)
            p = {"w": a.clone(), "step": step.clone()} | ({} if symmetric else {"lo": lo.clone()})
            for t in p.values():
                t.requires_grad_(True)
            params.append(p)
            base_lr = lr * step.mean().item()
            groups += [{"params": [p["w"]], "lr": base_lr},
                       {"params": [t for k, t in p.items() if k != "w"], "lr": base_lr * scale_lr}]

        def quant(i: int) -> Tensor:
            p = params[i]
            lo = -p["step"] * (levels - 1) / 2 if symmetric else p["lo"]
            return _fake_quant(p["w"], lo, p["step"], levels, group)

        def loss(xb: Tensor, y: Tensor, yb: Tensor) -> Tensor:
            return (mlp(xb, [quant(i) for i in range(3)]) - y).square().sum() / (yb - y).square().sum()

        with torch.no_grad():
            before = loss(xv, yv, ybv).item()
        opt = torch.optim.Adam([g for g in groups])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        with torch.enable_grad():
            for _ in range(steps):
                idx = torch.randint(0, len(xt), (min(batch, len(xt)),), generator=generator).to(xt.device)
                xb = xt[idx]
                with torch.no_grad():
                    y, yb = mlp(xb, dense), mlp(xb, [torch.zeros_like(d) for d in dense])
                opt.zero_grad(set_to_none=True)
                loss(xb, y, yb).backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    for p in params:
                        p["step"].clamp_(min=1e-12)
        with torch.no_grad():
            after = loss(xv, yv, ybv).item()
            for i, n in enumerate(names):
                getattr(moe, n)[e] = (quant(i) if after < before else init[i]).to(getattr(moe, n).dtype)
        out.append((before, after))
    return out

