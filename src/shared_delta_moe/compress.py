r"""Delta compression methods, simulated: each maps a dense delta :math:`\Delta` (R, C) to its compressed-then-restored
approximation :math:`\hat\Delta`, dense, and reports the storage it would take in bits per entry (dense bf16 = 16).
They are for measuring accuracy before a format gets packed storage and kernels (as DARE and BitDelta have,
``models/dare.py``, ``models/bitdelta.py``); results are in ``docs/guide/delta-compression.md``.

A method is a callable ``(delta, gram) -> (approx, bits)``. ``delta`` is one projection's delta in float32, the input
dimension last; ``gram`` is :math:`X^\top X` (C, C) of that projection's inputs on the tokens routed to the expert
(``calibration_grams``), used by the calibrated methods (GPTQ, SparseGPT, Wanda, importance DARE) and ignored by the
others. ``compress_deltas`` applies a method to every delta of a ``DeltaMoE`` or ``HierDeltaMoE`` in place.
"""

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


def _minmax(x: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    """Asymmetric min-max grid over the last dim: (offset, step)."""
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    return lo, (hi - lo).clamp_min(1e-12) / (2**bits - 1)


def _snap(x: Tensor, lo: Tensor, step: Tensor, bits: int) -> Tensor:
    return ((x - lo) / step).round().clamp(0, 2**bits - 1) * step + lo


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


def rtn(bits: int, group: int = 128) -> Method:
    """Round to nearest on an asymmetric min-max grid per group of ``group`` entries along the input dim, scale and
    offset in fp16.

    Args:
        bits: Bits per value.
        group: Group size; divides C.

    Returns:
        The method; bits per entry ``bits + 32 / group``.
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        r, c = delta.shape
        x = delta.reshape(r, c // group, group)
        return _snap(x, *_minmax(x, bits), bits).reshape(r, c), bits + 32 / group

    return f


def gptq(bits: int, group: int = 128, damp: float = 0.01, block: int = 128) -> Method:
    r"""GPTQ (Frantar et al., 2023) on the delta: columns quantised in order on ``rtn``'s grid (each group's grid
    fixed when its first column is reached), each one's error, weighted by the inverse Gram, pushed onto the columns
    not yet quantised. Minimises :math:`\|X(\hat\Delta - \Delta)^\top\|_F` greedily.

    Args:
        bits: Bits per value.
        group: Group size; divides C.
        damp: Diagonal damping, a fraction of the Gram's mean diagonal.
        block: Columns per lazy-update block.

    Returns:
        The method (needs ``gram``); bits per entry ``bits + 32 / group``.
    """

    def f(delta: Tensor, gram: Tensor | None) -> tuple[Tensor, float]:
        assert gram is not None, "gptq needs the inputs' Gram"
        w, hinv = _cholesky_inverse(delta, gram, damp)
        r, c = w.shape
        q = torch.zeros_like(w)
        lo = step = None
        for i1 in range(0, c, block):
            i2 = min(i1 + block, c)
            w1, err, hi = w[:, i1:i2].clone(), torch.zeros(r, i2 - i1, device=w.device), hinv[i1:i2, i1:i2]
            for j in range(i2 - i1):
                k = i1 + j
                if k % group == 0:  # this group's grid, from its current (error-updated) values
                    lo, step = _minmax(torch.cat([w1[:, j:j + group], w[:, i2:k + group]], 1), bits)
                col = w1[:, j:j + 1]
                qc = _snap(col, lo, step, bits)
                q[:, k:k + 1] = qc
                e = (col - qc) / hi[j, j]
                w1[:, j:] -= e @ hi[j:j + 1, j:]
                err[:, j:j + 1] = e
            w[:, i2:] -= err @ hinv[i1:i2, i2:]
        return q, bits + 32 / group

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
