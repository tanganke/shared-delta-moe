"""delta_gmm with seeded-DARE deltas (models/dare.py): the Philox mask is regenerated tile by tile (once per DARE
block), the kept values gathered from the packed storage and added to W_b in registers.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..models.dare import threshold
from .configs import _configs


@triton.autotune(configs=_configs(), key=["N", "K", "MAX_ROWS_BUCKET"])
@triton.jit
def _dare_gmm_kernel(
    x_ptr, wb_ptr, v_ptr, p_ptr, y_ptr, starts_ptr, ends_ptr,
    N, K, G, MAX_ROWS_BUCKET, seed, proj, thresh,
    sxm, sxk, swn, swk, sym, syn,
    B: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    g = tl.program_id(0)
    start = tl.load(starts_ptr + g)
    end = tl.load(ends_ptr + g)
    row0 = start + tl.program_id(1) * BM
    if row0 >= end:
        return
    rows = row0 + tl.arange(0, BM)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    rmask, cmask = rows < end, cols < N
    dcols = cmask & (g < G)
    run = tl.load(
        p_ptr + g * N + cols, mask=dcols, other=0
    )  # each weight row's next packed value
    stream = (g * 3 + proj).to(tl.int64) << 32  # Philox counter word 1 (dare.py spec)
    unit0 = cols.to(tl.int64)[:, None] * (K // B)
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + ks
        kmask = kk < K
        a = tl.load(
            x_ptr + rows[:, None] * sxm + kk[None, :] * sxk,
            mask=rmask[:, None] & kmask[None, :],
            other=0.0,
        )
        w = tl.load(
            wb_ptr + cols[:, None] * swn + kk[None, :] * swk,
            mask=cmask[:, None] & kmask[None, :],
            other=0.0,
        )
        # one Philox per DARE block in the tile, broadcast to its elements (needs BK % B == 0 or B % BK == 0)
        if B >= BK:
            u = tl.randint(seed, stream + unit0 + k0 // B).to(tl.int64)  # (BN, 1)
            m = tl.broadcast_to(u < thresh, (BN, BK))
        else:
            ub = tl.randint(
                seed, stream + unit0 + (k0 // B + tl.arange(0, BK // B))[None, :]
            )
            kb = tl.broadcast_to(
                (ub.to(tl.int64) < thresh)[:, :, None], (BN, BK // B, B)
            )
            m = tl.reshape(kb, (BN, BK))
        m = m & dcols[:, None] & kmask[None, :]
        mi = m.to(tl.int32)
        pos = run[:, None] + (tl.cumsum(mi, axis=1) - mi)
        dw = tl.load(v_ptr + pos, mask=m, other=0.0)
        run += tl.sum(mi, axis=1)
        acc = tl.dot(w + dw, tl.trans(a), acc)
    y = y_ptr + cols[:, None] * syn + rows[None, :] * sym
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=cmask[:, None] & rmask[None, :])


def dare_gmm(
    x: Tensor, wb: Tensor, v: Tensor, p: Tensor, seed: int, proj: int, keep: float, block: int,
    starts: Tensor, ends: Tensor, max_rows: int,
) -> Tensor:  # fmt: skip
    """delta_gmm with seeded-DARE deltas (dare.py): v the packed kept values, p (G, N) each row's start in v."""
    M, K = x.shape
    N, G = wb.shape[0], p.shape[0]
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    bucket = 1 << max(max_rows - 1, 0).bit_length()
    grid = lambda meta: (
        G + 1,
        triton.cdiv(max_rows, meta["BM"]),
        triton.cdiv(N, meta["BN"]),
    )
    # Triton launches on the current CUDA device, not on the tensors' one
    with torch.cuda.device(x.device):
        _dare_gmm_kernel[grid](
            x, wb, v, p, y, starts, ends, N, K, G, bucket, seed, proj, threshold(keep),
            *x.stride(), *wb.stride(), *y.stride(), B=block,
        )  # fmt: skip
    return y
