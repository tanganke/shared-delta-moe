"""Fused parameter aggregation in Triton: y = x @ (W_b + Delta_g)^T for each expert group g, with W_b + Delta_g formed
tile by tile in registers. Unlike materialising the sum it writes no weights, and unlike per-matrix output aggregation
it does one GEMM's FLOPs, not two. Group G (one past the deltas) is the zero expert: W_b alone.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from .configs import _configs


@triton.autotune(configs=_configs(), key=["N", "K", "MAX_ROWS_BUCKET"])
@triton.jit
def _delta_gmm_kernel(
    x_ptr, wb_ptr, d_ptr, y_ptr, starts_ptr, ends_ptr,
    N, K, G, MAX_ROWS_BUCKET,
    sxm, sxk, swn, swk, sdg, sdn, sdk, sym, syn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
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
    dcols = cmask & (
        g < G
    )  # zero expert: the delta load is masked off, not branched (keeps the loop pipelined)
    acc = tl.zeros(
        (BN, BM), dtype=tl.float32
    )  # y^T: the summed weight is operand A, which may live in registers
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
        dw = tl.load(
            d_ptr + g * sdg + cols[:, None] * sdn + kk[None, :] * sdk,
            mask=dcols[:, None] & kmask[None, :],
            other=0.0,
        )
        acc = tl.dot(w + dw, tl.trans(a), acc)
    y = y_ptr + cols[:, None] * syn + rows[None, :] * sym
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=cmask[:, None] & rmask[None, :])


def delta_gmm(
    x: Tensor, wb: Tensor, d: Tensor, starts: Tensor, ends: Tensor, max_rows: int
) -> Tensor:
    """x (M, K) sorted by group; wb (N, K); d (G, N, K); starts/ends (G+1,) row range of each group, the last being
    the zero expert. Returns (M, N), rows of group g multiplied by (wb + d[g])^T (by wb^T for g = G).
    """
    M, K = x.shape
    N, G = wb.shape[0], d.shape[0]
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    bucket = (
        1 << max(max_rows - 1, 0).bit_length()
    )  # autotune once per power of two of the largest group
    grid = lambda meta: (
        G + 1,
        triton.cdiv(max_rows, meta["BM"]),
        triton.cdiv(N, meta["BN"]),
    )
    # Triton launches on the current CUDA device, not on the tensors' one
    with torch.cuda.device(x.device):
        _delta_gmm_kernel[grid](
            x, wb, d, y, starts, ends, N, K, G, bucket,
            *x.stride(), *wb.stride(), *d.stride(), *y.stride(),
        )  # fmt: skip
    return y
