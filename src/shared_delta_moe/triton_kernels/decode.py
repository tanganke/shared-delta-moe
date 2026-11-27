r"""Decode 2-bit group-grid deltas (``models/grid.py``, mode 3) into full expert weights in one pass:
:math:`W_b + (\ell + s q)` for every delta expert, plus :math:`W_b` itself as the last expert (the zero experts), each
entry rounded as the materialised forward rounds it (the delta once to the model dtype, then the sum). Reads 2.25
bits and writes 16 bits per entry: memory-bound, a few ms per 27B layer on H200."""

import torch
import triton
import triton.language as tl
from torch import Tensor

from .fused import Delta


@triton.jit
def _decode_kernel(c_ptr, s_ptr, z_ptr, w_ptr, o_ptr, R, C, MW, NB, SB: tl.constexpr, BW: tl.constexpr,
                   BASE: tl.constexpr):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)  # (expert, row) of the (N, R, C) deltas
    wj = tl.program_id(1) * BW + tl.arange(0, BW)
    w = tl.load(c_ptr + row * MW + wj, mask=wj < MW, other=0)
    q = tl.reshape((w[:, None] >> (tl.arange(0, 16) * 2)[None, :]) & 3, (BW * 16,)).to(tl.float32)
    col = tl.program_id(1) * BW * 16 + tl.arange(0, BW * 16)
    cm = col < C
    s = tl.load(s_ptr + row * NB + col // SB, mask=cm, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + row * NB + col // SB, mask=cm, other=0.0).to(tl.float32)
    d = (z + s * q).to(o_ptr.dtype.element_ty)
    if BASE:
        d = tl.load(w_ptr + (row % R) * C + col, mask=cm, other=0.0) + d
    tl.store(o_ptr + row * C + col, d, mask=cm)


def decode_grid(d: Delta, base: Tensor | None = None) -> Tensor:
    """Full weights of every expert from one projection's grid deltas, or the deltas alone.

    Args:
        d: Mode 3 Delta: (N, R, C / 16) codes, (N, R, C / g) steps and lowest levels.
        base: (R, C) base weight :math:`W_b`; None: the deltas alone, in the steps' dtype.

    Returns:
        (N + 1, R, C) in base's dtype: :math:`W_b + \\Delta_e` for e < N, then :math:`W_b`; without base (N, R, C)
        :math:`\\Delta_e`.
    """
    assert d.mode == 3
    N, R, MW = d.d.shape
    C = MW * 16
    ref = d.p if base is None else base
    out = torch.empty(N + (base is not None), R, C, dtype=ref.dtype, device=ref.device)
    if base is not None:
        out[N].copy_(base)
    BW = 64
    c, s, z, b = (t.contiguous() for t in (d.d, d.p, d.m, ref))
    with torch.cuda.device(ref.device):
        _decode_kernel[(N * R, triton.cdiv(MW, BW))](c, s, z, b, out, R, C, MW, C // d.block, SB=d.block, BW=BW,
                                                       BASE=base is not None, num_warps=4)  # fmt: skip
    return out
