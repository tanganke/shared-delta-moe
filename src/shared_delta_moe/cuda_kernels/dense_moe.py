r"""Dense-delta MoE forward for prefill: materialise :math:`W_b + \Delta_g` into a reused scratch, then grouped GEMMs.

For many (token, slot) pairs the GEMMs are compute-bound and every weight tile is reused by many row tiles, so
writing the summed weights out once (a few hundred MB, under a millisecond on H200) costs less than what CUTLASS's
smem-operand wgmma (``F.grouped_mm``) gains over a GEMM whose weight operand is summed in registers (the Triton
kernel). Steps, no host synchronisation:

1. ``route`` (the Triton counting sort) sorts the pairs by group.
2. :math:`W + D` for every delta group with rows (a Triton add that skips empty groups on the device), and the
   base for the zero-expert group, into a (G + 1, N, K) scratch per projection, reused across forwards and layers.
3. Grouped GEMMs for gate and up, SiLU * up in torch (bf16 rounding of g, u, SiLU(g), h as in the PyTorch MLP), a
   grouped GEMM for down, then back to pair order times the routing weight (rounded to the model dtype), the
   slots summed.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

from ..triton_kernels.fused import Routing

_SCRATCH: dict[tuple, Tensor] = {}


def _key(name: str, w: Tensor, d: Tensor) -> tuple:
    """Scratch key: one buffer per projection, shape, dtype, device and stream (two streams never share one)."""
    return name, (d.shape[0] + 1, *w.shape), w.dtype, w.device, torch.cuda.current_stream(w.device).cuda_stream


def _scratch(name: str, w: Tensor, d: Tensor) -> Tensor:
    """A (G + 1, N, K) buffer reused across forwards and layers: a fixed address, so a forward can be captured in a
    CUDA graph once the buffer exists (``supports`` keeps capture off this path until then, so the buffer never comes
    from a graph pool). A normal tensor even when first made under inference mode."""
    # ponytail: kept for the process lifetime; _SCRATCH.clear() releases it
    key = _key(name, w, d)
    if key not in _SCRATCH:
        with torch.inference_mode(False):
            _SCRATCH[key] = torch.empty(key[1], dtype=w.dtype, device=w.device)
    return _SCRATCH[key]


@triton.jit
def _add_kernel(w_ptr, d_ptr, o_ptr, gstart_ptr, NK, sog, BLOCK: tl.constexpr):
    """o[g] = w + d[g] for the groups with rows (an empty group is skipped on the device: no host sync)."""
    g = tl.program_id(1)  # (axis 1 is capped at 65535 programs: the blocks go on axis 0)
    if tl.load(gstart_ptr + g + 1) == tl.load(gstart_ptr + g):
        return
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < NK
    w = tl.load(w_ptr + off, mask=m)
    d = tl.load(d_ptr + g.to(tl.int64) * NK + off, mask=m)
    tl.store(o_ptr + g.to(tl.int64) * sog + off, w + d, mask=m)


def _add(w: Tensor, d: Tensor, out: Tensor, gstart: Tensor) -> None:
    """out[g] = w + d[g] for every delta group g with rows (w, d, out[g] contiguous (N, K) blocks)."""
    G, NK, B = d.shape[0], w.numel(), 4096
    with torch.cuda.device(w.device):
        _add_kernel[(triton.cdiv(NK, B), G)](w, d, out, gstart, NK, out.stride(0), BLOCK=B)


def materialise(base: tuple[Tensor, Tensor, Tensor], deltas: tuple[Tensor, Tensor, Tensor],
                gstart: Tensor) -> list[Tensor]:  # fmt: skip
    r"""Summed expert weights of every group with rows, base last (the zero expert); a group without rows keeps
    whatever the scratch held (no GEMM reads it).

    Args:
        base: (:math:`W_g` (I, H), :math:`W_u` (I, H), :math:`W_d` (H, I)).
        deltas: (:math:`D_g` (G, I, H), :math:`D_u` (G, I, H), :math:`D_d` (G, H, I)), contiguous.
        gstart: (G + 2,) int32 group starts (``Routing.gstart``).

    Returns:
        [gate (G + 1, I, H), up (G + 1, I, H), down (G + 1, H, I)], scratch buffers that the next call overwrites.
    """
    out = []
    for name, w, d in zip(("gate", "up", "down"), base, deltas):
        s = _scratch(name, w, d)
        _add(w, d, s, gstart)
        s[-1].copy_(w)
        out.append(s)
    return out


@torch.no_grad()  # not differentiable (supports() refuses inputs that need a gradient)
def grouped_moe(x: Tensor, r: Routing, base: tuple[Tensor, Tensor, Tensor],
                deltas: tuple[Tensor, Tensor, Tensor]) -> Tensor:  # fmt: skip
    """Parameter-aggregation MoE forward through materialised weights and ``F.grouped_mm``.

    Args:
        x: (T, H) tokens.
        r: The sorted routing (``triton_kernels.fused.route``).
        base: (:math:`W_g` (I, H), :math:`W_u` (I, H), :math:`W_d` (H, I)).
        deltas: dense (:math:`D_g`, :math:`D_u`, :math:`D_d`), (G, N, K) each, contiguous.

    Returns:
        (T, H) MoE output in x's dtype.
    """
    wg, wu, wd = materialise(base, deltas, r.gstart)
    offs = r.gstart[1:]  # group ends, G + 1 groups
    xs = x[r.perm // r.top_k]
    h = F.silu(F.grouped_mm(xs, wg.mT, offs=offs)) * F.grouped_mm(xs, wu.mT, offs=offs)
    return unsort(F.grouped_mm(h, wd.mT, offs=offs), r)


def unsort(y: Tensor, r: Routing) -> Tensor:
    """Rows of y (sorted pair order) back to pair order, times the routing weight (rounded to y's dtype, as the
    PyTorch forward rounds it), the slots of each token summed.

    Args:
        y: (P, H) per sorted pair.
        r: The sorted routing.

    Returns:
        (T, H) in y's dtype.
    """
    inv = torch.empty_like(r.perm)
    inv[r.perm] = torch.arange(r.perm.numel(), dtype=r.perm.dtype, device=r.perm.device)
    o = y[inv] * r.weight.to(y.dtype)[:, None]
    return o if r.top_k == 1 else o.view(r.tokens, r.top_k, -1).sum(1)



def supports(x: Tensor, base: tuple[Tensor, Tensor, Tensor], deltas: tuple[Tensor, Tensor, Tensor]) -> bool:
    """Can ``grouped_moe`` take these?

    Needs sm90 (where it was measured) and ``F.grouped_mm``; bf16 everywhere (fp16 goes to torch's per-group
    fallback, which syncs with the host); contiguous weights and deltas, hidden and inter multiples of 8 (16-byte
    rows); no input that needs a gradient; and, while a CUDA graph is being captured, scratch buffers that already
    exist for this stream (they are never allocated during capture).
    """
    if not (x.is_cuda and hasattr(F, "grouped_mm") and torch.cuda.get_device_capability(x.device)[0] == 9):
        return False
    ts = (x, *base, *deltas)
    if not (all(t.dtype == torch.bfloat16 for t in ts) and all(t.is_contiguous() for t in ts[1:])
            and all(n % 8 == 0 for n in base[0].shape)):  # fmt: skip
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in ts):
        return False
    if torch.cuda.is_current_stream_capturing():
        return all(_key(n, w, d) in _SCRATCH for n, w, d in zip(("gate", "up", "down"), base, deltas))
    return True
