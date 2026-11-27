r"""CUDA kernels for packed deltas (seeded DARE, BitDelta) at decode sizes, loaded from a prebuilt library.

At a few tokens per expert the fused Triton kernels (``triton_kernels/fused.py``) are bound by unpacking the deltas
tile by tile, not by memory. These kernels (``packed_moe.cu``) treat decode as a GEMV: a warp owns 16 weight rows over
a K range and reads :math:`W_b` once. For each 64-wide K step and each routed group :math:`g` it forms
:math:`W_b + \Delta_g` in registers, rounded to the model dtype. For BitDelta, a byte of sign bits indexes a table of
sign-flip masks applied to the scale. For DARE, popcounts over the keep bitmap place the kept values. It then
multiplies that sum with the group's pairs on the tensor cores (mma.m16n8k16, the sorted pairs as the N side).
:math:`W_b` and the deltas stream through per-warp ``cp.async`` rings, several steps ahead; DARE's kept values take
only the pool space they need. The CTA keeps the pairs' activations for its K range in shared memory. Gate and up
share a warp and end in SiLU(g) * u; down ends in the routing weight. Split-K partials are summed in a fixed order by
the last warp of a row strip, so results are deterministic. The rounding is that of the Triton kernels and the
materialised forward.

``triton_kernels.fused.fused_moe`` uses them after the same ``route`` for packed deltas up to ``MAX_PAIRS``
(token, slot) pairs. Everything else falls back to Triton: more pairs, hidden or inter not a multiple of 64, DARE
blocks below 8, BitDelta scale blocks below 32, or the library not built. Under ``torch.compile`` they run inside the
same registered op (``shared_delta_moe::fused_moe_dare`` / ``_bit``), so CUDA graphs work as before.

``matrix_moe`` is the output-aggregation form (``forward_matrix`` of ``DareDeltaMoE`` / ``BitDeltaMoE``):
:math:`x W_g^T`, :math:`x W_u^T` and :math:`h W_d^T` by cuBLAS, and only the deltas' products in the kernels, so
:math:`W_b` is read once however many groups are routed. See ``docs/internals/cuda-decode.md``.

Only sm_90 (H100 / H200) is supported: the build is sm_90a only, and the layout needs Hopper's 227 KB of shared memory
per CTA; ``supports`` sends every other GPU to Triton. Build once per torch version, on a machine with nvcc:

.. code-block:: bash

   SHARED_DELTA_MOE_CUDA_DIR=/data/.../sdm-cuda python -m shared_delta_moe.cuda_kernels.packed_ops

This writes ``$SHARED_DELTA_MOE_CUDA_DIR/sdm_packed-<torch version>-<source hash>.so`` (default directory
``~/.cache/shared_delta_moe/cuda``). At runtime the library is loaded by that path, with the same variable and no
nvcc needed; without it (or after an edit of ``packed_moe.cu``, until rebuilt), if it fails to load, or with
``SHARED_DELTA_MOE_CUDA=0``, everything runs on Triton.
"""

import functools
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

MAX_PAIRS = 16  # H200, hidden 2560, inter 9216, 5 deltas: CUDA wins up to 16 (token, slot) pairs, Triton from 24
MATRIX_MAX_PAIRS = 1024  # forward_matrix: beyond this the inherited forward (it unpacks every delta)
ENABLED = os.environ.get("SHARED_DELTA_MOE_CUDA", "1") != "0"
DIR = Path(os.environ.get("SHARED_DELTA_MOE_CUDA_DIR", Path.home() / ".cache" / "shared_delta_moe" / "cuda"))
SRC = Path(__file__).with_name("packed_moe.cu")
HASH = hashlib.sha256(SRC.read_bytes()).hexdigest()[:12] if SRC.exists() else "nosrc"
LIB = DIR / f"sdm_packed-{torch.__version__}-{HASH}.so"  # never a stale build
ARCHS = ("90a",)  # sm_90 only: the layout wants Hopper's 227 KB of shared memory per CTA

_loaded: bool | None = None


def available() -> bool:
    """Load the prebuilt library once.

    Returns:
        True if ``torch.ops.sdm_packed`` is usable (built for this torch version, CUDA present, not disabled).
    """
    global _loaded
    if _loaded is None:
        _loaded = False
        if ENABLED and torch.cuda.is_available() and LIB.exists():
            try:
                torch.ops.load_library(str(LIB))
                _loaded = True
            except (OSError, RuntimeError):  # a broken or mismatched build: Triton
                pass
    return _loaded


def supports(x: Tensor, pairs: int, base: tuple[Tensor, Tensor, Tensor], deltas, matrix: bool = False) -> bool:
    """Whether these kernels take this call.

    Args:
        x: (T, H) tokens.
        pairs: (token, slot) pairs, T * top_k.
        base: (:math:`W_g`, :math:`W_u`, :math:`W_d`).
        deltas: The three ``Delta`` of the projections (mode 1 DARE or 2 BitDelta).
        matrix: For ``matrix_moe`` (output aggregation) instead of ``moe``: up to ``MATRIX_MAX_PAIRS`` pairs. Only
            metadata is read while ``torch.compile`` traces, so it also works on fake tensors.

    Returns:
        True on sm_90 for 1 .. ``MAX_PAIRS`` pairs of bf16/fp16, hidden and inter multiples of 64, DARE blocks of at
        least 8 (a power of 2), BitDelta with one scale per row or blocks of at least 32, every tensor contiguous and
        16-byte aligned, the layout within the device's shared memory, and the library loaded.
    """
    d = deltas[0]
    if d.mode == 3:
        return matrix and _grid_ok(x, pairs, base, deltas)
    if not (d.mode in (1, 2) and 0 < pairs <= (MATRIX_MAX_PAIRS if matrix else MAX_PAIRS) and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    if not available():  # first: unbuilt or disabled costs nothing more per call
        return False
    cap, smax = _device(x.device.index if x.device.index is not None else torch.cuda.current_device())
    if cap != (9, 0) or x.shape[1] % 64 or base[0].shape[0] % 64 or not _aligned(x.contiguous()):
        return False
    extra = (torch.int64, torch.int32) if d.mode == 1 else (x.dtype, torch.int32)  # (p, m) or (scales, sign words)
    if any(dl.mode != d.mode or dl.d.numel() == 0 for dl in deltas) or deltas[1].block != d.block:  # one gate/up block
        return False
    ts = [(w, x.dtype) for w in base] + [t for dl in deltas for t in zip((dl.p, dl.m if d.mode == 1 else dl.d), extra)]
    if any(t is None or t.dtype != dt or not t.is_contiguous() or not _aligned(t) for t, dt in ts):
        return False
    dd = x.dtype if d.mode == 1 else torch.int32  # kept values or sign words
    if any(dl.d.dtype != dd or not dl.d.is_contiguous() or not _aligned(dl.d) for dl in deltas):
        return False
    for dl in deltas:
        if dl.mode == 1 and (dl.block < 8 or dl.block & (dl.block - 1)) or dl.mode == 2 and dl.block and dl.block < 32:
            return False
    g = d.p.shape[0]
    if matrix:  # the mma kernel: a few KB; DARE's CUDA-core kernel: 8 warps x mats x (K / block) halfwords
        h, i = x.shape[1], base[0].shape[0]
        return d.mode == 2 or 8 * 2 * (h // d.block) * 2 <= smax and 8 * (i // deltas[2].block) * 2 <= smax
    return _smem(d.mode, 2, g, pairs, d.block) <= smax and _smem(d.mode, 1, g, pairs, deltas[2].block) <= smax


def _grid_ok(x: Tensor, pairs: int, base: tuple[Tensor | None, ...], deltas) -> bool:
    """``supports`` for 2-bit grids (mode 3, matrix kernels only): groups a power of 2 of at least 32, rows a
    multiple of 16, codes (int32), steps and lowest levels (x's dtype) contiguous and aligned. A base of None (the
    routed deltas: no base) is not checked."""
    if not (0 < pairs <= MATRIX_MAX_PAIRS and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    if not available():
        return False
    cap, _ = _device(x.device.index if x.device.index is not None else torch.cuda.current_device())
    if cap != (9, 0) or x.shape[1] % 64 or not _aligned(x.contiguous()):
        return False
    if any(w is not None and (w.dtype != x.dtype or not w.is_contiguous() or not _aligned(w)) for w in base):
        return False
    if len(deltas) == 3 and deltas[0].block != deltas[1].block:  # gate and up share the kernels' group
        return False
    for dl in deltas:
        if dl.mode != 3 or dl.block < 32 or dl.block & (dl.block - 1) or dl.d.shape[1] % 16:
            return False
        for t, dt in ((dl.d, torch.int32), (dl.p, x.dtype), (dl.m, x.dtype)):
            if t is None or t.dtype != dt or not t.is_contiguous() or not _aligned(t):
                return False
    return True


def _aligned(t: Tensor) -> bool:
    """16-byte aligned. Not checked while ``torch.compile`` traces (fake tensors have no address): ``matrix_moe_any``
    copies a misaligned x, and the op checks the rest."""
    return torch.compiler.is_compiling() or t.data_ptr() % 16 == 0


@functools.cache
def _device(index: int) -> tuple[tuple[int, int], int]:
    """(compute capability, opt-in shared memory per block) of a CUDA device."""
    p = torch.cuda.get_device_properties(index)
    return (p.major, p.minor), p.shared_memory_per_block_optin


def _smem(mode: int, mats: int, g: int, pairs: int, block: int, base: bool = True) -> int:
    """Shared memory of the smallest K split, as ``launch`` in ``packed_moe.cu`` computes it (keep the two in step)."""
    kc, warps, nt = 64, 8, next(n for n in (1, 2, 4, 8) if 8 * n >= pairs)
    hdr = ((2 * g + 4 + pairs) * 4 + 15) // 16 * 16
    ring = 4 * mats * 16 * kc * 2 if base else 0  # W_b ring (parameter aggregation only)
    if mode == 1:  # DARE pool and step records + row cursors and bitmap words of up to G + 1 groups (G: no base)
        words = kc // block // 32 + 2
        warp = ring + 256 * 16 + 8 * 36 * 4 + mats * min(g + base, pairs) * 16 * (8 + 4 * words)
    else:  # BitDelta steps, then the sign table
        warp, hdr = ring + 8 * mats * (64 * kc // 32 + 128), hdr + 4096
    return hdr + nt * 8 * (kc + 32) * 2 + warps * warp


def moe(x: Tensor, r, base: tuple[Tensor, Tensor, Tensor], deltas) -> Tensor:
    """``fused_moe`` after ``route``: gate/up with SiLU * up, then down with the routing weights.

    Args:
        x: (T, H) tokens.
        r: The sorted routing (``triton_kernels.fused.Routing``).
        base: (:math:`W_g` (I, H), :math:`W_u` (I, H), :math:`W_d` (H, I)).
        deltas: (:math:`D_g`, :math:`D_u`, :math:`D_d`) as ``Delta``, mode 1 or 2.

    Returns:
        (T, H) MoE output in x's dtype.
    """
    dg, du, dd = deltas
    x = x.contiguous()

    def extra(d):  # (p, m): DARE row starts and bitmap; BitDelta scales and a placeholder
        return (d.p, d.m) if d.mode == 1 else (d.p, d.d)

    pg, mg = extra(dg)
    pu, mu = extra(du)
    h = torch.ops.sdm_packed.gate_up(x, r.perm, r.gstart, base[0], base[1], dg.d, du.d, pg, pu, mg, mu, dg.mode,
                                     dg.block, r.top_k)  # fmt: skip
    pd, md = extra(dd)
    o = torch.ops.sdm_packed.down(h, r.perm, r.gstart, r.weight, base[2], dd.d, pd, md, dd.mode, dd.block, r.top_k)
    return o if r.top_k == 1 else o.view(r.tokens, r.top_k, -1).sum(1).to(x.dtype)


@torch.autocast("cuda", enabled=False)  # cuBLAS products in x's dtype, as the kernels check
def matrix_moe(x: Tensor, r, base: tuple[Tensor, Tensor, Tensor], deltas) -> Tensor:
    """Output aggregation (``DeltaMoE.forward_matrix``) after ``route``, rounded as ``_forward_matrix_grouped``:
    :math:`x W_g^T`, :math:`x W_u^T` for all tokens and :math:`h W_d^T` for all pairs by cuBLAS, the deltas' products
    by the kernels (gate/up adding them and SiLU * up per pair; down adding them and the routing weight).

    Args:
        x: (T, H) tokens.
        r: The sorted routing (``triton_kernels.fused.Routing``).
        base: (:math:`W_g` (I, H), :math:`W_u` (I, H), :math:`W_d` (H, I)).
        deltas: (:math:`D_g`, :math:`D_u`, :math:`D_d`) as ``Delta``, mode 1, 2 or 3.

    Returns:
        (T, H) MoE output in x's dtype.
    """
    dg, du, dd = deltas
    x = x.contiguous()

    def extra(d):
        return (d.p, d.d) if d.mode == 2 else (d.p, d.m)

    (pg, mg), (pu, mu), (pd, md) = extra(dg), extra(du), extra(dd)
    g0, u0 = F.linear(x, base[0]), F.linear(x, base[1])
    if all(_grid_fast(d, k) for d, k in ((dg, x.shape[1]), (dd, g0.shape[1]))):  # grid_kernel (factored, faster)
        h = torch.ops.sdm_packed.grid_gate_up(*_prep(x), r.perm, r.gstart, g0, u0, dg.d, du.d, pg, pu, mg, mu,
                                              dg.block, r.top_k)  # fmt: skip
        o = torch.ops.sdm_packed.grid_down(*_prep(h), r.perm, r.gstart, r.weight, F.linear(h, base[2]), dd.d, pd, md,
                                           dd.block, r.top_k)  # fmt: skip
        return o if r.top_k == 1 else o.view(r.tokens, r.top_k, -1).sum(1).to(x.dtype)
    h = torch.ops.sdm_packed.matrix_gate_up(x, r.perm, r.gstart, g0, u0, dg.d, du.d, pg, pu, mg, mu, dg.mode, dg.block,
                                            r.top_k)  # fmt: skip  # (P, I), sorted pairs
    o = torch.ops.sdm_packed.matrix_down(h, r.perm, r.gstart, r.weight, F.linear(h, base[2]), dd.d, pd, md, dd.mode,
                                         dd.block, r.top_k)  # fmt: skip  # (P, H) r_i y_i by pair
    return o if r.top_k == 1 else o.view(r.tokens, r.top_k, -1).sum(1).to(x.dtype)


def _grid_fast(d, K: int) -> bool:
    """The grid ops (``grid_gate_up`` / ``grid_down``: 2-bit grids factored per 128-entry block) take this Delta of
    input width K: groups a multiple of 128 and an even number of them per row."""
    return d.mode == 3 and d.block % 128 == 0 and (K // d.block) % 2 == 0


def _prep(v: Tensor) -> tuple[Tensor, Tensor]:
    """The grid ops' activations: v with each 16-entry chunk interleaved (entries i, i + 8 adjacent, as the kernel
    pairs the codes) and v's fp32 sums over 128-entry blocks."""
    rows, K = v.shape
    return v.view(rows, K // 16, 2, 8).transpose(2, 3).reshape(rows, K), v.float().view(rows, K // 128, 128).sum(-1)


def matrix_moe_any(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, base: tuple[Tensor, Tensor, Tensor],
                   deltas) -> Tensor:  # fmt: skip
    """``route`` then ``matrix_moe``; under ``torch.compile`` one registered op (``matrix_moe_dare`` / ``_bit``).

    Args:
        x: (T, H) tokens.
        idx: (T, k) expert per slot; ``idx >= n_delta``: a zero expert.
        rw: (T, k) routing weight per slot.
        n_delta: :math:`G`, number of delta experts.
        base: (:math:`W_g`, :math:`W_u`, :math:`W_d`).
        deltas: (:math:`D_g`, :math:`D_u`, :math:`D_d`) as ``Delta``, mode 1 or 2, for which ``supports(...,
            matrix=True)`` holds.

    Returns:
        (T, H) MoE output in x's dtype.
    """
    if not torch.compiler.is_compiling():
        return _matrix(x, idx, rw, n_delta, base, deltas)
    if deltas[0].mode == 3:
        return matrix_moe_grid(x, idx, rw, *base, *(d.d for d in deltas), *(d.p for d in deltas),
                               *(d.m for d in deltas), n_delta, deltas[0].block, deltas[2].block)  # fmt: skip
    if deltas[0].mode == 2:
        return matrix_moe_bit(x, idx, rw, *base, *(d.d for d in deltas), *(d.p for d in deltas), n_delta,
                              deltas[0].block, deltas[2].block)  # fmt: skip
    return matrix_moe_dare(x, idx, rw, *base, *(d.d for d in deltas), *(d.p for d in deltas),
                           *(d.m for d in deltas), n_delta, deltas[0].block)  # fmt: skip


def _matrix(x, idx, rw, n_delta, base, deltas):
    from ..triton_kernels.fused import route

    x = x.contiguous()
    if x.data_ptr() % 16:  # (a traced call checked the storage offset only)
        x = x.clone()
    return matrix_moe(x, route(idx, rw, n_delta), base, deltas)


@torch.library.custom_op("shared_delta_moe::matrix_moe_dare", mutates_args=())
def matrix_moe_dare(x: Tensor, idx: Tensor, rw: Tensor, wg: Tensor, wu: Tensor, wd: Tensor, vg: Tensor, vu: Tensor,
                    vd: Tensor, pg: Tensor, pu: Tensor, pd: Tensor, mg: Tensor, mu: Tensor, md: Tensor, n_delta: int,
                    block: int) -> Tensor:  # fmt: skip
    from ..triton_kernels import Delta

    deltas = tuple(Delta(1, v, p, m, block) for v, p, m in ((vg, pg, mg), (vu, pu, mu), (vd, pd, md)))
    return _matrix(x, idx, rw, n_delta, (wg, wu, wd), deltas)


@torch.library.custom_op("shared_delta_moe::matrix_moe_bit", mutates_args=())
def matrix_moe_bit(x: Tensor, idx: Tensor, rw: Tensor, wg: Tensor, wu: Tensor, wd: Tensor, bg: Tensor, bu: Tensor,
                   bd: Tensor, sg: Tensor, su: Tensor, sd: Tensor, n_delta: int, block_gu: int,
                   block_d: int) -> Tensor:  # fmt: skip
    from ..triton_kernels import Delta

    deltas = tuple(Delta(2, b, s, block=k) for b, s, k in ((bg, sg, block_gu), (bu, su, block_gu), (bd, sd, block_d)))
    return _matrix(x, idx, rw, n_delta, (wg, wu, wd), deltas)


@torch.library.custom_op("shared_delta_moe::matrix_moe_grid", mutates_args=())
def matrix_moe_grid(x: Tensor, idx: Tensor, rw: Tensor, wg: Tensor, wu: Tensor, wd: Tensor, cg: Tensor, cu: Tensor,
                    cd: Tensor, sg: Tensor, su: Tensor, sd: Tensor, zg: Tensor, zu: Tensor, zd: Tensor, n_delta: int,
                    block_gu: int, block_d: int) -> Tensor:  # fmt: skip
    from ..triton_kernels import Delta

    deltas = tuple(Delta(3, c, s, z, k) for c, s, z, k in ((cg, sg, zg, block_gu), (cu, su, zu, block_gu),
                                                            (cd, sd, zd, block_d)))  # fmt: skip
    return _matrix(x, idx, rw, n_delta, (wg, wu, wd), deltas)


def routed_matrix(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, d) -> Tensor:
    r"""Routed deltas of one linear layer, no base (``triton_kernels.fused.routed_deltas``), on the matrix down
    kernel: :math:`\sum_i r_i \Delta_{e_i} x` per token, the deltas' products for the sorted pairs over a zero base
    product.

    Args:
        x: (T, C) tokens.
        idx: (T, k) delta per slot (``>= n_delta``: none).
        rw: (T, k) routing weight per slot.
        n_delta: :math:`G`, deltas.
        d: The (G, R, C) deltas as a mode-3 ``Delta`` for which ``_grid_ok`` holds.

    Returns:
        (T, R) in x's dtype.
    """
    from ..triton_kernels.fused import route

    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    r = route(idx, rw, n_delta)
    zero = x.new_zeros(r.perm.shape[0], d.d.shape[1])
    if _grid_fast(d, x.shape[1]):
        xp, xs = _prep(x)
        tok = r.perm // r.top_k  # sorted pairs
        o = torch.ops.sdm_packed.grid_down(xp[tok], xs[tok], r.perm, r.gstart, r.weight, zero, d.d, d.p, d.m, d.block,
                                           r.top_k)  # fmt: skip
    else:
        h = x[r.perm // r.top_k]  # sorted pairs
        o = torch.ops.sdm_packed.matrix_down(h, r.perm, r.gstart, r.weight, zero, d.d, d.p, d.m, 3, d.block, r.top_k)
    return o.view(r.tokens, r.top_k, -1).sum(1).to(x.dtype)


@matrix_moe_grid.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)


@matrix_moe_dare.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)


@matrix_moe_bit.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)


def build(verbose: bool = True) -> Path:
    """Compile ``packed_moe.cu`` (needs nvcc and ninja) and copy the library to ``LIB``.

    Args:
        verbose: Show the compiler output.

    Returns:
        The library's path.
    """
    import shutil

    from torch.utils.cpp_extension import load

    tmp = DIR / f"build-{torch.__version__}"
    tmp.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-std=c++17", *(f"-gencode=arch=compute_{a},code=sm_{a}" for a in ARCHS)]
    load("sdm_packed", [str(SRC)], extra_cuda_cflags=flags, build_directory=str(tmp), is_python_module=False,
         verbose=verbose)  # fmt: skip
    shutil.copyfile(tmp / "sdm_packed.so", LIB)
    return LIB


if __name__ == "__main__":
    print(build())
