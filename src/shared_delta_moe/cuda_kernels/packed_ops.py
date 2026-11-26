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

Only sm_90 (H100 / H200) is supported: the build is sm_90a only, and the layout needs Hopper's 227 KB of shared memory
per CTA; ``supports`` sends every other GPU to Triton. Build once per torch version, on a machine with nvcc:

.. code-block:: bash

   SHARED_DELTA_MOE_CUDA_DIR=/data/.../sdm-cuda python -m shared_delta_moe.cuda_kernels.packed_ops

This writes ``$SHARED_DELTA_MOE_CUDA_DIR/sdm_packed-<torch version>-<source hash>.so`` (default directory
``~/.cache/shared_delta_moe/cuda``). At runtime the library is loaded by that path, with the same variable and no
nvcc needed; without it (or after an edit of ``packed_moe.cu``, until rebuilt), if it fails to load, or with
``SHARED_DELTA_MOE_CUDA=0``, everything runs on Triton.
"""

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor

MAX_PAIRS = 16  # H200, hidden 2560, inter 9216, 5 deltas: CUDA wins up to 16 (token, slot) pairs, Triton from 24
ENABLED = os.environ.get("SHARED_DELTA_MOE_CUDA", "1") != "0"
DIR = Path(os.environ.get("SHARED_DELTA_MOE_CUDA_DIR", Path.home() / ".cache" / "shared_delta_moe" / "cuda"))
SRC = Path(__file__).with_name("packed_moe.cu")
HASH = hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]
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


def supports(x: Tensor, pairs: int, base: tuple[Tensor, Tensor, Tensor], deltas) -> bool:
    """Whether these kernels take this call.

    Args:
        x: (T, H) tokens.
        pairs: (token, slot) pairs, T * top_k.
        base: (:math:`W_g`, :math:`W_u`, :math:`W_d`).
        deltas: The three ``Delta`` of the projections (mode 1 DARE or 2 BitDelta).

    Returns:
        True on sm_90 for 1 .. ``MAX_PAIRS`` pairs of bf16/fp16, hidden and inter multiples of 64, DARE blocks of at
        least 8 (a power of 2), BitDelta with one scale per row or blocks of at least 32, every tensor contiguous and
        16-byte aligned, the layout within the device's shared memory, and the library loaded.
    """
    d = deltas[0]
    if not (d.mode in (1, 2) and 0 < pairs <= MAX_PAIRS and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    if torch.cuda.get_device_capability(x.device) != (9, 0) or x.shape[1] % 64 or base[0].shape[0] % 64:
        return False
    extra = (torch.int64, torch.int32) if d.mode == 1 else (x.dtype, torch.int32)  # (p, m) or (scales, sign words)
    if any(dl.mode != d.mode or dl.d.numel() == 0 for dl in deltas) or deltas[1].block != d.block:  # one gate/up block
        return False
    ts = [(w, x.dtype) for w in base] + [t for dl in deltas for t in zip((dl.p, dl.m if d.mode == 1 else dl.d), extra)]
    if any(t is None or t.dtype != dt or not t.is_contiguous() or t.data_ptr() % 16 for t, dt in ts):
        return False
    dd = x.dtype if d.mode == 1 else torch.int32  # kept values or sign words
    if any(dl.d.dtype != dd or not dl.d.is_contiguous() or dl.d.data_ptr() % 16 for dl in deltas):
        return False
    for dl in deltas:
        if dl.mode == 1 and (dl.block < 8 or dl.block & (dl.block - 1)) or dl.mode == 2 and dl.block and dl.block < 32:
            return False
    smax = torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin
    g = d.p.shape[0]
    fits = _smem(d.mode, 2, g, pairs, d.block) <= smax and _smem(d.mode, 1, g, pairs, deltas[2].block) <= smax
    return fits and available()


def _smem(mode: int, mats: int, g: int, pairs: int, block: int) -> int:
    """Shared memory of the smallest K split, as ``launch`` in ``packed_moe.cu`` computes it (keep the two in step)."""
    kc, warps, nt = 64, 8, next(n for n in (1, 2, 4, 8) if 8 * n >= pairs)
    hdr = ((2 * g + 4 + pairs) * 4 + 15) // 16 * 16
    if mode == 1:  # W_b ring + DARE pool and step records + row cursors and bitmap words of up to G + 1 groups
        words = kc // block // 32 + 2
        warp = 4 * mats * 16 * kc * 2 + 256 * 16 + 8 * 36 * 4 + mats * min(g + 1, pairs) * 16 * (8 + 4 * words)
    else:  # W_b ring + BitDelta steps, then the sign table
        warp, hdr = 4 * mats * 16 * kc * 2 + 8 * mats * (64 * kc // 32 + 128), hdr + 4096
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
