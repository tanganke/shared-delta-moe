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

Build once per torch version, on a machine with nvcc (sm_90a; sm_89 and sm_80 also compiled, not tuned):

.. code-block:: bash

   SHARED_DELTA_MOE_CUDA_DIR=/data/.../sdm-cuda python -m shared_delta_moe.cuda_kernels.packed_ops

This writes ``$SHARED_DELTA_MOE_CUDA_DIR/sdm_packed-<torch version>.so`` (default directory
``~/.cache/shared_delta_moe/cuda``). At runtime the library is loaded by that path, with the same variable and no
nvcc needed; without it, or with ``SHARED_DELTA_MOE_CUDA=0``, everything runs on Triton.
"""

import os
from pathlib import Path

import torch
from torch import Tensor

MAX_PAIRS = 16  # H200, hidden 2560, inter 9216, 5 deltas: CUDA wins up to 16 (token, slot) pairs, Triton from 24
ENABLED = os.environ.get("SHARED_DELTA_MOE_CUDA", "1") != "0"
DIR = Path(os.environ.get("SHARED_DELTA_MOE_CUDA_DIR", Path.home() / ".cache" / "shared_delta_moe" / "cuda"))
LIB = DIR / f"sdm_packed-{torch.__version__}.so"
SRC = Path(__file__).with_name("packed_moe.cu")
ARCHS = ("90a", "89", "80")

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
            torch.ops.load_library(str(LIB))
            _loaded = True
    return _loaded


def supports(x: Tensor, pairs: int, base: tuple[Tensor, Tensor, Tensor], deltas) -> bool:
    """Whether these kernels take this call.

    Args:
        x: (T, H) tokens.
        pairs: (token, slot) pairs, T * top_k.
        base: (:math:`W_g`, :math:`W_u`, :math:`W_d`).
        deltas: The three ``Delta`` of the projections (mode 1 DARE or 2 BitDelta).

    Returns:
        True for 1 .. ``MAX_PAIRS`` pairs of bf16/fp16 on CUDA, hidden and inter multiples of 64, DARE blocks of at
        least 8 (a power of 2) with 16-byte aligned values, BitDelta with one scale per row or blocks of at least 32,
        and the library loaded.
    """
    d = deltas[0]
    if not (d.mode in (1, 2) and 0 < pairs <= MAX_PAIRS and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    if x.shape[1] % 64 or base[0].shape[0] % 64 or deltas[0].d.numel() == 0 or any(w.dtype != x.dtype for w in base):
        return False
    for dl in deltas:
        if dl.mode == 1 and (dl.block < 8 or dl.block & (dl.block - 1) or dl.d.data_ptr() % 16):
            return False
        if dl.mode == 2 and dl.block and dl.block < 32:
            return False
    return available()


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
