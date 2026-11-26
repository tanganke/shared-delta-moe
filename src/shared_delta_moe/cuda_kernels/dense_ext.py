r"""The hand-written Hopper CUDA kernels for dense-delta prefill (``dense_kernel.cu``), built ahead of time and loaded
from a prebuilt library, so that a runtime without nvcc (the vLLM image) can use them.

Build once per torch version, on a machine with nvcc and an sm_90 GPU toolchain (a 2B compute node: ``module load
cuda/13.0``)::

    SHARED_DELTA_MOE_CUDA_DIR=/data/.../sdm-cuda python -m shared_delta_moe.cuda_kernels.dense_ext

and point the runtime at the same directory (``SHARED_DELTA_MOE_CUDA_DIR``; default
``~/.cache/shared_delta_moe/cuda-<torch version>``). Without the library, or off sm90, ``available()`` is False and
the callers fall back to the Triton kernels.

Forward (``cuda_moe``): the pairs sorted by group (``route``), x gathered into sorted order, then two kernels:
``sdm_dense::gate_up`` (:math:`h = \mathrm{SiLU}(x (W_g + D_g)^\top) \odot x (W_u + D_u)^\top`, the weight sum
formed tile by tile in shared memory) and ``sdm_dense::down`` (:math:`h (W_d + D_d)^\top` times the routing weight,
written to each pair's slot); the slots summed in torch. bf16 only.
"""

import os
from pathlib import Path

import torch
from torch import Tensor

from ..triton_kernels.fused import Routing

SRC = Path(__file__).with_name("dense_kernel.cu")
NAME = "sdm_dense"


def lib_dir() -> Path:
    """Directory of the built library: ``SHARED_DELTA_MOE_CUDA_DIR``, else a per-torch-version cache directory."""
    d = os.environ.get("SHARED_DELTA_MOE_CUDA_DIR")
    return Path(d) if d else Path.home() / ".cache" / "shared_delta_moe" / f"cuda-{torch.__version__}"


def build(verbose: bool = True) -> Path:
    """Compile ``dense_kernel.cu`` for sm_90a into ``lib_dir()`` (needs nvcc and ninja) and load it.

    Returns:
        Path of the built ``sdm_dense.so``.
    """
    from torch.utils.cpp_extension import load

    d = lib_dir()
    d.mkdir(parents=True, exist_ok=True)
    load(NAME, [str(SRC)], build_directory=str(d), is_python_module=False, verbose=verbose,
         extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])  # fmt: skip
    return d / f"{NAME}.so"


_LOADED: bool | None = None


def available(device: torch.device | None = None) -> bool:
    """True if the prebuilt library loads (once per process) and the device is sm90 (Hopper)."""
    global _LOADED
    if _LOADED is None:
        so = lib_dir() / f"{NAME}.so"
        try:
            if not hasattr(torch.ops, NAME) or not hasattr(getattr(torch.ops, NAME), "gate_up"):
                torch.ops.load_library(str(so))
            _LOADED = True
        except (OSError, RuntimeError):
            _LOADED = False
    return _LOADED and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == (9, 0)


def supports(x: Tensor, base: tuple[Tensor, Tensor, Tensor], deltas: tuple[Tensor, Tensor, Tensor]) -> bool:
    """What the kernels take: bf16, contiguous, hidden and inter multiples of 8 (16-byte TMA rows)."""
    ts = (x, *base, *deltas)
    return all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in ts) and all(
        n % 8 == 0 for n in base[0].shape
    ) and available(x.device)


def cuda_moe(x: Tensor, r: Routing, base: tuple[Tensor, Tensor, Tensor],
             deltas: tuple[Tensor, Tensor, Tensor]) -> Tensor:  # fmt: skip
    """Parameter-aggregation MoE forward on the CUDA kernels (``supports`` must hold).

    Args:
        x: (T, H) tokens.
        r: The sorted routing (``triton_kernels.fused.route``).
        base: (:math:`W_g` (I, H), :math:`W_u` (I, H), :math:`W_d` (H, I)).
        deltas: dense (:math:`D_g`, :math:`D_u`, :math:`D_d`), (G, N, K) each.

    Returns:
        (T, H) MoE output in x's dtype.
    """
    ops = getattr(torch.ops, NAME)
    h = ops.gate_up(x[r.perm // r.top_k], r.gstart, base[0], base[1], deltas[0], deltas[1])
    o = ops.down(h, r.gstart, r.perm, r.weight, base[2], deltas[2])
    return o if r.top_k == 1 else o.view(r.tokens, r.top_k, -1).sum(1)


if __name__ == "__main__":
    print(build())
