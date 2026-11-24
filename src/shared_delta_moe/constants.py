"""Feature detection and the thresholds the public forwards use to pick an implementation."""

import importlib.util

import torch

HAS_TRITON = importlib.util.find_spec("triton") is not None

# ponytail: thresholds measured on H200 and RTX 6000 Ada only (hidden 2560, inter 9216, 6 experts); retune for others
GROUPED_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
)  # what F.grouped_mm accepts
GROUPED_MIN_PAIRS = (
    2  # one (token, slot) pair: sorting and counting cost more than the loop saves
)
TRITON_DTYPES = (torch.bfloat16, torch.float16)
TRITON_MIN_PAIRS = 2
TRITON_MAX_PAIRS_HOPPER = (
    4096  # sm90+: past this cuBLAS (wgmma) on materialised weights wins
)
TRITON_MAX_PAIRS_OLDER = (
    8192  # Ada and older: the Triton GEMM keeps up with cuBLAS longer
)
PARAM_MIN_ROWS = 256  # materialise W_b + Delta_e only for experts with at least this many routed rows
