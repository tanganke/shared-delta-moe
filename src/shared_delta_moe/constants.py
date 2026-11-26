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
TRITON_MIN_PAIRS = 1
TRITON_MAX_PAIRS_HOPPER = 8192  # sm90+: past this cuBLAS (wgmma) on materialised weights wins
TRITON_MAX_PAIRS_OLDER = 16384  # Ada and older: the Triton GEMM keeps up with cuBLAS longer
PARAM_MIN_ROWS = 256  # materialise W_b + Delta_e only for experts with at least this many routed rows
SORT_ONE_HOT = 16384  # elements of the Triton routing sort's per-chunk one-hot (32768 ran out of shared memory on Ada)
SORT_ONE_HOT_MAX = 65536  # past this even one token per chunk is untested: the materialised path instead
# dense deltas, sm90: from this many (token, slot) pairs per group (n_delta + 1) on, the fused forward materialises
# W_b + Delta and runs grouped GEMMs (cuda_kernels/dense_moe.py). H200 break-even: 5 deltas ~4096 pairs, the 27B
# shard (3 deltas) 2048..4096 (docs/dense-prefill.md)
DENSE_PREFILL_PAIRS_PER_GROUP = 640
