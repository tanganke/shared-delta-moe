"""Per-layer time of each implementation on CUDA, bf16, eager.

python benchmarks/bench.py [--top-k K] [--tokens 1,32,512,8192]
"""

import argparse
import time
from collections.abc import Callable

import torch

from shared_delta_moe import DeltaMoE, random_experts


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--tokens", default="1,32,512,8192")
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=9216)
    ap.add_argument("--deltas", type=int, default=5)
    a = ap.parse_args()
    torch.manual_seed(0)
    full = random_experts(a.hidden, a.inter, a.deltas + 1, torch.bfloat16, "cuda")
    router = torch.randn(a.deltas + 1, a.hidden, dtype=torch.bfloat16, device="cuda")
    m = DeltaMoE.from_experts(full[0], full[1:], router, a.top_k)
    del full
    experts = m.dense_experts()
    forms: dict[str, Callable] = {
        "dense loop": lambda x: m._forward_dense(x, experts),
        "dense grouped": lambda x: m._forward_dense_grouped(x, experts),
        "param": m._forward_param,
        "param hybrid": lambda x: m._forward_param(x, min_rows=256),
        "param triton": m._forward_param_triton,
        "matrix loop": m._forward_matrix,
        "matrix grouped": m._forward_matrix_grouped,
    }
    print(
        f"{torch.cuda.get_device_name()}, hidden {a.hidden}, inter {a.inter}, {a.deltas} deltas + zero expert, top-{a.top_k}, bf16, eager"
    )
    print(f"{'tokens':>7} " + " ".join(f"{k:>14}" for k in forms) + "   (ms per layer)")
    for t in map(int, a.tokens.split(",")):
        x = torch.randn(t, a.hidden, dtype=torch.bfloat16, device="cuda")
        row = []
        for f in forms.values():
            for _ in range(3):  # also runs Triton's autotuning
                f(x)
            torch.cuda.synchronize()
            s = time.perf_counter()
            for _ in range(20):
                f(x)
            torch.cuda.synchronize()
            row.append((time.perf_counter() - s) / 20 * 1e3)
        print(f"{t:>7} " + " ".join(f"{v:>14.3f}" for v in row))


if __name__ == "__main__":
    main()
