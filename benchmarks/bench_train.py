"""Training step of one MoE layer (forward + backward, bf16, CUDA): the two autograd Functions (triton_kernels/train.py:
fused Triton kernels, grouped GEMMs) against the materialised PyTorch path, router-only (frozen weights, the router-training case) or all weights trainable.
Prints ms per step and peak memory above the weights.

python benchmarks/bench_train.py [--top-k 2] [--tokens 4096,25000] [--all-weights]
"""

import argparse
import time

import torch

from shared_delta_moe import DeltaMoE, random_experts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--tokens", default="1024,4096,25000")
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=9216)
    ap.add_argument("--deltas", type=int, default=5)
    ap.add_argument("--all-weights", action="store_true", help="train W_b and the deltas too (default: router only)")
    a = ap.parse_args()
    torch.manual_seed(0)
    full = random_experts(a.hidden, a.inter, a.deltas + 1, torch.bfloat16, "cuda")
    router = torch.randn(a.deltas + 1, a.hidden, dtype=torch.bfloat16, device="cuda") * a.hidden**-0.5
    m = DeltaMoE.from_experts(full[0], full[1:], router, a.top_k)
    del full
    for n, p in m.named_parameters():
        p.requires_grad_(a.all_weights or n == "router.weight")
    from shared_delta_moe.triton_kernels.train import fused_moe_autograd, grouped_moe_autograd

    def via(fn):
        return lambda x: fn(x, *m.topk(x), (m.gate, m.up, m.down), (m.d_gate, m.d_up, m.d_down), m.n_delta)

    forms = {"fused": via(fused_moe_autograd), "grouped": via(grouped_moe_autograd), "materialised": m._forward_param}
    what = "all weights" if a.all_weights else "router only"
    print(f"{torch.cuda.get_device_name()}, hidden {a.hidden}, inter {a.inter}, {a.deltas} deltas + zero expert, "
          f"top-{a.top_k}, bf16, {what}; ms per forward+backward, peak GB above the weights")  # fmt: skip
    print(f"{'tokens':>7} " + " ".join(f"{k:>24}" for k in forms))
    for t in map(int, a.tokens.split(",")):
        x = torch.randn(t, a.hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        dy = torch.randn_like(x)
        row = []
        for f in forms.values():
            try:
                for _ in range(2):  # autotuning
                    f(x).backward(dy)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                s = time.perf_counter()
                for _ in range(5):
                    m.zero_grad(set_to_none=True)
                    x.grad = None
                    f(x).backward(dy)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - s) / 5 * 1e3
                row.append(f"{ms:9.2f} ms {(torch.cuda.max_memory_allocated() - base) / 2**30:6.2f} GB")
            except torch.OutOfMemoryError:
                row.append("OOM")
                torch.cuda.empty_cache()
        print(f"{t:>7} " + " ".join(f"{v:>24}" for v in row), flush=True)


if __name__ == "__main__":
    main()
