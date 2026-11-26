"""Training step of one MoE layer (forward + backward, bf16, CUDA): forward_param with triton_train (fused Triton
forward + backward, triton_kernels/train.py) against the materialised PyTorch path; --train picks what has a gradient.
--groups G makes a HierDeltaMoE: G groups of --deltas deltas + 1 zero expert, dense group routing, top-k per group.
Prints ms per step and peak memory above the weights.

python benchmarks/bench_train.py [--top-k 2] [--tokens 4096,25000] [--train router|base|deltas|all] [--groups 4]
"""

import argparse
import time

import torch

from shared_delta_moe import DeltaMoE, HierDeltaMoE, random_experts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--tokens", default="1024,4096,25000")
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=9216)
    ap.add_argument("--deltas", type=int, default=5)
    ap.add_argument("--groups", type=int, default=0, help="HierDeltaMoE with this many groups (0: plain DeltaMoE)")
    ap.add_argument("--train", default="router", choices=["router", "base", "deltas", "all"],
                    help="what gets a gradient besides x and the router: nothing, W_b, the deltas, or both")
    ap.add_argument("--no-remat", action="store_true", help="materialised path keeps W_b + Delta_e for the backward (remat=False)")
    a = ap.parse_args()
    torch.manual_seed(0)
    G = a.groups
    full = random_experts(a.hidden, a.inter, a.deltas * max(G, 1) + 1, torch.bfloat16, "cuda")
    if G:
        rows = G + G * (a.deltas + 1)
        router = torch.randn(rows, a.hidden, dtype=torch.bfloat16, device="cuda") * a.hidden**-0.5
        groups = [full[1 + i * a.deltas : 1 + (i + 1) * a.deltas] for i in range(G)]
        m: DeltaMoE = HierDeltaMoE.from_experts(full[0], groups, router, [1] * G, a.top_k)
    else:
        router = torch.randn(a.deltas + 1, a.hidden, dtype=torch.bfloat16, device="cuda") * a.hidden**-0.5
        m = DeltaMoE.from_experts(full[0], full[1:], router, a.top_k)
    del full
    base, deltas = {"gate", "up", "down"}, {"d_gate", "d_up", "d_down"}
    train = {"router": set(), "base": base, "deltas": deltas, "all": base | deltas}[a.train]
    for n, p in m.named_parameters():
        p.requires_grad_(n == "router.weight" or n in train)

    m.remat = not a.no_remat

    def triton(x):
        m.triton_train = True
        assert m._use_triton(x)
        return m.forward_param(x)

    def materialised(x):
        m.triton_train = False
        return m.forward_param(x)

    forms = {"triton": triton, "materialised": materialised}
    what = (f"router + {a.train}" if train else "router only") + (", no remat" if a.no_remat else "")
    kind = f"{G} groups x ({a.deltas} deltas + 1 zero), dense group routing, top-{a.top_k} per group" if G else (
        f"{a.deltas} deltas + zero expert, top-{a.top_k}")  # fmt: skip
    print(f"{torch.cuda.get_device_name()}, hidden {a.hidden}, inter {a.inter}, {kind}, bf16, {what}; "
          "ms per forward+backward, peak GB above the weights")  # fmt: skip
    print(f"{'tokens':>7} " + " ".join(f"{k:>24}" for k in forms))
    for t in map(int, a.tokens.split(",")):
        x = torch.randn(t, a.hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        dy = torch.randn_like(x)
        row = []
        for f in forms.values():
            try:
                for _ in range(2):  # autotuning
                    f(x).backward(dy)
                m.zero_grad(set_to_none=True)
                x.grad = None
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
                m.zero_grad(set_to_none=True)
                x.grad = None
                torch.cuda.empty_cache()
        print(f"{t:>7} " + " ".join(f"{v:>24}" for v in row), flush=True)


if __name__ == "__main__":
    main()
