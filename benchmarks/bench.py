"""Per-layer time of each implementation on CUDA, bf16, eager.

python benchmarks/bench.py [--top-k K] [--tokens 1,32,512,8192] [--dare 0.2:1,0.2:64] [--only NAME,...]
"""

import argparse
import time
from collections.abc import Callable

import torch

from shared_delta_moe import BitDeltaMoE, DareDeltaMoE, DeltaMoE, random_experts


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--tokens", default="1,32,512,8192")
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=9216)
    ap.add_argument("--deltas", type=int, default=5)
    ap.add_argument(
        "--dare",
        default="0.2:1,0.2:64",
        help="seeded-DARE settings keep:block, comma-separated",
    )
    ap.add_argument(
        "--only", default="", help="comma-separated implementation names to run"
    )
    a = ap.parse_args()
    torch.manual_seed(0)
    full = random_experts(a.hidden, a.inter, a.deltas + 1, torch.bfloat16, "cuda")
    router = torch.randn(a.deltas + 1, a.hidden, dtype=torch.bfloat16, device="cuda")
    m = DeltaMoE.from_experts(full[0], full[1:], router, a.top_k)
    del full
    experts = m.dense_experts()
    dare = {
        f"dare {s}": DareDeltaMoE.from_delta_moe(
            m, float(s.split(":")[0]), int(s.split(":")[1])
        )
        for s in a.dare.split(",")
        if s
    }
    forms: dict[str, Callable] = {
        "dense loop": lambda x: m._forward_dense(x, experts),
        "dense grouped": lambda x: m._forward_dense_grouped(x, experts),
        "param": m._forward_param,
        "param hybrid": lambda x: m._forward_param(x, min_rows=256),
        "param triton": m._forward_param_triton,
        "matrix loop": m._forward_matrix,
        "matrix grouped": m._forward_matrix_grouped,
        "param triton graph": m._forward_param_triton,  # replayed from a CUDA graph (needs no host sync)
    } | {k: d._forward_param_triton for k, d in dare.items()} | {
        "bitdelta": BitDeltaMoE.from_delta_moe(m, "row")._forward_param_triton,
        "bitdelta graph": BitDeltaMoE.from_delta_moe(m, "row")._forward_param_triton,
        "bitdelta b32": BitDeltaMoE.from_delta_moe(m, 32)._forward_param_triton,  # one scale per 1 x 32 block
        "bitdelta b128": BitDeltaMoE.from_delta_moe(m, 128)._forward_param_triton,
    }
    if a.only:
        forms = {k: f for k, f in forms.items() if k in a.only.split(",")}
    print(
        f"{torch.cuda.get_device_name()}, hidden {a.hidden}, inter {a.inter}, {a.deltas} deltas + zero expert, top-{a.top_k}, bf16, eager"
    )
    print(f"{'tokens':>7} " + " ".join(f"{k:>14}" for k in forms) + "   (ms per layer)")
    for t in map(int, a.tokens.split(",")):
        x = torch.randn(t, a.hidden, dtype=torch.bfloat16, device="cuda")
        row = []
        for name, f in forms.items():
            for _ in range(3):  # also runs Triton's autotuning
                f(x)
            if name.endswith(" graph"):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    f(x)
                f = lambda _: g.replay()  # noqa: E731
            torch.cuda.synchronize()
            s = time.perf_counter()
            for _ in range(20):
                f(x)
            torch.cuda.synchronize()
            row.append((time.perf_counter() - s) / 20 * 1e3)
        print(f"{t:>7} " + " ".join(f"{v:>14.3f}" for v in row))


if __name__ == "__main__":
    main()
