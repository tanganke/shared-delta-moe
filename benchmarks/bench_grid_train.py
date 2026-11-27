"""One Qwen3.5-27B MoE layer (H 5120, I 17408, 2-bit grids, group 128), router-only training under bf16 autocast:
the materialised path against GridDeltaMoE._forward_decoded. ms of forward / backward, peak GiB, and the decode alone.

    python benchmarks/bench_grid_train.py [B,A-top2,A-soft]
"""

import sys
import time

import torch

from shared_delta_moe.constants import PARAM_MIN_ROWS
from shared_delta_moe.models.grid import GridHierDeltaMoE

H, I = 5120, 17408
# (group sizes, zero experts per group, top-k per group, groups per token): the moe27-14 B / A layouts
CFG = {"B": ([14], [1], 2, None), "A-top2": ([4, 4, 4, 2], [1] * 4, 2, 2), "A-soft": ([4, 4, 4, 2], [1] * 4, 5, 2)}


def build(sizes, zeros, k1, k0):
    torch.manual_seed(0)
    m = GridHierDeltaMoE(H, I, sizes, zeros, k1, k0, "cuda", torch.bfloat16, 4, 128, False)
    with torch.no_grad():
        for n, b in m.named_buffers():
            b.random_(-(2**31), 2**31 - 1)  # random codes
        for n, p in m.named_parameters():
            if n.startswith("s_"):
                p.uniform_(2e-4, 6e-4)
            elif n.startswith("z_"):
                p.copy_(-1.5 * getattr(m, "s_" + n[2:]))
            else:
                p.normal_(std=0.02)
        m.router.weight.data = m.router.weight.data.float()  # fp32 router, as train_hier_router.py
    for n, p in m.named_parameters():
        p.requires_grad_(n == "router.weight")
    return m


def run(m, x0, dy, f, reps=3):
    ts = []
    for i in range(reps + 1):
        m.router.weight.grad = None
        x = x0.clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16):
            y = f(m, x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        y.backward(dy)
        torch.cuda.synchronize()
        if i:  # the first run warms up (autotuning, allocator)
            ts.append((t1 - t0, time.perf_counter() - t1))
    fw, bw = (sorted(v)[len(v) // 2] * 1e3 for v in zip(*ts))
    return fw, bw, torch.cuda.max_memory_allocated() / 2**30


def decode_ms(m, reps=5):
    with torch.no_grad():
        m._decoded()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            m._decoded()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


for name in (sys.argv[1] if len(sys.argv) > 1 else ",".join(CFG)).split(","):
    m = build(*CFG[name])
    print(f"{name}: decode of {m.n_delta + 1} experts x 3 projections {decode_ms(m):.1f} ms", flush=True)
    for T in (2048, 10240):
        torch.manual_seed(1)
        x0 = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x0) * 1e-2
        for meth, f in (("materialised", lambda m, x: m._forward_param(x, min_rows=PARAM_MIN_ROWS)),
                        ("decoded", lambda m, x: m(x))):  # fmt: skip
            fw, bw, mem = run(m, x0, dy, f)
            print(f"{name:7s} T={T:5d} pairs={T * m.top_k:6d} {meth:12s} fwd {fw:7.1f} ms  bwd {bw:7.1f} ms  peak {mem:5.1f} GiB",
                  flush=True)  # fmt: skip
            torch.cuda.empty_cache()
    del m
    torch.cuda.empty_cache()
