"""Dense-delta MoE forward at prefill sizes, per implementation: device time per layer (torch.profiler, sum of kernel
times) and % of H200 peak (989 TFLOPS bf16, 4.8 TB/s).

python benchmarks/bench_dense_prefill.py [--kernels]   (--kernels: per-kernel breakdown too)

triton   the fused Triton kernels (route, gate_up, down)
grouped  route + W_b + Delta materialised into scratch + F.grouped_mm (cuda_kernels/dense_moe.py)
plain    F.grouped_mm on pre-materialised experts (forward_dense, incl. its routing; not a delta model)
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

from shared_delta_moe import DeltaMoE, HierDeltaMoE, constants, random_experts
from shared_delta_moe.cuda_kernels import dense_moe
from shared_delta_moe.triton_kernels import fused as F

BW, FL = 4.8e12, 989e12
KERNELS = "--kernels" in sys.argv


def flat(H, I, n, k):
    full = random_experts(H, I, n + 1, torch.bfloat16, "cuda")
    r = torch.randn(n + 1, H, dtype=torch.bfloat16, device="cuda") * H**-0.5
    return DeltaMoE.from_experts(full[0], full[1:], r, k)


def hier(H, I, sizes, k1):
    full = random_experts(H, I, sum(sizes) + 1, torch.bfloat16, "cuda")
    gs, c = [], 1
    for s in sizes:
        gs.append(full[c : c + s])
        c += s
    r = torch.randn(len(sizes) + sum(sizes), H, dtype=torch.bfloat16, device="cuda") * H**-0.5
    return HierDeltaMoE.from_experts(full[0], gs, r, [0] * len(sizes), k1)


def device_ms(f, it=10):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(it):
            f()
        torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    tot = sum(e.self_device_time_total for e in ev) / it / 1e3
    if KERNELS:
        for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:8]:
            print(f"      {e.self_device_time_total / it / 1e3:7.3f}  {e.key[:100]}")
    return tot


@torch.no_grad()
def main():
    constants.DENSE_PREFILL_PAIRS_PER_GROUP = 1 << 62  # "triton" = the Triton kernels alone
    torch.manual_seed(0)
    cases = [("flat k1", lambda: flat(2560, 9216, 5, 1), (512, 2048, 4096, 8192)),
             ("flat k2", lambda: flat(2560, 9216, 5, 2), (512, 2048, 4096, 8192)),
             ("27B tp2", lambda: hier(5120, 8704, (2, 1), 1), (512, 1024, 2048, 8192))]  # fmt: skip
    print(f"{'config':>8} {'T':>5} {'pairs':>6} {'impl':>8} {'ms':>7} {'%FL':>5} {'%BW':>5}")
    for name, make, Ts in cases:
        m = make()
        experts = m.dense_experts()
        I, H = m.gate.shape
        base = (m.gate, m.up, m.down)
        dl = (m.d_gate, m.d_up, m.d_down)
        for T in Ts:
            x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
            idx, rw = m.topk(x)
            P = idx.numel()
            used = idx[idx < m.n_delta].unique().numel()
            wbytes = 3 * I * H * 2 * (1 + used)
            impls = {
                "triton": lambda: F.fused_moe(x, idx, rw, m.n_delta, base, tuple(F.Delta(0, d) for d in dl)),
                "grouped": lambda: dense_moe.grouped_moe(x, F.route(idx, rw, m.n_delta), base, dl),
                "plain": lambda: m._forward_dense_grouped(x, experts),
            }
            ref = m._forward_param(x).float()
            for impl, f in impls.items():
                err = ((f().float() - ref).abs().max() / ref.abs().max()).item()  # (norms overflow float32 here)
                ms = device_ms(f)
                print(f"{name:>8} {T:>5} {P:>6} {impl:>8} {ms:7.3f} {100 * 6 * P * I * H / (ms * 1e-3) / FL:5.0f} "
                      f"{100 * wbytes / (ms * 1e-3) / BW:5.0f}   (max err / max vs _forward_param {err:.1e})", flush=True)
        del m, experts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
