"""The fused Triton forward (triton_kernels/fused.py): the sort kernel, every autotune config of the
two GEMM kernels and split-K == the materialised parameter aggregation, and capture in a CUDA graph.
Needs CUDA and Triton (skipped otherwise). python tests/test_fused.py also runs it."""

import itertools

import torch
from test_delta_moe import make

from shared_delta_moe import DareDeltaMoE
from shared_delta_moe.constants import HAS_TRITON

GPU = torch.cuda.is_available() and HAS_TRITON


def rel(got, want):
    return ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()


def test_sort_matches_torch():
    """The sort kernel: a stable permutation of the pairs by group (zero experts -> group n), group starts."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    from shared_delta_moe.triton_kernels.fused import route

    for (z, k), t in itertools.product(((0, 1), (1, 1), (2, 2), (1, 3), (3, 6)), (1, 37, 1500)):
        n = 3
        idx = torch.rand(t, n + z, device="cuda").argsort(-1)[:, :k]  # k distinct experts per token
        w = torch.rand(t, k, device="cuda")
        r = route(idx, w, n)
        grp = torch.where(idx < n, idx, n).flatten()  # by pair id
        perm = r.perm.long()
        assert sorted(perm.tolist()) == list(range(t * k)), "a permutation of the pairs"
        g_sorted = grp[perm]
        assert (g_sorted[1:] >= g_sorted[:-1]).all(), "sorted by group"
        same = g_sorted[1:] == g_sorted[:-1]
        assert (perm[1:][same] > perm[:-1][same]).all(), "stable"
        counts = torch.bincount(grp, minlength=n + 1)
        assert r.gstart.tolist() == [0] + counts.cumsum(0).tolist(), (z, k, t)
        assert torch.equal(r.weight, w.flatten())


def _cases():
    """(module, x) for dense and DARE, top-1 and top-2, ragged hidden / inter (not multiples of the tiles)."""
    # every config compiles once per (top-k, delta mode, DARE block): four combinations keep it to minutes
    for (z, k), dare in (((1, 1), None), ((2, 2), None), ((2, 2), (0.5, 64)), ((1, 1), (0.25, 1))):
        m, _, x = make(z, k, 192, 320, torch.bfloat16, "cuda")
        yield (DareDeltaMoE.from_delta_moe(m, *dare, seed=5) if dare else m), x


def test_every_config_matches_materialised():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    from shared_delta_moe.triton_kernels import fused as F

    for m, x in _cases():
        dg, du, dd = m._kernel_deltas()
        for t in (1, 37, 200):
            xs = x[:t]
            want = m._forward_param(xs)
            r = F.route(*m.topk(xs), m.n_delta)
            h_ref = F.gate_up(xs, r, m.gate, m.up, dg, du, F.GATE_UP_CONFIGS[0])
            for c in F.GATE_UP_CONFIGS:
                if dg.mode == 1 and c.kwargs["BK"] % dg.block and dg.block % c.kwargs["BK"]:
                    continue
                try:
                    h = F.gate_up(xs, r, m.gate, m.up, dg, du, c)
                except ValueError:  # too much shared memory for this GPU
                    continue
                assert rel(h, h_ref) < 2e-2, ("gate_up", c, t)
            for c in F.DOWN_CONFIGS:
                if dd.mode == 1 and c.kwargs["BK"] % dd.block and dd.block % c.kwargs["BK"]:
                    continue
                try:
                    y = F.down(h_ref, r, m.down, dd, c, split_k=1)
                except ValueError:
                    continue
                assert rel(y, want) < 2e-2, ("down", c, t, rel(y, want))
            if dd.mode == 0:
                for s in (2, 4):
                    y = F.down(h_ref, r, m.down, dd, F.DOWN_CONFIGS[0], split_k=s)
                    assert rel(y, want) < 2e-2, ("split-K", s, t)
            print(f"  configs ok: {type(m).__name__} top-{m.top_k} mode {dg.mode} block {dg.block} tokens {t}", flush=True)
            got = m._forward_param_triton(xs)  # autotuned
            assert rel(got, want) < 2e-2, (type(m).__name__, t)


def test_rounds_like_the_model():
    """Real size (hidden 2560, inter 9216), deltas small next to W_b as in a fine-tuned MoE, 200 tokens: the fused
    forward matches _forward_param, which rounds g, u, SiLU(g), h and y to bf16 like the original model, to 1e-3 (norm
    relative). Computing the epilogue in fp32 instead was closer to exact but 4.5e-3 away from it per layer, which
    lowered argmax agreement of a 32-layer Qwen3.5 with the original from 98.8% to 98.4%."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    from shared_delta_moe import DeltaMoE

    red = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False  # cuBLAS split-K sums in bf16 else
    try:
        g = torch.Generator().manual_seed(0)
        H, I, n = 2560, 9216, 5
        base = [torch.randn(I, H, generator=g) * 0.02, torch.randn(I, H, generator=g) * 0.02,
                torch.randn(H, I, generator=g) * 0.02]  # fmt: skip
        experts = [[b + torch.randn(b.shape, generator=g) * 0.004 for b in base] for _ in range(n)]
        to = dict(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(200, H, generator=g).to(**to)
        for k in (1, 2):
            m = DeltaMoE.from_experts([b.to(**to) for b in base], [[w.to(**to) for w in e] for e in experts],
                                      (torch.randn(n + 1, H, generator=g) * H**-0.5).to(**to), k)  # fmt: skip
            with torch.no_grad():
                a, b = m._forward_param_triton(x).float(), m._forward_param(x).float()
            err = ((a - b).norm() / b.norm()).item()
            assert err < 1e-3, (k, err)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = red


def test_empty_and_all_zero_expert():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    m, _, x = make(1, 1, 192, 320, torch.bfloat16, "cuda")
    assert m._forward_param_triton(x[:0]).shape == (0, 192)
    with torch.no_grad():
        m.router.weight[-1] += 100  # every token to the zero expert: only group G is non-empty
    want = m._forward_param(x)
    assert rel(m._forward_param_triton(x), want) < 2e-2


def test_offsets_past_int32():
    """h (pairs x K) past 2^31 elements: row offsets must be int64 (they overflowed in int32). Needs ~5 GB."""
    if not GPU or torch.cuda.get_device_properties(0).total_memory < 24 * 2**30:
        print("skipped: no CUDA/Triton or < 24 GB")
        return
    from shared_delta_moe.triton_kernels import fused as F

    P, K, N = 140000, 16384, 64
    torch.manual_seed(0)
    h = torch.randn(P, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
    d = torch.zeros(1, N, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(P, 1, device="cuda", dtype=torch.long)  # every pair in group 0: perm = identity
    r = F.route(idx, torch.ones(P, 1, device="cuda"), 1)
    y = F.down(h, r, w, F.Delta(0, d), split_k=1)
    rows = torch.tensor([0, 2**31 // K - 1, 2**31 // K + 5, P - 1], device="cuda")
    assert rel(y[rows], h[rows].float() @ w.float().T) < 2e-2


def test_cuda_graph():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    for k in (1, 2):
        m, _, x = make(1, k, 256, 384, torch.bfloat16, "cuda")
        xs = x[:8].clone()
        with torch.no_grad():
            m._forward_param_triton(xs)  # autotune outside the capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                m._forward_param_triton(xs)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = m._forward_param_triton(xs)
            for _ in range(3):  # new inputs, new routing, same graph
                xs.copy_(torch.randn_like(xs))
                g.replay()
                assert rel(out, m._forward_param(xs)) < 2e-2, k


def test_torch_compile():
    """One opaque op under torch.compile: no graph break (fullgraph), same output, dense and DARE."""
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    m, _, x = make(1, 2, 256, 384, torch.bfloat16, "cuda")
    for mod in (m, DareDeltaMoE.from_delta_moe(m, 0.5, 64, seed=1)):
        torch._dynamo.reset()
        f = torch.compile(mod._forward_param_triton, fullgraph=True)
        with torch.no_grad():
            assert rel(f(x[:37]), mod._forward_param(x[:37])) < 2e-2, type(mod).__name__


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
