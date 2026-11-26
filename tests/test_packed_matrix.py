"""forward_matrix of DareDeltaMoE / BitDeltaMoE on the prebuilt CUDA kernels (cuda_kernels/packed_ops.py, matrix
ops) against _forward_matrix on the unpacked deltas."""

import itertools

import pytest
import torch
from test_delta_moe import make

from shared_delta_moe import BitDeltaMoE, DareDeltaMoE, DeltaMoE, random_experts
from shared_delta_moe.cuda_kernels import packed_ops

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and packed_ops.available()), reason="not built")


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def check(mod, x, ts=(1, 3, 8, 37, 100, 200)):
    for t in ts:
        xs = x[:t]
        with torch.no_grad():
            assert mod._matrix_cuda(xs), t
            got, want = mod.forward_matrix(xs), mod._forward_matrix(xs)
        assert rel(got, want) < 1e-2, (t, rel(got, want))


SIZES = ((256, 384), (384, 640), (256, 2048))


@pytest.mark.parametrize("z,k", list(itertools.product((0, 1, 2), (1, 2))))
def test_dare(z, k):
    for (hidden, inter), (keep, block) in itertools.product(SIZES, ((0.2, 64), (0.5, 8), (0.3, 16), (0.25, 32), (0.5, 128))):
        if hidden % block or inter % block:
            continue
        m, _, x = make(z, k, hidden, inter, torch.bfloat16, "cuda")
        check(DareDeltaMoE.from_delta_moe(m, keep, block, seed=2**33 + 1), x)


@pytest.mark.parametrize("z,k", list(itertools.product((0, 1, 2), (1, 2))))
def test_bitdelta(z, k):
    for (hidden, inter), scale in itertools.product(SIZES, ("matrix", "row", 32, 64, 128)):
        if isinstance(scale, int) and (hidden % scale or inter % scale):
            continue
        m, _, x = make(z, k, hidden, inter, torch.bfloat16, "cuda")
        check(BitDeltaMoE.from_delta_moe(m, scale), x)


def test_fp16_many_deltas_and_slots():
    m, _, x = make(1, 2, 256, 384, torch.float16, "cuda")
    check(DareDeltaMoE.from_delta_moe(m, 0.3, 64, seed=5), x)
    check(BitDeltaMoE.from_delta_moe(m, "row"), x)
    torch.manual_seed(0)
    full = random_experts(256, 384, 17, torch.bfloat16, "cuda")  # 16 deltas, top-3 (many slots per token)
    m = DeltaMoE.from_experts(full[0], full[1:], torch.randn(17, 256, dtype=torch.bfloat16, device="cuda"), top_k=3,
                              num_zero_experts=1)  # fmt: skip
    x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    check(DareDeltaMoE.from_delta_moe(m, 0.3, 8, seed=3), x, ts=(1, 5, 16, 64))
    check(BitDeltaMoE.from_delta_moe(m, 32), x, ts=(1, 5, 16, 64))


def test_falls_back(monkeypatch):
    m, _, x = make(1, 2, 256, 384, torch.bfloat16, "cuda")
    d = DareDeltaMoE.from_delta_moe(m, 0.3, 4, seed=5)  # block < 8
    b = BitDeltaMoE.from_delta_moe(m, "row")
    with torch.no_grad():
        assert not d._matrix_cuda(x[:4])
        assert b._matrix_cuda(x[:4])
        monkeypatch.setattr(packed_ops, "MATRIX_MAX_PAIRS", 8)
        assert not b._matrix_cuda(x[:8])  # too many pairs
    xg = x[:4].clone().requires_grad_()
    assert not b._matrix_cuda(xg)  # a gradient: the inherited forward
    b.forward_matrix(xg).sum().backward()
    assert xg.grad is not None


def test_cuda_graph_and_compile():
    m, _, x = make(1, 2, 256, 384, torch.bfloat16, "cuda")
    for mod in (DareDeltaMoE.from_delta_moe(m, 0.2, 64, seed=1), BitDeltaMoE.from_delta_moe(m, 64)):
        xs = x[:8].clone()
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                mod.forward_matrix(xs)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = mod.forward_matrix(xs)
            for _ in range(2):
                xs.copy_(torch.randn_like(xs))
                g.replay()
                assert rel(out, mod._forward_matrix(xs)) < 1e-2
            torch._dynamo.reset()
            f = torch.compile(mod.forward_matrix, fullgraph=True)
            assert rel(f(xs), mod._forward_matrix(xs)) < 1e-2
