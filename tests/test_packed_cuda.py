"""The prebuilt CUDA kernels for packed deltas (cuda_kernels/packed_moe.cu) match the materialised forward: DARE block
sizes, BitDelta scale variants, ragged groups, zero experts, top-2, split-K, fp16, a CUDA graph and torch.compile.
Skipped unless the library is built (python -m shared_delta_moe.cuda_kernels.packed_ops)."""

import itertools

import pytest
import torch
from test_delta_moe import make

from shared_delta_moe import BitDeltaMoE, DareDeltaMoE
from shared_delta_moe.cuda_kernels import packed_ops

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and packed_ops.available()), reason="not built")


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def check(mod, x, ts=(1, 3, 8, 16)):
    for t in ts:
        if t * mod.top_k > packed_ops.MAX_PAIRS:
            continue
        xs = x[:t]
        assert packed_ops.supports(xs, t * mod.top_k, (mod.gate, mod.up, mod.down), mod._kernel_deltas())
        with torch.no_grad():
            got, want = mod._forward_param_triton(xs), mod._forward_param(xs)
        assert rel(got, want) < 2e-2, (t, rel(got, want))


# (hidden, inter): multiples of 128 (the kernels' K step); inter 2048 so that down splits K more
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


def test_fp16_and_falls_back():
    m, _, x = make(1, 2, 256, 384, torch.float16, "cuda")
    check(DareDeltaMoE.from_delta_moe(m, 0.3, 64, seed=5), x)
    check(BitDeltaMoE.from_delta_moe(m, "row"), x)
    d = DareDeltaMoE.from_delta_moe(m, 0.3, 4, seed=5)  # block < 8: Triton
    assert not packed_ops.supports(x[:4], 8, (d.gate, d.up, d.down), d._kernel_deltas())
    assert rel(d._forward_param_triton(x[:4]), d._forward_param(x[:4])) < 2e-2


def test_cuda_graph_and_compile():
    m, _, x = make(1, 2, 256, 384, torch.bfloat16, "cuda")
    for mod in (DareDeltaMoE.from_delta_moe(m, 0.2, 64, seed=1), BitDeltaMoE.from_delta_moe(m, 64)):
        xs = x[:8].clone()
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                mod._forward_param_triton(xs)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = mod._forward_param_triton(xs)
            for _ in range(2):
                xs.copy_(torch.randn_like(xs))
                g.replay()
                assert rel(out, mod._forward_param(xs)) < 2e-2
            torch._dynamo.reset()
            f = torch.compile(mod._forward_param_triton, fullgraph=True)
            assert rel(f(x[:5]), mod._forward_param(x[:5])) < 2e-2
