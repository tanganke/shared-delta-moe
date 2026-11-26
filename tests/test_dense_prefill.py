"""The dense prefill path (cuda_kernels/dense_moe.py: materialised W_b + Delta and grouped GEMMs) == the materialised parameter aggregation ``_forward_param``: ragged and empty groups, every
pair on the zero expert, top-2, sizes that are not multiples of the tiles. Needs CUDA and Triton (skipped otherwise).
python tests/test_dense_prefill.py also runs it."""

import torch
from test_delta_moe import make
from test_fused import GPU, rel


def _impls():
    from shared_delta_moe.cuda_kernels import dense_moe

    yield "grouped", dense_moe.grouped_moe


def _cases():
    """(module, x): top-1 / top-2, one or two zero experts, ragged hidden / inter; tokens 1 .. 700."""
    for (z, k), (hidden, inter) in (((1, 1), (192, 320)), ((2, 2), (192, 320)), ((1, 2), (264, 712))):
        m, _, x = make(z, k, hidden, inter, torch.bfloat16, "cuda")
        x = torch.cat((x, torch.randn(500, hidden, dtype=x.dtype, device=x.device)))
        yield m, x


def _check(m, x, tag):
    from shared_delta_moe.triton_kernels import fused as F

    want = m._forward_param(x)
    r = F.route(*m.topk(x), m.n_delta)
    for name, f in _impls():
        got = f(x, r, (m.gate, m.up, m.down), (m.d_gate, m.d_up, m.d_down))
        assert got.shape == want.shape and got.dtype == want.dtype
        assert rel(got, want) < 2e-2, (name, tag, rel(got, want))


def test_matches_materialised():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    for m, x in _cases():
        for t in (1, 37, 300, 700):
            _check(m, x[:t], (m.top_k, m.n_delta, x.shape[1], t))


def test_empty_groups_and_zero_expert():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    m, _, x = make(1, 1, 192, 320, torch.bfloat16, "cuda")
    with torch.no_grad():
        m.router.weight[1] -= 100  # delta 1 never picked: an empty group between two non-empty ones
    _check(m, x, "delta 1 empty")
    with torch.no_grad():
        m.router.weight[-1] += 1000  # every pair on the zero expert: only the base group has rows
    _check(m, x, "all zero expert")


def test_forward_param_dispatch():
    """forward_param past the threshold goes through the prefill path (and still matches); empty input still works."""
    if not GPU or torch.cuda.get_device_capability()[0] != 9:
        print("skipped: needs sm90")
        return
    from shared_delta_moe import constants
    from shared_delta_moe.cuda_kernels import dense_moe

    m, _, x = make(1, 2, 192, 320, torch.bfloat16, "cuda")
    calls, prefill = [], dense_moe.grouped_moe
    old = constants.DENSE_PREFILL_PAIRS_PER_GROUP
    constants.DENSE_PREFILL_PAIRS_PER_GROUP, dense_moe.grouped_moe = 8, lambda *a: calls.append(1) or prefill(*a)
    try:
        with torch.no_grad():
            assert rel(m.forward_param(x), m._forward_param(x)) < 2e-2
            assert m.forward_param(x[:0]).shape == (0, 192)
    finally:
        constants.DENSE_PREFILL_PAIRS_PER_GROUP, dense_moe.grouped_moe = old, prefill
    assert calls, "prefill path not taken"


def test_torch_compile_prefill():
    """The prefill path inside the one opaque op under torch.compile (fullgraph): same output."""
    if not GPU or torch.cuda.get_device_capability()[0] != 9:
        print("skipped: needs sm90")
        return
    from shared_delta_moe import constants

    m, _, x = make(1, 2, 192, 320, torch.bfloat16, "cuda")
    old, constants.DENSE_PREFILL_PAIRS_PER_GROUP = constants.DENSE_PREFILL_PAIRS_PER_GROUP, 8
    try:
        torch._dynamo.reset()
        f = torch.compile(m._forward_param_triton, fullgraph=True)
        with torch.no_grad():
            assert rel(f(x), m._forward_param(x)) < 2e-2
    finally:
        constants.DENSE_PREFILL_PAIRS_PER_GROUP = old


def test_cuda_graph_prefill():
    """Capture past the threshold: with the scratch warmed up on the capture stream the graph runs the grouped path;
    with none (warm-up on another stream) it stays on Triton and allocates no scratch while capturing. Both replay
    correctly with new inputs."""
    if not GPU or torch.cuda.get_device_capability()[0] != 9:
        print("skipped: needs sm90")
        return
    from shared_delta_moe import constants
    from shared_delta_moe.cuda_kernels import dense_moe

    m, _, x = make(1, 2, 192, 320, torch.bfloat16, "cuda")
    old, constants.DENSE_PREFILL_PAIRS_PER_GROUP = constants.DENSE_PREFILL_PAIRS_PER_GROUP, 8
    calls, grouped = [], dense_moe.grouped_moe
    dense_moe.grouped_moe = lambda *a: calls.append(1) or grouped(*a)
    try:
        for warm in (True, False):
            dense_moe._SCRATCH.clear()
            xs = x.clone()
            s = torch.cuda.Stream()
            with torch.no_grad():
                constants.DENSE_PREFILL_PAIRS_PER_GROUP = 1 << 62
                m._forward_param_triton(xs)  # the Triton kernels compiled and tuned outside the capture
                constants.DENSE_PREFILL_PAIRS_PER_GROUP = 8
                m._forward_param_triton(xs)  # default stream: the add kernel compiled; scratch for this stream only
                torch.cuda.synchronize()
                if warm:
                    with torch.cuda.stream(s):
                        m._forward_param_triton(xs)
                    torch.cuda.synchronize()
                n = len(dense_moe._SCRATCH)
                calls.clear()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    out = m._forward_param_triton(xs)
                assert len(dense_moe._SCRATCH) == n, "scratch allocated during capture"
                assert bool(calls) == warm, (warm, calls)
                for _ in range(2):  # new inputs, new routing, same graph
                    xs.copy_(torch.randn_like(xs))
                    g.replay()
                    assert rel(out, m._forward_param(xs)) < 2e-2, warm
    finally:
        constants.DENSE_PREFILL_PAIRS_PER_GROUP, dense_moe.grouped_moe = old, grouped


def test_inference_mode_then_no_grad_and_grad():
    """Scratch first made under inference_mode still works under no_grad; inputs needing a gradient stay off it."""
    if not GPU or torch.cuda.get_device_capability()[0] != 9:
        print("skipped: needs sm90")
        return
    from shared_delta_moe.cuda_kernels import dense_moe
    from shared_delta_moe.triton_kernels import fused as F

    m, _, x = make(1, 2, 192, 320, torch.bfloat16, "cuda")
    base, ds = (m.gate, m.up, m.down), (m.d_gate, m.d_up, m.d_down)
    dense_moe._SCRATCH.clear()
    with torch.inference_mode():
        a = dense_moe.grouped_moe(x, F.route(*m.topk(x), m.n_delta), base, ds)
    with torch.no_grad():
        b = dense_moe.grouped_moe(x, F.route(*m.topk(x), m.n_delta), base, ds)
    assert torch.equal(a, b)
    assert not dense_moe.supports(x, base, ds)  # grad enabled, parameters require grad
    with torch.no_grad():
        assert dense_moe.supports(x, base, ds)
        assert not dense_moe.supports(x.half(), base, ds)  # bf16 only
        assert not dense_moe.supports(x, (m.gate.mT.contiguous().mT, m.up, m.down), ds)  # non-contiguous base


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
