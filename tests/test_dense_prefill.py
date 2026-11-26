"""The dense prefill path (cuda_kernels/dense_moe.py: materialised W_b + Delta and grouped GEMMs, and the CUDA
extension when built) == the materialised parameter aggregation ``_forward_param``: ragged and empty groups, every
pair on the zero expert, top-2, sizes that are not multiples of the tiles. Needs CUDA and Triton (skipped otherwise).
python tests/test_dense_prefill.py also runs it."""

import torch
from test_delta_moe import make
from test_fused import GPU, rel


def _impls():
    from shared_delta_moe.cuda_kernels import dense_moe

    yield "grouped", dense_moe.grouped_moe
    from shared_delta_moe.cuda_kernels import dense_ext

    if dense_ext.available():
        yield "cuda", dense_ext.cuda_moe
    else:
        print("cuda extension not built: grouped path only")


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
    calls, prefill = [], dense_moe.prefill
    old = constants.DENSE_PREFILL_PAIRS_PER_GROUP
    constants.DENSE_PREFILL_PAIRS_PER_GROUP, dense_moe.prefill = 8, lambda *a: calls.append(1) or prefill(*a)
    try:
        with torch.no_grad():
            assert rel(m.forward_param(x), m._forward_param(x)) < 2e-2
            assert m.forward_param(x[:0]).shape == (0, 192)
    finally:
        constants.DENSE_PREFILL_PAIRS_PER_GROUP, dense_moe.prefill = old, prefill
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


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
