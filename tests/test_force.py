"""Forced routing (DeltaMoE.force, force_route, hier_delta_moe.force in the config): every token on one expert gives
exactly that expert's MLP, W_b + Delta_e (dense, and 2-bit grids with their deltas rounded to the grid), for flat and
Hier modules, zero experts included; force_route restores the routing; the Qwen3.5 config field sets it. CPU, fp32.
The whole converted model (MLP, attention and lm_head deltas forced together): test_routed_linear.py."""

import torch
import test_hier as hier
from test_delta_moe import make

from shared_delta_moe.models.delta_moe import force_route
from shared_delta_moe.models.grid import GridDeltaMoE
from shared_delta_moe.ops import mlp


def _check(m, force, e, x):
    """m forced to ``force`` equals mlp(x; weights(e)) (e < 0: the base): exactly through the materialised forward,
    to rounding through the module's own (small batches: W_b x + Delta_e x, ``mlp_split``)."""
    with torch.no_grad():
        want = mlp(x, *m.weights(e))
        with force_route(m, force):
            exact, got = m._forward_param(x), m(x)
    assert torch.equal(exact, want), (force, e, (exact - want).abs().max().item())
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_flat():
    for z, k in ((1, 1), (2, 2)):
        m, _, x = make(z, k, 16, 32)
        for g in (m, GridDeltaMoE.from_delta_moe(m, 4, 8)):
            for e in range(g.n_delta + z):
                _check(g, e, e if e < g.n_delta else -1, x)


def test_hier():
    for k1, k0 in ((2, None), (1, 2)):
        m, _, x = hier.make(k1, k0, hidden=16, inter=32)
        for g in (m, GridDeltaMoE.from_delta_moe(m, 4, 8)):
            d = 0
            for gi, (n, zr) in enumerate(zip(g.group_sizes, g.group_zeros)):
                for e in range(n + zr):
                    _check(g, (gi, e), d + e if e < n else -1, x)
                d += n


def test_restores_and_rejects():
    m, _, x = hier.make(2, None)
    with torch.no_grad():
        y = m(x)
        with force_route(m, [0, 1]):  # a list, as the config gives it
            assert not torch.equal(m(x), y)
        assert m.force is None and torch.equal(m(x), y)
    m.force = (0, m.group_sizes[0] + m.group_zeros[0])  # past the group
    try:
        m(x)
    except AssertionError:
        pass
    else:
        raise AssertionError("an out-of-range force must fail")


def test_qwen_config_field():
    import sys

    sys.modules["fla"] = None  # CPU (see test_qwen3_5_hier_delta_moe.py)
    from types import SimpleNamespace

    from shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.modeling import Qwen3_5HierDeltaMoeBlock

    c = SimpleNamespace(hidden_size=16, intermediate_size=24, hier_delta_moe={"group_sizes": [2, 1], "group_zeros": [1, 0],
                                                                             "top_k": 1, "force": [1, 0]})  # fmt: skip
    assert Qwen3_5HierDeltaMoeBlock(c).force == [1, 0]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
