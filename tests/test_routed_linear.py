"""Token-routed attention / lm_head deltas (models/routed_linear.py, models/qwen3_5/common.py): the layer against
per-token weight mixing; a tiny Qwen3.5 converted with --routed-deltas and routed to one expert reproduces that
fine-tune (fp32, CPU), with dense and grid deltas; the routers get gradients."""

import sys
import tempfile
from pathlib import Path

import torch

sys.modules["fla"] = None  # CPU test (see test_qwen3_5_hier_delta_moe.py)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_qwen3_5_hier_delta_moe import tiny  # noqa: E402

from shared_delta_moe import compress as C  # noqa: E402
from shared_delta_moe.models.grid import pack_codes  # noqa: E402
from shared_delta_moe.models.delta_moe import force_route  # noqa: E402
from shared_delta_moe.models.hier import HierDeltaMoE  # noqa: E402
from shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert import main as convert_main  # noqa: E402
from shared_delta_moe.models.routed_linear import RoutedDeltaLinear  # noqa: E402


def test_layer_mixes_weights():
    torch.manual_seed(0)
    T, Cn, R, N, K = 10, 64, 24, 3, 2
    for levels in (None, 4):
        lin = RoutedDeltaLinear(Cn, R, N, levels, group=32)
        torch.nn.init.normal_(lin.weight)
        d = torch.randn(N, R, Cn) * 0.1
        if levels is None:
            lin.deltas.data.copy_(d)
        else:
            for e in range(N):
                q, lo, st = C.quantize(d[e], None, levels, 32)
                lin.codes[e], lin.lows.data[e], lin.steps.data[e] = pack_codes(q, 2), lo, st
                d[e] = C.dequantize(q, lo, st)
        idx = torch.randint(0, N + 1, (T, K))  # N: no delta
        r = torch.rand(T, K).softmax(-1)
        lin.route = (idx, r)
        x = torch.randn(2, T // 2, Cn)
        dz = torch.cat([d, torch.zeros(1, R, Cn)])
        w = lin.weight + (r[..., None, None] * dz[idx]).sum(1)  # T R C, each token's mixed weight
        torch.testing.assert_close(lin(x), torch.einsum("trc,tc->tr", w, x.reshape(T, Cn)).view(2, T // 2, R))


def test_converted_routes_every_weight():
    base, tuned = tiny()
    # fine-tunes that differ from the base only in routed weights (MLP, attention, lm_head), so one expert is exact
    for t in tuned:
        t.load_state_dict({k: v if v.dim() == 2 and "embed" not in k else base.state_dict()[k]
                           for k, v in t.state_dict().items()})  # fmt: skip
    x = torch.randint(0, 128, (2, 16))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base.save_pretrained(d / "base")
        for i, t in enumerate(tuned):
            t.save_pretrained(d / f"ft{i}")
        for levels in (None, 4):
            dst = d / f"moe{levels}"
            extra = [] if levels is None else ["--routed-levels", str(levels), "--routed-group", "8"]
            convert_main([str(d / "base"), str(dst), "--group", "a=" + ",".join(str(d / f"ft{i}") for i in range(3)),
                          "--routed-deltas", *extra])  # fmt: skip
            moe = Qwen3_5ForCausalLM.from_pretrained(dst).eval()
            routed = [m for m in moe.modules() if isinstance(m, RoutedDeltaLinear)]
            assert len(routed) == 5 + 4 + 1, "linear attention, full attention, lm_head"
            for e in range(3):
                want = tuned[e]
                if levels is not None:  # the fine-tune with its routed (non-MLP) deltas rounded to the grid
                    want = Qwen3_5ForCausalLM(base.config).eval()
                    sd = dict(tuned[e].state_dict())
                    for k, b in base.state_dict().items():
                        if b.dim() == 2 and "embed" not in k and ".mlp." not in k:
                            sd[k] = b + C.dequantize(*C.quantize(sd[k] - b, None, levels, 8))
                    want.load_state_dict(sd)
                with torch.no_grad(), force_route(moe, (0, e)):  # the MLP, attention and lm_head deltas of expert e
                    got, ref = moe(x).logits, want(x).logits
                err = ((got - ref).abs().max() / ref.abs().max()).item()
                assert err < 1e-4, (levels, e, err)
            # the routers get gradients through the routing weights
            moe = Qwen3_5ForCausalLM.from_pretrained(dst)
            for n, p in moe.named_parameters():
                if "router" in n:
                    torch.nn.init.normal_(p, std=0.5)
            moe(x).logits.square().mean().backward()
            grads = {n: p.grad for n, p in moe.named_parameters() if "delta_router" in n}
            assert len(grads) == 3 and all(g is not None and g.abs().sum() > 0 for g in grads.values())


def test_grid_mlp_and_routed_sets():
    """--levels (grid MLP deltas, Qwen3_5GridHierDeltaMoeBlock) with --routed-sets 2,1: expert e is its own MLP
    rounded to the grid plus its set's mean non-MLP delta rounded to the grid."""
    from shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe import Qwen3_5GridHierDeltaMoeBlock

    base, tuned = tiny()
    for t in tuned:
        t.load_state_dict({k: v if v.dim() == 2 and "embed" not in k else base.state_dict()[k]
                           for k, v in t.state_dict().items()})  # fmt: skip
    x = torch.randint(0, 128, (2, 16))
    q = lambda d: C.dequantize(*C.quantize(d, None, 4, 8))  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base.save_pretrained(d / "base")
        for i, t in enumerate(tuned):
            t.save_pretrained(d / f"ft{i}")
        dst = d / "moe"
        convert_main([str(d / "base"), str(dst), "--group", f"a={d / 'ft0'},{d / 'ft1'}", "--group", f"b={d / 'ft2'}",
                      "--zeros", "1", "--top-k", "2", "--levels", "4", "--grid-group", "8", "--routed-deltas",
                      "--routed-levels", "4", "--routed-group", "8", "--routed-sets", "2,1"])  # fmt: skip
        moe = Qwen3_5ForCausalLM.from_pretrained(dst).eval()
        assert all(isinstance(layer.mlp, Qwen3_5GridHierDeltaMoeBlock) for layer in moe.model.layers)
        assert moe.lm_head.n_delta == 2 and moe.lm_head.sets == [2, 1]
        sd0 = base.state_dict()
        for e, s_, ge in ((0, (0, 1), (0, 0)), (1, (0, 1), (0, 1)), (2, (2,), (1, 0))):  # ge: (group, expert in it)
            want = Qwen3_5ForCausalLM(base.config).eval()
            sd = {}
            for k, b in sd0.items():
                if b.dim() != 2 or "embed" in k:
                    sd[k] = b
                elif ".mlp." in k:
                    sd[k] = b + q(tuned[e].state_dict()[k] - b)
                else:
                    sd[k] = b + q(torch.stack([tuned[i].state_dict()[k] - b for i in s_]).mean(0))
            want.load_state_dict(sd)
            with torch.no_grad(), force_route(moe, ge):
                got, ref = moe(x).logits, want(x).logits
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-4, (e, err)


def test_top_k_wider_than_a_group():
    """k1 = 2 over groups of 2 and 1 experts (no zeros): dense within each group; the narrow group's extra slot is
    the base with weight 0."""
    m = HierDeltaMoE(16, 8, [2, 1], [0, 0], top_k=2)
    torch.nn.init.normal_(m.router.weight)
    idx, r = m.topk(torch.randn(5, 16))
    assert idx.shape == (5, 4)
    assert (idx[:, 3] == m.n_delta).all() and (r[:, 3] == 0).all() and (idx[:, 2] == 2).all()
    torch.testing.assert_close(r.sum(-1), torch.ones(5))


if __name__ == "__main__":
    test_layer_mixes_weights()
    test_converted_routes_every_weight()
    test_grid_mlp_and_routed_sets()
    test_top_k_wider_than_a_group()
    print("ok")
