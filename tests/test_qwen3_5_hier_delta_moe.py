"""Dense Qwen3.5 -> HierDeltaMoE upscaling in transformers: a tiny base and three perturbed "fine-tunes" in two domain
groups are converted, loaded with the patched decoder layer, given random routers, and compared (logits, fp32, CPU)
with the stock dense model whose MLPs are replaced by the two-level ensemble of the fine-tunes' full MLPs and whose
other weights are the merged ones. Also: the merged tensors and the merge methods."""

import sys
import tempfile
from pathlib import Path

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from test_hier import reference  # noqa: E402

from shared_delta_moe.merging import task_arithmetic, ties_merging  # noqa: E402
from shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe import Qwen3_5HierDeltaMoeBlock  # noqa: E402
from shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert import main as convert_main  # noqa: E402

GROUPS = {"a": [0, 1], "b": [2]}  # domain -> fine-tune indices


class Ensemble(torch.nn.Module):
    def __init__(self, src, k1, k0, zeros, shared):
        super().__init__()
        self.src, self.args = src, (k1, k0)
        self.kw = {"zeros": zeros, "shared": shared}

    def forward(self, h):
        return reference(self.src, *self.args, h.reshape(-1, h.shape[-1]), **self.kw).view(h.shape)


def tiny():
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=["linear_attention", "full_attention"], tie_word_embeddings=False,
    )  # fmt: skip
    torch.manual_seed(0)
    base = Qwen3_5ForCausalLM(cfg).eval()
    tuned = []
    for i in range(3):
        t = Qwen3_5ForCausalLM(cfg).eval()
        t.load_state_dict(base.state_dict())
        with torch.no_grad():
            for p in t.parameters():
                p.add_(torch.randn_like(p) * 0.05 * (1 + p.std() if p.numel() > 1 else 1))
        tuned.append(t)
    return base, tuned


def test_converted_matches_ensemble():
    base, tuned = tiny()
    x = torch.randint(0, 128, (2, 16))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base.save_pretrained(d / "base")
        for i, t in enumerate(tuned):
            t.save_pretrained(d / f"ft{i}")
        groups = [f"{n}=" + ",".join(str(d / f"ft{i}") for i in ix) for n, ix in GROUPS.items()]
        # (zeros, k1, k0, shared_base)
        for j, (zeros, k1, k0, shared) in enumerate(((0, 1, None, False), (1, 2, None, False), (1, 1, 1, True))):
            dst = d / f"moe{j}"
            args = [str(d / "base"), str(dst), *sum((["--group", g] for g in groups), []), "--zeros", str(zeros),
                    "--top-k", str(k1), "--shard-gb", "0.0001"] + (["--group-top-k", str(k0)] if k0 else []) + (
                    ["--shared-base"] if shared else [])  # fmt: skip
            convert_main(args)
            assert len(list(dst.glob("model-*.safetensors"))) > 1, "sharded"
            moe = Qwen3_5ForCausalLM.from_pretrained(dst).eval()
            blocks = [layer.mlp for layer in moe.model.layers]
            assert all(isinstance(b, Qwen3_5HierDeltaMoeBlock) for b in blocks)
            assert blocks[0].group_sizes == [2, 1] and blocks[0].shared_base == shared
            assert all(not b.router.weight.any() for b in blocks), "routers start at zero"
            sd = moe.state_dict()
            want = torch.stack([t.state_dict()["model.embed_tokens.weight"] for t in tuned]).mean(0)
            torch.testing.assert_close(sd["model.embed_tokens.weight"], want)  # simple_average
            torch.manual_seed(j)
            ref = Qwen3_5ForCausalLM(base.config).eval()
            ref.load_state_dict({k: v for k, v in sd.items() if ".mlp." not in k}, strict=False)
            for li, (b, layer) in enumerate(zip(blocks, ref.model.layers)):
                with torch.no_grad():
                    b.router.weight.normal_(std=1.0)
                mlp = lambda m: (m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)  # noqa: E731
                full = [[mlp(tuned[i].model.layers[li].mlp) for i in ix] for ix in GROUPS.values()]
                src = (mlp(base.model.layers[li].mlp), full, b.router.weight)
                layer.mlp = Ensemble(src, k1, k0, [zeros] * 2, shared)
            with torch.no_grad():
                got, want = moe(x).logits, ref(x).logits
            err = ((got - want).abs().max() / want.abs().max()).item()
            assert err < 1e-4, (j, err)


def test_merge_methods():
    torch.manual_seed(0)
    b = torch.randn(64, 32)
    ts = [b + torch.randn_like(b) for _ in range(3)]
    tv = torch.stack([t - b for t in ts])
    torch.testing.assert_close(task_arithmetic(b, ts, 0.5), b + 0.5 * tv.sum(0))
    # threshold 100: no trimming; the elected sign is that of the sum; mean over the agreeing entries
    agree = (tv.sign() == tv.sum(0).sign()).float()
    want = b + 0.3 * (tv * agree).sum(0) / agree.sum(0).clamp_min(1)
    torch.testing.assert_close(ties_merging(b, ts, 0.3, 100, "mean"), want)
    # threshold 10: at most ~10% of each task vector survives
    for th in (10, 0.1):  # a percentage, or a fraction (<= 1) as in fusion_bench
        one = ties_merging(b, ts[:1], 1.0, th)
        assert 0.08 < ((one - b) != 0).float().mean() < 0.12, th


class Dict:
    def __init__(self, d):
        self.d = d

    def keys(self):
        return list(self.d)

    def get(self, k):
        return self.d[k]


def test_vl_layout_and_missing_keys():
    """Qwen3.8's layout: decoder MLPs under model.language_model (converted), vision MLPs and mtp.* (merged), an
    integer tensor (kept); fine-tunes saved by transformers lack mtp.* (kept from the base)."""
    from shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert import Checkpoint, convert

    torch.manual_seed(0)
    p = "model.language_model.layers.0.mlp"
    base = {f"{p}.{q}_proj.weight": torch.randn(6, 4) if q != "down" else torch.randn(4, 6) for q in ("gate", "up", "down")}
    base |= {"model.visual.blocks.0.mlp.linear_fc1.weight": torch.randn(3, 3), "mtp.layers.0.mlp.gate_proj.weight":
             torch.randn(6, 4), "model.ids": torch.arange(5)}  # fmt: skip
    fts = [{k: v + 1.0 if v.is_floating_point() else v for k, v in base.items() if not k.startswith("mtp")} for _ in range(2)]
    with tempfile.TemporaryDirectory() as d:
        convert(Dict(base), {"a": [Dict(fts[0])], "b": [Dict(fts[1])]}, Path(d) / "out", [0, 0], log=lambda s: None)
        out = Checkpoint(Path(d) / "out")
        assert set(out.keys()) == {f"{p}.{q}" for q in ("gate", "up", "down", "d_gate", "d_up", "d_down", "router.weight")} | {
            "model.visual.blocks.0.mlp.linear_fc1.weight", "mtp.layers.0.mlp.gate_proj.weight", "model.ids"}  # fmt: skip
        torch.testing.assert_close(out.get(f"{p}.d_up"), torch.ones(2, 6, 4))
        torch.testing.assert_close(out.get("model.visual.blocks.0.mlp.linear_fc1.weight"), base["model.visual.blocks.0.mlp.linear_fc1.weight"] + 1)
        torch.testing.assert_close(out.get("mtp.layers.0.mlp.gate_proj.weight"), base["mtp.layers.0.mlp.gate_proj.weight"])
        assert out.get(f"{p}.router.weight").shape == (4, 4) and torch.equal(out.get("model.ids"), torch.arange(5))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
