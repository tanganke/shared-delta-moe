"""Dense Qwen3.5 -> HierDeltaMoE upscaling in transformers: a tiny base and three perturbed "fine-tunes" in two domain
groups are converted, loaded with the patched decoder layer, given random routers, and compared (logits, fp32, CPU)
with the stock dense model whose MLPs are replaced by the two-level ensemble of the fine-tunes' full MLPs and whose
other weights are the merged ones. Also: the merged tensors and the merge methods."""

import sys
import tempfile
from pathlib import Path

import torch

sys.modules["fla"] = None  # CPU test: with flash-linear-attention installed, transformers runs its Triton kernels even on CPU tensors
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from test_hier import reference  # noqa: E402

from shared_delta_moe.merging import simple_average, task_arithmetic, ties_merging  # noqa: E402
from shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe import Qwen3_5HierDeltaMoeBlock  # noqa: E402
from shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert import main as convert_main  # noqa: E402

GROUPS = {"a": [0, 1], "b": [2]}  # domain -> fine-tune indices


class Ensemble(torch.nn.Module):
    def __init__(self, src, k1, k0, zeros):
        super().__init__()
        self.src, self.args, self.zeros = src, (k1, k0), zeros

    def forward(self, h):
        return reference(self.src, *self.args, h.reshape(-1, h.shape[-1]), self.zeros).view(h.shape)


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
        # (zeros, k1, k0)
        for j, (zeros, k1, k0) in enumerate(((0, 1, None), (1, 2, None), (1, 1, 1))):
            dst = d / f"moe{j}"
            args = [str(d / "base"), str(dst), *sum((["--group", g] for g in groups), []), "--zeros", str(zeros),
                    "--top-k", str(k1), "--shard-gb", "0.0001"] + (["--group-top-k", str(k0)] if k0 else [])  # fmt: skip
            convert_main(args)
            assert len(list(dst.glob("model-*.safetensors"))) > 1, "sharded"
            moe = Qwen3_5ForCausalLM.from_pretrained(dst).eval()
            blocks = [layer.mlp for layer in moe.model.layers]
            assert all(isinstance(b, Qwen3_5HierDeltaMoeBlock) for b in blocks)
            assert blocks[0].group_sizes == [2, 1]
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
                layer.mlp = Ensemble(src, k1, k0, [zeros] * 2)
            with torch.no_grad():
                got, want = moe(x).logits, ref(x).logits
            err = ((got - want).abs().max() / want.abs().max()).item()
            assert err < 1e-4, (j, err)


def test_merge_methods():
    """The fusion_bench calls behind merging.py, on a state dict of a matrix and a 0-dim tensor in bf16 (merged in
    fp32, returned in bf16)."""
    torch.manual_seed(0)
    b = {"w": torch.randn(64, 32, dtype=torch.bfloat16), "s": torch.tensor(0.5, dtype=torch.bfloat16)}
    ts = [{k: v + torch.randn_like(v) for k, v in b.items()} for _ in range(3)]
    tv = {k: torch.stack([t[k].float() - v.float() for t in ts]) for k, v in b.items()}
    for k, v in task_arithmetic(b, ts, 0.5).items():
        assert v.dtype == torch.bfloat16 and v.shape == b[k].shape
        torch.testing.assert_close(v, (b[k].float() + 0.5 * tv[k].sum(0)).bfloat16())
    for k, v in simple_average(b, ts).items():
        torch.testing.assert_close(v, torch.stack([t[k].float() for t in ts]).mean(0).bfloat16())
    # one task vector, scale 1: the kept entries are its largest, unchanged; about threshold of them survive
    for th in (10, 0.1):  # a percentage, or a fraction (<= 1), as in fusion_bench
        one = ties_merging(b, ts[:1], 1.0, th)
        d = one["w"].float() - b["w"].float()
        kept = d != 0
        assert 0.08 < kept.float().mean() < 0.12, th
        torch.testing.assert_close(d[kept], (ts[0]["w"].float() - b["w"].float())[kept], atol=0.02, rtol=0.02)
        assert tv["w"][0].abs()[kept].min() >= tv["w"][0].abs()[~kept].max(), th
    m = ties_merging(b, ts, 0.3, 20, "mean")
    assert all(m[k].dtype == torch.bfloat16 and m[k].shape == b[k].shape for k in b)
    # fp64 is not rounded through fp32: a task vector of 1e-12 survives
    b64 = {"w": torch.ones(4, dtype=torch.float64)}
    t64 = [{"w": b64["w"] + 1e-12}]
    for f in (simple_average, task_arithmetic, ties_merging):
        out = f(b64, t64)["w"]
        assert out.dtype == torch.float64 and (out != 1).all(), f.__name__


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
        # one fine-tune in both groups: an expert in each, counted once in the merge (mean of ft0 + 1 and ft1 + 3)
        fts[1] = {k: v + 2.0 if v.is_floating_point() else v for k, v in fts[1].items()}
        shared = Dict(fts[0])
        convert(Dict(base), {"a": [shared, Dict(fts[1])], "b": [shared]}, Path(d) / "dup", [0, 0], log=lambda s: None)
        out = Checkpoint(Path(d) / "dup")
        torch.testing.assert_close(out.get(f"{p}.d_up"), torch.tensor([1.0, 3.0, 1.0])[:, None, None].expand(3, 6, 4))
        torch.testing.assert_close(out.get("model.visual.blocks.0.mlp.linear_fc1.weight"), base["model.visual.blocks.0.mlp.linear_fc1.weight"] + 2)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
