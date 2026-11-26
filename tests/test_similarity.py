"""similarity.py on tiny safetensors checkpoints: Gram statistics == direct computation, sharding, fits, specs."""

import torch
from safetensors.torch import save_file

from shared_delta_moe import similarity as S

P = "model.language_model.layers.0."
SHAPES = {P + "mlp.gate_proj.weight": (12, 8), P + "mlp.up_proj.weight": (12, 8), P + "mlp.down_proj.weight": (8, 12),
          P + "self_attn.q_proj.weight": (8, 8), P + "input_layernorm.weight": (8,),
          "model.language_model.embed_tokens.weight": (20, 8)}  # fmt: skip


def checkpoints(tmp_path):
    g = torch.Generator().manual_seed(0)
    base = {k: torch.randn(s, generator=g).to(torch.bfloat16) for k, s in SHAPES.items()}
    d1 = {k: 0.05 * torch.randn(s, generator=g) for k, s in SHAPES.items()}
    d2 = {k: 0.05 * torch.randn(s, generator=g) for k, s in SHAPES.items()}
    fts = {"a": d1, "b": d2, "c": {k: 2 * d1[k] - d2[k] for k in SHAPES}}
    paths = {}
    for name, sd in [("base", {})] + list(fts.items()):
        p = tmp_path / name
        p.mkdir()
        t = {k: (base[k].float() + sd[k]).to(torch.bfloat16) if sd else base[k] for k in SHAPES}
        t[P + "self_attn.k_proj.weight"] = base[P + "self_attn.q_proj.weight"].clone()  # unchanged in every fine-tune
        if name == "base":
            t["mtp.fc.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
        save_file(t, p / "model.safetensors")
        paths[name] = p
    return paths


def test_gram_matches_direct(tmp_path):
    p = checkpoints(tmp_path)
    fts = {n: p[n] for n in "abc"}
    g = S.delta_gram(p["base"], fts, device="cpu")
    assert "mtp.fc.weight" not in g["tensors"] and len(g["tensors"]) == len(SHAPES) + 1
    load = lambda n, k: S.Checkpoint(p[n]).get("model.language_model." + k).float()
    for name, r in g["tensors"].items():
        w = load("base", name)
        m = torch.stack([w.flatten()] + [(load(n, name) - w).flatten() for n in "abc"]).double()
        torch.testing.assert_close(r["gram"], m @ m.T)
        d = m[1:].abs()
        assert r["zero"].tolist() == (d == 0).sum(1).double().tolist()
        u = S.bf16_ulp(w.flatten()).double() * 1.001
        assert r["ulp1"].tolist() == ((d > 0) & (d <= u)).sum(1).double().tolist()
        assert r["numel"] == w.numel()
        if name.endswith("gate_proj.weight"):
            torch.testing.assert_close(r["neuron"], m[1:].reshape(3, 12, 8).square().sum(2).float())
        if name.endswith("down_proj.weight"):
            torch.testing.assert_close(r["neuron"], m[1:].reshape(3, 8, 12).square().sum(1).float())
    assert g["tensors"]["layers.0.self_attn.k_proj.weight"]["zero"].tolist() == [64.0] * 3


def test_shards_partition(tmp_path):
    p = checkpoints(tmp_path)
    fts = {n: p[n] for n in "ab"}
    whole = S.delta_gram(p["base"], fts, device="cpu")["tensors"]
    for i in range(3):
        torch.save(S.delta_gram(p["base"], fts, i, 3, device="cpu"), tmp_path / f"g{i}.pt")
    merged = S.load_grams([tmp_path / f"g{i}.pt" for i in range(3)])["tensors"]
    assert merged.keys() == whole.keys()
    for k in whole:
        torch.testing.assert_close(merged[k]["gram"], whole[k]["gram"])


def test_fit_and_report(tmp_path):
    p = checkpoints(tmp_path)
    g = S.delta_gram(p["base"], {n: p[n] for n in "abc"}, device="cpu")
    mlp = S.aggregate(g["tensors"], lambda n: S.group(n) == "mlp")["gram"]
    beta, resid = S.fit(mlp, 2, [0, 1])  # c = 2a - b up to bf16 rounding
    assert abs(beta[0] - 2) < 0.05 and abs(beta[1] + 1) < 0.05 and resid < 1e-3
    top, frac = S.shared_fraction(mlp, [0, 0])
    assert abs(top - 1) < 1e-9 and all(abs(f - 1) < 1e-9 for f in frac)
    r = S.report(g, fits=["c=a,b"], sets=["ab=a,b"], pairs=["a,c"])
    assert set(r["groups"]) >= {"all", "mlp", "attn", "embed", "norm"}
    assert r["groups"]["mlp"]["cos"][0][0] > 0.999
    assert 0 <= r["neurons"]["a,c"]["top_overlap"] <= 1
    S.print_report(r)


def test_bf16_ulp():
    w = torch.tensor([1.0, 0.75, -3.0, 1e-3])
    for x, u in zip(w, S.bf16_ulp(w)):
        b = x.to(torch.bfloat16)
        assert (b.float() + u * b.float().sign()).to(torch.bfloat16) != b
        assert u <= abs(x) * 2**-7


def test_parse_spec():
    assert S.parse_spec("A") == ("A", {"mlp": {"A": 1.0}, "rest": {"A": 1.0}})
    assert S.parse_spec("base") == ("base", {"mlp": {}, "rest": {}})
    assert S.parse_spec("avg=A*0.5+B*0.5")[1]["rest"] == {"A": 0.5, "B": 0.5}
    lab, g = S.parse_spec("fr=mlp:A;rest:A*0.5+B*0.5")
    assert lab == "fr" and g == {"mlp": {"A": 1.0}, "rest": {"A": 0.5, "B": 0.5}}
    assert S.kind("layers.3.mlp.up_proj.weight") == "mlp.up_proj" and S.layer("layers.3.mlp.up_proj.weight") == 3
    assert S.group("layers.3.linear_attn.norm.weight") == "norm" and S.group("lm_head.weight") == "embed"


def test_one_ulp_counted(tmp_path):
    g = torch.Generator().manual_seed(1)
    mag = (0.6 + 0.3 * torch.rand(16, 8, generator=g)) * 2.0 ** torch.randint(-6, 3, (16, 8), generator=g)
    w = (mag * torch.randn(16, 8, generator=g).sign()).to(torch.bfloat16)  # mid-binade: 2 ulp stays in the binade
    u = S.bf16_ulp(w.float())
    moved = w.float() + torch.where(torch.arange(16)[:, None] < 8, u, 2 * u) * w.float().sign()
    for n, t in (("base", w), ("ft", moved.to(torch.bfloat16))):
        (tmp_path / n).mkdir()
        save_file({"model.language_model.layers.0.mlp.up_proj.weight": t}, tmp_path / n / "model.safetensors")
    r = S.delta_gram(tmp_path / "base", {"ft": tmp_path / "ft"}, device="cpu")["tensors"]["layers.0.mlp.up_proj.weight"]
    assert r["ulp1"].tolist() == [64.0] and r["zero"].tolist() == [0.0]


class _Tok:
    """Chat template: every message is [role id, content tokens..., 0]; the generation prompt appends [9]."""

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        ids = [t for m in msgs for t in [1 if m["role"] == "user" else 9] + [int(c) for c in m["content"]] + [0]]
        return {"input_ids": ids + ([9] if add_generation_prompt else [])}

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [int(c) for c in text]})()


def test_read_docs(tmp_path):
    import json

    f = tmp_path / "d.jsonl"
    rows = [{"messages": [{"role": "user", "content": "34"}, {"role": "assistant", "content": "56"}]}, {"text": "12"},
            {"text": "1234"}]  # fmt: skip
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = S.read_docs(_Tok(), f, docs=5, max_length=10, min_tokens=3)
    assert out == [([1, 3, 4, 0, 9, 5, 6, 0], 5, 8), ([1, 2, 3, 4], 1, 4)]  # reply from its first token; short text dropped
    assert S.read_docs(_Tok(), f, 1, 10, score="content") == [([1, 3, 4, 0, 9, 5, 6, 0], 5, 7)]  # without the closing 0
    assert S.read_docs(_Tok(), f, 1, 10, score="last") == [([1, 3, 4, 0, 9, 5, 6, 0], 6, 7)]
    assert S.read_docs(_Tok(), f, 1, 5, score="content") == [([0, 9, 5, 6, 0], 2, 4)]  # long chat: keep the end
    assert S.read_docs(_Tok(), f, 1, 3, score="content") == []  # the reply itself does not fit
