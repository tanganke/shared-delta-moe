"""Looped Qwen3.5 (models/qwen3_5/looped.py, models/looped.py) and per-expert LoRA (models/lora.py), on a tiny model,
fp32, CPU: at init the loop is the plain model; fixed routing reuses round 0's routing (== a hand-rolled loop);
generation with the per-round cache slots == the full forward; checkpointed gradients == plain ones; LoRA == a
reference MoE with LoRA experts."""

import copy
import sys

import pytest
import torch

sys.modules["fla"] = None  # CPU test: no flash-linear-attention Triton kernels
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

from shared_delta_moe import DeltaMoE, GridDeltaMoE, mlp, random_experts  # noqa: E402
from shared_delta_moe.models.lora import add_expert_lora  # noqa: E402
from shared_delta_moe.models.qwen3_5.looped import loop_layers  # noqa: E402

H, I, N = 32, 48, 3
START, SIZE = 1, 2  # layers 1 (full attention) and 2 (linear attention)


def moe(seed: int) -> DeltaMoE:
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    base, *experts = random_experts(H, I, N + 1, torch.float32, "cpu")
    return DeltaMoE.from_experts(base, experts, torch.randn(N + 1, H, generator=g), top_k=2)


def tiny(lora: bool = False) -> Qwen3_5ForCausalLM:
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=H, intermediate_size=I, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, tie_word_embeddings=False,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    )  # fmt: skip
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(cfg).eval()
    for i in range(START, START + SIZE):
        m.model.layers[i].mlp = moe(i)
        if lora:
            add_expert_lora(m.model.layers[i].mlp, 4, 8)
    return m


def perturb(m: Qwen3_5ForCausalLM) -> None:
    torch.manual_seed(1)
    with torch.no_grad():
        m.model.loop_update.up.weight.normal_(0, 0.3)
        for n, p in m.named_parameters():
            if "lora_b" in n:
                p.normal_(0, 0.3)


@pytest.mark.parametrize("routing", ["fixed", "dynamic"])
def test_init_is_plain(routing):
    plain = tiny()
    looped = loop_layers(copy.deepcopy(plain), START, SIZE, 3, routing)
    x = torch.randint(0, 128, (2, 12))
    torch.testing.assert_close(looped(x).logits, plain(x).logits)  # feedback 0: every round repeats round 0
    perturb(looped)
    assert (looped(x).logits - plain(x).logits).abs().max() > 1e-3


def test_fixed_routing():
    m = loop_layers(tiny(), START, SIZE, 3, "fixed")
    perturb(m)
    calls = []
    for i in range(START, START + SIZE):
        m.model.layers[i].mlp.router.register_forward_hook(lambda mod, a, o: calls.append(mod))
    x = torch.randint(0, 128, (2, 12))
    got = m(x).logits
    assert len(calls) == SIZE  # the routers run in round 0 only

    # reference: the same model routing every round (dynamic), each block's topk replaced on the instance by one
    # that computes once and then returns that
    ref = copy.deepcopy(m)
    ref.model.config.loop = {**ref.model.config.loop, "routing": "dynamic"}
    for i in range(START, START + SIZE):
        b = ref.model.layers[i].mlp
        saved = []

        def topk(xx, router=None, b=b, saved=saved):
            if not saved:
                saved.append(DeltaMoE.topk(b, xx, router))
            return saved[0]

        b.topk = topk
    torch.testing.assert_close(got, ref(x).logits)
    torch.testing.assert_close(got, m(x).logits)  # and nothing left over between calls


def test_dynamic_routes_every_round():
    m = loop_layers(tiny(), START, SIZE, 3, "dynamic")
    calls = []
    for i in range(START, START + SIZE):
        m.model.layers[i].mlp.router.register_forward_hook(lambda mod, a, o: calls.append(mod))
    m(torch.randint(0, 128, (2, 12)))
    assert len(calls) == 3 * SIZE


@pytest.mark.parametrize("routing", ["fixed", "dynamic"])
def test_cache_matches_full_forward(routing):
    m = loop_layers(tiny(lora=True), START, SIZE, 3, routing)
    perturb(m)
    x = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        out = m.generate(x, max_new_tokens=6, do_sample=False, min_new_tokens=6, return_dict_in_generate=True,
                         output_logits=True, pad_token_id=0)  # fmt: skip
        assert len(out.past_key_values.layers) == 4 + 2 * SIZE
        full = m(out.sequences).logits[0, 9:-1]
    torch.testing.assert_close(torch.cat(out.logits), full, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("routing", ["fixed", "dynamic"])
def test_checkpoint_gradients(routing):
    m = loop_layers(tiny(lora=True), START, SIZE, 3, routing)
    perturb(m)
    m.train()
    x = torch.randint(0, 128, (2, 12))
    res = []
    for ckpt in (False, True):
        if ckpt:
            m.gradient_checkpointing_enable()
        m.zero_grad()
        loss = m(x, labels=x).loss + 0.01 * torch.stack(m.model.loop_aux).mean()
        assert len(m.model.loop_aux) == SIZE * (1 if routing == "fixed" else 3)
        loss.backward()
        res.append((loss.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    torch.testing.assert_close(res[0][0], res[1][0])
    assert res[0][1].keys() == res[1][1].keys()
    for k in ("model.loop_update.up.weight", "model.loop_update.update_logit", "model.layers.1.mlp.router.weight",
              "model.layers.2.mlp.lora_b_down"):  # fmt: skip
        assert k in res[0][1] and res[0][1][k].abs().sum() > 0, k
    for k in res[0][1]:
        torch.testing.assert_close(res[0][1][k], res[1][1][k], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("grid", [False, True])
def test_expert_lora(grid):
    m = moe(0)
    if grid:
        m = GridDeltaMoE.from_delta_moe(m, levels=4, group=16)
    x = torch.randn(20, H)
    with torch.no_grad():
        want0 = m(x)
    add_expert_lora(m, 4, 8)
    torch.testing.assert_close(m(x), want0)  # B = 0
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_b" in n:
                p.normal_(0, 0.3)
    got = m(x)
    idx, r = m.route(x)
    want = torch.zeros_like(x)
    for t in range(len(x)):
        for s in range(2):
            e = int(idx[t, s])
            g, u, d = m.weights(e)
            xt = x[t:t + 1]
            if e < 0:
                y = mlp(xt, g, u, d)
            else:
                lo = lambda p, v: v @ getattr(m, f"lora_a_{p}")[e].T @ getattr(m, f"lora_b_{p}")[e].T * 2  # noqa: E731
                hh = torch.nn.functional.silu(xt @ g.T + lo("gate", xt)) * (xt @ u.T + lo("up", xt))
                y = hh @ d.T + lo("down", hh)
            want[t] += r[t, s] * y[0]
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
    got.sum().backward()
    assert m.lora_b_up.grad.abs().sum() > 0 and m.router.weight.grad.abs().sum() > 0
