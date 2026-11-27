"""LoRA training on frozen 2-bit grids (models/lora.py ``grouped_lora_moe``): the grouped path == the
materialised per-expert loop, forward and the gradients to x, the router and the LoRA."""

import copy

import pytest
import torch

from shared_delta_moe import DeltaMoE, GridDeltaMoE, random_experts
from shared_delta_moe.constants import HAS_TRITON
from shared_delta_moe.models.lora import _ExpertLoRA, add_expert_lora, grouped_lora_moe

H, I, N = 64, 128, 3


def lora_moe(dtype, dev, grid=False, z=1):
    torch.manual_seed(0)
    base, *experts = random_experts(H, I, N + 1, dtype, dev)
    m = DeltaMoE.from_experts(base, experts, torch.randn(N + z, H, device=dev) * 0.3, top_k=2, num_zero_experts=z)
    if grid:
        m = GridDeltaMoE.from_delta_moe(m, levels=4, group=32)
    add_expert_lora(m, 16, 32)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_b" in n:
                p.normal_(0, 0.02)
    m.router.float()
    for n, p in m.named_parameters():
        p.requires_grad_(n == "router.weight" or n.startswith("lora_"))
    return m


def grads(m, x, f):
    x = x.detach().clone().requires_grad_()
    for p in m.parameters():
        p.grad = None
    y = f(x)
    y.float().square().sum().backward()
    return y, x.grad, {n: p.grad for n, p in m.named_parameters() if p.requires_grad}



def loop(m, x):  # the per-expert loop of _ExpertLoRA._forward_auto
    idx, r = m.route(x)
    out = torch.zeros_like(x)
    for e in idx.unique().tolist():
        rows, slots = (idx == e).nonzero(as_tuple=True)
        out = out.index_add(0, rows, (r[rows, slots, None] * m.expert_mlp(x[rows], e)).to(out.dtype))
    return out


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_grouped_matches_loop_cpu():
    m = lora_moe(torch.float32, "cpu")
    x = torch.randn(40, H)

    def grouped(x):
        idx, r = m.topk(x)
        with torch.no_grad():
            ws = [torch.stack([m.weights(e)[j] for e in range(N)] + [m.weights(-1)[j]]) for j in range(3)]
        loras = [(getattr(m, f"lora_a_{p}"), getattr(m, f"lora_b_{p}")) for p in ("gate", "up", "down")]
        return grouped_lora_moe(x, idx.clamp(max=N), r, ws, loras, m.lora_scale)

    y, gx, gp = grads(m, x, grouped)
    y0, gx0, gp0 = grads(m, x, lambda x: loop(m, x))
    torch.testing.assert_close(y, y0, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(gx, gx0, rtol=1e-4, atol=1e-5)
    for n in gp0:
        torch.testing.assert_close(gp[n], gp0[n], rtol=1e-4, atol=1e-6, msg=n)


@pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON), reason="CUDA + Triton")
@pytest.mark.parametrize("z", [0, 1])
def test_grid_decoded_matches_materialised(z):
    m = lora_moe(torch.bfloat16, "cuda", grid=True, z=z)
    ref = copy.deepcopy(m)
    x = torch.randn(512, H, device="cuda", dtype=torch.bfloat16)
    assert m._use_decoded(x.requires_grad_()), "takes the grouped LoRA path"
    y, gx, gp = grads(m, x, m)
    with torch.no_grad():  # the same path without grad (a reentrant checkpoint's forward)
        assert torch.equal(m(x), y)
    y0, gx0, gp0 = grads(ref, x, lambda x: loop(ref, x))
    assert rel(y, y0) < 1e-2 and rel(gx, gx0) < 2e-2, (rel(y, y0), rel(gx, gx0))
    for n in gp0:
        assert rel(gp[n], gp0[n]) < 5e-2, (n, rel(gp[n], gp0[n]))
