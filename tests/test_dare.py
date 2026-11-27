"""Seeded DARE: the Philox mask is the published Philox4x32-10, identical on CPU, CUDA and in Triton, and every
forward of DareDeltaMoE equals the plain MoE with W_b + mask * Delta / keep. python tests/test_dare.py also runs it.
"""

import itertools

import torch
from test_delta_moe import CASES, make, reference

from shared_delta_moe import DareDeltaMoE, DeltaMoE
from shared_delta_moe.constants import HAS_TRITON
from shared_delta_moe.models.dare import PROJ, keep_mask, mask_bits, philox

GPU = torch.cuda.is_available() and HAS_TRITON
SETTINGS = [
    (0.25, 1),
    (0.25, 4),
    (0.5, 8),
]  # (keep, block); hidden 16 and inter 24 are multiples of 8


def test_philox_known_answer():
    # Random123 kat_vectors, philox4x32 10 rounds, the "pi" vector: word 0 of the output
    c = [torch.tensor([v]) for v in (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)]
    assert philox(*c, 0xA4093822, 0x299F31D0).item() == 0xD16CFE09
    z = torch.zeros(1, dtype=torch.int64)
    assert philox(z, z, z, z, 0, 0).item() == 0x6627E8D5


def test_mask():
    m = keep_mask(7, 5, 64, 256, 0.25, 16)
    blocks = m.view(64, 16, 16)
    assert (blocks.all(-1) | ~blocks.any(-1)).all(), "whole blocks"
    assert abs(m.float().mean().item() - 0.25) < 0.05
    assert not torch.equal(m, keep_mask(7, 6, 64, 256, 0.25, 16)), "streams differ"
    if torch.cuda.is_available():
        assert torch.equal(
            m, keep_mask(7, 5, 64, 256, 0.25, 16, "cuda").cpu()
        ), "same mask on CPU and GPU"


def test_mask_bits_is_keep_mask():
    """The bitmap the fused kernel reads unpacks to keep_mask, ragged word counts included (C / B not a multiple of
    32, and over 32 blocks per row)."""
    for seed, stream, rows, cols, block in ((0, 0, 7, 40, 1), (12345, 7, 5, 64 * 40, 64), (2**40 + 3, 14, 3, 9216, 64),
                                            (9, 2, 4, 96, 32)):  # fmt: skip
        bits = mask_bits(seed, stream, rows, cols, 0.3, block)
        j = torch.arange(cols // block)
        unpacked = ((bits.long()[:, j // 32] >> (j % 32)) & 1).bool().repeat_interleave(block, 1)
        assert torch.equal(unpacked, keep_mask(seed, stream, rows, cols, 0.3, block)), (seed, block)


def test_bitmaps_after_load_state_dict():
    """The mask bitmaps are built at construction and after load_state_dict (also for a model made on meta), so the
    first forward of a loaded model can be captured in a CUDA graph (it builds nothing)."""
    m, _, x = make(1, 2, 64, 96, torch.bfloat16, "cuda") if GPU else make(1, 2)
    d = DareDeltaMoE.from_delta_moe(m, 0.3, 4, seed=5)
    I, H = m.gate.shape
    for device in ("meta", x.device):
        d2 = DareDeltaMoE(H, I, m.n_delta, 0.3, 4, 5, m.top_k, m.num_zero_experts, device, x.dtype)
        d2 = d2.to_empty(device=x.device)
        d2.load_state_dict(d.state_dict())  # sizes the packed buffers
        for name in ("gate", "up", "down"):
            assert torch.equal(getattr(d2, f"m_{name}"), getattr(d, f"m_{name}")), (device, name)
    for other in ((0.5, 4, 5), (0.3, 4, 6), (0.3, 8, 5)):  # keep, block, seed of another model: refused
        d3 = DareDeltaMoE(H, I, m.n_delta, *other, m.top_k, m.num_zero_experts, x.device, x.dtype)
        try:
            d3.load_state_dict(d.state_dict())
            raise AssertionError(other)
        except RuntimeError as err:
            assert "do not match" in str(err), err
    d4 = DareDeltaMoE(H, I, m.n_delta, 0.3, 4, 5, m.top_k, m.num_zero_experts, x.device, x.dtype)
    d4.load_state_dict({"router.weight": d.router.weight}, strict=False)  # packed deltas left for later: no check
    if not GPU:
        return
    with torch.no_grad():
        want = d._forward_param_triton(x[:37])
        xs = x[:37].clone()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = d2._forward_param_triton(xs)
        assert torch.equal(d2.m_gate, d.m_gate)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, want)


def test_no_deltas():
    """n_delta = 0 (only zero experts): packs, saves, loads and runs as the base MLP."""
    m, _, x = make(1, 1)
    base = DeltaMoE(16, 24, 0, 2, 2)
    with torch.no_grad():
        base.router.weight.normal_()
        for n in PROJ:
            getattr(base, n).copy_(getattr(m, n))
    d = DareDeltaMoE.from_delta_moe(base, 0.5, 4)
    d2 = DareDeltaMoE(16, 24, 0, 0.5, 4, 0, 2, 2)
    d2.load_state_dict(d.state_dict())
    want = base._forward_param(x)
    for mod in (d, d2):
        assert torch.allclose(mod._forward_param(x), want) and torch.allclose(mod._forward_matrix(x), want)


def test_dare_forwards_match_plain_moe():
    for (z, k), (keep, block) in itertools.product(CASES, SETTINGS):
        m, _, x = make(z, k)
        d = DareDeltaMoE.from_delta_moe(m, keep, block, seed=3)
        for name in ("gate", "up", "down"):
            dense, sparse = getattr(m, f"d_{name}"), getattr(d, f"d_{name}")
            kept = sparse != 0
            assert torch.allclose(
                sparse[kept], dense[kept] / keep
            ), "kept values rescaled"
            assert abs(kept.float().mean().item() - keep) < 0.15
        ref = reference(d, x)
        for name, f in {
            "param": d._forward_param,
            "matrix grouped": d._forward_matrix_grouped,
            "moe grouped": d.to_moe()._forward_grouped,
        }.items():
            err = (
                (f(x) - ref).abs().max() / ref.abs().max()
            ).item()  # deltas / keep make outputs larger
            assert err < 1e-5, (z, k, keep, block, name, err)


def test_unpack_is_keep_mask():
    """The bitmap unpack behind the materialised path equals Delta / keep under keep_mask: several words per row,
    a ragged last word and bit 31 (negative int32 words) included."""
    m, _, _ = make(1, 1, 80, 96)
    for keep, block in ((0.9, 1), (0.5, 1), (0.3, 2), (0.5, 8)):
        d = DareDeltaMoE.from_delta_moe(m, keep, block, seed=2**40 + 7)
        for j, n in enumerate(PROJ):
            dense = getattr(m, f"d_{n}")
            for e in range(m.n_delta):
                mask = keep_mask(2**40 + 7, 3 * e + j, *dense.shape[1:], keep, block)
                want = torch.where(mask, (dense[e].float() / keep).to(dense.dtype), 0)
                assert torch.equal(d.delta(e)[j], want), (keep, block, n, e)


def test_dare_training_gradients():
    """Training through the materialised path (deltas unpacked per expert from the bitmap): with the kept values made
    trainable, their gradient is the dense DeltaMoE's delta gradient at the kept positions; x's gradient matches."""
    for (z, k), (keep, block) in itertools.product(((1, 1), (2, 2)), SETTINGS):
        m, _, x = make(z, k)
        d = DareDeltaMoE.from_delta_moe(m, keep, block, seed=3)
        for n in PROJ:
            setattr(d, f"v_{n}", torch.nn.Parameter(d._buffers.pop(f"v_{n}")))
        dense = DeltaMoE(16, 24, m.n_delta, k, z)
        with torch.no_grad():
            dense.router.weight.copy_(d.router.weight)
            for n in PROJ:
                getattr(dense, n).copy_(getattr(d, n))
                getattr(dense, f"d_{n}").copy_(getattr(d, f"d_{n}"))
        w = torch.randn(200, 16)
        for name, f in (("param", lambda mod, x: mod._forward_param(x)), ("split", lambda mod, x: mod._forward_param(x, min_rows=10**9))):
            gx = []
            for mod in (d, dense):
                xg = x.clone().requires_grad_(True)
                mod.zero_grad()
                (f(mod, xg) * w).sum().backward()
                gx.append(xg.grad)
            assert torch.allclose(gx[0], gx[1], atol=1e-5), (z, k, keep, block, name)
            for j, n in enumerate(PROJ):
                g, (G, R, C) = getattr(dense, f"d_{n}").grad, d.shapes[n]
                want = torch.cat([g[e][keep_mask(3, 3 * e + j, R, C, keep, block)] for e in range(G)])
                assert torch.allclose(getattr(d, f"v_{n}").grad, want, atol=1e-5), (z, k, keep, block, name, n)


def test_dare_triton_matches_materialised():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    # few combinations: Triton autotunes again for every shape and block size; 192/320 leave ragged tiles
    combos = itertools.product(
        ((0, 1), (2, 2)), ((0.25, 1), (0.5, 64)), ((256, 384), (192, 320))
    )
    for (z, k), (keep, block), (hidden, inter) in combos:
        m, _, x = make(z, k, hidden, inter, torch.bfloat16, "cuda")
        d = DareDeltaMoE.from_delta_moe(m, keep, block, seed=2**33 + 1)
        for t in (1, 37, 200):
            want = d._forward_param(x[:t])
            got = d._forward_param_triton(x[:t])
            rel = ((got - want).abs().max() / want.abs().max()).item()
            assert rel < 2e-2, (z, k, keep, block, hidden, t, rel)


def test_triton_on_another_gpu():
    """Tensors on cuda:1 while cuda:0 is current (a model placed with device_map)."""
    if not GPU or torch.cuda.device_count() < 2:
        print("skipped: needs 2 GPUs")
        return
    m, _, x = make(1, 1, 256, 384, torch.bfloat16, "cuda:1")
    d = DareDeltaMoE.from_delta_moe(m, 0.5, 64, seed=1)
    for mod in (m, d):
        want = mod._forward_param(x[:37])
        rel = (
            (mod._forward_param_triton(x[:37]) - want).abs().max() / want.abs().max()
        ).item()
        assert rel < 2e-2, (type(mod).__name__, rel)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
