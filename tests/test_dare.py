"""Seeded DARE: the Philox mask is the published Philox4x32-10, identical on CPU, CUDA and in Triton, and every
forward of DareDeltaMoE equals the plain MoE with W_b + mask * Delta / keep. python tests/test_dare.py also runs it.
"""

import itertools

import torch
from test_delta_moe import CASES, make, reference

from shared_delta_moe import DareDeltaMoE
from shared_delta_moe.constants import HAS_TRITON
from shared_delta_moe.models.dare import keep_mask, mask_bits, philox

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
        for name in ("gate", "up", "down"):  # packed buffers start empty (see _init_deltas)
            setattr(d2, f"v_{name}", torch.empty_like(getattr(d, f"v_{name}")))
            setattr(d2, f"p_{name}", torch.empty_like(getattr(d, f"p_{name}")))
        d2.load_state_dict(d.state_dict())
        for name in ("gate", "up", "down"):
            assert torch.equal(getattr(d2, f"m_{name}"), getattr(d, f"m_{name}")), (device, name)
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
        experts = d.dense_experts()
        for name, f in {
            "param": d._forward_param,
            "matrix grouped": d._forward_matrix_grouped,
            "dense grouped": lambda x: d._forward_dense_grouped(x, experts),
        }.items():
            err = (
                (f(x) - ref).abs().max() / ref.abs().max()
            ).item()  # deltas / keep make outputs larger
            assert err < 1e-5, (z, k, keep, block, name, err)


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
