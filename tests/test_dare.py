"""Seeded DARE: the Philox mask is the published Philox4x32-10, identical on CPU, CUDA and in Triton, and every
forward of DareDeltaMoE equals the plain MoE with W_b + mask * Delta / keep. python tests/test_dare.py also runs it.
"""

import itertools

import torch
from test_delta_moe import CASES, make, reference

from shared_delta_moe import DareDeltaMoE
from shared_delta_moe.constants import HAS_TRITON
from shared_delta_moe.models.dare import keep_mask, philox

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


def test_triton_randint_is_the_same_philox():
    if not GPU:
        print("skipped: no CUDA/Triton")
        return
    import triton
    import triton.language as tl

    @triton.jit
    def k(out, seed, stream, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        u = tl.randint(seed, (stream.to(tl.int64) << 32) + i.to(tl.int64))
        tl.store(out + i, u.to(tl.int64), mask=i < n)

    for seed, stream in ((0, 0), (12345, 7), (2**40 + 3, 14)):
        got = torch.empty(5000, dtype=torch.int64, device="cuda")
        k[(5,)](got, seed, stream, 5000, BLOCK=1024)
        u = torch.arange(5000)
        z = torch.zeros_like(u)
        want = philox(u, z + stream, z, z, seed & 0xFFFFFFFF, seed >> 32)
        assert torch.equal(got.cpu(), want), (seed, stream)


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


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
