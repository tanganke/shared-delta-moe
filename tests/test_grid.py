"""GridDeltaMoE / GridHierDeltaMoE: code packing, agreement with compress.quantize, forwards, trainable grids."""

import pytest
import torch

from shared_delta_moe import DeltaMoE, GridDeltaMoE, GridHierDeltaMoE, HierDeltaMoE, random_experts
from shared_delta_moe import compress as C
from shared_delta_moe.models.grid import pack_codes, unpack_codes

H, I = 64, 128


def moe(hier):
    torch.manual_seed(0)
    full = random_experts(H, I, 4, torch.float32, "cpu")
    if hier:
        return HierDeltaMoE.from_experts(full[0], [full[1:3], full[3:4]], torch.randn(2 + 4, H), [0, 1], top_k=1)
    return DeltaMoE.from_experts(full[0], full[1:], torch.randn(4, H), top_k=1, num_zero_experts=1)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_pack_roundtrip(bits):
    q = torch.randint(0, 2**bits, (3, 5, 64))
    w = pack_codes(q, bits)
    assert w.dtype == torch.int32 and w.shape == (3, 5, 64 * bits // 32)
    assert torch.equal(unpack_codes(w, bits), q.to(torch.int32))


@pytest.mark.parametrize("hier", [False, True])
@pytest.mark.parametrize("levels,symmetric,gptq", [(4, False, False), (3, True, True), (2, False, True)])
def test_matches_simulated(hier, levels, symmetric, gptq):
    """The packed deltas are compress's simulated ones, and the module computes what the dense module with them
    computes."""
    m = moe(hier)
    x = [torch.randn(300, H) for _ in range(m.n_delta)] if gptq else None
    p = GridDeltaMoE.from_delta_moe(m, levels, 32, symmetric, clip=True, x=x)
    assert isinstance(p, GridHierDeltaMoE) == hier and isinstance(p, HierDeltaMoE) == hier
    method = (C.gptq if gptq else C.rtn)(levels=levels, group=32, symmetric=symmetric, clip=True)
    grams = None if x is None else C.calibration_grams(m, x)
    C.compress_deltas(m, method, grams)
    assert p.bits_per_entry() == p.bits + (1 if symmetric else 2) * 32 / 32  # float32 grids here
    for n in ("d_gate", "d_up", "d_down"):
        torch.testing.assert_close(getattr(p, n), getattr(m, n), rtol=0, atol=1e-6)
    t = torch.randn(20, H)
    torch.testing.assert_close(p.forward_param(t), m.forward_param(t), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(p(t), m(t), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("hier", [False, True])
def test_grids_trainable(hier):
    p = GridDeltaMoE.from_delta_moe(moe(hier), 4, 32)
    p.forward_param(torch.randn(20, H)).square().sum().backward()
    assert p.s_gate.grad is not None and p.s_gate.grad.abs().sum() > 0 and p.z_down.grad.abs().sum() > 0
    assert p.c_gate.grad is None  # the codes are buffers


def test_state_dict_roundtrip():
    p = GridDeltaMoE.from_delta_moe(moe(False), 3, 32, symmetric=True)
    q = GridDeltaMoE(H, I, 3, 1, 1, levels=3, group=32, symmetric=True)
    q.load_state_dict(p.state_dict())
    assert torch.equal(q.d_up, p.d_up) and not hasattr(q, "z_up")


@pytest.mark.parametrize("hier", [False, True])
def test_bf16_gptq(hier):
    """A bf16 module: GPTQ runs (in float32) and the deltas are the simulated ones up to bf16 rounding."""
    m = moe(hier).to(torch.bfloat16)
    x = [torch.randn(300, H, dtype=torch.bfloat16) for _ in range(m.n_delta)]
    p = GridDeltaMoE.from_delta_moe(m, 3, 32, symmetric=True, clip=True, x=x)
    assert p.s_up.dtype == torch.bfloat16 and p.bits_per_entry() == 2 + 16 / 32
    C.compress_deltas(m, C.gptq(levels=3, group=32, symmetric=True, clip=True), C.calibration_grams(m, x))
    torch.testing.assert_close(p.d_down.float(), m.d_down.float(), rtol=0, atol=float((m.d_down.abs().max()) / 64))


def test_flat_into_hier_refused():
    with pytest.raises(TypeError, match="route differently"):
        GridHierDeltaMoE.from_delta_moe(moe(False))
