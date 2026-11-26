"""compress.py: each method's storage and its basic property, and compress_deltas on DeltaMoE and HierDeltaMoE."""

import pytest
import torch

from shared_delta_moe import BitDeltaMoE, DeltaMoE, HierDeltaMoE, random_experts
from shared_delta_moe import compress as C

R, K = 48, 256


def data(seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(R, K, generator=g)
    x = torch.randn(512, K, generator=g) * torch.rand(K, generator=g) * 3  # uneven input scales
    return d, x


def out_err(approx, d, x):
    return ((x @ (approx - d).T).norm() / (x @ d.T).norm()).item()


def test_more_bits_less_error():
    d, x = data()
    for family in (C.rtn, lambda b: C.gptq(b)):
        errs = [out_err(family(b)(d, x.T @ x)[0], d, x) for b in (2, 3, 4)]
        assert errs[0] > errs[1] > errs[2], errs
    errs = [out_err(C.residual_sign(n)(d, None)[0], d, x) for n in (1, 2, 3)]
    assert errs[0] > errs[1] > errs[2], errs


def test_calibrated_beats_uncalibrated():
    """GPTQ / SparseGPT / Wanda use the inputs' Gram: lower output error than RTN / magnitude at the same storage."""
    d, x = data()
    h = x.T @ x
    assert out_err(C.gptq(2)(d, h)[0], d, x) < out_err(C.rtn(2)(d, h)[0], d, x)
    assert out_err(C.sparsegpt(0.3)(d, h)[0], d, x) < out_err(C.wanda(0.3)(d, h)[0], d, x) < out_err(C.magnitude(0.3)(d, h)[0], d, x)


def test_sparsity_and_bits():
    d, x = data()
    h = x.T @ x
    for m, keep in ((C.magnitude(0.25), 0.25), (C.wanda(0.25), 0.25), (C.wanda(0.25, 16), 0.25), (C.sparsegpt(0.25), 0.25)):
        a, bits = m(d, h)
        assert abs((a != 0).float().mean().item() - keep) < 0.02
    assert C.rtn(2, 64)(d, h)[1] == 2.5 and C.sparsegpt(0.25)(d, h)[1] == 5 and C.wanda(0.25, 16)(d, h)[1] == 4 + 1 / 16
    assert C.svd(K)(d, h)[0].sub(d).abs().max() < 1e-3  # full rank: exact


def test_dare_unbiased():
    """DARE and importance DARE are unbiased: the mean over many masks approaches the delta."""
    d, x = data()
    h = x.T @ x
    g = torch.Generator().manual_seed(1)
    for m in (C.dare(0.25, 16, generator=g), C.importance_dare(0.25, 16, generator=g),
              C.importance_dare(0.25, 16, rank1=False, generator=g)):  # fmt: skip
        mean = sum(m(d, h)[0] for _ in range(400)) / 400
        assert (mean - d).norm() / d.norm() < 0.15
        assert abs((m(d, h)[0] != 0).float().mean().item() - 0.25) < 0.03


def test_importance_dare_lower_variance():
    d, x = data()
    d = d * torch.rand(R, 1) * torch.rand(1, K) * 4  # uneven rows and columns, where importance helps
    h = x.T @ x
    g = torch.Generator().manual_seed(2)
    err = lambda m: sum(out_err(m(d, h)[0], d, x) for _ in range(20)) / 20
    assert err(C.importance_dare(0.2, 16, generator=g)) < err(C.dare(0.2, 16, generator=g))


@pytest.mark.parametrize("hier", [False, True])
def test_compress_deltas(hier):
    torch.manual_seed(0)
    H, I = 64, 128
    full = random_experts(H, I, 4, torch.float32, "cpu")
    if hier:
        m = HierDeltaMoE.from_experts(full[0], [full[1:3], full[3:4]], torch.randn(2 + 4, H), [0, 1], top_k=1)
    else:
        m = DeltaMoE.from_experts(full[0], full[1:], torch.randn(4, H), top_k=1, num_zero_experts=1)
    before = [t.clone() for t in (m.d_gate, m.d_up, m.d_down)]
    x = [torch.randn(200, H) for _ in range(m.n_delta)]
    grams = C.calibration_grams(m, x)
    assert grams[0][0].shape == (H, H) and grams[0][1].shape == (I, I)
    bits = C.compress_deltas(m, C.gptq(4, 64), grams)
    assert bits == 4.5
    for b, a in zip(before, (m.d_gate, m.d_up, m.d_down)):
        assert not torch.equal(a, b) and (a - b).norm() / b.norm() < 0.2
    y = m.forward_param(torch.randn(10, H))  # still a working module
    assert y.isfinite().all()


def test_packed_refused():
    full = random_experts(32, 64, 3, torch.float32, "cpu")
    m = DeltaMoE.from_experts(full[0], full[1:], torch.randn(3, 32), top_k=1, num_zero_experts=1)
    with pytest.raises(TypeError):
        C.compress_deltas(BitDeltaMoE.from_delta_moe(m), C.rtn(2, 32))


def test_edges():
    """keep=1 keeps everything; importance DARE with zero blocks stays finite."""
    d, x = data()
    h = x.T @ x
    for m in (C.magnitude(1.0), C.sparsegpt(1.0), C.wanda(1.0)):
        assert torch.allclose(m(d, h)[0], d, atol=1e-4)
    z = d.clone()
    z[:, :128] = 0
    for rank1 in (True, False):
        a = C.importance_dare(0.8, 16, rank1=rank1)(z, h)[0]
        assert a.isfinite().all() and torch.equal(a[:, :128], z[:, :128])


def test_gptq_on_grid():
    """Every GPTQ group has at most 2^bits distinct values."""
    d, x = data()
    q = C.gptq(2, 64)(d, x.T @ x)[0]
    assert max(len(g.unique()) for g in q.reshape(-1, 64)) <= 4
