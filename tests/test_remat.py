"""The materialised training path recomputes W_b + Delta_e in the backward instead of keeping it
(DeltaMoE._mlp_remat): outputs and every gradient equal those of the plain path, for DeltaMoE, HierDeltaMoE, DARE,
BitDelta and grid deltas (flat and Hier) and each set of trained weights (router only, base, deltas, all); the swap really happens (each
materialised weight is unpacked once per backward); under activation checkpointing the plain path runs. CPU, fp32."""

import torch
from torch.utils.checkpoint import checkpoint

from shared_delta_moe import BitDeltaMoE, DareDeltaMoE, DeltaMoE, GridDeltaMoE, HierDeltaMoE, random_experts

H, I = 64, 96


def models():
    torch.manual_seed(0)
    full = random_experts(H, I, 7, torch.float32, "cpu")
    flat = DeltaMoE.from_experts(full[0], full[1:], torch.randn(7, H) * 0.3, top_k=2, num_zero_experts=1)
    hier = HierDeltaMoE.from_experts(full[0], [full[1:4], full[4:7]], torch.randn(2 + 4 + 4, H) * 0.3, [1, 1], top_k=2)
    return {"flat": flat, "hier": hier, "dare": DareDeltaMoE.from_delta_moe(flat, keep=0.5, block=8, seed=0),
            "bitdelta": BitDeltaMoE.from_delta_moe(flat, scale="row"),
            "grid": GridDeltaMoE.from_delta_moe(flat, levels=4, group=32),
            "grid-hier": GridDeltaMoE.from_delta_moe(hier, levels=4, group=32)}


def run(m, x, plain):
    m.remat = not plain
    m.zero_grad(set_to_none=True)
    xx = x.clone().requires_grad_(True)
    y = m._forward_param(xx)
    y.backward(torch.ones_like(y))
    return y.detach(), xx.grad, {n: p.grad for n, p in m.named_parameters() if p.requires_grad}


def test_same_values_and_gradients():
    x = torch.randn(40, H)
    for name, m in models().items():
        for train in ("router", "base", "deltas", "all"):
            for n, p in m.named_parameters():
                base, delta = n in ("gate", "up", "down"), not n.startswith("router") and n not in ("gate", "up", "down")
                p.requires_grad_(n.startswith("router") or (base and train in ("base", "all")) or (delta and train in ("deltas", "all")))
            y0, gx0, g0 = run(m, x, plain=True)
            y1, gx1, g1 = run(m, x, plain=False)
            assert torch.equal(y0, y1) and torch.equal(gx0, gx1), (name, train)
            assert g0.keys() == g1.keys() and all(torch.equal(g0[k], g1[k]) for k in g0), (name, train)


def test_swap_happens_and_checkpoint_falls_back():
    m = models()["flat"]
    calls = []
    weights = m.weights
    m.weights = lambda e: calls.append(e) or weights(e)
    x = torch.randn(40, H, requires_grad=True)
    y = m._forward_param(x)
    n_fwd, n_delta = len(calls), sum(e >= 0 for e in calls)  # a zero expert (e < 0) is W_b itself: nothing to swap
    y.sum().backward()
    assert n_delta and len(calls) == n_fwd + n_delta, (n_fwd, n_delta, len(calls))  # one recompute per routed delta
    calls.clear()
    checkpoint(m._forward_param, x, use_reentrant=False).sum().backward()  # plain path under the checkpoint hooks
    assert len(calls) == 2 * n_fwd, len(calls)  # the checkpoint's own recompute of the forward, no weight recipe
    del m.weights


def test_materialised_weights_are_freed():
    import gc
    import weakref

    m = models()["flat"]
    refs = []
    weights = m.weights
    m.weights = lambda e: (w := weights(e), refs.extend(weakref.ref(t) for t in w if e >= 0))[0]
    y = m._forward_param(torch.randn(40, H, requires_grad=True))
    gc.collect()
    assert refs and all(r() is None for r in refs), sum(r() is not None for r in refs)  # nothing holds W_b + Delta_e
    y.sum().backward()
    del m.weights


if __name__ == "__main__":
    test_same_values_and_gradients()
    test_swap_happens_and_checkpoint_falls_back()
    test_materialised_weights_are_freed()
    print("ok")
