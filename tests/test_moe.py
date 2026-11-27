"""MoE / HierMoE, the plain MoEs with full experts: the loop and the grouped forward equal the MoE written out token by
token, with gradients (x, router, every expert weight); built on meta and loaded from a state dict.
python tests/test_moe.py also runs it."""

import torch
import torch.nn.functional as F

from shared_delta_moe import HierMoE, MoE, mlp, random_experts

H, I = 16, 24


def reference(m: MoE, x: torch.Tensor) -> torch.Tensor:
    """sum_i r_i MLP(x; W_{e_i}) per token, from the module's routing and its own weights."""
    idx, r = m.topk(x)
    return torch.stack([sum(ri.to(x.dtype) * mlp(xt, m.gate[e], m.up[e], m.down[e]) for e, ri in zip(es.tolist(), rs))
                        for xt, es, rs in zip(x, idx, r)])  # fmt: skip


def hier_reference(m: HierMoE, x: torch.Tensor) -> torch.Tensor:
    """The two-level MoE written out per token from the router weight, not through ``topk``: h = softmax over the
    groups (top-k0 renormalised), then top-k1 of each kept group's own softmax, renormalised."""
    G, out = len(m.group_sizes), []
    for xt in x:
        lg = (m.router.weight @ xt).float()
        h = lg[:G].softmax(-1)
        S = range(G) if m.group_top_k is None else h.topk(m.group_top_k).indices.tolist()
        y, c = 0, G
        for i, E in enumerate(m.group_sizes):
            if i in S:
                pv, pj = lg[c : c + E].softmax(-1).topk(m.k1)
                for v, j in zip(pv / pv.sum(), pj.tolist()):
                    e = c - G + j
                    y = y + (h[i] / h[list(S)].sum() * v).to(x.dtype) * mlp(xt, m.gate[e], m.up[e], m.down[e])
            c += E
        out.append(y)
    return torch.stack(out)


def flat(k: int) -> MoE:
    torch.manual_seed(0)
    return MoE.from_experts(random_experts(H, I, 5, torch.float32, "cpu"), torch.randn(5, H), k)


def hier(k1: int, k0: int | None, sizes=(2, 1, 3)) -> HierMoE:
    torch.manual_seed(0)
    full, groups = random_experts(H, I, sum(sizes), torch.float32, "cpu"), []
    for e in sizes:
        groups, full = groups + [full[:e]], full[e:]
    return HierMoE.from_experts(groups, torch.randn(len(sizes) + sum(sizes), H), k1, k0)


def modules():
    yield from (flat(k) for k in (1, 2, 5))  # k = E: soft routing over every expert
    yield from (hier(1, k0) for k0 in (None, 1, 2))
    yield from (hier(2, k0, (2, 3, 4)) for k0 in (None, 2))  # sizes that differ reversed: group order counts


def check(m: MoE, x: torch.Tensor) -> None:
    forms = {"loop": m._forward_loop, "module": m}
    if hasattr(F, "grouped_mm"):
        forms["grouped"] = m._forward_grouped
    grads = {}
    ref = (lambda x: hier_reference(m, x)) if isinstance(m, HierMoE) else (lambda x: reference(m, x))
    for name, f in ({"reference": ref} | forms).items():
        xg = x.clone().requires_grad_()
        m.zero_grad()
        y = f(xg)
        (y**2).sum().backward()
        grads[name] = [y, xg.grad] + [p.grad.clone() for p in m.parameters()]
    for name, g in grads.items():
        for a, b in zip(g, grads["reference"]):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4, msg=f"{name} top_k={m.top_k}")


def test_forwards_and_gradients_match_reference():
    x = torch.randn(60, H)
    for m in modules():
        idx, r = m.topk(x)
        assert idx.unique().numel() == m.n_experts, "every expert is hit"
        torch.testing.assert_close(r.sum(-1), torch.ones(len(x)))
        check(m, x)
        check(m, x[:1])  # one token: the module takes the loop


def test_bf16():
    """bf16 weights and tokens: every forward returns bf16, equal to the loop up to bf16 rounding."""
    m = flat(2).to(torch.bfloat16)
    x = torch.randn(40, H, dtype=torch.bfloat16)
    with torch.no_grad():
        want = m._forward_loop(x)
        forms = [m] + ([m._forward_grouped] if hasattr(F, "grouped_mm") else [])
        for f in forms:
            y = f(x)
            assert y.dtype == torch.bfloat16
            torch.testing.assert_close(y, want, atol=1e-2, rtol=1e-2)


def test_hier_routing_is_two_level():
    """A HierMoE slot weight is h_i p_ij: with one expert per group and dense routing it is the group softmax."""
    torch.manual_seed(0)
    full = random_experts(H, I, 3, torch.float32, "cpu")
    m = HierMoE.from_experts([[w] for w in full], torch.randn(6, H), 1, None)
    x = torch.randn(10, H)
    idx, r = m.topk(x)
    assert idx.tolist() == [[0, 1, 2]] * 10
    torch.testing.assert_close(r, m.router(x)[:, :3].softmax(-1))


def test_tables_from_inference_mode_train():
    """The routing tables are cached per layout and device across modules: ones first built under inference_mode
    must still let another module (or the same one) train."""
    x = torch.randn(8, H)
    with torch.inference_mode():
        hier(1, None, (2, 5))(x)
    hier(1, None, (2, 5))(x.requires_grad_()).sum().backward()


def test_meta_and_state_dict():
    x = torch.randn(30, H)
    for m, empty in ((flat(2), lambda: MoE(H, I, 5, 2, device="meta")),
                     (hier(1, 2), lambda: HierMoE(H, I, [2, 1, 3], 1, 2, device="meta"))):  # fmt: skip
        n = empty().to_empty(device="cpu")
        n.load_state_dict(m.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(n(x.view(3, 10, H)), m(x).view(3, 10, H))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
