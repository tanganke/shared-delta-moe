"""ops.replay: an activation-checkpoint recompute reuses the forward's routing picks."""

import torch
from torch.utils.checkpoint import checkpoint

from shared_delta_moe.models.delta_moe import DeltaMoE
from shared_delta_moe.models.hier import HierDeltaMoE


def _check(moe):
    """The recompute sees inputs that flip every pick (sign of x reversed); it must still route as the forward."""
    torch.manual_seed(0)
    for p in moe.parameters():
        torch.nn.init.normal_(p, std=0.1)
    torch.nn.init.normal_(moe.router.weight)
    x = torch.randn(16, 8)
    calls, seen = [0], []

    def f(x):
        calls[0] += 1
        xi = x if calls[0] == 1 else -x  # the recompute: different routing inputs
        seen.append(moe.topk(xi)[0])
        return moe(x).sum() + moe.topk(xi)[1].sum()

    checkpoint(f, x, use_reentrant=False).backward()
    assert calls[0] == 2 and torch.equal(seen[0], seen[1])
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0


def test_flat():
    _check(DeltaMoE(8, 12, 3, top_k=2))


def test_hier():
    _check(HierDeltaMoE(8, 12, [2, 2], [1, 1], 1, group_top_k=1))


def test_backward_order_and_no_checkpoint():
    """Two checkpointed micro-batches backpropagated oldest first each replay their own picks; outside a
    checkpoint nothing is kept."""
    from shared_delta_moe import ops

    moe = DeltaMoE(8, 12, 3, top_k=2)
    for p in moe.parameters():
        torch.nn.init.normal_(p, std=0.1)
    torch.nn.init.normal_(moe.router.weight)
    x1, x2 = torch.randn(16, 8), torch.randn(16, 8)
    seen: dict[int, list] = {1: [], 2: []}
    n = {1: 0, 2: 0}

    def f(x, k):
        n[k] += 1
        seen[k].append(moe.topk(x if n[k] == 1 else -x)[0])
        return moe(x).sum()

    l1 = checkpoint(f, x1, 1, use_reentrant=False)
    l2 = checkpoint(f, x2, 2, use_reentrant=False)
    l1.backward()
    l2.backward()
    assert torch.equal(seen[1][0], seen[1][1]) and torch.equal(seen[2][0], seen[2][1])
    moe(x1).sum().backward()
    assert len(ops._PICKS) == 0


def test_retain_graph():
    """Each backward of a retained graph recomputes; every recompute replays the forward's picks."""
    moe = DeltaMoE(8, 12, 3, top_k=2)
    for p in moe.parameters():
        torch.nn.init.normal_(p, std=0.1)
    torch.nn.init.normal_(moe.router.weight)
    x, seen = torch.randn(16, 8), []

    def f(x):
        seen.append(moe.topk(x if not seen else -x)[0])
        return moe(x).sum()

    loss = checkpoint(f, x, use_reentrant=False)
    loss.backward(retain_graph=True)
    loss.backward()
    assert len(seen) == 3 and all(torch.equal(seen[0], s) for s in seen)
