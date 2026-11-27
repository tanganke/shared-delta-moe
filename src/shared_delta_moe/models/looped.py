r"""Looped stacks: a stack of layers run several times on one carried state, the loop of qihuangz's
Mixture-of-Loop (MoL).

Round 0 runs the stack :math:`f` on its input, the anchor :math:`a`. Every later round runs it on the anchor plus a
low-rank feedback of the state and moves the state part of the way to the result:

.. math::

   h_0 = f(a), \qquad v_r = f\bigl(a + U D\,\mathrm{rms}(h_{r-1})\bigr), \qquad
   h_r = h_{r-1} + \sigma(u)\,(v_r - h_{r-1}),

with :math:`U` zero at init (so at init every round repeats round 0 and the loop is the plain stack) and
:math:`\sigma(u)` = ``update_init`` (0.1) at init.

MoE blocks in the stack (made ``loopable``) route each round (``"dynamic"``) or reuse round 0's indices and weights
in every later round (``"fixed"``; the weights keep their gradient to the router). The reuse is keyed by block and
router, so
a layer's routed attention deltas (``routed_linear``, routed through ``moe.topk(x, delta_router)``) are fixed with
it. Under activation checkpointing, wrap each checkpointed function of the stack in ``keep_routing``: the recompute
(during backward, after ``loop`` returned) then routes as that round did, replaying round 0's routing in later rounds
under fixed routing (and ``ops.replay`` keeps round 0's own picks).
"""

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .delta_moe import DeltaMoE
from .hier import HierDeltaMoE

ROUTINGS = ("fixed", "dynamic")


class LoopUpdate(nn.Module):
    r"""The feedback :math:`U D\,\mathrm{rms}(h)` and the update gate :math:`\sigma(u)` of a loop, in float32.

    Args:
        hidden: H, the state width.
        rank: Rank of the feedback.
        eps: RMS norm epsilon (the model's).
        update_init: :math:`\sigma(u)` at init, in (0, 1).
        device: Device of the parameters.
    """

    def __init__(self, hidden: int, rank: int, eps: float, update_init: float = 0.1,
                 device: torch.device | str | None = None) -> None:  # fmt: skip
        super().__init__()
        assert 0 < update_init < 1 and rank > 0
        self.eps = eps
        self.down = nn.Linear(hidden, rank, bias=False, device=device, dtype=torch.float32)  # kaiming uniform
        self.up = nn.Linear(rank, hidden, bias=False, device=device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        self.update_logit = nn.Parameter(torch.tensor(math.log(update_init / (1 - update_init)), device=device))

    def feedback(self, h: Tensor) -> Tensor:
        """(..., H) -> (..., H) in h's dtype: only the new branch is normalised."""
        f = h.float()
        return self.up(self.down(f * torch.rsqrt(f.square().mean(-1, keepdim=True) + self.eps))).to(h.dtype)

    def blend(self, h: Tensor, value: Tensor) -> Tensor:
        r""":math:`h + \sigma(u)(v - h)`: exactly h when v == h."""
        return h + self.update_logit.sigmoid().to(h.dtype) * (value - h)


class _Routes:
    def __init__(self, fixed: bool, aux: list | None) -> None:
        self.fixed, self.aux, self.replay, self.saved = fixed, aux, False, {}


def balance_loss(moe: DeltaMoE, x: Tensor, idx: Tensor) -> Tensor:
    r"""Switch balance loss of one routing call, :math:`E \sum_e f_e P_e` over the router's E rows (f: share of
    slots, stop-gradient; P: mean softmax). Flat DeltaMoE routers only (a HierDeltaMoE's rows are not its experts).

    Args:
        moe: The block.
        x: (T, H) its input.
        idx: (T, K) the router indices ``topk`` picked.

    Returns:
        Scalar, float32; 1 when balanced.
    """
    assert not isinstance(moe, HierDeltaMoE), "balance_loss: flat routers only"
    p = moe.router(x.to(moe.router.weight.dtype)).softmax(-1, dtype=torch.float)
    f = F.one_hot(idx, p.shape[-1]).float().mean(1).mean(0)
    return p.shape[-1] * (f.detach() * p.mean(0)).sum()


class _LoopRouting:
    """Mixin (``loopable``): ``topk`` as the class's, except inside ``loop`` (``_loop_routes`` set)."""

    _loop_routes: _Routes | None = None

    def topk(self, x: Tensor, router: nn.Module | None = None) -> tuple[Tensor, Tensor]:
        s = self._loop_routes
        if s is None:
            return super().topk(x, router)  # type: ignore[misc]
        key = id(self), id(router)  # one loop holds several blocks; None = the block's own router
        if s.replay:
            return s.saved[key]
        out = super().topk(x, router)  # type: ignore[misc]
        if s.fixed:
            s.saved[key] = out
        if s.aux is not None and router is None:
            s.aux.append(balance_loss(self, x, out[0]))  # type: ignore[arg-type]
        return out


_LOOPABLE: dict[type, type] = {}


def loopable(moe: DeltaMoE) -> DeltaMoE:
    """Let ``loop`` fix and collect ``moe``'s routing: its class becomes a subclass with ``_LoopRouting`` (in place,
    idempotent; nothing changes outside a loop).

    Args:
        moe: Any DeltaMoE (HierDeltaMoE, packed formats).

    Returns:
        moe.
    """
    cls = type(moe)
    if not issubclass(cls, _LoopRouting):
        if cls not in _LOOPABLE:
            _LOOPABLE[cls] = type(f"Looped{cls.__name__}", (_LoopRouting, cls), {})
        moe.__class__ = _LOOPABLE[cls]
    return moe


def keep_routing(fn: Callable[..., Any], moes: Iterable[Any]) -> Callable[..., Any]:
    """``fn`` run with the routing state of the loop round that is current now, also when called later: wrap the
    function an activation checkpoint inside ``stack`` recomputes. Outside a loop, ``fn`` itself.

    Args:
        fn: The checkpointed function (one or more layers of the stack).
        moes: The loop's ``loopable`` blocks (as passed to ``loop``).

    Returns:
        The wrapped function.
    """
    moes = list(moes)
    s = next((m._loop_routes for m in moes if m._loop_routes is not None), None)
    if s is None:
        return fn
    replay = s.replay

    def run(*args: Any, **kwargs: Any) -> Any:
        prev, was = [m._loop_routes for m in moes], s.replay
        for m in moes:
            m._loop_routes = s
        s.replay = replay
        try:
            return fn(*args, **kwargs)
        finally:
            s.replay = was
            for m, p in zip(moes, prev):
                m._loop_routes = p

    return run


def loop(stack: Callable[[Tensor, int], Tensor], x: Tensor, update: LoopUpdate | None, repeats: int,
         moes: Iterable[Any] = (), routing: str = "fixed", aux: list | None = None) -> Tensor:  # fmt: skip
    """Run ``stack`` ``repeats`` times on ``x`` as the module docstring says.

    Args:
        stack: ``stack(h, r)`` runs the layers once in round r (r picks per-round caches).
        x: The anchor, the stack's input.
        update: The loop's ``LoopUpdate``; None only for ``repeats`` 1.
        repeats: Rounds, >= 1.
        moes: The ``loopable`` MoE blocks of the stack.
        routing: ``"fixed"`` or ``"dynamic"``.
        aux: A list to append the balance loss of every routing call to (fixed: round 0 only); None = none. Read it
            before backward: checkpoint recomputes append again.

    Returns:
        The state after the last round.
    """
    assert routing in ROUTINGS and repeats >= 1 and (update is not None or repeats == 1)
    moes = list(moes)
    assert all(isinstance(m, _LoopRouting) for m in moes), "loopable(moe) first"
    s = _Routes(routing == "fixed", aux)
    for m in moes:
        m._loop_routes = s
    try:
        h = stack(x, 0)
        s.replay = s.fixed
        for r in range(1, repeats):
            h = update.blend(h, stack(x + update.feedback(h), r))  # type: ignore[union-attr]
    finally:
        for m in moes:
            m._loop_routes = None
    return h
