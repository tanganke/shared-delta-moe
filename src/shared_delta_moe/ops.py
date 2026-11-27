"""Stateless building blocks: the SwiGLU MLP, grouping routed rows, one grouped GEMM, and bit packing."""

import contextlib
import inspect
import weakref
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .constants import GROUPED_DTYPES, GROUPED_MIN_PAIRS
from .type_hints import (
    Group,
    GroupEnds,
    GroupedOut,
    GroupedRows,
    InWeight,
    OutWeight,
    Rows,
    SlotExperts,
    SlotWeights,
    StackedWeight,
    Tokens,
)


def pow2(n: int) -> int:
    r"""Smallest power of 2 that is at least :math:`n` (:math:`n \ge 1`)."""
    return 1 << (n - 1).bit_length()


def pack_bits(b: torch.Tensor) -> torch.Tensor:
    r"""Pack booleans along the last dimension into int32 words.

    Args:
        b: (..., C) bool.

    Returns:
        (..., ceil(C / 32)) int32: entry :math:`c` in bit :math:`c \bmod 32` of word :math:`\lfloor c / 32 \rfloor`,
        set where True; the padding bits of the last word are clear.
    """
    words = -(-b.shape[-1] // 32)
    b = F.pad(b, (0, words * 32 - b.shape[-1])).view(*b.shape[:-1], words, 32).long()
    w = (b << torch.arange(32, device=b.device)).sum(-1)
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def unpack_bits(w: torch.Tensor) -> torch.Tensor:
    """Unpack ``pack_bits`` words.

    Args:
        w: (..., W) int32 words.

    Returns:
        (..., 32 W) int32 of 0 and 1, padding bits included.
    """
    return ((w[..., None] >> torch.arange(32, dtype=torch.int32, device=w.device)) & 1).flatten(-2)


def mlp(x: Tokens, gate: InWeight, up: InWeight, down: OutWeight) -> Tokens:
    r"""SwiGLU MLP: :math:`\mathrm{down}(\mathrm{silu}(\mathrm{gate}\, x) \odot \mathrm{up}\, x)`.

    Args:
        x: (T, H) tokens.
        gate: (I, H) gate weight.
        up: (I, H) up weight.
        down: (H, I) down weight.

    Returns:
        (T, H) MLP output.
    """
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def softmax_topk(logits: torch.Tensor, k: int, pick: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k routing: float32 softmax (as transformers' MoE routers; bf16 can flip near-ties), top-k, renormalised.

    Args:
        logits: (T, E) router output.
        k: experts picked per token.
        pick: (T, k) indices to use instead of the top-k (``replay``); None = the top-k.

    Returns:
        (idx, r): idx (T, k) int64 expert indices; r (T, k) float32 weights, rows summing to 1.
    """
    p = logits.softmax(-1, dtype=torch.float)
    i = p.topk(k, dim=-1).indices if pick is None else pick
    r = p.gather(-1, i)
    return i, r / r.sum(-1, keepdim=True)


_PICKS: "weakref.WeakKeyDictionary[Any, dict[Any, Any]]" = weakref.WeakKeyDictionary()


def _checkpoint_frame() -> tuple[Any, Any] | None:
    """The non-reentrant ``torch.utils.checkpoint`` call this code runs in, from torch's own saved-tensor hooks
    (private: the forward's pack hook closes over ``frame``, the recompute's over ``target_frame_ref`` and ``gid``,
    its graph task): (frame, None) in the forward, (frame, gid) in a recompute; None outside one (or when torch's
    internals differ)."""
    top = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    hooks = top(False) if top is not None else None
    if not hooks:
        return None
    pack = inspect.unwrap(hooks[0])  # the recompute's is wrapped by torch._dynamo.disable
    if not getattr(pack, "__qualname__", "").startswith(("_checkpoint_hook.", "_recomputation_hook.")):
        return None  # another library's hooks
    cells = dict(zip(pack.__code__.co_freevars, pack.__closure__ or ()))
    if "frame" in cells:
        return cells["frame"].cell_contents, None
    if "target_frame_ref" in cells and "gid" in cells:
        frame = cells["target_frame_ref"].cell_contents()
        return (frame, cells["gid"].cell_contents) if frame is not None else None
    return None


def replay(router: torch.nn.Module, logits: torch.Tensor, pick: Callable[[], Any]) -> Any:
    """The routing decision ``pick()`` (indices, no gradient), recompute-safe under activation checkpointing.

    A checkpointed block recomputes its forward during backward, from inputs that need not be bitwise equal (atomic
    adds, non-deterministic kernels); a near-tie can then flip a top-k pick, change how many tokens an expert gets
    and fail the recompute (``CheckpointError``) or silently mismatch the gradients. So inside a non-reentrant
    checkpoint the forward keeps each router's picks on the checkpoint's own frame (freed with it), and every
    recompute (one per backward, ``retain_graph`` included) replays them in call order, whatever order the backwards
    run in; the weights are still recomputed from the logits, so the router gets its gradient. Elsewhere (no
    checkpoint, reentrant checkpoints, inference, torch.compile) it is ``pick()``.

    Args:
        router: The router module.
        logits: Its output for these tokens.
        pick: Computes the picks from the logits.

    Returns:
        The picks: this forward's, or in a recompute the forward's.

    Raises:
        RuntimeError: For a checkpoint nested in a recomputed checkpoint (not supported).
    """
    if torch.compiler.is_compiling() or not torch.is_grad_enabled():
        return pick()
    at = _checkpoint_frame()
    if at is None:
        return pick()
    frame, gid = at
    state = _PICKS.setdefault(frame, {"calls": {}, "read": {}})
    calls = state["calls"].setdefault(id(router), [])
    if gid is None:
        if torch._C._current_graph_task_id() != -1:
            raise RuntimeError("ops.replay: a checkpoint nested in a recomputed checkpoint would route afresh")
        calls.append((logits.shape[0], pick()))
        return calls[-1][1]
    key = (gid, id(router))
    n = state["read"].get(key, 0)
    if n >= len(calls):
        return pick()
    state["read"][key] = n + 1
    t, p = calls[n]
    return p if t == logits.shape[0] else pick()


def use_grouped(x: Tokens, top_k: int) -> bool:
    """Use grouped GEMM? Needs ``F.grouped_mm`` with this dtype on this device (CPU, or CUDA sm80+), and more than
    one (token, slot) pair (with one pair the sort and count cost more than the loop saves).

    Args:
        x: (T, H) tokens.
        top_k: slots per token.

    Returns:
        True to take the grouped-GEMM forward, False for the loop.
    """
    if not hasattr(F, "grouped_mm") or x.dtype not in GROUPED_DTYPES or x.device.type not in ("cuda", "cpu"):
        return False
    if x.device.type == "cuda" and torch.cuda.get_device_capability(x.device) < (8, 0):
        return False
    return x.shape[0] * top_k >= GROUPED_MIN_PAIRS


def groups(e: SlotExperts, r: SlotWeights) -> list[Group]:
    """Group the tokens of one routing slot by the expert they use.

    Args:
        e: (T,) int64 delta index per token for this slot, -1 = any zero expert.
        r: (T,) routing weight per token for this slot.

    Returns:
        One (k, rows, coef) per distinct expert in e: k the delta index or -1, rows (R,) int64 token indices routed
        to it, coef (R, 1) their routing weights.
    """
    out: list[Group] = []
    for k in e.unique().tolist():
        rows = (e == k).nonzero().squeeze(-1)
        out.append((k, rows, r[rows, None]))
    return out


def sort_rows(e: SlotExperts, n: int) -> tuple[Rows, GroupEnds]:
    r"""Rows with :math:`e \ge 0`, sorted by :math:`e`, and the cumulative group sizes that ``grouped_mm`` takes as ``offs``.

    Args:
        e: (T,) int64 group index per row; rows with :math:`e < 0` are dropped.
        n: Number of groups G; e must be < n.

    Returns:
        A tuple (rows, offs): rows (R,) int64 indices of the rows with :math:`e \ge 0`, sorted by :math:`e`; offs (G,) int32
        cumulative group sizes.
    """
    rows = (e >= 0).nonzero().squeeze(-1)
    rows = rows[e[rows].argsort()]
    return rows, torch.bincount(e[rows], minlength=n).cumsum(0).to(torch.int32)


def gmm(a: GroupedRows, w: StackedWeight, offs: GroupEnds) -> GroupedOut:
    r"""One grouped GEMM: rows ``offs[g-1]:offs[g]`` of ``a`` times :math:`w[g]^\top` (CUDA sm80+, or CPU;
    bf16/fp16/fp32).

    Args:
        a: (R, in) rows sorted by group.
        w: (G, out, in) one weight per group.
        offs: (G,) int32 cumulative group sizes, as from ``sort_rows``.

    Returns:
        (R, out) row i of a times the transposed weight of its group.
    """
    return F.grouped_mm(a, w.mT, offs=offs)


@contextlib.contextmanager
def recompute_saved(ws: "Sequence[torch.Tensor]", recompute: "Callable[[], Sequence[torch.Tensor]]") -> Iterator[None]:
    """Inside: autograd does not keep the tensors ``ws`` (or views of them) for the backward but a recipe, and
    ``recompute()`` (no grad; the same tensors in the same order) rebuilds them when the backward needs one, so a
    training step holds activations, not weights built per call (``DeltaMoE._mlp_remat``, the grid decode). A no-op
    under other saved-tensor hooks (activation checkpointing, offloading), which already decide what is kept, without
    grad, or without torch's (private) hook query. The recomputed tensors must equal the originals.

    Args:
        ws: Contiguous tensors built in the forward (each with its own storage).
        recompute: Rebuilds them.
    """
    top = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    if not torch.is_grad_enabled() or top is None or top(False) is not None:
        yield
        return
    # metadata only: autograd keeps the hooks with every saved tensor, so a closure over ws would keep them alive
    ptr = {t.untyped_storage().data_ptr(): (i, t.dtype, t.untyped_storage().nbytes()) for i, t in enumerate(ws)}
    cache: dict[int, torch.Tensor] = {}

    def pack(t: torch.Tensor) -> "torch.Tensor | tuple":
        # the tensor itself or a view of it (F.linear may save W.t()); ws[i] is contiguous, so offset 0 = same start
        i, dtype, nbytes = ptr.get(t.untyped_storage().data_ptr(), (-1, None, 0))
        if i < 0 or t.dtype != dtype or t.untyped_storage().nbytes() != nbytes:
            return t
        return i, t.shape, t.stride(), t.storage_offset()

    def unpack(p: "torch.Tensor | tuple") -> torch.Tensor:
        if isinstance(p, torch.Tensor):
            return p
        i, shape, stride, offset = p
        if i not in cache:
            with torch.no_grad():
                cache.update(enumerate(recompute()))
        return cache.pop(i).as_strided(shape, stride, offset)

    del ws
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield
