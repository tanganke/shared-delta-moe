r"""Upscale a dense Qwen3.5 / Qwen3.8 base and its domain fine-tunes into the HierDeltaMoE layout of ``modeling.py``.

One group per domain; each fine-tune's decoder MLP becomes an expert :math:`\theta_{ij}` of its group, stored as
:math:`W_b` (the base MLP) and :math:`\Delta_{ij} = \theta_{ij} - W_b` (subtracted in float32, stored in the checkpoint
dtype). Every other tensor (attention, linear attention, norms, embeddings, ``lm_head``, vision tower, MTP) is merged
from the fine-tunes with a ``merging.MERGE`` method (``fusion_bench``, extra ``merging``; default ``simple_average``),
except with ``--routed-deltas``: then every attention and linear-attention projection and ``lm_head`` keeps
:math:`W_b` and each fine-tune's delta, dense or rounded to a grid (``--routed-levels``, ``compress.quantize`` round
to nearest), routed per token (``models/routed_linear.py``). Routers start at zero: uniform over
groups and within each group. With one expert per group, no zero experts and dense group routing, each layer's MLP is
then the plain average of the fine-tunes' MLPs; a zero expert is the base MLP.

Usage::

    python -m shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert BASE DST \
        --group medical=FT1,FT2 --group code=FT3 [--zeros 0] [--top-k 1] [--group-top-k K] \
        [--merge ties_merging --merge-arg scaling_factor=0.3 --merge-arg threshold=20] \
        [--levels 4 --grid-group 128 [--symmetric]] \
        [--routed-deltas [--routed-levels 4 --routed-group 128 [--routed-symmetric]] [--routed-sets 4,4,4,2]]

``--levels`` stores the MLP deltas as group grids instead (``models/grid.py``, round to nearest, no data; loaded as
``Qwen3_5GridHierDeltaMoeBlock``). ``--routed-sets`` gives consecutive experts one shared routed delta, their mean.

Tensors are streamed one at a time from the safetensors shards and written in shards of ``--shard-gb``.
"""

import argparse
import functools
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from .... import compress as C
from ....merging import MERGE
from ...grid import code_bits, pack_codes
from ..common import Checkpoint, ShardWriter, copy_config

MLP = re.compile(r"(model\.(?:language_model\.)?layers\.\d+\.mlp)\.(gate|up|down)_proj\.weight")
# the RoutedDeltaLinears of models/qwen3_5/common.py; group 1 or 2: the module whose delta_router routes them
ROUTED = re.compile(r"(model\.(?:language_model\.)?layers\.\d+\.(?:self_attn|linear_attn))\."
                    r"(?:[qkvo]_proj|in_proj_(?:qkv|z|b|a)|out_proj)\.weight|(lm_head)\.weight")


class Source(Protocol):
    def keys(self) -> list[str]: ...
    def get(self, key: str) -> torch.Tensor: ...


class Mean:
    """One expert that is the mean of several fine-tunes (e.g. seeds), read one tensor at a time; fp32 sum, the
    first source's dtype. Keys: those every source has."""

    def __init__(self, sources: Sequence[Source]) -> None:
        self.sources = sources

    def keys(self) -> list[str]:
        have = [set(s.keys()) for s in self.sources[1:]]
        return [k for k in self.sources[0].keys() if all(k in h for h in have)]

    def get(self, key: str) -> torch.Tensor:
        ts = [s.get(key) for s in self.sources]
        return (sum(t.float() for t in ts) / len(ts)).to(ts[0].dtype)


def convert(base: Source, groups: dict[str, Sequence[Source]], dst: Path, zeros: Sequence[int], top_k: int = 1,
            group_top_k: int | None = None, merge: str = "simple_average",
            merge_args: dict[str, Any] | None = None, shard_gb: float = 5.0,
            log: Callable[[str], None] = print, router_std: float = 0.0,
            routed: dict[str, Any] | None = None, grid: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    r"""Write the converted tensors to ``dst`` (config and tokenizer files are ``main``'s job, ``copy_config``).

    Args:
        base: the dense base checkpoint; its decoder MLP is :math:`W_b`.
        groups: domain name -> its fine-tunes, one expert each, in order. The same source object may sit in
            several groups: one expert in each, but one fine-tune in the merge of the other tensors.
        dst: output directory (created if missing).
        zeros: zero experts per group.
        top_k: experts per group.
        group_top_k: groups per token; None = dense group routing.
        merge: a ``merging.MERGE`` method for every non-MLP tensor.
        merge_args: keyword arguments of the merge method.
        shard_gb: output shard size in GiB.
        log: progress messages.
        router_std: > 0 draws the routers from :math:`\mathcal{N}(0, \sigma^2)`, :math:`\sigma` = ``router_std``
            (seeded per layer) instead of zero, for tests.
        routed: The ``routed_deltas`` config entry (``levels``: None = dense, ``group``, ``symmetric``, ``sets``:
            consecutive expert counts sharing one delta, their mean); None = merge the attention and ``lm_head`` too.
        grid: ``{levels, group, symmetric}``: the MLP deltas as group grids, round to nearest; None = dense.

    Returns:
        The ``hier_delta_moe`` config entry.
    """
    names = list(groups)
    tuned = [s for n in names for s in groups[n]]
    uniq = list(dict.fromkeys(tuned))  # a fine-tune in several groups (the same object) is merged once
    assert len(zeros) == len(names) and all(groups[n] for n in names)
    widths = [len(groups[n]) + z for n, z in zip(names, zeros)]
    assert 1 <= top_k <= max(widths), f"top_k {top_k}: groups have {widths} experts + zero experts"
    assert group_top_k is None or 1 <= group_top_k <= len(names), f"group_top_k {group_top_k}: {len(names)} groups"
    merge_fn, kw = MERGE[merge], merge_args or {}
    keys = base.keys()
    assert any(MLP.fullmatch(k) for k in keys), "no decoder MLP found"
    have = [set(s.keys()) for s in uniq]
    for k in keys:
        if MLP.fullmatch(k):
            assert all(k in h for h in have), f"{k}: missing in a fine-tune"
    # a fine-tune saved by transformers lacks what it ignores on load (Qwen3.8's mtp.*): merged over the ones that
    # have it, or kept from the base when none does
    base_only = [k for k in keys if not any(k in h for h in have)]
    if base_only:
        log(f"{len(base_only)} tensors in no fine-tune, kept from the base: {base_only[:3]}...")
    dst.mkdir(parents=True, exist_ok=True)
    w = ShardWriter(dst, int(shard_gb * 2**30))
    done: set[str] = set()
    rows = len(names) + sum(len(groups[n]) + z for n, z in zip(names, zeros))

    def router(key: str, cols: int, dtype: torch.dtype) -> None:
        rt = torch.zeros(rows, cols)
        if router_std:
            rt.normal_(std=router_std, generator=torch.Generator().manual_seed(len(done)))
        w.add(key, rt.to(dtype))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if routed and routed.get("sets"):
        assert sum(routed["sets"]) == len(tuned), f"routed sets {routed['sets']} for {len(tuned)} experts"

    def put(codes: str, steps: str, lows: str, d: torch.Tensor, c: dict[str, Any], dtype: torch.dtype) -> None:
        """Deltas d (N, R, C) as group grids, round to nearest, no data (``models/grid.py`` layout)."""
        L, g, sym = c["levels"], c.get("group", 128), c.get("symmetric", False)
        qs = [C.quantize(de.to(dev), None, L, g, sym) for de in d]
        w.add(codes, torch.stack([pack_codes(q, code_bits(L)) for q, _, _ in qs]).cpu())
        w.add(steps, torch.stack([st for _, _, st in qs]).to(dtype).cpu())
        if not sym:
            w.add(lows, torch.stack([lo for _, lo, _ in qs]).to(dtype).cpu())
    hidden = base.get(next(k for k in keys if MLP.fullmatch(k) and "gate" in k)).shape[1]  # every router's input
    for i, k in enumerate(sorted(keys)):
        m = MLP.fullmatch(k)
        r = ROUTED.fullmatch(k) if routed is not None else None
        if r is not None:
            b = base.get(k)
            w.add(k, b)
            d = torch.stack([s.get(k).float() - b.float() for s in tuned])
            if routed.get("sets"):
                d = torch.stack([c.mean(0) for c in d.split(routed["sets"])])
            stem = k[: -len("weight")]
            if routed.get("levels") is None:
                w.add(stem + "deltas", d.to(b.dtype))
            else:
                put(stem + "codes", stem + "steps", stem + "lows", d, routed, b.dtype)
            block = r[1] or r[2]
            if block not in done:
                done.add(block)
                router(f"{block}.delta_router.weight", hidden, b.dtype)  # routes the block input, width H
            continue
        if m is None:
            b = base.get(k)
            ts = [s.get(k) for s, h in zip(uniq, have) if k in h]
            w.add(k, merge_fn({k: b}, [{k: t} for t in ts], **kw)[k] if b.is_floating_point() and ts else b)
            continue
        prefix = m[1]
        if prefix in done:
            continue
        done.add(prefix)
        for p in ("gate", "up", "down"):
            b = base.get(f"{prefix}.{p}_proj.weight")
            w.add(f"{prefix}.{p}", b)
            d = torch.stack([s.get(f"{prefix}.{p}_proj.weight").float() - b.float() for s in tuned])
            if grid is None:
                w.add(f"{prefix}.d_{p}", d.to(b.dtype))
            else:
                put(f"{prefix}.c_{p}", f"{prefix}.s_{p}", f"{prefix}.z_{p}", d, grid, b.dtype)
            if p == "gate":
                router(f"{prefix}.router.weight", b.shape[1], b.dtype)
        log(f"{prefix}: {len(tuned)} experts in {len(names)} groups")
    w.close()
    return {"groups": names, "group_sizes": [len(groups[n]) for n in names], "group_zeros": list(zeros),
            "top_k": top_k, "group_top_k": group_top_k,
            "merge": merge, "merge_args": kw, **(grid or {})}  # fmt: skip


def _value(v: str) -> Any:
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--group", action="append", required=True, help="NAME=DIR[,DIR...]: one domain's fine-tunes, one expert each; DIR+DIR+... is one expert, their mean")
    ap.add_argument("--zeros", default="0", help="zero experts per group: one int, or one per group (a,b,...)")
    ap.add_argument("--top-k", type=int, default=1, help="experts per group")
    ap.add_argument("--group-top-k", type=int, help="groups per token (default: dense over every group)")
    ap.add_argument("--merge", default="simple_average", choices=sorted(MERGE))
    ap.add_argument("--merge-arg", action="append", default=[], help="KEY=VALUE for the merge method")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    ap.add_argument("--routed-deltas", action="store_true", help="per-expert attention / lm_head deltas, routed per token")
    ap.add_argument("--routed-levels", type=int, help="grid values of those deltas (default: dense)")
    ap.add_argument("--routed-group", type=int, default=128)
    ap.add_argument("--routed-symmetric", action="store_true")
    ap.add_argument("--routed-sets", help="a,b,...: consecutive experts sharing one routed delta (their mean)")
    ap.add_argument("--levels", type=int, help="MLP deltas as group grids of this many values (default: dense)")
    ap.add_argument("--grid-group", type=int, default=128)
    ap.add_argument("--symmetric", action="store_true")
    a = ap.parse_args(argv)
    if not a.routed_deltas and (a.routed_levels or a.routed_symmetric or a.routed_sets):
        ap.error("--routed-levels / --routed-symmetric / --routed-sets need --routed-deltas")
    if a.symmetric and not a.levels:
        ap.error("--symmetric needs --levels")
    groups, ckpt = {}, functools.cache(lambda d: Checkpoint(Path(d).resolve()))  # one source per directory
    for g in a.group:
        name, dirs = g.split("=", 1)
        experts = [ckpt(d) if "+" not in d else Mean([ckpt(p) for p in d.split("+")]) for d in dirs.split(",")]
        groups.setdefault(name, []).extend(experts)  # a repeated name extends
    zeros = [int(z) for z in a.zeros.split(",")]
    zeros = zeros * len(groups) if len(zeros) == 1 else zeros
    merge_args = {k: _value(v) for k, v in (s.split("=", 1) for s in a.merge_arg)}
    a.dst.mkdir(parents=True)
    routed = None if not a.routed_deltas else {"levels": a.routed_levels, "group": a.routed_group,
                                                "symmetric": a.routed_symmetric}  # fmt: skip
    if routed and a.routed_sets:
        routed["sets"] = [int(v) for v in a.routed_sets.split(",")]
    grid = None if a.levels is None else {"levels": a.levels, "group": a.grid_group, "symmetric": a.symmetric}
    entry = convert(Checkpoint(a.base), groups, a.dst, zeros, a.top_k, a.group_top_k, a.merge,
                    merge_args, a.shard_gb, routed=routed, grid=grid)  # fmt: skip
    copy_config(a.base, a.dst, lambda text: text.update(hier_delta_moe=entry, **({"routed_deltas": routed} if routed else {})))
    print(f"{sum(entry['group_sizes'])} experts in groups {dict(zip(entry['groups'], entry['group_sizes']))}, "
          f"zeros {zeros}, top-{a.top_k} per group, groups {'dense' if a.group_top_k is None else a.group_top_k}, "
          f"merge {a.merge} {merge_args}")  # fmt: skip


if __name__ == "__main__":
    main()
