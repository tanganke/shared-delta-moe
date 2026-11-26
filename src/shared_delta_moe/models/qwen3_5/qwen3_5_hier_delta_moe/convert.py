r"""Upscale a dense Qwen3.5 / Qwen3.8 base and its domain fine-tunes into the HierDeltaMoE layout of ``modeling.py``.

One group per domain; each fine-tune's decoder MLP becomes an expert :math:`\theta_{ij}` of its group, stored as
:math:`W_b` (the base MLP) and :math:`\Delta_{ij} = \theta_{ij} - W_b` (subtracted in float32, stored in the checkpoint
dtype). Every other tensor (attention, linear attention, norms, embeddings, ``lm_head``, vision tower, MTP) is merged
from the fine-tunes with a ``merging.MERGE`` method (``fusion_bench``, extra ``merging``; default ``simple_average``). Routers start at zero: uniform over
groups and within each group. With one expert per group, no zero experts and dense group routing, each layer's MLP is
then the plain average of the fine-tunes' MLPs; a zero expert is the base MLP.

Usage::

    python -m shared_delta_moe.models.qwen3_5.qwen3_5_hier_delta_moe.convert BASE DST \
        --group medical=FT1,FT2 --group code=FT3 [--zeros 0] [--top-k 1] [--group-top-k K] \
        [--merge ties_merging --merge-arg scaling_factor=0.3 --merge-arg threshold=20]

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

from ....merging import MERGE
from ..common import Checkpoint, ShardWriter, copy_config

MLP = re.compile(r"(model\.(?:language_model\.)?layers\.\d+\.mlp)\.(gate|up|down)_proj\.weight")


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
            log: Callable[[str], None] = print, router_std: float = 0.0) -> dict[str, Any]:  # fmt: skip
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

    Returns:
        The ``hier_delta_moe`` config entry.
    """
    names = list(groups)
    tuned = [s for n in names for s in groups[n]]
    uniq = list(dict.fromkeys(tuned))  # a fine-tune in several groups (the same object) is merged once
    assert len(zeros) == len(names) and all(groups[n] for n in names)
    widths = [len(groups[n]) + z for n, z in zip(names, zeros)]
    assert 1 <= top_k <= min(widths), f"top_k {top_k}: groups have {widths} experts + zero experts"
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
    for i, k in enumerate(sorted(keys)):
        m = MLP.fullmatch(k)
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
            w.add(f"{prefix}.d_{p}", d.to(b.dtype))
            if p == "gate":
                rows = len(names) + sum(len(groups[n]) + z for n, z in zip(names, zeros))
                rt = torch.zeros(rows, b.shape[1])
                if router_std:
                    rt.normal_(std=router_std, generator=torch.Generator().manual_seed(len(done)))
                w.add(f"{prefix}.router.weight", rt.to(b.dtype))
        log(f"{prefix}: {len(tuned)} experts in {len(names)} groups")
    w.close()
    return {"groups": names, "group_sizes": [len(groups[n]) for n in names], "group_zeros": list(zeros),
            "top_k": top_k, "group_top_k": group_top_k,
            "merge": merge, "merge_args": kw}  # fmt: skip


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
    a = ap.parse_args(argv)
    groups, ckpt = {}, functools.cache(lambda d: Checkpoint(Path(d).resolve()))  # one source per directory
    for g in a.group:
        name, dirs = g.split("=", 1)
        experts = [ckpt(d) if "+" not in d else Mean([ckpt(p) for p in d.split("+")]) for d in dirs.split(",")]
        groups.setdefault(name, []).extend(experts)  # a repeated name extends
    zeros = [int(z) for z in a.zeros.split(",")]
    zeros = zeros * len(groups) if len(zeros) == 1 else zeros
    merge_args = {k: _value(v) for k, v in (s.split("=", 1) for s in a.merge_arg)}
    a.dst.mkdir(parents=True)
    entry = convert(Checkpoint(a.base), groups, a.dst, zeros, a.top_k, a.group_top_k, a.merge,
                    merge_args, a.shard_gb)  # fmt: skip
    copy_config(a.base, a.dst, lambda text: text.update(hier_delta_moe=entry))
    print(f"{sum(entry['group_sizes'])} experts in groups {dict(zip(entry['groups'], entry['group_sizes']))}, "
          f"zeros {zeros}, top-{a.top_k} per group, groups {'dense' if a.group_top_k is None else a.group_top_k}, "
          f"merge {a.merge} {merge_args}")  # fmt: skip


if __name__ == "__main__":
    main()
