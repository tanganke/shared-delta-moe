r"""Plain Qwen3.5 MoE checkpoint -> the DeltaMoE layout of ``modeling.py``: expert BASE becomes the shared
:math:`W_b` and the one zero expert, every other expert :math:`e` becomes :math:`\Delta_e = W_e - W_b` (subtracted in
the stored dtype), router rows reordered to [other experts in order, BASE]. The plain MoE's shared expert must have no
output (``down_proj`` all zero); it is dropped.

::

    python -m shared_delta_moe.models.qwen3_5.qwen3_5_delta_moe.convert SRC DST [--base INDEX] [--top-k K] [--shard-gb 5]

BASE defaults to the expert named ``"base"`` in ``SRC/upscale_info.json``. ``--top-k`` sets ``num_experts_per_tok``
(default: SRC's). Tensors are read one MLP at a time and written in shards of ``--shard-gb``.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from ..common import Checkpoint, ShardWriter, copy_config

SPLIT = re.compile(r"(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
FUSED = re.compile(r"(.+\.mlp)\.experts\.(gate_up_proj|down_proj)")
LAYER = re.compile(r"(.+\.mlp)\.")


def convert_state(state: dict[str, torch.Tensor], base: int) -> dict[str, torch.Tensor]:
    """Works on both checkpoint forms: one tensor per expert and projection, or the fused ``gate_up_proj`` /
    ``down_proj``."""
    # layer prefix -> projection -> fused tensor, or {expert index: tensor}
    experts: dict[str, dict[str, Any]] = {}
    for k in list(state):
        if m := SPLIT.fullmatch(k):
            experts.setdefault(m[1], {}).setdefault(m[3], {})[int(m[2])] = state.pop(k)
        elif m := FUSED.fullmatch(k):
            experts.setdefault(m[1], {})[m[2]] = state.pop(k)
    for prefix, e in experts.items():
        if "gate_up_proj" in e:
            gate, up = e.pop("gate_up_proj").chunk(2, dim=1)
            w = {"gate": gate, "up": up, "down": e["down_proj"]}
        else:
            w = {p: torch.stack([v[i] for i in sorted(v)]) for p, v in e.items()}
            w = {"gate": w["gate_proj"], "up": w["up_proj"], "down": w["down_proj"]}
        others = [i for i in range(w["gate"].shape[0]) if i != base]
        for p, t in w.items():
            state[f"{prefix}.{p}"] = t[base].contiguous()
            state[f"{prefix}.d_{p}"] = (t[others] - t[base]).contiguous()
        state[f"{prefix}.router.weight"] = state.pop(f"{prefix}.gate.weight")[
            others + [base]
        ].contiguous()
        shared = state.pop(f"{prefix}.shared_expert.down_proj.weight")
        assert (
            not shared.any()
        ), f"{prefix}: the shared expert has an output; this conversion drops it"
        for k in [k for k in state if k.startswith(f"{prefix}.shared_expert")]:
            del state[k]
    return state


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--base", type=int)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--shard-gb", type=float, default=5.0)
    a = ap.parse_args(argv)
    base = a.base
    if base is None:
        base = json.loads((a.src / "upscale_info.json").read_text())["expert_domains"].index("base")
    a.dst.mkdir(parents=True)
    text: dict[str, Any] = {}

    def edit(t: dict[str, Any]) -> None:
        t["delta_moe_zero_experts"] = 1
        t["num_experts_per_tok"] = a.top_k or t["num_experts_per_tok"]
        text.update(t)

    copy_config(a.src, a.dst, edit)
    # one MLP at a time: memory stays at one layer's experts
    src, w, layers = Checkpoint(a.src), ShardWriter(a.dst, int(a.shard_gb * 2**30)), {}
    for k in src.keys():
        if m := LAYER.match(k):
            layers.setdefault(m[1], []).append(k)
        else:
            w.add(k, src.get(k))
    for keys in layers.values():
        for k, t in convert_state({k: src.get(k) for k in keys}, base).items():
            w.add(k, t)
    w.close()
    print(f"base expert {base}; {text['num_experts'] - 1} deltas + 1 zero expert, top-{text['num_experts_per_tok']}")


if __name__ == "__main__":
    main()
