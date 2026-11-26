r"""How similar dense fine-tunes of one base are, before upscaling them into a ``DeltaMoE`` / ``HierDeltaMoE``
(guide: ``docs/guide/similarity.md``).

Three steps, each a subcommand of ``python -m shared_delta_moe.similarity``:

``gram``
    For every language-model tensor :math:`W` of the base (vision tower and ``mtp.*`` skipped) and fine-tunes
    :math:`\theta_1..\theta_n`, :math:`\Delta_i = \theta_i - W`: the Gram matrix of :math:`[W, \Delta_1, ..., \Delta_n]`
    (fp64 sums, rows streamed in chunks), the number of entries a delta left unchanged in the stored dtype and the
    number it moved by one bf16 ulp, and for MLP projections the delta energy per intermediate channel. Every
    weight-space statistic below is a function of these. ``--shard k/K`` splits the tensors over processes.
``report``
    From the merged Gram files: relative size :math:`\|\Delta_i\| / \|W\|` and pairwise cosine per tensor kind and per
    layer, the shared part of a set of deltas, and least-squares fits of one delta by others (does training on
    :math:`A \cup B` give :math:`\Delta_A + \Delta_B`?).
``loss``
    LM loss on raw-text dev sets of dense models :math:`W + \sum_i c_i \Delta_i`, with separate coefficients for the
    MLP and for every other tensor: single fine-tunes, averages, task arithmetic, and one expert of the upscaled MoE
    under forced routing (its MLP, merged shared weights).
"""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from shared_delta_moe.models.qwen3_5.common import Checkpoint

CHUNK = 1 << 25
MLP = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def tensor_files(path: str | Path) -> dict[str, tuple[str, str]]:
    """Language-model tensors of a checkpoint: name after ``language_model.`` -> (file, stored key)."""
    return {
        k.split("language_model.", 1)[-1]: (str(f), k)
        for k, f in Checkpoint(path).files.items()
        if "visual" not in k and not k.startswith("mtp.") and ".mtp." not in k
    }


def bf16_ulp(w: Tensor) -> Tensor:
    """One bf16 ulp at :math:`|w|` (8 significant bits): :math:`2^{e-8}` for :math:`w = m 2^e`, :math:`m \\in [0.5, 1)`."""
    _, e = torch.frexp(w)
    return torch.ldexp(torch.ones_like(w), e - 8)


class _Open:
    def __init__(self, device: str) -> None:
        self.device, self.h = device, {}

    def __call__(self, f: str) -> Any:
        if f not in self.h:
            self.h[f] = safe_open(f, "pt", device=self.device)
        return self.h[f]


def delta_gram(base: str | Path, fts: Mapping[str, str | Path], shard: int = 0, shards: int = 1,
               device: str = "cuda") -> dict[str, Any]:  # fmt: skip
    """Per-tensor Gram statistics of the deltas of ``fts`` against ``base`` (see the module docstring).

    Args:
        base: the base checkpoint directory.
        fts: fine-tune name -> checkpoint directory.
        shard, shards: only tensors whose index (in sorted name order) is ``shard`` mod ``shards``.
        device: where the arithmetic runs.

    Returns:
        ``{"names": [...], "tensors": {name: {"gram", "numel", "shape", "zero", "ulp1"[, "neuron"]}}}``; ``gram`` is
        (n+1, n+1) with row 0 the base tensor itself.
    """
    names = list(fts)
    b = tensor_files(base)
    f = [tensor_files(p) for p in fts.values()]
    common = sorted(set(b).intersection(*f))
    op = _Open(device)
    n, out = len(names), {}
    for t, name in enumerate(common):
        if t % shards != shard:
            continue
        shape = op(b[name][0]).get_slice(b[name][1]).get_shape()
        step = max(1, CHUNK // max(1, int(torch.tensor(shape[1:]).prod().item()) if len(shape) > 1 else 1))
        gram = torch.zeros(n + 1, n + 1, dtype=torch.float64, device=device)
        zero = torch.zeros(n, dtype=torch.float64, device=device)
        ulp1 = torch.zeros(n, dtype=torch.float64, device=device)
        neuron = None
        for r0 in range(0, shape[0], step):
            w = op(b[name][0]).get_slice(b[name][1])[r0 : r0 + step].float()
            d = torch.stack([op(x[name][0]).get_slice(x[name][1])[r0 : r0 + step].float() - w for x in f])
            m = torch.cat([w[None], d]).reshape(n + 1, -1)
            gram += (m @ m.T).double()
            ad = d.reshape(n, -1).abs()
            zero += (ad == 0).sum(1).double()
            ulp1 += ((ad > 0) & (ad <= bf16_ulp(w).reshape(1, -1) * 1.001)).sum(1).double()
            if name.endswith(MLP):
                e = d.square()
                if name.endswith("down_proj.weight"):
                    neuron = e.sum(1) if neuron is None else neuron + e.sum(1)
                else:
                    neuron = e.sum(2) if neuron is None else torch.cat([neuron, e.sum(2)], 1)
        rec = {"gram": gram.cpu(), "numel": int(torch.tensor(shape).prod().item()), "shape": list(shape),
               "zero": zero.cpu(), "ulp1": ulp1.cpu()}  # fmt: skip
        if neuron is not None:
            rec["neuron"] = neuron.float().cpu()
        out[name] = rec
    return {"names": names, "base": str(base), "fts": {k: str(v) for k, v in fts.items()}, "tensors": out}


def load_grams(files: Sequence[str | Path]) -> dict[str, Any]:
    """Merge the outputs of ``gram`` shards (same fine-tunes, disjoint tensors)."""
    res = None
    for p in files:
        d = torch.load(p, weights_only=False)
        if res is None:
            res = d
        else:
            assert d["names"] == res["names"], (p, d["names"])
            res["tensors"].update(d["tensors"])
    assert res is not None
    return res


def kind(name: str) -> str:
    """Tensor kind without the layer index: ``layers.3.mlp.up_proj.weight`` -> ``mlp.up_proj``."""
    parts = name.split(".")
    if parts[0] == "layers":
        parts = parts[2:]
    return ".".join(p for p in parts if p != "weight")


def group(name: str) -> str:
    """Coarse group: mlp, attn (full attention), linear_attn, embed (embeddings and lm_head), norm."""
    if ".mlp." in name:
        return "mlp"
    if "self_attn" in name and "norm" not in name:
        return "attn"
    if "linear_attn" in name and "norm" not in name:
        return "linear_attn"
    if "embed" in name or "lm_head" in name:
        return "embed"
    return "norm"


def layer(name: str) -> int | None:
    parts = name.split(".")
    return int(parts[1]) if parts[0] == "layers" else None


def aggregate(tensors: Mapping[str, Any], select: Any = None) -> dict[str, Tensor]:
    """Sum of the per-tensor statistics over the tensors ``select(name)`` is true for (all if None)."""
    tot: dict[str, Any] = {}
    for name, r in tensors.items():
        if select is not None and not select(name):
            continue
        for k in ("gram", "numel", "zero", "ulp1"):
            tot[k] = tot.get(k, 0) + r[k]
    return tot


def cosine(gram: Tensor) -> Tensor:
    """Cosine matrix of the deltas (row / column 0 of ``gram``, the base, dropped)."""
    g = gram[1:, 1:]
    d = g.diagonal().clamp_min(1e-300).sqrt()
    return g / d[:, None] / d[None, :]


def relative_norm(gram: Tensor) -> Tensor:
    r""":math:`\|\Delta_i\| / \|W\|` from a Gram matrix."""
    return (gram.diagonal()[1:] / gram[0, 0]).sqrt()


def fit(gram: Tensor, target: int, sources: Sequence[int]) -> tuple[list[float], float]:
    r"""Least-squares fit :math:`\Delta_t \approx \sum_j \beta_j \Delta_{s_j}` from the Gram matrix (indices count the
    fine-tunes from 0). Returns :math:`\beta` and the unexplained fraction of :math:`\|\Delta_t\|^2`."""
    g = gram[1:, 1:]
    s = list(sources)
    beta = torch.linalg.solve(g[s][:, s], g[s, target])
    return beta.tolist(), float(1 - g[s, target] @ beta / g[target, target])


def shared_fraction(gram: Tensor, members: Sequence[int]) -> tuple[float, list[float]]:
    r"""Shared part of a set of deltas: top eigenvalue of their cosine matrix over its trace (1/n: orthogonal, 1:
    parallel), and each delta's energy fraction along the unit-norm mean direction :math:`\bar u` of the
    normalized deltas, :math:`\cos^2(\Delta_i, \bar u)`."""
    c = cosine(gram)[list(members)][:, list(members)]
    top = float(torch.linalg.eigvalsh(c)[-1] / c.trace())
    s = c.sum(1)
    return top, (s.square() / c.sum()).tolist()


def neuron_overlap(tensors: Mapping[str, Any], i: int, j: int, top: float = 0.1) -> dict[str, float]:
    """Do two deltas concentrate on the same MLP channels? Mean over layers of the Pearson correlation of their
    per-channel energies (gate + up + down) and the overlap of their top ``top`` fraction of channels (chance =
    ``top``)."""
    by_layer: dict[int, Tensor] = {}
    for name, r in tensors.items():
        if "neuron" in r:
            L = layer(name)
            by_layer[L] = by_layer.get(L, 0) + r["neuron"]
    corr, ov = [], []
    for e in by_layer.values():
        a, b = e[i].double(), e[j].double()
        corr.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
        k = max(1, int(top * a.numel()))
        ta, tb = set(a.topk(k).indices.tolist()), set(b.topk(k).indices.tolist())
        ov.append(len(ta & tb) / k)
    return {"corr": sum(corr) / len(corr), "top_overlap": sum(ov) / len(ov)}


def report(g: Mapping[str, Any], fits: Sequence[str] = (), sets: Sequence[str] = (), pairs: Sequence[str] = ()) -> dict:
    """The summary ``report`` prints, as a dict (see the module docstring).

    Args:
        g: merged Gram statistics (``load_grams``).
        fits: ``target=src1,src2`` least-squares fits.
        sets: ``label=a,b,c`` sets for ``shared_fraction``.
        pairs: ``a,b`` pairs for ``neuron_overlap``.
    """
    names, T = g["names"], g["tensors"]
    ix = {n: i for i, n in enumerate(names)}
    out: dict[str, Any] = {"names": names, "groups": {}, "kinds": {}, "layers": {}, "fits": {}, "sets": {}, "neurons": {}}
    for grp in ("all", "mlp", "attn", "linear_attn", "embed", "norm"):
        a = aggregate(T, None if grp == "all" else (lambda n, grp=grp: group(n) == grp))
        if not a:
            continue
        gr = a["gram"]
        out["groups"][grp] = {"rel": relative_norm(gr).tolist(), "cos": cosine(gr).tolist(),
                              "cos_w": (gr[0, 1:] / gr[0, 0].sqrt() / gr.diagonal()[1:].sqrt()).tolist(),
                              "zero": (a["zero"] / a["numel"]).tolist(), "ulp1": (a["ulp1"] / a["numel"]).tolist()}  # fmt: skip
    for k in sorted({kind(n) for n in T}):
        a = aggregate(T, lambda n, k=k: kind(n) == k)
        out["kinds"][k] = {"rel": relative_norm(a["gram"]).tolist(), "zero": (a["zero"] / a["numel"]).tolist(),
                           "cos": cosine(a["gram"]).tolist()}  # fmt: skip
    for L in sorted({layer(n) for n in T if layer(n) is not None}):
        a = aggregate(T, lambda n, L=L: layer(n) == L and group(n) == "mlp")
        out["layers"][L] = {"rel": relative_norm(a["gram"]).tolist(), "cos": cosine(a["gram"]).tolist()}
    mlp = aggregate(T, lambda n: group(n) == "mlp")["gram"]
    for spec in fits:
        t, src = spec.split("=")
        beta, resid = fit(mlp, ix[t], [ix[s] for s in src.split(",")])
        out["fits"][spec] = {"beta": beta, "unexplained": resid}
    for spec in sets:
        lab, mem = spec.split("=")
        top, frac = shared_fraction(mlp, [ix[m] for m in mem.split(",")])
        out["sets"][lab] = {"members": mem.split(","), "top_eig": top, "mean_dir_energy": frac}
    for spec in pairs:
        a, b = spec.split(",")
        out["neurons"][spec] = neuron_overlap(T, ix[a], ix[b])
    return out


def _table(names: Sequence[str], m: Sequence[Sequence[float]], fmt: str = "{:.2f}") -> str:
    short = [n[:10] for n in names]
    lines = ["| | " + " | ".join(short) + " |", "|---" * (len(names) + 1) + "|"]
    lines += [f"| {short[i]} | " + " | ".join(fmt.format(v) for v in row) + " |" for i, row in enumerate(m)]
    return "\n".join(lines)


def print_report(r: Mapping[str, Any]) -> None:
    names = r["names"]
    print("## relative size %, unchanged %, one-ulp %, cos(delta, W)\n")
    print("| group | " + " | ".join(n[:10] for n in names) + " |\n" + "|---" * (len(names) + 1) + "|")
    for grp, v in r["groups"].items():
        print(f"| {grp} rel | " + " | ".join(f"{100 * x:.3f}" for x in v["rel"]) + " |")
    v = r["groups"]["all"]
    for key in ("zero", "ulp1"):
        print(f"| all {key} | " + " | ".join(f"{100 * x:.1f}" for x in v[key]) + " |")
    print("| all cos_w | " + " | ".join(f"{x:.3f}" for x in v["cos_w"]) + " |")
    for grp in ("mlp", "attn", "linear_attn", "embed"):
        if grp in r["groups"]:
            print(f"\n## cosine, {grp}\n\n" + _table(names, r["groups"][grp]["cos"]))
    print("\n## relative size % by tensor kind\n")
    print("| kind | " + " | ".join(n[:10] for n in names) + " |\n" + "|---" * (len(names) + 1) + "|")
    for k, v in r["kinds"].items():
        print(f"| {k} | " + " | ".join(f"{100 * x:.3f}" for x in v["rel"]) + " |")
    for title, key in (("fits (MLP)", "fits"), ("shared part (MLP)", "sets"), ("channel overlap", "neurons")):
        if r[key]:
            print(f"\n## {title}\n")
            for k, v in r[key].items():
                print(f"- {k}: {json.dumps(v)}")


def parse_spec(spec: str) -> tuple[str, dict[str, dict[str, float]]]:
    """``label=TERMS`` or ``label=mlp:TERMS;rest:TERMS``; TERMS is ``base`` or ``A*c+B*c...`` (``A`` means ``A*1``); a
    bare ``A`` is ``A=A``. Returns the label and ``{"mlp": {name: c}, "rest": {name: c}}``."""
    label, _, expr = spec.partition("=")
    expr = expr or label
    parts = dict(p.split(":", 1) for p in expr.split(";")) if ":" in expr else {"mlp": expr, "rest": expr}
    out = {}
    for grp in ("mlp", "rest"):
        terms: dict[str, float] = {}
        for t in parts[grp].split("+"):
            if t == "base":
                continue
            n, _, c = t.partition("*")
            terms[n] = terms.get(n, 0.0) + (float(c) if c else 1.0)
        out[grp] = terms
    return label, out


def _ids(x: Any) -> list[int]:
    return list(x["input_ids"] if isinstance(x, Mapping) or hasattr(x, "keys") else x)


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def read_docs(tok: Any, path: str | Path, docs: int, max_length: int, min_tokens: int = 256,
              score: str = "reply") -> list[tuple[list[int], int, int]]:  # fmt: skip
    """The first ``docs`` rows of a jsonl as ``(token ids, first scored position, end of scored span)``.

    A ``text`` row (raw text) needs at least ``min_tokens`` tokens, is cut to its first ``max_length`` and is scored
    on every token. A ``messages`` row (chat) is rendered with the tokenizer's chat template and scored on its last
    turn, by ``score``:

    - ``reply``: everything after the longest common prefix of the conversation and its prompt (all but the last
      turn plus the generation prompt), template tokens of the reply included;
    - ``content``: only the reply's own text, located as the part that differs from the same conversation with an
      empty reply;
    - ``last``: only the last token of that text (the option letter of an answer-only multiple-choice reply).

    Chat rows longer than ``max_length`` keep their last ``max_length`` tokens; rows whose scored span would be cut
    are dropped.
    """
    out = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "messages" in row:
                msgs = row["messages"]
                full = _ids(tok.apply_chat_template(msgs, tokenize=True))
                if score == "reply":
                    prompt = _ids(tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True))
                    start, end = _common_prefix(full, prompt), len(full)
                else:
                    empty = _ids(tok.apply_chat_template(msgs[:-1] + [{**msgs[-1], "content": ""}], tokenize=True))
                    start = _common_prefix(full, empty)
                    end = len(full) - _common_prefix(full[start:][::-1], empty[start:][::-1])
                    if score == "last":
                        start = end - 1
                cut = max(0, len(full) - max_length)
                if start - cut < 1 or end <= start:
                    continue
                out.append((full[cut:], start - cut, end - cut))
            else:
                ids = tok(row["text"], add_special_tokens=False).input_ids
                if len(ids) < min_tokens:
                    continue
                ids = ids[:max_length]
                out.append((ids, 1, len(ids)))
            if len(out) >= docs:
                break
    return out


def delta_loss(base: str | Path, fts: Mapping[str, str | Path], data: Mapping[str, str | Path], specs: Sequence[str],
               out: str | Path | None = None, docs: int = 128, max_length: int = 2048, batch: int = 8,
               score: str = "reply") -> dict:  # fmt: skip
    """Per-document mean NLL of each spec's dense model on each dev set (see ``parse_spec``). The base model is
    loaded once (transformers' Qwen3.5 class); each spec overwrites its language-model parameters. Results are
    written to ``out`` after every spec."""
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    files = {n: tensor_files(d) for n, d in fts.items()}
    parsed = [parse_spec(s) for s in specs]
    for label, g in parsed:
        unknown = (set(g["mlp"]) | set(g["rest"])) - set(files)
        assert not unknown, (label, unknown)
    tok = AutoTokenizer.from_pretrained(base)
    sets = {n: read_docs(tok, p, docs, max_length, score=score) for n, p in data.items()}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda").eval()
    params = {k: p for k, p in model.named_parameters() if "visual" not in k and not k.startswith("mtp.")}
    w0 = {k: p.detach().clone() for k, p in params.items()}
    op = _Open("cuda")

    @torch.no_grad()
    def build(g: Mapping[str, Mapping[str, float]]) -> None:
        for k, p in params.items():
            w = w0[k].float()
            v = w.clone()
            for n, c in g["mlp" if ".mlp." in k else "rest"].items():
                f, key = files[n][k.split("language_model.", 1)[-1]]
                v += c * (op(f).get_tensor(key).float() - w)
            p.copy_(v.to(p.dtype))

    @torch.no_grad()
    def nll(rows_all: list[tuple[list[int], int, int]]) -> list[float]:
        res = []
        for i in range(0, len(rows_all), batch):
            rows = rows_all[i : i + batch]
            width = max(len(r) for r, _, _ in rows)
            ids = torch.tensor([r + [tok.pad_token_id] * (width - len(r)) for r, _, _ in rows], device="cuda")
            mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r, _, _ in rows], device="cuda")
            hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            pos = torch.arange(width, device="cuda")
            scored = torch.stack([(pos >= s) & (pos < e) for _, s, e in rows])
            b, t = scored[:, 1:].nonzero(as_tuple=True)
            logits = model.lm_head(hidden[b, t]).float()
            x = torch.nn.functional.cross_entropy(logits, ids[b, t + 1], reduction="none")
            tot = torch.zeros(len(rows), device="cuda").index_add_(0, b, x)
            res += (tot / scored[:, 1:].sum(1)).tolist()
        return res

    res: dict[str, Any] = {"base": str(base), "fts": {k: str(v) for k, v in fts.items()}, "data": {k: str(v) for k, v in data.items()},
                           "docs": docs, "max_length": max_length, "score": score, "results": {}}  # fmt: skip
    for label, g in parsed:
        t = time.time()
        build(g)
        r = {n: nll(d) for n, d in sets.items()}
        res["results"][label] = {"spec": g, "doc_loss": r, "loss": {n: sum(v) / len(v) for n, v in r.items()}}
        print(label, f"{time.time() - t:.0f}s", json.dumps({n: round(v, 4) for n, v in res["results"][label]["loss"].items()}), flush=True)  # fmt: skip
        if out:
            Path(out).write_text(json.dumps(res))
    return res


def _pairs(xs: Sequence[str]) -> dict[str, str]:
    return dict(x.split("=", 1) for x in xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("gram")
    s.add_argument("--base", required=True)
    s.add_argument("--ft", nargs="+", required=True, help="name=DIR")
    s.add_argument("--out", required=True)
    s.add_argument("--shard", default="0/1", help="k/K")
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r = sub.add_parser("report")
    r.add_argument("grams", nargs="+")
    r.add_argument("--fit", nargs="*", default=[], help="target=src1,src2")
    r.add_argument("--set", nargs="*", default=[], help="label=a,b,c")
    r.add_argument("--pair", nargs="*", default=[], help="a,b")
    r.add_argument("--json")
    lo = sub.add_parser("loss")
    lo.add_argument("--base", required=True)
    lo.add_argument("--ft", nargs="*", default=[], help="name=DIR")
    lo.add_argument("--data", nargs="+", required=True, help="name=FILE.jsonl")
    lo.add_argument("--spec", nargs="+", required=True)
    lo.add_argument("--out", required=True)
    lo.add_argument("--docs", type=int, default=128)
    lo.add_argument("--max-length", type=int, default=2048)
    lo.add_argument("--batch", type=int, default=8)
    lo.add_argument("--score", choices=("reply", "content", "last"), default="reply", help="chat rows: see read_docs")
    a = ap.parse_args()
    if a.cmd == "gram":
        k, K = map(int, a.shard.split("/"))
        torch.save(delta_gram(a.base, _pairs(a.ft), k, K, a.device), a.out)
    elif a.cmd == "report":
        rep = report(load_grams(a.grams), a.fit, a.set, a.pair)
        print_report(rep)
        if a.json:
            Path(a.json).write_text(json.dumps(rep))
    else:
        delta_loss(a.base, _pairs(a.ft), _pairs(a.data), a.spec, a.out, a.docs, a.max_length, a.batch, a.score)


if __name__ == "__main__":
    main()
