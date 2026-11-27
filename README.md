# shared-delta-moe

A small Python library: a top-k SwiGLU MoE stored as **one shared base $W_b$ plus a delta per expert**, $W_e = W_b+\Delta_e$ (dense, seeded-DARE or 1-bit BitDelta deltas; also a two-level `HierDeltaMoE`), with several forward implementations that are exactly equivalent to the original MoE, including a Triton kernel that fuses $W_b+\Delta_e$ into the GEMM.

```
src/shared_delta_moe/models/delta_moe.py  DeltaMoE: from_experts, routing, the forward implementations
src/shared_delta_moe/ops.py               stateless helpers: mlp, grouping routed rows, one grouped GEMM, bit packing, pow2
src/shared_delta_moe/triton_kernels/      fused.py: the fused parameter-aggregation forward (sort, gate_up + SiLU,
                                          down + routing weight; dense, seeded-DARE or BitDelta deltas; Hopper TMA); train.py:
                                          its backward for training (dense deltas, triton_train)
src/shared_delta_moe/models/moe.py        MoE / HierMoE: the plain MoEs (full experts, no base), routed as DeltaMoE /
                                          HierDeltaMoE; to_moe() of either delta form
src/shared_delta_moe/models/hier.py       HierDeltaMoE: groups of deltas + zero experts (= the base), two-level routing
src/shared_delta_moe/models/packed.py     PackedDeltaMoE: the common part of the packed formats (DARE, BitDelta)
src/shared_delta_moe/models/dare.py       seeded DARE: Philox mask, its bitmap, pack, DareDeltaMoE
src/shared_delta_moe/models/bitdelta.py   BitDelta: 1-bit sign deltas with per-matrix or per-row scales, BitDeltaMoE
src/shared_delta_moe/models/qwen3_5/common.py
                                          shared by both Qwen3.5 integrations: block forward, decoder-layer patch,
                                          streaming checkpoint reader / shard writer, config copy
src/shared_delta_moe/models/qwen3_5/qwen3_5_delta_moe/
                                          transformers' Qwen3.5 MoE with DeltaMoE blocks (forward_param): modeling.py
                                          (patches the decoder layer for configs with delta_moe_zero_experts), convert.py
                                          (plain MoE checkpoint -> delta layout)
src/shared_delta_moe/models/qwen3_5/qwen3_5_hier_delta_moe/
                                          dense Qwen3.5 / Qwen3.8 with HierDeltaMoE MLPs: modeling.py, convert.py
                                          (base + domain fine-tunes -> one group per domain, other tensors merged)
src/shared_delta_moe/merging.py           merge methods for the shared tensors: fusion_bench calls (extra merging)
src/shared_delta_moe/similarity.py        how alike the fine-tunes are before upscaling: delta Gram statistics, report,
                                          LM loss of merged / forced-routing dense models (docs/guide/similarity.md)
src/shared_delta_moe/constants.py         HAS_TRITON and the thresholds that pick an implementation
src/shared_delta_moe/type_hints.py        shape-annotated tensor aliases (T tokens, H hidden, ...)
src/shared_delta_moe/testing.py           random_experts for tests and benchmarks
tests/test_delta_moe.py                   every forward == plain MoE, token by token (Triton part needs CUDA)
tests/test_dare.py                        Philox known-answer, CPU == CUDA masks, bitmap == mask, DARE forwards and gradients
tests/test_similarity.py                  Gram statistics == direct computation on tiny checkpoints, shards, fits, ulp counts
tests/test_moe.py                         MoE / HierMoE: loop == grouped == token by token, gradients, meta + state dict
tests/test_hier.py                        HierDeltaMoE forwards and gradients == the two-level ensemble, token by token
tests/test_qwen3_5_hier_delta_moe.py     tiny dense Qwen3.5 upscaled == ensemble of its fine-tunes' MLPs
tests/test_train.py                       the Triton backward == the materialised path (x, router, every weight; Hier)
tests/test_bitdelta.py                    sign packing, BitDelta scales, forwards == plain MoE, scale gradients, Triton mode 2
tests/test_qwen3_5_delta_moe.py           tiny Qwen3.5 MoE: converted == stock MoE at top-1, top-2, soft routing
tests/test_fused.py                       the fused kernels: sort, every autotune config, split-K, int64 offsets,
                                          CUDA graph, torch.compile
benchmarks/bench.py                       per-layer time of each implementation (CUDA)
benchmarks/bench_train.py                 training step: Triton backward vs materialised, time and memory
```

```bash
git clone https://github.com/tanganke/fusion_bench ../fusion_bench   # uv uses it editable (merging.py, tests)
uv sync                                  # or: pip install -e .   (Triton ships with torch on Linux + CUDA)
uv run pytest                            # or: python tests/test_delta_moe.py
uv run python benchmarks/bench.py --top-k 2 --tokens 1,32,512,8192
```

```python
from shared_delta_moe import DeltaMoE
m = DeltaMoE.from_experts(base, experts, router, top_k=1, num_zero_experts=1)
y = m.forward_param(x)   # or forward_matrix(x); m.to_moe() is the same MoE with full experts (the baseline)
```

## Docs

Details live in `docs/`, a Sphinx site: guide pages in MyST Markdown, one topic per file, and an API reference
(`docs/api.md`) generated from the docstrings. This README keeps only the layout, setup and this index.

```bash
uv run --group docs --extra transformers sphinx-build -W -b html docs docs/_build/html   # open docs/_build/html/index.html
```

Published at https://tanganke.github.io/shared-delta-moe/, built by `.github/workflows/docs.yml` on every push to
main (pull requests build only).

- A new page goes into a toctree: `docs/index.md`, or a parent page such as `guide/delta-formats.md` or
  `internals/kernels.md` (and this list). A new module needs nothing: `docs/api.md` runs
  autosummary recursively over the package (one page per module, generated into `docs/_api/`); a new subpackage needs an
  `__init__.py` with a one-line docstring.
- Public docstrings are Google style (summary line, then `Args:` / `Returns:` / `Raises:`, shapes in each argument),
  rendered by napoleon; the text inside is reStructuredText: aligned blocks after `::` and a blank line, `|x|` as
  ``` ``|x|`` ```. NamedTuple fields go in `:ivar name:` entries, not `Attributes:`. The build runs with `-W`, so a
  malformed docstring or a page missing from a toctree fails it.
- Diagrams are ```` ```mermaid ```` blocks (sphinxcontrib-mermaid; GitHub renders them too). `docs/_static/compact.css`
  holds the layout changes over Furo.


- [getting-started.md](docs/getting-started.md): install, one MoE layer, a whole model
- User guide (`docs/guide/`): [deltamoe.md](docs/guide/deltamoe.md) (the delta form, zero experts, storage),
  [hier.md](docs/guide/hier.md) (two-level routing), [moe.md](docs/guide/moe.md) (the plain MoE / HierMoE, `to_moe()`), [delta-formats.md](docs/guide/delta-formats.md) with
  [dare.md](docs/guide/dare.md) and [bitdelta.md](docs/guide/bitdelta.md), [qwen3_5.md](docs/guide/qwen3_5.md)
  (converting Qwen3.5 models), [vllm.md](docs/guide/vllm.md), [training.md](docs/guide/training.md),
  [router-training.md](docs/guide/router-training.md) (what worked when training routers)
- Internals (`docs/internals/`): [dispatch.md](docs/internals/dispatch.md) (which forward runs when),
  [kernels.md](docs/internals/kernels.md) with [fused-triton.md](docs/internals/fused-triton.md),
  [cuda-decode.md](docs/internals/cuda-decode.md) and [dense-prefill.md](docs/internals/dense-prefill.md),
  [benchmarks.md](docs/internals/benchmarks.md) (every speed table)
- Notes: [transformers-decode.md](docs/internals/transformers-decode.md) (why the plain MoE looked slow in
  transformers' `generate()`), [cuda-dense-kernel.md](docs/internals/cuda-dense-kernel.md) (the hand-written dense
  kernel, not kept)
- [api.md](docs/api.md): API reference, one page per module in the package layout, generated from the docstrings
