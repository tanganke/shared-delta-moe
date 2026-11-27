# Looped layers and expert LoRA

A loop runs a stack of decoder layers several times on one carried state; with MoE layers in the stack this is a
looped MoE. It is the loop of qihuangz's Mixture-of-Loop (MoL) experiments on Qwen3.8-27B: layers 28 to 31, 3 rounds.
`models/looped.py` has the loop, `models/qwen3_5/looped.py` puts it into transformers' Qwen3.5 text model, and
`models/lora.py` gives every delta expert its own LoRA, as MoL trains its experts.

## The loop

Round 0 runs the stack $f$ on its input, the anchor $a$. Every later round runs $f$ on the anchor plus a low-rank
feedback of the state, then moves the state part of the way towards the result:

$$
h_0 = f(a), \qquad v_r = f\bigl(a + U D\,\mathrm{rms}(h_{r-1})\bigr), \qquad h_r = h_{r-1} + \sigma(u)\,(v_r - h_{r-1})
$$

`LoopUpdate` holds $D$, $U$ and $u$ in float32. $U$ starts at zero, so at init every round repeats round 0 and the
looped model computes the same function as the plain model. $\sigma(u)$ starts at `update_init` (0.1).

**Routing.** MoE blocks in the stack route in one of two ways:

| `routing` | rounds after round 0 |
|---|---|
| `"fixed"` | reuse round 0's indices and weights; the router runs once per forward, and the reused weights keep their gradient |
| `"dynamic"` | route again |

`loopable(moe)` turns a block's `topk` into one that does this inside a loop and is unchanged outside one. Any
DeltaMoE takes it, including HierDeltaMoE and the packed formats. Saved routes are keyed by block and router. A layer's
routed attention deltas ([routed-deltas.md](routed-deltas.md)) route through `moe.topk(x, delta_router)`, so they are
fixed together with the MLP.

**Balance loss.** `loop(..., aux=[])` collects the Switch balance loss $E \sum_e f_e P_e$ of every routing call: round
0 only under fixed routing, every round under dynamic routing. It works for flat routers only.

## Qwen3.5

```python
from shared_delta_moe.models.qwen3_5.looped import loop_layers
loop_layers(model, start=28, size=4, repeats=3, routing="fixed", feedback_rank=64, update_init=0.1)
```

`loop_layers` changes a loaded model in place. The layers can have any MLP: stock, DeltaMoE or HierDeltaMoE, with or
without LoRA.

- **Caches.** Each round of a layer needs its own KV cache and its own linear-attention state. Round $r > 0$ of layer
  `start + j` runs an alias of that layer: it shares every parameter, but its attention has
  `layer_idx = num_hidden_layers + (r - 1) * size + j`. The text config's `layer_types` gets the types of those
  slots appended, so the `DynamicCache` that `generate` builds already has them. The aliases are not submodules, so
  the state dict has no duplicate keys.
- **Checkpointing.** Under gradient checkpointing every layer call is its own checkpoint, inside the loop too. Each
  is wrapped in `keep_routing`, so its recompute during backward routes as its round did: later rounds replay round
  0's routing under fixed routing, and `ops.replay` keeps round 0's own picks. (One checkpoint around the whole loop
  ran out of memory on an H200 for an 18.8k-token Qwen3.8-27B row.)
- **Training mode.** `model.model.loop_aux` holds the balance losses of the last forward; read it before backward.
- **Saving.** Save the adapters (`loop_update`, routers, LoRA) as a state dict. No `from_pretrained` path builds the
  loop, and no vLLM path exists yet.

## Expert LoRA

```python
from shared_delta_moe.models.lora import add_expert_lora
add_expert_lora(moe, rank=16, alpha=16)
```

`add_expert_lora` gives every delta expert $e$ a LoRA on its gate, up and down projections. Each projection gets
$\tfrac{\alpha}{r} B_{e} A_{e}\,x$ added in activation space: the LoRA is not merged, because merging it into bf16
weights would round small updates away.

- **Init.** $A$ is kaiming uniform, as `nn.Linear` initialises; $B$ is zero, so the output is unchanged at init.
- **Zero experts.** A zero expert (the base alone) gets no LoRA. To give the base its own LoRA, store it as an
  all-zero delta.
- **Forward path.** With LoRA, the block always runs the materialised per-expert loop on `weights(e)`. The fused
  Triton, grouped and matrix forwards read the deltas directly, so they would skip the LoRA.

`tests/test_looped.py` checks the following:

- At init the loop equals the plain model.
- Fixed routing equals a dynamic loop whose `topk` replays round 0.
- Generation with cache equals the full forward.
- Checkpointed gradients equal plain ones.
- Expert LoRA (dense and grid deltas) equals a reference MoE.
