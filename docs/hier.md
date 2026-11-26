# Two-level shared delta MoE (`HierDeltaMoE`)

A shared base expert plus delta experts arranged in groups, routed first over groups and then within the chosen groups
(`models/hier.py`). Group $i$ holds $E_i$ delta experts with weights $W_b+\Delta_{ij}$ and $Z_i$ zero experts (no
parameters, output 0); $E_i$ and $Z_i$ may differ between groups, and a group may hold only zero experts.

- Group router: $h = \mathrm{softmax}(R_0 x)$ over the $G$ groups; dense / soft (`group_top_k=None`: every group,
  weight $h_i$) or top-$k_0$ renormalised over the chosen groups.
- Within group $i$: softmax over its $E_i+Z_i$ logits, top-$k_1$ (`top_k`), renormalised within the group; zero
  experts compete like the others.

$$
y = \mathrm{MLP}(x; W_b) + \sum_{i \in S(x)} h_i(x) \sum_{j \in \text{top-}k_1,\ j \text{ a delta}} \tilde p_{ij}(x)\, \mathrm{MLP}(x; W_b+\Delta_{ij})
$$

This equals the plain two-level ensemble of the full experts $\theta_{ij}=W_b+\Delta_{ij}$ with the base as a shared
expert of weight 1 (`from_experts` takes those full experts and stores $\Delta_{ij}=\theta_{ij}-W_b$). One linear layer
holds both routers: rows $0..G-1$ are $R_0$, then group by group its $E_i$ delta rows and $Z_i$ zero-expert rows.

**Implementation.** `topk` flattens the routing into ordinary `DeltaMoE` slots: per token $S\cdot k_1$ slots
($S=G$ dense, else $k_0$) weighted $h_i\tilde p_{ij}$, plus one base slot of weight 1; a picked zero expert becomes a base
slot of weight 0. Every `DeltaMoE` forward (materialised, split, grouped, fused Triton) and the Triton training path
([training.md](training.md)) then compute $y$ unchanged, gradients through the two softmaxes included.

**Limits.** Zero-weight base slots are still computed (one base MLP each in the parameter-aggregation forwards):
waste that grows with $S\cdot k_1$ minus the deltas picked. Dense deltas only: `DareDeltaMoE.from_delta_moe` refuses a
`HierDeltaMoE`. The routing tables are non-persistent buffers built in `__init__`, so a module made on the meta device
needs them rebuilt (not handled). `tests/test_hier.py` checks every forward and the gradients against the ensemble
written out token by token.
