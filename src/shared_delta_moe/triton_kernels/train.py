r"""Backward of the fused parameter-aggregation MoE, for training through ``forward_param`` with
``DeltaMoE.triton_train`` (dense deltas; any top-k, any pair count; HierDeltaMoE through its flattened slots).

.. math::

   y = \sum_i r_i\, \mathrm{MLP}(x; W + D_{e_i}), \qquad
   \mathrm{MLP}(x; W) = (\mathrm{SiLU}(x W_g^\top) \odot (x W_u^\top))\, W_d^\top

Per (token, slot) pair :math:`i`, with :math:`g_i, u_i` the gate / up pre-activations,
:math:`h_i = \mathrm{SiLU}(g_i) \odot u_i` and :math:`dh'_i = dy\,(W_d + D_d)` (not yet weighted by :math:`r_i`):

.. math::

   \frac{\partial L}{\partial r_i} &= \langle dy,\ h_i (W_d + D_d)^\top \rangle = \langle dh'_i, h_i \rangle \\
   \frac{\partial L}{\partial g_i} &= r_i\, dh'_i \odot u_i \odot \mathrm{SiLU}'(g_i), \qquad
   \frac{\partial L}{\partial u_i} = r_i\, dh'_i \odot \mathrm{SiLU}(g_i) \\
   \frac{\partial L}{\partial x} &= \sum_i \frac{\partial L}{\partial g_i} (W_g + D_g)
   + \frac{\partial L}{\partial u_i} (W_u + D_u)

The forward saves only :math:`x`, the routing (index and weight per slot) and the weights; nothing per pair. The
backward recomputes :math:`g` and :math:`u` inside the kernel that forms :math:`dh'` (all three GEMMs share the output
tile, pairs :math:`\times` inter, and the reduction over hidden), so neither :math:`g`, :math:`u` nor :math:`h` is ever
written: ``_bwd_kernel`` writes :math:`\partial L/\partial g` and :math:`\partial L/\partial u`, and
:math:`\partial L/\partial r` by atomic adds; :math:`\partial L/\partial x` is then the down kernel on
:math:`[\partial L/\partial g \mid \partial L/\partial u]` against :math:`[W_g; W_u]^\top`. Weight gradients (only when
a weight requires grad) use torch grouped GEMMs over the sorted pairs.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from .fused import (
    AUTOTUNE,
    Delta,
    Routing,
    _cfg,
    _dense_delta,
    _fit,
    _grid,
    _groups,
    _launch_fixed,
    _smem_limit,
    _tile,
    down,
    gate_up,
    route,
)

_BWD_CONFIGS = [
    _cfg(bm, bn, bk, w, s, 1)
    for bm, bn, bk, w, s in ((16, 64, 64, 4, 3), (32, 64, 64, 4, 3), (64, 64, 64, 8, 3), (64, 64, 32, 4, 4),
                             (128, 64, 32, 8, 3), (64, 128, 32, 8, 3), (128, 128, 32, 8, 2))  # fmt: skip
]


def _prune_bwd(configs, named_args, **kwargs):
    a = {**named_args, **kwargs}
    limit = _smem_limit(a["gstart_ptr"])
    out = [_fit(c, 6, limit, rows=2) for c in configs if c.kwargs["BM"] <= max(16, a["P_BUCKET"])]  # x, dy; 6 weights
    out = [c for c in out if c is not None]
    assert out, "no config fits"
    return out[: None if AUTOTUNE else 1]


@triton.autotune(configs=_BWD_CONFIGS, key=["N", "K", "P_BUCKET", "TOPK"], prune_configs_by={"early_config_prune": _prune_bwd},
                 reset_to_zero=["dr_ptr"])  # fmt: skip
@triton.jit
def _bwd_kernel(
    x_ptr, dy_ptr, perm_ptr, gstart_ptr, rw_ptr, wg_ptr, wu_ptr, wd_ptr, dg_ptr, du_ptr, dd_ptr,
    dgo_ptr, duo_ptr, dr_ptr,
    N, K, G, P_BUCKET,
    sxm, sxk, sym, syk, sgn, sgk, sun, suk, sdn, sdk, sDg, sDn, sDk, sFg, sFn, sFk, sEg, sEn, sEk, som, son,
    TOPK: tl.constexpr, GP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, OCC: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
    """Tile (sorted pairs rows, inter columns cols): g, u = x_tok (W + D)^T for gate / up and dh' = dy_tok (W_d + D_d)
    over the same columns (W_d read transposed, strides sdn / sdk give its (inter, hidden) view). sg* / su* / sd*:
    strides of W_g, W_u and W_d^T; sD* / sF* / sE*: of D_g, D_u and the transposed D_d."""
    lo, hi, mt, mend, MT = _groups(gstart_ptr, G, BM, GP)
    NT = tl.cdiv(N, BN)
    ks = tl.arange(0, BK)
    for t in range(tl.program_id(0), MT * NT, tl.num_programs(0)):
        g, row0, rend, ni, _ = _tile(t, MT, NT, lo, hi, mt, mend, BM, GP, GROUP_M)
        rows = row0 + tl.arange(0, BM)
        rmask = rows < rend
        pair = tl.load(perm_ptr + rows, mask=rmask, other=0)
        tok = pair // TOPK
        cols = ni * BN + tl.arange(0, BN)
        cmask = cols < N
        dcols = cmask & (g < G)
        acc_g = tl.zeros((BN, BM), tl.float32)
        acc_u = tl.zeros((BN, BM), tl.float32)
        acc_d = tl.zeros((BN, BM), tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + ks
            kmask = kk < K
            amask = rmask[:, None] & kmask[None, :]
            wmask = cmask[:, None] & kmask[None, :]
            dmask = dcols[:, None] & kmask[None, :]
            a = tl.load(x_ptr + tok.to(tl.int64)[:, None] * sxm + kk[None, :] * sxk, mask=amask, other=0.0)
            b = tl.load(dy_ptr + tok.to(tl.int64)[:, None] * sym + kk[None, :] * syk, mask=amask, other=0.0)
            wg = tl.load(wg_ptr + cols[:, None] * sgn + kk[None, :] * sgk, mask=wmask, other=0.0)
            wu = tl.load(wu_ptr + cols[:, None] * sun + kk[None, :] * suk, mask=wmask, other=0.0)
            wd = tl.load(wd_ptr + cols[:, None] * sdn + kk[None, :] * sdk, mask=wmask, other=0.0)
            Dg = _dense_delta(dg_ptr, g, cols, kk, dmask, sDg, sDn, sDk)
            Du = _dense_delta(du_ptr, g, cols, kk, dmask, sFg, sFn, sFk)
            Dd = _dense_delta(dd_ptr, g, cols, kk, dmask, sEg, sEn, sEk)
            at = tl.trans(a)
            acc_g = tl.dot(wg + Dg, at, acc_g)
            acc_u = tl.dot(wu + Du, at, acc_u)
            acc_d = tl.dot(wd + Dd, tl.trans(b), acc_d)
        r = tl.load(rw_ptr + pair, mask=rmask, other=0.0)
        # g, u, SiLU(g) and h rounded as the forward (gate_up) rounds them, so dL/dr is the gradient of that forward
        dt = dgo_ptr.dtype.element_ty
        gr = acc_g.to(dt).to(tl.float32)
        ur = acc_u.to(dt).to(tl.float32)
        sg = tl.sigmoid(gr)
        silu = (gr * sg).to(dt).to(tl.float32)
        keep = cmask[:, None] & rmask[None, :]
        h = (silu * ur).to(dt).to(tl.float32)
        tl.atomic_add(dr_ptr + pair, tl.sum(tl.where(keep, acc_d * h, 0.0), 0), mask=rmask, sem="relaxed")
        dh = acc_d * r[None, :]
        dgv = dh * ur * sg * (1.0 + gr * (1.0 - sg))
        duv = dh * silu
        out = rows.to(tl.int64)[None, :] * som + cols[:, None] * son
        tl.store(dgo_ptr + out, dgv.to(dgo_ptr.dtype.element_ty), mask=keep)
        tl.store(duo_ptr + out, duv.to(duo_ptr.dtype.element_ty), mask=keep)


def _bwd(x: Tensor, dy: Tensor, r: Routing, w: tuple[Tensor, Tensor, Tensor], d: tuple[Tensor, Tensor, Tensor],
         config: triton.Config | None = None) -> tuple[Tensor, Tensor]:  # fmt: skip
    """dL/d[g | u] (P, 2I) in sorted order, the two halves being dL/dg and dL/du, and dL/dr (P,) by pair id."""
    wg, wu, wd = w
    dg, du, dd = d
    P, (N, K), G = r.perm.numel(), wg.shape, r.n_delta
    dgu = torch.empty(P, 2 * N, dtype=x.dtype, device=x.device)
    dr = torch.zeros(P, dtype=torch.float32, device=x.device)
    grid, _ = _grid(x.device, P, G, N, 1)
    wdT, ddT = wd.t(), dd.transpose(1, 2)  # (I, H) and (G, I, H) views
    args = (x, dy, r.perm, r.gstart, r.weight, wg, wu, wdT, dg, du, ddT, dgu, dgu[:, N:], dr, N, K, G,
            triton.next_power_of_2(P), *x.stride(), *dy.stride(), *wg.stride(), *wu.stride(), *wdT.stride(),
            *dg.stride(), *du.stride(), *ddT.stride(), *dgu.stride())  # fmt: skip
    kw = dict(TOPK=r.top_k, GP=triton.next_power_of_2(G + 2))
    with torch.cuda.device(x.device):
        if config is None:
            _bwd_kernel[grid](*args, **kw)
        else:
            _launch_fixed(_bwd_kernel, grid, config, 6, args, kw, rows=2)
    return dgu, dr


def _weight_grads(x: Tensor, dy: Tensor, r: Routing, h: Tensor, dgo: Tensor, duo: Tensor, needs: tuple[bool, ...]):
    """Per-group weight gradients with torch grouped GEMMs (pairs as the reduction dimension): for each group,
    dW_d+D_d = (r dy)^T h and dW_g+D_g = dg^T x, dW_u+D_u = du^T x. The base gets the sum over all groups (it is in
    every expert), a delta its own group's. needs: requires-grad of (W_g, W_u, W_d, D_g, D_u, D_d)."""
    perm, G = r.perm.long(), r.n_delta
    tok = perm // r.top_k
    offs = r.gstart[1:].contiguous()  # cumulative group ends, groups 0 .. G
    xs = x[tok]
    dys = (dy[tok].float() * r.weight[perm, None]).to(dy.dtype)
    out: list[Tensor | None] = [None] * 6
    for j, (a, b) in enumerate(((dgo, xs), (duo, xs), (dys, h))):
        if needs[j] or needs[3 + j]:
            gw = F.grouped_mm(a.t(), b, offs=offs)  # (G + 1, rows, cols)
            out[j] = gw.float().sum(0).to(gw.dtype) if needs[j] else None  # the base: every group
            out[3 + j] = gw[:G] if needs[3 + j] else None
    return out


class _FusedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, rw, wg, wu, wd, dg, du, dd, n_delta: int, top_k: int):  # type: ignore[override]
        r = route(idx, rw.detach(), n_delta)
        y = down(gate_up(x, r, wg, wu, Delta(0, dg), Delta(0, du)), r, wd, Delta(0, dd), split_k=1)
        ctx.save_for_backward(x, idx, rw, wg, wu, wd, dg, du, dd)
        ctx.n_delta, ctx.top_k = n_delta, top_k
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):  # type: ignore[override]
        x, idx, rw, wg, wu, wd, dg, du, dd = ctx.saved_tensors
        T, H = x.shape
        r = route(idx, rw.detach(), ctx.n_delta)
        dy = dy.to(x.dtype).contiguous()  # (an fp32 dy from an autocast region downstream)
        dgu, dr = _bwd(x, dy, r, (wg, wu, wd), (dg, du, dd))
        dgo, duo = dgu.chunk(2, 1)
        dx = None
        if ctx.needs_input_grad[0]:
            # dL/dx = [dg | du] [W_g; W_u] summed over each token's slots: the down kernel with weight 1 per pair
            ones = Routing(r.perm, r.gstart, torch.ones_like(r.weight), T, r.top_k, r.n_delta)
            wcat = torch.cat((wg, wu)).t()  # (H, 2I)
            dcat = torch.cat((dg, du), 1).transpose(1, 2)  # (G, H, 2I)
            dx = down(dgu, ones, wcat, Delta(0, dcat), split_k=1)
        needs = ctx.needs_input_grad[3:9]
        grads: list = [None] * 6
        if any(needs):
            h = gate_up(x, r, wg, wu, Delta(0, dg), Delta(0, du))
            grads = _weight_grads(x, dy, r, h, dgo, duo, needs)
        return (dx, None, dr.view(T, ctx.top_k).to(rw.dtype), *grads, None, None)


def fused_moe_autograd(x: Tensor, idx: Tensor, rw: Tensor, base: tuple[Tensor, Tensor, Tensor],
                       deltas: tuple[Tensor, Tensor, Tensor], n_delta: int) -> Tensor:  # fmt: skip
    r"""Differentiable fused forward of the parameter-aggregation MoE, with the Triton backward.

    :math:`y = \sum_i \mathrm{rw}_i\, \mathrm{MLP}(x; W + D_{\mathrm{idx}_i})` over each token's :math:`k` slots.
    Gradients flow to ``x``, ``rw`` and every weight that requires grad; ``idx`` gets none.

    Args:
        x: (T, H) tokens.
        idx: (T, k) int expert per slot (>= ``n_delta``: a zero expert).
        rw: (T, k) routing weights (differentiable).
        base: (:math:`W_g` (I, H), :math:`W_u` (I, H), :math:`W_d` (H, I)), the base weights.
        deltas: dense deltas (:math:`D_g` (n_delta, I, H), :math:`D_u` (n_delta, I, H), :math:`D_d` (n_delta, H, I)).
        n_delta: the number of delta experts.

    Returns:
        (T, H) output :math:`y`; empty when :math:`T = 0`.
    """
    if x.shape[0] == 0:
        return torch.empty_like(x)
    return _FusedMoE.apply(x, idx, rw, *base, *deltas, n_delta, idx.shape[1])


