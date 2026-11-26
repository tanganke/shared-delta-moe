"""Fused parameter-aggregation MoE in three launches after the routing, with no host synchronisation:

  sort     one program: a stable counting sort of the (token, slot) pairs by group (delta e -> group e, every zero
           expert -> group G), for a routing chosen in torch (DeltaMoE.topk: the same choice as every other forward
           and as the autograd path, ties included). Writes perm (sorted position -> pair) and gstart (G + 2 group
           starts in sorted order).
  gate_up  h = SiLU(x (W_g + D_g)^T) * (x (W_u + D_u)^T) for every sorted pair, x rows gathered through perm, the
           sums W + D formed tile by tile in registers; h is written in sorted order.
  down     y = h (W_d + D_d)^T, scaled by the routing weight and written to the pair's slot (split-K: atomically
           added to the token's float32 row, which also sums the top-k slots).

The GEMM kernels are persistent: each program walks tiles (group, row tile, column tile, K split) with a grouped
order, so programs running at the same time share weight and activation tiles in L2. The tile list is derived on
the device from gstart, which is what lets a whole forward run under a CUDA graph.

Deltas come in two storage modes (MODE): 0 dense (G, N, K); 1 seeded DARE (models/dare.py), the keep bits
of each tile read from a 1-bit-per-block mask bitmap and the kept values gathered from the packed storage. On Hopper (sm90+) with dense deltas the
weight tiles are loaded with TMA (tensor descriptors), so that address arithmetic and masking leave the main loop.
Row offsets (row index times row stride) are int64: h (pairs x inter) passes 2^31 elements in training.
"""

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.tools.tensor_descriptor import TensorDescriptor

from ..constants import SORT_ONE_HOT

AUTOTUNE = os.environ.get("SHARED_DELTA_MOE_AUTOTUNE", "1") != "0"  # "0": first fitting config only (tests)

# ----------------------------------------------------------------------------------------------------------- sort


@triton.jit
def _groups_of(idx_ptr, t0, T, n_delta, TOPK: tl.constexpr, KP: tl.constexpr, BT: tl.constexpr):
    """Group (BT * KP,), validity and pair id of the slots of tokens t0 .. t0 + BT (idx (T, TOPK) contiguous)."""
    tt = t0 + tl.arange(0, BT)
    kar = tl.arange(0, KP)
    valid = (tt < T)[:, None] & (kar < TOPK)[None, :]
    pair = tt[:, None] * TOPK + kar[None, :]
    idx = tl.load(idx_ptr + pair, mask=valid, other=0).to(tl.int32)
    grp = tl.where(idx < n_delta, idx, n_delta)  # zero experts -> group G = n_delta
    return tl.reshape(grp, (BT * KP,)), tl.reshape(valid, (BT * KP,)), tl.reshape(pair, (BT * KP,))


@triton.jit
def _sort_kernel(idx_ptr, perm_ptr, gstart_ptr, T, n_delta, TOPK: tl.constexpr, KP: tl.constexpr, GP: tl.constexpr,
                 BT: tl.constexpr):  # fmt: skip
    gar = tl.arange(0, GP)
    counts = tl.zeros((GP,), tl.int32)
    for t0 in range(0, T, BT):  # pass 1: group sizes
        grp, valid, pair = _groups_of(idx_ptr, t0, T, n_delta, TOPK, KP, BT)
        counts += tl.sum(((grp[:, None] == gar[None, :]) & valid[:, None]).to(tl.int32), 0)
    run = tl.cumsum(counts, 0) - counts  # group starts; entry G + 1 is the pair count
    tl.store(gstart_ptr + gar, run, mask=gar <= n_delta + 1)
    for t0 in range(0, T, BT):  # pass 2: each pair's position = its group's running start + its rank in the chunk
        grp, valid, pair = _groups_of(idx_ptr, t0, T, n_delta, TOPK, KP, BT)
        oh = ((grp[:, None] == gar[None, :]) & valid[:, None]).to(tl.int32)
        pos = tl.sum(oh * (tl.cumsum(oh, 0) - oh + run[None, :]), 1)
        tl.store(perm_ptr + pos, pair, mask=valid)
        run += tl.sum(oh, 0)


class Routing(NamedTuple):
    perm: Tensor  # (P,) int32, sorted position -> pair (token * top_k + slot)
    gstart: Tensor  # (G + 2,) int32, start of each group in sorted order; [G + 1] = P
    weight: Tensor  # (P,) float32, renormalised routing weight of each pair
    tokens: int
    top_k: int
    n_delta: int


def route(idx: Tensor, weight: Tensor, n_delta: int) -> Routing:
    """Sort a routing: idx (T, k) expert per slot (>= n_delta: a zero expert), weight (T, k) its routing weight."""
    T, k = idx.shape
    idx = idx.contiguous()
    perm = torch.empty(T * k, dtype=torch.int32, device=idx.device)
    gstart = torch.empty(n_delta + 2, dtype=torch.int32, device=idx.device)
    KP, GP = triton.next_power_of_2(k), triton.next_power_of_2(n_delta + 2)
    BT = max(1, min(512 // KP, SORT_ONE_HOT // (KP * GP)))  # the (BT * KP, GP) one-hot fits in shared memory
    with torch.cuda.device(idx.device):
        _sort_kernel[(1,)](idx, perm, gstart, T, n_delta, TOPK=k, KP=KP, GP=GP, BT=BT, num_warps=4)  # fmt: skip
    return Routing(perm, gstart, weight.float().reshape(-1).contiguous(), T, k, n_delta)


# ---------------------------------------------------------------------------------------------- tile scheduling


@triton.jit
def _groups(gstart_ptr, G, BM: tl.constexpr, GP: tl.constexpr):
    """Per group (vectors of GP): row start, row end, row tiles, cumulative row tiles; and the total row tiles."""
    gar = tl.arange(0, GP)
    lo = tl.load(gstart_ptr + gar, mask=gar <= G, other=0)
    hi = tl.load(gstart_ptr + gar + 1, mask=gar <= G, other=0)
    mt = (hi - lo + BM - 1) // BM
    mend = tl.cumsum(mt, 0)
    return lo, hi, mt, mend, tl.sum(mt, 0)


@triton.jit
def _tile(t, MT, NT, lo, hi, mt, mend, BM: tl.constexpr, GP: tl.constexpr, GROUP_M: tl.constexpr):
    """Tile t -> (group, first row, row end, column tile, K split). Row tiles of all groups are numbered in group
    order; GROUP_M consecutive row tiles sweep the column tiles together (they share weight tiles in L2, and in
    decode, where each group is one row tile, the groups of one column tile run side by side and share W_b)."""
    MN = MT * NT
    s = t // MN
    q = t % MN
    width = GROUP_M * NT
    first = (q // width) * GROUP_M
    gsz = tl.minimum(MT - first, GROUP_M)
    mi = first + (q % width) % gsz
    ni = (q % width) // gsz
    sel = tl.arange(0, GP) == tl.sum((mend <= mi).to(tl.int32), 0)
    row0 = tl.sum(tl.where(sel, lo + (mi - mend + mt) * BM, 0), 0)
    rend = tl.sum(tl.where(sel, hi, 0), 0)
    g = tl.sum(tl.where(sel, tl.arange(0, GP), 0), 0)
    return g, row0, rend, ni, s


# ------------------------------------------------------------------------------------------------- delta tiles


@triton.jit
def _dense_delta(d_ptr, g, cols, kk, dmask, sdg, sdn, sdk):
    return tl.load(d_ptr + g.to(tl.int64) * sdg + cols[:, None] * sdn + kk[None, :] * sdk, mask=dmask, other=0.0)


@triton.jit
def _mask_words(m_ptr, mrow, dcols, MW, MWP: tl.constexpr):
    """A row's whole mask bitmap, (BN, MWP) words, when it is short (MWP: words padded to a power of 2; 0 = read per
    tile instead): loaded once per tile so the K loop has no load the delta gather has to wait for."""
    W1: tl.constexpr = MWP if MWP > 0 else 1
    ar = tl.arange(0, W1)[None, :]
    return tl.load(m_ptr + mrow + ar, mask=dcols[:, None] & (ar < MW) & (MWP > 0), other=0)


@triton.jit
def _dare_delta(v_ptr, run, k0, kk, dmask, m_ptr, mrow, words, KB, B: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, MWP: tl.constexpr):  # fmt: skip
    """Seeded-DARE delta tile: the keep bit of each DARE block from the mask bitmap (block j of a row in bit j % 32
    of word j // 32; KB blocks per row): from words (_mask_words) when MWP > 0, else read at m_ptr + mrow. Kept values
    gathered from v at each row's running position (run, advanced by the tile's kept count). Needs BK % B == 0 or
    B % BK == 0."""
    J: tl.constexpr = BK // B if B < BK else 1
    j = (k0 // B + tl.arange(0, J))[None, :]  # (1, J) blocks of this tile
    if MWP > 0:  # pick word j // 32 from the registers
        sel = (j // 32)[:, :, None] == tl.arange(0, MWP)[None, None, :]
        w = tl.sum(tl.where(sel, words[:, None, :], 0), axis=2)  # (BN, J)
    else:
        rm = tl.max(dmask.to(tl.int32), axis=1)[:, None] > 0  # rows with a delta
        w = tl.load(m_ptr + mrow + j // 32, mask=rm & (j < KB), other=0)  # (BN, J), mostly one word
    bit = ((w >> (j % 32)) & 1) != 0
    if B >= BK:
        m = tl.broadcast_to(bit, (BN, BK))
    else:
        m = tl.reshape(tl.broadcast_to(bit[:, :, None], (BN, J, B)), (BN, BK))
    m = m & dmask
    mi = m.to(tl.int32)
    dw = tl.load(v_ptr + run[:, None] + (tl.cumsum(mi, axis=1) - mi), mask=m, other=0.0)
    return dw, run + tl.sum(mi, axis=1)


@triton.jit
def _bit_delta(b_ptr, brow, k0, dmask, scale, MW, BN: tl.constexpr, BK: tl.constexpr):
    """BitDelta tile, float32: entry c of a row is +scale if bit c % 32 of its word c // 32 is set, else -scale (MW
    words per row from brow; BK a multiple of 32)."""
    W: tl.constexpr = BK // 32
    wj = k0 // 32 + tl.arange(0, W)[None, :]
    rm = tl.max(dmask.to(tl.int32), axis=1)[:, None] > 0  # rows with a delta
    w = tl.load(b_ptr + brow + wj, mask=rm & (wj < MW), other=0)  # (BN, W)
    bit = (w[:, :, None] >> tl.arange(0, 32)[None, None, :]) & 1
    sign = (tl.reshape(bit, (BN, BK)) * 2 - 1).to(tl.float32)
    return tl.where(dmask, sign * scale[:, None], 0.0)


# ------------------------------------------------------------------------------------------------ GEMM kernels


def _set_blocks(nargs):
    """TMA descriptors take the box shape of the config being launched (autotune pre_hook)."""
    bm, bn, bk = nargs["BM"], nargs["BN"], nargs["BK"]
    for name, shape in (("wg_desc", [bn, bk]), ("wu_desc", [bn, bk]), ("w_desc", [bn, bk]), ("h_desc", [bm, bk]),
                        ("dg_desc", [1, bn, bk]), ("du_desc", [1, bn, bk]), ("d_desc", [1, bn, bk])):  # fmt: skip
        if nargs.get(name) is not None:
            nargs[name].block_shape = shape


def _cfg(bm, bn, bk, w, s, occ, gm=8):
    return triton.Config({"BM": bm, "BN": bn, "BK": bk, "OCC": occ, "GROUP_M": gm}, num_warps=w, num_stages=s,
                         pre_hook=_set_blocks)  # fmt: skip


# (BM, BN, BK, warps, stages, programs per SM); small BM for decode, large for prefill
_SMALL = [(16, 32, 128, 4, 4), (16, 64, 128, 4, 4), (16, 64, 256, 4, 3), (16, 128, 64, 4, 4), (32, 64, 128, 4, 4),
          (32, 128, 64, 4, 4), (64, 64, 64, 4, 4)]  # fmt: skip
_LARGE = [(64, 128, 64, 8, 3), (128, 128, 64, 8, 3), (128, 128, 32, 8, 4), (64, 128, 32, 4, 5), (128, 64, 64, 4, 4),
          (128, 256, 64, 8, 3), (256, 128, 64, 8, 3)]  # fmt: skip
GATE_UP_CONFIGS = [_cfg(*c, o) for c in _SMALL for o in (1, 2, 4)] + [
    _cfg(*c, 1) for c in _LARGE if c[0] * c[1] <= 128 * 128  # two accumulators
]
DOWN_CONFIGS = [_cfg(*c, o) for c in _SMALL for o in (1, 2, 4)] + [_cfg(*c, 1) for c in _LARGE]


def _smem_limit(t: Tensor) -> int:
    return torch.cuda.get_device_properties(t.device).shared_memory_per_block_optin


def _fit(c: triton.Config, tiles: int, limit: int, tma: bool = False, rows: int = 1) -> triton.Config | None:
    """c with fewer pipeline stages if its (stages - 1) * (rows * BM + tiles * BN) * BK bf16 buffers exceed the shared
    memory limit (tiles: weight tiles, rows: activation tiles per K step); None if even 2 stages do not fit, or
    if c's weight box is 256 wide under TMA."""
    # ponytail: Triton 3.7.1 on H200 gives wrong results with a 256-row TMA box (down, BN 256: rel err 0.3, pointer
    # loads fine); those configs are left out under TMA until a Triton release fixes it
    if tma and c.kwargs["BN"] > 128:
        return None
    per = (rows * c.kwargs["BM"] + tiles * c.kwargs["BN"]) * c.kwargs["BK"] * 2
    st = min(c.num_stages, limit // per + 1)
    if st < 2:
        return None
    return c if st == c.num_stages else triton.Config(c.kwargs, num_warps=c.num_warps, num_stages=st, pre_hook=c.pre_hook)


def _pruner(tiles: int):
    def prune(configs, named_args, **kwargs):
        """Row tiles beyond the pair count are wasted; DARE needs BK and the block size to divide one another;
        pipeline depth cut to the device's shared memory."""
        a = {**named_args, **kwargs}
        pb, b, limit = a["P_BUCKET"], a["B"], _smem_limit(a["gstart_ptr"])
        out = [_fit(c, tiles, limit, tma=a["TMA"]) for c in configs
               if c.kwargs["BM"] <= max(16, pb) and (c.kwargs["BK"] % b == 0 or b % c.kwargs["BK"] == 0)]  # fmt: skip
        out = [c for c in out if c is not None]
        assert out, "no config fits"
        return out[: None if AUTOTUNE else 1]

    return prune


_KEY = ["N", "K", "P_BUCKET", "TOPK", "MODE", "B", "S", "TMA"]


@triton.autotune(configs=GATE_UP_CONFIGS, key=_KEY, prune_configs_by={"early_config_prune": _pruner(4)})
@triton.jit
def _gate_up_kernel(
    x_ptr, perm_ptr, gstart_ptr, wg_ptr, wu_ptr, dg_ptr, du_ptr, pg_ptr, pu_ptr, mg_ptr, mu_ptr, h_ptr,
    wg_desc, wu_desc, dg_desc, du_desc,
    N, K, G, P_BUCKET, MW,
    sxm, sxk, swn, swk, sun, suk, sdg, sdn, sdk, seg, sen, sek, shm, shn,
    TOPK: tl.constexpr, MODE: tl.constexpr, B: tl.constexpr, S: tl.constexpr, GP: tl.constexpr, TMA: tl.constexpr,
    MWP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, OCC: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
    """sw* / su*: strides of W_g / W_u; sd* / se*: of D_g / D_u (dense mode)."""
    lo, hi, mt, mend, MT = _groups(gstart_ptr, G, BM, GP)
    NT = tl.cdiv(N, BN)
    ks = tl.arange(0, BK)
    for t in range(tl.program_id(0), MT * NT, tl.num_programs(0)):
        g, row0, rend, ni, _ = _tile(t, MT, NT, lo, hi, mt, mend, BM, GP, GROUP_M)
        rows = row0 + tl.arange(0, BM)
        rmask = rows < rend
        tok = tl.load(perm_ptr + rows, mask=rmask, other=0) // TOPK
        cols = ni * BN + tl.arange(0, BN)
        cmask = cols < N
        dcols = cmask & (g < G)  # zero expert: delta loads masked off, not branched (keeps the loop pipelined)
        rg = tl.zeros((BN,), tl.int64)
        ru = tl.zeros((BN,), tl.int64)
        sg = tl.zeros((BN,), tl.float32)
        su = tl.zeros((BN,), tl.float32)
        if MODE == 1:
            rg = tl.load(pg_ptr + g * N + cols, mask=dcols, other=0)
            ru = tl.load(pu_ptr + g * N + cols, mask=dcols, other=0)
        if MODE == 2:  # p: the BitDelta row scales
            sg = tl.load(pg_ptr + g * N + cols, mask=dcols, other=0.0).to(tl.float32)
            su = tl.load(pu_ptr + g * N + cols, mask=dcols, other=0.0).to(tl.float32)
        mrow = (g * N + cols).to(tl.int64)[:, None] * MW  # mask words (DARE) / sign words (BitDelta) of each row
        wg_bits = _mask_words(mg_ptr, mrow, dcols, MW, MWP)
        wu_bits = _mask_words(mu_ptr, mrow, dcols, MW, MWP)
        acc_g = tl.zeros((BN, BM), tl.float32)  # y^T: the summed weight is operand A
        acc_u = tl.zeros((BN, BM), tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + ks
            kmask = kk < K
            wmask = cmask[:, None] & kmask[None, :]
            dmask = dcols[:, None] & kmask[None, :]
            a = tl.load(x_ptr + tok.to(tl.int64)[:, None] * sxm + kk[None, :] * sxk, mask=rmask[:, None] & kmask[None, :], other=0.0)
            if TMA:  # out-of-bounds boxes read as zeros: ragged N / K, and the zero expert (g = G) in the deltas
                wg = wg_desc.load([ni * BN, k0])
                wu = wu_desc.load([ni * BN, k0])
            else:
                wg = tl.load(wg_ptr + cols[:, None] * swn + kk[None, :] * swk, mask=wmask, other=0.0)
                wu = tl.load(wu_ptr + cols[:, None] * sun + kk[None, :] * suk, mask=wmask, other=0.0)
            if MODE == 0 and TMA:
                dg = dg_desc.load([g, ni * BN, k0]).reshape(BN, BK)
                du = du_desc.load([g, ni * BN, k0]).reshape(BN, BK)
            elif MODE == 0:
                dg = _dense_delta(dg_ptr, g, cols, kk, dmask, sdg, sdn, sdk)
                du = _dense_delta(du_ptr, g, cols, kk, dmask, seg, sen, sek)
            elif MODE == 1:
                dg, rg = _dare_delta(dg_ptr, rg, k0, kk, dmask, mg_ptr, mrow, wg_bits, K // B, B, BN, BK, MWP)
                du, ru = _dare_delta(du_ptr, ru, k0, kk, dmask, mu_ptr, mrow, wu_bits, K // B, B, BN, BK, MWP)
            else:  # +-scale is exact in the model dtype, so W_b + delta rounds as in the materialised forward
                dg = _bit_delta(dg_ptr, mrow, k0, dmask, sg, MW, BN, BK).to(wg.dtype)
                du = _bit_delta(du_ptr, mrow, k0, dmask, su, MW, BN, BK).to(wu.dtype)
            at = tl.trans(a)
            acc_g = tl.dot(wg + dg, at, acc_g)
            acc_u = tl.dot(wu + du, at, acc_u)
        # rounded where the PyTorch MLP rounds (g, u, SiLU(g), h: each a bf16/fp16 tensor there), so the result
        # matches the model's own numerics; fp32 throughout is closer to exact but drifts from what the model
        # was trained with (32-layer Qwen3.5: argmax agreement with the original 98.4% vs 98.8%)
        dt = h_ptr.dtype.element_ty
        g = acc_g.to(dt).to(tl.float32)
        h = (g * tl.sigmoid(g)).to(dt).to(tl.float32) * acc_u.to(dt).to(tl.float32)
        tl.store(h_ptr + rows.to(tl.int64)[None, :] * shm + cols[:, None] * shn, h.to(h_ptr.dtype.element_ty),
                 mask=cmask[:, None] & rmask[None, :])  # fmt: skip


@triton.autotune(configs=DOWN_CONFIGS, key=_KEY, prune_configs_by={"early_config_prune": _pruner(2)},
                 reset_to_zero=["o_ptr"])  # fmt: skip
@triton.jit
def _down_kernel(
    h_ptr, perm_ptr, gstart_ptr, rw_ptr, w_ptr, d_ptr, p_ptr, m_ptr, o_ptr,
    h_desc, w_desc, d_desc,
    N, K, G, P_BUCKET, MW,
    shm, shk, swn, swk, sdg, sdn, sdk, som, son,
    TOPK: tl.constexpr, MODE: tl.constexpr, B: tl.constexpr, S: tl.constexpr, GP: tl.constexpr, TMA: tl.constexpr,
    MWP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, OCC: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
    lo, hi, mt, mend, MT = _groups(gstart_ptr, G, BM, GP)
    NT = tl.cdiv(N, BN)
    ks = tl.arange(0, BK)
    kper = tl.cdiv(tl.cdiv(K, BK), S) * BK  # K range of one split
    for t in range(tl.program_id(0), MT * NT * S, tl.num_programs(0)):
        g, row0, rend, ni, s = _tile(t, MT, NT, lo, hi, mt, mend, BM, GP, GROUP_M)
        rows = row0 + tl.arange(0, BM)
        rmask = rows < rend
        cols = ni * BN + tl.arange(0, BN)
        cmask = cols < N
        dcols = cmask & (g < G)
        run = tl.zeros((BN,), tl.int64)
        sc = tl.zeros((BN,), tl.float32)
        if MODE == 1:
            run = tl.load(p_ptr + g * N + cols, mask=dcols, other=0)
        if MODE == 2:
            sc = tl.load(p_ptr + g * N + cols, mask=dcols, other=0.0).to(tl.float32)
        mrow = (g * N + cols).to(tl.int64)[:, None] * MW
        bits = _mask_words(m_ptr, mrow, dcols, MW, MWP)
        acc = tl.zeros((BN, BM), tl.float32)
        for k0 in range(s * kper, tl.minimum(K, s * kper + kper), BK):
            kk = k0 + ks
            kmask = kk < K
            dmask = dcols[:, None] & kmask[None, :]
            if TMA:  # h rows past the group's end belong to the next group: computed, not stored
                a = h_desc.load([row0, k0])
                w = w_desc.load([ni * BN, k0])
            else:
                a = tl.load(h_ptr + rows.to(tl.int64)[:, None] * shm + kk[None, :] * shk, mask=rmask[:, None] & kmask[None, :], other=0.0)
                w = tl.load(w_ptr + cols[:, None] * swn + kk[None, :] * swk, mask=cmask[:, None] & kmask[None, :], other=0.0)
            if MODE == 0 and TMA:
                dw = d_desc.load([g, ni * BN, k0]).reshape(BN, BK)
            elif MODE == 0:
                dw = _dense_delta(d_ptr, g, cols, kk, dmask, sdg, sdn, sdk)
            elif MODE == 1:
                dw, run = _dare_delta(d_ptr, run, k0, kk, dmask, m_ptr, mrow, bits, K // B, B, BN, BK, MWP)
            else:
                dw = _bit_delta(d_ptr, mrow, k0, dmask, sc, MW, BN, BK).to(w.dtype)
            acc = tl.dot(w + dw, tl.trans(a), acc)
        pair = tl.load(perm_ptr + rows, mask=rmask, other=0)
        rw = tl.load(rw_ptr + pair, mask=rmask, other=0.0)
        omask = cmask[:, None] & rmask[None, :]
        if S == 1:  # each pair's own slot; y and the routing weight rounded as in the PyTorch forward
            dt = o_ptr.dtype.element_ty
            y = acc.to(dt).to(tl.float32) * rw.to(dt).to(tl.float32)[None, :]
            tl.store(o_ptr + pair.to(tl.int64)[None, :] * som + cols[:, None] * son, y.to(o_ptr.dtype.element_ty), mask=omask)
        else:  # partial sums of all splits and slots into the token's float32 row
            y = acc * rw[None, :]
            tl.atomic_add(o_ptr + (pair // TOPK).to(tl.int64)[None, :] * som + cols[:, None] * son, y, mask=omask, sem="relaxed")


# --------------------------------------------------------------------------------------------------- host side


class Delta(NamedTuple):
    """One projection's deltas as the kernels take them. mode 0: dense (G, N, K) in d; mode 1: seeded DARE, d the
    packed kept values, p (G, N) each row's start in d, m (G, N, words) int32 the keep bit of each DARE block
    (models/dare.py mask_bits); mode 2: BitDelta, d (G, N, K / 32) int32 sign bits, p (G, N) row scales, block 32
    (models/bitdelta.py)."""

    mode: int
    d: Tensor
    p: Tensor | None = None
    m: Tensor | None = None
    block: int = 1

    def ptrs(self) -> tuple[Tensor, Tensor, int]:
        """(p, m, mask words per row); dense mode: placeholders the kernel never reads."""
        if self.mode == 0:
            return self.d, self.d, 0
        if self.mode == 1:
            return self.p, self.m, self.m.shape[-1]  # type: ignore[return-value]
        return self.p, self.d, self.d.shape[-1]  # type: ignore[return-value]

    def mwp(self) -> int:
        """Mask words per row padded to a power of 2 when at most 8 (kept in registers), else 0 (read per tile)."""
        mw = self.ptrs()[2]
        return triton.next_power_of_2(mw) if self.mode == 1 and 0 < mw <= 8 else 0

    def strides(self) -> tuple[int, int, int]:
        return tuple(self.d.stride()) if self.mode == 0 else (0, 0, 0)  # type: ignore[return-value]


_SMS: dict[int, int] = {}


def _sms(dev: torch.device) -> int:
    if dev.index not in _SMS:
        _SMS[dev.index] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SMS[dev.index]


def _grid(dev: torch.device, P: int, G: int, N: int, S: int):
    sms = _sms(dev)

    def grid(meta):  # at most min(P, G + 1) groups, each with a ragged last row tile
        tiles = (min(triton.cdiv(P, meta["BM"]) + G + 1, P)) * triton.cdiv(N, meta["BN"]) * S
        return (max(1, min(sms * meta["OCC"], tiles)),)

    return grid, sms


def _split_k(P: int, G: int, N: int, K: int, sms: int, d: Delta) -> int:
    """Split K when there are too few output tiles to occupy the SMs (decode with one or two tokens); not for DARE
    (its rows are read from their start)."""
    if d.mode == 1:
        return 1
    tiles, s = min(P, G + 1 + triton.cdiv(P, 64)) * triton.cdiv(N, 64), 1  # row tiles (64 rows) x column tiles
    while s < 8 and tiles * s < sms and K // (2 * s) >= 1024:
        s *= 2
    return s


TMA = os.environ.get("SHARED_DELTA_MOE_TMA", "1") != "0"  # "0": pointer loads on Hopper too (for comparison)
TMA_MIN_PAIRS = 256  # H200: pointer loads win below (1 token: 0.18 vs 0.27 ms), TMA above (8192: 3.26 vs 3.49 ms)


def _tma(P: int, *ts: Tensor) -> bool:
    """TMA descriptors on Hopper and newer from TMA_MIN_PAIRS pairs (below, building the descriptors costs more than
    TMA saves), for tensors TMA can address: 16-byte aligned base and strides, contiguous last dimension, no empty
    dimension."""
    t0 = ts[0]
    if not (TMA and P >= TMA_MIN_PAIRS and t0.is_cuda and torch.cuda.get_device_capability(t0.device) >= (9, 0)):
        return False
    return all(
        t.data_ptr() % 16 == 0 and t.stride(-1) == 1 and all(s * t.element_size() % 16 == 0 for s in t.stride()[:-1])
        and all(n > 0 for n in t.shape)
        for t in ts
    )


def _desc(t: Tensor) -> TensorDescriptor:
    return TensorDescriptor.from_tensor(t, [1] * t.dim())  # box shape set per config by _set_blocks


def _bucket(P: int) -> int:
    return triton.next_power_of_2(P)


def _launch(kernel, grid, config: triton.Config | None, *args, **kw):
    """Autotuned launch, or one fixed config (tests check every config this way)."""
    if config is None:
        return kernel[grid](*args, **kw)
    return _launch_fixed(kernel, grid, config, 4 if kernel is _gate_up_kernel else 2, args, kw)


def _launch_fixed(kernel, grid, config: triton.Config, tiles: int, args: tuple, kw: dict, rows: int = 1):
    """One config, its pipeline depth fitted to shared memory (tiles, rows: see _fit)."""
    fitted = _fit(config, tiles, _smem_limit(args[0]), kw.get("TMA", False), rows)
    if fitted is None:
        raise ValueError(f"config does not fit (shared memory, or a 256-wide TMA box): {config}")
    meta = fitted.all_kwargs()
    _set_blocks({**dict(zip(kernel.arg_names, args)), **kw, **meta})
    return kernel.fn[grid(meta)](*args, **kw, **meta)


def gate_up(x: Tensor, r: Routing, wg: Tensor, wu: Tensor, dg: Delta, du: Delta,
            config: triton.Config | None = None) -> Tensor:  # fmt: skip
    """h (P, I) in sorted pair order."""
    P, (N, K), G = r.perm.numel(), wg.shape, r.n_delta
    h = torch.empty(P, N, dtype=x.dtype, device=x.device)
    grid, _ = _grid(x.device, P, G, N, 1)
    tma = dg.mode == 0 and _tma(P, wg, wu, dg.d, du.d)  # DARE: pointer loads (TMA on W_b alone: 2.3x slower, H200)
    descs = (_desc(wg), _desc(wu), *((_desc(dg.d), _desc(du.d)) if dg.mode == 0 else (None, None))) if tma else (None,) * 4
    with torch.cuda.device(x.device):
        _launch(_gate_up_kernel, grid, config,
            x, r.perm, r.gstart, wg, wu, dg.d, du.d, dg.ptrs()[0], du.ptrs()[0], dg.ptrs()[1], du.ptrs()[1], h, *descs,
            N, K, G, _bucket(P), dg.ptrs()[2],
            *x.stride(), *wg.stride(), *wu.stride(), *dg.strides(), *du.strides(), *h.stride(),
            TOPK=r.top_k, MODE=dg.mode, B=dg.block, S=1, GP=triton.next_power_of_2(G + 2), TMA=tma, MWP=dg.mwp(),
        )  # fmt: skip
    return h


def down(h: Tensor, r: Routing, w: Tensor, d: Delta, config: triton.Config | None = None,
         split_k: int | None = None) -> Tensor:  # fmt: skip
    """(T, H): sum over slots of r_i * h_i (W_d + D_d)^T."""
    P, (N, K), G, T = h.shape[0], w.shape, r.n_delta, r.tokens
    S = split_k or _split_k(P, G, N, K, _sms(h.device), d)
    assert S == 1 or d.mode != 1, "DARE rows are read from their start: no split-K"
    grid, _ = _grid(h.device, P, G, N, S)
    o = torch.empty(P, N, dtype=h.dtype, device=h.device) if S == 1 else torch.zeros(T, N, dtype=torch.float32, device=h.device)
    tma = d.mode == 0 and _tma(P, h, w, d.d)
    descs = (_desc(h), _desc(w), _desc(d.d) if d.mode == 0 else None) if tma else (None,) * 3
    with torch.cuda.device(h.device):
        _launch(_down_kernel, grid, config,
            h, r.perm, r.gstart, r.weight, w, d.d, *d.ptrs()[:2], o, *descs,
            N, K, G, _bucket(P), d.ptrs()[2],
            *h.stride(), *w.stride(), *d.strides(), *o.stride(),
            TOPK=r.top_k, MODE=d.mode, B=d.block, S=S, GP=triton.next_power_of_2(G + 2), TMA=tma, MWP=d.mwp(),
        )  # fmt: skip
    if S > 1:
        return o.to(h.dtype)
    return o if r.top_k == 1 else o.view(T, r.top_k, N).sum(1).to(h.dtype)  # (autocast would make the sum fp32)


def fused_moe(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, base: tuple[Tensor, Tensor, Tensor],
              deltas: tuple[Delta, Delta, Delta]) -> Tensor:  # fmt: skip
    """Parameter-aggregation MoE forward: x (T, H); idx / rw (T, k) expert and routing weight per slot (>= n_delta:
    a zero expert); base = (W_g, W_u, W_d)."""
    if x.shape[0] == 0:
        return x.new_empty(x.shape)
    r = route(idx, rw, n_delta)
    return down(gate_up(x, r, base[0], base[1], deltas[0], deltas[1]), r, base[2], deltas[2])


# ------------------------------------------------------------------------------------------- torch.compile ops
# torch.compile sees the forward as one opaque op (the Triton launches, autotuning and descriptors stay in eager
# Python), so compiled models keep it in one graph and mode="reduce-overhead" captures it in a CUDA graph. Eager
# calls skip the dispatcher (its overhead matters at decode).


@torch.library.custom_op("shared_delta_moe::fused_moe_dense", mutates_args=())
def fused_moe_dense(x: Tensor, idx: Tensor, rw: Tensor, wg: Tensor, wu: Tensor, wd: Tensor, dg: Tensor, du: Tensor,
                    dd: Tensor, n_delta: int) -> Tensor:  # fmt: skip
    return fused_moe(x, idx, rw, n_delta, (wg, wu, wd), (Delta(0, dg), Delta(0, du), Delta(0, dd)))


@torch.library.custom_op("shared_delta_moe::fused_moe_dare", mutates_args=())
def fused_moe_dare(x: Tensor, idx: Tensor, rw: Tensor, wg: Tensor, wu: Tensor, wd: Tensor, vg: Tensor, vu: Tensor,
                   vd: Tensor, pg: Tensor, pu: Tensor, pd: Tensor, mg: Tensor, mu: Tensor, md: Tensor, n_delta: int,
                   block: int) -> Tensor:  # fmt: skip
    deltas = tuple(Delta(1, v, p, m, block) for v, p, m in ((vg, pg, mg), (vu, pu, mu), (vd, pd, md)))
    return fused_moe(x, idx, rw, n_delta, (wg, wu, wd), deltas)  # type: ignore[arg-type]


@torch.library.custom_op("shared_delta_moe::fused_moe_bit", mutates_args=())
def fused_moe_bit(x: Tensor, idx: Tensor, rw: Tensor, wg: Tensor, wu: Tensor, wd: Tensor, bg: Tensor, bu: Tensor,
                  bd: Tensor, sg: Tensor, su: Tensor, sd: Tensor, n_delta: int) -> Tensor:  # fmt: skip
    deltas = tuple(Delta(2, b, s, block=32) for b, s in ((bg, sg), (bu, su), (bd, sd)))
    return fused_moe(x, idx, rw, n_delta, (wg, wu, wd), deltas)  # type: ignore[arg-type]


@fused_moe_dense.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)  # contiguous, as the op returns (empty_like would keep x's strides)


@fused_moe_dare.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)


@fused_moe_bit.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)


def fused_moe_any(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, base: tuple[Tensor, Tensor, Tensor],
                  deltas: tuple[Delta, Delta, Delta]) -> Tensor:  # fmt: skip
    """fused_moe, through the registered op while torch.compile traces."""
    if not torch.compiler.is_compiling():
        return fused_moe(x, idx, rw, n_delta, base, deltas)
    x = x.contiguous()  # Inductor may hand the op another layout of x than the one it traced
    if deltas[0].mode == 0:
        return fused_moe_dense(x, idx, rw, *base, *(d.d for d in deltas), n_delta)
    if deltas[0].mode == 2:
        return fused_moe_bit(x, idx, rw, *base, *(d.d for d in deltas), *(d.p for d in deltas), n_delta)  # type: ignore[misc]
    d0 = deltas[0]
    return fused_moe_dare(x, idx, rw, *base, *(d.d for d in deltas), *(d.p for d in deltas),  # type: ignore[misc]
                          *(d.m for d in deltas), n_delta, d0.block)  # fmt: skip
