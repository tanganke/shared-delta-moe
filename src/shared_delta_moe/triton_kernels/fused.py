r"""Fused parameter-aggregation MoE in three launches after the routing, with no host synchronisation:

sort
    One program: a stable counting sort of the (token, slot) pairs by group (delta :math:`e` to group :math:`e`,
    every zero expert to group :math:`G`), for a routing chosen in torch (``DeltaMoE.topk``: the same choice as every
    other forward and as the autograd path, ties included). Writes ``perm`` (sorted position to pair) and ``gstart``
    (:math:`G + 2` group starts in sorted order).
gate_up
    :math:`h = \mathrm{SiLU}(x (W_g + D_g)^\top) \odot (x (W_u + D_u)^\top)` for every sorted pair, the rows of
    :math:`x` gathered through ``perm``, the sums :math:`W + D` formed tile by tile in registers; :math:`h` is written
    in sorted order.
down
    :math:`y = h (W_d + D_d)^\top`, scaled by the routing weight and written to the pair's slot (split-K: atomically
    added to the token's float32 row, which also sums the top-k slots).

The GEMM kernels are persistent: each program walks tiles (group, row tile, column tile, K split) with a grouped
order, so programs running at the same time share weight and activation tiles in L2. The tile list is derived on
the device from ``gstart``, which is what lets a whole forward run under a CUDA graph.

Deltas come in four storage modes (``MODE``):

- 0: dense, (G, N, K).
- 1: seeded DARE (``models/dare.py``), the keep bits of each tile read from a 1-bit-per-block mask bitmap and the
  kept values gathered from the packed storage.
- 2: BitDelta (``models/bitdelta.py``), sign bits expanded to :math:`\pm` scale per tile, the scales one per row or
  per block.
- 3: group grids (``models/grid.py``), 2-bit codes :math:`q` expanded to :math:`\ell + s q` per tile, one step
  :math:`s` and lowest level :math:`\ell` per group of entries along K.

On Hopper (sm90+) with dense deltas the weight tiles are loaded with TMA (tensor descriptors), so that address
arithmetic and masking leave the main loop. Row offsets (row index times row stride) are int64: :math:`h`
(pairs :math:`\times` inter) passes :math:`2^{31}` elements in training.
"""

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.tools.tensor_descriptor import TensorDescriptor

from .. import constants
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
    r"""A routing sorted by group, as ``route`` returns it; :math:`P` = tokens :math:`\cdot` top_k pairs,
    :math:`G` = ``n_delta``.

    Attributes:
        perm: (P,) int32, sorted position -> pair (token * top_k + slot).
        gstart: (G + 2,) int32, start of each group in sorted order; ``gstart[G + 1]`` = :math:`P`.
        weight: (P,) float32, renormalised routing weight of each pair.
        tokens: T, number of tokens.
        top_k: k, slots per token.
        n_delta: :math:`G`, number of delta experts (every zero expert is group :math:`G`).
    """

    perm: Tensor  # (P,) int32, sorted position -> pair (token * top_k + slot)
    gstart: Tensor  # (G + 2,) int32, start of each group in sorted order; [G + 1] = P
    weight: Tensor  # (P,) float32, renormalised routing weight of each pair
    tokens: int
    top_k: int
    n_delta: int


def route(idx: Tensor, weight: Tensor, n_delta: int) -> Routing:
    """Sort a routing by group, on the device in one program (a stable counting sort, no host synchronisation).

    Args:
        idx: (T, k) expert per slot; ``idx >= n_delta``: a zero expert.
        weight: (T, k) routing weight of each slot.
        n_delta: :math:`G`, number of delta experts.

    Returns:
        The Routing: delta :math:`e` to group :math:`e`, every zero expert to group :math:`G`.
    """
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
def _bit_delta(b_ptr, s_ptr, s_row, row, k0, K, dmask, MW, SB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """BitDelta tile, float32: entry c of a row is +s if bit c % 32 of its word c // 32 is set, else -s, s the row's
    scale c // SB (row (BN, 1): the rows' index in (G, N); MW sign words, cdiv(K, SB) scales per row; BK a multiple
    of 32, SB a power of 2). SB = 0: one scale per row, s_row (BN,), loaded once before the K loop."""
    W: tl.constexpr = BK // 32
    wj = k0 // 32 + tl.arange(0, W)[None, :]
    rm = tl.max(dmask.to(tl.int32), axis=1)[:, None] > 0  # rows with a delta
    w = tl.load(b_ptr + row * MW + wj, mask=rm & (wj < MW), other=0)  # (BN, W)
    bit = (w[:, :, None] >> tl.arange(0, 32)[None, None, :]) & 1
    sign = (tl.reshape(bit, (BN, BK)) * 2 - 1).to(tl.float32)
    if SB == 0:
        s = s_row[:, None]
    elif SB >= BK:  # one scale per row for the whole tile
        s = tl.load(s_ptr + row * tl.cdiv(K, SB) + k0 // SB, mask=rm, other=0.0).to(tl.float32)  # (BN, 1)
    else:  # BK / SB scales per row, each repeated over its block
        J: tl.constexpr = BK // SB
        NB = tl.cdiv(K, SB)
        j = k0 // SB + tl.arange(0, J)[None, :]
        s = tl.load(s_ptr + row * NB + j, mask=rm & (j < NB), other=0.0).to(tl.float32)  # (BN, J)
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (BN, J, SB)), (BN, BK))
    return tl.where(dmask, sign * s, 0.0)


@triton.jit
def _grid_delta(c_ptr, s_ptr, z_ptr, row, k0, K, dmask, MW, SB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Group-grid tile, float32: entry c of a row is z + s q, q the 2-bit code in bits 2 (c % 16) of word c // 16, s / z
    the step / lowest level of group c // SB (row (BN, 1): the rows' index in (G, N); MW code words, K / SB groups per
    row; BK a multiple of 16, SB a power of 2)."""
    W: tl.constexpr = BK // 16
    wj = k0 // 16 + tl.arange(0, W)[None, :]
    rm = tl.max(dmask.to(tl.int32), axis=1)[:, None] > 0  # rows with a delta
    w = tl.load(c_ptr + row * MW + wj, mask=rm & (wj < MW), other=0)  # (BN, W)
    q = tl.reshape((w[:, :, None] >> (tl.arange(0, 16) * 2)[None, None, :]) & 3, (BN, BK)).to(tl.float32)
    NB = K // SB
    if SB >= BK:  # one group for the whole tile
        s = tl.load(s_ptr + row * NB + k0 // SB, mask=rm, other=0.0).to(tl.float32)  # (BN, 1)
        z = tl.load(z_ptr + row * NB + k0 // SB, mask=rm, other=0.0).to(tl.float32)
    else:  # BK / SB groups per row, each repeated over its entries
        J: tl.constexpr = BK // SB
        j = k0 // SB + tl.arange(0, J)[None, :]
        s = tl.load(s_ptr + row * NB + j, mask=rm & (j < NB), other=0.0).to(tl.float32)  # (BN, J)
        z = tl.load(z_ptr + row * NB + j, mask=rm & (j < NB), other=0.0).to(tl.float32)
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (BN, J, SB)), (BN, BK))
        z = tl.reshape(tl.broadcast_to(z[:, :, None], (BN, J, SB)), (BN, BK))
    return tl.where(dmask, z + s * q, 0.0)


# ------------------------------------------------------------------------------------------------ GEMM kernels


def _set_blocks(nargs):
    """TMA descriptors take the box shape of the config being launched (autotune pre_hook)."""
    bm, bn, bk = nargs["BM"], nargs["BN"], nargs["BK"]
    for name, shape in (("wg_desc", [bn, bk]), ("wu_desc", [bn, bk]), ("w_desc", [bn, bk]), ("h_desc", [bm, bk]),
                        ("x_desc", [bm, bk]), ("dg_desc", [1, bn, bk]), ("du_desc", [1, bn, bk]), ("d_desc", [1, bn, bk])):  # fmt: skip
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
               if c.kwargs["BM"] <= max(16, pb) and (b == 0 or c.kwargs["BK"] % b == 0 or b % c.kwargs["BK"] == 0)]  # fmt: skip
        out = [c for c in out if c is not None]
        assert out, "no config fits"
        return out[: None if AUTOTUNE else 1]

    return prune


_KEY = ["N", "K", "P_BUCKET", "TOPK", "MODE", "B", "S", "TMA"]


@triton.autotune(configs=GATE_UP_CONFIGS, key=_KEY, prune_configs_by={"early_config_prune": _pruner(4)},
                 cache_results=True)  # fmt: skip
@triton.jit
def _gate_up_kernel(
    x_ptr, perm_ptr, gstart_ptr, wg_ptr, wu_ptr, dg_ptr, du_ptr, pg_ptr, pu_ptr, mg_ptr, mu_ptr, h_ptr,
    x_desc, wg_desc, wu_desc, dg_desc, du_desc,
    N, K, G, P_BUCKET, MW,
    sxm, sxk, swn, swk, sun, suk, sdg, sdn, sdk, seg, sen, sek, shm, shn,
    TOPK: tl.constexpr, MODE: tl.constexpr, B: tl.constexpr, S: tl.constexpr, GP: tl.constexpr, TMA: tl.constexpr,
    MWP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, OCC: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
    """sw* / su*: strides of W_g / W_u; sd* / se*: of D_g / D_u (dense mode). TMA: x_desc over x gathered in sorted
    pair order (rows past the group's end belong to the next group: computed, not stored)."""
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
        if MODE == 1:
            rg = tl.load(pg_ptr + g * N + cols, mask=dcols, other=0)
            ru = tl.load(pu_ptr + g * N + cols, mask=dcols, other=0)
        sg = tl.zeros((BN,), tl.float32)
        su = tl.zeros((BN,), tl.float32)
        if MODE == 2 and B == 0:  # BitDelta, one scale per row
            sg = tl.load(pg_ptr + g * N + cols, mask=dcols, other=0.0).to(tl.float32)
            su = tl.load(pu_ptr + g * N + cols, mask=dcols, other=0.0).to(tl.float32)
        brow = (g * N + cols).to(tl.int64)[:, None]  # BitDelta: rows of the sign words and scales (p)
        mrow = (g * N + cols).to(tl.int64)[:, None] * MW  # DARE: mask words of each row
        wg_bits = _mask_words(mg_ptr, mrow, dcols, MW, MWP)
        wu_bits = _mask_words(mu_ptr, mrow, dcols, MW, MWP)
        acc_g = tl.zeros((BN, BM), tl.float32)  # y^T: the summed weight is operand A
        acc_u = tl.zeros((BN, BM), tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + ks
            kmask = kk < K
            wmask = cmask[:, None] & kmask[None, :]
            dmask = dcols[:, None] & kmask[None, :]
            if TMA:  # out-of-bounds boxes read as zeros: ragged N / K, and the zero expert (g = G) in the deltas
                a = x_desc.load([row0, k0])
                wg = wg_desc.load([ni * BN, k0])
                wu = wu_desc.load([ni * BN, k0])
            else:
                a = tl.load(x_ptr + tok.to(tl.int64)[:, None] * sxm + kk[None, :] * sxk, mask=rmask[:, None] & kmask[None, :], other=0.0)
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
            elif MODE == 2:  # +-scale is exact in the model dtype, so W_b + delta rounds as in the materialised forward
                dg = _bit_delta(dg_ptr, pg_ptr, sg, brow, k0, K, dmask, MW, B, BN, BK).to(wg.dtype)
                du = _bit_delta(du_ptr, pu_ptr, su, brow, k0, K, dmask, MW, B, BN, BK).to(wu.dtype)
            else:  # z + s q rounded once to the model dtype, as the materialised forward's addcmul
                dg = _grid_delta(dg_ptr, pg_ptr, mg_ptr, brow, k0, K, dmask, MW, B, BN, BK).to(wg.dtype)
                du = _grid_delta(du_ptr, pu_ptr, mu_ptr, brow, k0, K, dmask, MW, B, BN, BK).to(wu.dtype)
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
                 reset_to_zero=["o_ptr"], cache_results=True)  # fmt: skip
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
        if MODE == 1:
            run = tl.load(p_ptr + g * N + cols, mask=dcols, other=0)
        sc = tl.zeros((BN,), tl.float32)
        if MODE == 2 and B == 0:
            sc = tl.load(p_ptr + g * N + cols, mask=dcols, other=0.0).to(tl.float32)
        brow = (g * N + cols).to(tl.int64)[:, None]
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
            elif MODE == 2:
                dw = _bit_delta(d_ptr, p_ptr, sc, brow, k0, K, dmask, MW, B, BN, BK).to(w.dtype)
            else:
                dw = _grid_delta(d_ptr, p_ptr, m_ptr, brow, k0, K, dmask, MW, B, BN, BK).to(w.dtype)
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
    """One projection's deltas as the kernels take them, for G delta experts of (N, K) matrices.

    :ivar mode: Storage mode. 0: dense; 1: seeded DARE (``models/dare.py``); 2: BitDelta (``models/bitdelta.py``);
        3: 2-bit group grids (``models/grid.py``).
    :ivar d: Mode 0: (G, N, K) dense deltas. Mode 1: the packed kept values. Mode 2: (G, N, K / 32) int32 sign bits.
        Mode 3: (G, N, K / 16) int32 2-bit codes.
    :ivar p: Mode 1: (G, N) each row's start in d. Mode 2: (G, N, cdiv(K, block)) scales. Mode 3: (G, N, K / block)
        steps. Unused in mode 0.
    :ivar m: Mode 1: (G, N, words) int32, the keep bit of each DARE block (``models/dare.py`` ``mask_bits``). Mode 3:
        (G, N, K / block) lowest levels. Unused otherwise.
    :ivar block: Mode 1: the DARE block size. Mode 2: the entries per scale (a power of 2), 0 for one scale per row.
        Mode 3: the grid group (a power of 2 dividing K).
    """

    mode: int
    d: Tensor
    p: Tensor | None = None
    m: Tensor | None = None
    block: int = 1

    def ptrs(self) -> tuple[Tensor, Tensor, int]:
        """Pointer arguments of the kernels.

        Returns:
            (p, m, mask words per row): mode 1 (p, m, m's words per row); mode 2 (p, d, d's words per row); dense
            mode: placeholders the kernel never reads (d, d, 0).
        """
        if self.mode == 0:
            return self.d, self.d, 0
        if self.mode == 1:
            return self.p, self.m, self.m.shape[-1]  # type: ignore[return-value]
        if self.mode == 3:
            return self.p, self.m, self.d.shape[-1]  # type: ignore[return-value]
        return self.p, self.d, self.d.shape[-1]  # type: ignore[return-value]

    def mwp(self) -> int:
        """Mask words per row as the kernels keep them.

        Returns:
            Mode 1 with at most 8 words per row: the count padded to a power of 2 (kept in registers); else 0 (read
            per tile).
        """
        mw = self.ptrs()[2]
        return triton.next_power_of_2(mw) if self.mode == 1 and 0 < mw <= 8 else 0

    def strides(self) -> tuple[int, int, int]:
        """Strides of the dense deltas.

        Returns:
            d's (G, N, K) strides in mode 0, else (0, 0, 0).
        """
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
    """Autotune key for P pairs: powers of 4 from 16, so a server meets few new keys after its warmup (each new key
    tunes every config, seconds to minutes per projection shape)."""
    b = 16
    while b < P:
        b *= 4
    return b


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
    r"""First GEMM, for every sorted pair:

    .. math::

       h = \mathrm{SiLU}(x (W_g + D_g)^\top) \odot (x (W_u + D_u)^\top).


    Args:
        x: (T, H) tokens; each pair's row is gathered through ``r.perm``.
        r: The sorted routing (``route``).
        wg: (I, H) base gate weight :math:`W_g`.
        wu: (I, H) base up weight :math:`W_u`.
        dg: Gate deltas :math:`D_g`, (G, I, H) as a Delta.
        du: Up deltas :math:`D_u`, same mode as dg.
        config: One fixed Triton config instead of autotuning.

    Returns:
        :math:`h` (P, I) in sorted pair order, in x's dtype.

    Raises:
        ValueError: If config does not fit (shared memory, or a 256-wide TMA box).
    """
    P, (N, K), G = r.perm.numel(), wg.shape, r.n_delta
    h = torch.empty(P, N, dtype=x.dtype, device=x.device)
    grid, _ = _grid(x.device, P, G, N, 1)
    tma = dg.mode == 0 and _tma(P, wg, wu, dg.d, du.d)  # DARE: pointer loads (TMA on W_b alone: 2.3x slower, H200)
    # TMA: x gathered into sorted pair order first (P x K, a few % of the GEMM), so its tiles come in by TMA too
    descs = (_desc(x[r.perm // r.top_k]), _desc(wg), _desc(wu), _desc(dg.d), _desc(du.d)) if tma else (None,) * 5
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
    r"""Second GEMM, :math:`y = h (W_d + D_d)^\top`, scaled by each pair's routing weight and summed over the slots.

    Args:
        h: (P, I) ``gate_up`` output, in sorted pair order.
        r: The sorted routing (``route``).
        w: (H, I) base down weight :math:`W_d`.
        d: Down deltas :math:`D_d`, (G, H, I) as a Delta.
        config: One fixed Triton config instead of autotuning.
        split_k: K splits (atomic float32 adds into each token's row); None: chosen from the tile count and the
            SMs (1 for DARE, whose rows are read from their start).

    Returns:
        (T, H) in h's dtype: :math:`\sum_i r_i\, h_i (W_d + D_d)^\top` over the slots :math:`i`.

    Raises:
        AssertionError: If ``split_k > 1`` with DARE deltas.
        ValueError: If config does not fit (shared memory, or a 256-wide TMA box).
    """
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
    """Parameter-aggregation MoE forward: ``route``, ``gate_up``, ``down``; dense deltas on sm90 from
    ``DENSE_PREFILL_PAIRS_PER_GROUP`` * (``n_delta`` + 1) pairs on: ``route`` and ``cuda_kernels.dense_moe.grouped_moe``.

    Args:
        x: (T, H) tokens.
        idx: (T, k) expert per slot; ``idx >= n_delta``: a zero expert.
        rw: (T, k) routing weight per slot.
        n_delta: :math:`G`, number of delta experts.
        base: (:math:`W_g` (I, H), :math:`W_u` (I, H), :math:`W_d` (H, I)).
        deltas: (:math:`D_g`, :math:`D_u`, :math:`D_d`), one Delta per projection, all in the same mode.

    Returns:
        (T, H) MoE output in x's dtype (empty for :math:`T = 0`).
    """
    if x.shape[0] == 0:
        return x.new_empty(x.shape)
    r = route(idx, rw, n_delta)
    if deltas[0].mode:  # packed deltas at decode sizes: the prebuilt CUDA kernels, when built
        from ..cuda_kernels import packed_ops

        if packed_ops.supports(x, r.perm.numel(), base, deltas):
            return packed_ops.moe(x, r, base, deltas)
    elif idx.numel() >= constants.DENSE_PREFILL_PAIRS_PER_GROUP * (n_delta + 1):  # dense deltas, prefill sizes
        from ..cuda_kernels import dense_moe  # prefill: materialised W_b + Delta and grouped GEMMs

        ds = tuple(d.d for d in deltas)
        if dense_moe.supports(x, base, ds):  # type: ignore[arg-type]
            return dense_moe.grouped_moe(x, r, base, ds)  # type: ignore[arg-type]
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
                  bd: Tensor, sg: Tensor, su: Tensor, sd: Tensor, n_delta: int, block_gu: int,
                  block_d: int) -> Tensor:  # fmt: skip
    deltas = tuple(Delta(2, b, s, block=k) for b, s, k in ((bg, sg, block_gu), (bu, su, block_gu), (bd, sd, block_d)))
    return fused_moe(x, idx, rw, n_delta, (wg, wu, wd), deltas)  # type: ignore[arg-type]


@torch.library.custom_op("shared_delta_moe::fused_moe_grid", mutates_args=())
def fused_moe_grid(x: Tensor, idx: Tensor, rw: Tensor, wg: Tensor, wu: Tensor, wd: Tensor, cg: Tensor, cu: Tensor,
                   cd: Tensor, sg: Tensor, su: Tensor, sd: Tensor, zg: Tensor, zu: Tensor, zd: Tensor, n_delta: int,
                   group: int) -> Tensor:  # fmt: skip
    deltas = tuple(Delta(3, c, s, z, group) for c, s, z in ((cg, sg, zg), (cu, su, zu), (cd, sd, zd)))
    return grid_moe(x, idx, rw, n_delta, (wg, wu, wd), deltas)  # type: ignore[arg-type]


def grid_moe(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, base: tuple[Tensor, Tensor, Tensor],
             deltas: tuple[Delta, Delta, Delta]) -> Tensor:  # fmt: skip
    """The 2-bit grid MoE forward, chosen per call at run time (inside ``fused_moe_grid`` under torch.compile, so
    a trace at one batch size does not fix it): up to ``packed_ops.MATRIX_MAX_PAIRS`` pairs the CUDA matrix kernels
    where they run (decode: :math:`W_b` read once), else ``fused_moe``. Same arguments as ``fused_moe``."""
    from ..cuda_kernels import packed_ops

    if packed_ops.supports(x, idx.numel(), base, deltas, matrix=True):
        return packed_ops._matrix(x, idx, rw, n_delta, base, deltas)
    return fused_moe(x, idx, rw, n_delta, base, deltas)


def grid_moe_any(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, base: tuple[Tensor, Tensor, Tensor],
                 deltas: tuple[Delta, Delta, Delta]) -> Tensor:  # fmt: skip
    """``grid_moe``, through ``fused_moe_grid`` while ``torch.compile`` traces."""
    if not torch.compiler.is_compiling():
        return grid_moe(x, idx, rw, n_delta, base, deltas)
    return fused_moe_any(x, idx, rw, n_delta, base, deltas)


@fused_moe_grid.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)


@fused_moe_dense.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)  # contiguous, as the op returns (empty_like would keep x's strides)


@fused_moe_dare.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)


@fused_moe_bit.register_fake
def _(x, *args, **kwargs):
    return x.new_empty(x.shape)


def routed_deltas(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, d: Delta) -> Tensor:
    r"""Routed deltas of one linear layer, no base: :math:`\sum_i r_i \Delta_{e_i} x` per token
    (``models/routed_linear.py``), by ``route`` and the down kernel over a stride-0 zero base weight (every base load
    the same cached zero); pairs with ``idx >= n_delta`` contribute 0.

    Args:
        x: (T, C) tokens.
        idx: (T, k) delta per slot.
        rw: (T, k) routing weight per slot.
        n_delta: :math:`G`, deltas.
        d: The deltas, (G, R, C) as a Delta.

    Returns:
        (T, R) in x's dtype.
    """
    R = d.d.shape[1]
    if x.shape[0] == 0:
        return x.new_empty(0, R)
    r = route(idx, rw, n_delta)
    zero = x.new_zeros(1, 1).expand(R, x.shape[1])
    return down(x[r.perm // r.top_k], r, zero, d)


def routed_auto(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, d: Delta) -> Tensor:
    """``routed_deltas``, chosen per call at run time: up to ``ROUTED_MATRIX_MAX_PAIRS`` pairs (decode) the CUDA
    matrix kernel (``packed_ops.routed_matrix``) where it runs, else the Triton down kernel."""
    from ..cuda_kernels import packed_ops

    if idx.numel() <= ROUTED_MATRIX_MAX_PAIRS and packed_ops.supports(x, idx.numel(), (None,), (d,), matrix=True):
        return packed_ops.routed_matrix(x, idx, rw, n_delta, d)
    return routed_deltas(x, idx, rw, n_delta, d)


ROUTED_MATRIX_MAX_PAIRS = 64  # H200, Qwen3.8-27B shards: the matrix kernel wins to 32 pairs, ties at 128 (lm_head loses)


@torch.library.custom_op("shared_delta_moe::routed_deltas_grid", mutates_args=())
def routed_deltas_grid(x: Tensor, idx: Tensor, rw: Tensor, codes: Tensor, steps: Tensor, lows: Tensor, n_delta: int,
                       group: int) -> Tensor:  # fmt: skip
    return routed_auto(x, idx, rw, n_delta, Delta(3, codes, steps, lows, group))


@routed_deltas_grid.register_fake
def _(x, idx, rw, codes, *args, **kwargs):
    return x.new_empty(x.shape[0], codes.shape[1])


def routed_deltas_any(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, d: Delta) -> Tensor:
    """``routed_auto`` (2-bit grids, mode 3), through the registered op ``routed_deltas_grid`` while
    ``torch.compile`` traces (the choice is made inside, per call)."""
    if not torch.compiler.is_compiling():
        return routed_auto(x, idx, rw, n_delta, d)
    return routed_deltas_grid(x.contiguous(), idx, rw, d.d, d.p, d.m, n_delta, d.block)  # type: ignore[arg-type]


def fused_moe_any(x: Tensor, idx: Tensor, rw: Tensor, n_delta: int, base: tuple[Tensor, Tensor, Tensor],
                  deltas: tuple[Delta, Delta, Delta]) -> Tensor:  # fmt: skip
    """``fused_moe``, through the registered op while ``torch.compile`` traces.

    Eager calls run ``fused_moe`` directly (skipping the dispatcher); under ``torch.compile`` the forward is one opaque
    op (``fused_moe_dense``, ``fused_moe_dare``, ``fused_moe_bit`` or ``fused_moe_grid``, by the deltas' mode).

    Args:
        x: (T, H) tokens.
        idx: (T, k) expert per slot; ``idx >= n_delta``: a zero expert.
        rw: (T, k) routing weight per slot.
        n_delta: :math:`G`, number of delta experts.
        base: (:math:`W_g` (I, H), :math:`W_u` (I, H), :math:`W_d` (H, I)).
        deltas: (:math:`D_g`, :math:`D_u`, :math:`D_d`), one Delta per projection, all in the same mode.

    Returns:
        (T, H) MoE output in x's dtype.
    """
    if not torch.compiler.is_compiling():
        return fused_moe(x, idx, rw, n_delta, base, deltas)
    x = x.contiguous()  # Inductor may hand the op another layout of x than the one it traced
    if deltas[0].mode == 0:
        return fused_moe_dense(x, idx, rw, *base, *(d.d for d in deltas), n_delta)
    if deltas[0].mode == 3:
        return fused_moe_grid(x, idx, rw, *base, *(d.d for d in deltas), *(d.p for d in deltas),  # type: ignore[misc]
                              *(d.m for d in deltas), n_delta, deltas[0].block)  # fmt: skip
    if deltas[0].mode == 2:
        return fused_moe_bit(x, idx, rw, *base, *(d.d for d in deltas), *(d.p for d in deltas), n_delta,  # type: ignore[misc]
                             deltas[0].block, deltas[2].block)  # fmt: skip
    d0 = deltas[0]
    return fused_moe_dare(x, idx, rw, *base, *(d.d for d in deltas), *(d.p for d in deltas),  # type: ignore[misc]
                          *(d.m for d in deltas), n_delta, d0.block)  # fmt: skip
