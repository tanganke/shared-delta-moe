// Decode-size MoE kernels for packed deltas (seeded DARE, BitDelta): see packed_ops.py for the dispatch. Built ahead
// of time into a plain shared library (python -m shared_delta_moe.cuda_kernels.packed_ops), ops registered as
// torch.ops.sdm_packed.{gate_up,down} (parameter aggregation) and {matrix_gate_up,matrix_down} (output
// aggregation, the kernels at the end of the file).
//
// A CTA owns one K range (split-K across CTAs; the warp that finishes a row strip last sums the splits in a fixed
// order) and holds the routed pairs' activations for it in shared memory; each of its warps owns 16 output rows of
// one or two weight matrices (gate and up together). The pairs, sorted by group (route() in
// triton_kernels/fused.py), are the N side of mma.m16n8k16 (tiles of 8 sorted pairs); the weight rows are the M side.
// A warp walks steps (K chunk of 64, group): W_b of the chunk is read once into a ring, and per group W_b + delta_g is
// formed in registers, rounded to the model dtype, and multiplied with the pairs of group g only (the tile's other
// columns zeroed). W_b and the deltas stream in by cp.async, as many steps ahead as the ring, the delta pool and
// MAXLEAD allow (DARE's kept values take only the pool space they need), so a warp keeps several KB in flight.
//
// k permutation: lane (row r, quad q) loads 16 contiguous bytes (k = 8q .. 8q + 7 of a 32-wide sub-chunk) of its
// rows and of its pair; the two mma k-steps take k = 8q + 4s + {0, 1} as fragment slots 2q + {0, 1} and
// 8q + 4s + {2, 3} as 2q + 8 + {0, 1}, identically for both operands, so the sum over k is unchanged.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>

#include <map>
#include <type_traits>

namespace {

constexpr int WARPS = 8;     // warps per CTA, one row strip each
constexpr int KC = 64;       // K per step
constexpr int U = KC / 32;   // its 32-wide sub-chunks (one mma pair each)
constexpr int CR = 4;        // W_b ring slots (chunks) per warp
constexpr int MAXLEAD = 8;   // steps issued ahead of the one consumed
constexpr int PG = 256;      // DARE: kept-value pool per warp, 16-byte granules (one step's worst case: 256)
constexpr int XPAD = 32;     // activation rows padded by 64 bytes: no bank conflicts between neighbouring pairs
constexpr int XMAX = 32 * 1024;  // activation bytes per CTA (more: split K further)
constexpr int TAB = 4096;    // BitDelta: sign-flip masks of every byte, 256 x uint4

struct Params {
  const void* x;       // gate_up: (T, K) tokens, row perm[i] / topk; down: (P, K) h in sorted order
  const int* perm;     // (P,) sorted position -> pair
  const int* gstart;   // (G + 2,) group starts
  const float* rw;     // (P,) routing weight per pair (down)
  const void* w[2];    // (N, K) base weights
  const void* d[2];    // DARE: kept values; BitDelta: (G, N, K / 32) sign words
  const void* p[2];    // DARE: (G, N) int64 row starts; BitDelta: (G, N, nsb) scales
  const int* m[2];     // DARE: (G, N, mw) keep bitmap
  void* out;           // gate_up: (P, N) h sorted; down: (P, N) per pair id
  float* ws;           // split-K partials
  int* counter;        // split-K arrivals per strip
  int N, K, G, P, topk, mw, nsb, bs, sbs, S, cps, gather, ngmax, nw;
};


__device__ __forceinline__ unsigned su32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(su32(dst)), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(su32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(su32(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait_async() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <typename T>
struct Ty;
template <>
struct Ty<__nv_bfloat16> {
  __device__ static unsigned add2(unsigned a, unsigned b) {
    __nv_bfloat162 r = __hadd2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
    return *reinterpret_cast<unsigned*>(&r);
  }
  __device__ static void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  __device__ static __nv_bfloat16 of(float f) { return __float2bfloat16_rn(f); }
  __device__ static unsigned pack(float lo, float hi) {
    __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&r);
  }
  __device__ static float rnd(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
};
template <>
struct Ty<__half> {
  __device__ static unsigned add2(unsigned a, unsigned b) {
    __half2 r = __hadd2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
    return *reinterpret_cast<unsigned*>(&r);
  }
  __device__ static void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  __device__ static __half of(float f) { return __float2half_rn(f); }
  __device__ static unsigned pack(float lo, float hi) {
    __half2 r = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&r);
  }
  __device__ static float rnd(float f) { return __half2float(__float2half_rn(f)); }
};

// Keep mask of the 32 entries kc .. kc + 31 of a row (bit i = entry kc + i), DARE block 2^bs >= 8, from the bitmap
// word holding block kc >> bs.
__device__ __forceinline__ unsigned chunk_mask(unsigned w, int kc, int bs) {
  int blk = kc >> bs, off = blk & 31;
  if (bs >= 5) return (w >> off) & 1 ? 0xFFFFFFFFu : 0u;
  int nb = 32 >> bs, B = 1 << bs;
  unsigned f = (w >> off) & ((1u << nb) - 1), e = 0, ones = (1u << B) - 1;
  for (int i = 0; i < nb; i++)
    if ((f >> i) & 1) e |= ones << (i * B);
  return e;
}

// Kept entries of kc .. kc + KC - 1 (kc a multiple of KC) of a row, from the bitmap word holding block kc >> bs.
__device__ __forceinline__ int chunk_kept(unsigned w, int kc, int bs) {
  int blk = kc >> bs, off = blk & 31;
  if ((1 << bs) >= KC) return (w >> off) & 1 ? KC : 0;
  int nb = KC >> bs;
  return __popc((w >> off) & ((1u << nb) - 1)) << bs;
}


// cp.async.wait_group takes an immediate: all but the newest n groups complete
__device__ __forceinline__ void wait_n(int n) {
  switch (n) {
#define W_(i) \
  case i:     \
    wait_async<i>(); \
    break;
    W_(0) W_(1) W_(2) W_(3) W_(4) W_(5) W_(6) W_(7) W_(8) W_(9) W_(10) W_(11) W_(12)
#undef W_
    default:
      wait_async<0>();
  }
}
static_assert(MAXLEAD <= 12, "wait_n cases");

template <int MODE, int MATS>
struct Layout {  // per-warp shared memory, bytes
  static constexpr int W = CR * MATS * 16 * KC * 2;  // W_b ring: CR chunks x 16 rows x KC
  static constexpr int BSL = 64 * U + 128;          // BitDelta step, per matrix: per row U sign words, then 4 scales
  static constexpr int BSLOT = MATS * BSL;
  static constexpr int REC = 36;                    // DARE step record: 32 row offsets, granules used
  static constexpr int D = MODE == 1 ? PG * 16 + MAXLEAD * REC * 4 : MAXLEAD * BSLOT;
  static constexpr int fixed = W + D;
  static_assert(fixed % 16 == 0, "alignment");
};

__host__ __device__ inline int dare_extra(int mode, int mats, int ngmax, int nw) {
  return mode == 1 ? mats * ngmax * 16 * (8 + 4 * nw) : 0;  // row cursors (int64) and bitmap words
}

template <int MODE, int MATS, int NT, typename T>
__global__ void __launch_bounds__(WARPS * 32) packed_kernel(const Params p) {
  using L = Layout<MODE, MATS>;
  extern __shared__ __align__(16) unsigned char smem[];
  const int G = p.G, P = p.P, N = p.N, K = p.K;
  int* s_gs = reinterpret_cast<int*>(smem);
  int* s_gl = s_gs + G + 2;
  int* s_ng = s_gl + G + 1;
  int* s_perm = s_ng + 1;
  const int hdr = ((2 * G + 4 + P) * 4 + 15) / 16 * 16;
  for (int i = threadIdx.x; i < G + 2; i += blockDim.x) s_gs[i] = p.gstart[i];
  for (int i = threadIdx.x; i < P; i += blockDim.x) s_perm[i] = p.perm[i];
  uint4* tab = reinterpret_cast<uint4*>(smem + hdr);  // BitDelta: tab[b].j flips the halves of pair j whose bit is clear
  if (MODE == 2)
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
      unsigned m[4], nb = ~b;
      for (int j = 0; j < 4; j++) m[j] = (((nb >> (2 * j)) & 1u) << 15) | (((nb >> (2 * j + 1)) & 1u) << 31);
      tab[b] = make_uint4(m[0], m[1], m[2], m[3]);
    }
  __syncthreads();
  if (threadIdx.x == 0) {
    int n = 0;
    for (int g = 0; g <= G; g++)
      if (s_gs[g + 1] > s_gs[g]) s_gl[n++] = g;
    *s_ng = n;
  }
  __syncthreads();

  const int split = blockIdx.x % p.S, sgrp = blockIdx.x / p.S;
  const int c_lo = split * p.cps, nch = min(K / KC, c_lo + p.cps) - c_lo, xs = p.cps * KC + XPAD;
  T* X = reinterpret_cast<T*>(smem + hdr + (MODE == 2 ? TAB : 0));
  {  // the pairs' activations over this K range, shared by the CTA's warps (cp.async group 0)
    const T* x = reinterpret_cast<const T*>(p.x);
    const int gpr = nch * KC / 8;
    for (int i = threadIdx.x; i < NT * 8 * gpr; i += blockDim.x) {
      int pr = i / gpr, k8 = i % gpr, ok = pr < P, row = ok ? (p.gather ? s_perm[pr] / p.topk : pr) : 0;
      cp16(X + pr * xs + k8 * 8, x + (long long)row * K + c_lo * KC + k8 * 8, ok);
    }
    commit();
  }

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, r0 = lane >> 2, q = lane & 3;
  const int strips = (N + 15) / 16, strip = sgrp * WARPS + warp, n0 = strip * 16;
  const bool active = strip < strips;
  const int ng = *s_ng, nsteps = active ? nch * ng : 0;
  unsigned char* base = smem + hdr + (MODE == 2 ? TAB : 0) + NT * 8 * xs * 2 + warp * (L::fixed + dare_extra(MODE, MATS, p.ngmax, p.nw));
  T* Wr = reinterpret_cast<T*>(base);
  unsigned char* Dr = base + L::W;
  int* rec = reinterpret_cast<int*>(Dr + PG * 16);                          // DARE: MAXLEAD records
  long long* run = reinterpret_cast<long long*>(base + L::fixed);          // DARE: (MATS, ngmax, 16) cursors
  unsigned* bits = reinterpret_cast<unsigned*>(run + MATS * p.ngmax * 16);  // DARE: (MATS, ngmax, 16, nw)
  const int wlo = MODE == 1 ? ((c_lo * KC) >> p.bs) >> 5 : 0;

  if (MODE == 1 && active) {  // the range's bitmap words, and where each row's kept values of the range start
    for (int i = lane; i < MATS * ng * 16; i += 32) {
      int mat = i / (ng * 16), gi = (i / 16) % ng, r = i % 16, g = s_gl[gi], n = n0 + r;
      int o = (mat * p.ngmax + gi) * 16 + r;
      if (g >= G || n >= N) continue;
      const int* mrow = p.m[mat] + ((long long)g * N + n) * p.mw;
      for (int w = 0; w < p.nw; w++) bits[o * p.nw + w] = wlo + w < p.mw ? mrow[wlo + w] : 0;
      int kb = c_lo * KC, blk = kb >> p.bs;
      long long kept = 0;
      for (int w = 0; w < (blk >> 5); w++) kept += __popc(mrow[w]);
      unsigned wb = mrow[blk >> 5];
      kept += __popc(wb & ((1u << (blk & 31)) - 1));
      kept <<= p.bs;
      if ((wb >> (blk & 31)) & 1) kept += kb & ((1 << p.bs) - 1);  // the range starts inside a kept block
      run[o] = reinterpret_cast<const long long*>(p.p[mat])[(long long)g * N + n] + kept;
    }
    __syncwarp();
  }

  // W_b rows of this lane's copies: rows wr + RP t, 16-byte column wk (N is a multiple of 16 here)
  constexpr int GR = KC / 8, RP = 32 / GR;  // granules per row, rows per pass
  const int wr = lane / GR, wk = lane % GR;
  const T* wsrc[MATS];
#pragma unroll
  for (int mat = 0; mat < MATS; mat++)
    wsrc[mat] = reinterpret_cast<const T*>(p.w[mat]) + (long long)min(n0 + wr, N - 1) * K + c_lo * KC + wk * 8;
  auto issue_w = [&](int c) {  // W_b chunk c; the 64-byte sub-chunks of odd rows swapped in pairs (bank conflicts)
    T* dst = Wr + (c % CR) * MATS * 16 * KC + wr * KC + (((wk >> 2) ^ (wr & 1)) * 4 + (wk & 3)) * 8;
#pragma unroll
    for (int mat = 0; mat < MATS; mat++)
#pragma unroll
      for (int t = 0; t < 16 / RP; t++)
        cp16(dst + mat * 16 * KC + RP * t * KC, wsrc[mat] + (long long)(RP * t) * K + c * KC, true);
  };

  int issued = 0, pc = 0, pgi = 0, head = 0, tail = 0;  // producer: next step (chunk pc, group pgi); DARE pool
  auto try_issue = [&](int c_cons) -> bool {           // c_cons: the chunk of the next step to consume
    if (pgi == 0 && pc >= c_cons + CR) return false;  // W_b ring full
    const int g = s_gl[pgi], kc = (c_lo + pc) * KC, mat = lane >> 4, r = lane & 15, n = n0 + r;
    const bool row = g < G && mat < MATS && n < N;
    if (MODE == 1) {
      const int o = (mat * p.ngmax + pgi) * 16 + r;
      const int cnt = row ? chunk_kept(bits[o * p.nw + ((kc >> p.bs) >> 5) - wlo], kc, p.bs) : 0;
      int gr = cnt >> 3, inc = gr;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        int v = __shfl_up_sync(0xFFFFFFFFu, inc, d);
        if (lane >= d) inc += v;
      }
      const int total = __shfl_sync(0xFFFFFFFFu, inc, 31);
      if (head - tail + total > PG) return false;  // pool full
      const int at = head + inc - gr, sl = issued % MAXLEAD;
      rec[sl * L::REC + lane] = at;
      if (lane == 0) rec[sl * L::REC + 32] = total;
      if (pgi == 0) issue_w(pc);
      if (row) {
        const T* v = reinterpret_cast<const T*>(p.d[mat]) + run[o];
        for (int i = 0; i < gr; i++) cp16(Dr + ((at + i) % PG) * 16, v + 8 * i, true);
        run[o] += cnt;
      }
      head += total;
    } else {
      if (pgi == 0) issue_w(pc);
      if (row) {
        unsigned char* dst = Dr + (issued % MAXLEAD) * L::BSLOT + mat * L::BSL;
        long long rw = (long long)g * N + n, e = rw * p.nsb + (p.sbs >= 0 ? kc >> p.sbs : 0);
        const int* b = reinterpret_cast<const int*>(p.d[mat]) + rw * p.mw + (kc >> 5);
        if (U == 4) cp16(dst + r * 16, b, true);
        else cp8(dst + r * 8, b);
        const T* sc = reinterpret_cast<const T*>(p.p[mat]) + (e & ~3LL);  // the aligned 4 scales holding e
        cp4(dst + 64 * U + r * 8, sc);
        cp4(dst + 64 * U + r * 8 + 4, sc + 2);
      }
    }
    commit();
    issued++;
    if (++pgi == ng) pgi = 0, pc++;
    return true;
  };

  float acc[MATS][NT][4], acc2[MATS][NT][4];  // two chains per tile (even / odd sub-chunks): mma latency overlaps
#pragma unroll
  for (int mat = 0; mat < MATS; mat++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[mat][j][e] = acc2[mat][j][e] = 0.f;

  while (issued < nsteps && issued < MAXLEAD && try_issue(0)) {
  }
  wait_n(issued);  // the activations (group 0)
  __syncthreads();
  if (!active) return;

  int c = 0, gi = 0;
  for (int s = 0; s < nsteps; s++) {
    wait_n(issued - s - 1);  // step s's group (its W_b chunk came with it or earlier)
    __syncwarp();
    const int g = s_gl[gi], lo = s_gs[g], hi = s_gs[g + 1], slot = c % CR, kc = (c_lo + c) * KC, sl = s % MAXLEAD;
#pragma unroll
    for (int mat = 0; mat < MATS; mat++) {
      const T* wm = Wr + (slot * MATS + mat) * 16 * KC;
      const unsigned* bm = reinterpret_cast<const unsigned*>(Dr + sl * L::BSLOT + mat * L::BSL);
      // per row half h (rows r0, r0 + 8) and sub-chunk u: DARE keep masks, and where u starts in the row's pool
      // run; BitDelta sign words and the scale in both halves
      unsigned mk[2][U] = {}, s2[2][U] = {};
      int st[2][U] = {};
      if (g < G) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int r = r0 + 8 * h;
          if (MODE == 1) {
            const int o = (mat * p.ngmax + gi) * 16 + r;
            const unsigned wd = n0 + r < N ? bits[o * p.nw + ((kc >> p.bs) >> 5) - wlo] : 0u;
            int at = rec[sl * L::REC + mat * 16 + r] * 8;  // in entries
#pragma unroll
            for (int u = 0; u < U; u++) {
              mk[h][u] = chunk_mask(wd, kc + 32 * u, p.bs);
              st[h][u] = at;
              at += __popc(mk[h][u]);
            }
          } else {
            const unsigned sw0 = bm[16 * U + 2 * r], sw1 = bm[16 * U + 2 * r + 1];
            const unsigned e0 = (unsigned)(g * N + n0 + r) * (unsigned)p.nsb + (p.sbs >= 0 ? (unsigned)(kc >> p.sbs) : 0u);
#pragma unroll
            for (int u = 0; u < U; u++) {
              mk[h][u] = bm[U * r + u];
              const unsigned i = (e0 & 3u) + (p.sbs >= 0 ? (unsigned)(((kc + 32 * u) >> p.sbs) - (kc >> p.sbs)) : 0u);
              const unsigned sw = i >= 2 ? sw1 : sw0, s1 = (i & 1) ? sw >> 16 : sw & 0xFFFF;
              s2[h][u] = s1 | (s1 << 16);
            }
          }
        }
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int col = ((u ^ (r0 & 1)) * 4 + q) * 8;  // rows r0 and r0 + 8 have the same parity
        uint4 w[2];
        w[0] = *reinterpret_cast<const uint4*>(wm + r0 * KC + col);
        w[1] = *reinterpret_cast<const uint4*>(wm + (r0 + 8) * KC + col);
        if (g < G) {
#pragma unroll
          for (int h = 0; h < 2; h++) {  // W_b + delta_g, entries kc + 32u + 8q .. + 7 of row r0 + 8h
            uint4 d;
            if (MODE == 1) {
              const unsigned e = mk[h][u];
              if (!((e >> (8 * q)) & 1)) continue;  // blocks >= 8: the lane's 8 entries are kept or dropped together
              const int off = st[h][u] + __popc(e & ((1u << (8 * q)) - 1));
              d = *reinterpret_cast<const uint4*>(Dr + ((off >> 3) % PG) * 16);
            } else {
              const uint4 t = tab[(mk[h][u] >> (8 * q)) & 0xFF];
              d = make_uint4(s2[h][u] ^ t.x, s2[h][u] ^ t.y, s2[h][u] ^ t.z, s2[h][u] ^ t.w);
            }
            w[h].x = Ty<T>::add2(w[h].x, d.x), w[h].y = Ty<T>::add2(w[h].y, d.y);
            w[h].z = Ty<T>::add2(w[h].z, d.z), w[h].w = Ty<T>::add2(w[h].w, d.w);
          }
        }
        const T* xk = X + c * KC + 32 * u + q * 8;
#pragma unroll
        for (int j = 0; j < NT; j++) {
          if (8 * j >= hi || 8 * j + 8 <= lo) continue;  // warp-uniform
          const int pr = 8 * j + r0;
          uint4 b = *reinterpret_cast<const uint4*>(xk + pr * xs);
          if (pr < lo || pr >= hi) b = make_uint4(0, 0, 0, 0);
          float* a = u & 1 ? acc2[mat][j] : acc[mat][j];
          Ty<T>::mma(a, w[0].x, w[1].x, w[0].y, w[1].y, b.x, b.y);
          Ty<T>::mma(a, w[0].z, w[1].z, w[0].w, w[1].w, b.z, b.w);
        }
      }
    }
    if (MODE == 1) tail += rec[sl * L::REC + 32];  // read before the refill below can reuse the slot
    __syncwarp();
    if (++gi == ng) gi = 0, c++;
    while (issued < nsteps && issued - s - 1 < MAXLEAD && try_issue(c)) {
    }
  }

#pragma unroll
  for (int mat = 0; mat < MATS; mat++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[mat][j][e] += acc2[mat][j][e];
  if (p.S > 1) {  // partials in a fixed layout; the last warp of the strip sums them in split order
    float4* ws = reinterpret_cast<float4*>(p.ws);
#pragma unroll
    for (int mat = 0; mat < MATS; mat++)
#pragma unroll
      for (int j = 0; j < NT; j++)
        ws[(((long long)(strip * p.S + split) * MATS + mat) * NT + j) * 32 + lane] =
            make_float4(acc[mat][j][0], acc[mat][j][1], acc[mat][j][2], acc[mat][j][3]);
    __threadfence();
    __syncwarp();  // every lane's partial is published before lane 0 counts the warp in
    int old = 0;
    if (lane == 0) old = atomicAdd(p.counter + strip, 1);
    old = __shfl_sync(0xFFFFFFFFu, old, 0);
    if (old != p.S - 1) return;
    __threadfence();
#pragma unroll
    for (int mat = 0; mat < MATS; mat++)
#pragma unroll
      for (int j = 0; j < NT; j++) {
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int sp = 0; sp < p.S; sp++) {
          float4 v = __ldcg(ws + (((long long)(strip * p.S + sp) * MATS + mat) * NT + j) * 32 + lane);
          a.x += v.x, a.y += v.y, a.z += v.z, a.w += v.w;
        }
        acc[mat][j][0] = a.x, acc[mat][j][1] = a.y, acc[mat][j][2] = a.z, acc[mat][j][3] = a.w;
      }
  }

  // epilogue, rounded where the PyTorch MLP rounds: g, u, SiLU(g), h (gate_up); y and the routing weight (down)
  T* out = reinterpret_cast<T*>(p.out);
#pragma unroll
  for (int j = 0; j < NT; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      int n = n0 + r0 + (e >> 1) * 8, pr = 8 * j + 2 * q + (e & 1);
      if (n >= N || pr >= P) continue;
      if (MATS == 2) {
        float gv = Ty<T>::rnd(acc[0][j][e]), uv = Ty<T>::rnd(acc[MATS - 1][j][e]);
        out[(long long)pr * N + n] = Ty<T>::of(Ty<T>::rnd(gv / (1.f + expf(-gv))) * uv);
      } else {
        int pid = s_perm[pr];
        out[(long long)pid * N + n] = Ty<T>::of(Ty<T>::rnd(acc[0][j][e]) * Ty<T>::rnd(p.rw[pid]));
      }
    }
}

template <int MODE, int MATS, int NT, typename T>
void launch(Params& p, const at::Tensor& like, at::Tensor& ws_keep, at::Tensor& cnt_keep) {
  auto k = packed_kernel<MODE, MATS, NT, T>;
  using L = Layout<MODE, MATS>;
  const int hdr = ((2 * p.G + 4 + p.P) * 4 + 15) / 16 * 16, nchunks = p.K / KC, strips = (p.N + 15) / 16;
  const int sgroups = (strips + WARPS - 1) / WARPS, dev = like.get_device();
  const int sms = at::cuda::getDeviceProperties(dev)->multiProcessorCount;
  const int smax = (int)at::cuda::getDeviceProperties(dev)->sharedMemPerBlockOptin;
  auto nw_of = [&](int cps) { return MODE == 1 ? ((cps * KC) >> p.bs >> 5) + 2 : 0; };
  auto smem_of = [&](int cps) {
    return hdr + (MODE == 2 ? TAB : 0) + NT * 8 * (cps * KC + XPAD) * 2 + WARPS * (L::fixed + dare_extra(MODE, MATS, p.ngmax, nw_of(cps)));
  };
  static std::map<std::pair<int, int>, int> occ;  // (device, smem) -> CTAs per SM
  auto per_sm = [&](int smem) {
    auto key = std::make_pair(dev, smem);
    auto it = occ.find(key);
    if (it != occ.end()) return it->second;
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smax));
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, WARPS * 32, smem));
    return occ[key] = n;
  };
  // K splits: the activations of a split within XMAX, then the best wave efficiency (CTAs over resident slots,
  // rounded up to whole waves), fewer splits on a tie
  int cps = 0;
  double best = -1;
  for (int S = 1; S <= nchunks; S++) {
    int c = (nchunks + S - 1) / S;
    if (cps && c == cps) continue;
    int smem = smem_of(c);
    if (NT * 8 * (c * KC + XPAD) * 2 > XMAX || smem > smax) continue;
    long long ctas = (long long)sgroups * ((nchunks + c - 1) / c), slots = (long long)max(1, per_sm(smem)) * sms;
    double eff = (double)ctas / (double)(((ctas + slots - 1) / slots) * slots);
    if (eff > best + 0.05) best = eff, cps = c;
    if (c <= 4 || eff > 0.95) break;
  }
  TORCH_CHECK(cps > 0, "sdm_packed: no K split fits in shared memory");
  p.cps = cps, p.S = (nchunks + cps - 1) / cps, p.nw = nw_of(cps);
  const int smem = smem_of(cps);
  per_sm(smem);  // sets the shared memory attribute on this device
  if (p.S > 1) {
    ws_keep = at::empty({(long long)strips * p.S * MATS * NT * 32 * 4}, like.options().dtype(at::kFloat));
    cnt_keep = at::zeros({strips}, like.options().dtype(at::kInt));
    p.ws = ws_keep.data_ptr<float>();
    p.counter = cnt_keep.data_ptr<int>();
  }
  k<<<sgroups * p.S, WARPS * 32, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MODE, int MATS, typename T>
void by_nt(Params& p, const at::Tensor& like, at::Tensor& a, at::Tensor& b) {
  int nt = (p.P + 7) / 8;
  if (nt <= 1) return launch<MODE, MATS, 1, T>(p, like, a, b);
  if (nt <= 2) return launch<MODE, MATS, 2, T>(p, like, a, b);
  if (nt <= 4) return launch<MODE, MATS, 4, T>(p, like, a, b);
  return launch<MODE, MATS, 8, T>(p, like, a, b);
}

template <int MATS>
void dispatch(int mode, Params& p, const at::Tensor& like, at::Tensor& a, at::Tensor& b) {
  bool bf = like.scalar_type() == at::kBFloat16;
  if (mode == 1) return bf ? by_nt<1, MATS, __nv_bfloat16>(p, like, a, b) : by_nt<1, MATS, __half>(p, like, a, b);
  return bf ? by_nt<2, MATS, __nv_bfloat16>(p, like, a, b) : by_nt<2, MATS, __half>(p, like, a, b);
}

int log2i(int64_t v) {
  int r = 0;
  while ((1LL << r) < v) r++;
  TORCH_CHECK((1LL << r) == v, "sdm_packed: block ", v, " is not a power of 2");
  return r;
}

void check(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.data_ptr() != nullptr && (uintptr_t)t.data_ptr() % 16 == 0,
              "sdm_packed: ", name, " must be a contiguous, 16-byte aligned CUDA tensor");
}

Params make(int64_t mode, int64_t block, int64_t topk, const at::Tensor& x, const at::Tensor& perm,
            const at::Tensor& gstart, const at::Tensor& w, const at::Tensor& d, const at::Tensor& pp,
            const at::Tensor& m) {
  TORCH_CHECK(mode == 1 || mode == 2, "sdm_packed: mode 1 (DARE) or 2 (BitDelta)");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "sdm_packed: bf16 or fp16");
  TORCH_CHECK(w.scalar_type() == x.scalar_type() && w.dim() == 2 && w.size(1) % KC == 0 && w.size(0) % 16 == 0, "sdm_packed: weights (K a multiple of 64)");
  TORCH_CHECK(perm.scalar_type() == at::kInt && gstart.scalar_type() == at::kInt, "sdm_packed: int32 routing");
  TORCH_CHECK(perm.numel() <= 64, "sdm_packed: at most 64 pairs");
  check(x, "x");
  check(w, "w");
  check(d, "d");
  Params p{};
  p.x = x.data_ptr(), p.perm = perm.data_ptr<int>(), p.gstart = gstart.data_ptr<int>();
  p.N = w.size(0), p.K = w.size(1), p.G = gstart.numel() - 2, p.P = perm.numel(), p.topk = topk;
  p.ngmax = std::min<int64_t>(p.G + 1, p.P);
  if (mode == 1) {
    TORCH_CHECK(block >= 8 && pp.scalar_type() == at::kLong && m.scalar_type() == at::kInt, "sdm_packed: DARE");
    p.bs = log2i(block), p.mw = m.size(-1), p.sbs = -1;
  } else {
    TORCH_CHECK(block == 0 || block >= 32, "sdm_packed: BitDelta scale block 0 or >= 32");
    TORCH_CHECK(d.size(-1) == p.K / 32 && pp.scalar_type() == x.scalar_type(), "sdm_packed: BitDelta");
    p.sbs = block ? log2i(block) : -1, p.mw = d.size(-1), p.nsb = pp.size(-1);
  }
  return p;
}

at::Tensor gate_up(const at::Tensor& x, const at::Tensor& perm, const at::Tensor& gstart, const at::Tensor& wg,
                   const at::Tensor& wu, const at::Tensor& dg, const at::Tensor& du, const at::Tensor& pg,
                   const at::Tensor& pu, const at::Tensor& mg, const at::Tensor& mu, int64_t mode, int64_t block,
                   int64_t topk) {
  c10::cuda::CUDAGuard guard(x.device());
  Params p = make(mode, block, topk, x, perm, gstart, wg, dg, pg, mg);
  check(wu, "wu");
  check(du, "du");
  TORCH_CHECK(x.size(1) == p.K && wu.sizes() == wg.sizes(), "sdm_packed: shapes");
  p.w[0] = wg.data_ptr(), p.w[1] = wu.data_ptr(), p.d[0] = dg.data_ptr(), p.d[1] = du.data_ptr();
  p.p[0] = pg.data_ptr(), p.p[1] = pu.data_ptr();
  p.m[0] = mode == 1 ? mg.data_ptr<int>() : nullptr, p.m[1] = mode == 1 ? mu.data_ptr<int>() : nullptr;
  p.gather = 1;
  at::Tensor h = at::empty({p.P, p.N}, x.options()), a, b;
  p.out = h.data_ptr();
  if (p.P) dispatch<2>(mode, p, x, a, b);
  return h;
}

at::Tensor down(const at::Tensor& h, const at::Tensor& perm, const at::Tensor& gstart, const at::Tensor& rw,
                const at::Tensor& w, const at::Tensor& d, const at::Tensor& pp, const at::Tensor& m, int64_t mode,
                int64_t block, int64_t topk) {
  c10::cuda::CUDAGuard guard(h.device());
  Params p = make(mode, block, topk, h, perm, gstart, w, d, pp, m);
  TORCH_CHECK(h.size(0) == p.P && h.size(1) == p.K && rw.scalar_type() == at::kFloat && rw.numel() == p.P,
              "sdm_packed: shapes");
  p.w[0] = p.w[1] = w.data_ptr(), p.d[0] = p.d[1] = d.data_ptr(), p.p[0] = p.p[1] = pp.data_ptr();
  p.m[0] = p.m[1] = mode == 1 ? m.data_ptr<int>() : nullptr;
  p.rw = rw.data_ptr<float>();
  p.gather = 0;
  at::Tensor o = at::empty({p.P, p.N}, h.options()), a, b;
  p.out = o.data_ptr();
  if (p.P) dispatch<1>(mode, p, h, a, b);
  return o;
}

// ------------------------------------------------------------------------------------- output aggregation
// forward_matrix: x W_b^T comes from cuBLAS (g0 / u0 (T, N) for gate/up, d0 (P, N) sorted for down); these kernels
// add the deltas' products. The grid's y is the group, the last one the zero expert's, which adds nothing; a CTA
// whose group has no pairs exits.
// DARE (dare_kernel): a warp owns one row and its group's pairs in passes of MT, on CUDA cores (20% of the entries
// are kept; tensor cores would multiply the dropped ones too). The row's kept block indices go to shared memory once
// (a popcount scan of the bitmap), then each lane takes 8 kept values at a time and the 8 activations they meet.
// mma_kernel (BitDelta; DARE from about 5 pairs per group): a CTA owns 16 rows, its 8 warps split K, and the pairs
// are the N side of mma.m16n8k16 in passes of 8. Lane (row r0, quad q) takes entries 8q .. 8q + 7 of each 32-wide K
// chunk of rows r0 and r0 + 8, and 16 contiguous bytes of its pair's activations there (the k permutation of
// packed_kernel). BitDelta: byte q of the sign word indexes a 256-entry table of the sign flips of 4 pairs, XORed
// into the scale. DARE: the 8 entries are kept or dropped together (blocks >= 8); a running popcount of the row's
// bitmap gives where they sit in the kept values. Partials sum in warp order.
// Rounded as _forward_matrix_grouped: g = x W_b^T + (x delta^T), both rounded, likewise u and y; h and r y as in the
// parameter-aggregation epilogue.

__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }

struct MParams {
  const void* x;      // gate_up: (T, K) tokens; down: (P, K) h, sorted
  const int* perm;    // (P,) sorted position -> pair
  const int* gstart;  // (G + 2,)
  const float* rw;    // (P,) routing weights (down)
  const void* b0[2];  // x W_b^T: gate_up (T, N) gate and up; down (P, N) sorted
  const void* d[2];   // DARE kept values; BitDelta (G, N, K / 32) sign words
  const void* p[2];   // DARE (G, N) int64 row starts; BitDelta (G, N, nsb) scales
  const int* m[2];    // DARE (G, N, mw) keep bitmap; grid (G, N, nsb) lowest levels (as T)
  void* out;          // gate_up: (P, N) h sorted; down: (P, N) r y by pair
  const float* xs;    // grid_kernel: (rows of x, K / 128) sums of x over each 128-entry block
  int N, K, G, topk, mw, nsb, bs, sbs, gather;
};

// output (sorted pair pr, row n) from the deltas' products dg (and du)
template <int MATS, typename T>
__device__ __forceinline__ void epilogue(const MParams& p, int pr, int n, float dg, float du) {
  T* out = reinterpret_cast<T*>(p.out);
  const int N = p.N;
  if (MATS == 2) {
    const long long i = (long long)(p.perm[pr] / p.topk) * N + n;
    const float gv = Ty<T>::rnd(to_f(reinterpret_cast<const T*>(p.b0[0])[i]) + Ty<T>::rnd(dg));
    const float uv = Ty<T>::rnd(to_f(reinterpret_cast<const T*>(p.b0[1])[i]) + Ty<T>::rnd(du));
    out[(long long)pr * N + n] = Ty<T>::of(Ty<T>::rnd(gv / (1.f + expf(-gv))) * uv);
  } else {
    const int pid = p.perm[pr];
    const float y = Ty<T>::rnd(to_f(reinterpret_cast<const T*>(p.b0[0])[(long long)pr * N + n]) + Ty<T>::rnd(dg));
    out[(long long)pid * N + n] = Ty<T>::of(y * Ty<T>::rnd(p.rw[pid]));
  }
}

__device__ __forceinline__ float2 f2(unsigned v, __nv_bfloat16*) {
  return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xFFFF0000u));
}
__device__ __forceinline__ float2 f2(unsigned v, __half*) { return __half22float2(*reinterpret_cast<__half2*>(&v)); }
template <typename T>
__device__ __forceinline__ float dot8(uint4 a, uint4 b) {
  float2 a0 = f2(a.x, (T*)0), a1 = f2(a.y, (T*)0), a2 = f2(a.z, (T*)0), a3 = f2(a.w, (T*)0);
  float2 b0 = f2(b.x, (T*)0), b1 = f2(b.y, (T*)0), b2 = f2(b.z, (T*)0), b3 = f2(b.w, (T*)0);
  return a0.x * b0.x + a0.y * b0.y + a1.x * b1.x + a1.y * b1.y + a2.x * b2.x + a2.y * b2.y + a3.x * b3.x + a3.y * b3.y;
}

template <int MATS, int MT, typename T>
__global__ void __launch_bounds__(WARPS * 32) dare_kernel(const MParams p) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n = blockIdx.x * WARPS + warp, g = blockIdx.y;
  const int lo = p.gstart[g], hi = p.gstart[g + 1], N = p.N, K = p.K;
  if (n >= N || lo == hi) return;
  const T* x = reinterpret_cast<const T*>(p.x);
  const int nblk = K >> p.bs;
  unsigned short* list = reinterpret_cast<unsigned short*>(smem) + warp * MATS * nblk;  // kept blocks
  int kept[MATS] = {};
  long long start[MATS] = {};
  const bool delta = g < p.G;
  if (delta) {  // both matrices' bitmap words in flight together, then a popcount scan each
    const int* mrow[MATS];
#pragma unroll
    for (int mat = 0; mat < MATS; mat++) {
      mrow[mat] = p.m[mat] + ((long long)g * N + n) * p.mw;
      start[mat] = reinterpret_cast<const long long*>(p.p[mat])[(long long)g * N + n];
    }
    int base[MATS] = {};
    for (int w0 = 0; w0 < p.mw; w0 += 32) {
      const int w = w0 + lane;
      unsigned bits[MATS];
#pragma unroll
      for (int mat = 0; mat < MATS; mat++) bits[mat] = w < p.mw ? __ldg(mrow[mat] + w) : 0u;
#pragma unroll
      for (int mat = 0; mat < MATS; mat++) {
        unsigned b = bits[mat];
        if (32 * w + 32 > nblk) b &= w * 32 < nblk ? (1u << (nblk - 32 * w)) - 1 : 0u;
        int c = __popc(b), inc = c;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          int v = __shfl_up_sync(0xFFFFFFFFu, inc, d);
          if (lane >= d) inc += v;
        }
        int at = base[mat] + inc - c;
        while (b) {
          list[mat * nblk + at++] = (unsigned short)(32 * w + __ffs(b) - 1);
          b &= b - 1;
        }
        base[mat] += __shfl_sync(0xFFFFFFFFu, inc, 31);
      }
    }
#pragma unroll
    for (int mat = 0; mat < MATS; mat++) kept[mat] = base[mat] << p.bs;  // kept entries
    __syncwarp();
  }
  constexpr int UN = 4;  // 8-value chunks per lane in flight
  for (int t0 = lo; t0 < hi; t0 += MT) {
    const int nt = min(MT, hi - t0);
    int xr[MT];
#pragma unroll
    for (int t = 0; t < MT; t++) {
      const int pr = t0 + min(t, nt - 1);
      xr[t] = (p.gather ? p.perm[pr] / p.topk : pr) * K;
    }
    float acc[MATS][MT] = {};
    if (delta) {
#pragma unroll
      for (int mat = 0; mat < MATS; mat++) {
        const T* v = reinterpret_cast<const T*>(p.d[mat]) + start[mat];
        for (int e0 = lane * 8; e0 < kept[mat]; e0 += 256 * UN) {
          uint4 a[UN];
          int k[UN];
#pragma unroll
          for (int u = 0; u < UN; u++) {
            const int e = e0 + 256 * u;
            a[u] = e < kept[mat] ? __ldg(reinterpret_cast<const uint4*>(v + e)) : make_uint4(0, 0, 0, 0);
            k[u] = e < kept[mat] ? list[mat * nblk + (e >> p.bs)] * (1 << p.bs) + (e & ((1 << p.bs) - 1)) : 0;
          }
#pragma unroll
          for (int t = 0; t < MT; t++)
            if (t < nt)
#pragma unroll
              for (int u = 0; u < UN; u++)
                acc[mat][t] += dot8<T>(a[u], __ldg(reinterpret_cast<const uint4*>(x + xr[t] + k[u])));
        }
      }
#pragma unroll
      for (int mat = 0; mat < MATS; mat++)
#pragma unroll
        for (int t = 0; t < MT; t++)
          if (t < nt)
#pragma unroll
            for (int d = 16; d; d >>= 1) acc[mat][t] += __shfl_xor_sync(0xFFFFFFFFu, acc[mat][t], d);
    }
#pragma unroll
    for (int t = 0; t < MT; t++)
      if (lane == t && t < nt) epilogue<MATS, T>(p, t0 + t, n, acc[0][t], acc[MATS - 1][t]);
  }
}

template <int MODE, int MATS, typename T>
__global__ void __launch_bounds__(WARPS * 32) mma_kernel(const MParams p) {
  __shared__ uint4 tab[256];                   // tab[b].j flips the halves of pair j whose bit is clear
  __shared__ float4 red[WARPS][MATS][32];      // per warp partials
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, r0 = lane >> 2, q = lane & 3, g = blockIdx.y;
  const int lo = p.gstart[g], hi = p.gstart[g + 1], N = p.N, K = p.K, n0 = blockIdx.x * 16;
  if (lo == hi) return;  // (uniform over the CTA)
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    unsigned m[4], nb = ~b;
    for (int j = 0; j < 4; j++) m[j] = (((nb >> (2 * j)) & 1u) << 15) | (((nb >> (2 * j + 1)) & 1u) << 31);
    tab[b] = make_uint4(m[0], m[1], m[2], m[3]);
  }
  __syncthreads();
  const T* x = reinterpret_cast<const T*>(p.x);
  const bool delta = g < p.G;
  const int mw = p.K / 32, per = (mw + WARPS - 1) / WARPS, c_lo = min(mw, warp * per), c_hi = min(mw, c_lo + per);
  const long long rw0 = (long long)g * N + n0 + r0, rw1 = rw0 + 8;
  // DARE: per matrix and row half, the bitmap and kept values of the row, and the kept blocks in words [0, cw)
  const int* mrow[MATS][2];
  const T* vrow[MATS][2];
  int cw[MATS][2] = {}, ck[MATS][2] = {};
  if (MODE == 1 && delta) {
#pragma unroll
    for (int mat = 0; mat < MATS; mat++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const long long rw = h ? rw1 : rw0;
        mrow[mat][h] = p.m[mat] + rw * p.mw;
        vrow[mat][h] = reinterpret_cast<const T*>(p.d[mat]) + reinterpret_cast<const long long*>(p.p[mat])[rw];
      }
  }
  for (int t0 = lo; t0 < hi; t0 += 8) {
    const int nt = min(8, hi - t0);
    float acc[MATS][4] = {};
    if (delta) {
      const int pr = t0 + min(r0, nt - 1);  // this lane's activation row: pair r0 of the pass
      const T* xr = x + (long long)(p.gather ? p.perm[pr] / p.topk : pr) * K + 8 * q;
#pragma unroll 4
      for (int c = c_lo; c < c_hi; c++) {
        uint4 b = __ldg(reinterpret_cast<const uint4*>(xr + 32 * c));
        if (r0 >= nt) b = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int mat = 0; mat < MATS; mat++) {
          if (MODE == 1) {  // entries 32c + 8q .. + 7 of rows r0, r0 + 8: kept or dropped together (blocks >= 8)
            const int k = 32 * c + 8 * q, blk = k >> p.bs, wi = blk >> 5;
            uint4 a[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
              if (wi < cw[mat][h]) cw[mat][h] = 0, ck[mat][h] = 0;  // a new pass: count from the start again
              while (cw[mat][h] < wi) ck[mat][h] += __popc(__ldg(mrow[mat][h] + cw[mat][h]++));
              const unsigned wd = __ldg(mrow[mat][h] + wi);
              const int rank = ck[mat][h] + __popc(wd & ((1u << (blk & 31)) - 1));
              a[h] = (wd >> (blk & 31)) & 1
                         ? __ldg(reinterpret_cast<const uint4*>(vrow[mat][h] + ((long long)rank << p.bs) + (k & ((1 << p.bs) - 1))))
                         : make_uint4(0, 0, 0, 0);
            }
            Ty<T>::mma(acc[mat], a[0].x, a[1].x, a[0].y, a[1].y, b.x, b.y);
            Ty<T>::mma(acc[mat], a[0].z, a[1].z, a[0].w, a[1].w, b.z, b.w);
            continue;
          }
          if (MODE == 3) {  // 2-bit grid: entries 32c + 8q .. + 7 are the 8 codes in half word q & 1 of word 2c + q / 2
            const unsigned* cw3 = reinterpret_cast<const unsigned*>(p.d[mat]);
            const T* st = reinterpret_cast<const T*>(p.p[mat]);
            const T* lw = reinterpret_cast<const T*>(p.m[mat]);
            const int wi = 2 * c + (q >> 1), sh = 16 * (q & 1), si = (32 * c) >> p.sbs;
            const unsigned w[2] = {__ldg(cw3 + rw0 * p.mw + wi) >> sh, __ldg(cw3 + rw1 * p.mw + wi) >> sh};
            const float sv[2] = {to_f(st[rw0 * p.nsb + si]), to_f(st[rw1 * p.nsb + si])};
            const float lv[2] = {to_f(lw[rw0 * p.nsb + si]), to_f(lw[rw1 * p.nsb + si])};
            unsigned a[2][4];
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
              for (int j = 0; j < 4; j++)  // entries 2j, 2j + 1: low, high half
                a[h][j] = Ty<T>::pack(lv[h] + sv[h] * float((w[h] >> (4 * j)) & 3u),
                                      lv[h] + sv[h] * float((w[h] >> (4 * j + 2)) & 3u));
            Ty<T>::mma(acc[mat], a[0][0], a[1][0], a[0][1], a[1][1], b.x, b.y);
            Ty<T>::mma(acc[mat], a[0][2], a[1][2], a[0][3], a[1][3], b.z, b.w);
            continue;
          }
          const unsigned* bits = reinterpret_cast<const unsigned*>(p.d[mat]);
          const unsigned short* sc = reinterpret_cast<const unsigned short*>(p.p[mat]);
          const int si = p.sbs >= 0 ? (32 * c) >> p.sbs : 0;
          const unsigned w0 = __ldg(bits + rw0 * p.mw + c), w1 = __ldg(bits + rw1 * p.mw + c);
          const unsigned s0 = __ldg(sc + rw0 * p.nsb + si), s1 = __ldg(sc + rw1 * p.nsb + si);
          const unsigned a0 = s0 | (s0 << 16), a1 = s1 | (s1 << 16);
          const uint4 m0 = tab[(w0 >> (8 * q)) & 0xFF], m1 = tab[(w1 >> (8 * q)) & 0xFF];
          Ty<T>::mma(acc[mat], a0 ^ m0.x, a1 ^ m1.x, a0 ^ m0.y, a1 ^ m1.y, b.x, b.y);
          Ty<T>::mma(acc[mat], a0 ^ m0.z, a1 ^ m1.z, a0 ^ m0.w, a1 ^ m1.w, b.z, b.w);
        }
      }
    }
#pragma unroll
    for (int mat = 0; mat < MATS; mat++) red[warp][mat][lane] = make_float4(acc[mat][0], acc[mat][1], acc[mat][2], acc[mat][3]);
    __syncthreads();
    if (warp == 0) {
      float s[MATS][4] = {};
      for (int w = 0; w < WARPS; w++)
#pragma unroll
        for (int mat = 0; mat < MATS; mat++) {
          const float4 v = red[w][mat][lane];
          s[mat][0] += v.x, s[mat][1] += v.y, s[mat][2] += v.z, s[mat][3] += v.w;
        }
#pragma unroll
      for (int e = 0; e < 4; e++) {  // C fragment: row r0 + 8 (e >> 1), pair 2q + (e & 1)
        const int j = 2 * q + (e & 1);
        if (j < nt) epilogue<MATS, T>(p, t0 + j, n0 + r0 + 8 * (e >> 1), s[0][e], s[MATS - 1][e]);
      }
    }
    __syncthreads();
  }
}

template <int MATS>
void matrix_launch(int mode, MParams& p, const at::Tensor& like, int64_t P) {
  const bool bf = like.scalar_type() == at::kBFloat16;
  auto stream = at::cuda::getCurrentCUDAStream();
  // DARE on the tensor cores from about 5 pairs per group (H200: CUDA cores win below, the mma kernel above)
  if (mode != 1 || P >= 5 * (p.G + 1)) {
    const dim3 grid(p.N / 16, p.G + 1);
    if (mode == 3)
      bf ? mma_kernel<3, MATS, __nv_bfloat16><<<grid, WARPS * 32, 0, stream>>>(p)
         : mma_kernel<3, MATS, __half><<<grid, WARPS * 32, 0, stream>>>(p);
    else if (mode == 2)
      bf ? mma_kernel<2, MATS, __nv_bfloat16><<<grid, WARPS * 32, 0, stream>>>(p)
         : mma_kernel<2, MATS, __half><<<grid, WARPS * 32, 0, stream>>>(p);
    else
      bf ? mma_kernel<1, MATS, __nv_bfloat16><<<grid, WARPS * 32, 0, stream>>>(p)
         : mma_kernel<1, MATS, __half><<<grid, WARPS * 32, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const dim3 grid((p.N + WARPS - 1) / WARPS, p.G + 1);
  const int smem = WARPS * MATS * (p.K >> p.bs) * 2;
  auto go = [&](auto k) {
    if (smem > 48 * 1024) C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    k<<<grid, WARPS * 32, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  auto by_t = [&](auto mt_t) {  // pairs per pass: at most P
    constexpr int R = decltype(mt_t)::value;
    bf ? go(dare_kernel<MATS, R, __nv_bfloat16>) : go(dare_kernel<MATS, R, __half>);
  };
  if (P <= 1) by_t(std::integral_constant<int, 1>{});
  else if (P <= 2) by_t(std::integral_constant<int, 2>{});
  else if (P <= 4) by_t(std::integral_constant<int, 4>{});
  else by_t(std::integral_constant<int, 8>{});
}

MParams mmake(int64_t mode, int64_t block, int64_t topk, const at::Tensor& x, const at::Tensor& perm,
              const at::Tensor& gstart, int64_t N, const at::Tensor& d, const at::Tensor& pp, const at::Tensor& m) {
  TORCH_CHECK(mode >= 1 && mode <= 3, "sdm_packed: matrix mode 1 (DARE), 2 (BitDelta) or 3 (2-bit grid)");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "sdm_packed: bf16 or fp16");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 32 == 0, "sdm_packed: K a multiple of 32");
  TORCH_CHECK(perm.scalar_type() == at::kInt && gstart.scalar_type() == at::kInt && perm.is_contiguous() &&
                  gstart.is_contiguous(), "sdm_packed: int32 routing");
  check(x, "x");
  check(d, "d");
  MParams p{};
  p.x = x.data_ptr(), p.perm = perm.data_ptr<int>(), p.gstart = gstart.data_ptr<int>();
  p.N = N, p.K = x.size(1), p.G = gstart.numel() - 2, p.topk = topk;
  if (mode == 1) {
    TORCH_CHECK(block >= 8 && pp.scalar_type() == at::kLong && m.scalar_type() == at::kInt && m.is_contiguous() &&
                    pp.is_contiguous(), "sdm_packed: DARE");
    p.bs = log2i(block), p.mw = m.size(-1), p.sbs = -1;
    TORCH_CHECK((p.K >> p.bs) <= 65535 && (p.K & (block - 1)) == 0, "sdm_packed: DARE blocks");
  } else if (mode == 3) {
    TORCH_CHECK(block >= 32 && (block & (block - 1)) == 0 && p.K % block == 0, "sdm_packed: grid group a power of 2 >= 32");
    TORCH_CHECK(d.scalar_type() == at::kInt && d.size(-1) == p.K / 16 && pp.scalar_type() == x.scalar_type() &&
                    m.scalar_type() == x.scalar_type() && pp.is_contiguous() && m.is_contiguous() &&
                    pp.size(-1) == p.K / block && m.sizes() == pp.sizes() && N % 16 == 0, "sdm_packed: grid");
    p.sbs = log2i(block), p.mw = d.size(-1), p.nsb = pp.size(-1);
  } else {
    TORCH_CHECK(block == 0 || block >= 32, "sdm_packed: BitDelta scale block 0 or >= 32");
    TORCH_CHECK(d.size(-1) == p.K / 32 && pp.scalar_type() == x.scalar_type() && pp.is_contiguous() &&
                    N % 16 == 0, "sdm_packed: BitDelta");
    p.sbs = block ? log2i(block) : -1, p.mw = d.size(-1), p.nsb = pp.size(-1);
  }
  return p;
}

at::Tensor matrix_gate_up(const at::Tensor& x, const at::Tensor& perm, const at::Tensor& gstart, const at::Tensor& g0,
                          const at::Tensor& u0, const at::Tensor& dg, const at::Tensor& du, const at::Tensor& pg,
                          const at::Tensor& pu, const at::Tensor& mg, const at::Tensor& mu, int64_t mode,
                          int64_t block, int64_t topk) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(g0.dim() == 2 && g0.size(0) == x.size(0) && u0.sizes() == g0.sizes() &&
                  g0.scalar_type() == x.scalar_type() && u0.scalar_type() == x.scalar_type(),
              "sdm_packed: matrix_gate_up shapes");
  check(g0, "g0");
  check(u0, "u0");
  check(du, "du");
  MParams p = mmake(mode, block, topk, x, perm, gstart, g0.size(1), dg, pg, mg);
  p.d[0] = dg.data_ptr(), p.d[1] = du.data_ptr(), p.p[0] = pg.data_ptr(), p.p[1] = pu.data_ptr();
  p.m[0] = mode != 2 ? static_cast<const int*>(mg.data_ptr()) : nullptr;
  p.m[1] = mode != 2 ? static_cast<const int*>(mu.data_ptr()) : nullptr;
  p.b0[0] = g0.data_ptr(), p.b0[1] = u0.data_ptr(), p.gather = 1;
  at::Tensor h = at::empty({perm.numel(), p.N}, x.options());
  p.out = h.data_ptr();
  if (perm.numel()) matrix_launch<2>(mode, p, x, perm.numel());
  return h;
}

at::Tensor matrix_down(const at::Tensor& h, const at::Tensor& perm, const at::Tensor& gstart, const at::Tensor& rw,
                       const at::Tensor& d0, const at::Tensor& d, const at::Tensor& pp, const at::Tensor& m,
                       int64_t mode, int64_t block, int64_t topk) {
  c10::cuda::CUDAGuard guard(h.device());
  TORCH_CHECK(d0.dim() == 2 && d0.size(0) == h.size(0) && d0.scalar_type() == h.scalar_type() &&
                  rw.scalar_type() == at::kFloat && rw.numel() == h.size(0) && perm.numel() == h.size(0),
              "sdm_packed: matrix_down shapes");
  check(d0, "d0");
  MParams p = mmake(mode, block, topk, h, perm, gstart, d0.size(1), d, pp, m);
  p.d[0] = p.d[1] = d.data_ptr(), p.p[0] = p.p[1] = pp.data_ptr();
  p.m[0] = p.m[1] = mode != 2 ? static_cast<const int*>(m.data_ptr()) : nullptr;
  p.rw = rw.data_ptr<float>(), p.b0[0] = d0.data_ptr(), p.gather = 0;
  at::Tensor o = at::empty({perm.numel(), p.N}, h.options());
  p.out = o.data_ptr();
  if (perm.numel()) matrix_launch<1>(mode, p, h, perm.numel());
  return o;
}

// grid_kernel (2-bit group grids, groups a multiple of 128): a CTA owns 16 rows, its 4 warps split K (128-entry
// blocks; the next block's codes load while one computes), the group's pairs in passes of 8 NT. Per 128-entry block b of a row (one group: step s, lowest level l), with
// X_b = sum of the pair's activations over b (xs, precomputed): delta . x = s sum q x + l X_b. The tensor cores take
// v = 1 + q / 4 (exact: q in the top mantissa bits of 1.0), so the block's mma gives P = X_b + (sum q x) / 4 and
// delta . x = 4 s P + (l - 4 s) X_b, folded in fp32. Unlike mma_kernel the delta entries are not rounded to the
// model dtype first (exact s q + l).
// k permutation: lane q takes entries 32q .. 32q + 31 of the block (words 8b + 2q, 8b + 2q + 1: one uint2 per row);
// register i of word w holds entries 16w + i and 16w + i + 8 (codes at bits 2i and 2i + 16: a shift and a lop3), and
// the activations come pre-permuted alike (grid_prep in packed_ops: x' [16w + 2i + j] = x[16w + i + 8j]). The mma's k slots 2q, 2q + 1 take register i0 = 2 (m % 4) of
// word m / 4, slots 2q + 8, 2q + 9 register i0 + 1, for both operands.
template <typename T>
struct Magic;
template <>
struct Magic<__nv_bfloat16> {
  static constexpr int SH = 5;                  // q / 4 in mantissa bits 5, 6 of bf16 1.0
  static constexpr unsigned ONE = 0x3F803F80u;
};
template <>
struct Magic<__half> {
  static constexpr int SH = 8;
  static constexpr unsigned ONE = 0x3C003C00u;
};

template <typename T, int I>
__device__ __forceinline__ unsigned qv(unsigned w) {  // codes at bits 2I, 2I + 16 -> 1 + q / 4 in both halves
  constexpr int s = 2 * I - Magic<T>::SH;
  unsigned t;
  if constexpr (s >= 0) t = w >> s;
  else t = w << (-s);
  return (t & (0x00030003u << Magic<T>::SH)) | Magic<T>::ONE;
}
template <typename T>
__device__ __forceinline__ void dec8(unsigned w, unsigned* a) {
  a[0] = qv<T, 0>(w), a[1] = qv<T, 1>(w), a[2] = qv<T, 2>(w), a[3] = qv<T, 3>(w);
  a[4] = qv<T, 4>(w), a[5] = qv<T, 5>(w), a[6] = qv<T, 6>(w), a[7] = qv<T, 7>(w);
}

// Shared-memory layout of one pipeline stage (one 128-entry block b): the codes of the CTA's 64 rows, the steps and
// lowest levels of b's group si and its neighbour (si & ~1, si | 1: cp.async needs 4-byte pieces), the pass's
// activations (rows padded to 272 bytes: the lanes' 8-byte reads of 8 pairs fall in distinct banks) and their block
// sums X_b.
template <int MATS, int NT>
struct GridStage {
  static constexpr int XR = 272, CB = MATS * 64 * 32, SB = MATS * 2 * 64 * 4, XB = NT * 8 * XR, XS = NT * 8 * 4;
  static constexpr int BYTES = CB + SB + XB + XS;
  static constexpr int STAGES = NT == 2 ? 4 : 3;
};

template <int MATS, int NT, typename T>
__global__ void __launch_bounds__(128) grid_kernel(const MParams p) {
  using L = GridStage<MATS, NT>;
  constexpr int S = L::STAGES;
  extern __shared__ __align__(16) unsigned char sm[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, r0 = lane >> 2, q = lane & 3;
  int g = 0;  // the blockIdx.y-th group with pairs (the grid has min(P, G + 1) of them)
  for (int c = blockIdx.y;; g++) {
    if (g > p.G) return;
    if (p.gstart[g + 1] > p.gstart[g] && c-- == 0) break;
  }
  const int lo = p.gstart[g], hi = p.gstart[g + 1], N = p.N, K = p.K, nc = blockIdx.x * 64, n0 = nc + warp * 16;
  const bool delta = g < p.G, rows = n0 < N;  // (rows: uniform per warp)
  const T* x = reinterpret_cast<const T*>(p.x);
  const int nb = K >> 7;
  for (int t0 = lo; t0 < hi; t0 += 8 * NT) {
    const int nt = min(8 * NT, hi - t0);
    float acc[MATS][NT][4] = {};
    if (delta) {
      // this thread's cp.async sources for block 0 (block b: + b * the per-block stride) and destinations
      constexpr int NX = (NT * 8 * 16 + 127) / 128;  // activation chunks per thread
      const char* csrc[MATS];
      const T* ssrc[MATS];
      int cdst[MATS], sdst[MATS];
      bool cok[MATS];
#pragma unroll
      for (int i = 0; i < MATS; i++) {
        const int c = threadIdx.x + 128 * i, mat = c >> 7, row = (c & 127) >> 1, n = min(nc + row, N - 1);
        csrc[i] = reinterpret_cast<const char*>(p.d[mat]) + ((long long)g * N + n) * p.mw * 4 + (c & 1) * 16;
        cdst[i] = mat * 2048 + row * 32 + (c & 1) * 16, cok[i] = nc + row < N;
        ssrc[i] = reinterpret_cast<const T*>((c & 1) ? p.m[mat] : p.p[mat]) + ((long long)g * N + n) * p.nsb;
        sdst[i] = L::CB + ((mat * 2 + (c & 1)) * 64 + row) * 4;
      }
      const T* xsrc[NX];
      int xdst[NX];
#pragma unroll
      for (int i = 0; i < NX; i++) {
        const int c = threadIdx.x + 128 * i, pr = t0 + min(c >> 4, nt - 1);
        xsrc[i] = x + (long long)(p.gather ? p.perm[pr] / p.topk : pr) * K + (c & 15) * 8;
        xdst[i] = c < nt * 16 ? L::CB + L::SB + (c >> 4) * L::XR + (c & 15) * 16 : -1;
      }
      const float* ssum = nullptr;
      if (threadIdx.x < nt) {
        const int pr = t0 + threadIdx.x;
        ssum = p.xs + (long long)(p.gather ? p.perm[pr] / p.topk : pr) * nb;
      }
      auto issue = [&](int b) {  // block b into its stage (one commit group per block, maybe empty)
        if (b < nb) {
          unsigned char* st = sm + (b % S) * L::BYTES;
#pragma unroll
          for (int i = 0; i < MATS; i++) {
            cp16(st + cdst[i], csrc[i] + b * 32, cok[i]);
            cp4(st + sdst[i], ssrc[i] + (((b << 7) >> p.sbs) & ~1));  // groups si & ~1, si | 1
          }
#pragma unroll
          for (int i = 0; i < NX; i++)
            if (xdst[i] >= 0) cp16(st + xdst[i], xsrc[i] + (b << 7), true);
          if (ssum) cp4(st + L::CB + L::SB + L::XB + threadIdx.x * 4, ssum + b);
        }
        commit();
      };
#pragma unroll
      for (int u = 0; u < S - 1; u++) issue(u);
#pragma unroll 1
      for (int b = 0; b < nb; b++) {
        wait_async<S - 2>();  // block b landed (every block commits one group)
        __syncthreads();      // ... for every thread; and block b - 1's stage is free again
        issue(b + S - 1);
        if (!rows) continue;
        const unsigned char* st = sm + (b % S) * L::BYTES;
        uint2 w[MATS][2];
        float sv[MATS][2], zv[MATS][2];
#pragma unroll
        for (int mat = 0; mat < MATS; mat++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int row = warp * 16 + r0 + 8 * h;
            w[mat][h] = *reinterpret_cast<const uint2*>(st + mat * 2048 + row * 32 + q * 8);
            const int si = ((b << 7) >> p.sbs) & 1;  // the block's group (of the two staged)
            const T* sp = reinterpret_cast<const T*>(st + L::CB + (mat * 2 * 64 + row) * 4) + si;
            const T* lp = reinterpret_cast<const T*>(st + L::CB + ((mat * 2 + 1) * 64 + row) * 4) + si;
            sv[mat][h] = 4.f * to_f(*sp), zv[mat][h] = to_f(*lp) - sv[mat][h];
          }
        float P[MATS][NT][4] = {};
        const unsigned char* xb = st + L::CB + L::SB + r0 * L::XR + q * 64;
#pragma unroll
        for (int m = 0; m < 8; m++) {
          const int wd = m >> 2;  // word: entries 0..15 or 16..31 of the lane's 32
          unsigned a[MATS][2][2];
#pragma unroll
          for (int mat = 0; mat < MATS; mat++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
              const unsigned v = wd ? w[mat][h].y : w[mat][h].x;
              switch (m & 3) {  // registers i0 = 2 (m & 3), i0 + 1 of the word
                case 0: a[mat][h][0] = qv<T, 0>(v), a[mat][h][1] = qv<T, 1>(v); break;
                case 1: a[mat][h][0] = qv<T, 2>(v), a[mat][h][1] = qv<T, 3>(v); break;
                case 2: a[mat][h][0] = qv<T, 4>(v), a[mat][h][1] = qv<T, 5>(v); break;
                default: a[mat][h][0] = qv<T, 6>(v), a[mat][h][1] = qv<T, 7>(v);
              }
            }
#pragma unroll
          for (int j = 0; j < NT; j++) {
            if (8 * j >= nt) break;  // (uniform)
            const uint2 bb = *reinterpret_cast<const uint2*>(xb + j * 8 * L::XR + wd * 32 + (m & 3) * 8);
#pragma unroll
            for (int mat = 0; mat < MATS; mat++)
              Ty<T>::mma(P[mat][j], a[mat][0][0], a[mat][1][0], a[mat][0][1], a[mat][1][1], bb.x, bb.y);
          }
        }
        const float* xsb = reinterpret_cast<const float*>(st + L::CB + L::SB + L::XB);
#pragma unroll
        for (int j = 0; j < NT; j++) {
          if (8 * j >= nt) break;
          const float X0 = xsb[8 * j + 2 * q], X1 = xsb[8 * j + 2 * q + 1];  // (pairs past nt: unused)
#pragma unroll
          for (int mat = 0; mat < MATS; mat++) {
            acc[mat][j][0] += sv[mat][0] * P[mat][j][0] + zv[mat][0] * X0;
            acc[mat][j][1] += sv[mat][0] * P[mat][j][1] + zv[mat][0] * X1;
            acc[mat][j][2] += sv[mat][1] * P[mat][j][2] + zv[mat][1] * X0;
            acc[mat][j][3] += sv[mat][1] * P[mat][j][3] + zv[mat][1] * X1;
          }
        }
      }
      wait_async<0>();
      __syncthreads();  // the next pass reuses the stages
    }
    if (!rows) continue;
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {  // C fragment: row r0 + 8 (e >> 1), pair 8j + 2q + (e & 1)
        const int t = 8 * j + 2 * q + (e & 1);
        if (t < nt) epilogue<MATS, T>(p, t0 + t, n0 + r0 + 8 * (e >> 1), acc[0][j][e], acc[MATS - 1][j][e]);
      }
  }
}

// grid_small_kernel: grid_kernel's arithmetic for few pairs per group (one pass of <= 8): a CTA owns 16 rows and its
// 4 warps split K (4x the CTAs: parallelism at batch 1), next block's operands prefetched in registers.
template <int MATS, typename T>
__global__ void __launch_bounds__(128) grid_small_kernel(const MParams p) {
  constexpr int NT = 1, RT = 1, KS = 4;
  __shared__ float4 red[RT * KS][MATS][NT][32];  // per warp partials (the KS warps of a row tile split K)
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, r0 = lane >> 2, q = lane & 3;
  const int rt = warp % RT, ks = warp / RT;
  int g = 0;  // the blockIdx.y-th group with pairs (the grid has min(P, G + 1) of them)
  for (int c = blockIdx.y;; g++) {
    if (g > p.G) return;
    if (p.gstart[g + 1] > p.gstart[g] && c-- == 0) break;
  }
  const int lo = p.gstart[g], hi = p.gstart[g + 1], N = p.N, K = p.K, n0 = (blockIdx.x * RT + rt) * 16;
  const T* x = reinterpret_cast<const T*>(p.x);
  const int nb = K >> 7, per = (nb + KS - 1) / KS, b_lo = min(nb, ks * per), b_hi = g < p.G ? min(nb, b_lo + per) : b_lo;
  const long long row0 = (long long)g * N + n0 + r0, row1 = row0 + 8;
  // block b's code words (rows r0, r0 + 8) and 4 s, l - 4 s of both rows, per matrix
  auto load = [&](int b, uint2 (&w)[MATS][2], float (&sz)[MATS][4]) {
    const int si = (b << 7) >> p.sbs;
#pragma unroll
    for (int mat = 0; mat < MATS; mat++) {
      const uint2* cw = reinterpret_cast<const uint2*>(p.d[mat]);
      w[mat][0] = __ldg(cw + ((row0 * p.mw + 8 * b) >> 1) + q);
      w[mat][1] = __ldg(cw + ((row1 * p.mw + 8 * b) >> 1) + q);
      const T* st = reinterpret_cast<const T*>(p.p[mat]);
      const T* lw = reinterpret_cast<const T*>(p.m[mat]);
      sz[mat][0] = 4.f * to_f(st[row0 * p.nsb + si]), sz[mat][1] = 4.f * to_f(st[row1 * p.nsb + si]);
      sz[mat][2] = to_f(lw[row0 * p.nsb + si]) - sz[mat][0], sz[mat][3] = to_f(lw[row1 * p.nsb + si]) - sz[mat][1];
    }
  };
  for (int t0 = lo; t0 < hi; t0 += 8 * NT) {
    const int nt = min(8 * NT, hi - t0);
    float acc[MATS][NT][4] = {};
    const T* xr[NT];          // B operand: pair 8j + r0 of the pass
    const float* xsr[NT][2];  // C fragment: pairs 8j + 2q, 8j + 2q + 1
#pragma unroll
    for (int j = 0; j < NT; j++) {
      const int pr = t0 + min(8 * j + r0, nt - 1);
      xr[j] = x + (long long)(p.gather ? p.perm[pr] / p.topk : pr) * K + 32 * q;
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int pc = t0 + min(8 * j + 2 * q + e, nt - 1);
        xsr[j][e] = p.xs + (long long)(p.gather ? p.perm[pc] / p.topk : pc) * nb;
      }
    }
    // block b's codes, scales, X and activations, loaded one block ahead (NT == 1)
    uint2 wc[MATS][2];
    float sc[MATS][4], xc[2];
    uint4 vc[4];
    auto loadx = [&](int b, float (&X)[2], uint4 (&v)[4]) {
      X[0] = __ldg(xsr[0][0] + b), X[1] = __ldg(xsr[0][1] + b);
      const uint4* xp = reinterpret_cast<const uint4*>(xr[0] + (b << 7));
      v[0] = __ldg(xp), v[1] = __ldg(xp + 1), v[2] = __ldg(xp + 2), v[3] = __ldg(xp + 3);
    };
    if (b_lo < b_hi) load(b_lo, wc, sc), loadx(b_lo, xc, vc);
#pragma unroll 1
    for (int b = b_lo; b < b_hi; b++) {
      uint2 wn[MATS][2];
      float sn[MATS][4], xn[2];
      uint4 vn[4];
      if (b + 1 < b_hi) load(b + 1, wn, sn), loadx(b + 1, xn, vn);  // in flight while block b computes
      unsigned a[MATS][2][16];  // rows r0, r0 + 8: word 0 registers 0..7, word 1 registers 8..15
#pragma unroll
      for (int mat = 0; mat < MATS; mat++)
#pragma unroll
        for (int h = 0; h < 2; h++) dec8<T>(wc[mat][h].x, a[mat][h]), dec8<T>(wc[mat][h].y, a[mat][h] + 8);
      const unsigned r[16] = {vc[0].x, vc[0].y, vc[0].z, vc[0].w, vc[1].x, vc[1].y, vc[1].z, vc[1].w,
                              vc[2].x, vc[2].y, vc[2].z, vc[2].w, vc[3].x, vc[3].y, vc[3].z, vc[3].w};
      float P[MATS][4] = {};
#pragma unroll
      for (int m = 0; m < 8; m++) {
        const int w = m >> 2, i0 = 2 * (m & 3), i1 = i0 + 1;
#pragma unroll
        for (int mat = 0; mat < MATS; mat++)
          Ty<T>::mma(P[mat], a[mat][0][8 * w + i0], a[mat][1][8 * w + i0], a[mat][0][8 * w + i1],
                     a[mat][1][8 * w + i1], r[8 * w + i0], r[8 * w + i1]);
      }
#pragma unroll
      for (int mat = 0; mat < MATS; mat++) {
        acc[mat][0][0] += sc[mat][0] * P[mat][0] + sc[mat][2] * xc[0];
        acc[mat][0][1] += sc[mat][0] * P[mat][1] + sc[mat][2] * xc[1];
        acc[mat][0][2] += sc[mat][1] * P[mat][2] + sc[mat][3] * xc[0];
        acc[mat][0][3] += sc[mat][1] * P[mat][3] + sc[mat][3] * xc[1];
#pragma unroll
        for (int h = 0; h < 2; h++) wc[mat][h] = wn[mat][h];
#pragma unroll
        for (int e = 0; e < 4; e++) sc[mat][e] = sn[mat][e];
      }
      xc[0] = xn[0], xc[1] = xn[1];
#pragma unroll
      for (int u = 0; u < 4; u++) vc[u] = vn[u];
    }
#pragma unroll
    for (int mat = 0; mat < MATS; mat++)
#pragma unroll
      for (int j = 0; j < NT; j++) red[warp][mat][j][lane] = make_float4(acc[mat][j][0], acc[mat][j][1], acc[mat][j][2], acc[mat][j][3]);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NT; j++) {
      if (j % KS != ks || 8 * j >= nt) continue;  // n-tile j's epilogue on the row tile's warp j % KS
      float s[MATS][4] = {};
      for (int k = 0; k < KS; k++)  // fixed order
#pragma unroll
        for (int mat = 0; mat < MATS; mat++) {
          const float4 v = red[k * RT + rt][mat][j][lane];
          s[mat][0] += v.x, s[mat][1] += v.y, s[mat][2] += v.z, s[mat][3] += v.w;
        }
#pragma unroll
      for (int e = 0; e < 4; e++) {  // C fragment: row r0 + 8 (e >> 1), pair 8j + 2q + (e & 1)
        const int t = 8 * j + 2 * q + (e & 1);
        if (t < nt) epilogue<MATS, T>(p, t0 + t, n0 + r0 + 8 * (e >> 1), s[0][e], s[MATS - 1][e]);
      }
    }
    __syncthreads();
  }
}

template <int MATS>
void grid_launch(MParams& p, const at::Tensor& like, int64_t P) {
  const bool bf = like.scalar_type() == at::kBFloat16;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t per = (2 * P + p.G) / (p.G + 1);  // pairs of a busy group, roughly
  auto go = [&](auto nt) {
    constexpr int NT = decltype(nt)::value;
    using L = GridStage<MATS, NT>;
    const int smem = L::STAGES * L::BYTES;
    const dim3 grid((p.N + 63) / 64, std::min<int64_t>(P, p.G + 1));  // only groups with pairs
    auto k = bf ? grid_kernel<MATS, NT, __nv_bfloat16> : grid_kernel<MATS, NT, __half>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    k<<<grid, 128, smem, stream>>>(p);
  };
  if (per <= 8) {  // few pairs: grid_small_kernel (4x the CTAs; H200 Qwen3.8-27B shapes: faster to 8)
    const dim3 grid(p.N / 16, std::min<int64_t>(P, p.G + 1));
    bf ? grid_small_kernel<MATS, __nv_bfloat16><<<grid, 128, 0, stream>>>(p)
       : grid_small_kernel<MATS, __half><<<grid, 128, 0, stream>>>(p);
  } else if (per <= 16) go(std::integral_constant<int, 2>{});
  else go(std::integral_constant<int, 4>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grid_check(const MParams& p, const at::Tensor& x, const at::Tensor& xs, int64_t block) {
  TORCH_CHECK(block % 128 == 0 && p.N % 16 == 0 && p.nsb % 2 == 0, "sdm_packed: grid kernel needs groups of 128k, "
              "N % 16, an even group count per row");
  TORCH_CHECK(xs.is_cuda() && xs.device() == x.device() && xs.scalar_type() == at::kFloat && xs.is_contiguous() &&
                  xs.dim() == 2 && xs.size(0) == x.size(0) && xs.size(1) == p.K / 128,
              "sdm_packed: xs (rows, K / 128) float32 on x's device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "sdm_packed: x 16-byte aligned");
}

at::Tensor grid_gate_up(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& perm, const at::Tensor& gstart,
                        const at::Tensor& g0, const at::Tensor& u0, const at::Tensor& dg, const at::Tensor& du,
                        const at::Tensor& pg, const at::Tensor& pu, const at::Tensor& mg, const at::Tensor& mu,
                        int64_t block, int64_t topk) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(g0.dim() == 2 && g0.size(0) == x.size(0) && u0.sizes() == g0.sizes() &&
                  g0.scalar_type() == x.scalar_type() && u0.scalar_type() == x.scalar_type(),
              "sdm_packed: grid_gate_up shapes");
  check(g0, "g0");
  check(u0, "u0");
  check(du, "du");
  MParams p = mmake(3, block, topk, x, perm, gstart, g0.size(1), dg, pg, mg);
  grid_check(p, x, xs, block);
  p.d[0] = dg.data_ptr(), p.d[1] = du.data_ptr(), p.p[0] = pg.data_ptr(), p.p[1] = pu.data_ptr();
  p.m[0] = static_cast<const int*>(mg.data_ptr()), p.m[1] = static_cast<const int*>(mu.data_ptr());
  p.b0[0] = g0.data_ptr(), p.b0[1] = u0.data_ptr(), p.gather = 1, p.xs = xs.data_ptr<float>();
  at::Tensor h = at::empty({perm.numel(), p.N}, x.options());
  p.out = h.data_ptr();
  if (perm.numel()) grid_launch<2>(p, x, perm.numel());
  return h;
}

at::Tensor grid_down(const at::Tensor& h, const at::Tensor& hs, const at::Tensor& perm, const at::Tensor& gstart,
                     const at::Tensor& rw, const at::Tensor& d0, const at::Tensor& d, const at::Tensor& pp,
                     const at::Tensor& m, int64_t block, int64_t topk) {
  c10::cuda::CUDAGuard guard(h.device());
  TORCH_CHECK(d0.dim() == 2 && d0.size(0) == h.size(0) && d0.scalar_type() == h.scalar_type() &&
                  rw.scalar_type() == at::kFloat && rw.numel() == h.size(0) && perm.numel() == h.size(0),
              "sdm_packed: grid_down shapes");
  check(d0, "d0");
  MParams p = mmake(3, block, topk, h, perm, gstart, d0.size(1), d, pp, m);
  grid_check(p, h, hs, block);
  p.d[0] = p.d[1] = d.data_ptr(), p.p[0] = p.p[1] = pp.data_ptr();
  p.m[0] = p.m[1] = static_cast<const int*>(m.data_ptr());
  p.rw = rw.data_ptr<float>(), p.b0[0] = d0.data_ptr(), p.gather = 0, p.xs = hs.data_ptr<float>();
  at::Tensor o = at::empty({perm.numel(), p.N}, h.options());
  p.out = o.data_ptr();
  if (perm.numel()) grid_launch<1>(p, h, perm.numel());
  return o;
}

// routed_small_kernel: routed deltas (no base) for few tokens, without routing: CTA (16 rows, token t) reads t's slots
// itself; its 4 warps split K. The arithmetic of grid_kernel (2-bit grids factored per 128-entry block, v = 1 + q / 4
// on the tensor cores) with t as B column 0 (the other 7 columns zero: tensor-core throughput is not the limit here)
// and each slot's routing weight r folded into its scales: sum_j r_j Delta_{e_j} x_t in one accumulator. x_t goes to
// shared memory interleaved as packed_ops._prep does, with its block sums X_b. Out (T, N).
template <typename T>
__global__ void __launch_bounds__(128) routed_small_kernel(const T* __restrict__ x, const int64_t* __restrict__ idx,
                                                           const float* __restrict__ rw, const int* __restrict__ codes,
                                                           const T* __restrict__ steps, const T* __restrict__ lows,
                                                           T* __restrict__ out, int N, int K, int G, int k, int sbs) {
  extern __shared__ __align__(16) unsigned char rsm[];
  constexpr int KS = 4;
  const int t = blockIdx.y, nb = K >> 7, nsb = K >> sbs, W = K / 16;
  T* xp = reinterpret_cast<T*>(rsm);                   // (K) x_t, x'[16 w + 2 i + j] = x[16 w + i + 8 j]
  float* xb = reinterpret_cast<float*>(rsm + 2 * K);   // (nb) X_b
  float4* red = reinterpret_cast<float4*>(xb + nb + (-(nb) & 3));  // (KS, 32) partials
  for (int c = threadIdx.x; c < K; c += 128) {
    const int w = c >> 4, i = (c & 15) >> 1, jj = c & 1;
    xp[c] = x[(long long)t * K + 16 * w + i + 8 * jj];
  }
  for (int b = threadIdx.x / 32; b < nb; b += 4) {  // a warp per block sum
    float v = 0.f;
    for (int c = threadIdx.x % 32; c < 128; c += 32) v += to_f(x[(long long)t * K + (b << 7) + c]);
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x % 32 == 0) xb[b] = v;
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, r0 = lane >> 2, q = lane & 3, n0 = blockIdx.x * 16;
  const int per = (nb + KS - 1) / KS, b_lo = min(nb, warp * per), b_hi = min(nb, b_lo + per);
  const bool col0 = r0 == 0;  // B column 0 (the token) is lanes 0..3's
  float acc[4] = {};
  for (int j = 0; j < k; j++) {
    const int64_t e = idx[(long long)t * k + j];
    if (e < 0 || e >= G) continue;  // no delta (uniform)
    const float r = rw[(long long)t * k + j];
    const long long row0 = e * N + n0 + r0, row1 = row0 + 8;
    const uint2* cw = reinterpret_cast<const uint2*>(codes);
    for (int b = b_lo; b < b_hi; b++) {
      const uint2 w0 = __ldg(cw + ((row0 * W + 8 * b) >> 1) + q), w1 = __ldg(cw + ((row1 * W + 8 * b) >> 1) + q);
      const int si = (b << 7) >> sbs;
      const float s0 = 4.f * r * to_f(steps[row0 * nsb + si]), s1 = 4.f * r * to_f(steps[row1 * nsb + si]);
      const float z0 = r * to_f(lows[row0 * nsb + si]) - s0, z1 = r * to_f(lows[row1 * nsb + si]) - s1;
      unsigned a0[16], a1[16];
      dec8<T>(w0.x, a0), dec8<T>(w0.y, a0 + 8), dec8<T>(w1.x, a1), dec8<T>(w1.y, a1 + 8);
      unsigned bx[16] = {};
      if (col0) {
        const uint4* v = reinterpret_cast<const uint4*>(xp + (b << 7) + 32 * q);
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const uint4 vv = v[u];
          bx[4 * u] = vv.x, bx[4 * u + 1] = vv.y, bx[4 * u + 2] = vv.z, bx[4 * u + 3] = vv.w;
        }
      }
      float P[4] = {};
#pragma unroll
      for (int m = 0; m < 8; m++) {
        const int w = m >> 2, i0 = 2 * (m & 3), i1 = i0 + 1;
        Ty<T>::mma(P, a0[8 * w + i0], a1[8 * w + i0], a0[8 * w + i1], a1[8 * w + i1], bx[8 * w + i0], bx[8 * w + i1]);
      }
      const float X = xb[b];
      acc[0] += s0 * P[0] + z0 * X, acc[2] += s1 * P[2] + z1 * X;  // (column 0: q == 0's c0, c2)
    }
  }
  red[warp * 32 + lane] = make_float4(acc[0], 0.f, acc[2], 0.f);
  __syncthreads();
  if (warp == 0 && q == 0) {
    float y0 = 0.f, y1 = 0.f;
    for (int w = 0; w < KS; w++) y0 += red[w * 32 + lane].x, y1 += red[w * 32 + lane].z;  // fixed order
    out[(long long)t * N + n0 + r0] = Ty<T>::of(y0), out[(long long)t * N + n0 + r0 + 8] = Ty<T>::of(y1);
  }
}

at::Tensor routed_small(const at::Tensor& x, const at::Tensor& idx, const at::Tensor& rw, const at::Tensor& codes,
                        const at::Tensor& steps, const at::Tensor& lows, int64_t block) {
  c10::cuda::CUDAGuard guard(x.device());
  const bool bf = x.scalar_type() == at::kBFloat16;
  TORCH_CHECK(bf || x.scalar_type() == at::kHalf, "sdm_packed: routed_small x bf16 / fp16");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && idx.dim() == 2 && idx.size(0) == x.size(0) &&
                  idx.scalar_type() == at::kLong && idx.is_contiguous() && rw.sizes() == idx.sizes() &&
                  rw.scalar_type() == at::kFloat && rw.is_contiguous(), "sdm_packed: routed_small x, idx, rw");
  const int64_t K = x.size(1), G = codes.size(0), N = codes.size(1);
  TORCH_CHECK(block % 128 == 0 && (block & (block - 1)) == 0 && K % block == 0 && N % 16 == 0,
              "sdm_packed: routed_small groups a power of 2 >= 128, N % 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0, "sdm_packed: routed_small codes aligned");
  TORCH_CHECK(codes.dim() == 3 && codes.size(2) * 16 == K && codes.scalar_type() == at::kInt && codes.is_contiguous(),
              "sdm_packed: routed_small codes (G, N, K / 16) int32");
  for (const at::Tensor* s : {&steps, &lows})
    TORCH_CHECK(s->sizes() == at::IntArrayRef({G, N, K / block}) && s->scalar_type() == x.scalar_type() &&
                    s->is_contiguous(), "sdm_packed: routed_small steps / lows (G, N, K / group)");
  for (const at::Tensor* a : {&idx, &rw, &codes, &steps, &lows})
    TORCH_CHECK(a->is_cuda() && a->device() == x.device(), "sdm_packed: routed_small tensors on x's device");
  at::Tensor o = at::empty({x.size(0), N}, x.options());
  if (!o.numel()) return o;
  const int smem = int(2 * K + (K / 128 + 3) / 4 * 16 + 4 * 32 * 16);
  const dim3 grid(N / 16, x.size(0));
  const int sbs = __builtin_ctzll(block);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto* tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(routed_small_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    routed_small_kernel<T><<<grid, 128, smem, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()), idx.data_ptr<int64_t>(), rw.data_ptr<float>(), codes.data_ptr<int>(),
        reinterpret_cast<const T*>(steps.data_ptr()), reinterpret_cast<const T*>(lows.data_ptr()),
        reinterpret_cast<T*>(o.data_ptr()), int(N), int(K), int(G), int(idx.size(1)), sbs);
  };
  bf ? go((__nv_bfloat16*)nullptr) : go((__half*)nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

}  // namespace

TORCH_LIBRARY(sdm_packed, lib) {
  lib.def(
      "gate_up(Tensor x, Tensor perm, Tensor gstart, Tensor wg, Tensor wu, Tensor dg, Tensor du, Tensor pg, "
      "Tensor pu, Tensor mg, Tensor mu, int mode, int block, int topk) -> Tensor",
      &gate_up);
  lib.def(
      "down(Tensor h, Tensor perm, Tensor gstart, Tensor rw, Tensor w, Tensor d, Tensor p, Tensor m, int mode, "
      "int block, int topk) -> Tensor",
      &down);
  lib.def(
      "matrix_gate_up(Tensor x, Tensor perm, Tensor gstart, Tensor g0, Tensor u0, Tensor dg, Tensor du, Tensor pg, "
      "Tensor pu, Tensor mg, Tensor mu, int mode, int block, int topk) -> Tensor",
      &matrix_gate_up);
  lib.def(
      "matrix_down(Tensor h, Tensor perm, Tensor gstart, Tensor rw, Tensor d0, Tensor d, Tensor p, Tensor m, "
      "int mode, int block, int topk) -> Tensor",
      &matrix_down);
  lib.def(
      "grid_gate_up(Tensor x, Tensor xs, Tensor perm, Tensor gstart, Tensor g0, Tensor u0, Tensor dg, Tensor du, "
      "Tensor pg, Tensor pu, Tensor mg, Tensor mu, int block, int topk) -> Tensor",
      &grid_gate_up);
  lib.def(
      "grid_down(Tensor h, Tensor hs, Tensor perm, Tensor gstart, Tensor rw, Tensor d0, Tensor d, Tensor p, Tensor m, "
      "int block, int topk) -> Tensor",
      &grid_down);
  lib.def("routed_small(Tensor x, Tensor idx, Tensor rw, Tensor codes, Tensor steps, Tensor lows, int block) -> Tensor",
          &routed_small);
}
