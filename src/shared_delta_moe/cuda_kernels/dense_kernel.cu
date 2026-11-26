// Dense-delta MoE GEMMs for prefill on Hopper (sm_90a), hand-written: TMA + mbarrier pipeline + wgmma.
//
// One kernel template, two uses (torch ops sdm_dense::gate_up and sdm_dense::down):
//   gate_up: h = silu(xs (W_g + D_g)^T) * (xs (W_u + D_u)^T) for the pairs in sorted order (xs = x gathered by perm)
//   down:    o[perm[row]] = r[perm[row]] * h[row] (W_d + D_d)^T, in pair order
// Rows are grouped by expert (gstart, G + 2 starts; group G = the zero expert: W_b alone).
//
// CTA: 3 warpgroups. WG0 (one thread) issues TMA loads of a 256 x 64 activation tile, a 128 x 64 W_b tile and a
// 128 x 64 delta tile per K step into a 3-stage ring; WG1 and WG2 add the delta tile into the W_b tile in shared
// memory (both tiles have the same 128B swizzle, so the add is elementwise and layout-free; bf16 sum rounded as the
// materialised W_b + Delta is), then issue wgmma (SS, m64n128k16) on 128 rows each. The add of stage k overlaps
// the wgmma of stage k - 1. For gate_up the B tile is 64 gate rows over 64 up rows, so each thread holds gate
// column c and up column c + 64 of the same row: SiLU * up in registers. Epilogue rounding as the PyTorch MLP:
// g, u, silu(g), h (gate_up); y and the routing weight (down) to bf16.
// Tiles: 1-D grid, grouped order (GROUP_M row tiles sweep the column tiles), derived from gstart on the device:
// no host sync, CUDA-graph safe. Rows past a group's end belong to the next group: computed, not stored.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/library.h>

namespace {

constexpr int BM = 256, BN = 128, BK = 64, ST = 3;
constexpr int A_ELEMS = BM * BK, B_ELEMS = BN * BK;

struct Smem {
  alignas(1024) __nv_bfloat16 a[ST][A_ELEMS];
  alignas(1024) __nv_bfloat16 b[ST][B_ELEMS];
  alignas(1024) __nv_bfloat16 d[ST][B_ELEMS];
  uint64_t full[ST], empty[ST];
};

__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(b)), "r"(n));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int phase) {
  uint32_t ok;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(su32(b)), "r"(phase) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(su32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(su32(bar)), "r"(c0), "r"(c1) : "memory");
}

// K-major tile of 64 bf16 (128 B) rows, 128B swizzle (as TMA writes it): 8-row core groups 1024 B apart
__device__ __forceinline__ uint64_t desc(const void* p) {
  return ((uint64_t)(su32(p) & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void wgmma128(float (&d)[64], uint64_t a, uint64_t b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(a), "l"(b), "r"(1));
}

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

template <bool GATE_UP>
__global__ void __launch_bounds__(384, 1)
dense_kernel(const __grid_constant__ CUtensorMap xm, const __grid_constant__ CUtensorMap w0m,
             const __grid_constant__ CUtensorMap w1m, const __grid_constant__ CUtensorMap d0m,
             const __grid_constant__ CUtensorMap d1m, const int* __restrict__ gstart, const int* __restrict__ perm,
             const float* __restrict__ rw, __nv_bfloat16* __restrict__ out, int G, int N, int K, int ldo,
             int group_m) {
  // ---- tile from gstart: row tiles of all groups in group order, then the grouped (GROUP_M) sweep
  const int NT = GATE_UP ? (N + 63) / 64 : (N + BN - 1) / BN;  // gate_up: N = I, 64 h columns per tile
  int MT = 0;
  for (int g = 0; g <= G; ++g) MT += (gstart[g + 1] - gstart[g] + BM - 1) / BM;
  const int t = blockIdx.x;
  if (t >= MT * NT) return;
  const int width = group_m * NT, first = (t / width) * group_m, gsz = min(MT - first, group_m);
  const int mi = first + (t % width) % gsz, ni = (t % width) / gsz;
  int grp = 0, row0 = 0, rend = 0;
  for (int g = 0, cum = 0; g <= G; ++g) {
    const int lo = gstart[g], hi = gstart[g + 1], mt = (hi - lo + BM - 1) / BM;
    if (mi < cum + mt) { grp = g; row0 = lo + (mi - cum) * BM; rend = hi; break; }
    cum += mt;
  }
  const bool has_delta = grp < G;
  const int nk = (K + BK - 1) / BK;

  extern __shared__ uint8_t raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
  if (threadIdx.x == 0) {
    for (int i = 0; i < ST; ++i) {
      mbar_init(&s.full[i], 1);
      mbar_init(&s.empty[i], 8);  // the consumer warps
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int wg = threadIdx.x / 128;
  if (wg == 0) {  // ---------------------------------------------------------------- producer
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (threadIdx.x == 0) {
      const uint32_t bytes = (A_ELEMS + B_ELEMS * (has_delta ? 2 : 1)) * 2;
      const int drow = grp * N;  // the group's rows in the (G * N, K) deltas
      for (int k = 0; k < nk; ++k) {
        const int st = k % ST;
        if (k >= ST) mbar_wait(&s.empty[st], ((k / ST) + 1) & 1);
        mbar_expect_tx(&s.full[st], bytes);
        tma2d(s.a[st], &xm, &s.full[st], k * BK, row0);
        if (GATE_UP) {
          tma2d(s.b[st], &w0m, &s.full[st], k * BK, ni * 64);
          tma2d(s.b[st] + 64 * BK, &w1m, &s.full[st], k * BK, ni * 64);
          if (has_delta) {
            tma2d(s.d[st], &d0m, &s.full[st], k * BK, drow + ni * 64);
            tma2d(s.d[st] + 64 * BK, &d1m, &s.full[st], k * BK, drow + ni * 64);
          }
        } else {
          tma2d(s.b[st], &w0m, &s.full[st], k * BK, ni * BN);
          if (has_delta) tma2d(s.d[st], &d0m, &s.full[st], k * BK, drow + ni * BN);
        }
      }
    }
    return;
  }
  // ------------------------------------------------------------------------------------ consumers
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  const int ct = threadIdx.x - 128, m0 = (wg - 1) * 128;
  float d0[64], d1[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) d0[i] = d1[i] = 0.f;
  for (int k = 0; k < nk; ++k) {
    const int st = k % ST;
    mbar_wait(&s.full[st], (k / ST) & 1);
    if (has_delta) {  // W_b tile += delta tile, in place, 16 B per access
      uint4* b = reinterpret_cast<uint4*>(s.b[st]);
      const uint4* dd = reinterpret_cast<const uint4*>(s.d[st]);
#pragma unroll
      for (int i = 0; i < B_ELEMS / 8 / 256; ++i) {
        uint4 w = b[ct + i * 256], v = dd[ct + i * 256];
        __nv_bfloat162* wp = reinterpret_cast<__nv_bfloat162*>(&w);
        const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) wp[j] = __hadd2(wp[j], vp[j]);
        b[ct + i * 256] = w;
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic writes -> wgmma (async proxy) reads
      asm volatile("bar.sync 1, 256;" ::: "memory");
    }
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      const uint64_t bd = desc(s.b[st] + kk * 16);
      wgmma128(d0, desc(s.a[st] + m0 * BK + kk * 16), bd);
      wgmma128(d1, desc(s.a[st] + (m0 + 64) * BK + kk * 16), bd);
    }
    wg_commit();
    wg_wait<1>();  // stage k - 1's wgmma done: release its buffers
    if (k > 0 && (ct % 32) == 0) mbar_arrive(&s.empty[(k - 1) % ST]);
  }
  wg_wait<0>();

  // ------------------------------------------------------------------------------------ epilogue
  const int w = (ct % 128) / 32, l = ct % 32;
  auto store = [&](const float(&acc)[64], int x) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = row0 + m0 + x * 64 + w * 16 + l / 4 + h * 8;
      if (row >= rend) continue;
      if constexpr (GATE_UP) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int col = ni * 64 + i * 8 + (l % 4) * 2;
          if (col >= N) continue;
          float hv[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float g = bf(acc[i * 4 + h * 2 + e]), u = bf(acc[(i + 8) * 4 + h * 2 + e]);
            hv[e] = bf(g / (1.f + expf(-g))) * u;
          }
          *reinterpret_cast<__nv_bfloat162*>(out + (int64_t)row * ldo + col) = __floats2bfloat162_rn(hv[0], hv[1]);
        }
      } else {
        const int pair = perm[row];
        const float r = bf(rw[pair]);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int col = ni * BN + i * 8 + (l % 4) * 2;
          if (col >= N) continue;
          *reinterpret_cast<__nv_bfloat162*>(out + (int64_t)pair * ldo + col) =
              __floats2bfloat162_rn(bf(acc[i * 4 + h * 2]) * r, bf(acc[i * 4 + h * 2 + 1]) * r);
        }
      }
    }
  };
  store(d0, 0);
  store(d1, 1);
}

// ---------------------------------------------------------------------------------------------------- host side

using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                            const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                            CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

Encode encode() {
  static Encode f = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) ==
                        cudaSuccess && q == cudaDriverEntryPointSuccess,
                "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<Encode>(p);
  }();
  return f;
}

// (rows, cols) row-major bf16, box (box_rows, 64), 128B swizzle; out-of-bounds reads are zeros
CUtensorMap tmap(const at::Tensor& t, int64_t rows, int64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {64, box_rows}, es[2] = {1, 1};
  CUresult r = encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

void check(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              ": contiguous CUDA bf16 expected");
  TORCH_CHECK(t.size(-1) % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
              ": last dimension a multiple of 8, 16-byte aligned");
}

template <bool GATE_UP>
void launch(const CUtensorMap* maps, const at::Tensor& gstart, const int* perm, const float* rw, at::Tensor& out,
            int G, int N, int K, int P, int64_t group_m) {
  const int NT = GATE_UP ? (N + 63) / 64 : (N + BN - 1) / BN;
  const int tiles = ((P + BM - 1) / BM + G + 1) * NT;  // at most this many row tiles; the rest exit
  const int smem = sizeof(Smem) + 1024;
  auto k = dense_kernel<GATE_UP>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  k<<<tiles, 384, smem, at::cuda::getCurrentCUDAStream()>>>(maps[0], maps[1], maps[2], maps[3], maps[4],
                                                              gstart.data_ptr<int>(), perm, rw,
                                                              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), G, N,
                                                              K, out.stride(0), (int)group_m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor gate_up(const at::Tensor& xs, const at::Tensor& gstart, const at::Tensor& wg, const at::Tensor& wu,
                   const at::Tensor& dg, const at::Tensor& du, int64_t group_m) {
  check(xs, "xs"), check(wg, "wg"), check(wu, "wu"), check(dg, "dg"), check(du, "du");
  c10::cuda::CUDAGuard guard(xs.device());
  const int P = xs.size(0), K = xs.size(1), N = wg.size(0), G = dg.size(0);
  auto h = at::empty({P, N}, xs.options());
  if (P == 0) return h;
  const at::Tensor &dgm = G ? dg : wg, &dum = G ? du : wu;  // no deltas: placeholders, never read
  CUtensorMap maps[5] = {tmap(xs, P, K, BM), tmap(wg, N, K, 64), tmap(wu, N, K, 64),
                         tmap(dgm, (int64_t)std::max(G, 1) * N, K, 64), tmap(dum, (int64_t)std::max(G, 1) * N, K, 64)};
  launch<true>(maps, gstart, nullptr, nullptr, h, G, N, K, P, group_m);
  return h;
}

at::Tensor down(const at::Tensor& h, const at::Tensor& gstart, const at::Tensor& perm, const at::Tensor& rw,
                const at::Tensor& wd, const at::Tensor& dd, int64_t group_m) {
  check(h, "h"), check(wd, "wd"), check(dd, "dd");
  c10::cuda::CUDAGuard guard(h.device());
  const int P = h.size(0), K = h.size(1), N = wd.size(0), G = dd.size(0);
  auto o = at::empty({P, N}, h.options());
  if (P == 0) return o;
  const CUtensorMap wm = tmap(wd, N, K, BN), dm = G ? tmap(dd, (int64_t)G * N, K, BN) : wm;
  CUtensorMap maps[5] = {tmap(h, P, K, BM), wm, wm, dm, dm};
  launch<false>(maps, gstart, perm.data_ptr<int>(), rw.data_ptr<float>(), o, G, N, K, P, group_m);
  return o;
}

}  // namespace

TORCH_LIBRARY(sdm_dense, m) {
  m.def("gate_up(Tensor xs, Tensor gstart, Tensor wg, Tensor wu, Tensor dg, Tensor du, int group_m=4) -> Tensor");
  m.def("down(Tensor h, Tensor gstart, Tensor perm, Tensor rw, Tensor wd, Tensor dd, int group_m=4) -> Tensor");
}

TORCH_LIBRARY_IMPL(sdm_dense, CUDA, m) {
  m.impl("gate_up", &gate_up);
  m.impl("down", &down);
}
