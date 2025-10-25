/**********************************************************************************
 * Copyright (c) 2025 SandAI. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *********************************************************************************/

/**********************************************************************************
 * Copyright (c) 2025 DeepSeek. All Rights Reserved.
 *
 * Licensed under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *********************************************************************************/

#pragma once

#include "configs.cuh"
#include "exception.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Helper Macros
///////////////////////////////////////////////////////////////////////////////////////////////////

#define DEVICE_INLINE __device__ __forceinline__

#define HOST_DEVICE __host__ __device__

#define GLOBAL_LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                   \
  {                                                                                                                                 \
    constexpr int kUnRollLoopStride = WARP_SIZE * (UNROLL_FACTOR);                                                                  \
    const int __unroll_loop_iters = ((N) / kUnRollLoopStride) * kUnRollLoopStride;                                                  \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type __unroll_buf[(UNROLL_FACTOR)];                               \
    auto __src = (SRC);                                                                                                             \
    auto __dst = (DST);                                                                                                             \
    for (int __i = (LANE_ID); __i < __unroll_loop_iters; __i += kUnRollLoopStride) {                                                \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) __unroll_buf[__j] = LD_FUNC(__src + __j * WARP_SIZE + __i); \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __j * WARP_SIZE + __i, __unroll_buf[__j]);  \
    }                                                                                                                               \
    for (int __i = __unroll_loop_iters + (LANE_ID); __i < (N); __i += WARP_SIZE)                                                    \
      ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                                                                                   \
  }

#define UNROLLED_WARP_CAST_COPY(UNROLL_FACTOR, LANE_ID, N, NLOAD, DST, SRC, LD_FUNC, ST_FUNC, CAST_FUNC)                                            \
  {                                                                                                                                                 \
    constexpr int kUnRollLoopStride = WARP_SIZE * (UNROLL_FACTOR) * (NLOAD);                                                                        \
    const int __unroll_loop_iters = ((N) / kUnRollLoopStride) * kUnRollLoopStride;                                                                  \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type __unroll_buf[(UNROLL_FACTOR) * (NLOAD)];                                     \
    auto __src = (SRC);                                                                                                                             \
    auto __dst = (DST);                                                                                                                             \
    for (int __i = (LANE_ID); __i < __unroll_loop_iters; __i += kUnRollLoopStride) {                                                                \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                                           \
        _Pragma("unroll") for (int __k = 0; __k < (NLOAD); ++__k) __unroll_buf[__j * (NLOAD) + __k] =                                               \
            LD_FUNC(__src + __j * WARP_SIZE * (NLOAD) + (__i - (LANE_ID)) + (LANE_ID) * (NLOAD) + __k);                                             \
      }                                                                                                                                             \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                                                                             \
          ST_FUNC(__dst + __j * WARP_SIZE + (__i - (LANE_ID)) / (NLOAD) + (LANE_ID), CAST_FUNC(__unroll_buf + __j * (NLOAD)));                      \
    }                                                                                                                                               \
    for (int __i = __unroll_loop_iters + (LANE_ID); __i < (N); __i += WARP_SIZE * (NLOAD)) {                                                        \
      _Pragma("unroll") for (int __k = 0; __k < (NLOAD); ++__k) __unroll_buf[__k] = LD_FUNC(__src + (__i - (LANE_ID)) + (LANE_ID) * (NLOAD) + __k); \
      ST_FUNC(__dst + (__i - (LANE_ID)) / (NLOAD) + (LANE_ID), CAST_FUNC(__unroll_buf));                                                            \
    }                                                                                                                                               \
  }

namespace magi_attn_comm::grpcoll {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Common Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

template <typename dtype_t>
HOST_DEVICE constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
HOST_DEVICE constexpr dtype_t align(dtype_t a, dtype_t b) {
  return ceil_div<dtype_t>(a, b) * b;
}

DEVICE_INLINE float log2f_approx(const float& x) {
  float ret;
  asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
}

DEVICE_INLINE float exp2f_approx(const float& x) {
  float ret;
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
}

DEVICE_INLINE void trap() {
  asm("trap;");
}

template <typename dtype_t>
DEVICE_INLINE dtype_t get_neg_inf() {
  return -std::numeric_limits<dtype_t>::infinity();
}

template <typename dtype_t>
DEVICE_INLINE bool is_neg_inf(const dtype_t val) {
  return val == get_neg_inf<dtype_t>();
}

template <typename dtype_t>
DEVICE_INLINE dtype_t safe_subtract(const dtype_t a, const dtype_t b) {
  if (is_neg_inf(a) && is_neg_inf(b)) {
    // "-inf" - "-inf" will result in "nan"
    // but we want it to be still "-inf"
    return get_neg_inf<dtype_t>();
  }
  return a - b;
}

template <typename dtype_t>
DEVICE_INLINE dtype_t safe_exp(const dtype_t val) {
  if constexpr (std::is_same_v<dtype_t, half>) { // for fp16
    return hexp(val);
  } else if constexpr (std::is_same_v<dtype_t, nv_bfloat16>) { // for bf16
    return __float2bfloat16(expf(__bfloat162float(val)));
  } else { // for fp32 and fp64
    return std::exp(val);
  }
}

template <typename dtype_t>
DEVICE_INLINE dtype_t safe_log1p(const dtype_t val) {
  if constexpr (std::is_same_v<dtype_t, half>) { // for fp16
    return __float2half(log1pf(__half2float(val)));
  } else if constexpr (std::is_same_v<dtype_t, nv_bfloat16>) { // for bf16
    return __float2bfloat16(log1pf(__bfloat162float(val)));
  } else { // for fp32 and fp64
    return std::log1p(val);
  }
}

template <typename dtype_t>
DEVICE_INLINE dtype_t get_lse_rescale_weight(const dtype_t lse_to_rescale, const dtype_t rescaled_lse) {
  // formula derivation: wi = exp(lsei - lse)
  dtype_t rescale_weight = safe_exp(safe_subtract(lse_to_rescale, rescaled_lse));
  return rescale_weight;
}

template <typename reduce_dtype_t, typename src_dtype_t>
DEVICE_INLINE void lse_reduce(reduce_dtype_t& reduced_lse, const src_dtype_t src_lse) {
  auto src_lse_reduce_dtype = static_cast<reduce_dtype_t>(src_lse);

  // formula derivation:
  // lse = log(exp(lse1) + exp(lse2))
  //     = lse1 + log(1 + exp(lse2 - lse1))
  //     = max_lse + log(1 + exp(min_lse - max_lse))
  //     = max_lse + log1p(exp(min_lse - max_lse))
  //     = max_lse + softplus(min_lse - max_lse)
  auto max_lse = std::max(reduced_lse, src_lse_reduce_dtype);
  auto min_lse = std::min(reduced_lse, src_lse_reduce_dtype);
  reduced_lse = max_lse + safe_log1p(safe_exp(safe_subtract(min_lse, max_lse)));
}

DEVICE_INLINE float fast_pow2(int x) {
  // We can ensure `-126 <= x and x <= 127`
  uint32_t bits_x = (x + 127) << 23;
  return *reinterpret_cast<float*>(&bits_x);
}

DEVICE_INLINE int fast_log2_ceil(float x) {
  auto bits_x = *reinterpret_cast<uint32_t*>(&x);
  auto exp_x = (bits_x >> 23) & 0xff;
  auto man_bits = bits_x & ((1 << 23) - 1);
  return exp_x - 127 + (man_bits != 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Vectiorized Integer Dtype
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBytes>
struct VecInt {};
template <>
struct VecInt<1> {
  using vec_t = int8_t;
};
template <>
struct VecInt<2> {
  using vec_t = int16_t;
};
template <>
struct VecInt<4> {
  using vec_t = int;
};
template <>
struct VecInt<8> {
  using vec_t = int64_t;
};
template <>
struct VecInt<16> {
  using vec_t = int4;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Memory Fence Funcs
///////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE void memory_fence() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

DEVICE_INLINE void memory_fence_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

DEVICE_INLINE void memory_fence_cta() {
  asm volatile("fence.acq_rel.cta;" ::: "memory");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Load/Store Funcs with Memory Order
///////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE void st_relaxed_sys_global(const int* ptr, int val) {
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

DEVICE_INLINE void st_release_sys_global(const int* ptr, int val) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

DEVICE_INLINE void st_release_cta(const int* ptr, int val) {
  asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

DEVICE_INLINE int ld_acquire_sys_global(const int* ptr) {
  int ret;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE uint64_t ld_acquire_sys_global(const uint64_t* ptr) {
  uint64_t ret;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE int ld_acquire_global(const int* ptr) {
  int ret;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE int atomic_add_release_sys_global(const int* ptr, int value) {
  int ret;
  asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
  return ret;
}

DEVICE_INLINE int atomic_add_release_global(const int* ptr, int value) {
  int ret;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
  return ret;
}

DEVICE_INLINE int ld_acquire_cta(const int* ptr) {
  int ret;
  asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE uint8_t ld_na_relaxed(const uint8_t* ptr) {
  uint16_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return static_cast<uint8_t>(ret);
}

DEVICE_INLINE uint16_t ld_na_relaxed(const uint16_t* ptr) {
  uint16_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE uint32_t ld_na_relaxed(const uint32_t* ptr) {
  uint32_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE uint64_t ld_na_relaxed(const uint64_t* ptr) {
  uint64_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE int ld_volatile_global(const int* ptr) {
  int ret;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE float ld_volatile_global(const float* ptr) {
  float ret;
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE int64_t ld_volatile_global(const int64_t* ptr) {
  int64_t ret;
  asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

DEVICE_INLINE int64_t ld_volatile_global(const uint64_t* ptr) {
  int64_t ret;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Non-Cached Load/Store Funcs
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global.L2::256B"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t>
DEVICE_INLINE dtype_t ld_nc_global(const dtype_t* ptr) {
  auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
  return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
DEVICE_INLINE uint8_t ld_nc_global(const uint8_t* ptr) {
  uint16_t ret;
  // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
  asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return static_cast<uint8_t>(ret);
}

template <>
DEVICE_INLINE int ld_nc_global(const int* ptr) {
  int ret;
  asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

template <>
DEVICE_INLINE int64_t ld_nc_global(const int64_t* ptr) {
  int64_t ret;
  asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

template <>
DEVICE_INLINE float ld_nc_global(const float* ptr) {
  float ret;
  asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
}

template <>
DEVICE_INLINE int2 ld_nc_global(const int2* ptr) {
  int2 ret;
  asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
  return ret;
}

template <>
DEVICE_INLINE int4 ld_nc_global(const int4* ptr) {
  int4 ret;
  asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
  return ret;
}

DEVICE_INLINE void st_na_relaxed(const uint8_t* ptr, uint8_t val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));
}

DEVICE_INLINE void st_na_relaxed(const uint16_t* ptr, uint16_t val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
}

DEVICE_INLINE void st_na_relaxed(const uint32_t* ptr, uint32_t val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

DEVICE_INLINE void st_na_relaxed(const int* ptr, int val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

DEVICE_INLINE void st_na_relaxed(const int4* ptr, int4 val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

DEVICE_INLINE void st_na_release(const int* ptr, int val) {
  asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

DEVICE_INLINE void st_na_release(const uint32_t* ptr, uint32_t val) {
  asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

DEVICE_INLINE void st_na_release(const uint64_t* ptr, uint64_t val) {
  asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
// NOTES: `L1::no_allocate` informs the compiler not to cache the data in L1 cache
// since the data to be stored (i.e. recv data) won't be read
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
DEVICE_INLINE void st_na_global(const dtype_t* ptr, const dtype_t& value) {
  st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr), *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
DEVICE_INLINE void st_na_global(const int* ptr, const int& value) {
  asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
DEVICE_INLINE void st_na_global(const int64_t* ptr, const int64_t& value) {
  asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <>
DEVICE_INLINE void st_na_global(const float* ptr, const float& value) {
  asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <>
DEVICE_INLINE void st_na_global(const int4* ptr, const int4& value) {
  asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TMA Helper Funcs
///////////////////////////////////////////////////////////////////////////////////////////////////

// TMA PTX instructions
#ifndef DISABLE_SM90_FEATURES

template <typename ptr_t>
DEVICE_INLINE uint32_t to_shared_ptr_value(const ptr_t* ptr) {
  // Transfer pointer from generic space to shared space
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

DEVICE_INLINE uint32_t elect_one_sync(int lane_id) {
  uint32_t pred = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "      elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "      mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(lane_id), "+r"(pred)
      : "r"(0xffffffff));
  return pred;
}

DEVICE_INLINE void fence_view_async_shared() {
  asm volatile("fence.proxy.async.shared::cta; \n" ::);
}

DEVICE_INLINE void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
}

DEVICE_INLINE void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
  auto mbar_int_ptr = to_shared_ptr_value<uint64_t>(mbar_ptr);
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count), "r"(mbar_int_ptr));
}

DEVICE_INLINE void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& stage, int num_tma_stages = 2) {
  auto mbar_int_ptr = to_shared_ptr_value<uint64_t>(mbar_ptr);
  asm volatile(
      "{\n\t"
      ".reg .pred       P1; \n\t"
      "LAB_WAIT: \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
      "@P1 bra DONE; \n\t"
      "bra     LAB_WAIT; \n\t"
      "DONE: \n\t"
      "}" ::"r"(mbar_int_ptr),
      "r"(stage),
      "r"(0x989680));

  // NOTES: stage is updated inplace
  if (num_tma_stages == 2)
    stage ^= 1;
  else
    stage = (stage + 1) % num_tma_stages;
}

DEVICE_INLINE void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
  auto mbar_int_ptr = to_shared_ptr_value<uint64_t>(mbar_ptr);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(num_bytes), "r"(mbar_int_ptr));
}

DEVICE_INLINE void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;");
}

constexpr uint64_t kEvictFirst = 0x12f0000000000000; // the data should be evicted first in the L2 cache
constexpr uint64_t kEvictNormal = 0x1000000000000000;

DEVICE_INLINE void tma_load_1d(const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes, bool evict_first = true) {
  auto mbar_int_ptr = to_shared_ptr_value<uint64_t>(mbar_ptr);
  auto smem_int_ptr = to_shared_ptr_value<void>(smem_ptr);
  const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
               "l"(gmem_ptr),
               "r"(num_bytes),
               "r"(mbar_int_ptr),
               "l"(cache_hint)
               : "memory");
}

DEVICE_INLINE void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes, bool evict_first = true) {
  auto smem_int_ptr = to_shared_ptr_value<void>(smem_ptr);
  const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(gmem_ptr), "r"(smem_int_ptr), "r"(num_bytes), "l"(cache_hint)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;");
}

template <int N = 0>
DEVICE_INLINE void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Sync/Barrier/Lock Helper Funcs
///////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE void sync_warp_group(int group_flag, int group_size) {
  asm volatile("bar.sync %0, %1;" ::"r"(group_flag), "r"(group_size));
}

template <int kNumRanks, bool kSyncOnly = false>
DEVICE_INLINE void barrier_block(int** barrier_signal_ptrs, int rank) {
  auto thread_id = static_cast<int>(threadIdx.x);

  // For non-sync-only cases, the memory operations by other threads in the block must be visible to the `sys` scope
  if constexpr (not kSyncOnly) {
    memory_fence();
    __syncthreads();
  }

  // Add self-ranks, sub other ranks
  if (thread_id < kNumRanks) {
    atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
    atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
  }
  GRPCOLL_DEVICE_ASSERT(kNumRanks <= blockDim.x);

  // Check timeout
  auto start_time = clock64();
  while (true) {
    auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
    if (__all_sync(0xffffffff, value <= 0))
      break;

    if (clock64() - start_time > NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
      printf("grpcoll timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
      trap();
    }
  }
  __syncthreads();
}

DEVICE_INLINE int atomic_cas_cta_acquire(int* addr, int x, int y) {
  int ret;
  asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;" : "=r"(ret) : "l"(addr), "r"(x), "r"(y) : "memory");
  return ret;
}

DEVICE_INLINE int atomic_exch_cta_release(int* addr, int x) {
  int ret;
  asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(x) : "memory");
  return ret;
}

DEVICE_INLINE void acquire_lock(int* mutex) {
  // To make later memory operations valid, we must use `acquire` for memory semantics
  while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)
    ;
}

DEVICE_INLINE void release_lock(int* mutex) {
  // To make previous memory operations visible to other threads, we must use `release` for memory semantics
  atomic_exch_cta_release(mutex, 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Warp Sync Funcs
///////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE int broadcast_in_warp(int val, int src_lane = 0) {
  return __shfl_sync(0xffffffff, val, src_lane);
}

DEVICE_INLINE int any_in_warp(int pred) {
  return __any_sync(0xffffffff, pred);
}

DEVICE_INLINE int all_in_warp(int pred) {
  return __all_sync(0xffffffff, pred);
}

template <typename dtype_t>
DEVICE_INLINE dtype_t broadcast_ptr_in_warp(dtype_t& ptr, int src_lane = 0) {
  GRPCOLL_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");

  auto send_int_vals = reinterpret_cast<int*>(&ptr);
  int recv_int_vals[sizeof(dtype_t) / sizeof(int)];

#pragma unroll
  for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
    recv_int_vals[i] = broadcast_in_warp(send_int_vals[i], src_lane);

  return *reinterpret_cast<dtype_t*>(recv_int_vals);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Warp Reduce Funcs
///////////////////////////////////////////////////////////////////////////////////////////////////

// Operation functors
template <typename T>
struct ReduceSum {
  __device__ T operator()(T a, T b) const {
    return a + b;
  }
};
template <typename T>
struct ReduceMax {
  __device__ T operator()(T a, T b) const {
    return a > b ? a : b;
  }
};
template <typename T>
struct ReduceMin {
  __device__ T operator()(T a, T b) const {
    return a < b ? a : b;
  }
};

// Unified reduction function
template <uint32_t kNumLanes, typename T, typename Op>
DEVICE_INLINE T warp_reduce(T value, Op op) {
  GRPCOLL_STATIC_ASSERT(kNumLanes == 32 or kNumLanes == 16 or kNumLanes == 8 or kNumLanes == 4 or kNumLanes == 2 or kNumLanes == 1, "Invalid number of lanes");

  if constexpr (kNumLanes >= 32)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 16));
  if constexpr (kNumLanes >= 16)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 8));
  if constexpr (kNumLanes >= 8)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 4));
  if constexpr (kNumLanes >= 4)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 2));
  if constexpr (kNumLanes >= 2)
    value = op(value, __shfl_xor_sync(0xffffffff, value, 1));
  return value;
}

// Convenience aliases
template <uint32_t kNumLanes = 32, typename T>
DEVICE_INLINE T warp_reduce_sum(T value) {
  return warp_reduce<kNumLanes, T>(value, ReduceSum<T>{});
}

template <uint32_t kNumLanes = 32, typename T>
DEVICE_INLINE T warp_reduce_max(T value) {
  return warp_reduce<kNumLanes, T>(value, ReduceMax<T>{});
}

template <uint32_t kNumLanes = 32, typename T>
DEVICE_INLINE T warp_reduce_min(T value) {
  return warp_reduce<kNumLanes, T>(value, ReduceMin<T>{});
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ForEach Funcs
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename dst_dtype_t, int kArrayLength>
DEVICE_INLINE void foreach_fill(dst_dtype_t* dst_ptr, const dst_dtype_t value) {
#pragma unroll
  for (int i = 0; i < kArrayLength; ++i)
    dst_ptr[i] = value;
}

template <typename dst_dtype_t, int kArrayLength>
DEVICE_INLINE void foreach_mul(dst_dtype_t* dst_ptr, const dst_dtype_t value) {
#pragma unroll
  for (int i = 0; i < kArrayLength; ++i)
    dst_ptr[i] *= value;
}

template <typename dst_dtype_t, int kArrayLength>
DEVICE_INLINE void foreach_div(dst_dtype_t* dst_ptr, const dst_dtype_t value) {
#pragma unroll
  for (int i = 0; i < kArrayLength; ++i)
    dst_ptr[i] /= value;
}

template <typename dst_dtype_t, typename src_dtype_t, int kArrayLength>
DEVICE_INLINE void foreach_assign(dst_dtype_t* dst_ptr, const src_dtype_t* src_ptr) {
#pragma unroll
  for (int i = 0; i < kArrayLength; ++i)
    dst_ptr[i] = static_cast<dst_dtype_t>(src_ptr[i]);
}

template <typename reduce_dtype_t, typename src_dtype_t, int kArrayLength>
DEVICE_INLINE void foreach_reduce_add(reduce_dtype_t* reduce_ptr, const src_dtype_t* src_ptr) {
#pragma unroll
  for (int i = 0; i < kArrayLength; ++i)
    reduce_ptr[i] += static_cast<reduce_dtype_t>(src_ptr[i]);
}

template <typename reduce_dtype_t, typename src_dtype_t, typename lse_dtype_t, int kArrayLength>
DEVICE_INLINE void foreach_reduce_lse(reduce_dtype_t* reduce_ptr, const reduce_dtype_t reduced_lse, const src_dtype_t* src_ptr, const lse_dtype_t src_lse) {
  // formula derivation:
  // out += exp(lsei - lse) * srci, where `lse` is the reduced lse, and `out` is the reduced buf
  reduce_dtype_t rescale_weight = get_lse_rescale_weight(static_cast<reduce_dtype_t>(src_lse), reduced_lse);

#pragma unroll
  for (int i = 0; i < kArrayLength; ++i)
    reduce_ptr[i] += rescale_weight * static_cast<reduce_dtype_t>(src_ptr[i]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Other Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FuncT>
struct PatternVisitor {
  FuncT func;

  HOST_DEVICE explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

  HOST_DEVICE auto operator[](const uint32_t& i) {
    return func(i);
  }
};

DEVICE_INLINE void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
  int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
  token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
  token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
DEVICE_INLINE dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
  GRPCOLL_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
  dtype_b_t packed;
  auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
  unpacked_ptr[0] = x, unpacked_ptr[1] = y;
  return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
DEVICE_INLINE void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
  GRPCOLL_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
  auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
  x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t, typename lp_dtype_t, typename vec_dtype_t>
DEVICE_INLINE vec_dtype_t vec_downcast(vec_dtype_t* vec_ptr) {
  constexpr int kDtypePerVec = sizeof(vec_dtype_t) / sizeof(dtype_t);
  constexpr int kLowPrecisionDtypePerDtype = sizeof(dtype_t) / sizeof(lp_dtype_t);

  vec_dtype_t downcast_val_vec;
  lp_dtype_t* downcast_val_ptr_lp = reinterpret_cast<lp_dtype_t*>(&downcast_val_vec);

#pragma unroll
  for (int i = 0; i < kLowPrecisionDtypePerDtype; ++i) {
    dtype_t* ith_dtype_ptr = reinterpret_cast<dtype_t*>(vec_ptr + i);
#pragma unroll
    for (int j = 0; j < kDtypePerVec; ++j)
      downcast_val_ptr_lp[i * kDtypePerVec + j] = static_cast<lp_dtype_t>(ith_dtype_ptr[j]);
  }

  return downcast_val_vec;
}

constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;

DEVICE_INLINE void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
  if (round_scale) {
    auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
    scale = fast_pow2(-exp_scale_inv);
    scale_inv = fast_pow2(exp_scale_inv);
  } else {
    scale_inv = amax * kFinfoAmaxInvE4M3;
    scale = kFinfoAmaxE4M3 / amax;
  }
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
DEVICE_INLINE out_dtype_t extract_required_scale_format(float value) {
  if constexpr (kIsUE8M0) {
    return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
  } else {
    return value;
  }
}

} // namespace magi_attn_comm::grpcoll
