// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

#include <cutlass/numeric_types.h>
#include <torch/extension.h>

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                       \
  [&] {                                                                                          \
    if (COND) {                                                                                  \
      constexpr static bool CONST_NAME = true;                                                   \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static bool CONST_NAME = false;                                                  \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_CLUSTER
  #define CLUSTER_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define CLUSTER_SWITCH BOOL_SWITCH
#endif

#define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                                        \
[&] {                                                                                            \
  if (ARCH == 90) {                                                                              \
    constexpr static int ARCH_NAME = 90;                                                         \
    return __VA_ARGS__();                                                                        \
  } else {                                                                                       \
    TORCH_CHECK(false, "Unsupported architecture: ", ARCH);                                      \
  }                                                                                              \
}()

#define HEADDIM_SWITCH(HEADDIM, ...)                                                             \
  [&] {                                                                                          \
    if (HEADDIM == 64) {                                                                         \
      constexpr static int kHeadSize = 64;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 128) {                                                                 \
      constexpr static int kHeadSize = 128;                                                      \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 192) {                                                                 \
      constexpr static int kHeadSize = 192;                                                      \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 256) {                                                                 \
      constexpr static int kHeadSize = 256;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

#define COMPUTE_DTYPE_SWITCH(DTYPE, TYPE_NAME, ...)                                              \
  [&] {                                                                                          \
    if (DTYPE == at::ScalarType::BFloat16) {                                                     \
      using TYPE_NAME = cutlass::bfloat16_t;                                                     \
      return __VA_ARGS__();                                                                      \
    } else if (DTYPE == at::ScalarType::Half) {                                                  \
      using TYPE_NAME = cutlass::half_t;                                                         \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      TORCH_CHECK(false, "Unsupported data type");                                               \
    }                                                                                            \
  }()

#define OUT_DTYPE_SWITCH(DTYPE, TYPE_NAME, ...)                                                  \
  [&] {                                                                                          \
    if (DTYPE == at::ScalarType::BFloat16) {                                                     \
      using TYPE_NAME = cutlass::bfloat16_t;                                                     \
      return __VA_ARGS__();                                                                      \
    } else if (DTYPE == at::ScalarType::Half) {                                                  \
      using TYPE_NAME = cutlass::half_t;                                                         \
      return __VA_ARGS__();                                                                      \
    } else if (DTYPE == at::ScalarType::Float) {                                                 \
      using TYPE_NAME = float;                                                                   \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      TORCH_CHECK(false, "Unsupported data type");                                               \
    }                                                                                            \
  }()
