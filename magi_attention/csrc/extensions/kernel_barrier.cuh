/**********************************************************************************
 * Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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

#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h> // Required for torch::Tensor
#include <optional>
#include <stdexcept>

namespace magi_attn_ext {

// ==========================================
// 1. Device View
// ==========================================

// A POD (Plain Old Data) structure designed to be passed to CUDA kernels by value.
// It acts as a lightweight handle and does not own the underlying memory.
struct KernelBarrierView {
  int* d_val;

  KernelBarrierView() : d_val(nullptr) {}
  KernelBarrierView(int* d_val_) : d_val(d_val_) {}
  KernelBarrierView(const KernelBarrierView& other) : d_val(other.d_val) {}

#ifdef __CUDACC__
  // Signal arrival. Restricted to a single thread to prevent race conditions or redundant adds.
  __device__ __forceinline__ void arrive() {
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence();
      atomicAdd(d_val, 1);
    }
  }

  // Busy-wait (spin lock) until the counter reaches the target value.
  __device__ __forceinline__ void wait_impl(int target) {
    volatile int* v_ptr = d_val;
    while (*v_ptr < target) {
      // spin
    }
    __threadfence();
  }
#endif
};

// ==========================================
// 2. Host Class Declaration
// ==========================================

class KernelBarrier {
 public:
  // Use torch::Tensor for RAII memory management.
  // The Tensor holds the underlying int32 counter on the GPU.
  torch::Tensor buffer;
  int target_val;

  KernelBarrier(int target);

  // Copy constructor.
  // Performs a shallow copy of the Tensor (increments reference count),
  // allowing multiple objects to share the same physical barrier on the GPU.
  KernelBarrier(const KernelBarrier& other);

  // Default destructor.
  // The GPU memory is automatically freed by PyTorch when the Tensor's reference count drops to zero.
  ~KernelBarrier() = default;

  void reset();
  int get_value();
  void synchronize();

  // Returns the lightweight view for kernel arguments.
  KernelBarrierView view();
};

// ==========================================
// 3. Global Producer Function
// ==========================================

// Helper function to trigger the barrier signal.
void produce(std::optional<KernelBarrier>& counter);

} // namespace magi_attn_ext
