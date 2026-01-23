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

#include <c10/cuda/CUDAStream.h>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>
#include "kernel_barrier.cuh"

namespace magi_attn_ext {

// ==========================================
// Kernel Definitions
// ==========================================

__global__ void wait_kernel(KernelBarrierView counter, int target) {
  // Only the first thread of the first block performs the wait operation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    counter.wait_impl(target);
  }
}

__global__ void producer_kernel(KernelBarrierView counter) {
  // Signal arrival by incrementing the counter
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    counter.arrive();
  }
}

// ==========================================
// Host Class Implementation
// ==========================================

KernelBarrier::KernelBarrier(int target) {
  this->target_val = target;

  // Initialize a scalar Int32 tensor on the current CUDA device to act as the counter.
  // Using torch::zeros ensures the memory is managed by PyTorch's allocator.
  this->buffer = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

KernelBarrier::KernelBarrier(const KernelBarrier& other) {
  this->target_val = other.target_val;

  // Shallow copy the tensor.
  // This ensures that multiple KernelBarrier objects (e.g., passed by value)
  // share the same underlying GPU storage/counter.
  this->buffer = other.buffer;
}

// Note: No explicit destructor is needed.
// The underlying GPU memory will be automatically freed by PyTorch
// when the last Tensor reference goes out of scope.

void KernelBarrier::reset() {
  // Reset the counter to 0 asynchronously using an in-place operation.
  this->buffer.zero_();
}

int KernelBarrier::get_value() {
  // Retrieve the value from Device to Host.
  // .item<int>() triggers a synchronous copy.
  return this->buffer.item<int>();
}

KernelBarrierView KernelBarrier::view() {
  // extract the raw device pointer from the Tensor to pass into CUDA kernels.
  return KernelBarrierView{this->buffer.data_ptr<int>()};
}

void KernelBarrier::synchronize() {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  // Launch the wait kernel to block execution on the GPU until the target is reached.
  wait_kernel<<<1, 1, 0, stream.stream()>>>(this->view(), this->target_val);
}

// ==========================================
// Global Producer Function
// ==========================================
void produce(std::optional<KernelBarrier>& counter) {
  if (!counter.has_value()) {
    return;
  }
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  // Launch the producer kernel to signal completion/arrival.
  producer_kernel<<<1, 1, 0, stream.stream()>>>(counter.value().view());
}

} // namespace magi_attn_ext
