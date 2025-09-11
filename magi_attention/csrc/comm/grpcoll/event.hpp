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

#include <ATen/cuda/CUDAContext.h>
#include <memory>

#include "kernels/exception.cuh"

namespace magi_attn_comm::grpcoll {

struct EventHandle {
  std::shared_ptr<torch::Event> event;

  EventHandle() {
    event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(at::cuda::getCurrentCUDAStream());
  }

  explicit EventHandle(const at::cuda::CUDAStream& stream) {
    event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(stream);
  }

  EventHandle(const EventHandle& other) = default;

  void current_stream_wait() const {
    at::cuda::getCurrentCUDAStream().unwrap().wait(*event);
  }
};

torch::Event create_event(const at::cuda::CUDAStream& s) {
  auto event = torch::Event(torch::kCUDA);
  event.record(s);
  return event;
}

void stream_wait(const at::cuda::CUDAStream& s_0, const at::cuda::CUDAStream& s_1) {
  EP_HOST_ASSERT(s_0.id() != s_1.id());
  s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const at::cuda::CUDAStream& s, const EventHandle& event) {
  s.unwrap().wait(*event.event);
}

} // namespace magi_attn_comm::grpcoll
