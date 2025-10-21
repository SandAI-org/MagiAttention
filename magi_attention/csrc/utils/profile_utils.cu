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

#include <iostream>
#include <stdexcept>
#include "cuda_check.h"
#include "profile_utils.h"

std::map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> MagiEvents::magi_events;

void MagiEvents::destroy() {
  for (auto const& [key, val] : magi_events) {
    CHECK_CUDA(cudaEventDestroy(val.first));
    CHECK_CUDA(cudaEventDestroy(val.second));
  }
  magi_events.clear();
}

void MagiEvents::start(const std::string& key) {
  auto it = magi_events.find(key);
  if (it == magi_events.end()) {
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    it = magi_events.emplace(key, std::make_pair(start_event, stop_event)).first;
  }

  CHECK_CUDA(cudaEventRecord(it->second.first, 0));
}

void MagiEvents::stop(const std::string& key) {
  auto it = magi_events.find(key);
  if (it == magi_events.end()) {
    throw std::runtime_error("[MagiEvents] Error: You must call start() for key '" + key + "' before stop().");
  }

  CHECK_CUDA(cudaEventRecord(it->second.second, 0));
}

float MagiEvents::elapsed_ms(const std::string& key) {
  auto it = magi_events.find(key);
  if (it == magi_events.end()) {
    return -1;
  }
  CHECK_CUDA(cudaEventSynchronize(it->second.second));
  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, it->second.first, it->second.second));

  return milliseconds;
}
