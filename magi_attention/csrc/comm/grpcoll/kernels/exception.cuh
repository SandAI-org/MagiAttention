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

#include <exception>
#include <string>

#include "configs.cuh"

#ifndef GRPCOLL_STATIC_ASSERT
#define GRPCOLL_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class GrpCollException : public std::exception {
 private:
  std::string message = {};

 public:
  explicit GrpCollException(const char* name, const char* file, const int line, const std::string& error) {
    message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
  }

  const char* what() const noexcept override {
    return message.c_str();
  }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                          \
  do {                                                                           \
    cudaError_t e = (cmd);                                                       \
    if (e != cudaSuccess) {                                                      \
      throw GrpCollException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                            \
  } while (0)
#endif

#ifndef GRPCOLL_HOST_ASSERT
#define GRPCOLL_HOST_ASSERT(cond, msg)                                                           \
  do {                                                                                           \
    if (not(cond)) {                                                                             \
      throw GrpCollException("Assertion", __FILE__, __LINE__, std::string(#cond) + " " + (msg)); \
    }                                                                                            \
  } while (0)
#endif

#ifndef GRPCOLL_DEVICE_ASSERT
#define GRPCOLL_DEVICE_ASSERT(cond, fmt, ...)                                                                         \
  do {                                                                                                                \
    if (!(cond)) {                                                                                                    \
      printf("Assertion failed: %s:%d, condition: %s, message: " fmt "\n", __FILE__, __LINE__, #cond, ##__VA_ARGS__); \
      asm("trap;");                                                                                                   \
    }                                                                                                                 \
  } while (0)
#endif
