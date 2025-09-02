# Copyright (c) 2025 DeepSeek. All Rights Reserved.
#
# Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(add_deep_ep_library target_name source_file)
    add_library(${target_name} STATIC ${source_file})
    set_target_properties(${target_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 17
            CUDA_STANDARD 17
            CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${target_name} PUBLIC nvshmem cudart cudadevrt mlx5)
endfunction()

add_deep_ep_library(runtime_cuda runtime.cu)
add_deep_ep_library(layout_cuda layout.cu)
add_deep_ep_library(intranode_cuda intranode.cu)
add_deep_ep_library(internode_cuda internode.cu)
add_deep_ep_library(internode_ll_cuda internode_ll.cu)

# Later, we should link all libraries in `EP_CUDA_LIBRARIES`
set(EP_CUDA_LIBRARIES runtime_cuda layout_cuda intranode_cuda internode_cuda internode_ll_cuda PARENT_SCOPE)
