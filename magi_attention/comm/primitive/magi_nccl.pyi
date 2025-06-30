# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import timedelta

import torch
import torch.distributed as dist

class GroupCastOptions:
    timeout: timedelta
    asyncOp: bool

class GroupReduceOptions:
    timeout: timedelta
    asyncOp: bool

# NOTE: MagiNCCLBackend is NOT a subclass of ProcessGroupNCCL
class MagiNCCLBackend(dist.Backend):
    def __init__(
        self,
        store: dist.Store,
        rank: int,
        size: int,
    ) -> None: ...
    @property
    def nccl_stream(self) -> torch.cuda.Stream: ...
    def group_cast(
        input: torch.Tensor,
        output: torch.Tensor,
        input_split_size_list: list[int],
        output_split_size_list: list[int],
        dst_indices_list: list[list[int]],
        src_index_list: list[int],
        opts=...,
        **kwargs,
    ) -> dist.Work: ...
    def group_reduce(
        input: torch.Tensor,
        output: torch.Tensor,
        input_split_size_list: list[int],
        output_split_size_list: list[int],
        dst_index_list: list[int],
        src_indices_list: list[list[int]],
        opts=...,
        **kwargs,
    ) -> dist.Work: ...
