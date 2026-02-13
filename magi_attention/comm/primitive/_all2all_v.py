# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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


import torch
import torch.distributed as dist

from magi_attention.utils import nvtx

from ..work import GeneralWork

__all__ = ["all2all_v"]


@torch.no_grad()
@nvtx.instrument_nvtx
def all2all_v(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    group: dist.ProcessGroup,
    async_op: bool = False,
) -> GeneralWork:
    """All-to-All-V

    Args:
        input (torch.Tensor): input tensor
        output (torch.Tensor): output tensor
        input_split_size_list (list[int]): input split size list
        output_split_size_list (list[int]): output split size list
        group (dist.ProcessGroup): process group
        async_op (bool, optional): whether to use async op. Defaults to False.

    Returns:
        work (GeneralWork): the work object with a ``wait`` method,
            which will wait for the comm kernel done and the output ready
    """

    assert (
        len(input_split_size_list)
        == len(output_split_size_list)
        == dist.get_world_size(group)
    )
    assert input.stride() == output.stride()

    work_nccl = dist.all_to_all_single(
        output=output,
        input=input,
        output_split_sizes=output_split_size_list,
        input_split_sizes=input_split_size_list,
        group=group,
        async_op=async_op,
    )

    work = GeneralWork(work=work_nccl)

    return work
