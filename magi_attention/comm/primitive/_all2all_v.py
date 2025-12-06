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

from typing import Literal

import torch
import torch.distributed as dist

from magi_attention.utils import nvtx

from ..work import GeneralWork

__all__ = ["all2all_v"]


def _calc_a2v_comm_bytes(
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    stride0: int,
    dtype: torch.dtype,
    rank: int,
    reduce_op: Literal["sum", "max"] = "max",
) -> int:
    num_loads = sum(input_split_size_list) - input_split_size_list[rank]
    num_stores = sum(output_split_size_list) - output_split_size_list[rank]

    total_num = 0
    match reduce_op:
        case "sum":
            total_num = num_loads + num_stores
        case "max":
            total_num = max(num_loads, num_stores)
        case _:
            raise ValueError(f"Invalid reduce_op: {reduce_op}")

    return total_num * stride0 * dtype.itemsize


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

    a2av_comm_bytes = _calc_a2v_comm_bytes(
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        stride0=input.stride(0),
        dtype=input.dtype,
        rank=dist.get_rank(group),
    )

    # only for debug
    # if dist.get_rank(group) == 0 and a2av_comm_bytes != 0:
    #     print(f"{a2av_comm_bytes=}")

    with nvtx.add_nvtx_event(
        (
            f"a2av: "
            f"{a2av_comm_bytes=} | "
            f"{input.shape=} | "
            f"{output.shape=} | "
            f"{input_split_size_list=} | "
            f"{output_split_size_list=}"
        )
    ):
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
