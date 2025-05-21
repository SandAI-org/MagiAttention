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

from collections import defaultdict
from itertools import accumulate, chain

import torch
import torch.distributed
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.comm.primitive import group_cast_collective, group_reduce_collective
from magi_attention.comm.primitive.utils import (
    _calc_group_cast_a2a_input_args,
    _calc_group_reduce_a2a_input_args,
    _reduce_to_tensor,
    _unpermute_tensor,
)
from magi_attention.testing.dist_common import DistTestBase, with_comms


def _seqlens2curanges(
    seqlens: list[int],
) -> list[tuple[int, int]]:
    cu_seqlens = list(accumulate(seqlens))
    return [
        (cu_seqlens[i - 1], cu_seqlens[i]) if i > 0 else (0, cu_seqlens[i])
        for i in range(len(cu_seqlens))
    ]


def _unpermute_tensor_ref(
    input_tensor: torch.Tensor,
    unperm_index: torch.LongTensor,
) -> torch.Tensor:
    """unpermute a2a output to output (deprecated as reference)
    as a post-processing func for group_cast_collective
    """

    return input_tensor.index_select(
        dim=0,
        index=unperm_index,
    )


def _calc_unpermute_index_tensor(
    tensor: torch.Tensor,
    unpermute_index_list: list[int],
    tensor_size_list: list[int],
) -> torch.LongTensor:
    tensor_cum_size_list = list(accumulate(tensor_size_list))
    tensor_size_ranges = [
        (tensor_cum_size_list[i - 1], tensor_cum_size_list[i])
        if i > 0
        else (0, tensor_cum_size_list[i])
        for i in range(len(tensor_cum_size_list))
    ]
    unperm_index_tensor = torch.tensor(
        list(
            chain(*(list(range(*tensor_size_ranges[i])) for i in unpermute_index_list))
        ),
        dtype=torch.int32,
        device=tensor.device,
    )

    return unperm_index_tensor


def _calc_group_cast_a2a_input_meta_args_ref(
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    world_size: int,
) -> tuple[list[int], torch.LongTensor]:
    input_size_ranges = _seqlens2curanges(input_split_size_list)

    a2a_input_size_repeat_ranges_with_rank = sorted(  # stable sort
        list(
            chain(
                *(
                    [(input_size_ranges[i], dst_rank) for dst_rank in dst_indices]
                    for i, dst_indices in enumerate(dst_indices_list)
                )
            )
        ),
        key=lambda x: x[1],
    )

    a2a_input_split_size_dict: dict[int, int] = defaultdict(int)
    for (start, end), rank in a2a_input_size_repeat_ranges_with_rank:
        a2a_input_split_size_dict[rank] += end - start
    a2a_input_split_size = [
        a2a_input_split_size_dict[rank] for rank in range(world_size)
    ]

    a2a_input_unperm_index_tensor = torch.tensor(
        list(
            chain(
                *(
                    list(range(*a2a_input_size_repeat_range_with_rank[0]))
                    for a2a_input_size_repeat_range_with_rank in a2a_input_size_repeat_ranges_with_rank
                )
            )
        ),
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )

    return (
        a2a_input_split_size,
        a2a_input_unperm_index_tensor,
    )


def _calc_group_cast_a2a_input_args_ref(
    input: torch.Tensor,
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    world_size: int,
) -> tuple[torch.Tensor, list[int]]:
    (
        a2a_input_split_size,
        a2a_input_unperm_index_tensor,
    ) = _calc_group_cast_a2a_input_meta_args_ref(
        input_split_size_list=input_split_size_list,
        dst_indices_list=dst_indices_list,
        world_size=world_size,
    )

    a2a_input = input.index_select(
        dim=0,
        index=a2a_input_unperm_index_tensor,
    )

    return a2a_input, a2a_input_split_size


def _calc_group_cast_a2a_output_meta_args_ref(
    output_split_size_list: list[int],
    src_index_list: list[int],
    world_size: int,
) -> tuple[list[int], torch.LongTensor]:
    a2a_output_split_size_per_rank: list[list[int]] = [[] for _ in range(world_size)]
    a2a_output_permute_index_list_per_rank: list[list[int]] = [
        [] for _ in range(world_size)
    ]
    for i, src_index in enumerate(src_index_list):
        a2a_output_split_size_per_rank[src_index].append(output_split_size_list[i])
        a2a_output_permute_index_list_per_rank[src_index].append(i)
    a2a_output_split_size: list[int] = [sum(x) for x in a2a_output_split_size_per_rank]
    a2a_output_tensor_size_list: list[int] = list(
        chain(*a2a_output_split_size_per_rank)
    )
    a2a_output_permute_index_list = list(chain(*a2a_output_permute_index_list_per_rank))
    a2a_output_unpermute_index_list: list[int] = sorted(
        range(len(a2a_output_permute_index_list)),
        key=lambda x: a2a_output_permute_index_list[x],
    )

    # ---------    calc post-process args    --------- #

    tensor_size_ranges = _seqlens2curanges(a2a_output_tensor_size_list)
    a2a_output_unperm_index_tensor = torch.tensor(
        list(
            chain(
                *(
                    list(range(*tensor_size_ranges[i]))
                    for i in a2a_output_unpermute_index_list
                )
            )
        ),
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )

    return (
        a2a_output_split_size,
        a2a_output_unperm_index_tensor,
    )


def _reduce_to_tensor_ref(
    output: torch.Tensor,
    a2a_output: torch.Tensor,
    reduce_index: torch.LongTensor,
) -> torch.Tensor:
    """sum-reduce a2a output to output (deprecated as reference)
    as a post-processing func for group_reduce_collective
    """

    return output.index_add_(
        dim=0,
        index=reduce_index,
        source=a2a_output,
    )


def _calc_reduce_index_tensor(
    a2a_output_unpermute_index_list: list[int],
    a2a_output_tensor_size_list: list[int],
    output_split_size_list: list[int],
    num_src_list: list[int],
) -> torch.LongTensor:
    tensor_size_ranges = _seqlens2curanges(a2a_output_tensor_size_list)
    output_size_ranges = _seqlens2curanges(output_split_size_list)
    cum_src_ranges = _seqlens2curanges(num_src_list)

    a2a_output_reduce_index = [0] * sum(a2a_output_tensor_size_list)
    for cum_src_range, output_size_range in zip(cum_src_ranges, output_size_ranges):
        output_size_range_idxs = list(range(*output_size_range))
        for idx in a2a_output_unpermute_index_list[cum_src_range[0] : cum_src_range[1]]:
            a2a_output_reduce_index[
                tensor_size_ranges[idx][0] : tensor_size_ranges[idx][1]
            ] = output_size_range_idxs

    a2a_output_reduce_index_tensor = torch.tensor(
        a2a_output_reduce_index,
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )

    return a2a_output_reduce_index_tensor


def _calc_group_reduce_a2a_input_meta_args_ref(
    input_split_size_list: list[int],
    dst_index_list: list[int],
    world_size: int,
) -> tuple[list[int], torch.LongTensor]:
    input_size_ranges = _seqlens2curanges(input_split_size_list)

    unperm_input_size_ranges_with_rank = sorted(
        [(input_size_ranges[i], dst_rank) for i, dst_rank in enumerate(dst_index_list)],
        key=lambda x: x[1],
    )

    a2a_input_split_size_dict: dict[int, int] = defaultdict(int)
    for (start, end), rank in unperm_input_size_ranges_with_rank:
        a2a_input_split_size_dict[rank] += end - start
    a2a_input_split_size = [
        a2a_input_split_size_dict[rank] for rank in range(world_size)
    ]

    a2a_input_unperm_index_tensor = torch.tensor(
        list(
            chain(
                *(
                    list(range(*unperm_input_size_range_with_rank[0]))
                    for unperm_input_size_range_with_rank in unperm_input_size_ranges_with_rank
                )
            )
        ),
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )

    return (
        a2a_input_split_size,
        a2a_input_unperm_index_tensor,
    )


def _calc_group_reduce_a2a_input_args_ref(
    input: torch.Tensor,
    input_split_size_list: list[int],
    dst_index_list: list[int],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int]]:
    a2a_input_split_size = kwargs.get("a2a_input_split_size", None)
    a2a_input_unperm_index_tensor = kwargs.get("a2a_input_unperm_index_tensor", None)

    if a2a_input_split_size is None or a2a_input_unperm_index_tensor is None:
        (
            a2a_input_split_size,
            a2a_input_unperm_index_tensor,
        ) = _calc_group_reduce_a2a_input_meta_args_ref(
            input_split_size_list=input_split_size_list,
            dst_index_list=dst_index_list,
            world_size=world_size,
        )

    a2a_input = input.index_select(dim=0, index=a2a_input_unperm_index_tensor)

    return a2a_input, a2a_input_split_size


def _calc_group_reduce_a2a_output_meta_args_ref(
    output_split_size_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
) -> tuple[list[int], torch.LongTensor]:
    a2a_output_split_size = [0 for _ in range(world_size)]
    size_src_index_i_list = []
    idx = 0
    for output_split_size, src_indices in zip(output_split_size_list, src_indices_list):
        for src_index in src_indices:
            a2a_output_split_size[src_index] += output_split_size
            size_src_index_i_list.append((output_split_size, src_index, idx))
            idx += 1
    size_src_index_i_list.sort(key=lambda x: x[1])
    a2a_output_permute_index_list = [x[2] for x in size_src_index_i_list]
    a2a_output_unpermute_index_list = sorted(
        range(len(a2a_output_permute_index_list)),
        key=lambda x: a2a_output_permute_index_list[x],
    )
    a2a_output_tensor_size_list = [x[0] for x in size_src_index_i_list]

    # ---------    calc post-process args    --------- #

    num_src_list = [len(src_indices) for src_indices in src_indices_list]
    tensor_size_ranges = _seqlens2curanges(a2a_output_tensor_size_list)
    output_size_ranges = _seqlens2curanges(output_split_size_list)
    cum_src_ranges = _seqlens2curanges(num_src_list)

    a2a_output_reduce_index = [0] * sum(a2a_output_tensor_size_list)
    for cum_src_range, output_size_range in zip(cum_src_ranges, output_size_ranges):
        output_size_range_idxs = list(range(*output_size_range))
        for idx in a2a_output_unpermute_index_list[cum_src_range[0] : cum_src_range[1]]:
            a2a_output_reduce_index[
                tensor_size_ranges[idx][0] : tensor_size_ranges[idx][1]
            ] = output_size_range_idxs

    a2a_output_reduce_index_tensor = torch.tensor(
        a2a_output_reduce_index,
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )

    return (
        a2a_output_split_size,
        a2a_output_reduce_index_tensor,
    )


def _calc_group_reduce_a2a_output_phase2_meta_args_ref(
    a2a_output_tensor_size_list: list[int],
    a2a_output_unpermute_index_list: list[int],
    output_split_size_list: list[int],
    num_src_list: list[int],
):
    a2a_output_size_ranges = _seqlens2curanges(a2a_output_tensor_size_list)
    output_size_ranges = _seqlens2curanges(output_split_size_list)
    cum_src_ranges = _seqlens2curanges(num_src_list)
    a2a_output_reduce_ranges_list = []
    for start, end in cum_src_ranges:
        a2a_output_reduce_ranges_list.append(
            [
                a2a_output_size_ranges[index]
                for index in a2a_output_unpermute_index_list[start:end]
            ]
        )

    return (
        a2a_output_reduce_ranges_list,
        output_size_ranges,
    )


class TestMultiCastCollective(DistTestBase):
    def init_pg(self):
        super().init_pg()

        # -----    set up for hier comm   ---- #

        if magi_attention.is_hierarchical_comm_enable() and self.world_size in [
            4,
            6,
            8,
        ]:
            world_size_inter_node, world_size_intra_node = {
                4: (2, 2),
                6: (3, 2),
                8: (2, 4),
            }[self.world_size]
            device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(world_size_inter_node, world_size_intra_node),
                mesh_dim_names=("inter", "intra"),
            )
            self.intra_group = device_mesh.get_group("intra")
            self.inter_group = device_mesh.get_group("inter")
        else:
            self.intra_group = None
            self.inter_group = None

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_naive_group_cast_like_a2a(self):
        dtype = torch.int32
        device = torch.cuda.current_device()

        input_tensor = torch.tensor(
            [self.rank] * self.world_size, dtype=dtype, device=device
        )
        output_tensor = torch.tensor([-1] * self.world_size, dtype=dtype, device=device)
        input_split_size_list = [1] * self.world_size
        output_split_size_list = [1] * self.world_size
        dst_indices_list = [[i] for i in range(self.world_size)]
        # dst_indices_list:
        # r0, r1, r2, r3: [[0],
        #                  [1],
        #                  [2],
        #                  [3]]
        src_index_list = [i for i in range(self.world_size)]
        # src_index_list:
        # r0, r1, r2, r3: [0, 1, 2, 3]

        work = group_cast_collective(
            input=input_tensor,
            output=output_tensor,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            group=self.process_group,
            async_op=True,
        )
        output_tensor = work.wait_post_process(output_tensor)

        expected_output = torch.tensor(
            [i for i in range(self.world_size)], dtype=dtype, device=device
        )
        # expected_output:
        # r0, r1, r2, r3: [0, 1, 2, 3]
        self.assertTrue(torch.equal(output_tensor, expected_output))

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_naive_group_reduce(self):
        dtype = torch.int32
        device = torch.cuda.current_device()

        # Do group-reduce collective as following:
        # r0: input: [0, 0, 0, 0] output: [0, 0, 0, 0] ---> [0, 1, 2, 3]
        # r1: input: [1, 1, 1, 1] output: [1, 1, 1, 1] ---> [1, 2, 3, 4]
        # r2: input: [2, 2, 2, 2] output: [2, 2, 2, 2] ---> [2, 3, 4, 5]
        # r3: input: [3, 3, 3, 3] output: [3, 3, 3, 3] ---> [3, 4, 5, 6]
        expected_tensor_per_rank = [
            torch.tensor([0, 1, 2, 3], dtype=dtype, device=device),
            torch.tensor([1, 2, 3, 4], dtype=dtype, device=device),
            torch.tensor([2, 3, 4, 5], dtype=dtype, device=device),
            torch.tensor([3, 4, 5, 6], dtype=dtype, device=device),
        ]

        input_tensor = torch.tensor(
            [self.rank] * self.world_size, dtype=dtype, device=device
        )
        output_tensor = torch.tensor(
            [self.rank] * self.world_size, dtype=dtype, device=device
        )
        input_split_size_list = [1] * self.world_size
        output_split_size_list = [1] * self.world_size
        src_indices_list = [[i] for i in range(self.world_size)]
        # src_indices_list:
        # r0, r1, r2, r3: [[0],
        #                  [1],
        #                  [2],
        #                  [3]]
        dst_index_list = [i for i in range(self.world_size)]
        # dst_index_list:
        # r0, r1, r2, r3: [0, 1, 2, 3]

        work = group_reduce_collective(
            input=input_tensor,
            output=output_tensor,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_index_list=dst_index_list,
            src_indices_list=src_indices_list,
            group=self.process_group,
            async_op=True,
        )
        output_tensor = work.wait_post_process(output_tensor)

        self.assertTrue(torch.equal(output_tensor, expected_tensor_per_rank[self.rank]))

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_group_cast_collective(self):
        dtype = torch.int32
        device = torch.cuda.current_device()

        # Do multi-cast collective as following:
        # r0: [0, 1, 2, 3] -------> [5, 9, 13]
        # r1: [4, 5, 6, 7] -------> [0, 1, 10, 11, 2, 12, 13]
        # r2: [8, 9, 10, 11] -----> [2, 3, 6, 7, 14, 15]
        # r3: [12, 13, 14, 15] ---> [4, 5, 8, 9]
        expected_tensor_per_rank = [
            torch.tensor([5, 9, 13], dtype=dtype, device=device),
            torch.tensor([0, 1, 10, 11, 2, 12, 13], dtype=dtype, device=device),
            torch.tensor([2, 3, 6, 7, 14, 15], dtype=dtype, device=device),
            torch.tensor([4, 5, 8, 9], dtype=dtype, device=device),
        ]
        input_split_size_list_per_rank = [[2, 1, 1], [1, 1, 2], [1, 1, 2], [1, 1, 2]]
        output_split_size_list_per_rank = [
            [1, 1, 1],
            [2, 2, 1, 2],
            [1, 1, 2, 2],
            [1, 1, 1, 1],
        ]
        dst_indices_list_per_rank = [
            [[1], [1, 2], [2]],
            [[3], [0, 3], [2]],
            [[3], [0, 3], [1]],
            [[1], [0, 1], [2]],
        ]
        src_index_list_per_rank = [[1, 2, 3], [0, 2, 0, 3], [0, 0, 1, 3], [1, 1, 2, 2]]
        input_tensor = torch.tensor(
            [
                i
                for i in range(
                    self.world_size * self.rank, self.world_size * (self.rank + 1)
                )
            ],
            dtype=dtype,
            device=device,
        )
        output_tensor = torch.full_like(
            expected_tensor_per_rank[self.rank], -1, dtype=dtype, device=device
        )
        input_split_size_list = input_split_size_list_per_rank[self.rank]
        output_split_size_list = output_split_size_list_per_rank[self.rank]
        dst_indices_list = dst_indices_list_per_rank[self.rank]
        src_index_list = src_index_list_per_rank[self.rank]

        expected_tensor = expected_tensor_per_rank[self.rank]

        work = group_cast_collective(
            input=input_tensor,
            output=output_tensor,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            group=self.process_group,
            async_op=True,
            intra_group=self.intra_group,
            inter_group=self.inter_group,
        )
        output_tensor = work.wait_post_process(output_tensor)

        self.assertTrue(torch.equal(output_tensor, expected_tensor))

    def test_unpermute_tensor(self):
        # ---------    normal unperm idxs     --------- #

        x = torch.randn(6, 3, 4, device=torch.cuda.current_device())

        unperm_idxs1 = [0, 2, 1]
        split_sizes1 = [2, 1, 3]
        y1_ref = _unpermute_tensor_ref(
            input_tensor=x,
            unperm_index=_calc_unpermute_index_tensor(x, unperm_idxs1, split_sizes1),
        )
        y1 = _unpermute_tensor(
            x,
            unperm_after_a2a_kwargs={
                "ranges": torch.tensor(
                    [[0, 2], [3, 6], [2, 3]],
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                ),
                "cu_range_sizes": torch.tensor(
                    [0, 2, 5, 6], dtype=torch.int32, device=torch.cuda.current_device()
                ),
                "total_size": 6,
                "dim": 0,
            },
        )

        self.assertTrue(y1.is_contiguous())
        self.assertTrue(torch.equal(y1, y1_ref))
        self.assertTrue(
            torch.equal(
                y1,
                torch.cat(
                    [  # split_sizes_after_unperm = (2,3,1)
                        x[:2],
                        x[-3:],
                        x[2:3],
                    ]
                ),
            )
        )

        unperm_idxs2 = [2, 1, 0]
        split_sizes2 = [2, 3, 1]
        y2_ref = _unpermute_tensor_ref(
            input_tensor=x,
            unperm_index=_calc_unpermute_index_tensor(x, unperm_idxs2, split_sizes2),
        )
        y2 = _unpermute_tensor(
            x,
            unperm_after_a2a_kwargs={
                "ranges": torch.tensor(
                    [[5, 6], [2, 5], [0, 2]],
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                ),
                "cu_range_sizes": torch.tensor(
                    [0, 1, 4, 6], dtype=torch.int32, device=torch.cuda.current_device()
                ),
                "total_size": 6,
                "dim": 0,
            },
        )
        self.assertTrue(y2.is_contiguous())
        self.assertTrue(torch.equal(y2, y2_ref))
        self.assertTrue(
            torch.equal(
                y2,
                torch.cat(
                    [  # split_sizes_after_unperm = (1,3,2)
                        x[-1:],
                        x[2:-1],
                        x[:2],
                    ]
                ),
            )
        )

        unperm_idxs3 = [2, 0, 1]
        split_sizes3 = [3, 1, 2]
        y3_ref = _unpermute_tensor_ref(
            input_tensor=x,
            unperm_index=_calc_unpermute_index_tensor(x, unperm_idxs3, split_sizes3),
        )
        y3 = _unpermute_tensor(
            x,
            unperm_after_a2a_kwargs={
                "ranges": torch.tensor(
                    [[4, 6], [0, 3], [3, 4]],
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                ),
                "cu_range_sizes": torch.tensor(
                    [0, 2, 5, 6], dtype=torch.int32, device=torch.cuda.current_device()
                ),
                "total_size": 6,
                "dim": 0,
            },
        )
        self.assertTrue(y3_ref.is_contiguous())
        self.assertTrue(torch.equal(y3, y3_ref))
        self.assertTrue(
            torch.equal(
                y3,
                torch.cat(
                    [  # split_sizes_after_unperm = (2,3,1)
                        x[-2:],
                        x[:3],
                        x[3:4],
                    ]
                ),
            )
        )

        # ---------    empty unperm idxs     --------- #

        x = torch.randn(6, 3, 4, device=torch.cuda.current_device())
        emp = torch.empty(0, 3, 4, device=torch.cuda.current_device())
        unperm_idxs4 = []
        split_sizes4 = [1, 2, 3]

        y4_ref = _unpermute_tensor_ref(
            input_tensor=x,
            unperm_index=_calc_unpermute_index_tensor(x, unperm_idxs4, split_sizes4),
        )
        y4 = _unpermute_tensor(
            x,
            unperm_after_a2a_kwargs={
                "ranges": torch.tensor(
                    [], dtype=torch.int32, device=torch.cuda.current_device()
                ),
                "cu_range_sizes": torch.tensor(
                    [], dtype=torch.int32, device=torch.cuda.current_device()
                ),
                "total_size": 0,
                "dim": 0,
            },
        )
        self.assertTrue(y4.is_contiguous())
        self.assertTrue(torch.equal(y4, y4_ref))
        self.assertTrue(torch.equal(y4, emp))

    def test_reduce_to_tensor(self):
        # ---------    init data     --------- #

        h = 128
        a2a_output_tensor_size_list = [4, 2, 3, 3, 4]  # [3, 3, 4, 4, 2]
        a2a_output_unpermute_index_list = [3, 2, 0, 4, 1]
        num_src_list = [2, 2, 1]
        output_split_size_list = [3, 4, 2]

        a2a_output = torch.randn(
            sum(a2a_output_tensor_size_list), h, device=torch.cuda.current_device()
        )
        output = torch.randn(
            sum(output_split_size_list), h, device=torch.cuda.current_device()
        )
        output_ref = output.clone()

        # ---------    ref     --------- #

        reduce_index_tensor = _calc_reduce_index_tensor(
            a2a_output_unpermute_index_list=a2a_output_unpermute_index_list,
            a2a_output_tensor_size_list=a2a_output_tensor_size_list,
            output_split_size_list=output_split_size_list,
            num_src_list=num_src_list,
        )

        reduced_output_ref = _reduce_to_tensor_ref(
            output=output,
            a2a_output=a2a_output,
            reduce_index=reduce_index_tensor,
        )

        # ---------    impl     --------- #

        (
            a2a_output_reduce_ranges_list,
            output_size_ranges,
        ) = _calc_group_reduce_a2a_output_phase2_meta_args_ref(
            a2a_output_tensor_size_list=a2a_output_tensor_size_list,
            a2a_output_unpermute_index_list=a2a_output_unpermute_index_list,
            output_split_size_list=output_split_size_list,
            num_src_list=num_src_list,
        )

        # calc range_reduce kwargs
        input_ranges = []
        output_ranges = []
        cu_range_sizes = [0]
        total_size = 0
        for (out_start, out_end), reduce_ranges in zip(
            output_size_ranges, a2a_output_reduce_ranges_list
        ):
            for reduce_start, reduce_end in reduce_ranges:
                input_ranges.append([reduce_start, reduce_end])
                output_ranges.append([out_start, out_end])
                cu_range_sizes.append(reduce_end - reduce_start)
                total_size += reduce_end - reduce_start

        input_ranges = torch.tensor(input_ranges, dtype=torch.int32)
        output_ranges = torch.tensor(output_ranges, dtype=torch.int32)
        cu_range_sizes = torch.tensor(cu_range_sizes, dtype=torch.int32)
        cu_range_sizes = torch.cumsum(cu_range_sizes, dim=0)

        range_reduce_kwargs = {
            "input_ranges": input_ranges.to(torch.cuda.current_device()),
            "output_ranges": output_ranges.to(torch.cuda.current_device()),
            "cu_range_sizes": cu_range_sizes.to(torch.cuda.current_device()),
            "total_size": total_size,
        }

        reduced_output = _reduce_to_tensor(
            output=output_ref,
            a2a_output=a2a_output,
            range_reduce_kwargs=range_reduce_kwargs,
        )

        # ---------    check     --------- #

        # NOTE: since the add order is different, we can not expect all-equal but all-close
        # self.assertTrue(torch.equal(reduced_output_ref, reduced_output))
        torch.testing.assert_close(
            reduced_output_ref,
            reduced_output,
        )

    def test_calc_group_cast_a2a_input_args(self):
        # ---------    normal dst indices list     --------- #

        # ---------    init data     --------- #

        h = 128
        input_split_size_list = [3, 4, 2, 6, 1]
        dst_indices_list = [[0, 1], [2, 3], [1, 2, 3], [], [0, 1, 2, 3]]
        world_size = 4
        tensor = torch.randn(
            (sum(input_split_size_list), h), device=torch.cuda.current_device()
        )

        # ---------    ref     --------- #

        a2a_input_ref, a2a_input_split_size_ref = _calc_group_cast_a2a_input_args_ref(
            input=tensor,
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            world_size=world_size,
        )

        # ---------    impl     --------- #

        a2a_input, a2a_input_split_size = _calc_group_cast_a2a_input_args(
            input=tensor,
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            world_size=world_size,
        )

        # ---------    check     --------- #

        self.assertTrue(torch.equal(a2a_input_ref, a2a_input))
        self.assertEqual(a2a_input_split_size_ref, a2a_input_split_size)

        # ---------    empty dst indices list     --------- #

        # ---------    init data     --------- #

        h = 128
        input_split_size_list = [3, 4, 2, 6, 1]
        dst_indices_list = [[], [], [], [], []]
        world_size = 4
        tensor = torch.randn(
            (sum(input_split_size_list), h), device=torch.cuda.current_device()
        )

        # ---------    ref     --------- #

        a2a_input_ref, a2a_input_split_size_ref = _calc_group_cast_a2a_input_args_ref(
            input=tensor,
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            world_size=world_size,
        )

        # ---------    impl     --------- #

        a2a_input, a2a_input_split_size = _calc_group_cast_a2a_input_args(
            input=tensor,
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            world_size=world_size,
        )

        # ---------    check     --------- #

        self.assertTrue(torch.equal(a2a_input_ref, a2a_input))
        self.assertEqual(a2a_input_split_size_ref, a2a_input_split_size)

    def test_calc_group_reduce_a2a_input_args(self):
        # ---------    init data     --------- #

        h = 128
        input_split_size_list = [3, 4, 2, 6, 1]
        dst_index_list = [0, 3, 1, 2, 0]
        world_size = 4
        tensor = torch.randn(
            (sum(input_split_size_list), h), device=torch.cuda.current_device()
        )

        # ---------    ref     --------- #

        a2a_input_ref, a2a_input_split_size_ref = _calc_group_reduce_a2a_input_args_ref(
            input=tensor,
            input_split_size_list=input_split_size_list,
            dst_index_list=dst_index_list,
            world_size=world_size,
        )

        # ---------    impl     --------- #

        a2a_input, a2a_input_split_size = _calc_group_reduce_a2a_input_args(
            input=tensor,
            input_split_size_list=input_split_size_list,
            dst_index_list=dst_index_list,
            world_size=world_size,
        )

        # ---------    check     --------- #

        self.assertTrue(torch.equal(a2a_input_ref, a2a_input))
        self.assertEqual(a2a_input_split_size_ref, a2a_input_split_size)


if __name__ == "__main__":
    run_tests()
