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

from functools import partial
from itertools import chain
from typing import Callable, overload

import torch
import torch.distributed as dist

import magi_attention
from magi_attention.comm.work import GeneralWork, WorkWithPostProcessFn
from magi_attention.common.enum import GroupReduceOp
from magi_attention.common.range_op import range_gather, range_reduce
from magi_attention.utils import is_list_type_all, nvtx

from .._all2all_v import all2all_v
from .utils import (
    _calc_group_cast_a2a_input_meta_args,
    _calc_group_cast_a2a_output_meta_args,
    _calc_group_reduce_a2a_input_meta_args,
    _calc_group_reduce_a2a_output_meta_args,
    _calc_unperm_range_gather_kwargs_from_split_size_list,
    sum_reduce_to_tensor,
    unpermute_tensor,
)

__all__ = [
    "init_hier_group_cast_meta_solver",
    "hier_group_cast_impl_with_a2av",
    "init_hier_group_reduce_meta_solver",
    "hier_group_reduce_impl_with_a2av",
]

# ------------------        hierarchical a2av group cast       ------------------ #


class HierGroupCastMetaSolver:
    def __init__(
        self,
        input_split_size_list: list[int],
        output_split_size_list: list[int],
        dst_indices_list: list[list[int]],
        src_index_list: list[int],
        rank: int,
        world_size: int,
        intra_group: dist.ProcessGroup,
        inter_group: dist.ProcessGroup,
        use_a2av_impl: bool = True,
    ):
        self.use_a2av_impl = use_a2av_impl

        # --------   prepare env info  -------- #

        self._prepare_env_info(rank, world_size, intra_group, inter_group)

        # ----   build group-cast meta for pre-intra  ---- #

        self._build_group_cast_meta_args_pre_intra(
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            output_split_size_list=output_split_size_list,
            src_index_list=src_index_list,
            return_local_rank=True,
        )

        # ----   build a2a meta for pre-intra  ---- #

        if self.use_a2av_impl:
            self._build_group_cast_a2a_meta_args_pre_intra()

        # ----   build group-cast meta for inter + half post-intra  ---- #

        self._build_group_cast_meta_args_inter_plus_half_post_intra(
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            return_local_rank=True,
        )

        # ----   build a2a meta for inter  ---- #

        if self.use_a2av_impl:
            self._build_group_cast_a2a_meta_args_inter()

        # ----   build group-cast meta for post-intra  ---- #

        self._build_group_cast_meta_args_rest_half_post_intra(
            return_local_rank=True,
        )

        # ----   build a2a meta for post-intra  ---- #

        if self.use_a2av_impl:
            self._build_group_cast_a2a_meta_args_post_intra()

        # ----   build hier group-cast a2a post-process fn ---- #

        if self.use_a2av_impl:
            self._build_group_cast_a2a_post_process_fn(
                output_split_size_list=output_split_size_list,
                src_index_list=src_index_list,
            )

    def get_group_cast_meta_pre_intra(self):
        return (
            self.input_split_size_list_pre_intra,
            self.output_split_size_list_pre_intra,
            self.dst_indices_list_pre_intra,
            self.src_index_list_pre_intra,
        )

    def get_group_cast_meta_inter(self):
        return (
            self.input_split_size_list_inter,
            self.output_split_size_list_inter,
            self.dst_indices_list_inter,
            self.src_index_list_inter,
        )

    def get_group_cast_meta_post_intra(self):
        return (
            self.input_split_size_list_post_intra,
            self.output_split_size_list_post_intra,
            self.dst_indices_list_post_intra,
            self.src_index_list_post_intra,
        )

    def make_group_cast_a2a_post_process_fn(
        self,
        stashed_tensors: list[torch.Tensor] | None = None,
    ):
        core_post_process_fn_hier = partial(
            unpermute_tensor,
            unperm_after_a2a_kwargs=self.unperm_after_a2a_kwargs_hier,
        )

        def post_process_fn_hier(tensor: torch.Tensor):
            try:
                return core_post_process_fn_hier(tensor)
            finally:
                nonlocal stashed_tensors
                if stashed_tensors is not None:
                    stashed_tensors.clear()

        return post_process_fn_hier

    def _prepare_env_info(
        self,
        rank: int,
        world_size: int,
        intra_group: dist.ProcessGroup,
        inter_group: dist.ProcessGroup,
    ):
        self.rank = rank
        self.world_size = world_size
        self.intra_group = intra_group
        self.inter_group = inter_group

        self.device = torch.cuda.current_device()

        self.world_size_intra_node = dist.get_world_size(intra_group)
        self.world_size_inter_node = dist.get_world_size(inter_group)
        assert (
            self.world_size == self.world_size_intra_node * self.world_size_inter_node
        ), (
            f"The {self.world_size=} should be equal to the product of "
            f"{self.world_size_inter_node=} x {self.world_size_intra_node=}."
        )

        self.local_rank_intra_node = self._to_local_rank_intra_node(self.rank)
        self.local_rank_inter_node = self._to_local_rank_inter_node(self.rank)

        self.first_rank_this_intra_node = (
            self.local_rank_inter_node * self.world_size_intra_node
        )
        self.last_rank_this_intra_node = (
            self.first_rank_this_intra_node + self.world_size_intra_node - 1
        )

    def _to_local_rank_intra_node(self, r):
        return r % self.world_size_intra_node

    def _to_local_rank_inter_node(self, r):
        return r // self.world_size_intra_node

    def _is_rank_within_this_intra_node(self, r):
        return self.first_rank_this_intra_node <= r <= self.last_rank_this_intra_node

    def _to_rank_with_same_local_rank_intra_node(self, r):
        return (
            self._to_local_rank_inter_node(r) * self.world_size_intra_node
            + self.local_rank_intra_node
        )

    def _filter_dst_indices_within_this_intra_node(
        self, dst_indices: list[int], return_local_rank: bool = False
    ) -> list[int]:
        return list(
            map(
                self._to_local_rank_intra_node if return_local_rank else lambda r: r,
                filter(
                    self._is_rank_within_this_intra_node,
                    dst_indices,
                ),
            )
        )

    def _build_group_cast_meta_args_pre_intra(
        self,
        input_split_size_list: list[int],
        dst_indices_list: list[list[int]],
        output_split_size_list: list[int],
        src_index_list: list[int],
        return_local_rank: bool = False,
    ):
        # input split size list does not need to change
        input_split_size_list_pre_intra = input_split_size_list

        # filter dst indices list within this intra node
        dst_indices_list_pre_intra = [
            self._filter_dst_indices_within_this_intra_node(
                dst_indices=dst_indices,
                return_local_rank=return_local_rank,
            )
            for dst_indices in dst_indices_list
        ]

        # filter src index list along with output split size list
        # within this intra node
        src_index_list_pre_intra = []
        output_split_size_list_pre_intra = []
        for src_index, output_split_size in zip(src_index_list, output_split_size_list):
            if self._is_rank_within_this_intra_node(src_index):
                src_index_list_pre_intra.append(
                    self._to_local_rank_intra_node(src_index)
                    if return_local_rank
                    else src_index
                )
                output_split_size_list_pre_intra.append(output_split_size)

        # set pre-intra group-cast meta
        self.a2a_output_seqlen_pre_intra = sum(output_split_size_list_pre_intra)
        self.input_split_size_list_pre_intra = input_split_size_list_pre_intra
        self.dst_indices_list_pre_intra = dst_indices_list_pre_intra
        self.output_split_size_list_pre_intra = output_split_size_list_pre_intra
        self.src_index_list_pre_intra = src_index_list_pre_intra

    def _build_group_cast_a2a_meta_args_pre_intra(self):
        (
            a2a_input_split_size_pre_intra,
            perm_range_gather_kwargs_pre_intra,
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_pre_intra,
            dst_indices_list=self.dst_indices_list_pre_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
        )
        (
            a2a_output_split_size_pre_intra,
            _,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_pre_intra,
            src_index_list=self.src_index_list_pre_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
            calc_unperm_after_a2a_kwargs=False,
        )

        # set pre-intra a2a meta
        self.a2a_input_split_size_pre_intra = a2a_input_split_size_pre_intra
        self.a2a_output_split_size_pre_intra = a2a_output_split_size_pre_intra
        self.perm_range_gather_kwargs_pre_intra = perm_range_gather_kwargs_pre_intra

    def _build_group_cast_meta_args_inter_plus_half_post_intra(
        self,
        input_split_size_list: list[int],
        dst_indices_list: list[list[int]],
        return_local_rank: bool = False,
    ):
        # input split size list does not need to change
        input_split_size_list_inter = input_split_size_list

        # build dst indices list within this inter node
        dst_indices_list_inter = [
            list(
                set(
                    map(
                        self._to_local_rank_inter_node
                        if return_local_rank
                        else lambda r: r,
                        map(
                            self._to_rank_with_same_local_rank_intra_node,
                            filter(
                                lambda r: not self._is_rank_within_this_intra_node(r),
                                dst_indices,
                            ),
                        ),
                    )
                )
            )
            for dst_indices in dst_indices_list
        ]

        # all gather send info per rank within this inter node
        send_info_this_rank_inter = (
            input_split_size_list_inter,
            dst_indices_list_inter,
            dst_indices_list,
        )
        send_info_per_rank_inter: list[
            tuple[list[int], list[list[int]], list[list[int]]]
        ] = [
            None  # type: ignore
        ] * self.world_size_inter_node
        dist.all_gather_object(
            send_info_per_rank_inter,
            send_info_this_rank_inter,
            group=self.inter_group,
        )

        # build naive src index list and output split size list
        # within this inter node
        # as well as preparing group cast input meta args for the third step
        src_index_list_inter = []
        output_split_size_list_inter = []
        dst_indices_list_post_intra = []
        is_this_rank_in_dst_indices_inter = (  # noqa: E731
            lambda dst_indices: (self.local_rank_inter_node in dst_indices)
            if return_local_rank
            else (self.rank in dst_indices)
        )
        for i, (
            input_split_size_list_inter_ith_rank,
            dst_indices_list_inter_ith_rank,
            dst_indices_list_ith_rank,
        ) in enumerate(send_info_per_rank_inter):
            if i == self.local_rank_inter_node:
                continue

            src_index_inter = (
                i
                if return_local_rank
                else (i * self.world_size_intra_node + self.local_rank_intra_node)
            )
            for input_split_size_inter, dst_indices_inter, dst_indices in zip(
                input_split_size_list_inter_ith_rank,
                dst_indices_list_inter_ith_rank,
                dst_indices_list_ith_rank,
            ):
                if is_this_rank_in_dst_indices_inter(dst_indices_inter):
                    # append src index with corr. input split size for the second step
                    src_index_list_inter.append(src_index_inter)
                    output_split_size_list_inter.append(input_split_size_inter)

                    # append filtered dst indices for the third step
                    dst_indices_list_post_intra.append(
                        self._filter_dst_indices_within_this_intra_node(
                            dst_indices=dst_indices,
                            return_local_rank=return_local_rank,
                        )
                    )

        input_split_size_list_post_intra = output_split_size_list_inter

        # set inter group-cast meta
        self.a2a_output_seqlen_inter = sum(output_split_size_list_inter)
        self.input_split_size_list_inter = input_split_size_list_inter
        self.dst_indices_list_inter = dst_indices_list_inter
        self.output_split_size_list_inter = output_split_size_list_inter
        self.src_index_list_inter = src_index_list_inter

        # set half post-intra group-cast meta
        self.input_split_size_list_post_intra = input_split_size_list_post_intra
        self.dst_indices_list_post_intra = dst_indices_list_post_intra

    def _build_group_cast_a2a_meta_args_inter(self):
        (
            a2a_input_split_size_inter,
            perm_range_gather_kwargs_inter,
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_inter,
            dst_indices_list=self.dst_indices_list_inter,
            world_size=self.world_size_inter_node,
            device=self.device,
        )
        (
            a2a_output_split_size_inter,
            _,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_inter,
            src_index_list=self.src_index_list_inter,
            world_size=self.world_size_inter_node,
            device=self.device,
            calc_unperm_after_a2a_kwargs=False,
        )

        # set inter a2a meta
        self.a2a_input_split_size_inter = a2a_input_split_size_inter
        self.a2a_output_split_size_inter = a2a_output_split_size_inter
        self.perm_range_gather_kwargs_inter = perm_range_gather_kwargs_inter

    def _build_group_cast_meta_args_rest_half_post_intra(
        self,
        return_local_rank: bool = False,
    ):
        # all gather send info per rank within this intra node
        send_info_this_rank_post_intra = (
            self.input_split_size_list_post_intra,
            self.dst_indices_list_post_intra,
        )
        send_info_per_rank_post_intra: list[tuple[list[int], list[list[int]]]] = [
            None  # type: ignore
        ] * self.world_size_intra_node
        dist.all_gather_object(
            send_info_per_rank_post_intra,
            send_info_this_rank_post_intra,
            group=self.intra_group,
        )

        # build naive src index list and output split size list
        # within this intra node
        src_index_list_post_intra = []
        output_split_size_list_post_intra = []
        is_this_rank_in_dst_indices_post_intra = (  # noqa: E731
            lambda dst_indices: (self.local_rank_intra_node in dst_indices)
            if return_local_rank
            else (self.rank in dst_indices)
        )
        for i, (
            input_split_size_list_post_intra_ith_rank,
            dst_indices_list_post_intra_ith_rank,
        ) in enumerate(send_info_per_rank_post_intra):
            src_index_post_intra = (
                i if return_local_rank else (i + self.first_rank_this_intra_node)
            )
            for input_split_size_post_intra, dst_indices_post_intra in zip(
                input_split_size_list_post_intra_ith_rank,
                dst_indices_list_post_intra_ith_rank,
            ):
                if is_this_rank_in_dst_indices_post_intra(dst_indices_post_intra):
                    # append src index with corr. input split size for the second step
                    src_index_list_post_intra.append(src_index_post_intra)
                    output_split_size_list_post_intra.append(
                        input_split_size_post_intra
                    )

        # set post-intra group-cast meta
        self.a2a_output_seqlen_post_intra = sum(output_split_size_list_post_intra)
        self.output_split_size_list_post_intra = output_split_size_list_post_intra
        self.src_index_list_post_intra = src_index_list_post_intra

    def _build_group_cast_a2a_meta_args_post_intra(self):
        (
            a2a_input_split_size_post_intra,
            perm_range_gather_kwargs_post_intra,
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_post_intra,
            dst_indices_list=self.dst_indices_list_post_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
        )
        (
            a2a_output_split_size_post_intra,
            _,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_post_intra,
            src_index_list=self.src_index_list_post_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
            calc_unperm_after_a2a_kwargs=False,
        )

        # set post-intra a2a meta
        self.a2a_input_seqlen_post_intra = sum(a2a_input_split_size_post_intra)
        self.a2a_input_split_size_post_intra = a2a_input_split_size_post_intra
        self.a2a_output_split_size_post_intra = a2a_output_split_size_post_intra
        self.perm_range_gather_kwargs_post_intra = perm_range_gather_kwargs_post_intra

        # HACK: this is a helper side stream
        # to apply async intermediate range-gather before post-intra a2a for hierarchical group-cast
        # NOTE: we need to allocate each comm for corr. stage a separate stream
        # to avoid the side stream being blocked by other comms for later stages
        # since we probably set `CUDA_DEVICE_MAX_CONNECTIONS > 1`
        # when all the comms are issued in advance of all the calcs
        # however, this will introduce cuda-malloc ops when applying range-gather for each comm
        # TODO: use the nccl stream to synchronize directly with magi nccl backend
        self.a2a_post_intra_side_stream = torch.cuda.Stream()

    def _build_group_cast_a2a_post_process_fn(
        self,
        output_split_size_list: list[int],
        src_index_list: list[int],
    ):
        rank_list_intra = list(
            range(self.first_rank_this_intra_node, self.last_rank_this_intra_node + 1)
        )
        rank_list_per_inter_wo_this_intra_node = [
            [
                r
                for inter_offset in range(self.world_size_inter_node)
                if not self._is_rank_within_this_intra_node(
                    r := inter_offset * self.world_size_intra_node + intra_offset
                )
            ]
            for intra_offset in range(self.world_size_intra_node)
        ]

        self.reorder_list_hier = rank_list_intra + list(
            chain(*rank_list_per_inter_wo_this_intra_node)
        )

        (
            _,
            self.unperm_after_a2a_kwargs_hier,
            # verbose
            self.unperm_split_size_list_hier,
            self.perm_index_list_hier,
            self.unperm_index_list_hier,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=output_split_size_list,
            src_index_list=src_index_list,
            world_size=self.world_size,
            device=self.device,
            reorder_list=self.reorder_list_hier,
            calc_unperm_after_a2a_kwargs=True,
            return_verbose=True,
        )


def init_hier_group_cast_meta_solver(
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    dst_indices: list[list[int]],
    src_index: list[int],
    rank: int,
    world_size: int,
    intra_group: dist.ProcessGroup,
    inter_group: dist.ProcessGroup,
    use_a2av_impl: bool = True,
    **kwargs,
) -> HierGroupCastMetaSolver:
    if "hier_group_cast_meta_solver" in kwargs:
        # if pre-calculated, directly return
        return kwargs.pop("hier_group_cast_meta_solver")

    return HierGroupCastMetaSolver(
        input_split_size_list=input_split_sizes,
        output_split_size_list=output_split_sizes,
        dst_indices_list=dst_indices,
        src_index_list=src_index,
        rank=rank,
        world_size=world_size,
        intra_group=intra_group,
        inter_group=inter_group,
        use_a2av_impl=use_a2av_impl,
    )


# host meta interface
@overload
def hier_group_cast_impl_with_a2av(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor | None,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    dst_indices: list[list[int]],
    src_index: list[int],
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


# device meta interface
@overload
def hier_group_cast_impl_with_a2av(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor | None,
    input_split_sizes: torch.Tensor,
    output_split_sizes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_index: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


@nvtx.instrument_nvtx
def hier_group_cast_impl_with_a2av(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_indices: list[list[int]] | torch.Tensor,
    src_index: list[int] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Hierarchical group-cast implementation based on all2all_v"""
    # ----    check     ---- #

    # check functionalities
    assert (
        not magi_attention.comm.is_native_grpcoll_enable()
    ), "A2A-based hierarchical group-cast is not compatible with native grpcoll implementation"
    assert async_op, "A2A-based hierarchical group-cast only supports async_op for now"
    assert (
        output_tensor is not None
    ), "A2A-based hierarchical group-cast only supports output is given"
    assert is_list_type_all(
        [input_split_sizes, output_split_sizes, dst_indices, src_index], list
    ), (
        "A2A-based hierarchical group-cast only supports host meta interface, "
        "thus the input_split_sizes, output_split_sizes, dst_indices, src_index should all be list type"
    )
    assert (
        not cast_lse
    ), "A2A-based hierarchical group-cast does not support lse cast for now"

    # check shapes
    assert len(input_split_sizes) == len(dst_indices), (
        f"The length of input_split_sizes and dst_indices should be the same, "
        f"but got {len(input_split_sizes)=} and {len(dst_indices)=}"
    )
    assert len(output_split_sizes) == len(src_index), (
        f"The length of output_split_sizes and src_index should be the same, "
        f"but got {len(output_split_sizes)=} and {len(src_index)=}"
    )
    assert input_tensor.shape[0] == sum(input_split_sizes), (
        f"The sum of input_split_sizes should be equal to input_seqlen, "
        f"but got {sum(input_split_sizes)=} and {input_tensor.shape[0]=}"
    )
    assert output_tensor.shape[0] == sum(output_split_sizes), (
        f"The sum of output_split_sizes should be equal to output_seqlen, "
        f"but got {sum(output_split_sizes)=} and {output_tensor.shape[0]=}"
    )

    rank = kwargs.pop("rank", dist.get_rank(group))
    world_size = kwargs.pop("world_size", dist.get_world_size(group))

    intra_group = kwargs.pop("intra_group", None)
    inter_group = kwargs.pop("inter_group", None)
    assert intra_group is not None and inter_group is not None

    # ----    get hier group-cast meta solver     ---- #

    meta_solver: HierGroupCastMetaSolver = init_hier_group_cast_meta_solver(
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        dst_indices=dst_indices,
        src_index=src_index,
        rank=rank,
        world_size=world_size,
        intra_group=intra_group,
        inter_group=inter_group,
        use_a2av_impl=True,  # for now, only support a2av impl
        **kwargs,
    )

    # ----    prepare a2a output buffer for inter     ---- #

    output_other_shape = output_tensor.shape[1:]
    a2a_output_inter = torch.empty(
        size=[meta_solver.a2a_output_seqlen_inter, *output_other_shape],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # ----    prepare a2a input buffer for inter     ---- #

    a2a_input_inter = range_gather(
        input=input_tensor,
        **meta_solver.perm_range_gather_kwargs_inter,
    )

    # ----    apply a2a for inter     ---- #

    work_inter = all2all_v(
        input=a2a_input_inter,
        output=a2a_output_inter,
        input_split_size_list=meta_solver.a2a_input_split_size_inter,
        output_split_size_list=meta_solver.a2a_output_split_size_inter,
        group=inter_group,
        async_op=async_op,
    )

    # ----    prepare a2a output buffer for pre-/post-intra      ---- #

    a2a_output_pre_intra, a2a_output_post_intra = torch.split(
        output_tensor,
        [
            meta_solver.a2a_output_seqlen_pre_intra,
            meta_solver.a2a_output_seqlen_post_intra,
        ],
        dim=0,
    )

    # ----    prepare a2a input buffer for pre-intra      ---- #

    a2a_input_pre_intra = range_gather(
        input=input_tensor,
        **meta_solver.perm_range_gather_kwargs_pre_intra,
    )

    # ----    apply a2a for pre-intra     ---- #

    # work_pre_intra = \
    all2all_v(
        input=a2a_input_pre_intra,
        output=a2a_output_pre_intra,
        input_split_size_list=meta_solver.a2a_input_split_size_pre_intra,
        output_split_size_list=meta_solver.a2a_output_split_size_pre_intra,
        group=intra_group,
        async_op=async_op,
    )

    # ----    allocate a2a input buffer for post-intra      ---- #

    a2a_input_post_intra = torch.empty(
        size=[meta_solver.a2a_input_seqlen_post_intra, *output_other_shape],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    side_stream = meta_solver.a2a_post_intra_side_stream
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        # ----    prepare a2a input buffer for post-intra     ---- #

        work_inter.wait()
        range_gather(
            input=a2a_output_inter,
            output=a2a_input_post_intra,
            **meta_solver.perm_range_gather_kwargs_post_intra,
        )

        # ----    apply a2a for post-intra     ---- #

        work_post_intra = all2all_v(
            input=a2a_input_post_intra,
            output=a2a_output_post_intra,
            input_split_size_list=meta_solver.a2a_input_split_size_post_intra,
            output_split_size_list=meta_solver.a2a_output_split_size_post_intra,
            group=intra_group,
            async_op=async_op,
        )
        work_post_intra.wait()

    # ----    prepare work with hier group-cast post-process fn    ---- #

    post_process_fn_hier = meta_solver.make_group_cast_a2a_post_process_fn(
        stashed_tensors=[
            # pre-intra
            a2a_input_pre_intra,
            # a2a_output_pre_intra,
            # inter
            a2a_input_inter,
            a2a_output_inter,
            # post-intra
            a2a_input_post_intra,
            # a2a_output_post_intra,
        ],
    )

    # NOTE: no need to wait for work_pre_intra explicitly here
    # since side_stream will wait for work_post_intra,
    # which is issued after work_pre_intra's completion
    # thus we only need to wait for side_stream
    work_with_post_process_fn = WorkWithPostProcessFn(
        # work=GeneralWork([work_pre_intra, side_stream]),
        work=GeneralWork(side_stream),
        post_process_fn=post_process_fn_hier,
        async_op=async_op,
    )

    return work_with_post_process_fn


# ------------------        hierarchical a2av group reduce       ------------------ #


class HierGroupReduceMetaSolver(HierGroupCastMetaSolver):
    def __init__(
        self,
        input_split_size_list: list[int],
        output_split_size_list: list[int],
        dst_index_list: list[int],
        src_indices_list: list[list[int]],
        rank: int,
        world_size: int,
        intra_group: dist.ProcessGroup,
        inter_group: dist.ProcessGroup,
        use_a2av_impl: bool = True,
        deterministic: bool = False,
    ):
        # --------   init the symmetric hier group-cast meta solver  -------- #

        super().__init__(
            input_split_size_list=output_split_size_list,
            output_split_size_list=input_split_size_list,
            dst_indices_list=src_indices_list,
            src_index_list=dst_index_list,
            rank=rank,
            world_size=world_size,
            intra_group=intra_group,
            inter_group=inter_group,
            use_a2av_impl=use_a2av_impl,
        )

        self._build(deterministic)

    @classmethod
    def make_from_sym_hier_group_cast_meta_solver(
        cls,
        sym_hier_group_cast_meta_solver: HierGroupCastMetaSolver,
        deterministic: bool = False,
    ) -> "HierGroupReduceMetaSolver":
        instance = cls.__new__(cls)
        instance.__dict__.update(sym_hier_group_cast_meta_solver.__dict__)
        instance._build(deterministic)
        return instance

    def get_group_reduce_meta_pre_intra(self):
        return (
            self.input_split_size_list_pre_intra,
            self.output_split_size_list_pre_intra,
            self.dst_index_list_pre_intra,
            self.src_indices_list_pre_intra,
        )

    def get_group_reduce_meta_inter(self):
        return (
            self.input_split_size_list_inter,
            self.output_split_size_list_inter,
            self.dst_index_list_inter,
            self.src_indices_list_inter,
        )

    def get_group_reduce_meta_post_intra(self):
        return (
            self.input_split_size_list_post_intra,
            self.output_split_size_list_post_intra,
            self.dst_index_list_post_intra,
            self.src_indices_list_post_intra,
        )

    def make_group_reduce_a2a_post_process_fn(
        self,
        a2a_output_pre_intra: torch.Tensor,
        a2a_output_inter: torch.Tensor,
        stashed_tensors: list[torch.Tensor] | None = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        post_process_fn_pre_intra = partial(
            sum_reduce_to_tensor,  # TODO: support other reduce ops
            a2a_output=a2a_output_pre_intra,
            range_reduce_kwargs=self.range_reduce_kwargs_pre_intra,
        )

        post_process_fn_inter = partial(
            sum_reduce_to_tensor,  # TODO: support other reduce ops
            a2a_output=a2a_output_inter,
            range_reduce_kwargs=self.range_reduce_kwargs_inter,
        )

        def post_process_fn_hier(output_tensor: torch.Tensor) -> torch.Tensor:
            try:
                post_process_fn_pre_intra(output_tensor)
                post_process_fn_inter(output_tensor)
            finally:
                nonlocal stashed_tensors
                if stashed_tensors is not None:
                    stashed_tensors.clear()

            return output_tensor

        return post_process_fn_hier

    def _build(self, deterministic: bool = False):
        self.deterministic = deterministic

        # ----   build group-reduce meta for pre-intra  ---- #

        self._build_group_reduce_meta_args_pre_intra()

        # ----   build a2a meta for pre-intra  ---- #

        if self.use_a2av_impl:
            self._build_group_reduce_a2a_meta_args_pre_intra()

        # ----   build group-reduce meta for post-intra  ---- #

        self._build_group_reduce_meta_args_post_intra()

        # ----   build a2a meta for post-intra  ---- #

        if self.use_a2av_impl:
            self._build_group_reduce_a2a_meta_args_post_intra()

        # ----   build group-reduce meta for inter  ---- #

        self._build_group_reduce_meta_args_inter()

        # ----   build a2a meta for inter  ---- #

        if self.use_a2av_impl:
            self._build_group_reduce_a2a_meta_args_inter()

        # ----   build hier group-reduce a2a pre-process fn ---- #

        if self.use_a2av_impl:
            self._build_group_reduce_a2a_pre_process_fn()

    def _build_group_reduce_meta_args_pre_intra(self):
        # build from symmetric group-cast pre-intra meta args
        (
            self.output_split_size_list_pre_intra,
            self.input_split_size_list_pre_intra,
            self.src_indices_list_pre_intra,
            self.dst_index_list_pre_intra,
        ) = self.get_group_cast_meta_pre_intra()

    def _build_group_reduce_a2a_meta_args_pre_intra(self):
        # a2a input meta args
        (
            self.a2a_input_split_size_pre_intra,
            _,  # perm_range_gather_kwargs_pre_intra,
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_pre_intra,
            dst_index_list=self.dst_index_list_pre_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
        )

        # a2a output meta args
        (
            self.a2a_output_split_size_pre_intra,
            self.range_reduce_kwargs_pre_intra,
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_pre_intra,
            src_indices_list=self.src_indices_list_pre_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
            deterministic=self.deterministic,
        )

        # get the pre-intra a2a input seqlen from the corr. a2a output seqlen for hier group-cast
        self.a2a_input_seqlen_pre_intra = self.a2a_output_seqlen_pre_intra
        # sanity check
        assert self.a2a_input_seqlen_pre_intra == sum(
            self.a2a_input_split_size_pre_intra
        )

        # reset the pre-intra a2a output seqlen
        self.a2a_output_seqlen_pre_intra = sum(self.a2a_output_split_size_pre_intra)

    def _build_group_reduce_meta_args_post_intra(self):
        # build from symmetric group-cast post-intra meta args
        (
            self.output_split_size_list_post_intra,
            self.input_split_size_list_post_intra,
            self.src_indices_list_post_intra,
            self.dst_index_list_post_intra,
        ) = self.get_group_cast_meta_post_intra()

    def _build_group_reduce_a2a_meta_args_post_intra(self):
        # a2a input meta args
        (
            self.a2a_input_split_size_post_intra,
            _,  # perm_range_gather_kwargs_post_intra,
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_post_intra,
            dst_index_list=self.dst_index_list_post_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
        )

        # a2a output meta args
        (
            self.a2a_output_split_size_post_intra,
            self.range_reduce_kwargs_post_intra,
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_post_intra,
            src_indices_list=self.src_indices_list_post_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
            deterministic=self.deterministic,
        )

        # get the post-intra a2a input/output seqlen from the corr. a2a output/input seqlen for hier group-cast
        (
            self.a2a_input_seqlen_post_intra,
            self.a2a_output_seqlen_post_intra,
        ) = (
            self.a2a_output_seqlen_post_intra,
            self.a2a_input_seqlen_post_intra,
        )

        # sanity check
        assert self.a2a_input_seqlen_post_intra == sum(
            self.a2a_input_split_size_post_intra
        )
        assert self.a2a_output_seqlen_post_intra == sum(
            self.a2a_output_split_size_post_intra
        )

    def _build_group_reduce_meta_args_inter(self):
        # build from symmetric group-cast inter meta args
        (
            self.output_split_size_list_inter,
            self.input_split_size_list_inter,
            self.src_indices_list_inter,
            self.dst_index_list_inter,
        ) = self.get_group_cast_meta_inter()

    def _build_group_reduce_a2a_meta_args_inter(self):
        # a2a input meta args
        (
            self.a2a_input_split_size_inter,
            _,  # perm_range_gather_kwargs_inter
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_inter,
            dst_index_list=self.dst_index_list_inter,
            world_size=self.world_size_inter_node,
            device=self.device,
        )

        # a2a output meta args
        (
            self.a2a_output_split_size_inter,
            self.range_reduce_kwargs_inter,
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_inter,
            src_indices_list=self.src_indices_list_inter,
            world_size=self.world_size_inter_node,
            device=self.device,
            deterministic=self.deterministic,
        )

        # get the inter a2a input seqlen from the corr. a2a output seqlen for hier group-cast
        self.a2a_input_seqlen_inter = self.a2a_output_seqlen_inter
        # sanity check
        assert self.a2a_input_seqlen_inter == sum(self.a2a_input_split_size_inter)

        # reset the inter a2a output seqlen
        self.a2a_output_seqlen_inter = sum(self.a2a_output_split_size_inter)

        # HACK: this is a helper side stream
        # to apply async intermediate range-reduce before inter a2a for hierarchical group-reduce
        # NOTE: we need to allocate each comm for corr. stage a separate stream
        # to avoid the side stream being blocked by other comms for later stages
        # since we probably set `CUDA_DEVICE_MAX_CONNECTIONS > 1`
        # when all the comms are issued in advance of all the calcs
        # however, this will introduce cuda-malloc ops when applying range-gather for each comm
        # TODO: use the nccl stream to synchronize directly with magi nccl backend
        self.a2a_inter_side_stream = torch.cuda.Stream()

    def _build_group_reduce_a2a_pre_process_fn(self):
        self.perm_split_size_list_hier = [
            self.unperm_split_size_list_hier[idx] for idx in self.unperm_index_list_hier
        ]
        self.perm_before_a2a_kwargs_hier = (
            _calc_unperm_range_gather_kwargs_from_split_size_list(
                split_size_list=self.perm_split_size_list_hier,
                unpermute_index_list=self.perm_index_list_hier,
                device=self.device,
            )
        )

        self.pre_process_fn_hier = partial(
            unpermute_tensor, unperm_after_a2a_kwargs=self.perm_before_a2a_kwargs_hier
        )


def init_hier_group_reduce_meta_solver(
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    dst_index: list[int],
    src_indices: list[list[int]],
    rank: int,
    world_size: int,
    intra_group: dist.ProcessGroup,
    inter_group: dist.ProcessGroup,
    use_a2av_impl: bool = True,
    **kwargs,
) -> HierGroupReduceMetaSolver:
    if "hier_group_reduce_meta_solver" in kwargs:
        # if pre-calculated, directly return
        return kwargs.pop("hier_group_reduce_meta_solver")

    if "sym_hier_group_cast_meta_solver" in kwargs:
        # if the symmetric hier group-cast meta solver exists
        # then use it to build the hier group-reduce meta solver to avoid re-calculating
        return HierGroupReduceMetaSolver.make_from_sym_hier_group_cast_meta_solver(
            sym_hier_group_cast_meta_solver=kwargs.pop(
                "sym_hier_group_cast_meta_solver"
            ),
            deterministic=kwargs.pop("deterministic", False),
        )

    return HierGroupReduceMetaSolver(
        input_split_size_list=input_split_sizes,
        output_split_size_list=output_split_sizes,
        dst_index_list=dst_index,
        src_indices_list=src_indices,
        rank=rank,
        world_size=world_size,
        intra_group=intra_group,
        inter_group=inter_group,
        use_a2av_impl=use_a2av_impl,
        deterministic=kwargs.pop("deterministic", False),
    )


# host meta interface
@overload
def hier_group_reduce_impl_with_a2av(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor | None,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    dst_index: list[int],
    src_indices: list[list[int]],
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


# device meta interface
@overload
def hier_group_reduce_impl_with_a2av(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor | None,
    input_split_sizes: torch.Tensor,
    output_split_sizes: torch.Tensor,
    dst_index: torch.Tensor,
    src_indices: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


@nvtx.instrument_nvtx
def hier_group_reduce_impl_with_a2av(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_index: list[int] | torch.Tensor,
    src_indices: list[list[int]] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Hierarchical group-reduce implementation based on all2all_v"""
    # ----    check     ---- #

    # check functionalities
    assert (
        not magi_attention.comm.is_native_grpcoll_enable()
    ), "A2A-based hierarchical group-reduce is not compatible with native grpcoll implementation"
    assert (
        acc_reduce and output_tensor is not None
    ), "A2A-based hierarchical group-reduce only supports acc_reduce=True and output is given"
    assert (
        async_op
    ), "A2A-based hierarchical group-reduce only supports async_op for now"
    assert (
        reduce_op == "sum"
    ), "A2A-based hierarchical group-reduce only supports sum reduction by now"
    assert is_list_type_all(
        [input_split_sizes, output_split_sizes, dst_index, src_indices], list
    ), (
        "A2A-based hierarchical group-reduce only supports host meta interface, "
        "thus the input_split_sizes, output_split_sizes, dst_index, src_indice should all be list type"
    )

    # check shape
    assert len(input_split_sizes) == len(dst_index), (
        f"input_split_sizes and dst_index should have the same length, "
        f"but got {len(input_split_sizes)=} and {len(dst_index)=}"
    )
    assert len(output_split_sizes) == len(src_indices), (
        f"output_split_sizes and src_indices should have the same length, "
        f"but got {len(output_split_sizes)=} and {len(src_indices)=}"
    )
    assert input_tensor.shape[0] == sum(input_split_sizes), (
        f"The sum of input_split_sizes should be equal to input_seqlen, "
        f"but got {sum(input_split_sizes)=} and {input_tensor.shape[0]=}"
    )
    assert output_tensor.shape[0] == sum(output_split_sizes), (
        f"The sum of output_split_sizes should be equal to output_seqlen, "
        f"but got {sum(output_split_sizes)=} and {output_tensor.shape[0]=}"
    )

    rank = kwargs.pop("rank", dist.get_rank(group))
    world_size = kwargs.pop("world_size", dist.get_world_size(group))

    intra_group = kwargs.pop("intra_group", None)
    inter_group = kwargs.pop("inter_group", None)
    assert intra_group is not None and inter_group is not None

    # ----    get hier group-reduce meta solver     ---- #

    meta_solver: HierGroupReduceMetaSolver = init_hier_group_reduce_meta_solver(
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        dst_index=dst_index,
        src_indices=src_indices,
        rank=rank,
        world_size=world_size,
        intra_group=intra_group,
        inter_group=inter_group,
        use_a2av_impl=True,  # for now, only support a2av impl
        **kwargs,
    )

    # ----    hier group-reduce a2a pre-process    ---- #

    perm_input_tensor = meta_solver.pre_process_fn_hier(input_tensor)

    # ----    prepare a2a input buffer for pre-/post-intra    ---- #

    a2a_input_pre_intra, a2a_input_post_intra = torch.split(
        perm_input_tensor,
        [
            meta_solver.a2a_input_seqlen_pre_intra,
            meta_solver.a2a_input_seqlen_post_intra,
        ],
        dim=0,
    )

    # ----    allocate a2a output buffer for post-intra    ---- #

    output_other_shape = output_tensor.shape[1:]
    a2a_output_post_intra = torch.empty(
        size=[meta_solver.a2a_output_seqlen_post_intra, *output_other_shape],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # ----    apply a2a for post-intra     ---- #

    work_post_intra = all2all_v(
        input=a2a_input_post_intra,
        output=a2a_output_post_intra,
        input_split_size_list=meta_solver.a2a_input_split_size_post_intra,
        output_split_size_list=meta_solver.a2a_output_split_size_post_intra,
        group=intra_group,
        async_op=async_op,
    )

    # ----    allocate a2a output buffer for inter    ---- #

    a2a_output_inter = torch.empty(
        size=[meta_solver.a2a_output_seqlen_inter, *output_other_shape],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # ----    allocate a2a input buffer for inter    ---- #

    # NOTE: since it needs to be reduced, we need a zero buffer
    a2a_input_inter = torch.zeros(
        size=[meta_solver.a2a_input_seqlen_inter, *output_other_shape],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    side_stream = meta_solver.a2a_inter_side_stream
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        # ----    prepare a2a input buffer for inter    ---- #

        work_post_intra.wait()
        range_reduce(
            input=a2a_output_post_intra,
            output=a2a_input_inter,
            **meta_solver.range_reduce_kwargs_post_intra,
        )

        # ----    apply a2a for inter     ---- #

        work_inter = all2all_v(
            input=a2a_input_inter,
            output=a2a_output_inter,
            input_split_size_list=meta_solver.a2a_input_split_size_inter,
            output_split_size_list=meta_solver.a2a_output_split_size_inter,
            group=inter_group,
            async_op=async_op,
        )
        work_inter.wait()

    # ----    allocate a2a output buffer for pre-intra    ---- #

    a2a_output_pre_intra = torch.empty(
        size=[meta_solver.a2a_output_seqlen_pre_intra, *output_other_shape],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # ----    apply a2a for pre-intra     ---- #

    work_pre_intra = all2all_v(
        input=a2a_input_pre_intra,
        output=a2a_output_pre_intra,
        input_split_size_list=meta_solver.a2a_input_split_size_pre_intra,
        output_split_size_list=meta_solver.a2a_output_split_size_pre_intra,
        group=intra_group,
        async_op=async_op,
    )

    # ----    prepare work with hier group-reduce post-process fn    ---- #

    post_process_fn_hier = meta_solver.make_group_reduce_a2a_post_process_fn(
        a2a_output_pre_intra=a2a_output_pre_intra,
        a2a_output_inter=a2a_output_inter,
        stashed_tensors=[
            # pre-intra
            # a2a_input_pre_intra,
            # a2a_output_pre_intra,
            # inter
            a2a_input_inter,
            # a2a_output_inter,
            # post-intra
            # a2a_input_post_intra,
            a2a_output_post_intra,
        ],
    )

    # NOTE: different from hier group-cast,
    # we have to wait for work_pre_intra explicitly here
    # since waiting for side_stream only guarantees
    # work_post_intra and work_inter is done
    work_with_post_process_fn = WorkWithPostProcessFn(
        work=GeneralWork([work_pre_intra, side_stream]),
        post_process_fn=post_process_fn_hier,
        async_op=async_op,
    )

    return work_with_post_process_fn
