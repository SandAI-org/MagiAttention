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

from dataclasses import dataclass
from typing import Literal

import torch
from torch.distributed.device_mesh import DeviceMesh

import magi_attention
from magi_attention.comm.primitive._group_collective_hier import (
    init_hier_group_cast_meta_solver,
    init_hier_group_reduce_meta_solver,
)
from magi_attention.comm.primitive.utils import (
    _calc_group_cast_a2a_input_meta_args,
    _calc_group_cast_a2a_output_meta_args,
    _calc_group_reduce_a2a_input_meta_args,
    _calc_group_reduce_a2a_output_meta_args,
)


@dataclass
class GroupCollectiveArg:
    """The native args for group cast/reduce collective"""

    input_split_size_list: list[int]
    output_split_size_list: list[int]
    dst_indices_list: list[list[int]]
    src_index_list: list[int]

    rank: int
    world_size: int
    device_mesh: DeviceMesh | None = None

    deterministic: bool = False

    def __post_init__(self):
        self.device = torch.cuda.current_device()

    def to_group_cast_args(self) -> dict:
        return dict(
            input_split_size_list=self.input_split_size_list,
            output_split_size_list=self.output_split_size_list,
            dst_indices_list=self.dst_indices_list,
            src_index_list=self.src_index_list,
        )

    def to_group_reduce_args(self) -> dict:
        return dict(
            input_split_size_list=self.output_split_size_list,
            output_split_size_list=self.input_split_size_list,
            dst_index_list=self.src_index_list,
            src_indices_list=self.dst_indices_list,
        )


@dataclass
class A2AVBasedGroupCollectiveArg(GroupCollectiveArg):
    """The a2a-v based args for group cast/reduce collective"""

    packed_times: int = 1
    reduce_op: Literal["sum", "avg", "lse"] = "sum"
    init_group_reduce: bool = True

    def __post_init__(self):
        super().__post_init__()

        # ----   group cast args dict for packed tensors  ---- #

        self._group_cast_args_dict_packed = {
            # pack tensors along split dim by `self.packed_times` times
            k: v * self.packed_times  # type: ignore[operator]
            for k, v in {
                "input_split_size_list": self.input_split_size_list,
                "output_split_size_list": self.output_split_size_list,
                "dst_indices_list": self.dst_indices_list,
                "src_index_list": self.src_index_list,
            }.items()
        }

        # ----   group reduce args dict for packed tensors  ---- #

        if self.init_group_reduce:
            # symmetric to group-cast
            self._group_reduce_args_dict_packed = dict(
                input_split_size_list=self._group_cast_args_dict_packed[
                    "output_split_size_list"
                ],
                output_split_size_list=self._group_cast_args_dict_packed[
                    "input_split_size_list"
                ],
                dst_index_list=self._group_cast_args_dict_packed["src_index_list"],
                src_indices_list=self._group_cast_args_dict_packed["dst_indices_list"],
                reduce_op=self.reduce_op,
            )

        # ----   additional kwargs  ---- #

        if magi_attention.comm.is_hierarchical_comm_enable():
            assert self.device_mesh.ndim == 2, (  # type: ignore[union-attr]
                f"The hierarchical comm is only supported for 2D device mesh, "
                f"but got {self.device_mesh.ndim=}."  # type: ignore[union-attr]
            )

            # fetch the intra/inter groups from the device mesh
            self.intra_group = self.device_mesh.get_group(1)  # type: ignore[union-attr]
            self.inter_group = self.device_mesh.get_group(0)  # type: ignore[union-attr]

            # init meta kwargs for hierarchical group-cast/reduce
            self._init_meta_kwargs_for_hier_group_cast()
            if self.init_group_reduce:
                self._init_meta_kwargs_for_hier_group_reduce()
        else:
            # init a2a meta kwargs for group-cast/reduce
            self._init_a2a_meta_kwargs_for_group_cast()
            if self.init_group_reduce:
                self._init_a2a_meta_kwargs_for_group_reduce()

    def _init_meta_kwargs_for_hier_group_cast(self):
        self._group_cast_args_dict_packed.update(
            dict(
                rank=self.rank,
                world_size=self.world_size,
                intra_group=self.intra_group,
                inter_group=self.inter_group,
            )
        )

        # init the hierarchial group-cast meta solver
        (
            self._group_cast_args_dict_packed["hier_group_cast_meta_solver"]
        ) = init_hier_group_cast_meta_solver(
            **self._group_cast_args_dict_packed,
        )

    def _init_meta_kwargs_for_hier_group_reduce(self):
        self._group_reduce_args_dict_packed.update(
            dict(
                rank=self.rank,
                world_size=self.world_size,
                intra_group=self.intra_group,
                inter_group=self.inter_group,
            )
        )

        # init the hierarchial group-reduce meta solver
        (
            self._group_reduce_args_dict_packed["hier_group_reduce_meta_solver"]
        ) = init_hier_group_reduce_meta_solver(
            **self._group_reduce_args_dict_packed,
            sym_hier_group_cast_meta_solver=(
                self._group_cast_args_dict_packed.get(
                    "hier_group_cast_meta_solver", None
                )
            ),
            deterministic=self.deterministic,
        )

    def _init_a2a_meta_kwargs_for_group_cast(self):
        (
            self._group_cast_args_dict_packed["a2a_input_split_size"],
            self._group_cast_args_dict_packed["perm_before_a2a_kwargs"],
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self._group_cast_args_dict_packed[
                "input_split_size_list"
            ],
            dst_indices_list=self._group_cast_args_dict_packed["dst_indices_list"],
            world_size=self.world_size,
            device=self.device,
        )

        (
            self._group_cast_args_dict_packed["a2a_output_split_size"],
            self._group_cast_args_dict_packed["unperm_after_a2a_kwargs"],
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self._group_cast_args_dict_packed[
                "output_split_size_list"
            ],
            src_index_list=self._group_cast_args_dict_packed["src_index_list"],
            world_size=self.world_size,
            device=self.device,
        )

    def _init_a2a_meta_kwargs_for_group_reduce(self):
        (
            self._group_reduce_args_dict_packed["a2a_input_split_size"],
            self._group_reduce_args_dict_packed["perm_before_a2a_kwargs"],
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=self._group_reduce_args_dict_packed[
                "input_split_size_list"
            ],
            dst_index_list=self._group_reduce_args_dict_packed["dst_index_list"],
            world_size=self.world_size,
            device=self.device,
        )

        (
            self._group_reduce_args_dict_packed["a2a_output_split_size"],
            self._group_reduce_args_dict_packed["range_reduce_kwargs"],
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=self._group_reduce_args_dict_packed[
                "output_split_size_list"
            ],
            src_indices_list=self._group_reduce_args_dict_packed["src_indices_list"],
            world_size=self.world_size,
            device=self.device,
            deterministic=self.deterministic,
        )

    def to_group_cast_args(self) -> dict:
        return self._group_cast_args_dict_packed

    def to_group_reduce_args(self) -> dict:
        assert self.init_group_reduce, (
            "The group-reduce args dict has not been initialized yet."
            "Please set `init_group_reduce=True`."
        )
        return self._group_reduce_args_dict_packed


@dataclass
class CommMeta:
    # for kv comm in fwd and dkv comm in bwd
    # NOTE: this denotes sk or sv, not sk + sv
    num_remote_kv_tokens_per_stage: list[int]
    kv_group_collective_args_list: list[GroupCollectiveArg]

    # for qo comm in fwd and q,o,do,dq,lse comm in bwd
    # NOTE: this denotes sq or so, not sq + so
    num_remote_qo_tokens_per_stage: list[int]
    qo_group_collective_args_list: list[GroupCollectiveArg]

    # NOTE: The following variables are automatically generated by `__post_init__`
    # and serve as the meta arguments for dist-attn comm.
    #   num_remote_qo_do_tokens_per_stage: list[int]
    #   qo_do_group_collective_args_list: list[GroupCollectiveArg]
    #   num_remote_out_lse_tokens_per_stage: list[int]
    #   out_lse_group_collective_args_list: list[GroupCollectiveArg]

    @property
    def overlap_degree(self) -> int:
        return len(self.num_remote_kv_tokens_per_stage)

    def __post_init__(self):
        assert (
            len(self.num_remote_kv_tokens_per_stage)
            == len(self.kv_group_collective_args_list)
            == len(self.num_remote_qo_tokens_per_stage)
            == len(self.qo_group_collective_args_list)
        ), (
            f"Got inconsistent overlap degree: "
            f"{len(self.num_remote_kv_tokens_per_stage)=}, "
            f"{len(self.kv_group_collective_args_list)=}, "
            f"{len(self.num_remote_qo_tokens_per_stage)=}, "
            f"{len(self.qo_group_collective_args_list)=}, "
        )

        # -----     init a2a-v based group collective args     ----- #

        self.num_remote_qo_do_tokens_per_stage: list[int] = []
        self.qo_do_group_collective_args_list: list[A2AVBasedGroupCollectiveArg] = []
        self.num_remote_out_lse_tokens_per_stage: list[int] = []
        self.out_lse_group_collective_args_list: list[A2AVBasedGroupCollectiveArg] = []
        for stage in range(self.overlap_degree):
            # --- for fetch kv and reduce dkv --- #

            self.num_remote_kv_tokens_per_stage[stage] *= 2
            kv_group_collective_arg = self.kv_group_collective_args_list[stage]
            self.kv_group_collective_args_list[stage] = A2AVBasedGroupCollectiveArg(
                input_split_size_list=kv_group_collective_arg.input_split_size_list,
                output_split_size_list=kv_group_collective_arg.output_split_size_list,
                dst_indices_list=kv_group_collective_arg.dst_indices_list,
                src_index_list=kv_group_collective_arg.src_index_list,
                rank=kv_group_collective_arg.rank,
                world_size=kv_group_collective_arg.world_size,
                device_mesh=kv_group_collective_arg.device_mesh,
                deterministic=kv_group_collective_arg.deterministic,
                packed_times=2,  # pack kv along seqlen dim
                reduce_op="sum",  # sum-reduce dkv
                init_group_reduce=True,
            )

            if magi_attention.comm.is_qo_comm_enable():
                # --- for fetch q, fetch lse and reduce dq --- #

                qo_group_collective_arg = self.qo_group_collective_args_list[stage]
                self.qo_group_collective_args_list[stage] = A2AVBasedGroupCollectiveArg(
                    input_split_size_list=qo_group_collective_arg.input_split_size_list,
                    output_split_size_list=qo_group_collective_arg.output_split_size_list,
                    dst_indices_list=qo_group_collective_arg.dst_indices_list,
                    src_index_list=qo_group_collective_arg.src_index_list,
                    rank=qo_group_collective_arg.rank,
                    world_size=qo_group_collective_arg.world_size,
                    device_mesh=qo_group_collective_arg.device_mesh,
                    deterministic=qo_group_collective_arg.deterministic,
                    packed_times=1,  # q, lse, dq along
                    reduce_op="sum",  # sum-reduce dq
                    init_group_reduce=True,
                )

                # --- for fetch q,o,do --- #

                self.num_remote_qo_do_tokens_per_stage.append(
                    self.num_remote_qo_tokens_per_stage[stage] * 3
                )
                self.qo_do_group_collective_args_list.append(
                    A2AVBasedGroupCollectiveArg(
                        input_split_size_list=qo_group_collective_arg.input_split_size_list,
                        output_split_size_list=qo_group_collective_arg.output_split_size_list,
                        dst_indices_list=qo_group_collective_arg.dst_indices_list,
                        src_index_list=qo_group_collective_arg.src_index_list,
                        rank=qo_group_collective_arg.rank,
                        world_size=qo_group_collective_arg.world_size,
                        device_mesh=qo_group_collective_arg.device_mesh,
                        deterministic=qo_group_collective_arg.deterministic,
                        packed_times=3,  # pack q, o, do along seqlen dim
                        reduce_op="sum",
                        init_group_reduce=False,  # no reduce
                    )
                )

                # --- for reduce out with lse --- #

                self.num_remote_out_lse_tokens_per_stage.append(
                    self.num_remote_qo_tokens_per_stage[stage]
                )
                self.out_lse_group_collective_args_list.append(
                    A2AVBasedGroupCollectiveArg(
                        input_split_size_list=qo_group_collective_arg.input_split_size_list,
                        output_split_size_list=qo_group_collective_arg.output_split_size_list,
                        dst_indices_list=qo_group_collective_arg.dst_indices_list,
                        src_index_list=qo_group_collective_arg.src_index_list,
                        rank=qo_group_collective_arg.rank,
                        world_size=qo_group_collective_arg.world_size,
                        device_mesh=qo_group_collective_arg.device_mesh,
                        deterministic=qo_group_collective_arg.deterministic,
                        packed_times=1,  # out with lse along
                        reduce_op="lse",  # lse-reduce out and lse
                        init_group_reduce=True,
                    )
                )
