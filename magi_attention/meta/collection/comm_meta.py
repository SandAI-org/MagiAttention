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
    """The arg dataclass for group cast/reduce collective ops"""

    input_split_size_list: list[int]
    output_split_size_list: list[int]
    dst_indices_list: list[list[int]]
    src_index_list: list[int]

    rank: int
    world_size: int
    device_mesh: DeviceMesh | None = None

    deterministic: bool = False

    # NOTE: The following variables are automatically generated by `__post_init__`
    # and serve as the meta arguments for group collective.
    #   group_cast_args_dict_kv_packed: dict
    #   group_reduce_args_dict_kv_packed: dict

    def __post_init__(self):
        self.device = torch.cuda.current_device()

        # ----   group cast args dict for packed kv  ---- #

        self.group_cast_args_dict_kv_packed = {
            k: v * 2  # concat kv along seqlen dim
            for k, v in {
                "input_split_size_list": self.input_split_size_list,
                "output_split_size_list": self.output_split_size_list,
                "dst_indices_list": self.dst_indices_list,
                "src_index_list": self.src_index_list,
            }.items()
        }

        # ----   group reduce args dict for packed kv  ---- #

        # symmetric to group-cast
        self.group_reduce_args_dict_kv_packed = dict(
            input_split_size_list=self.group_cast_args_dict_kv_packed[
                "output_split_size_list"
            ],
            output_split_size_list=self.group_cast_args_dict_kv_packed[
                "input_split_size_list"
            ],
            dst_index_list=self.group_cast_args_dict_kv_packed["src_index_list"],
            src_indices_list=self.group_cast_args_dict_kv_packed["dst_indices_list"],
        )

        # ----   additional kwargs  ---- #

        if magi_attention.comm.is_hierarchical_comm_enable():
            assert self.device_mesh.ndim == 2, (
                f"The hierarchical comm is only supported for 2D device mesh, "
                f"but got {self.device_mesh.ndim=}."
            )

            # fetch the intra/inter groups from the device mesh
            self.intra_group = self.device_mesh.get_group(1)
            self.inter_group = self.device_mesh.get_group(0)

            # init meta kwargs for hierarchical group-cast/reduce
            self._init_meta_kwargs_for_hier_group_cast()
            self._init_meta_kwargs_for_hier_group_reduce()
        else:
            # init a2a meta kwargs for group-cast/reduce
            self._init_a2a_meta_kwargs_for_group_cast()
            self._init_a2a_meta_kwargs_for_group_reduce()

    def _init_meta_kwargs_for_hier_group_cast(self):
        self.group_cast_args_dict_kv_packed.update(
            dict(
                rank=self.rank,
                world_size=self.world_size,
                intra_group=self.intra_group,
                inter_group=self.inter_group,
            )
        )

        # init the hierarchial group-cast meta solver
        (
            self.group_cast_args_dict_kv_packed["hier_group_cast_meta_solver"]
        ) = init_hier_group_cast_meta_solver(
            **self.group_cast_args_dict_kv_packed,
        )

    def _init_meta_kwargs_for_hier_group_reduce(self):
        self.group_reduce_args_dict_kv_packed.update(
            dict(
                rank=self.rank,
                world_size=self.world_size,
                intra_group=self.intra_group,
                inter_group=self.inter_group,
            )
        )

        # init the hierarchial group-reduce meta solver
        (
            self.group_reduce_args_dict_kv_packed["hier_group_reduce_meta_solver"]
        ) = init_hier_group_reduce_meta_solver(
            **self.group_reduce_args_dict_kv_packed,
            sym_hier_group_cast_meta_solver=(
                self.group_cast_args_dict_kv_packed.get(
                    "hier_group_cast_meta_solver", None
                )
            ),
            deterministic=self.deterministic,
        )

    def _init_a2a_meta_kwargs_for_group_cast(self):
        (
            self.group_cast_args_dict_kv_packed["a2a_input_split_size"],
            self.group_cast_args_dict_kv_packed["perm_before_a2a_kwargs"],
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self.group_cast_args_dict_kv_packed[
                "input_split_size_list"
            ],
            dst_indices_list=self.group_cast_args_dict_kv_packed["dst_indices_list"],
            world_size=self.world_size,
            device=self.device,
        )

        (
            self.group_cast_args_dict_kv_packed["a2a_output_split_size"],
            self.group_cast_args_dict_kv_packed["unperm_after_a2a_kwargs"],
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self.group_cast_args_dict_kv_packed[
                "output_split_size_list"
            ],
            src_index_list=self.group_cast_args_dict_kv_packed["src_index_list"],
            world_size=self.world_size,
            device=self.device,
        )

    def _init_a2a_meta_kwargs_for_group_reduce(self):
        (
            self.group_reduce_args_dict_kv_packed["a2a_input_split_size"],
            self.group_reduce_args_dict_kv_packed["perm_before_a2a_kwargs"],
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=self.group_reduce_args_dict_kv_packed[
                "input_split_size_list"
            ],
            dst_index_list=self.group_reduce_args_dict_kv_packed["dst_index_list"],
            world_size=self.world_size,
            device=self.device,
        )

        (
            self.group_reduce_args_dict_kv_packed["a2a_output_split_size"],
            self.group_reduce_args_dict_kv_packed["range_reduce_kwargs"],
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=self.group_reduce_args_dict_kv_packed[
                "output_split_size_list"
            ],
            src_indices_list=self.group_reduce_args_dict_kv_packed["src_indices_list"],
            world_size=self.world_size,
            device=self.device,
            deterministic=self.deterministic,
        )

    def to_group_cast_args(self) -> dict:
        return self.group_cast_args_dict_kv_packed

    def to_group_reduce_args(self) -> dict:
        return self.group_reduce_args_dict_kv_packed


@dataclass
class CommMeta:
    num_remote_tokens_per_stage: list[int]
    group_collective_args_list: list[GroupCollectiveArg]

    @property
    def overlap_degree(self) -> int:
        return len(self.num_remote_tokens_per_stage)

    def __post_init__(self):
        assert len(self.num_remote_tokens_per_stage) == len(
            self.group_collective_args_list
        ), (
            f"Got inconsistent overlap degree: "
            f"{len(self.num_remote_tokens_per_stage)=} and "
            f"{len(self.group_collective_args_list)=}."
        )
