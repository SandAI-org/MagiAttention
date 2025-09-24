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
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention
from magi_attention.comm.primitive.grpcoll._group_collective_hier import (
    init_hier_group_cast_meta_solver,
    init_hier_group_reduce_meta_solver,
)
from magi_attention.comm.primitive.grpcoll.utils import (
    _calc_group_cast_a2a_input_meta_args,
    _calc_group_cast_a2a_output_meta_args,
    _calc_group_reduce_a2a_input_meta_args,
    _calc_group_reduce_a2a_output_meta_args,
    get_a2av_perm_idxs_from_group_cast_meta,
    get_dispatch_layout_from_group_cast_meta,
)
from magi_attention.common.enum import ReduceOp
from magi_attention.utils import format_dict_field, format_list_field


@dataclass(repr=False)
class GroupCollectiveArg:
    """The native args for group cast/reduce collective

    NOTE: for now, we always use the sub-class `A2AVBasedGroupCollectiveArg`
    since this parent class has incomplete attributes
    """

    input_split_size_list: list[int]
    output_split_size_list: list[int]
    dst_indices_list: list[list[int]]
    src_index_list: list[int]

    rank: int
    world_size: int
    group: dist.ProcessGroup
    device_mesh: DeviceMesh | None = None

    deterministic: bool = False

    def __post_init__(self):
        self.device = torch.cuda.current_device()

    def to_group_cast_args(self) -> dict:
        return dict(
            input_split_sizes=self.input_split_size_list,
            output_split_sizes=self.output_split_size_list,
            dst_indices=self.dst_indices_list,
            src_index=self.src_index_list,
        )

    def to_group_reduce_args(self) -> dict:
        return dict(
            input_split_sizes=self.output_split_size_list,
            output_split_sizes=self.input_split_size_list,
            dst_index=self.src_index_list,
            src_indices=self.dst_indices_list,
        )

    def __repr__(self) -> str:
        indent = ""
        repr_str = "GroupCollectiveArg(\n"

        repr_str += f"{indent}    rank={self.rank},\n"
        repr_str += f"{indent}    world_size={self.world_size},\n"
        repr_str += f"{indent}    device_mesh={repr(self.device_mesh)},\n"
        repr_str += f"{indent}    deterministic={self.deterministic},\n"
        repr_str += f"{indent}    device='{self.device}',\n"

        repr_str += f"{indent}    input_split_size_list={self.input_split_size_list},\n"
        repr_str += (
            f"{indent}    output_split_size_list={self.output_split_size_list},\n"
        )
        repr_str += f"{indent}    dst_indices_list={self.dst_indices_list},\n"
        repr_str += f"{indent}    src_index_list={self.src_index_list},\n"

        repr_str = repr_str.rstrip(",\n") + "\n)"
        return repr_str


@dataclass(repr=False)
class A2AVBasedGroupCollectiveArg(GroupCollectiveArg):
    """The a2a-v based args for group cast/reduce collective"""

    packed_times: int = 1
    reduce_op: ReduceOp = "sum"
    init_group_reduce: bool = True

    def __post_init__(self):
        super().__post_init__()

        # NOTE: only sum-reduce has non-deterministic kernel by now
        self.deterministic |= self.reduce_op != "sum"

        # ----   group cast args dict for packed tensors  ---- #

        self._group_cast_args_dict_packed = {
            # pack tensors along split dim by `self.packed_times` times
            k: v * self.packed_times  # type: ignore[operator]
            for k, v in {
                "input_split_sizes": self.input_split_size_list,
                "output_split_sizes": self.output_split_size_list,
                "dst_indices": self.dst_indices_list,
                "src_index": self.src_index_list,
            }.items()
        }

        # ----   group reduce args dict for packed tensors  ---- #

        if self.init_group_reduce:
            # symmetric to group-cast
            self._group_reduce_args_dict_packed = dict(
                input_split_sizes=self._group_cast_args_dict_packed[
                    "output_split_sizes"
                ],
                output_split_sizes=self._group_cast_args_dict_packed[
                    "input_split_sizes"
                ],
                dst_index=self._group_cast_args_dict_packed["src_index"],
                src_indices=self._group_cast_args_dict_packed["dst_indices"],
                reduce_op=self.reduce_op,
            )

        # ----   additional kwargs  ---- #

        if magi_attention.comm.is_hierarchical_comm_enable():
            assert (
                not magi_attention.comm.is_native_grpcoll_enable()
            ), "Hierarchical comm mode is not compatible with native grpcoll for now."
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
            if magi_attention.comm.is_native_grpcoll_enable():
                # init meta kwargs for native group-cast/reduce
                self._init_meta_kwargs_for_native_group_cast()
                if self.init_group_reduce:
                    self._init_meta_kwargs_for_native_group_reduce()
            else:
                # init meta kwargs for a2av group-cast/reduce
                self._init_meta_kwargs_for_a2av_group_cast()
                if self.init_group_reduce:
                    self._init_meta_kwargs_for_a2av_group_reduce()

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

    def _init_meta_kwargs_for_native_group_cast(self):
        # transfer group-cast meta args to dispatch meta args
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
        ) = get_dispatch_layout_from_group_cast_meta(
            input_split_size_list=self._group_cast_args_dict_packed[
                "input_split_sizes"
            ],
            dst_indices_list=self._group_cast_args_dict_packed["dst_indices"],
            group=self.group,
            num_nodes=1,  # TODO: support internode
        )

        # for group-cast/group-reduce, perm_to_a2av_idx is the post_perm_idx/pre_perm_idx
        _, post_perm_idx = get_a2av_perm_idxs_from_group_cast_meta(
            output_split_size_list=self._group_cast_args_dict_packed[
                "output_split_sizes"
            ],
            src_index_list=self._group_cast_args_dict_packed["src_index"],
            world_size=self.world_size,
        )

        self._group_cast_args_dict_packed["native_group_cast_meta_dict"] = dict(
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            post_perm_idx=post_perm_idx,
        )

        self._group_cast_args_dict_packed["native_grpcoll_handle_dict"] = dict(
            group_cast=None,
        )

    def _init_meta_kwargs_for_native_group_reduce(self):
        pre_perm_idx = self._group_cast_args_dict_packed["native_group_cast_meta_dict"][
            "post_perm_idx"
        ]

        self._group_reduce_args_dict_packed["native_group_reduce_meta_dict"] = dict(
            pre_perm_idx=pre_perm_idx,
        )
        # HACK: the symmetric group-cast handle dict is shared with symmetric group-reduce
        # since the "group_reduce" handle is not known until the "group_cast" returns
        self._group_reduce_args_dict_packed[
            "native_grpcoll_handle_dict"
        ] = self._group_cast_args_dict_packed["native_grpcoll_handle_dict"]

    def _init_meta_kwargs_for_a2av_group_cast(self):
        (
            self._group_cast_args_dict_packed["a2a_input_split_size"],
            self._group_cast_args_dict_packed["perm_before_a2a_kwargs"],
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self._group_cast_args_dict_packed[
                "input_split_sizes"
            ],
            dst_indices_list=self._group_cast_args_dict_packed["dst_indices"],
            world_size=self.world_size,
            device=self.device,
        )

        (
            self._group_cast_args_dict_packed["a2a_output_split_size"],
            self._group_cast_args_dict_packed["unperm_after_a2a_kwargs"],
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self._group_cast_args_dict_packed[
                "output_split_sizes"
            ],
            src_index_list=self._group_cast_args_dict_packed["src_index"],
            world_size=self.world_size,
            device=self.device,
        )

    def _init_meta_kwargs_for_a2av_group_reduce(self):
        (
            self._group_reduce_args_dict_packed["a2a_input_split_size"],
            self._group_reduce_args_dict_packed["perm_before_a2a_kwargs"],
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=self._group_reduce_args_dict_packed[
                "input_split_sizes"
            ],
            dst_index_list=self._group_reduce_args_dict_packed["dst_index"],
            world_size=self.world_size,
            device=self.device,
        )

        (
            self._group_reduce_args_dict_packed["a2a_output_split_size"],
            self._group_reduce_args_dict_packed["range_reduce_kwargs"],
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=self._group_reduce_args_dict_packed[
                "output_split_sizes"
            ],
            src_indices_list=self._group_reduce_args_dict_packed["src_indices"],
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

    def __repr__(self) -> str:
        # Get the representation of the base class
        base_repr_str = super().__repr__()

        # Replace the class name in the base representation
        # Find the first '(' and replace the part before it
        first_paren_idx = base_repr_str.find("(")
        if first_paren_idx != -1:
            base_repr_str = (
                f"{self.__class__.__name__}{base_repr_str[first_paren_idx:]}"
            )

        # Remove the final ')' from the base representation
        base_repr_lines = base_repr_str.splitlines()
        base_repr_str_without_closing = "\n".join(base_repr_lines[:-1])
        indent = ""  # This repr will add its own indentation starting from here

        repr_str = f"{base_repr_str_without_closing.rstrip(',')},\n"  # Remove trailing comma if exists and add our own

        repr_str += f"{indent}    packed_times={self.packed_times},\n"
        repr_str += f"{indent}    reduce_op='{self.reduce_op}',\n"
        repr_str += f"{indent}    init_group_reduce={self.init_group_reduce},\n"

        repr_str += f"{indent}    # Generated by __post_init__:\n"
        repr_str += format_dict_field(
            "_group_cast_args_dict_packed", self._group_cast_args_dict_packed, indent
        )

        if self.init_group_reduce:
            repr_str += format_dict_field(
                "_group_reduce_args_dict_packed",
                self._group_reduce_args_dict_packed,
                indent,
            )

        if magi_attention.comm.is_hierarchical_comm_enable():
            repr_str += f"{indent}    intra_group={repr(self.intra_group)},\n"
            repr_str += f"{indent}    inter_group={repr(self.inter_group)},\n"

        repr_str += f"{indent})"  # Add the new closing parenthesis

        return repr_str


@dataclass(repr=False)
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
        assert (
            self.overlap_degree >= 1
        ), f"Overlap degree must be >= 1, but got {self.overlap_degree=}"

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
                group=kv_group_collective_arg.group,
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
                    group=qo_group_collective_arg.group,
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
                        group=qo_group_collective_arg.group,
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
                        group=qo_group_collective_arg.group,
                        device_mesh=qo_group_collective_arg.device_mesh,
                        deterministic=qo_group_collective_arg.deterministic,
                        packed_times=1,  # out with lse along
                        reduce_op="lse",  # lse-reduce out and lse
                        init_group_reduce=True,
                    )
                )

    def __repr__(self) -> str:
        indent = ""
        repr_str = f"CommMeta(overlap_degree={self.overlap_degree},\n"

        # num_remote_kv_tokens_per_stage
        repr_str += f"{indent}    num_remote_kv_tokens_per_stage={self.num_remote_kv_tokens_per_stage},\n"
        # kv_group_collective_args_list
        repr_str += format_list_field(
            "kv_group_collective_args_list", self.kv_group_collective_args_list, indent
        )

        # num_remote_qo_tokens_per_stage
        repr_str += f"{indent}    num_remote_qo_tokens_per_stage={self.num_remote_qo_tokens_per_stage},\n"
        # qo_group_collective_args_list
        repr_str += format_list_field(
            "qo_group_collective_args_list", self.qo_group_collective_args_list, indent
        )

        # Generated fields from __post_init__
        repr_str += f"{indent}    # Generated by __post_init__:\n"

        # num_remote_qo_do_tokens_per_stage
        repr_str += f"{indent}    num_remote_qo_do_tokens_per_stage={self.num_remote_qo_do_tokens_per_stage},\n"
        # qo_do_group_collective_args_list
        repr_str += format_list_field(
            "qo_do_group_collective_args_list",
            self.qo_do_group_collective_args_list,
            indent,
        )

        # num_remote_out_lse_tokens_per_stage
        repr_str += f"{indent}    num_remote_out_lse_tokens_per_stage={self.num_remote_out_lse_tokens_per_stage},\n"
        # out_lse_group_collective_args_list
        repr_str += format_list_field(
            "out_lse_group_collective_args_list",
            self.out_lse_group_collective_args_list,
            indent,
        )

        repr_str = (
            repr_str.rstrip(",\n") + "\n)"
        )  # Remove trailing comma before final paren
        return repr_str
