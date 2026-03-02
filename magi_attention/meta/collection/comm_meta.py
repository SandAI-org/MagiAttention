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

from dataclasses import dataclass
from typing import Literal

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
    get_native_group_cast_meta,
    get_num_rdma_recv_tokens,
)
from magi_attention.common.enum import GroupReduceOp
from magi_attention.utils import format_dict_field, format_list_field


@dataclass(repr=False)
class GroupCollectiveArg:
    """The basic comm args for group collective implementations"""

    input_split_size_list: list[int]
    output_split_size_list: list[int]
    dst_indices_list: list[list[int]]
    src_index_list: list[int]

    rank: int
    world_size: int
    group: dist.ProcessGroup
    device_mesh: DeviceMesh | None = None

    deterministic: bool = False

    split_alignment: int = 1

    def __post_init__(self):
        pass

    def to_group_cast_args(self) -> dict:
        return self.to_packed_group_cast_args()

    def to_group_reduce_args(self) -> dict:
        return self.to_packed_group_reduce_args()

    def to_packed_group_cast_args(self, packed_times: int = 1) -> dict:
        # pack args along split dim by `packed_times` times
        return dict(
            input_split_sizes=self.input_split_size_list * packed_times,
            output_split_sizes=self.output_split_size_list * packed_times,
            dst_indices=self.dst_indices_list * packed_times,
            src_index=self.src_index_list * packed_times,
        )

    def to_packed_group_reduce_args(self, packed_times: int = 1) -> dict:
        # symmetric to group-cast
        # pack args along split dim by `packed_times` times
        return dict(
            input_split_sizes=self.output_split_size_list * packed_times,
            output_split_sizes=self.input_split_size_list * packed_times,
            dst_index=self.src_index_list * packed_times,
            src_indices=self.dst_indices_list * packed_times,
        )

    def compute_send_recv_token_counts(
        self, reduce_op: Literal["sum", "max"] = "max"
    ) -> None:
        self._compute_group_cast_send_recv_token_counts(reduce_op=reduce_op)
        self._compute_group_reduce_send_recv_token_counts(reduce_op=reduce_op)

    def _compute_group_cast_send_recv_token_counts(
        self, reduce_op: Literal["sum", "max"] = "max"
    ) -> None:
        group_cast_args = self.to_group_cast_args()

        # calculate for group cast
        cast_input_split_size_list: list[int] = group_cast_args["input_split_sizes"]
        cast_output_split_size_list: list[int] = group_cast_args["output_split_sizes"]
        cast_dst_indices_list: list[list[int]] = group_cast_args["dst_indices"]

        cast_send_tokens = sum(
            [
                split_size * len(dst_indice)
                for split_size, dst_indice in zip(
                    cast_input_split_size_list, cast_dst_indices_list
                )
            ]
        )
        cast_recv_tokens = sum(cast_output_split_size_list)

        self.group_cast_comm_tokens = self._reduce_send_recv_tokens(
            cast_send_tokens, cast_recv_tokens, reduce_op
        )

    def _compute_group_reduce_send_recv_token_counts(
        self, reduce_op: Literal["sum", "max"] = "max"
    ) -> None:
        group_reduce_args = self.to_group_reduce_args()

        # calculate for group reduce
        reduce_input_split_size_list: list[int] = group_reduce_args["input_split_sizes"]
        reduce_output_split_size_list: list[int] = group_reduce_args[
            "output_split_sizes"
        ]
        reduce_src_indices_list: list[list[int]] = group_reduce_args["src_indices"]

        reduce_send_tokens = sum(reduce_input_split_size_list)
        reduce_recv_tokens = sum(
            [
                split_size * len(dst_indice)
                for split_size, dst_indice in zip(
                    reduce_output_split_size_list, reduce_src_indices_list
                )
            ]
        )

        self.group_reduce_comm_tokens = self._reduce_send_recv_tokens(
            reduce_send_tokens, reduce_recv_tokens, reduce_op
        )

    def _reduce_send_recv_tokens(
        self,
        send_tokens: int,
        recv_tokens: int,
        reduce_op: Literal["sum", "max"] = "max",
    ) -> int:
        match reduce_op:
            case "sum":
                return send_tokens + recv_tokens
            case "max":
                return max(send_tokens, recv_tokens)
            case _:
                raise ValueError(f"Invalid reduce_op: {reduce_op}")

    def __repr__(self) -> str:  # pragma: no cover
        indent = ""
        repr_str = "GroupCollectiveArg(\n"

        repr_str += f"{indent}    rank={self.rank},\n"
        repr_str += f"{indent}    world_size={self.world_size},\n"
        repr_str += f"{indent}    device_mesh={repr(self.device_mesh)},\n"
        repr_str += f"{indent}    deterministic={self.deterministic},\n"
        repr_str += f"{indent}    split_alignment={self.split_alignment},\n"

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
    """The args for a2a-v based group collective implementations"""

    packed_times: int = 1
    reduce_op: GroupReduceOp = "sum"
    init_group_reduce: bool = True

    def __post_init__(self):
        super().__post_init__()

        assert (
            not magi_attention.comm.is_native_grpcoll_enable()
        ), "This arg dataclass is not supported for native grpcoll."

        self.device = torch.cuda.current_device()

        # NOTE: only sum-reduce has non-deterministic kernel by now
        self.deterministic |= self.reduce_op != "sum"

        # ----   packed group cast args dict  ---- #

        self._group_cast_args_dict_packed = self.to_packed_group_cast_args(
            packed_times=self.packed_times
        )

        # ----   packed group reduce args dict  ---- #

        if self.init_group_reduce:
            self._group_reduce_args_dict_packed = self.to_packed_group_reduce_args(
                packed_times=self.packed_times
            )
            self._group_reduce_args_dict_packed.update(reduce_op=self.reduce_op)

        # ----   compute cast and reduce comm tokens ---- #

        self.compute_send_recv_token_counts()

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

    def compute_send_recv_token_counts(
        self, reduce_op: Literal["sum", "max"] = "max"
    ) -> None:
        self._compute_group_cast_send_recv_token_counts(reduce_op=reduce_op)
        if self.init_group_reduce:
            self._compute_group_reduce_send_recv_token_counts(reduce_op=reduce_op)

    def __repr__(self) -> str:  # pragma: no cover
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
        repr_str += f"{indent}    device={self.device},\n"
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
class NativeGroupCollectiveArg(GroupCollectiveArg):
    """The comm args for native group collective implementation"""

    def __post_init__(self):
        super().__post_init__()

        assert (
            magi_attention.comm.is_native_grpcoll_enable()
        ), "This arg dataclass is only supported for native grpcoll."
        assert (
            not magi_attention.comm.is_hierarchical_comm_enable()
        ), "This arg dataclass is not supported for hierarchical comm for now."

        self._check_split_alignment()

        self.device = torch.cuda.current_device()

        # transfer group-cast meta args to dispatch meta args
        # HACK: for now, we only support internode grpcoll
        # with intranode world size of 8
        self.num_nodes = max(1, self.group.size() // 8)

        # ----   original group cast args dict  ---- #

        self._group_cast_args_dict = super().to_group_cast_args()

        # ----   original group reduce args dict  ---- #

        self._group_reduce_args_dict = super().to_group_reduce_args()

        # ----   compute cast and reduce comm tokens ---- #

        self.compute_send_recv_token_counts()

        # ----   additional kwargs  ---- #

        # init meta kwargs for native grpcoll
        self._init_meta_kwargs_for_native_group_cast()
        self._init_meta_kwargs_for_native_group_reduce()

    def _check_split_alignment(self):
        if self.split_alignment > 1:  # non-trivial alignment
            for idx, split in enumerate(self.input_split_size_list):
                assert split % self.split_alignment == 0, (
                    f"Each input split size must be multiple of {self.split_alignment} "
                    f"for better performance, but got {self.input_split_size_list=}, where the {idx}-th {split=}"
                )
            for idx, split in enumerate(self.output_split_size_list):
                assert split % self.split_alignment == 0, (
                    f"Each output split size must be multiple of {self.split_alignment} "
                    f"for better performance, but got {self.output_split_size_list=}, where the {idx}-th {split=}"
                )

    def _init_meta_kwargs_for_native_group_cast(self):
        self._preprocess_args_for_split_alignment()

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
        ) = get_native_group_cast_meta(
            input_split_sizes=self._group_cast_args_dict["input_split_sizes"],
            dst_indices=self._group_cast_args_dict["dst_indices"],
            group=self.group,
            num_nodes=self.num_nodes,
        )

        if num_tokens_per_rdma_rank is not None:
            assert num_tokens_per_rdma_rank.size(0) == self.num_nodes
            # NOTE: for internode grpcoll, besides providing output buffer,
            # we have to pass extra `internode_output_seqlen` to fully avoid GPU-CPU sync
            self._group_cast_args_dict[
                "internode_output_seqlen"
            ] = get_num_rdma_recv_tokens(
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                group=self.group,
            )

        # for group-cast/group-reduce,
        # `perm_to_a2av_idx`` is used as `post_perm_idx` / `pre_perm_idx` resp.
        post_perm_idx = get_a2av_perm_idxs_from_group_cast_meta(
            output_split_sizes=self._group_cast_args_dict["output_split_sizes"],
            src_index=self._group_cast_args_dict["src_index"],
            num_ranks=self.world_size,
        )

        self._group_cast_args_dict["native_group_cast_meta_dict"] = dict(
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            post_perm_idx=post_perm_idx,
        )

        self._group_cast_args_dict["native_grpcoll_handle_dict"] = dict(
            group_cast=None,
        )

    def _init_meta_kwargs_for_native_group_reduce(self):
        pre_perm_idx = self._group_cast_args_dict["native_group_cast_meta_dict"][
            "post_perm_idx"
        ]

        self._group_reduce_args_dict["native_group_reduce_meta_dict"] = dict(
            pre_perm_idx=pre_perm_idx,
        )
        # HACK: the symmetric group-cast handle dict is shared with symmetric group-reduce
        # since the "group_reduce" handle is not known until the "group_cast" returns
        self._group_reduce_args_dict[
            "native_grpcoll_handle_dict"
        ] = self._group_cast_args_dict["native_grpcoll_handle_dict"]

    def _preprocess_args_for_split_alignment(self):
        if self.split_alignment > 1:
            self._group_cast_args_dict["input_split_sizes"] = [
                split // self.split_alignment
                for split in self._group_cast_args_dict["input_split_sizes"]
            ]
            self._group_cast_args_dict["output_split_sizes"] = [
                split // self.split_alignment
                for split in self._group_cast_args_dict["output_split_sizes"]
            ]
            self._group_reduce_args_dict[
                "input_split_sizes"
            ] = self._group_cast_args_dict["output_split_sizes"]
            self._group_reduce_args_dict[
                "output_split_sizes"
            ] = self._group_cast_args_dict["input_split_sizes"]

            self._group_cast_args_dict["split_alignment"] = self.split_alignment
            self._group_reduce_args_dict["split_alignment"] = self.split_alignment

    def to_group_cast_args(self) -> dict:
        return self._group_cast_args_dict

    def to_group_reduce_args(self) -> dict:
        return self._group_reduce_args_dict

    def __repr__(self) -> str:  # pragma: no cover
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

        repr_str += f"{indent}    # Generated by __post_init__:\n"
        repr_str += f"{indent}    device={self.device},\n"
        repr_str += f"{indent}    num_nodes={self.num_nodes},\n"
        repr_str += format_dict_field(
            "_group_cast_args_dict", self._group_cast_args_dict, indent
        )
        repr_str += format_dict_field(
            "_group_reduce_args_dict",
            self._group_reduce_args_dict,
            indent,
        )

        repr_str += f"{indent})"  # Add the new closing parenthesis

        return repr_str


@dataclass(repr=False)
class CommMeta:
    # for k,v comm in fwd and dk,dv comm in bwd
    # NOTE: this denotes sk or sv, not sk + sv
    num_remote_kv_tokens_per_stage: list[int]
    kv_group_collective_args_list: list[GroupCollectiveArg]

    # for q,o,lse comm in fwd and q,o,do,dq,lse comm in bwd
    # NOTE: this denotes sq or so, not sq + so
    num_remote_qo_tokens_per_stage: list[int]
    qo_group_collective_args_list: list[GroupCollectiveArg]

    num_heads_q: int
    num_heads_kv: int
    head_dim: int

    @property
    def overlap_degree(self) -> int:
        return len(self.num_remote_kv_tokens_per_stage)

    @property
    def num_heads_per_group(self) -> int:
        return self._num_heads_per_group

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
            self.overlap_degree >= 0
        ), f"Overlap degree must be >= 0, but got {self.overlap_degree=}"

        assert self.num_heads_q % self.num_heads_kv == 0, (
            f"num_heads_q must be divisible by num_heads_kv, "
            f"but got {self.num_heads_q=} and {self.num_heads_kv=}"
        )
        self._num_heads_per_group = self.num_heads_q // self.num_heads_kv

        if magi_attention.comm.is_native_grpcoll_enable():
            self._init_native_grpcoll_args()
        else:
            self._init_a2av_based_grpcoll_args()

    def _init_a2av_based_grpcoll_args(self):
        """If using a2a-v based group collective,
        the following variables are generated within `__post_init__`
        and serve as the meta arguments for dist-attn communication.

        - num_remote_qo_do_tokens_per_stage: list[int]
        - qo_do_group_collective_args_list: list[A2AVBasedGroupCollectiveArg]
        - num_remote_out_lse_tokens_per_stage: list[int]
        - out_lse_group_collective_args_list: list[A2AVBasedGroupCollectiveArg]

        And the original args will also be inplace modified

        - num_remote_kv_tokens_per_stage will time by 2
        - kv_group_collective_args_list will become `list[A2AVBasedGroupCollectiveArg]`
        with packed_times=2 and reduce_op="sum"

        - num_remote_qo_tokens_per_stage will stay the same
        - qo_group_collective_args_list will become `list[A2AVBasedGroupCollectiveArg]`
        with packed_times=1 and reduce_op="sum"
        """

        # -----     init a2a-v based group collective args     ----- #

        self.num_remote_qo_do_tokens_per_stage: list[int] = []
        self.qo_do_group_collective_args_list: list[A2AVBasedGroupCollectiveArg] = []
        self.num_remote_out_lse_tokens_per_stage: list[int] = []
        self.out_lse_group_collective_args_list: list[A2AVBasedGroupCollectiveArg] = []
        for stage in range(self.overlap_degree):
            num_remote_kv_tokens = self.num_remote_kv_tokens_per_stage[stage]
            kv_group_collective_kwargs = vars(self.kv_group_collective_args_list[stage])

            # --- for fetch packed kv and reduce packed dkv  --- #

            self.num_remote_kv_tokens_per_stage[stage] = num_remote_kv_tokens * 2
            self.kv_group_collective_args_list[stage] = A2AVBasedGroupCollectiveArg(
                **kv_group_collective_kwargs,
                packed_times=2,  # pack kv/dkv along seqlen dim
                reduce_op="sum",  # sum-reduce dkv
                init_group_reduce=True,
            )

            if magi_attention.comm.is_qo_comm_enable():
                num_remote_qo_tokens = self.num_remote_qo_tokens_per_stage[stage]
                qo_group_collective_kwargs = vars(
                    self.qo_group_collective_args_list[stage]
                )

                # --- for fetch q and reduce dq  --- #

                self.qo_group_collective_args_list[stage] = A2AVBasedGroupCollectiveArg(
                    **qo_group_collective_kwargs,
                    packed_times=1,  # q,dq,(q,o,do),lse
                    reduce_op="sum",  # sum-reduce dq
                    init_group_reduce=True,
                )

                # --- for fetch packed q,o,do --- #

                self.num_remote_qo_do_tokens_per_stage.append(num_remote_qo_tokens * 3)
                self.qo_do_group_collective_args_list.append(
                    A2AVBasedGroupCollectiveArg(
                        **qo_group_collective_kwargs,
                        packed_times=3,  # pack q,o,do along seqlen dim
                        reduce_op="sum",
                        init_group_reduce=False,  # no reduce
                    )
                )

                # --- for reduce out with lse --- #

                self.num_remote_out_lse_tokens_per_stage.append(num_remote_qo_tokens)
                self.out_lse_group_collective_args_list.append(
                    A2AVBasedGroupCollectiveArg(
                        **qo_group_collective_kwargs,
                        packed_times=1,  # out with lse
                        reduce_op="lse",  # lse-reduce out and lse
                        init_group_reduce=True,
                    )
                )

    def _init_native_grpcoll_args(self):
        """If using native group collective,
        the original args will be inplace modified

        - num_remote_kv_tokens_per_stage will stay the same
        - kv_group_collective_args_list will become `list[NativeGroupCollectiveArg]`

        - num_remote_qo_tokens_per_stage will stay the same
        - qo_group_collective_args_list will become `list[NativeGroupCollectiveArg]`
        """

        # -----     init native group collective args     ----- #

        for stage in range(self.overlap_degree):
            # --- for fetch tupled kv, and reduce tupled dkv  --- #

            kv_group_collective_kwargs = vars(self.kv_group_collective_args_list[stage])

            self.kv_group_collective_args_list[stage] = NativeGroupCollectiveArg(
                **kv_group_collective_kwargs,
            )

            if magi_attention.comm.is_qo_comm_enable():
                # --- for fetch q, reduce dq, fetch tupled q,o,do,lse and reduce out with lse  --- #

                qo_group_collective_kwargs = vars(
                    self.qo_group_collective_args_list[stage]
                )
                self.qo_group_collective_args_list[stage] = NativeGroupCollectiveArg(
                    **qo_group_collective_kwargs,
                )

    def __repr__(self) -> str:  # pragma: no cover
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

        # num_heads_q, num_heads_kv and num_heads_per_group
        repr_str += f"{indent}    num_heads_q={self.num_heads_q},\n"
        repr_str += f"{indent}    num_heads_kv={self.num_heads_kv},\n"
        repr_str += f"{indent}    head_dim={self.head_dim},\n"

        # Generated fields from __post_init__
        repr_str += f"{indent}    # Generated by __post_init__:\n"
        repr_str += f"{indent}    num_heads_per_group={self.num_heads_per_group},\n"
        if not magi_attention.comm.is_native_grpcoll_enable():
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

        # Remove trailing comma before final paren
        repr_str = repr_str.rstrip(",\n") + "\n)"

        return repr_str
