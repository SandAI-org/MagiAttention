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


import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta.algorithms import FastSNFDynamicAttnAlgorithm
from magi_attention.meta.collection.calc_meta import CalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta
from magi_attention.meta.collection.dispatch_meta import DispatchMeta
from magi_attention.meta.solver.dist_attn_solver import (
    BaseDistAttnSolver,
    DistAttnSolver,
)
from magi_attention.meta.solver.dynamic_attn_solver import DynamicAttnSolver
from magi_attention.meta.solver.overlap_solver import OverlapConfig
from magi_attention.utils import nvtx


@nvtx.instrument_nvtx
def make_attn_meta_from_dispatch_meta(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    dispatch_meta_q: DispatchMeta,
    dispatch_meta_k: DispatchMeta,
    cp_group: dist.ProcessGroup,
    overlap_config: OverlapConfig,
    cp_mesh: DeviceMesh | None = None,
    num_heads_q: int = 1,
    num_heads_kv: int = 1,
) -> tuple[CommMeta, CalcMeta, BaseDistAttnSolver]:
    """Make the communication and calculation meta from the dispatch meta

    Args:
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (list[AttnMaskType]): attn mask type (list)

        dispatch_meta_q (DispatchMeta): The dispatch meta for query
        dispatch_meta_k (DispatchMeta): The dispatch meta for key

        cp_group (dist.ProcessGroup): The NCCL process group

        overlap_config (OverlapConfig): The overlap config, including the overlap mode, overlap degree, overlap chunk size, etc

        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now

        num_heads_q (int): number of heads of query. Default: 1
        num_heads_kv (int): number of heads of key/value. Default: 1

    Returns:
        tuple[CommMeta, CalcMeta, BaseDistAttnSolver]:
            the communication meta, calculation meta and the attn solver
    """

    attn_solver: BaseDistAttnSolver
    if magi_attention.comm.is_qo_comm_enable():
        # NOTE: for now, we use dynamic attn solver when and only when enabling qo comm
        # however, we will unify the static/dynamic attn solver in the future
        attn_solver = DynamicAttnSolver(
            algorithm=FastSNFDynamicAttnAlgorithm(),
            cp_group=cp_group,
            dispatch_meta_q=dispatch_meta_q,
            dispatch_meta_k=dispatch_meta_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            cp_mesh=cp_mesh,
        )
        attn_solver.solve(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            flatten_head_groups=magi_attention.is_flatten_head_groups_enable(),
        )
        # only for debug: visualize the buckets
        # if cp_group.rank() == 0:
        #     attn_solver.output_solve_result(
        #         visualize=True, save_path="/home/littsk/lijin/MagiAttention/buckets.png"
        #     )
    else:
        attn_solver = DistAttnSolver(
            cp_group=cp_group,
            overlap_config=overlap_config,
            cp_mesh=cp_mesh,
        )
        attn_solver.solve(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            dispatch_meta_q=dispatch_meta_q,
            dispatch_meta_k=dispatch_meta_k,
        )

    assert attn_solver.is_solved
    comm_meta = attn_solver.make_comm_meta()
    calc_meta = attn_solver.make_calc_meta()

    assert comm_meta.overlap_degree == calc_meta.overlap_degree, (
        "The overlap degree is inconsistent between "
        f"comm meta ({comm_meta.overlap_degree=}) and calc meta ({calc_meta.overlap_degree=})."
    )

    return comm_meta, calc_meta, attn_solver
