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

import logging
from logging import getLogger

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta.algorithms import BinaryGreedyParallelDynamicAttnAlgorithm
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

logger = getLogger(__name__)


@nvtx.instrument_nvtx
def make_attn_meta_from_dispatch_meta(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    dispatch_meta_q: DispatchMeta,
    dispatch_meta_k: DispatchMeta,
    overlap_config: OverlapConfig,
    cp_group: dist.ProcessGroup,
    cp_mesh: DeviceMesh | None = None,
) -> tuple[CommMeta, CalcMeta, BaseDistAttnSolver]:
    """Make the communication and calculation meta from the dispatch meta

    Args:
        q_ranges (AttnRanges): the global query ranges.
        k_ranges (AttnRanges): the global key ranges.
        attn_mask_type (list[AttnMaskType]): the attn mask type list.

        num_heads_q (int): the number of heads of query.
        num_heads_kv (int): the number of heads of key/value.
        head_dim (int): the dimension of each attention head.

        dispatch_meta_q (DispatchMeta): the dispatch meta for query.
        dispatch_meta_k (DispatchMeta): the dispatch meta for key/value.
        overlap_config (OverlapConfig): the overlap config.

        cp_group (dist.ProcessGroup): the process group.
        cp_mesh (DeviceMesh, optional): the process mesh. Defaults to ``None``.

    Returns:
        tuple[CommMeta, CalcMeta, BaseDistAttnSolver]:
            the communication meta, calculation meta and the attn solver.
    """

    # Solve attention
    attn_solver: BaseDistAttnSolver
    if magi_attention.comm.is_qo_comm_enable():
        # NOTE: for now, we use dynamic attn solver when and only when enabling qo comm
        # however, we will unify the static/dynamic attn solver in the future
        attn_solver = DynamicAttnSolver(
            algorithm=BinaryGreedyParallelDynamicAttnAlgorithm(),
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            cp_group=cp_group,
            dispatch_meta_q=dispatch_meta_q,
            dispatch_meta_k=dispatch_meta_k,
            cp_mesh=cp_mesh,
        )
        attn_solver.solve(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
        )
        # Visualize the buckets only for debug
        if logger.isEnabledFor(logging.DEBUG) and cp_group.rank() == 0:
            logger.debug("Visualizing the buckets...")
            attn_solver.output_solve_result(
                visualize=True, save_path="./dyn_solver_buckets.png"
            )
    else:
        attn_solver = DistAttnSolver(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            overlap_config=overlap_config,
            cp_group=cp_group,
            cp_mesh=cp_mesh,
        )
        attn_solver.solve(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            dispatch_meta_q=dispatch_meta_q,
            dispatch_meta_k=dispatch_meta_k,
        )

    # Make comm/calc meta
    assert attn_solver.is_solved
    comm_meta = attn_solver.make_comm_meta()
    calc_meta = attn_solver.make_calc_meta()

    # Sanity check
    assert comm_meta.overlap_degree == calc_meta.overlap_degree, (
        "The overlap degree is inconsistent between "
        f"comm meta ({comm_meta.overlap_degree=}) and calc meta ({calc_meta.overlap_degree=})."
    )

    return comm_meta, calc_meta, attn_solver
