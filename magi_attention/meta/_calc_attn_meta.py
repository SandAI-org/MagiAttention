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

from magi_attention.meta.collection.calc_meta import AttnCalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta
from magi_attention.meta.collection.dispatch_meta import DispatchMeta
from magi_attention.meta.container.bucket import AttnBucket
from magi_attention.meta.solver.dist_attn_solver import DistAttnSolver
from magi_attention.meta.solver.overlap_solver import OverlapConfig


def calc_attn_meta_from_dispatch_meta(
    dispatch_meta_q: DispatchMeta,
    dispatch_meta_k: DispatchMeta,
    bucket_per_rank: list[AttnBucket],
    cp_group: dist.ProcessGroup,
    overlap_config: OverlapConfig,
    cp_mesh: DeviceMesh | None = None,
) -> tuple[CommMeta, AttnCalcMeta, DistAttnSolver]:
    """Calculate the communication and calculation meta from the dispatch meta

    Args:
        dispatch_meta_q (DispatchMeta): The dispatch meta for query
        dispatch_meta_k (DispatchMeta): The dispatch meta for key
        bucket_per_rank (list[AttnBucket]): The bucket per rank
        cp_group (dist.ProcessGroup): The NCCL process group
        overlap_config (OverlapConfig): The overlap config, including the overlap mode, overlap degree, overlap chunk size, etc
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now.

    Returns:
        tuple[CommMeta, AttnCalcMeta]: The communication and calculation meta
    """

    attn_solver = DistAttnSolver(
        bucket_per_rank=bucket_per_rank,
        dispatch_meta_q=dispatch_meta_q,
        dispatch_meta_k=dispatch_meta_k,
        cp_group=cp_group,
        overlap_config=overlap_config,
        cp_mesh=cp_mesh,
    )

    comm_meta = attn_solver.calc_comm_meta()
    calc_meta = attn_solver.calc_attn_calc_meta()

    assert comm_meta.overlap_degree == calc_meta.overlap_degree, (
        "The overlap degree is inconsistent between "
        f"comm meta ({comm_meta.overlap_degree}) and calc meta ({calc_meta.overlap_degree})."
    )

    return comm_meta, calc_meta, attn_solver
