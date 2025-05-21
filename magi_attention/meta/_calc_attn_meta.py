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
    high_bandwith_domain_size: int,
    overlap_config: OverlapConfig,
    cp_intra_group: dist.ProcessGroup | None = None,
    cp_inter_group: dist.ProcessGroup | None = None,
) -> tuple[CommMeta, AttnCalcMeta, DistAttnSolver]:
    """Calculate the communication and calculation meta from the dispatch meta

    Args:
        dispatch_meta_q (DispatchMeta): The dispatch meta for query
        dispatch_meta_k (DispatchMeta): The dispatch meta for key
        bucket_per_rank (list[AttnBucket]): The bucket per rank
        cp_group (dist.ProcessGroup): The NCCL process group
        high_bandwith_domain_size (int): The high bandwith domain size
        overlap_config (OverlapConfig): The overlap config, including the overlap mode, overlap degree, overlap chunk size, etc

    Returns:
        tuple[CommMeta, AttnCalcMeta]: The communication and calculation meta
    """

    attn_solver = DistAttnSolver(
        bucket_per_rank=bucket_per_rank,
        dispatch_meta_q=dispatch_meta_q,
        dispatch_meta_k=dispatch_meta_k,
        cp_group=cp_group,
        high_bandwith_domain_size=high_bandwith_domain_size,
        overlap_config=overlap_config,
        cp_intra_group=cp_intra_group,
        cp_inter_group=cp_inter_group,
    )

    comm_meta = attn_solver.calc_comm_meta()
    calc_meta = attn_solver.calc_attn_calc_meta()

    assert comm_meta.overlap_degree == calc_meta.overlap_degree, (
        "The overlap degree is inconsistent between "
        f"comm meta ({comm_meta.overlap_degree}) and calc meta ({calc_meta.overlap_degree})."
    )

    # DE-BUG: log attn solver, comm meta and calc meta
    # from magi_attention.utils import write_rank
    # write_rank(repr(attn_solver), "attn_solver.log")
    # write_rank(repr(comm_meta), "comm_meta.log")
    # write_rank(repr(calc_meta), "calc_meta.log")

    return comm_meta, calc_meta, attn_solver
