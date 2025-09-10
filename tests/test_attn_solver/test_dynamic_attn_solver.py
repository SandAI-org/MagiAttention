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


import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.mask import MaskIterator
from magi_attention.common import AttnRanges, AttnRectangles
from magi_attention.meta.algorithms import (
    GRGDynamicAttnAlgorithm,
    NCQDynamicAttnAlgorithm,
)
from magi_attention.meta.solver.dynamic_attn_solver import DynamicAttnSolver
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms

WORLD_SIZE = 4
SEED = 42

# MaskIterator settings
TOTAL_SEQLEN = 100
MASK_GENERATE_TIMES = 1


class TestDynamicAttnSolver(DistTestBase):
    """DynamicAttnSolver test"""

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    @property
    def seed(self) -> int:
        return SEED

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    @parameterize(
        "mask_type",
        [
            FlashMaskType.FULL,
            FlashMaskType.CAUSAL,
            FlashMaskType.CAUSAL_DOCUMENT,
            FlashMaskType.FULL_DOCUMENT,
            FlashMaskType.SHARE_QUESTION,
            FlashMaskType.CAUSAL_BLOCKWISE,
            FlashMaskType.PREFIX_LM_DOCUMENT,
            FlashMaskType.PREFIX_LM_CAUSAL,
            FlashMaskType.QK_SPARSE,
        ],
    )
    @parameterize(
        "algorithm_type",
        [
            "ncq",
            "grg",
        ],
    )
    @parameterize(
        "heads_config",
        [
            {"num_heads_q": 1, "num_heads_kv": 1},  # MHA (Multi-Head Attention)
            {"num_heads_q": 8, "num_heads_kv": 8},  # MHA (Multi-Head Attention)
            {"num_heads_q": 8, "num_heads_kv": 1},  # GQA (Grouped Query Attention)
            {"num_heads_q": 16, "num_heads_kv": 4},  # GQA with different ratio
        ],
    )
    def test_dynamic_attn_solver_solve_function(
        self, mask_type, algorithm_type, heads_config
    ):
        """test DynamicAttnSolver solve function correctness"""
        # --------------      setup       -------------- #

        rank = self.rank
        cp_size = self.world_size
        manual_seed = self.seed
        torch.manual_seed(manual_seed)

        # --------------      init parameters      -------------- #

        mask_iterator = MaskIterator(
            generate_times=MASK_GENERATE_TIMES,
            generate_mask=mask_type,
            total_seqlen=TOTAL_SEQLEN,
            to_attn_ranges=True,
            seed=manual_seed,
        )

        for q_ranges, k_ranges, attn_mask_type in mask_iterator:
            total_seqlen_q: int = TOTAL_SEQLEN
            total_seqlen_k: int = TOTAL_SEQLEN
            num_heads_q: int = heads_config["num_heads_q"]
            num_heads_kv: int = heads_config["num_heads_kv"]

            # generate sequential host ranges
            def create_ranges_per_rank(
                total_seqlen: int, world_size: int
            ) -> list[AttnRanges]:
                """generate sequential ranges for each rank"""
                ranges_per_rank = []
                seqlen_per_rank = total_seqlen // world_size

                for rank_idx in range(world_size):
                    start_seq = rank_idx * seqlen_per_rank
                    end_seq = (
                        (rank_idx + 1) * seqlen_per_rank
                        if rank_idx < world_size - 1
                        else total_seqlen
                    )

                    rank_ranges = [[start_seq, end_seq]]
                    ranges_per_rank.append(AttnRanges.from_ranges(rank_ranges))

                return ranges_per_rank

            host_q_ranges_global_this_rank = create_ranges_per_rank(
                total_seqlen_q, cp_size
            )
            host_k_ranges_global_this_rank = create_ranges_per_rank(
                total_seqlen_k, cp_size
            )

            # --------------      init algorithm      -------------- #

            if algorithm_type == "ncq":
                algorithm = NCQDynamicAttnAlgorithm()
            elif algorithm_type == "grg":
                algorithm = GRGDynamicAttnAlgorithm()
            else:
                raise ValueError(f"Unknown algorithm type: {algorithm_type}")

            # --------------      init solver      -------------- #

            solver = DynamicAttnSolver(
                algorithm=algorithm,
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
                host_ranges_q=host_q_ranges_global_this_rank,
                host_ranges_k=host_k_ranges_global_this_rank,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                cp_rank=rank,
                cp_size=cp_size,
                calc_local_range=True,
            )

            # --------------      solve      -------------- #
            if rank == 0:
                print(
                    f"solve: {algorithm_type=}, {heads_config=}, {mask_type=}, {q_ranges=}, {k_ranges=}, {attn_mask_type=}"
                )
            solver.solve(
                q_ranges=q_ranges, k_ranges=k_ranges, mask_types=attn_mask_type
            )

            # --------------      verify results      -------------- #
            # verify each q k attn position is correct
            rects = AttnRectangles.from_ranges(
                q_ranges=q_ranges, k_ranges=k_ranges, mask_types=attn_mask_type
            )

            # all q k attn position should be 1 in sample
            map = [[0 for _ in range(total_seqlen_k)] for _ in range(total_seqlen_q)]
            for rect in rects:
                for i in range(rect.q_range.start, rect.q_range.end):
                    for j in range(rect.k_range.start, rect.k_range.end):
                        if j - i >= rect.d_range.start and j - i <= rect.d_range.end:
                            map[i][j] += 1
                            if map[i][j] > 1:
                                raise ValueError(
                                    f"overlap sample: q_range: {rect.q_range}, k_range: {rect.k_range}, position: {i}, {j}"
                                )

            # all q k attn position should be add 1 and equal2 in solver result
            for i in range(cp_size):
                rects = solver.bucket_per_rank[i]
                for rect in rects:
                    for i in range(rect.q_range.start, rect.q_range.end):
                        for j in range(rect.k_range.start, rect.k_range.end):
                            if (
                                j - i >= rect.d_range.start
                                and j - i <= rect.d_range.end
                            ):
                                map[i][j] += 1
                                if map[i][j] > 2:
                                    raise ValueError(
                                        (
                                            f"overlap solver result: rank {i}, q_range: {rect.q_range}, "
                                            f"k_range: {rect.k_range}, position: {i}, {j}"
                                        )
                                    )
                                if map[i][j] == 1:
                                    raise ValueError(
                                        (
                                            f"excess solver result: rank {i}, q_range: {rect.q_range}, "
                                            f"k_range: {rect.k_range}, position: {i}, {j}"
                                        )
                                    )

            # all q k attn position should be 2 in solver result
            for i in range(total_seqlen_q):
                for j in range(total_seqlen_k):
                    if map[i][j] == 1:
                        raise ValueError(
                            f"missing solver result: q_range: {i}, k_range: {j}, position: {i}, {j}"
                        )

            # --------------      test calc_local_range meta generation      -------------- #

            attn_calc_meta = solver.calc_attn_calc_meta()

            comm_meta = solver.calc_comm_meta()

            # --------------      verify local calc meta generation      -------------- #

            local_attn_arg = attn_calc_meta.local_attn_arg
            job_num = len(local_attn_arg.q_ranges)
            host_range_q_max_len = max(
                [range_q.end for range_q in solver.host_ranges_q[rank]]
            )
            host_range_k_max_len = max(
                [range_k.end for range_k in solver.host_ranges_k[rank]]
            )
            for i in range(job_num):
                q_range = local_attn_arg.q_ranges[i]
                k_range = local_attn_arg.k_ranges[i]
                # verify q_range in comm_meta range
                if not (q_range.start >= 0 and q_range.end <= host_range_q_max_len):
                    raise ValueError(
                        f"local calc meta generation: q_range: {q_range}, host_range_q_max_len: {host_range_q_max_len}"
                    )

                # verify k_range in comm_meta range
                if not (k_range.start >= 0 and k_range.end <= host_range_k_max_len):
                    raise ValueError(
                        f"local calc meta generation: k_range: {k_range}, host_range_k_max_len: {host_range_k_max_len}"
                    )

            # --------------      verify remote calc meta generation      -------------- #

            remote_attn_args_list = attn_calc_meta.remote_attn_args_list
            stage = 0
            for remote_attn_arg in remote_attn_args_list:
                job_num = len(remote_attn_arg.q_ranges)
                remote_range_q_max_len = comm_meta.num_remote_qo_tokens_per_stage[stage]
                remote_range_k_max_len = comm_meta.num_remote_kv_tokens_per_stage[stage]
                for i in range(job_num):
                    q_range = remote_attn_arg.q_ranges[i]
                    k_range = remote_attn_arg.k_ranges[i]
                    # verify q_range in comm_meta remote range
                    if not (
                        q_range.start >= 0 and q_range.end <= remote_range_q_max_len
                    ):
                        raise ValueError(
                            (
                                f"remote calc meta generation: q_range: {q_range}, "
                                f"remote_range_q_max_len: {remote_range_q_max_len}"
                            )
                        )

                    # verify k_range in comm_meta remote range
                    if not (
                        k_range.start >= 0 and k_range.end <= remote_range_k_max_len
                    ):
                        raise ValueError(
                            (
                                f"remote calc meta generation: k_range: {k_range}, "
                                f"remote_range_k_max_len: {remote_range_k_max_len}"
                            )
                        )

                stage += 1


if __name__ == "__main__":
    run_tests()
