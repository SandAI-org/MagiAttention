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

import magi_attention
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import DispatchConfig, MinHeapDispatchAlg
from magi_attention.functional import dispatch_func, undispatch_func
from magi_attention.meta import calc_dispatch_meta_from_qk_ranges
from magi_attention.meta._calc_dispatch_meta import (
    cu_seqlens2seqlens,
    seqlens2cu_seqlens,
)
from magi_attention.testing.dist_common import DistTestBase, with_comms

WORLD_SIZE = 4
SEED = 42


class TestDispatcher(DistTestBase):
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
    def test_dispatch_and_undispatch(self):
        # --------------      setup       -------------- #

        rank = self.rank
        cp_size = self.world_size
        manual_seed = self.seed
        device = torch.cuda.current_device()
        torch.manual_seed(manual_seed)

        # --------------      init sample meta      -------------- #

        # TODO: limited to self-attn settings for now
        is_same_source = True
        is_q_permutable = True
        is_k_permutable = True

        q_ranges = AttnRanges.from_ranges(
            [
                (0, 1),
                (1, 5),
                (5, 12),
                (12, 16),
            ],
        )

        k_ranges = AttnRanges.from_ranges(
            [
                (0, 1),
                (1, 4),
                (5, 10),
                (12, 13),
            ]
        )

        attn_mask_type = [  # TODO: limited to all full attn masks for now
            AttnMaskType.FULL for _ in range(len(q_ranges))
        ]

        chunk_size = 4
        seq_dim = 0
        # TODO: test top-p minhp dispatch alg
        dispatch_config = DispatchConfig(alg=MinHeapDispatchAlg())

        # --------------      init global q, k       -------------- #

        global_q = torch.arange(q_ranges[-1].end * 2).view(-1, 2).to(device)  # (sq, 2)
        global_k = global_q * -1  # (sq, 2), due to self-attn

        # --------------      compute meta       -------------- #

        assert self.world_size % 2 == 0
        meta_q, meta_k, buckets_per_rank = calc_dispatch_meta_from_qk_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=q_ranges.end,
            total_seqlen_k=q_ranges.end,  # self-attn
            chunk_size=chunk_size,
            cp_rank=rank,
            cp_size=cp_size,
            dispatch_config=dispatch_config,
            is_same_source=is_same_source,
            is_q_permutable=is_q_permutable,
            is_k_permutable=is_k_permutable,
            # pass a placeholder since required
            high_bandwith_domain_size=1
            if magi_attention.is_hierarchical_comm_enable()
            else self.world_size // 2,
        )

        self.assertEqual(len(buckets_per_rank), cp_size)

        # --------------      dispatch to get host q, k       -------------- #

        host_q = dispatch_func(
            x_global=global_q,
            group=self.process_group,
            meta=meta_q,
            seq_dim=seq_dim,
        )

        host_k = dispatch_func(
            x_global=global_k,
            group=self.process_group,
            meta=meta_k,
            seq_dim=seq_dim,
        )

        # --------------      undispatch to restore global q, k       -------------- #

        global_q_und = undispatch_func(
            x_local=host_q,
            group=self.process_group,
            meta=meta_q,
            seq_dim=seq_dim,
        )

        global_k_und = undispatch_func(
            x_local=host_k,
            group=self.process_group,
            meta=meta_k,
            seq_dim=seq_dim,
        )

        # --------------      check       -------------- #

        self.assertTrue(torch.equal(global_q_und, global_q))
        self.assertTrue(torch.equal(global_k_und, global_k))

    def test_seqlens2cu_seqlens(self):
        # ---------    multi-elem seqlens    --------- #

        self.assertEqual(
            seqlens2cu_seqlens([5, 4, 3, 2, 1]),
            [0, 5, 9, 12, 14, 15],
        )

        # ---------    single-elem seqlens    --------- #

        self.assertEqual(
            seqlens2cu_seqlens([12]),
            [0, 12],
        )

        # ---------    empty seqlens    --------- #

        self.assertEqual(
            seqlens2cu_seqlens([0]),
            [0, 0],
        )

        self.assertEqual(
            seqlens2cu_seqlens([0, 0]),
            [0, 0, 0],
        )

        self.assertEqual(
            seqlens2cu_seqlens([]),
            [0],
        )

    def test_cu_seqlens2seqlens(self):
        # ---------    multi-elem cu_seqlens    --------- #

        self.assertEqual(
            cu_seqlens2seqlens([0, 5, 9, 12, 14, 15]),
            [5, 4, 3, 2, 1],
        )

        # ---------    single-elem cu_seqlens    --------- #

        self.assertEqual(
            cu_seqlens2seqlens([0, 12]),
            [12],
        )

        # ---------    empty cu_seqlens    --------- #

        self.assertEqual(
            cu_seqlens2seqlens([0, 0]),
            [0],
        )

        self.assertEqual(
            cu_seqlens2seqlens([0]),
            [],
        )

        self.assertEqual(
            cu_seqlens2seqlens([]),
            [],
        )


if __name__ == "__main__":
    run_tests()
