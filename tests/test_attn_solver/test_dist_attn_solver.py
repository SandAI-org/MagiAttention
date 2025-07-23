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

import types
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRange, AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import DispatchConfig, MinHeapDispatchAlg
from magi_attention.meta import calc_dispatch_meta_from_qk_ranges
from magi_attention.meta.container.rank_entry import HostRankEntry, RemoteRankEntry
from magi_attention.meta.container.slice import AttnSlice, MultiKAttnSlice
from magi_attention.meta.solver.dist_attn_solver import DistAttnSolver
from magi_attention.meta.solver.overlap_solver import (
    OverlapConfig,
    OverlapSolver,
    UniformOverlapAlg,
)
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms

WORLD_SIZE = 4
SEED = 42


def add_range_to_array(
    array: np.ndarray,
    q_range: AttnRange,
    k_range: AttnRange,
    masktype: AttnMaskType = AttnMaskType.FULL,
    check: bool = False,
):
    # get start and end of range
    x_start, x_end = q_range.start, q_range.end
    y_start, y_end = k_range.start, k_range.end

    if check:
        # check whether the current slice has been filled
        assert np.all(array[x_start:x_end, y_start:y_end] == 0), (
            f"Part of the area has been added," f"when {q_range=} and {k_range=}"
        )

    # fill the area according to the type of the mask.
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            if masktype == AttnMaskType.FULL:
                array[i][j] = 1
            elif masktype == AttnMaskType.CAUSAL:
                b = y_end - x_end
                fx = i + b
                if j <= fx:
                    array[i][j] = 1
            elif masktype == AttnMaskType.INVCAUSAL:
                b = y_start - x_start
                fx = i + b
                if j >= fx:
                    array[i][j] = 1
            elif masktype == AttnMaskType.BICAUSAL:
                causal_b = y_end - x_end
                f_causal = i + causal_b

                inv_causal_b = y_start - x_start
                f_inv_causal = i + inv_causal_b
                if j <= f_causal and j >= f_inv_causal:
                    array[i][j] = 1

    return array


def make_range_global(
    global_ranges: AttnRanges,
    local_range: AttnRange,
) -> AttnRanges:
    """convert local_range to global_ranges with base global_ranges

    Args:
        global_ranges (AttnRanges): the actual base global ranges
        local_range (AttnRange): range need to convert

    Returns:
        AttnRanges: converted multiple ranges since local range may
            be converted to multiple segments of ranges
    """
    assert local_range.seqlen <= global_ranges.total_seqlen

    ranges_ = AttnRanges()

    local_start, local_length = local_range.start, local_range.seqlen

    global_index = 0
    current_global_length = 0
    start_length = local_start

    while global_index < len(global_ranges):
        if global_ranges[global_index].seqlen <= start_length:
            start_length -= global_ranges[global_index].seqlen
            global_index += 1
        else:
            current_global_length = start_length
            break

    while global_index < len(global_ranges):
        if global_ranges[global_index].seqlen - current_global_length < local_length:
            range_ = AttnRange(
                start=global_ranges[global_index].start + current_global_length,
                end=global_ranges[global_index].end,
            )
            local_length = (
                local_length
                - global_ranges[global_index].seqlen
                + current_global_length
            )
            global_index += 1
            current_global_length = 0
            ranges_.append(range_)
        else:
            range_ = AttnRange(
                start=global_ranges[global_index].start + current_global_length,
                end=global_ranges[global_index].start
                + current_global_length
                + local_length,
            )
            ranges_.append(range_)
            break

    return ranges_


def determine_ith_range_masktype(
    i: int,
    length: int,
    masktype: AttnMaskType = AttnMaskType.FULL,
):
    """
    determine mask type in tests for Slice,
    when convert local range with one single masktype to global range with multi masktypes
    """
    if length == 1 and masktype is AttnMaskType.BICAUSAL:
        return AttnMaskType.BICAUSAL
    if i == 0 and masktype is AttnMaskType.BICAUSAL:
        return AttnMaskType.INVCAUSAL
    if i == length - 1 and masktype is AttnMaskType.BICAUSAL:
        return AttnMaskType.CAUSAL
    if i == 0 and masktype is AttnMaskType.INVCAUSAL:
        return AttnMaskType.INVCAUSAL
    if i == length - 1 and masktype is AttnMaskType.CAUSAL:
        return AttnMaskType.CAUSAL
    return AttnMaskType.FULL


class TestDistAttnSolver(DistTestBase):
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    @property
    def seed(self) -> int:
        return SEED

    @property
    def high_bandwith_domain_size(self) -> int:
        # TODO: add test when high_bandwith_domain_size > 1
        return 1

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    @parameterize(
        "testcase",
        [
            {
                "name": "testcase_1",
                "q_ranges": AttnRanges.from_ranges(
                    ranges=[[0, 1], [1, 5], [5, 12], [12, 16]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    ranges=[[0, 1], [1, 4], [5, 10], [12, 13]]
                ),
                "attn_mask_type": [
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                ],
                "host_q_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[8, 12]]),
                    AttnRanges.from_ranges(ranges=[[4, 8]]),
                    AttnRanges.from_ranges(ranges=[[0, 4]]),
                    AttnRanges.from_ranges(ranges=[[12, 16]]),
                ],
                "host_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[8, 12]]),
                    AttnRanges.from_ranges(ranges=[[4, 8]]),
                    AttnRanges.from_ranges(ranges=[[0, 4]]),
                    AttnRanges.from_ranges(ranges=[[12, 16]]),
                ],
                "remote_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[5, 8]]),
                    AttnRanges.from_ranges(ranges=[[1, 4], [8, 10]]),
                    AttnRanges.from_ranges(ranges=[]),
                    AttnRanges.from_ranges(ranges=[]),
                ],
                "chunk_size": 4,
            },
            {
                "name": "testcase_2",
                "q_ranges": AttnRanges.from_ranges(
                    ranges=[[0, 10], [10, 16], [16, 30], [30, 43], [43, 61], [61, 64]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    ranges=[[0, 11], [5, 18], [18, 32], [32, 45], [45, 64], [53, 64]]
                ),
                "attn_mask_type": [
                    AttnMaskType.FULL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                ],
                "host_q_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[16, 24], [48, 56]]),
                    AttnRanges.from_ranges(ranges=[[32, 48]]),
                    AttnRanges.from_ranges(ranges=[[24, 32], [56, 64]]),
                    AttnRanges.from_ranges(ranges=[[0, 16]]),
                ],
                "host_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[16, 24], [48, 56]]),
                    AttnRanges.from_ranges(ranges=[[32, 48]]),
                    AttnRanges.from_ranges(ranges=[[24, 32], [56, 64]]),
                    AttnRanges.from_ranges(ranges=[[0, 16]]),
                ],
                "remote_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[24, 26], [45, 48], [56, 64]]),
                    AttnRanges.from_ranges(ranges=[[48, 64]]),
                    AttnRanges.from_ranges(ranges=[[18, 24], [32, 34], [45, 56]]),
                    AttnRanges.from_ranges(ranges=[[16, 18]]),
                ],
                "chunk_size": 8,
            },
            {
                "name": "testcase_3_causal_1",
                "q_ranges": AttnRanges.from_ranges(
                    ranges=[
                        [0, 8],
                        [8, 24],
                        [24, 38],
                        [38, 57],
                        [57, 83],
                        [83, 92],
                        [92, 96],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    ranges=[
                        [12, 20],
                        [27, 43],
                        [43, 48],
                        [48, 67],
                        [5, 74],
                        [31, 86],
                        [67, 96],
                    ]
                ),
                "attn_mask_type": [
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                ],
                "host_q_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[0, 8], [40, 48], [72, 80]]),
                    AttnRanges.from_ranges(ranges=[[8, 24], [80, 88]]),
                    AttnRanges.from_ranges(ranges=[[32, 40], [48, 56], [64, 72]]),
                    AttnRanges.from_ranges(ranges=[[24, 32], [56, 64], [88, 96]]),
                ],
                "host_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[0, 8], [40, 48], [72, 80]]),
                    AttnRanges.from_ranges(ranges=[[8, 24], [80, 88]]),
                    AttnRanges.from_ranges(ranges=[[32, 40], [48, 56], [64, 72]]),
                    AttnRanges.from_ranges(ranges=[[24, 32], [56, 64], [88, 96]]),
                ],
                "remote_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[8, 40], [48, 71]]),
                    AttnRanges.from_ranges(ranges=[[5, 8], [24, 80]]),
                    AttnRanges.from_ranges(ranges=[[5, 32], [40, 48], [56, 64]]),
                    AttnRanges.from_ranges(ranges=[[5, 24], [32, 56], [64, 88]]),
                ],
                "chunk_size": 8,
            },
            {
                "name": "testcase_4_causal_2",
                "q_ranges": AttnRanges.from_ranges(
                    ranges=[[0, 30], [30, 53], [53, 74], [74, 96]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    ranges=[[0, 50], [61, 71], [34, 47], [57, 90]]
                ),
                "attn_mask_type": [
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                ],
                "host_q_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[16, 24], [48, 56], [64, 72]]),
                    AttnRanges.from_ranges(ranges=[[24, 32], [40, 48], [72, 80]]),
                    AttnRanges.from_ranges(ranges=[[8, 16], [56, 64], [80, 88]]),
                    AttnRanges.from_ranges(ranges=[[0, 8], [32, 40], [88, 96]]),
                ],
                "host_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[16, 24], [48, 56], [64, 72]]),
                    AttnRanges.from_ranges(ranges=[[24, 32], [40, 48], [72, 80]]),
                    AttnRanges.from_ranges(ranges=[[8, 16], [56, 64], [80, 88]]),
                    AttnRanges.from_ranges(ranges=[[0, 8], [32, 40], [88, 96]]),
                ],
                "remote_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[0, 16], [24, 45], [61, 64]]),
                    AttnRanges.from_ranges(
                        ranges=[[0, 24], [32, 40], [48, 50], [57, 72]]
                    ),
                    AttnRanges.from_ranges(ranges=[[0, 8], [16, 37], [64, 80]]),
                    AttnRanges.from_ranges(ranges=[[8, 28], [57, 88]]),
                ],
                "chunk_size": 8,
            },
            {
                "name": "testcase_5",
                "q_ranges": AttnRanges.from_ranges(
                    ranges=[
                        [0, 2],
                        [2, 5],
                        [5, 8],
                        [8, 10],
                        [10, 14],
                        [14, 16],
                        [16, 22],
                        [22, 29],
                        [29, 32],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    ranges=[
                        [0, 2],
                        [2, 5],
                        [5, 8],
                        [8, 16],
                        [8, 16],
                        [8, 16],
                        [4, 20],
                        [17, 24],
                        [19, 31],
                    ]
                ),
                "attn_mask_type": [
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.FULL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.FULL,
                ],
                "host_q_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[16, 24]]),
                    AttnRanges.from_ranges(ranges=[[24, 32]]),
                    AttnRanges.from_ranges(ranges=[[8, 16]]),
                    AttnRanges.from_ranges(ranges=[[0, 8]]),
                ],
                "host_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[16, 24]]),
                    AttnRanges.from_ranges(ranges=[[24, 32]]),
                    AttnRanges.from_ranges(ranges=[[8, 16]]),
                    AttnRanges.from_ranges(ranges=[[0, 8]]),
                ],
                "remote_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[4, 16]]),
                    AttnRanges.from_ranges(ranges=[[17, 24]]),
                    AttnRanges.from_ranges(ranges=[]),
                    AttnRanges.from_ranges(ranges=[]),
                ],
                "chunk_size": 8,
            },
            {
                "name": "testcase_6_all_full",
                "q_ranges": AttnRanges.from_ranges(
                    ranges=[
                        [0, 8],
                        [8, 22],
                        [22, 38],
                        [38, 51],
                        [51, 56],
                        [56, 64],
                        [64, 72],
                        [72, 89],
                        [89, 96],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    ranges=[
                        [15, 30],
                        [56, 74],
                        [25, 87],
                        [7, 58],
                        [71, 90],
                        [62, 90],
                        [71, 90],
                        [3, 96],
                        [7, 49],
                    ]
                ),
                "attn_mask_type": [
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                ],
                "host_q_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[0, 8], [48, 56], [80, 88]]),
                    AttnRanges.from_ranges(ranges=[[8, 24], [72, 80]]),
                    AttnRanges.from_ranges(ranges=[[24, 32], [64, 72], [88, 96]]),
                    AttnRanges.from_ranges(ranges=[[32, 48], [56, 64]]),
                ],
                "host_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[0, 8], [48, 56], [80, 88]]),
                    AttnRanges.from_ranges(ranges=[[8, 24], [72, 80]]),
                    AttnRanges.from_ranges(ranges=[[24, 32], [64, 72], [88, 96]]),
                    AttnRanges.from_ranges(ranges=[[32, 48], [56, 64]]),
                ],
                "remote_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[8, 48], [56, 80], [88, 96]]),
                    AttnRanges.from_ranges(ranges=[[3, 8], [24, 72], [80, 96]]),
                    AttnRanges.from_ranges(ranges=[[3, 24], [32, 64], [72, 88]]),
                    AttnRanges.from_ranges(ranges=[[7, 32], [48, 56], [64, 90]]),
                ],
                "chunk_size": 8,
            },
            {
                "name": "testcase_7",
                "q_ranges": AttnRanges.from_ranges(ranges=[[0, 96]]),
                "k_ranges": AttnRanges.from_ranges(ranges=[[48, 51]]),
                "attn_mask_type": [AttnMaskType.CAUSAL],
                "host_q_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[0, 16], [88, 96]]),
                    AttnRanges.from_ranges(ranges=[[64, 88]]),
                    AttnRanges.from_ranges(ranges=[[40, 64]]),
                    AttnRanges.from_ranges(ranges=[[16, 40]]),
                ],
                "host_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[0, 16], [88, 96]]),
                    AttnRanges.from_ranges(ranges=[[64, 88]]),
                    AttnRanges.from_ranges(ranges=[[40, 64]]),
                    AttnRanges.from_ranges(ranges=[[16, 40]]),
                ],
                "remote_k_ranges_global_this_rank": [
                    AttnRanges.from_ranges(ranges=[[48, 51]]),
                    AttnRanges.from_ranges(ranges=[]),
                    AttnRanges.from_ranges(ranges=[]),
                    AttnRanges.from_ranges(ranges=[]),
                ],
                "chunk_size": 8,
            },
        ],
    )
    def test_init_host_remote_ranges_global(self, testcase):
        # --------------      setup       -------------- #

        rank = self.rank
        cp_size = self.world_size
        manual_seed = self.seed
        testcase_name = testcase["name"]
        torch.manual_seed(manual_seed)

        # --------------      init sample meta      -------------- #

        # TODO: limited to self-attn settings for now
        is_same_source = True
        is_q_permutable = True
        is_k_permutable = True

        # TODO: test top-p minhp dispatch alg
        dispatch_config = DispatchConfig(alg=MinHeapDispatchAlg())

        # ------------  init SimpleNamespace class ------------ #

        test_solver_class = SimpleNamespace()
        test_solver_class.cp_rank = rank
        test_solver_class.high_bandwith_domain_size = self.high_bandwith_domain_size
        test_solver_class._init_host_remote_ranges_global_this_rank = types.MethodType(
            DistAttnSolver._init_host_remote_ranges_global_this_rank, test_solver_class
        )

        # --------------      compute meta       -------------- #

        q_ranges: AttnRanges = testcase.get("q_ranges")
        k_ranges: AttnRanges = testcase.get("k_ranges")
        attn_mask_type: list[AttnMaskType] = testcase.get("attn_mask_type")
        chunk_size: int = testcase.get("chunk_size")

        expected_host_q_ranges_global_this_rank: AttnRanges = testcase.get(
            "host_q_ranges_global_this_rank"
        )[rank]
        expected_host_k_ranges_global_this_rank: AttnRanges = testcase.get(
            "host_k_ranges_global_this_rank"
        )[rank]
        expected_remote_k_ranges_global_this_rank: AttnRanges = testcase.get(
            "remote_k_ranges_global_this_rank"
        )[rank]

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
            high_bandwith_domain_size=self.high_bandwith_domain_size,
        )

        # ----------- compute host and remote ranges ------------ #

        bucket_this_rank = buckets_per_rank[rank]
        (
            host_q_ranges_global_this_rank,
            host_k_ranges_global_this_rank,
            remote_k_ranges_global_this_rank,
            # TODO: test hb domain and lb domain
            remote_k_ranges_global_hb_domain,
            remote_k_ranges_global_lb_domain,
        ) = test_solver_class._init_host_remote_ranges_global_this_rank(
            dispatch_meta_q=meta_q,
            dispatch_meta_k=meta_k,
            bucket_this_rank=bucket_this_rank,
        )

        assert (
            host_q_ranges_global_this_rank == expected_host_q_ranges_global_this_rank
        ), (
            f"Get {host_q_ranges_global_this_rank=}, "
            f"when expected result={expected_host_q_ranges_global_this_rank} in {testcase_name}"
        )
        assert (
            host_k_ranges_global_this_rank == expected_host_k_ranges_global_this_rank
        ), (
            f"Get {host_k_ranges_global_this_rank=}, "
            f"when expected result={expected_host_k_ranges_global_this_rank} in {testcase_name}"
        )
        assert (
            remote_k_ranges_global_this_rank
            == expected_remote_k_ranges_global_this_rank
        ), (
            f"Get {remote_k_ranges_global_this_rank=}, "
            f"when expected result={expected_remote_k_ranges_global_this_rank} in {testcase_name}"
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    @parameterize(
        "testcase",
        [
            {
                "name": "testcase_2",
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 10], [10, 16], [16, 30], [30, 43], [43, 61], [61, 64]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [[0, 11], [5, 18], [18, 32], [32, 45], [45, 64], [53, 64]]
                ),
                "attn_mask_type": [
                    AttnMaskType.FULL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                ],
                "attn_calc_host_slice_local_list": [
                    [
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[0, 6]),
                            k_range=AttnRange.from_range(attn_range=[2, 8]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[6, 8]),
                            k_range=AttnRange.from_range(attn_range=[2, 8]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[8, 16]),
                            k_range=AttnRange.from_range(attn_range=[8, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                    ],
                    [
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[0, 8]),
                            k_range=AttnRange.from_range(attn_range=[0, 10]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[8, 11]),
                            k_range=AttnRange.from_range(attn_range=[0, 13]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[11, 16]),
                            k_range=AttnRange.from_range(attn_range=[13, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                    ],
                    [
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[0, 6]),
                            k_range=AttnRange.from_range(attn_range=[0, 8]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[8, 13]),
                            k_range=AttnRange.from_range(attn_range=[8, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[13, 16]),
                            k_range=AttnRange.from_range(attn_range=[8, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                    ],
                    [
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[0, 8]),
                            k_range=AttnRange.from_range(attn_range=[0, 11]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[8, 10]),
                            k_range=AttnRange.from_range(attn_range=[0, 11]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[10, 14]),
                            k_range=AttnRange.from_range(attn_range=[5, 16]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[14, 16]),
                            k_range=AttnRange.from_range(attn_range=[5, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                    ],
                ],
                "attn_calc_remote_slice_list_per_chunk": [
                    [
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[6, 8]),
                                k_ranges=AttnRanges.from_ranges([[24, 26]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 16]),
                                k_ranges=AttnRanges.from_ranges([[45, 47]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 16]),
                                k_ranges=AttnRanges.from_ranges([[47, 48], [56, 59]]),
                                mask_types=[AttnMaskType.FULL, AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 16]),
                                k_ranges=AttnRanges.from_ranges([[59, 63]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 16]),
                                k_ranges=AttnRanges.from_ranges([[63, 64]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                    ],
                    [
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 16]),
                                k_ranges=AttnRanges.from_ranges([[48, 52]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 16]),
                                k_ranges=AttnRanges.from_ranges([[52, 56]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 16]),
                                k_ranges=AttnRanges.from_ranges([[56, 60]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 16]),
                                k_ranges=AttnRanges.from_ranges([[60, 64]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                    ],
                    [
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 6]),
                                k_ranges=AttnRanges.from_ranges([[18, 22]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 6]),
                                k_ranges=AttnRanges.from_ranges([[22, 24]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[6, 8]),
                                k_ranges=AttnRanges.from_ranges([[32, 34]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 13]),
                                k_ranges=AttnRanges.from_ranges([[45, 49]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 13]),
                                k_ranges=AttnRanges.from_ranges([[49, 53]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 13]),
                                k_ranges=AttnRanges.from_ranges([[53, 56]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[13, 16]),
                                k_ranges=AttnRanges.from_ranges([[53, 56]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                    ],
                    [
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_ranges=AttnRanges.from_ranges([[16, 18]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            )
                        ]
                    ],
                ],
                "chunk_size": 8,
                "min_chunk_size": 4,
                "max_num_chunks": 16,
            },
            {
                "name": "testcase_3_all_causal_1",
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 8], [8, 24], [24, 38], [38, 57], [57, 83], [83, 92], [92, 96]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [12, 20],
                        [27, 43],
                        [43, 48],
                        [48, 67],
                        [5, 74],
                        [31, 86],
                        [67, 96],
                    ]
                ),
                "attn_mask_type": [
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                ],
                "attn_calc_host_slice_local_list": [
                    [
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[16, 24]),
                            k_range=AttnRange.from_range(attn_range=[5, 16]),
                            mask_type=AttnMaskType.FULL,
                        )
                    ],
                    [
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[16, 19]),
                            k_range=AttnRange.from_range(attn_range=[0, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[22, 24]),
                            k_range=AttnRange.from_range(attn_range=[16, 18]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                    ],
                    [
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[6, 8]),
                            k_range=AttnRange.from_range(attn_range=[8, 10]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[8, 16]),
                            k_range=AttnRange.from_range(attn_range=[8, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[14, 16]),
                            k_range=AttnRange.from_range(attn_range=[16, 18]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[16, 17]),
                            k_range=AttnRange.from_range(attn_range=[0, 16]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[17, 24]),
                            k_range=AttnRange.from_range(attn_range=[0, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                    ],
                    [
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[8, 9]),
                            k_range=AttnRange.from_range(attn_range=[8, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[9, 16]),
                            k_range=AttnRange.from_range(attn_range=[0, 8]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[16, 20]),
                            k_range=AttnRange.from_range(attn_range=[7, 16]),
                            mask_type=AttnMaskType.FULL,
                        ),
                        AttnSlice(
                            q_range=AttnRange.from_range(attn_range=[20, 24]),
                            k_range=AttnRange.from_range(attn_range=[16, 24]),
                            mask_type=AttnMaskType.CAUSAL,
                        ),
                    ],
                ],
                "attn_calc_remote_slice_list_per_chunk": [
                    [
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 4]),
                                k_ranges=AttnRanges.from_ranges([[12, 16]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[4, 8]),
                                k_ranges=AttnRanges.from_ranges([[12, 16]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[8, 16]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[4, 8]),
                                k_ranges=AttnRanges.from_ranges([[16, 20]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[16, 24]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[24, 32]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[32, 40]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 14]),
                                k_ranges=AttnRanges.from_ranges([[48, 56]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_ranges=AttnRanges.from_ranges([[48, 56]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[48, 56]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_ranges=AttnRanges.from_ranges([[56, 58]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_ranges=AttnRanges.from_ranges([[56, 64]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 24]),
                                k_ranges=AttnRanges.from_ranges([[56, 64]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 24]),
                                k_ranges=AttnRanges.from_ranges([[64, 71]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            )
                        ],
                    ],
                    [
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 2]),
                                k_ranges=AttnRanges.from_ranges([[27, 29]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[2, 8]),
                                k_ranges=AttnRanges.from_ranges([[27, 29]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 16]),
                                k_ranges=AttnRanges.from_ranges([[27, 29]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_ranges=AttnRanges.from_ranges([[5, 8], [24, 29]]),
                                mask_types=[AttnMaskType.FULL, AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[2, 8]),
                                k_ranges=AttnRanges.from_ranges([[29, 35]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 10]),
                                k_ranges=AttnRanges.from_ranges([[29, 37]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[10, 16]),
                                k_ranges=AttnRanges.from_ranges([[29, 37]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_ranges=AttnRanges.from_ranges([[29, 37]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_ranges=AttnRanges.from_ranges([[31, 37]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[10, 16]),
                                k_ranges=AttnRanges.from_ranges([[37, 43]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_ranges=AttnRanges.from_ranges([[37, 45]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_ranges=AttnRanges.from_ranges([[37, 45]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_ranges=AttnRanges.from_ranges([[45, 53]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_ranges=AttnRanges.from_ranges([[45, 53]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_ranges=AttnRanges.from_ranges([[53, 61]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_ranges=AttnRanges.from_ranges([[53, 61]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_ranges=AttnRanges.from_ranges([[61, 69]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_ranges=AttnRanges.from_ranges([[61, 69]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_ranges=AttnRanges.from_ranges([[69, 74]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_ranges=AttnRanges.from_ranges([[69, 77]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 22]),
                                k_ranges=AttnRanges.from_ranges([[77, 80]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_ranges=AttnRanges.from_ranges([[77, 80]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                    ],
                    [
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[5, 13]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[13, 21]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[21, 29]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[1, 3]),
                                k_ranges=AttnRanges.from_ranges([[43, 45]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[3, 6]),
                                k_ranges=AttnRanges.from_ranges([[43, 45]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_ranges=AttnRanges.from_ranges([[29, 32], [40, 45]]),
                                mask_types=[AttnMaskType.FULL, AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[3, 6]),
                                k_ranges=AttnRanges.from_ranges([[45, 48]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 11]),
                                k_ranges=AttnRanges.from_ranges([[56, 61]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 16]),
                                k_ranges=AttnRanges.from_ranges([[56, 61]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_ranges=AttnRanges.from_ranges([[45, 48]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 22]),
                                k_ranges=AttnRanges.from_ranges([[45, 48], [56, 61]]),
                                mask_types=[AttnMaskType.FULL, AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_ranges=AttnRanges.from_ranges([[45, 48], [56, 61]]),
                                mask_types=[AttnMaskType.FULL, AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 14]),
                                k_ranges=AttnRanges.from_ranges([[61, 64]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_ranges=AttnRanges.from_ranges([[61, 64]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_ranges=AttnRanges.from_ranges([[61, 63]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                        ],
                    ],
                    [
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 16]),
                                k_ranges=AttnRanges.from_ranges([[5, 13]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 16]),
                                k_ranges=AttnRanges.from_ranges([[13, 21]]),
                                mask_types=[AttnMaskType.FULL],
                            )
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 16]),
                                k_ranges=AttnRanges.from_ranges([[21, 24], [32, 37]]),
                                mask_types=[AttnMaskType.FULL, AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 20]),
                                k_ranges=AttnRanges.from_ranges([[32, 37]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 16]),
                                k_ranges=AttnRanges.from_ranges([[37, 45]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 20]),
                                k_ranges=AttnRanges.from_ranges([[37, 45]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 9]),
                                k_ranges=AttnRanges.from_ranges([[48, 53]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 14]),
                                k_ranges=AttnRanges.from_ranges([[45, 53]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_ranges=AttnRanges.from_ranges([[45, 53]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 20]),
                                k_ranges=AttnRanges.from_ranges([[45, 53]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 9]),
                                k_ranges=AttnRanges.from_ranges([[53, 56], [64, 67]]),
                                mask_types=[AttnMaskType.FULL, AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_ranges=AttnRanges.from_ranges([[53, 55]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 20]),
                                k_ranges=AttnRanges.from_ranges([[53, 56], [64, 69]]),
                                mask_types=[AttnMaskType.FULL, AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_ranges=AttnRanges.from_ranges([[67, 69]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 20]),
                                k_ranges=AttnRanges.from_ranges([[69, 77]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_ranges=AttnRanges.from_ranges([[69, 77]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_ranges=AttnRanges.from_ranges([[77, 85]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 20]),
                                k_ranges=AttnRanges.from_ranges([[77, 85]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_ranges=AttnRanges.from_ranges([[77, 85]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                        [
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 20]),
                                k_ranges=AttnRanges.from_ranges([[85, 86]]),
                                mask_types=[AttnMaskType.CAUSAL],
                            ),
                            MultiKAttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_ranges=AttnRanges.from_ranges([[85, 88]]),
                                mask_types=[AttnMaskType.FULL],
                            ),
                        ],
                    ],
                ],
                "chunk_size": 8,
                "min_chunk_size": 8,
                "max_num_chunks": 16,
            },
        ],
    )
    def test_init_host_rank_entry(self, testcase):
        # --------------      setup       -------------- #

        rank = self.rank
        cp_size = self.world_size
        manual_seed = self.seed
        testcase_name = testcase["name"]
        torch.manual_seed(manual_seed)

        # --------------      init sample meta      -------------- #

        # TODO: limited to self-attn settings for now
        is_same_source = True
        is_q_permutable = True
        is_k_permutable = True

        # TODO: test top-p minhp dispatch alg
        dispatch_config = DispatchConfig(alg=MinHeapDispatchAlg())

        # ------------  init SimpleNamespace class ------------ #

        test_solver_class = SimpleNamespace()
        test_solver_class.cp_rank = rank
        test_solver_class.high_bandwith_domain_size = self.high_bandwith_domain_size
        _init_host_remote_ranges_global_this_rank = types.MethodType(
            DistAttnSolver._init_host_remote_ranges_global_this_rank, test_solver_class
        )
        _init_host_rank_entry_this_rank = types.MethodType(
            DistAttnSolver._init_host_rank_entry_this_rank, test_solver_class
        )
        test_solver_class._chunk_remote_k_ranges_global = types.MethodType(
            DistAttnSolver._chunk_remote_k_ranges_global, test_solver_class
        )
        test_solver_class._make_ith_attn_calc_host_slice = types.MethodType(
            DistAttnSolver._make_ith_attn_calc_host_slice, test_solver_class
        )
        test_solver_class._make_ith_attn_calc_remote_slice_per_chunk = types.MethodType(
            DistAttnSolver._make_ith_attn_calc_remote_slice_per_chunk, test_solver_class
        )
        test_solver_class._make_ith_attn_calc_remote_slice = types.MethodType(
            DistAttnSolver._make_ith_attn_calc_remote_slice, test_solver_class
        )

        # --------------      compute meta       -------------- #

        q_ranges: AttnRanges = testcase.get("q_ranges")
        k_ranges: AttnRanges = testcase.get("k_ranges")
        attn_mask_type: list[AttnMaskType] = testcase.get("attn_mask_type")
        chunk_size: int = testcase.get("chunk_size")
        min_chunk_size: int = testcase.get("min_chunk_size")
        max_num_chunks: int = testcase.get("max_num_chunks")
        test_solver_class.overlap_config = OverlapConfig(
            min_chunk_size=min_chunk_size, max_num_chunks=max_num_chunks
        )

        expected_attn_calc_host_slice_local_list = testcase.get(
            "attn_calc_host_slice_local_list"
        )[rank]
        expected_attn_calc_remote_slice_list_per_chunk = testcase.get(
            "attn_calc_remote_slice_list_per_chunk"
        )[rank]

        # --------------      compute meta       -------------- #

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
            high_bandwith_domain_size=self.high_bandwith_domain_size,
        )

        # ----------- compute host and remote ranges ------------ #

        bucket_this_rank = buckets_per_rank[rank]
        (
            host_q_ranges_global_this_rank,
            host_k_ranges_global_this_rank,
            remote_k_ranges_global_this_rank,
            remote_k_ranges_global_hb_domain,
            remote_k_ranges_global_lb_domain,
        ) = _init_host_remote_ranges_global_this_rank(
            dispatch_meta_q=meta_q,
            dispatch_meta_k=meta_k,
            bucket_this_rank=bucket_this_rank,
        )

        # -------------- compute host rank entry -------------- #

        host_rank_entry_this_rank = _init_host_rank_entry_this_rank(
            host_q_ranges_global=host_q_ranges_global_this_rank,
            host_k_ranges_global=host_k_ranges_global_this_rank,
            remote_k_ranges_global_hb_domain=remote_k_ranges_global_hb_domain,
            remote_k_ranges_global_lb_domain=remote_k_ranges_global_lb_domain,
            attn_calc_slice_global_list=bucket_this_rank.attn_slices,
        )

        assert (
            host_rank_entry_this_rank.attn_calc_host_slice_local_list
            == expected_attn_calc_host_slice_local_list
        ), (
            f"Get host_slice_local_list={host_rank_entry_this_rank.attn_calc_host_slice_local_list}, \n"
            f"when expected result={expected_attn_calc_host_slice_local_list} \n"
            f"in {testcase_name}"
        )
        assert (
            host_rank_entry_this_rank.attn_calc_remote_slice_list_per_chunk
            == expected_attn_calc_remote_slice_list_per_chunk
        ), (
            f"Get remote_slice_list={host_rank_entry_this_rank.attn_calc_remote_slice_list_per_chunk}, \n"
            f"when expected result={expected_attn_calc_remote_slice_list_per_chunk} \n"
            f"in {testcase_name}"
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    @parameterize(
        "testcase",
        [
            {
                "name": "testcase_1",
                "chunk_size": 8,
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "degree": 1,
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 10], [10, 16], [16, 30], [30, 43], [43, 61], [61, 64]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [[0, 11], [5, 18], [18, 32], [32, 45], [45, 64], [53, 64]]
                ),
                "attn_mask_type": [
                    AttnMaskType.FULL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.FULL,
                    AttnMaskType.FULL,
                ],
                "remote_k_ranges_global": [
                    [AttnRanges.from_ranges([[24, 26], [45, 48], [56, 64]])],
                    [AttnRanges.from_ranges([[48, 64]])],
                    [AttnRanges.from_ranges([[18, 24], [32, 34], [45, 56]])],
                    [AttnRanges.from_ranges([[16, 18]])],
                ],
                "attn_calc_remote_slice_local_list": [
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[6, 8]),
                                k_range=AttnRange.from_range(attn_range=[0, 2]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 16]),
                                k_range=AttnRange.from_range(attn_range=[2, 13]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ]
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            )
                        ]
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 6]),
                                k_range=AttnRange.from_range(attn_range=[0, 6]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[6, 8]),
                                k_range=AttnRange.from_range(attn_range=[6, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 13]),
                                k_range=AttnRange.from_range(attn_range=[8, 19]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[13, 16]),
                                k_range=AttnRange.from_range(attn_range=[16, 19]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ]
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 2]),
                                mask_type=AttnMaskType.CAUSAL,
                            )
                        ]
                    ],
                ],
            },
            {
                "name": "testcase_2",
                "chunk_size": 8,
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "degree": 1,
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 8], [8, 24], [24, 38], [38, 57], [57, 83], [83, 92], [92, 96]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [12, 20],
                        [27, 43],
                        [43, 48],
                        [48, 67],
                        [5, 74],
                        [31, 86],
                        [67, 96],
                    ]
                ),
                "attn_mask_type": [
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                ],
                "remote_k_ranges_global": [
                    [AttnRanges.from_ranges([[8, 40], [48, 71]])],
                    [AttnRanges.from_ranges([[5, 8], [24, 80]])],
                    [AttnRanges.from_ranges([[5, 32], [40, 48], [56, 64]])],
                    [AttnRanges.from_ranges([[5, 24], [32, 56], [64, 88]])],
                ],
                "attn_calc_remote_slice_local_list": [
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 4]),
                                k_range=AttnRange.from_range(attn_range=[4, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[4, 8]),
                                k_range=AttnRange.from_range(attn_range=[4, 12]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 14]),
                                k_range=AttnRange.from_range(attn_range=[32, 40]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[32, 42]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_range=AttnRange.from_range(attn_range=[0, 48]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 55]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                        ]
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 2]),
                                k_range=AttnRange.from_range(attn_range=[6, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[2, 8]),
                                k_range=AttnRange.from_range(attn_range=[6, 14]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 10]),
                                k_range=AttnRange.from_range(attn_range=[6, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[10, 16]),
                                k_range=AttnRange.from_range(attn_range=[6, 22]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 53]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 22]),
                                k_range=AttnRange.from_range(attn_range=[10, 59]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_range=AttnRange.from_range(attn_range=[10, 59]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ]
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[1, 3]),
                                k_range=AttnRange.from_range(attn_range=[30, 32]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[3, 6]),
                                k_range=AttnRange.from_range(attn_range=[30, 35]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 11]),
                                k_range=AttnRange.from_range(attn_range=[35, 40]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 14]),
                                k_range=AttnRange.from_range(attn_range=[35, 43]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[35, 43]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_range=AttnRange.from_range(attn_range=[0, 35]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 22]),
                                k_range=AttnRange.from_range(attn_range=[0, 40]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 42]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                        ]
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 9]),
                                k_range=AttnRange.from_range(attn_range=[35, 46]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 14]),
                                k_range=AttnRange.from_range(attn_range=[0, 40]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 42]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[19, 64]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 20]),
                                k_range=AttnRange.from_range(attn_range=[19, 65]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_range=AttnRange.from_range(attn_range=[46, 67]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ]
                    ],
                ],
            },
            {
                "name": "testcase_3",
                "chunk_size": 8,
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "degree": 3,
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 8], [8, 24], [24, 38], [38, 57], [57, 83], [83, 92], [92, 96]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [12, 20],
                        [27, 43],
                        [43, 48],
                        [48, 67],
                        [5, 74],
                        [31, 86],
                        [67, 96],
                    ]
                ),
                "attn_mask_type": [
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                ],
                "remote_k_ranges_global": [
                    [
                        AttnRanges.from_ranges([[8, 24]]),
                        AttnRanges.from_ranges([[24, 40], [48, 56]]),
                        AttnRanges.from_ranges([[56, 71]]),
                    ],
                    [
                        AttnRanges.from_ranges([[5, 8], [24, 37]]),
                        AttnRanges.from_ranges([[37, 61]]),
                        AttnRanges.from_ranges([[61, 80]]),
                    ],
                    [
                        AttnRanges.from_ranges([[5, 21]]),
                        AttnRanges.from_ranges([[21, 32], [40, 45]]),
                        AttnRanges.from_ranges([[45, 48], [56, 64]]),
                    ],
                    [
                        AttnRanges.from_ranges([[5, 24], [32, 37]]),
                        AttnRanges.from_ranges([[37, 56], [64, 69]]),
                        AttnRanges.from_ranges([[69, 88]]),
                    ],
                ],
                "attn_calc_remote_slice_local_list": [
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 4]),
                                k_range=AttnRange.from_range(attn_range=[4, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[4, 8]),
                                k_range=AttnRange.from_range(attn_range=[4, 12]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 14]),
                                k_range=AttnRange.from_range(attn_range=[16, 24]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[16, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 2]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_range=AttnRange.from_range(attn_range=[0, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 15]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                        ],
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 2]),
                                k_range=AttnRange.from_range(attn_range=[6, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[2, 8]),
                                k_range=AttnRange.from_range(attn_range=[6, 14]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 10]),
                                k_range=AttnRange.from_range(attn_range=[6, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[10, 16]),
                                k_range=AttnRange.from_range(attn_range=[6, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_range=AttnRange.from_range(attn_range=[10, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[10, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 6]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 13]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 22]),
                                k_range=AttnRange.from_range(attn_range=[0, 19]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 19]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            )
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[1, 3]),
                                k_range=AttnRange.from_range(attn_range=[14, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[3, 6]),
                                k_range=AttnRange.from_range(attn_range=[14, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[3, 6]),
                                k_range=AttnRange.from_range(attn_range=[0, 3]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 11]),
                                k_range=AttnRange.from_range(attn_range=[3, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 14]),
                                k_range=AttnRange.from_range(attn_range=[3, 11]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[3, 11]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_range=AttnRange.from_range(attn_range=[0, 3]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 22]),
                                k_range=AttnRange.from_range(attn_range=[0, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 10]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                        ],
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 20]),
                                k_range=AttnRange.from_range(attn_range=[19, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 9]),
                                k_range=AttnRange.from_range(attn_range=[11, 22]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 14]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 18]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 20]),
                                k_range=AttnRange.from_range(attn_range=[0, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_range=AttnRange.from_range(attn_range=[22, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 20]),
                                k_range=AttnRange.from_range(attn_range=[0, 17]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 19]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                    ],
                ],
            },
            {
                "name": "testcase_4",
                "chunk_size": 8,
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "degree": 3,
                "random_costs": True,
                "random_seed": 42,
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 8], [8, 24], [24, 38], [38, 57], [57, 83], [83, 92], [92, 96]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [12, 20],
                        [27, 43],
                        [43, 48],
                        [48, 67],
                        [5, 74],
                        [31, 86],
                        [67, 96],
                    ]
                ),
                "attn_mask_type": [
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.CAUSAL,
                ],
                "remote_k_ranges_global": [
                    [
                        AttnRanges.from_ranges([[16, 24], [32, 40]]),
                        AttnRanges.from_ranges([[24, 32], [48, 56], [64, 71]]),
                        AttnRanges.from_ranges([[8, 16], [56, 64]]),
                    ],
                    [
                        AttnRanges.from_ranges([[45, 61]]),
                        AttnRanges.from_ranges([[37, 45], [69, 80]]),
                        AttnRanges.from_ranges([[5, 8], [24, 37], [61, 69]]),
                    ],
                    [
                        AttnRanges.from_ranges([[13, 21], [29, 32], [40, 45]]),
                        AttnRanges.from_ranges([[21, 29], [45, 48], [56, 61]]),
                        AttnRanges.from_ranges([[5, 13], [61, 64]]),
                    ],
                    [
                        AttnRanges.from_ranges([[37, 45], [69, 85]]),
                        AttnRanges.from_ranges(
                            [[21, 24], [32, 37], [45, 53], [85, 88]]
                        ),
                        AttnRanges.from_ranges([[5, 21], [53, 56], [64, 69]]),
                    ],
                ],
                "attn_calc_remote_slice_local_list": [
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[4, 8]),
                                k_range=AttnRange.from_range(attn_range=[0, 4]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 14]),
                                k_range=AttnRange.from_range(attn_range=[8, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[8, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 23]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 4]),
                                k_range=AttnRange.from_range(attn_range=[4, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[4, 8]),
                                k_range=AttnRange.from_range(attn_range=[4, 8]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[8, 10]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[10, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 6]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 13]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 22]),
                                k_range=AttnRange.from_range(attn_range=[0, 19]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 19]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[0, 2]),
                                k_range=AttnRange.from_range(attn_range=[6, 8]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[2, 8]),
                                k_range=AttnRange.from_range(attn_range=[6, 14]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 10]),
                                k_range=AttnRange.from_range(attn_range=[6, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[10, 16]),
                                k_range=AttnRange.from_range(attn_range=[6, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 24]),
                                k_range=AttnRange.from_range(attn_range=[10, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[1, 3]),
                                k_range=AttnRange.from_range(attn_range=[14, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[3, 6]),
                                k_range=AttnRange.from_range(attn_range=[14, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[3, 6]),
                                k_range=AttnRange.from_range(attn_range=[8, 11]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 11]),
                                k_range=AttnRange.from_range(attn_range=[11, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 16]),
                                k_range=AttnRange.from_range(attn_range=[11, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 17]),
                                k_range=AttnRange.from_range(attn_range=[0, 11]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[17, 22]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[11, 14]),
                                k_range=AttnRange.from_range(attn_range=[8, 11]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[8, 11]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 22]),
                                k_range=AttnRange.from_range(attn_range=[0, 8]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[22, 24]),
                                k_range=AttnRange.from_range(attn_range=[0, 10]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                        ],
                    ],
                    [
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 8]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[0, 24]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 20]),
                                k_range=AttnRange.from_range(attn_range=[0, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_range=AttnRange.from_range(attn_range=[8, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 9]),
                                k_range=AttnRange.from_range(attn_range=[11, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 14]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 19]),
                                k_range=AttnRange.from_range(attn_range=[3, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[19, 20]),
                                k_range=AttnRange.from_range(attn_range=[3, 17]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_range=AttnRange.from_range(attn_range=[16, 19]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                        [
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[8, 9]),
                                k_range=AttnRange.from_range(attn_range=[16, 22]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[9, 14]),
                                k_range=AttnRange.from_range(attn_range=[0, 16]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[14, 16]),
                                k_range=AttnRange.from_range(attn_range=[0, 18]),
                                mask_type=AttnMaskType.CAUSAL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[16, 20]),
                                k_range=AttnRange.from_range(attn_range=[16, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                            AttnSlice(
                                q_range=AttnRange.from_range(attn_range=[20, 24]),
                                k_range=AttnRange.from_range(attn_range=[22, 24]),
                                mask_type=AttnMaskType.FULL,
                            ),
                        ],
                    ],
                ],
            },
        ],
    )
    def test_init_remote_rank_entry_per_stage_this_rank(self, testcase):
        # --------------      setup       -------------- #

        rank = self.rank
        cp_size = self.world_size
        manual_seed = self.seed
        testcase_name = testcase["name"]
        torch.manual_seed(manual_seed)

        # --------------      init sample meta      -------------- #

        # TODO: limited to self-attn settings for now
        is_same_source = True
        is_q_permutable = True
        is_k_permutable = True

        # TODO: test top-p minhp dispatch alg
        dispatch_config = DispatchConfig(alg=MinHeapDispatchAlg())

        # ------------  init SimpleNamespace class ------------ #

        test_solver_class = DistAttnSolver.__new__(DistAttnSolver)
        test_solver_class.overlap_config = OverlapConfig(
            min_chunk_size=8, max_num_chunks=16, degree=2
        )
        test_solver_class.cp_rank = rank
        test_solver_class.high_bandwith_domain_size = self.high_bandwith_domain_size

        # --------------      compute meta       -------------- #

        q_ranges: AttnRanges = testcase.get("q_ranges")
        k_ranges: AttnRanges = testcase.get("k_ranges")
        attn_mask_type: list[AttnMaskType] = testcase.get("attn_mask_type")
        chunk_size: int = testcase.get("chunk_size")
        min_chunk_size: int = testcase.get("min_chunk_size")
        max_num_chunks: int = testcase.get("max_num_chunks")

        degree: int = testcase.get("degree", None)
        random_costs = testcase.get("random_costs", False)
        random_seed = testcase.get("random_seed", None)

        test_solver_class.overlap_config = OverlapConfig(
            min_chunk_size=min_chunk_size,
            max_num_chunks=max_num_chunks,
            degree=degree,
        )
        test_solver_class.overlap_solver = OverlapSolver(
            UniformOverlapAlg(random_costs=random_costs, random_seed=random_seed)
        )

        expected_remote_k_ranges_global = testcase.get("remote_k_ranges_global")[rank]
        expected_attn_calc_remote_slice_local_list = testcase.get(
            "attn_calc_remote_slice_local_list"
        )[rank]

        # --------------      compute meta       -------------- #

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
            high_bandwith_domain_size=self.high_bandwith_domain_size,
        )

        # ----------- compute host and remote ranges ------------ #

        bucket_this_rank = buckets_per_rank[rank]
        (
            host_q_ranges_global_this_rank,
            host_k_ranges_global_this_rank,
            remote_k_ranges_global_this_rank,
            remote_k_ranges_global_hb_domain,
            remote_k_ranges_global_lb_domain,
        ) = test_solver_class._init_host_remote_ranges_global_this_rank(
            dispatch_meta_q=meta_q,
            dispatch_meta_k=meta_k,
            bucket_this_rank=bucket_this_rank,
        )

        # -------------- compute host rank entry -------------- #

        host_rank_entry_this_rank: HostRankEntry = (
            test_solver_class._init_host_rank_entry_this_rank(
                host_q_ranges_global=host_q_ranges_global_this_rank,
                host_k_ranges_global=host_k_ranges_global_this_rank,
                remote_k_ranges_global_hb_domain=remote_k_ranges_global_hb_domain,
                remote_k_ranges_global_lb_domain=remote_k_ranges_global_lb_domain,
                attn_calc_slice_global_list=bucket_this_rank.attn_slices,
            )
        )

        # ---------- compute remote rank entry this rank ---------- #

        remote_rank_entry_per_stage_this_rank: list[
            RemoteRankEntry
        ] = test_solver_class._init_remote_rank_entry_per_stage_this_rank(
            host_rank_entry_this_rank
        )

        for index, remote_rank_entry in enumerate(
            remote_rank_entry_per_stage_this_rank
        ):
            assert (
                remote_rank_entry.remote_k_ranges_global
                == expected_remote_k_ranges_global[index]
            ), (
                f"Get remote_k_ranges_global={remote_rank_entry.remote_k_ranges_global}, "
                f"when expected resule is {expected_remote_k_ranges_global[index]}, "
                f"in {index}th remote rank entry in rank {rank} in {testcase_name}"
            )
            assert (
                remote_rank_entry.attn_calc_remote_slice_local_list
                == expected_attn_calc_remote_slice_local_list[index]
            ), (
                f"Get remote_slice_local_list={remote_rank_entry.attn_calc_remote_slice_local_list}, "
                f"when expected result is {expected_attn_calc_remote_slice_local_list}, "
                f"in {index}th remote rank entry in rank {rank} in {testcase_name}"
            )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    @parameterize(
        "testcase",
        [
            {
                "name": "testcase_1",
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 10], [10, 16], [16, 30], [30, 43], [43, 61], [61, 64]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [[0, 11], [5, 18], [18, 32], [32, 45], [45, 64], [53, 64]]
                ),
            },
            {
                "name": "testcase_2",
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 8], [8, 24], [24, 38], [38, 57], [57, 83], [83, 92], [92, 96]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [12, 20],
                        [27, 43],
                        [43, 48],
                        [48, 67],
                        [5, 74],
                        [31, 86],
                        [67, 96],
                    ]
                ),
            },
            {
                "name": "testcase_3",
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 8], [8, 24], [24, 38], [38, 57], [57, 83], [83, 92], [92, 96]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [12, 20],
                        [27, 43],
                        [43, 48],
                        [48, 67],
                        [5, 74],
                        [31, 86],
                        [67, 96],
                    ]
                ),
            },
            {
                "name": "testcase_4",
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "q_ranges": AttnRanges.from_ranges(
                    [[0, 8], [8, 24], [24, 38], [38, 57], [57, 83], [83, 92], [92, 96]]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [12, 20],
                        [27, 43],
                        [43, 48],
                        [48, 67],
                        [5, 74],
                        [31, 86],
                        [67, 96],
                    ]
                ),
            },
            {
                "name": "testcase_5_overlap",
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 16],
                        [0, 16],
                        [16, 32],
                        [16, 32],
                        [32, 64],
                        [32, 64],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 16],
                        [48, 64],
                        [0, 32],
                        [32, 64],
                        [0, 32],
                        [32, 64],
                    ]
                ),
            },
            {
                "name": "testcase_6_overlap",
                "min_chunk_size": 8,
                "max_num_chunks": 16,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 16],
                        [16, 32],
                        [32, 48],
                        [48, 64],
                        [64, 80],
                        [80, 96],
                        [0, 30],
                        [0, 20],
                        [40, 75],
                        [80, 90],
                        [90, 96],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 16],
                        [8, 24],
                        [16, 32],
                        [24, 40],
                        [32, 48],
                        [40, 56],
                        [56, 96],
                        [24, 40],
                        [60, 90],
                        [56, 70],
                        [70, 96],
                    ]
                ),
            },
        ],
    )
    @parameterize(
        "attn_mask_type",
        [
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.INVCAUSAL,
            AttnMaskType.BICAUSAL,
        ],
    )
    @parameterize(
        "degree",
        [1, 2, 3],
    )
    @parameterize(
        "chunk_size",
        [4, 8],
    )
    @parameterize(
        "random_param",
        [
            (False, None),
            (True, 42),
        ],
    )
    def test_init_remote_rank_entry_this_rank_with_numpy(
        self,
        testcase,
        attn_mask_type,
        degree,
        chunk_size,
        random_param,
    ):
        # --------------      setup       -------------- #

        rank = self.rank
        cp_size = self.world_size
        manual_seed = self.seed
        torch.manual_seed(manual_seed)

        # --------------      init sample meta      -------------- #

        # TODO: limited to self-attn settings for now
        is_same_source = True
        is_q_permutable = True
        is_k_permutable = True

        # TODO: test top-p minhp dispatch alg
        dispatch_config = DispatchConfig(alg=MinHeapDispatchAlg())

        # ------------  init SimpleNamespace class ------------ #

        test_solver_class = DistAttnSolver.__new__(DistAttnSolver)
        test_solver_class.overlap_config = OverlapConfig(
            min_chunk_size=8, max_num_chunks=16, degree=2
        )
        test_solver_class.cp_rank = rank
        test_solver_class.high_bandwith_domain_size = self.high_bandwith_domain_size

        # --------------      compute meta       -------------- #

        q_ranges: AttnRanges = testcase.get("q_ranges")
        k_ranges: AttnRanges = testcase.get("k_ranges")
        attn_mask_type: list[AttnMaskType] = [attn_mask_type] * len(q_ranges)
        min_chunk_size: int = testcase.get("min_chunk_size")
        max_num_chunks: int = testcase.get("max_num_chunks")
        random_costs, random_seed = random_param

        test_solver_class.overlap_config = OverlapConfig(
            min_chunk_size=min_chunk_size,
            max_num_chunks=max_num_chunks,
            degree=degree,
        )
        test_solver_class.overlap_solver = OverlapSolver(
            UniformOverlapAlg(random_costs=random_costs, random_seed=random_seed)
        )

        # --------------      compute meta       -------------- #

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
            high_bandwith_domain_size=self.high_bandwith_domain_size,
        )

        # ----------- compute host and remote ranges ------------ #

        bucket_this_rank = buckets_per_rank[rank]
        (
            host_q_ranges_global_this_rank,
            host_k_ranges_global_this_rank,
            remote_k_ranges_global_this_rank,
            remote_k_ranges_global_hb_domain,
            remote_k_ranges_global_lb_domain,
        ) = test_solver_class._init_host_remote_ranges_global_this_rank(
            dispatch_meta_q=meta_q,
            dispatch_meta_k=meta_k,
            bucket_this_rank=bucket_this_rank,
        )

        # -------------- compute host rank entry -------------- #

        host_rank_entry_this_rank: HostRankEntry = (
            test_solver_class._init_host_rank_entry_this_rank(
                host_q_ranges_global=host_q_ranges_global_this_rank,
                host_k_ranges_global=host_k_ranges_global_this_rank,
                remote_k_ranges_global_hb_domain=remote_k_ranges_global_hb_domain,
                remote_k_ranges_global_lb_domain=remote_k_ranges_global_lb_domain,
                attn_calc_slice_global_list=bucket_this_rank.attn_slices,
            )
        )

        # ---------- compute remote rank entry this rank ---------- #

        remote_rank_entry_per_stage_this_rank: list[
            RemoteRankEntry
        ] = test_solver_class._init_remote_rank_entry_per_stage_this_rank(
            host_rank_entry_this_rank
        )

        # ------------------ init numpy array ----------------------- #

        answer = np.zeros((q_ranges.end, q_ranges.end), dtype=np.int32)
        result = np.zeros((q_ranges.end // cp_size, q_ranges.end), dtype=np.int32)

        for q_range, k_range, masktype in zip(q_ranges, k_ranges, attn_mask_type):
            add_range_to_array(
                array=answer,
                q_range=q_range,
                k_range=k_range,
                masktype=masktype,
                check=True,
            )

        for attn_slice in host_rank_entry_this_rank.attn_calc_host_slice_local_list:
            k_ranges_local = make_range_global(
                global_ranges=host_rank_entry_this_rank.host_q_ranges_global,
                local_range=attn_slice.k_range,
            )
            for i, k_range_local in enumerate(k_ranges_local):
                add_range_to_array(
                    array=result,
                    q_range=attn_slice.q_range,
                    k_range=k_range_local,
                    masktype=determine_ith_range_masktype(
                        i=i,
                        length=len(k_ranges_local),
                        masktype=attn_slice.mask_type,
                    ),
                    check=True,
                )

        for remote_rank_entry in remote_rank_entry_per_stage_this_rank:
            for attn_slice in remote_rank_entry.attn_calc_remote_slice_local_list:
                k_ranges_local = make_range_global(
                    global_ranges=remote_rank_entry.remote_k_ranges_global,
                    local_range=attn_slice.k_range,
                )
                for i, k_range_local in enumerate(k_ranges_local):
                    add_range_to_array(
                        array=result,
                        q_range=attn_slice.q_range,
                        k_range=k_range_local,
                        masktype=determine_ith_range_masktype(
                            i=i,
                            length=len(k_ranges_local),
                            masktype=attn_slice.mask_type,
                        ),
                        check=True,
                    )

        partitions = meta_q.partitions[rank]
        mask = np.zeros(q_ranges.end, dtype=bool)
        for idx in partitions:
            mask[chunk_size * idx : chunk_size * (idx + 1)] = True

        answer_current_rank = answer[mask]
        assert np.array_equal(answer_current_rank, result), (
            f"There's wrong with {rank=}, "
            f"when {q_ranges=}, {k_ranges=}, {attn_mask_type=}"
        )


if __name__ == "__main__":
    run_tests()
