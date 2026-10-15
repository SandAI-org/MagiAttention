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

"""q/k-range features of the SM100/SM110 kernels: per-range mask types,
overlapping q ranges (atomic O merge), the unique-writer contracts, and O/LSE
accumulation over several calls."""

import math
import unittest
from dataclasses import dataclass

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import MT_MAP, flex_flash_attn_func
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import with_run_in_mp
from tests.test_kernel.cutedsl.cutedsl_test_utils import (
    GQA_HD128,
    MHA_HD128,
    SM100_FAMILY_REQUIRED,
    AttnTensors,
    CuteDSLDistTestBase,
    HeadConfig,
    Ranges,
    assert_close_to_torch_ref,
    cycle_mask_types,
    is_sm100_family,
    make_qkv,
    mask_types_arg,
    ranges_from_lengths,
    seed_test_case,
    to_ranges_tensor,
    torch_reference_out_lse,
)

HEAD_CONFIGS = [MHA_HD128, GQA_HD128]
DTYPE = torch.bfloat16

# Relations of the batched cases. Each q range is shorter than its k range, so
# bi_causal stays a band: on the diagonal P == 1 and the reference dq vanishes.
NUM_RELATIONS = 8
Q_RANGE_LEN = 256
K_RANGE_LEN = 512

MASK_PATTERNS = {
    "full": [MT_MAP.full],
    "causal": [MT_MAP.causal],
    "inv_causal": [MT_MAP.inv_causal],
    "bi_causal": [MT_MAP.bi_causal],
    "mixed": [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal],
}

# Length unit of the overlap and accumulation geometries.
SPAN = 128

# Accumulation tensors: fp32 O buffer, the atomic-merge precision.
ACCUM_ATOL = ACCUM_RTOL = 1e-2
LSE_ATOL = LSE_RTOL = 1e-3


@dataclass(frozen=True)
class UniqueWriterCase:
    name: str
    heads: HeadConfig
    disable_fwd_atomic_reduction: bool
    disable_bwd_dkv_atomic_reduction: bool


UNIQUE_WRITER_CASES = [
    UniqueWriterCase("direct_o", MHA_HD128, True, False),
    UniqueWriterCase("direct_dkv", MHA_HD128, False, True),
    UniqueWriterCase("direct_o_and_dkv", MHA_HD128, True, True),
    # Direct dK/dV store is MHA-only; GQA keeps the dK/dV reduction.
    UniqueWriterCase("direct_o_gqa", GQA_HD128, True, False),
]


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRanges(CuteDSLDistTestBase):
    def _run_and_check(
        self,
        *,
        q_ranges: Ranges,
        k_ranges: Ranges,
        attn_type_map: list[int],
        heads: HeadConfig,
        test_case: str,
        disable_fwd_atomic_reduction: bool = False,
        disable_bwd_dkv_atomic_reduction: bool = False,
    ) -> None:
        seed_test_case(test_case)
        q, k, v = make_qkv(
            (max(end for _, end in q_ranges),),
            (max(end for _, end in k_ranges),),
            heads,
            DTYPE,
            self.device,
        )
        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=to_ranges_tensor(q_ranges, self.device),
            k_ranges=to_ranges_tensor(k_ranges, self.device),
            mask_types=mask_types_arg(attn_type_map, self.device),
            max_seqlen_q=max(end - start for start, end in q_ranges),
            max_seqlen_k=max(end - start for start, end in k_ranges),
            disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
            disable_bwd_dkv_atomic_reduction=disable_bwd_dkv_atomic_reduction,
        )
        do = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
        assert_close_to_torch_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("pattern", ["bi_causal", "mixed"])
    @parameterize("heads", HEAD_CONFIGS)
    def test_per_range_mask_types(self, pattern: str, heads: HeadConfig):
        """Mask types beyond full/causal go through the per-range kernel."""
        self._run_and_check(
            q_ranges=ranges_from_lengths([Q_RANGE_LEN] * NUM_RELATIONS),
            k_ranges=ranges_from_lengths([K_RANGE_LEN] * NUM_RELATIONS),
            attn_type_map=cycle_mask_types(MASK_PATTERNS[pattern], NUM_RELATIONS),
            heads=heads,
            test_case=f"[test_per_range_mask_types][{pattern}][{heads.name}]",
        )

    @with_run_in_mp
    @parameterize("mask_type", [MT_MAP.full, MT_MAP.causal])
    @parameterize("heads", HEAD_CONFIGS)
    def test_overlapping_q_ranges(self, mask_type: int, heads: HeadConfig):
        """Two relations share q rows; the atomic O merge combines them.

        Their k ranges are disjoint, so every (q, k) pair is attended once.
        """
        self._run_and_check(
            q_ranges=[[0, 4 * SPAN], [2 * SPAN, 6 * SPAN]],
            k_ranges=[[0, 4 * SPAN], [4 * SPAN, 6 * SPAN]],
            attn_type_map=[mask_type] * 2,
            heads=heads,
            test_case=f"[test_overlapping_q_ranges][{mask_type=}][{heads.name}]",
        )

    @with_run_in_mp
    @parameterize("case", UNIQUE_WRITER_CASES)
    @parameterize("pattern", ["full", "causal", "mixed"])
    def test_unique_writer_contracts(self, case: UniqueWriterCase, pattern: str):
        """Sorted disjoint ranges let O / dK,dV skip the atomic reduction."""
        self._run_and_check(
            q_ranges=ranges_from_lengths([Q_RANGE_LEN] * NUM_RELATIONS),
            k_ranges=ranges_from_lengths([K_RANGE_LEN] * NUM_RELATIONS),
            attn_type_map=cycle_mask_types(MASK_PATTERNS[pattern], NUM_RELATIONS),
            heads=case.heads,
            disable_fwd_atomic_reduction=case.disable_fwd_atomic_reduction,
            disable_bwd_dkv_atomic_reduction=case.disable_bwd_dkv_atomic_reduction,
            test_case=f"[test_unique_writer_contracts][{case.name}][{pattern}]",
        )


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRangeAccumulation(unittest.TestCase):
    """Atomic-merge forward calls accumulating into caller-owned O/LSE."""

    def setUp(self) -> None:
        self.device = torch.cuda.current_device()
        self.heads = MHA_HD128
        seed_test_case(self.id())

    def _accumulate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        q_ranges: Ranges,
        k_ranges: Ranges,
    ) -> None:
        _flex_flash_attn_fwd(
            q,
            k,
            v,
            out=out,
            lse=lse,
            q_ranges=to_ranges_tensor(q_ranges, self.device),
            k_ranges=to_ranges_tensor(k_ranges, self.device),
            mask_types=MT_MAP.full,
            max_seqlen_q=max(end - start for start, end in q_ranges),
            max_seqlen_k=max(end - start for start, end in k_ranges),
        )

    def _empty_state(self, total_q: int) -> tuple[torch.Tensor, torch.Tensor]:
        """O needs no initialization where LSE is -inf."""
        heads = self.heads
        out = torch.full(
            (total_q, heads.num_heads_q, heads.head_dim),
            math.nan,
            device=self.device,
            dtype=torch.float32,
        )
        lse = torch.full((total_q, heads.num_heads_q), -math.inf, device=self.device)
        return out, lse

    def test_calls_accumulate_to_their_union(self):
        """Three calls equal one call over the union of their relations.

        k ranges are disjoint across calls; the last q span stays uncovered
        and ends with O = 0, LSE = -inf.
        """
        calls = [
            ([[0, 2 * SPAN]], [[0, 2 * SPAN]]),
            ([[SPAN, 4 * SPAN]], [[2 * SPAN, 3 * SPAN]]),
            ([[3 * SPAN, 5 * SPAN]], [[3 * SPAN, 4 * SPAN]]),
        ]
        total_q, total_k = 6 * SPAN, 4 * SPAN
        q, k, v = make_qkv((total_q,), (total_k,), self.heads, DTYPE, self.device)
        out, lse = self._empty_state(total_q)

        for q_ranges, k_ranges in calls:
            self._accumulate(q, k, v, out, lse, q_ranges, k_ranges)

        out_ref, lse_ref = torch_reference_out_lse(
            q=q,
            k=k,
            v=v,
            q_ranges=[r for q_ranges, _ in calls for r in q_ranges],
            k_ranges=[r for _, k_ranges in calls for r in k_ranges],
        )
        torch.testing.assert_close(out, out_ref, atol=ACCUM_ATOL, rtol=ACCUM_RTOL)
        torch.testing.assert_close(lse, lse_ref, atol=LSE_ATOL, rtol=LSE_RTOL)

    def test_empty_k_call_keeps_the_accumulated_state(self):
        """A call without keys leaves accumulated rows and zeroes empty rows."""
        total_q = 4 * SPAN
        accumulated = slice(0, 2 * SPAN)
        empty = slice(2 * SPAN, total_q)
        q, k, v = make_qkv((total_q,), (0,), self.heads, DTYPE, self.device)
        out, lse = self._empty_state(total_q)
        out[accumulated] = torch.randn_like(out[accumulated])
        lse[accumulated] = torch.randn_like(lse[accumulated])
        out_prev, lse_prev = out.clone(), lse.clone()

        self._accumulate(q, k, v, out, lse, [[0, total_q]], [[0, 0]])

        self.assertTrue(torch.all(out[empty] == 0), "empty-row O is not zero")
        torch.testing.assert_close(
            out[accumulated], out_prev[accumulated], atol=0, rtol=0
        )
        torch.testing.assert_close(lse, lse_prev, atol=0, rtol=0)


if __name__ == "__main__":
    run_tests()
