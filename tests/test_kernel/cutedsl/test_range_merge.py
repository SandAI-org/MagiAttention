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

import unittest

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import MT_MAP, RangeMergePlan, plan_range_merge
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.range_merge import bwd_range_merge_arg
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import (
    MHA_HD64,
    MHA_HD128,
    SM100_FAMILY_REQUIRED,
    AttnTensors,
    CuteDSLDistTestBase,
    HeadConfig,
    assert_close_to_torch_ref,
    is_sm100_family,
    make_qkv,
    mask_types_arg,
    ranges_from_lengths,
    seed_test_case,
    to_ranges_tensor,
)
from magi_attention.testing.dist_common import with_run_in_mp

# Unique-writer dK/dV, which the backward merge requires, is MHA-only.
DTYPE = torch.bfloat16

# (heads, mask pattern): one on-the-fly plan and one precomputed plan, at both
# supported head dims.
RANGE_MERGE_CASES = [
    (MHA_HD64, "full"),
    (MHA_HD128, "mixed"),
]

# Lengths off the tile sizes leave partial tiles inside every group.
Q_GROUP_LENGTHS = (257, 129, 193)
K_GROUP_LENGTHS = (129, 257, 193)
# (q group, k group) of each relation. Each q group pairs with several k
# groups (merged by the forward) and each k group with several q groups
# (merged by the backward). Distinct groups are disjoint, so the merged
# groups give O and dK/dV one writer per row.
RELATION_GROUPS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]

MASK_PATTERNS = {
    "full": [MT_MAP.full] * len(RELATION_GROUPS),
    # bi_causal / causal pairs leave K tiles without any Q block.
    "mixed": [
        MT_MAP.bi_causal,
        MT_MAP.causal,
        MT_MAP.full,
        MT_MAP.full,
        MT_MAP.inv_causal,
        MT_MAP.full,
    ],
}


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRangeMerge(CuteDSLDistTestBase):
    @with_run_in_mp
    @parameterize("case", RANGE_MERGE_CASES)
    def test_range_merge_fwd_bwd(self, case: tuple[HeadConfig, str]):
        """``range_merge=True`` plans on the fly; a precomputed
        :class:`RangeMergePlan` carries both the Q- and the K-merge.

        The forward and backward are called separately: O has one writer per
        row after the Q-merge, but the relations repeat q ranges, so dQ keeps
        its reducing path (``disable_fwd_atomic_reduction`` only in the
        forward).
        """
        heads, pattern = case
        test_case = f"[test_range_merge_fwd_bwd][{pattern}][{heads.name}]"
        seed_test_case(test_case)
        q_groups = ranges_from_lengths(Q_GROUP_LENGTHS)
        k_groups = ranges_from_lengths(K_GROUP_LENGTHS)
        q_ranges = [q_groups[q_group] for q_group, _ in RELATION_GROUPS]
        k_ranges = [k_groups[k_group] for _, k_group in RELATION_GROUPS]
        attn_type_map = MASK_PATTERNS[pattern]
        q, k, v = (
            t.detach()
            for t in make_qkv(
                (sum(Q_GROUP_LENGTHS),),
                (sum(K_GROUP_LENGTHS),),
                heads,
                DTYPE,
                self.device,
            )
        )
        q_ranges_t = to_ranges_tensor(q_ranges, self.device)
        k_ranges_t = to_ranges_tensor(k_ranges, self.device)
        mask_types = mask_types_arg(attn_type_map, self.device)
        range_merge: bool | RangeMergePlan = (
            True
            if pattern == "full"
            else plan_range_merge(q_ranges_t, k_ranges_t, mask_types)
        )

        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_types=mask_types,
            max_seqlen_q=max(Q_GROUP_LENGTHS),
            max_seqlen_k=max(K_GROUP_LENGTHS),
            disable_fwd_atomic_reduction=True,
            range_merge=range_merge,
        )
        do = torch.randn_like(out)
        dq, dk, dv, _ = _flex_flash_attn_bwd(
            q,
            k,
            v,
            out,
            lse,
            do,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_types=mask_types,
            max_seqlen_q=max(Q_GROUP_LENGTHS),
            max_seqlen_k=max(K_GROUP_LENGTHS),
            disable_bwd_dkv_atomic_reduction=True,
            range_merge=bwd_range_merge_arg(range_merge),
        )

        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )


if __name__ == "__main__":
    run_tests()
