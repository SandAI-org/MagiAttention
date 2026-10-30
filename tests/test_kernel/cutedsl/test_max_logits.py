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

import math
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from magi_attention.kernel.cutedsl import MT_MAP, flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import (
    MHA_HD64,
    MHA_HD128,
    SM100_FAMILY_REQUIRED,
    HeadConfig,
    is_sm100_family,
    make_qkv,
    ranges_from_lengths,
    seed_test_case,
    to_ranges_tensor,
)
from magi_attention.testing.precision import assert_close

DTYPE = torch.bfloat16


SCALE = 0.5

SOFTCAP = 16.0

Q_GROUP_LENGTHS = (257, 129)
K_GROUP_LENGTHS = (129, 257)

RELATION_GROUPS = [(0, 0), (0, 1), (1, 0)]

MAX_LOGITS_RTOL = 5e-4


def _reference_max_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    q_ranges: list[list[int]],
    k_ranges: list[list[int]],
    softcap: float,
) -> torch.Tensor:
    """Per-head max of the softmax input over every pair of full-mask relations."""
    out = torch.full((q.shape[1],), -math.inf, device=q.device, dtype=torch.float32)
    for (q_start, q_end), (k_start, k_end) in zip(q_ranges, k_ranges):
        rows = q[q_start:q_end].float()
        keys = k[k_start:k_end].float()
        scores = torch.einsum("qhd,khd->hqk", rows, keys) * SCALE
        if softcap > 0.0:
            scores = softcap * torch.tanh(scores / softcap)
        out = torch.maximum(out, scores.amax(dim=(1, 2)))
    return out


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestMaxLogits(TestCase):
    """``return_max_logits`` on the softcap and range-merge paths."""

    def setUp(self) -> None:
        self.device = torch.cuda.current_device()

    def _check(self, heads: HeadConfig, softcap: float, range_merge: bool) -> None:
        test_case = f"[test_max_logits][{heads.name}][{softcap=}][{range_merge=}]"
        seed_test_case(test_case)
        q_groups = ranges_from_lengths(Q_GROUP_LENGTHS)
        k_groups = ranges_from_lengths(K_GROUP_LENGTHS)
        q_ranges = [q_groups[q_group] for q_group, _ in RELATION_GROUPS]
        k_ranges = [k_groups[k_group] for _, k_group in RELATION_GROUPS]
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

        _, meta = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=to_ranges_tensor(q_ranges, self.device),
            k_ranges=to_ranges_tensor(k_ranges, self.device),
            mask_types=MT_MAP.full,
            max_seqlen_q=max(Q_GROUP_LENGTHS),
            max_seqlen_k=max(K_GROUP_LENGTHS),
            softmax_scale=SCALE,
            softcap=softcap,
            # The forward range merge requires a unique writer for O.
            disable_fwd_atomic_reduction=range_merge,
            range_merge=range_merge,
            return_max_logits=True,
        )
        assert meta.max_logits is not None
        self.assertEqual(meta.max_logits.dtype, torch.float32)
        self.assertEqual(tuple(meta.max_logits.shape), (heads.num_heads_q,))
        assert_close(
            meta.max_logits,
            _reference_max_logits(q, k, q_ranges, k_ranges, softcap),
            rtol=MAX_LOGITS_RTOL,
            test_case=f"{test_case} => max_logits",
        )

    def test_softcap(self):
        """The reported maximum is ``softcap * tanh(scale * q.k / softcap)``."""
        self._check(MHA_HD64, SOFTCAP, range_merge=False)

    @parameterize("heads", [MHA_HD64, MHA_HD128])
    def test_range_merge(self, heads: HeadConfig):
        """The merged softmax loop covers every relation of a q group."""
        self._check(heads, softcap=0.0, range_merge=True)


if __name__ == "__main__":
    run_tests()
