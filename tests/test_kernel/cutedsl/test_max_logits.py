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

# A non-unit scale that is not 1/sqrt(head_dim), so a missing or an extra
# factor of it changes the result.
SCALE = 0.5
SOFTCAP = 2.0

SEGMENTS = 8
SEG_LEN = 256

# The softcap path uses tanh.approx, so the reported maximum can differ from an
# exact tanh in the last digits of the approximation.
MAX_LOGITS_ATOL = 5e-3
MAX_LOGITS_RTOL = 5e-3


def _reference_max_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    lengths: list[int],
    *,
    softcap: float,
) -> torch.Tensor:
    """Per-head max of the softmax input over every attended (q, k) pair.

    Full-mask relations only: the mask does not change the unit of the tracked
    maximum, which is what this test pins.
    """
    heads = q.shape[1]
    out = torch.full((heads,), -math.inf, device=q.device, dtype=torch.float32)
    start = 0
    for length in lengths:
        rows = q[start : start + length].float()
        keys = k[start : start + length].float()
        scores = torch.einsum("qhd,khd->hqk", rows, keys) * SCALE
        if softcap > 0.0:
            scores = softcap * torch.tanh(scores / softcap)
        out = torch.maximum(out, scores.amax(dim=(1, 2)))
        start += length
    return out


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestMaxLogits(TestCase):
    """``return_max_logits`` reports the maximum in natural-log units.

    A score modifier receives the scaled score, so the kernel must not scale
    the tracked maximum a second time.
    """

    def setUp(self) -> None:
        self.device = torch.cuda.current_device()

    def _check(self, heads: HeadConfig, softcap: float) -> None:
        lengths = [SEG_LEN] * SEGMENTS
        q_ranges = ranges_from_lengths(lengths)
        test_case = f"[test_max_logits][{heads.name}][{softcap=}]"
        seed_test_case(test_case)
        q, k, v = make_qkv(
            (SEGMENTS * SEG_LEN,), (SEGMENTS * SEG_LEN,), heads, DTYPE, self.device
        )

        _, meta = flex_flash_attn_func(
            q.detach(),
            k.detach(),
            v.detach(),
            q_ranges=to_ranges_tensor(q_ranges, self.device),
            k_ranges=to_ranges_tensor(q_ranges, self.device),
            mask_types=MT_MAP.full,
            max_seqlen_q=SEG_LEN,
            max_seqlen_k=SEG_LEN,
            softmax_scale=SCALE,
            softcap=softcap,
            return_max_logits=True,
        )
        assert meta.max_logits is not None
        self.assertEqual(meta.max_logits.dtype, torch.float32)
        self.assertEqual(tuple(meta.max_logits.shape), (heads.num_heads_q,))
        assert_close(
            meta.max_logits,
            _reference_max_logits(
                q=q.detach(), k=k.detach(), lengths=lengths, softcap=softcap
            ),
            atol=MAX_LOGITS_ATOL,
            rtol=MAX_LOGITS_RTOL,
            test_case=f"{test_case} => max_logits",
        )

    @parameterize("softcap", [0.0, SOFTCAP])
    def test_scaled_max_logits(self, softcap: float):
        """The raw maximum is scaled once; a soft-capped one is already scaled."""
        self._check(MHA_HD64, softcap)


if __name__ == "__main__":
    run_tests()
