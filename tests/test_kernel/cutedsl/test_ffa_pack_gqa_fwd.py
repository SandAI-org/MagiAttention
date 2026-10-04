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

"""GQA row packing on the SM100 CuTeDSL range atomic forward.

The range atomic path sizes q_stage by the Q rows the MMA actually sees:
``max_seqlen_q * G`` when packed, ``max_seqlen_q`` otherwise.
"""

from contextlib import contextmanager
from typing import Iterator
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.utils import make_attn_mask_from_ffa_args


@contextmanager
def _record_fwd_kernels() -> Iterator[list[FFAFwdSm100]]:
    """Record the SM100 fwd kernel objects built within the context.

    The specialization is only visible on the kernel object, which a JIT cache
    hit never builds, so the context also swaps in an empty compile cache.
    """
    built: list[FFAFwdSm100] = []
    init = FFAFwdSm100.__init__

    def record(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    with mock.patch.object(FFAFwdSm100, "__init__", record), mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ):
        yield built


class TestFfaPackGqaFwd(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] != 10:
            self.skipTest("the q_stage rule is SM100-specific")
        torch.manual_seed(42)

    @parameterize(
        "case",
        [
            # (atomic, len_q, out_dtype, q_stage, 2-CTA, fp32-O K/V borrowing)
            # 128 unpacked rows fit one stage; the G-scaled count (512) would
            # pick two stages with the second one empty.
            (True, 128, torch.float32, 1, False, True),
            (True, 129, torch.float32, 2, False, True),
            # The direct-store path keeps the G-scaled count, so 512 rows
            # still select two stages and the 2-CTA kernel.
            (False, 128, torch.bfloat16, 2, True, False),
        ],
    )
    def test_unpacked_gqa_q_stage_counts_unpacked_rows(self, case):
        """The unpacked GQA range kernel's q_stage and the configs derived
        from it; without sm_margin a range launch is persistent iff fp32 O
        borrows K/V smem."""
        atomic, len_q, out_dtype, q_stage, two_cta, borrow_kv = case
        device, dtype, head_dim = "cuda", torch.bfloat16, 128
        num_head, num_head_kv, num_rel, len_k = 16, 4, 8, 256
        q_ranges = [[i * len_q, (i + 1) * len_q] for i in range(num_rel)]
        k_ranges = [[i * len_k, (i + 1) * len_k] for i in range(num_rel)]
        q = torch.randn(num_rel * len_q, num_head, head_dim, device=device, dtype=dtype)
        k, v = (
            torch.randn(
                num_rel * len_k, num_head_kv, head_dim, device=device, dtype=dtype
            )
            for _ in range(2)
        )

        with _record_fwd_kernels() as built:
            out, _ = _flex_flash_attn_fwd(
                q,
                k,
                v,
                q_ranges=torch.tensor(q_ranges, dtype=torch.int32, device=device),
                k_ranges=torch.tensor(k_ranges, dtype=torch.int32, device=device),
                mask_types=MT_MAP.full,
                max_seqlen_q=len_q,
                max_seqlen_k=len_k,
                pack_gqa=False,
                disable_fwd_atomic_reduction=not atomic,
                out_dtype=out_dtype,
            )
        self.assertEqual(len(built), 1)
        kernel = built[0]
        self.assertFalse(kernel.pack_gqa)
        self.assertEqual(kernel.disable_fwd_atomic_reduction, not atomic)
        self.assertEqual(kernel.q_stage, q_stage)
        self.assertEqual(kernel.use_2cta_instrs, two_cta)
        self.assertEqual(kernel.sO_borrow_kv, borrow_kv)
        self.assertEqual(kernel.is_persistent, borrow_kv)

        mask = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_ranges),
            k_ranges=AttnRanges.from_ranges(k_ranges),
            attn_type_map=[MT_MAP.full] * num_rel,
            total_seqlen_q=q.shape[0],
            total_seqlen_k=k.shape[0],
            device=device,
        )
        out_ref, _ = ref_attn_func(
            q=q, k=k, v=v, mask=mask, layout="thd", high_precision=True
        )
        torch.testing.assert_close(out.float(), out_ref.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
