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

"""Output dtypes other than the input dtype (SM100/SM110): the forward O and
the backward dQ / dK / dV."""

import unittest
from dataclasses import dataclass

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import MT_MAP, flex_flash_attn_func
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
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
    is_sm100_family,
    make_qkv,
    ranges_from_lengths,
    seed_test_case,
    to_ranges_tensor,
)

DTYPE = torch.bfloat16
HEAD_CONFIGS = [MHA_HD128, GQA_HD128]

# Length unit of the range geometries.
SPAN = 128


@dataclass(frozen=True)
class StorePath:
    name: str
    q_ranges: Ranges
    k_ranges: Ranges
    disable_fwd_atomic_reduction: bool
    low_prec_out_dtype: torch.dtype | None


STORE_PATHS = [
    # Two relations share q rows: None already merges in fp32 (covered in
    # test_ranges), so pass DTYPE explicitly to exercise the bf16 atomic merge.
    StorePath(
        "atomic_merge",
        q_ranges=[[0, 4 * SPAN], [2 * SPAN, 6 * SPAN]],
        k_ranges=[[0, 4 * SPAN], [4 * SPAN, 6 * SPAN]],
        disable_fwd_atomic_reduction=False,
        low_prec_out_dtype=DTYPE,
    ),
    # Sorted disjoint q ranges: O is converted once from the fp32 registers.
    StorePath(
        "direct_store",
        q_ranges=ranges_from_lengths([2 * SPAN] * 3),
        k_ranges=ranges_from_lengths([2 * SPAN] * 3),
        disable_fwd_atomic_reduction=True,
        low_prec_out_dtype=None,
    ),
]

GRAD_RANGES = ranges_from_lengths([2 * SPAN] * 4)


def _total(ranges: Ranges) -> int:
    return max(end for _, end in ranges)


def _max_len(ranges: Ranges) -> int:
    return max(end - start for start, end in ranges)


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestOutDtype(CuteDSLDistTestBase):
    @with_run_in_mp
    @parameterize("path", STORE_PATHS)
    @parameterize("fp32_out", [False, True])
    @parameterize("heads", HEAD_CONFIGS)
    def test_fwd_out_dtype(self, path: StorePath, fp32_out: bool, heads: HeadConfig):
        """Input-dtype vs fp32 forward O across atomic-merge and direct-store paths."""
        out_dtype = torch.float32 if fp32_out else path.low_prec_out_dtype
        test_case = f"[test_fwd_out_dtype][{path.name}][{out_dtype=}][{heads.name}]"
        seed_test_case(test_case)
        q, k, v = make_qkv(
            (_total(path.q_ranges),),
            (_total(path.k_ranges),),
            heads,
            DTYPE,
            self.device,
        )

        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=to_ranges_tensor(path.q_ranges, self.device),
            k_ranges=to_ranges_tensor(path.k_ranges, self.device),
            mask_types=MT_MAP.full,
            max_seqlen_q=_max_len(path.q_ranges),
            max_seqlen_k=_max_len(path.k_ranges),
            disable_fwd_atomic_reduction=path.disable_fwd_atomic_reduction,
            out_dtype=out_dtype,
        )
        self.assertEqual(out.dtype, DTYPE if out_dtype is None else out_dtype)
        do = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)

        assert_close_to_torch_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=path.q_ranges,
            k_ranges=path.k_ranges,
            attn_type_map=[MT_MAP.full] * len(path.q_ranges),
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("grad_dtype", [torch.float32, DTYPE])
    @parameterize("heads", HEAD_CONFIGS)
    def test_bwd_grad_dtype(self, grad_dtype: torch.dtype, heads: HeadConfig):
        """dQ / dK / dV come back in the requested dtype."""
        test_case = f"[test_bwd_grad_dtype][{grad_dtype=}][{heads.name}]"
        seed_test_case(test_case)
        total = _total(GRAD_RANGES)
        q, k, v = (
            t.detach() for t in make_qkv((total,), (total,), heads, DTYPE, self.device)
        )
        q_ranges = to_ranges_tensor(GRAD_RANGES, self.device)
        k_ranges = to_ranges_tensor(GRAD_RANGES, self.device)
        max_len = _max_len(GRAD_RANGES)

        # The backward reads O in the input dtype.
        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=MT_MAP.full,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            out_dtype=DTYPE,
        )
        do = torch.randn_like(out)
        dq, dk, dv, _ = _flex_flash_attn_bwd(
            q,
            k,
            v,
            out,
            lse,
            do,
            dq_type=grad_dtype,
            dk_type=grad_dtype,
            dv_type=grad_dtype,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=MT_MAP.full,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
        )
        for name, grad in (("dq", dq), ("dk", dk), ("dv", dv)):
            self.assertEqual(grad.dtype, grad_dtype, f"{test_case} {name}")

        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=GRAD_RANGES,
            k_ranges=GRAD_RANGES,
            attn_type_map=[MT_MAP.full] * len(GRAD_RANGES),
            test_case=test_case,
        )


if __name__ == "__main__":
    run_tests()
