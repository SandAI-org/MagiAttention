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


import random
import unittest
from typing import Any
from unittest import TestCase

import pytest
import torch

import magi_attention.testing
from magi_attention.common import AttnRanges
from magi_attention.functional import flex_flash_attn_func
from magi_attention.functional.dist_attn import result_correction
from magi_attention.functional.flex_flash_attn import (
    _flex_flash_attn_backward,
    _flex_flash_attn_forward,
    merge_ranges,
)
from magi_attention.testing import parameterize
from magi_attention.testing.precision import (
    EPSILON,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
    torch_attn_ref,
)
from magi_attention.utils import get_attn_mask_from_ffa_args, is_list_value_any


class TestFlexFlashAttn(TestCase):
    @property
    def seed(self):
        return 42

    @property
    def device(self):
        return torch.cuda.current_device()

    def setUp(self):
        torch.manual_seed(self.seed)

    def generate_non_overlapping_qk_pairs(
        self,
        total_seqlen_q: int,
        total_seqlen_k: int,
        num_pairs: int,
        min_len_q: int = 16,
        max_len_q: int = 128,
        min_len_k: int = 16,
        max_len_k: int = 128,
        max_consecutive_failures: int = 500,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Generates non-overlapping q_ranges and k_ranges on a potentially non-square attention area.

        The function attempts to generate a specified number of attention pairs, defined by `num_pairs`.
        It has two termination conditions:
        1. Success: When the number of generated pairs reaches `num_pairs`.
        2. Saturation: When it fails to find a new, non-overlapping (q,k) pair for a number of
        consecutive attempts (controlled by `max_consecutive_failures`), it terminates even if
        `num_pairs` has not been reached.

        Core Constraint (No Area Overlap):
        Imagine each (q_range, k_range) pair as a rectangle on a 2D plane. This function ensures
        that no two such generated rectangles have overlapping areas.

        Args:
            total_seqlen_q (int): The total sequence length of the Query vector.
            total_seqlen_k (int): The total sequence length of the Key vector.
            num_pairs (int): The target number of (q_range, k_range) pairs to generate.
            min_len_q (int): The minimum length for a q_range.
            max_len_q (int): The maximum length for a q_range.
            min_len_k (int): The minimum length for a k_range.
            max_len_k (int): The maximum length for a k_range.
            max_consecutive_failures (int):
                The upper limit for consecutive failed attempts, used to determine if the space is saturated.

        Returns:
            tuple[list[list[int]], list[list[int]]]:
            A tuple containing two lists: q_ranges and k_ranges.
        """
        q_ranges: list = []
        k_ranges: list = []

        consecutive_failures = 0

        def _create_one_random_q_range() -> list[int]:
            """Generates a random range [start, end] for the Query axis."""
            effective_max = min(max_len_q, total_seqlen_q)
            if min_len_q > effective_max:
                raise ValueError(
                    f"min_len_q ({min_len_q}) cannot be greater than its effective max ({effective_max})"
                )

            length = random.randint(min_len_q, effective_max)
            start = random.randint(0, total_seqlen_q - length)
            return [start, start + length]

        def _create_one_random_k_range() -> list[int]:
            """Generates a random range [start, end] for the Key axis."""
            effective_max = min(max_len_k, total_seqlen_k)
            if min_len_k > effective_max:
                raise ValueError(
                    f"min_len_k ({min_len_k}) cannot be greater than its effective max ({effective_max})"
                )

            length = random.randint(min_len_k, effective_max)
            start = random.randint(0, total_seqlen_k - length)
            return [start, start + length]

        # Main loop, continues until one of the termination conditions is met:
        # the target number of pairs is reached, or the consecutive failure threshold is exceeded.
        while (
            len(q_ranges) < num_pairs
            and consecutive_failures < max_consecutive_failures
        ):
            candidate_q_range = _create_one_random_q_range()
            candidate_k_range = _create_one_random_k_range()

            is_valid_placement = True

            for i in range(len(q_ranges)):
                q_axis_overlaps = (
                    candidate_q_range[0] < q_ranges[i][1]
                    and candidate_q_range[1] > q_ranges[i][0]
                )
                k_axis_overlaps = (
                    candidate_k_range[0] < k_ranges[i][1]
                    and candidate_k_range[1] > k_ranges[i][0]
                )

                if q_axis_overlaps and k_axis_overlaps:
                    is_valid_placement = False
                    break

            if is_valid_placement:
                q_ranges.append(candidate_q_range)
                k_ranges.append(candidate_k_range)
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            # Provide an informative printout at the end of the function.
            """
            if len(q_ranges) == num_pairs:
                print(
                    f"Successfully reached the target, generating {num_pairs} non-overlapping qk-ranges."
                )
            else:
                print(
                    f"Terminated after {max_consecutive_failures} consecutive failures. "
                    f"Target was {num_pairs} pairs, actually generated {len(q_ranges)} pairs."
                )
            """
        return q_ranges, k_ranges

    def test_compiled_flex_flash_attn(self):
        s, h, d = 2048, 6, 128
        hk = 3

        q = torch.randn(s, h, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(s, hk, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        do = torch.randn_like(q)

        [x.requires_grad_(True) for x in (q, k, v)]

        q_ranges = AttnRanges.from_ranges([[0, s // 2], [s // 2, s]])
        k_ranges = AttnRanges.from_ranges([[0, s // 2], [s // 2, s]])
        attn_type_map = [0, 1]
        max_seqlen_q = s // 2
        max_seqlen_k = s // 2

        compiled_ffa_func = torch.compile(fullgraph=True)(flex_flash_attn_func)

        o, lse = compiled_ffa_func(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges.to_tensor("cuda"),
            k_ranges=k_ranges.to_tensor("cuda"),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_type_map=torch.tensor(attn_type_map, dtype=torch.int32, device="cuda"),
            # FIXME: compiling does not support auto_range_merge
            # due to custom unique_consecutive_pairs kernel with dynamic output shape
            auto_range_merge=False,
        )
        o.backward(do)
        dq, dk, dv = q.grad, k.grad, v.grad

        self.assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=s,
            total_seqlen_k=s,
            total_q=q,
            total_k=k,
            total_v=v,
            total_out=o,
            grad_total_q=dq,
            grad_total_k=dk,
            grad_total_v=dv,
            grad_total_out=do,
            dtype=torch.bfloat16,
            test_case="test_compiled_flex_flash_attn",
        )

    def check_deterministic(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        q_ranges_tensor,
        k_ranges_tensor,
        max_seqlen_q,
        max_seqlen_k,
        attn_type_map_tensor,
        auto_range_merge,
        test_case,
        o_ref: torch.Tensor,
        dq_ref: torch.Tensor,
        dk_ref: torch.Tensor,
        dv_ref: torch.Tensor,
    ):
        # Check deterministic behavior
        # If deterministic is True, we will compare the output and gradients with a second run
        # If any of them is not equal, we will collect the error messages
        err_msg_list: list[str] = []
        q = q.clone().detach().requires_grad_(True)
        k = k.clone().detach().requires_grad_(True)
        v = v.clone().detach().requires_grad_(True)
        do = do.clone()
        o, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges_tensor,
            k_ranges_tensor,
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map_tensor,
            auto_range_merge=auto_range_merge,
            deterministic=True,
        )
        o.backward(do)

        try:
            assert torch.equal(
                o, o_ref
            ), f"For {test_case=}: forward output not deterministic"

            assert torch.equal(
                q.grad, dq_ref
            ), f"For {test_case=}: backward dq not deterministic"

            assert torch.equal(
                k.grad, dk_ref
            ), f"For {test_case=}: backward dk not deterministic"

            assert torch.equal(
                v.grad, dv_ref
            ), f"For {test_case=}: backward dv not deterministic"
        except Exception as e:
            err_msg_list.append(str(e))

        return err_msg_list

    def check_flex_flash_attn_accumulation(
        self,
        q,
        k,
        v,
        do,
        q_ranges_tensor,
        k_ranges_tensor,
        max_seqlen_q,
        max_seqlen_k,
        attn_type_map_tensor,
        auto_range_merge,
        deterministic,
        test_case,
    ):
        t, h, d = q.shape
        o_acc = torch.randn_like(q, dtype=torch.float32)
        lse_acc = torch.randn([h, t], device=q.device, dtype=torch.float32)

        softmax_scale = 1.0 / (d**0.5)

        if auto_range_merge:
            (
                merge_q_ranges,
                fwd_q_ranges,
                fwd_k_ranges,
                fwd_attn_type_map,
                fwd_qk_map,
                fwd_unique_count,
            ) = merge_ranges(q_ranges_tensor, k_ranges_tensor, attn_type_map_tensor)
            (
                merge_k_ranges,
                bwd_k_ranges,
                bwd_q_ranges,
                bwd_attn_type_map,
                bwd_kq_map,
                bwd_unique_count,
            ) = merge_ranges(k_ranges_tensor, q_ranges_tensor, attn_type_map_tensor)
        else:
            fwd_q_ranges = q_ranges_tensor
            fwd_k_ranges = k_ranges_tensor
            bwd_q_ranges = q_ranges_tensor
            bwd_k_ranges = k_ranges_tensor
            fwd_attn_type_map = attn_type_map_tensor
            bwd_attn_type_map = attn_type_map_tensor
            merge_q_ranges = None
            merge_k_ranges = None
            fwd_qk_map = None
            bwd_kq_map = None
            fwd_unique_count = None
            bwd_unique_count = None

        o, lse = _flex_flash_attn_forward(
            q=q,
            k=k,
            v=v,
            out=None,
            lse=None,
            q_ranges=fwd_q_ranges,
            k_ranges=fwd_k_ranges,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_type_map=fwd_attn_type_map,
            merge_q_ranges=merge_q_ranges,
            qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_fwd_atomic_reduction=False,
            out_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
        )

        o_ref, lse_ref = result_correction(out_list=[o, o_acc], lse_list=[lse, lse_acc])

        # NOTE: The auto accumulation call must follow the non-auto accumulation call,
        # as the latter modifies the input tensors, and the former relies on these modified tensors.
        o_auto_acc, lse_auto_acc = _flex_flash_attn_forward(
            q=q,
            k=k,
            v=v,
            out=o_acc,
            lse=lse_acc,
            q_ranges=fwd_q_ranges,
            k_ranges=fwd_k_ranges,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_type_map=fwd_attn_type_map,
            merge_q_ranges=merge_q_ranges,
            qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_fwd_atomic_reduction=False,
            out_type=None,
            deterministic=deterministic,
            sm_margin=0,
        )

        assert_close(
            o_auto_acc,
            o_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => o",
        )
        assert_close(
            lse_auto_acc,
            lse_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => lse",
        )

        dq_acc = torch.randn_like(q, dtype=torch.float32)
        dk_acc = torch.randn_like(k, dtype=torch.float32)
        dv_acc = torch.randn_like(v, dtype=torch.float32)

        dq_ref, dk_ref, dv_ref, _ = _flex_flash_attn_backward(
            do,
            q,
            k,
            v,
            o_ref.to(q.dtype),
            None,  # dq
            None,  # dk
            None,  # dv
            lse_ref,
            bwd_q_ranges,
            bwd_k_ranges,
            max_seqlen_q,
            max_seqlen_k,
            bwd_attn_type_map,
            merge_k_ranges,
            bwd_kq_map,
            bwd_unique_count,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
        )

        dq_ref += dq_acc
        dk_ref += dk_acc
        dv_ref += dv_acc

        dq_acc, dk_acc, dv_acc, _ = _flex_flash_attn_backward(
            do,
            q,
            k,
            v,
            o_ref.to(q.dtype),
            dq_acc,  # dq
            dk_acc,  # dk
            dv_acc,  # dv
            lse_ref,
            bwd_q_ranges,
            bwd_k_ranges,
            max_seqlen_q,
            max_seqlen_k,
            bwd_attn_type_map,
            merge_k_ranges,
            bwd_kq_map,
            bwd_unique_count,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
        )

        assert_close(
            dq_acc,
            dq_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dq",
        )
        assert_close(
            dk_acc,
            dk_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dk",
        )
        assert_close(
            dv_acc,
            dv_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dv",
        )

    def assert_close_to_torch_ref(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_type_map: list[int],
        total_seqlen_q: int,
        total_seqlen_k: int,
        total_q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_out: torch.Tensor,
        grad_total_q: torch.Tensor,
        grad_total_k: torch.Tensor,
        grad_total_v: torch.Tensor,
        grad_total_out: torch.Tensor,
        dtype: torch.dtype,
        test_case: str = "",
        err_msg_list: list[str] = [],
    ) -> None:
        # -----   customize tolerance threshold  ---- #
        o_atol = EPSILON
        o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)

        dq_atol = EPSILON
        dq_rtol = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)

        dk_atol = EPSILON
        dk_rtol = {torch.bfloat16: 0.15, torch.float16: 0.08}.get(dtype, 0.08)

        dv_atol = EPSILON
        dv_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)

        method_name = self._testMethodName

        # NOTE: an experimental value from magi_attention testing
        mismatch_thres_ratio: float = 2.0
        # NOTE: an experimental value from fa testing
        norm_rtol_ratio: float = 2.0

        # -----   build attn mask   ---- #
        mask = get_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=self.device,
        )

        # -----   ref1. torch ref with high precision (fp32)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None

        total_out_ref_high_precision = torch_attn_ref(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            layout="thd",
            high_precision=True,
        )
        total_out_ref_high_precision.backward(grad_total_out)
        (
            grad_total_q_ref_high_precision,
            grad_total_k_ref_high_precision,
            grad_total_v_ref_high_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )

        # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None

        total_out_ref_low_precision = torch_attn_ref(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            layout="thd",
            high_precision=False,
        )

        total_out_ref_low_precision.backward(grad_total_out)
        (
            grad_total_q_ref_low_precision,
            grad_total_k_ref_low_precision,
            grad_total_v_ref_low_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )

        # -----   assert close for fwd out   ---- #

        # fa style with Linf norm
        out_norm = calc_inf_norm(total_out, total_out_ref_high_precision)
        out_ref_norm = calc_inf_norm(
            total_out_ref_low_precision, total_out_ref_high_precision
        )
        try:
            self.assertLessEqual(
                out_norm,
                norm_rtol_ratio * out_ref_norm,
                msg=f"For {test_case=}: {out_norm=} should be no greater than {norm_rtol_ratio}x of {out_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        o_thres = extract_mismatch_threshold(
            actual=total_out_ref_low_precision,
            expected=total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        try:
            magi_attention.testing.assert_close(
                total_out,
                total_out_ref_high_precision,
                atol=o_atol,
                rtol=o_rtol,
                mismatch_threshold=o_thres,
                test_case=f"{test_case} => o",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dq   ---- #

        # fa style with Linf norm
        dq_norm = calc_inf_norm(grad_total_q, grad_total_q_ref_high_precision)
        dq_ref_norm = calc_inf_norm(
            grad_total_q_ref_low_precision, grad_total_q_ref_high_precision
        )
        try:
            self.assertLessEqual(
                dq_norm,
                norm_rtol_ratio * dq_ref_norm,
                msg=f"For {test_case=}: {dq_norm=} should be no greater than {norm_rtol_ratio}x of {dq_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dq_thres = extract_mismatch_threshold(
            actual=grad_total_q_ref_low_precision,
            expected=grad_total_q_ref_high_precision,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        if method_name == "test_flex_attn_random":
            dq_thres = 1.0
        try:
            magi_attention.testing.assert_close(
                grad_total_q,
                grad_total_q_ref_high_precision,
                atol=dq_atol,
                rtol=dq_rtol,
                mismatch_threshold=dq_thres,
                test_case=f"{test_case} => dq",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dk   ---- #

        # fa style with Linf norm
        dk_norm = calc_inf_norm(grad_total_k, grad_total_k_ref_high_precision)
        dk_ref_norm = calc_inf_norm(
            grad_total_k_ref_low_precision, grad_total_k_ref_high_precision
        )
        try:
            self.assertLessEqual(
                dk_norm,
                norm_rtol_ratio * dk_ref_norm,
                msg=f"For {test_case=}: {dk_norm=} should be no greater than {norm_rtol_ratio}x of {dk_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dk_thres = extract_mismatch_threshold(
            actual=grad_total_k_ref_low_precision,
            expected=grad_total_k_ref_high_precision,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        try:
            magi_attention.testing.assert_close(
                grad_total_k,
                grad_total_k_ref_high_precision,
                atol=dk_atol,
                rtol=dk_rtol,
                mismatch_threshold=dk_thres,
                test_case=f"{test_case} => dk",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dv   ---- #

        # fa style with Linf norm

        dv_norm = calc_inf_norm(grad_total_v, grad_total_v_ref_high_precision)
        dv_ref_norm = calc_inf_norm(
            grad_total_v_ref_low_precision, grad_total_v_ref_high_precision
        )
        try:
            self.assertLessEqual(
                dv_norm,
                norm_rtol_ratio * dv_ref_norm,
                msg=f"For {test_case=}: {dv_norm=} should be no greater than {norm_rtol_ratio}x of {dv_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dv_thres = extract_mismatch_threshold(
            actual=grad_total_v_ref_low_precision,
            expected=grad_total_v_ref_high_precision,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        try:
            magi_attention.testing.assert_close(
                grad_total_v,
                grad_total_v_ref_high_precision,
                atol=dv_atol,
                rtol=dv_rtol,
                mismatch_threshold=dv_thres,
                test_case=f"{test_case} => dv",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   raise error if any error occurs   ---- #

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

    def run_test_case(
        self,
        seqlen_q,
        seqlen_kv,
        model_config,
        dtype,
        q_ranges,
        k_ranges,
        attn_type_map,
        auto_range_merge,
        deterministic,
        test_accumulation_inplace,
        test_case,
    ):
        if auto_range_merge and deterministic:
            return

        # FIXME: for square bi-causal mask, i.e. when only the main diagonal is valid
        # ffa bwd kernel encounters with some precision issue with dq/dk,
        # thus we skip here and will fix it asap
        if is_list_value_any(attn_type_map, 3):
            return

        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]

        # construct data
        q = torch.randn(
            (seqlen_q, num_heads_q, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (seqlen_kv, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (seqlen_kv, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)

        # construct meta args
        max_seqlen_q = q_ranges.max_seqlen
        max_seqlen_k = k_ranges.max_seqlen
        q_ranges_tensor = q_ranges.to_tensor(device=self.device)
        k_ranges_tensor = k_ranges.to_tensor(device=self.device)
        attn_type_map_tensor = torch.tensor(
            attn_type_map, dtype=torch.int32, device=self.device
        )

        if test_accumulation_inplace:
            # If test_accumulation_inplace is True, we will test the accumulation and return
            self.check_flex_flash_attn_accumulation(
                q=q,
                k=k,
                v=v,
                do=do,
                q_ranges_tensor=q_ranges_tensor,
                k_ranges_tensor=k_ranges_tensor,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                attn_type_map_tensor=attn_type_map_tensor,
                auto_range_merge=auto_range_merge,
                deterministic=deterministic,
                test_case=test_case,
            )
            return

        # run ffa forward
        o, lse = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges_tensor,
            k_ranges_tensor,
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map_tensor,
            auto_range_merge=auto_range_merge,
            deterministic=deterministic,
        )
        o.backward(do)

        err_msg_list = []

        if deterministic:
            # If deterministic is True, check deterministic behavior and return
            err_msg_list = self.check_deterministic(
                q=q,
                k=k,
                v=v,
                do=do,
                q_ranges_tensor=q_ranges_tensor,
                k_ranges_tensor=k_ranges_tensor,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                attn_type_map_tensor=attn_type_map_tensor,
                auto_range_merge=auto_range_merge,
                test_case=test_case,
                o_ref=o,
                dq_ref=q.grad,
                dk_ref=k.grad,
                dv_ref=v.grad,
            )

        # compare with reference
        self.assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=seqlen_q,
            total_seqlen_k=seqlen_kv,
            total_q=q,
            total_k=k,
            total_v=v,
            total_out=o,
            grad_total_q=q.grad,
            grad_total_k=k.grad,
            grad_total_v=v.grad,
            grad_total_out=do,
            dtype=dtype,
            test_case=test_case,
            err_msg_list=err_msg_list,
        )

    MODEL_CONFIGS = [
        {
            "name": "mha_nh8_hd128",
            "num_heads_q": 8,
            "num_heads_kv": 8,
            "head_dim": 128,
        },
        {
            "name": "gqa_nhq16_nhkv4_hd128",
            "num_heads_q": 16,
            "num_heads_kv": 4,
            "head_dim": 128,
        },
        {
            "name": "mha_nh1_hd64",
            "num_heads_q": 1,
            "num_heads_kv": 1,
            "head_dim": 64,
        },
        {
            "name": "gqa_nhq4_nhkv2_hd64",
            "num_heads_q": 4,
            "num_heads_kv": 2,
            "head_dim": 64,
        },
    ]

    @parameterize(
        "attn_mask_config",
        [
            {
                "name": "full_4k",
                "seqlen": 4096,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 4096],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 4096],
                    ]
                ),
                "attn_type_map": [0],
            },
            {
                "name": "varlen_full_4k",
                "seqlen": 4096,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                        [2048, 4096],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                        [2048, 4096],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "block_causal_2k",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 512],
                        [0, 1024],
                        [0, 1280],
                        [0, 1536],
                        [0, 1792],
                        [0, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "varlen_block_causal_2k",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 512],
                        [0, 1024],
                        [1024, 1280],
                        [1024, 1536],
                        [1024, 1792],
                        [1024, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "sparse_attn_2k",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 256],
                        [0, 256],
                        [256, 512],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],  # [0, 256]
                        [512, 768],
                        [1011, 1123],
                        [0, 512],  # [256, 512]
                        [777, 888],
                        [0, 1024],  # [512, 1024]
                        [1024, 1280],  # [1024, 1280]
                        [0, 128],  # [1280, 1536],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],  # [1536, 1792],
                        [1024, 2048],  # [1792, 2048]
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "varlen_block_causal_2k_with_disjoint_ranges",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 512],
                        [0, 1024],
                        [1024, 1280],
                        [1024, 1536],
                        [1024, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0],
            },
            {
                "name": "sparse_attn_2k_with_disjoint_ranges",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 256],
                        [0, 256],
                        [256, 512],
                        [256, 512],
                        [1024, 1280],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],  # [0, 256]
                        [512, 768],
                        [1011, 1123],
                        [0, 512],  # [256, 512]
                        [777, 888],
                        [1024, 1280],  # [1024, 1280]
                        [0, 128],  # [1280, 1536],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],  # [1536, 1792],
                        [1024, 2048],  # [1792, 2048]
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "deterministic_sample",
                "seqlen": 2500,
                "q_ranges": AttnRanges.from_ranges(
                    [[i * 50, (i + 1) * 50] for i in range(50) for j in range(50)]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [[i * 50, (i + 1) * 50] for i in range(50)] * 50
                ),
                "attn_type_map": [0, 1] * 1250,
            },
            {
                "name": "sparse_attn_2k_with_same_k_ranges",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 256],
                        [0, 256],
                        [256, 512],
                        [256, 512],
                        [1024, 1280],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],  # [0, 256]
                        [512, 768],
                        [1011, 1123],
                        [0, 256],  # [256, 512]
                        [777, 888],
                        [1024, 1536],  # [1024, 1280]
                        [0, 128],  # [1280, 1536],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],  # [1536, 1792],
                        [1024, 2048],  # [1792, 2048]
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        ],
    )
    @parameterize("model_config", MODEL_CONFIGS)
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("random_attn_type_map", [False, True])
    @parameterize("auto_range_merge", [False, True])
    @parameterize("deterministic", [False, True])
    @parameterize("test_accumulation_inplace", [False, True])
    def test_flex_attn(
        self,
        attn_mask_config: dict[str, Any],
        model_config: dict[str, Any],
        dtype: torch.dtype,
        random_attn_type_map: bool,
        auto_range_merge: bool,
        deterministic: bool,
        test_accumulation_inplace: bool,
    ):
        # extract config
        seqlen = attn_mask_config["seqlen"]
        q_ranges: AttnRanges = attn_mask_config["q_ranges"]
        k_ranges: AttnRanges = attn_mask_config["k_ranges"]
        attn_type_map: list[int] = attn_mask_config["attn_type_map"]
        assert len(q_ranges) == len(k_ranges) == len(attn_type_map), (
            "q_ranges, k_ranges and attn_type_map should have the same length"
            f", but got {len(q_ranges)=}, {len(k_ranges)=}, {len(attn_type_map)=}"
        )

        if random_attn_type_map:
            # we now support attn type idx in {0, 1, 2, 3}
            attn_type_map = torch.randint(0, 4, (len(attn_type_map),)).tolist()

        test_case = (
            f"[{attn_mask_config['name']}]"
            f"[{model_config['name']}]"
            f"[dtype={dtype}]"
            f"[random_attn_type_map={random_attn_type_map}]"
            f"[auto_range_merge={auto_range_merge}]"
            f"[deterministic={deterministic}]"
            f"[test_accumulation_inplace={test_accumulation_inplace}]"
        )

        self.run_test_case(
            seqlen_q=seqlen,
            seqlen_kv=seqlen,
            model_config=model_config,
            dtype=dtype,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            auto_range_merge=auto_range_merge,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            test_case=test_case,
        )

    @pytest.mark.slow
    @parameterize("model_config", MODEL_CONFIGS)
    @parameterize(
        "generate_config",
        [
            {
                "name": "dense_test_q_1024_k_1024",
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "min_len_q": 16,
                "max_len_q": 64,
                "min_len_k": 16,
                "max_len_k": 64,
            },
            {
                "name": "dense_test_q_2048_k_2048",
                "total_seqlen_q": 2048,
                "total_seqlen_k": 2048,
                "min_len_q": 32,
                "max_len_q": 128,
                "min_len_k": 32,
                "max_len_k": 128,
            },
            {
                "name": "sparse_test_q_2048_k_4096",
                "total_seqlen_q": 2048,
                "total_seqlen_k": 4096,
                "min_len_q": 16,
                "max_len_q": 256,
                "min_len_k": 64,
                "max_len_k": 512,
            },
            {
                "name": "sparse_test_q_4096_k_2048",
                "total_seqlen_q": 4096,
                "total_seqlen_k": 2048,
                "min_len_q": 64,
                "max_len_q": 512,
                "min_len_k": 16,
                "max_len_k": 256,
            },
            # FIXME: ut fail in the following 2 test case, which may because the very small qk range.
            # {
            #    "name": "strange_test_1",
            #    "total_seqlen_q": 513,
            #    "total_seqlen_k": 123,
            #    "min_len_q": 1,
            #    "max_len_q": 513,
            #    "min_len_k": 1,
            #    "max_len_k": 123,
            # },
            # {
            #    "name": "strange_test_2",
            #    "total_seqlen_q": 1023,
            #    "total_seqlen_k": 2077,
            #    "min_len_q": 1,
            #    "max_len_q": 1023,
            #    "min_len_k": 1,
            #    "max_len_k": 2077,
            # },
        ],
    )
    @parameterize("num_pairs", [10, 100, 1000])  # the max qk range pairs to generate
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize(
        "attn_type", [0, 1, 2, 3, 4]
    )  # 0 - 3 means attn type are all 0/1/2/3, 4 means random attn type.
    @parameterize("auto_range_merge", [False, True])
    @parameterize("deterministic", [False, True])
    @parameterize("test_accumulation_inplace", [False, True])
    def test_flex_attn_random(
        self,
        model_config: dict[str, Any],
        generate_config: dict[str, Any],
        num_pairs: int,
        dtype: torch.dtype,
        attn_type: int,
        auto_range_merge: bool,
        deterministic: bool,
        test_accumulation_inplace: bool,
    ):
        """in this test, we generate q,k range randomly and as complicate as possible"""
        # extract config
        total_seqlen_q = generate_config["total_seqlen_q"]
        total_seqlen_k = generate_config["total_seqlen_k"]
        min_len_q = generate_config["min_len_q"]
        min_len_k = generate_config["min_len_k"]
        max_len_q = generate_config["max_len_q"]
        max_len_k = generate_config["min_len_k"]

        q_list, k_list = self.generate_non_overlapping_qk_pairs(
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            num_pairs=num_pairs,
            min_len_q=min_len_q,
            max_len_q=max_len_q,
            min_len_k=min_len_k,
            max_len_k=max_len_k,
            max_consecutive_failures=200,
        )
        q_ranges: AttnRanges = AttnRanges.from_ranges(q_list)
        k_ranges: AttnRanges = AttnRanges.from_ranges(k_list)

        attn_type_map = [attn_type] * q_ranges.size
        if attn_type == 4:
            attn_type_map = torch.randint(0, 4, (len(attn_type_map),)).tolist()

        assert len(q_ranges) == len(k_ranges) == len(attn_type_map), (
            "q_ranges, k_ranges and attn_type_map should have the same length"
            f", but got {len(q_ranges)=}, {len(k_ranges)=}, {len(attn_type_map)=}"
        )

        test_case = (
            f"[{model_config['name']}]"
            f"[{generate_config['name']}]"
            f"[num_pairs={num_pairs}]"
            f"[dtype={dtype}]"
            f"[attn_type_map={attn_type_map}]"
            f"[auto_range_merge={auto_range_merge}]"
            f"[deterministic={deterministic}]"
            f"[test_accumulation_inplace={test_accumulation_inplace}"
        )

        self.run_test_case(
            seqlen_q=total_seqlen_q,
            seqlen_kv=total_seqlen_k,
            model_config=model_config,
            dtype=dtype,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            auto_range_merge=auto_range_merge,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            test_case=test_case,
        )


if __name__ == "__main__":
    unittest.main()
