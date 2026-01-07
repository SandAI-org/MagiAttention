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

import random
from typing import Any

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnSinkLayout
from magi_attention.functional import flex_flash_attn_func
from magi_attention.functional.flex_flash_attn import (
    _flex_flash_attn_backward,
    _flex_flash_attn_forward,
    merge_ranges,
)
from magi_attention.functional.utils import correct_attn_fwd_result
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.flag_generator import FlagCombGenerator
from magi_attention.testing.precision import (
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.utils import is_list_value_any, make_attn_mask_from_ffa_args


class TestFlexFlashAttn(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init flag generator and its iterator
        self.flag_generator = FlagCombGenerator(
            flags=[
                "test_accumulation_inplace",
                "deterministic",
                "auto_range_merge",
                "random_attn_type_map",
            ],
            options={},
            defaults={},
            groups=[],
            strategy="heuristic",
        )
        self.flag_iterator = iter(self.flag_generator)

    @property
    def seed(self):
        return 42

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 8

    @property
    def timeout(self) -> int:
        return 4000

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

    def check_deterministic(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor | None,
        sink_layout: AttnSinkLayout,
        do: torch.Tensor,
        q_ranges_tensor: torch.Tensor,
        k_ranges_tensor: torch.Tensor,
        attn_type_map_tensor: torch.Tensor,
        auto_range_merge: bool,
        o_ref: torch.Tensor,
        lse_ref: torch.Tensor,
        dq_ref: torch.Tensor,
        dk_ref: torch.Tensor,
        dv_ref: torch.Tensor,
        dsink_ref: torch.Tensor | None,
        swap_ab: bool,
        ref_block_size: tuple[int, int],
        test_case: str,
    ) -> list[str]:
        """Check deterministic behavior
        in which we will compare the output and gradients with a second run
        and if any of them is not equal, we will collect the error messages to return
        """
        err_msg_list: list[str] = []

        q = q.clone().detach().requires_grad_(True)
        k = k.clone().detach().requires_grad_(True)
        v = v.clone().detach().requires_grad_(True)
        sink = sink.clone().detach().requires_grad_(True) if sink is not None else None
        do = do.clone()

        o, lse = flex_flash_attn_func(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map_tensor,
            sink=sink,
            sink_layout=sink_layout,
            auto_range_merge=auto_range_merge,
            deterministic=True,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
        )
        o.backward(do)

        try:
            assert torch.equal(
                o, o_ref
            ), f"For {test_case=}: forward output not deterministic"

            assert torch.equal(
                lse, lse_ref
            ), f"For {test_case=}: forward lse not deterministic"

            assert torch.equal(
                q.grad, dq_ref
            ), f"For {test_case=}: backward dq not deterministic"

            assert torch.equal(
                k.grad, dk_ref
            ), f"For {test_case=}: backward dk not deterministic"

            assert torch.equal(
                v.grad, dv_ref
            ), f"For {test_case=}: backward dv not deterministic"

            if sink is not None:
                assert torch.equal(
                    sink.grad, dsink_ref
                ), f"For {test_case=}: backward dsink not deterministic"
        except Exception as e:
            err_msg_list.append(str(e))

        return err_msg_list

    def check_flex_flash_attn_accumulation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        q_ranges_tensor: torch.Tensor,
        k_ranges_tensor: torch.Tensor,
        attn_type_map_tensor: torch.Tensor,
        auto_range_merge: bool,
        deterministic: bool,
        test_case: str,
    ):
        t, h, d = q.shape
        o_acc = torch.randn_like(q, dtype=torch.float32)
        lse_acc = torch.randn([t, h], device=q.device, dtype=torch.float32)

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
            sink=None,
            sink_layout="sh",
            out=None,
            lse=None,
            q_ranges=fwd_q_ranges,
            k_ranges=fwd_k_ranges,
            attn_type_map=fwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_fwd_atomic_reduction=False,
            out_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
            auto_range_merge=auto_range_merge,
            merge_q_ranges=merge_q_ranges,
            qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            ref_block_size=None,
            swap_ab=False,
        )

        o_ref, lse_ref = correct_attn_fwd_result(
            out_list=[o, o_acc], lse_list=[lse, lse_acc]
        )

        # NOTE: The auto accumulation call must follow the non-auto accumulation call,
        # as the latter modifies the input tensors, and the former relies on these modified tensors.
        o_auto_acc, lse_auto_acc = _flex_flash_attn_forward(
            q=q,
            k=k,
            v=v,
            sink=None,
            sink_layout="sh",
            out=o_acc,
            lse=lse_acc,
            q_ranges=fwd_q_ranges,
            k_ranges=fwd_k_ranges,
            attn_type_map=fwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_fwd_atomic_reduction=False,
            out_type=None,
            deterministic=deterministic,
            sm_margin=0,
            auto_range_merge=auto_range_merge,
            merge_q_ranges=merge_q_ranges,
            qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            ref_block_size=None,
            swap_ab=False,
        )

        assert_close(
            o_auto_acc,
            o_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => o",
            print_rank=-1,
        )
        assert_close(
            lse_auto_acc,
            lse_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => lse",
            print_rank=-1,
        )

        dq_acc = torch.randn_like(q, dtype=torch.float32)
        dk_acc = torch.randn_like(k, dtype=torch.float32)
        dv_acc = torch.randn_like(v, dtype=torch.float32)

        dq_ref, dk_ref, dv_ref, _ = _flex_flash_attn_backward(
            do,
            q,
            k,
            v,
            None,  # sink
            "sh",  # sink_layout
            o_ref.to(q.dtype),
            lse_ref,
            None,  # dq
            None,  # dk
            None,  # dv
            None,  # dsink
            bwd_q_ranges,
            bwd_k_ranges,
            bwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,  # TODO: test when it's `True`
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
            auto_range_merge=auto_range_merge,
            merge_k_ranges=merge_k_ranges,
            bwd_kq_map=bwd_kq_map,
            bwd_unique_count=bwd_unique_count,
            swap_bwd_qk_loop=False,  # TODO: test when it's `True`
        )

        dq_ref += dq_acc
        dk_ref += dk_acc
        dv_ref += dv_acc

        dq_acc, dk_acc, dv_acc, _ = _flex_flash_attn_backward(
            do,
            q,
            k,
            v,
            None,  # sink
            "sh",  # sink_layout
            o_ref.to(q.dtype),
            lse_ref,
            dq_acc,  # dq
            dk_acc,  # dk
            dv_acc,  # dv
            None,  # dsink
            bwd_q_ranges,
            bwd_k_ranges,
            bwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,  # TODO: test when it's `True`
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
            auto_range_merge=auto_range_merge,
            merge_k_ranges=merge_k_ranges,
            bwd_kq_map=bwd_kq_map,
            bwd_unique_count=bwd_unique_count,
            swap_bwd_qk_loop=False,  # TODO: test when it's `True`
        )

        assert_close(
            dq_acc,
            dq_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dq",
            print_rank=-1,
        )
        assert_close(
            dk_acc,
            dk_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dk",
            print_rank=-1,
        )
        assert_close(
            dv_acc,
            dv_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dv",
            print_rank=-1,
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
        total_sink: torch.Tensor | None,
        total_out: torch.Tensor,
        total_lse: torch.Tensor,
        grad_total_q: torch.Tensor,
        grad_total_k: torch.Tensor,
        grad_total_v: torch.Tensor,
        grad_total_sink: torch.Tensor | None,
        grad_total_out: torch.Tensor,
        dtype: torch.dtype,
        sink_layout: AttnSinkLayout,
        test_case: str = "",
        err_msg_list: list[str] = [],
        err_ratio_dict: dict[str, float] = {},
    ) -> None:
        # -----   customize tolerance / threshold  ---- #

        if total_sink is not None:
            assert isinstance(total_sink, torch.Tensor)
            has_sink = True
        else:
            has_sink = False

        o_atol = EPSILON
        o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        o_norm_rtol_ratio = err_ratio_dict.get("o_norm_rtol_ratio", NORM_RTOL_RATIO)
        o_min_norm_rtol = err_ratio_dict.get("o_min_norm_rtol", 0.0)
        o_mismatch_thres_ratio = err_ratio_dict.get(
            "o_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        o_min_mismatch_thres = err_ratio_dict.get("o_min_mismatch_thres", 0.0)
        o_max_mismatch_thres = err_ratio_dict.get(
            "o_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        lse_atol = EPSILON
        lse_rtol = 0.001
        lse_norm_rtol_ratio = err_ratio_dict.get("lse_norm_rtol_ratio", NORM_RTOL_RATIO)
        lse_min_norm_rtol = err_ratio_dict.get("lse_min_norm_rtol", 0.0)
        lse_mismatch_thres_ratio = err_ratio_dict.get(
            "lse_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        lse_min_mismatch_thres = err_ratio_dict.get("lse_min_mismatch_thres", 0.0)
        lse_max_mismatch_thres = err_ratio_dict.get(
            "lse_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dq_atol = EPSILON
        dq_rtol = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)
        dq_norm_rtol_ratio = err_ratio_dict.get("dq_norm_rtol_ratio", NORM_RTOL_RATIO)
        dq_min_norm_rtol = err_ratio_dict.get("dq_min_norm_rtol", 0.0)
        dq_mismatch_thres_ratio = err_ratio_dict.get(
            "dq_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dq_min_mismatch_thres = err_ratio_dict.get("dq_min_mismatch_thres", 0.0)
        dq_max_mismatch_thres = err_ratio_dict.get(
            "dq_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dk_atol = EPSILON
        dk_rtol = {torch.bfloat16: 0.15, torch.float16: 0.08}.get(dtype, 0.08)
        dk_norm_rtol_ratio = err_ratio_dict.get("dk_norm_rtol_ratio", NORM_RTOL_RATIO)
        dk_min_norm_rtol = err_ratio_dict.get("dk_min_norm_rtol", 0.0)
        dk_mismatch_thres_ratio = err_ratio_dict.get(
            "dk_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dk_min_mismatch_thres = err_ratio_dict.get("dk_min_mismatch_thres", 0.0)
        dk_max_mismatch_thres = err_ratio_dict.get(
            "dk_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dv_atol = EPSILON
        dv_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        dv_norm_rtol_ratio = err_ratio_dict.get("dv_norm_rtol_ratio", NORM_RTOL_RATIO)
        dv_min_norm_rtol = err_ratio_dict.get("dv_min_norm_rtol", 0.0)
        dv_mismatch_thres_ratio = err_ratio_dict.get(
            "dv_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dv_min_mismatch_thres = err_ratio_dict.get("dv_min_mismatch_thres", 0.0)
        dv_max_mismatch_thres = err_ratio_dict.get(
            "dv_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dsink_atol = err_ratio_dict.get("dsink_atol", EPSILON)
        dsink_rtol = err_ratio_dict.get("dsink_rtol", 0.05)
        dsink_norm_rtol_ratio = err_ratio_dict.get(
            "dsink_norm_rtol_ratio", NORM_RTOL_RATIO
        )
        dsink_min_norm_rtol = err_ratio_dict.get("dsink_min_norm_rtol", 0.0)
        dsink_mismatch_thres_ratio = err_ratio_dict.get(
            "dsink_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dsink_min_mismatch_thres = err_ratio_dict.get("dsink_min_mismatch_thres", 0.0)
        dsink_max_mismatch_thres = err_ratio_dict.get(
            "dsink_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        # -----   build attn mask   ---- #

        mask = make_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=self.device,
        )

        # -----   ref1. torch ref with high precision (fp64)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None
        if has_sink:
            total_sink.grad = None

        total_out_ref_high_precision, total_lse_ref_high_precision = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            sink=total_sink,
            mask=mask,
            layout="thd",
            sink_layout=sink_layout,
            high_precision=True,
            backend="torch" if has_sink else "sdpa",
            return_lse=True,
        )
        total_out_ref_high_precision.backward(grad_total_out)
        (
            grad_total_q_ref_high_precision,
            grad_total_k_ref_high_precision,
            grad_total_v_ref_high_precision,
            grad_total_sink_ref_high_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
            total_sink.grad if has_sink else None,
        )

        # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None
        if has_sink:
            total_sink.grad = None

        total_out_ref_low_precision, total_lse_ref_low_precision = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            sink=total_sink,
            mask=mask,
            layout="thd",
            sink_layout=sink_layout,
            backend="torch" if has_sink else "sdpa",
            high_precision=False,
            return_lse=True,
        )

        total_out_ref_low_precision.backward(grad_total_out)
        (
            grad_total_q_ref_low_precision,
            grad_total_k_ref_low_precision,
            grad_total_v_ref_low_precision,
            grad_total_sink_ref_low_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
            total_sink.grad if has_sink else None,
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
                max(o_min_norm_rtol, o_norm_rtol_ratio * out_ref_norm),
                msg=(
                    f"For {test_case=}: {out_norm=} should be no greater than "
                    f"max({o_min_norm_rtol}, {o_norm_rtol_ratio} x {out_ref_norm=})",
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        o_thres = extract_mismatch_threshold(
            actual=total_out_ref_low_precision,
            expected=total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=o_mismatch_thres_ratio,
            min_mismatch_thres=o_min_mismatch_thres,
            max_mismatch_thres=o_max_mismatch_thres,
        )
        try:
            assert_close(
                total_out,
                total_out_ref_high_precision,
                atol=o_atol,
                rtol=o_rtol,
                mismatch_threshold=o_thres,
                test_case=f"{test_case} => o",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for fwd lse   ---- #

        # fa style with Linf norm
        lse_norm = calc_inf_norm(total_lse, total_lse_ref_high_precision)
        lse_ref_norm = calc_inf_norm(
            total_lse_ref_low_precision, total_lse_ref_high_precision
        )
        try:
            self.assertLessEqual(
                lse_norm,
                max(lse_min_norm_rtol, lse_norm_rtol_ratio * lse_ref_norm),
                msg=(
                    f"For {test_case=}: {lse_norm=} should be no greater than "
                    f"max({lse_min_norm_rtol}, {lse_norm_rtol_ratio} x {lse_ref_norm=})"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        lse_thres = extract_mismatch_threshold(
            actual=total_lse_ref_low_precision,
            expected=total_lse_ref_high_precision,
            atol=lse_atol,
            rtol=lse_rtol,
            mismatch_thres_ratio=lse_mismatch_thres_ratio,
            min_mismatch_thres=lse_min_mismatch_thres,
            max_mismatch_thres=lse_max_mismatch_thres,
        )
        try:
            assert_close(
                total_lse,
                total_lse_ref_high_precision,
                atol=lse_atol,
                rtol=lse_rtol,
                mismatch_threshold=lse_thres,
                test_case=f"{test_case} => lse",
                print_rank=-1,
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
                max(dq_min_norm_rtol, dq_norm_rtol_ratio * dq_ref_norm),
                msg=(
                    f"For {test_case=}: {dq_norm=} should be no greater than "
                    f"max({dq_min_norm_rtol}, {dq_norm_rtol_ratio} x {dq_ref_norm=})"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dq_thres = extract_mismatch_threshold(
            actual=grad_total_q_ref_low_precision,
            expected=grad_total_q_ref_high_precision,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_thres_ratio=dq_mismatch_thres_ratio,
            min_mismatch_thres=dq_min_mismatch_thres,
            max_mismatch_thres=dq_max_mismatch_thres,
        )
        try:
            assert_close(
                grad_total_q,
                grad_total_q_ref_high_precision,
                atol=dq_atol,
                rtol=dq_rtol,
                mismatch_threshold=dq_thres,
                test_case=f"{test_case} => dq",
                print_rank=-1,
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
                max(dk_min_norm_rtol, dk_norm_rtol_ratio * dk_ref_norm),
                msg=(
                    f"For {test_case=}: {dk_norm=} should be no greater than "
                    f"max({dk_min_norm_rtol}, {dk_norm_rtol_ratio} x {dk_ref_norm=})"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dk_thres = extract_mismatch_threshold(
            actual=grad_total_k_ref_low_precision,
            expected=grad_total_k_ref_high_precision,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_thres_ratio=dk_mismatch_thres_ratio,
            min_mismatch_thres=dk_min_mismatch_thres,
            max_mismatch_thres=dk_max_mismatch_thres,
        )
        try:
            assert_close(
                grad_total_k,
                grad_total_k_ref_high_precision,
                atol=dk_atol,
                rtol=dk_rtol,
                mismatch_threshold=dk_thres,
                test_case=f"{test_case} => dk",
                print_rank=-1,
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
                max(dv_min_norm_rtol, dv_norm_rtol_ratio * dv_ref_norm),
                msg=(
                    f"For {test_case=}: {dv_norm=} should be no greater than "
                    f"max({dv_min_norm_rtol}, {dv_norm_rtol_ratio} x {dv_ref_norm=})"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dv_thres = extract_mismatch_threshold(
            actual=grad_total_v_ref_low_precision,
            expected=grad_total_v_ref_high_precision,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_thres_ratio=dv_mismatch_thres_ratio,
            min_mismatch_thres=dv_min_mismatch_thres,
            max_mismatch_thres=dv_max_mismatch_thres,
        )
        try:
            assert_close(
                grad_total_v,
                grad_total_v_ref_high_precision,
                atol=dv_atol,
                rtol=dv_rtol,
                mismatch_threshold=dv_thres,
                test_case=f"{test_case} => dv",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dsink   ---- #

        if has_sink:
            # fa style with Linf norm
            dsink_norm = calc_inf_norm(
                grad_total_sink, grad_total_sink_ref_high_precision
            )
            dsink_ref_norm = calc_inf_norm(
                grad_total_sink_ref_low_precision, grad_total_sink_ref_high_precision
            )
            try:
                self.assertLessEqual(
                    dsink_norm,
                    max(dsink_min_norm_rtol, dsink_norm_rtol_ratio * dsink_ref_norm),
                    msg=(
                        f"For {test_case=}: {dsink_norm=} should be no greater than "
                        f"max({dsink_min_norm_rtol}, {dsink_norm_rtol_ratio} x {dsink_ref_norm=})"
                    ),
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            dsink_thres = extract_mismatch_threshold(
                actual=grad_total_sink_ref_low_precision,
                expected=grad_total_sink_ref_high_precision,
                atol=dsink_atol,
                rtol=dsink_rtol,
                mismatch_thres_ratio=dsink_mismatch_thres_ratio,
                min_mismatch_thres=dsink_min_mismatch_thres,
                max_mismatch_thres=dsink_max_mismatch_thres,
            )
            try:
                assert_close(
                    grad_total_sink,
                    grad_total_sink_ref_high_precision,
                    atol=dsink_atol,
                    rtol=dsink_rtol,
                    mismatch_threshold=dsink_thres,
                    test_case=f"{test_case} => dsink",
                    print_rank=-1,
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   raise error if any error occurs   ---- #

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

    def run_test_case(
        self,
        seqlen_q: int,
        seqlen_kv: int,
        seqlen_sink: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        dtype: torch.dtype,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_type_map: list[int],
        auto_range_merge: bool,
        deterministic: bool,
        test_accumulation_inplace: bool,
        sink_layout: AttnSinkLayout,
        swap_ab: bool,
        ref_block_size: tuple[int, int],
        test_case: str,
        err_ratio_dict: dict[str, float] = {},
    ) -> None:
        if auto_range_merge and deterministic:
            return

        # FIXME: for square bi-causal mask, i.e. when only the main diagonal is valid
        # ffa bwd kernel encounters with some precision issue with dq/dk,
        # thus we skip here and will fix it asap
        if is_list_value_any(attn_type_map, 3):
            return

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
        has_sink = seqlen_sink > 0
        if has_sink:
            match sink_layout:
                case "sh":
                    sink = torch.randn(
                        (seqlen_sink, num_heads_q),
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=True,
                    )
                case "ssh":
                    sink = torch.randn(
                        (seqlen_q, seqlen_sink, num_heads_q),
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=True,
                    )
                case "shd":
                    raise NotImplementedError(
                        f"sink_layout {sink_layout} is not supported yet"
                    )
                case _:
                    raise ValueError(f"Invalid sink_layout {sink_layout}")
        else:
            sink = None

        # construct meta args
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
                attn_type_map_tensor=attn_type_map_tensor,
                auto_range_merge=auto_range_merge,
                deterministic=deterministic,
                test_case=test_case,
            )
            return

        # run ffa forward
        o, lse = flex_flash_attn_func(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map_tensor,
            sink=sink,
            sink_layout=sink_layout,
            auto_range_merge=auto_range_merge,
            deterministic=deterministic,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
        )

        # run ffa backward
        o.backward(do)

        err_msg_list = []

        # if deterministic is True, check deterministic behavior and return error messages
        if deterministic:
            err_msg_list = self.check_deterministic(
                q=q,
                k=k,
                v=v,
                sink=sink,
                sink_layout=sink_layout,
                do=do,
                q_ranges_tensor=q_ranges_tensor,
                k_ranges_tensor=k_ranges_tensor,
                attn_type_map_tensor=attn_type_map_tensor,
                auto_range_merge=auto_range_merge,
                o_ref=o,
                lse_ref=lse,
                dq_ref=q.grad,
                dk_ref=k.grad,
                dv_ref=v.grad,
                dsink_ref=sink.grad if has_sink else None,
                swap_ab=swap_ab,
                ref_block_size=ref_block_size,
                test_case=test_case,
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
            total_sink=sink,
            total_out=o,
            total_lse=lse,
            grad_total_q=q.grad,
            grad_total_k=k.grad,
            grad_total_v=v.grad,
            grad_total_sink=sink.grad if has_sink else None,
            grad_total_out=do,
            dtype=dtype,
            sink_layout=sink_layout,
            test_case=test_case,
            err_msg_list=err_msg_list,
            err_ratio_dict=err_ratio_dict,
        )

    MODEL_CONFIGS = [
        {
            "name": "mha_nh8_hd128",
            "num_heads_q": 8,
            "num_heads_kv": 8,
            "head_dim": 128,
        },
        {
            "name": "gqa_nhq32_nhkv1_hd128",
            "num_heads_q": 32,
            "num_heads_kv": 1,
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

    @with_run_in_mp
    @parameterize(
        "attn_mask_config",
        [
            {
                "name": "full_4k",
                "seqlen": 4096,
                "seqlen_sink": 1,
                "sink_layout": "ssh",
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
                "name": "varlen_full_1k",
                "seqlen": 1024,
                "seqlen_sink": 2,
                "sink_layout": "sh",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 366],
                        [366, 391],
                        [391, 471],
                        [471, 835],
                        [835, 984],
                        [984, 1005],
                        [1005, 1017],
                        [1017, 1020],
                        [1020, 1023],
                        [1023, 1024],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 366],
                        [366, 391],
                        [391, 471],
                        [471, 835],
                        [835, 984],
                        [984, 1005],
                        [1005, 1017],
                        [1017, 1020],
                        [1020, 1023],
                        [1023, 1024],
                    ]
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "varlen_full_4k",
                "seqlen": 4096,
                "seqlen_sink": 4,
                "sink_layout": "ssh",
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
                "seqlen_sink": 6,
                "sink_layout": "sh",
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
                "seqlen_sink": 8,
                "sink_layout": "ssh",
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
                        [0, 256],
                        [512, 768],
                        [1011, 1123],
                        [0, 512],
                        [777, 888],
                        [0, 1024],
                        [1024, 1280],
                        [0, 128],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],
                        [1024, 2048],
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
                        [0, 256],
                        [512, 768],
                        [1011, 1123],
                        [0, 512],
                        [777, 888],
                        [1024, 1280],
                        [0, 128],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],
                        [1024, 2048],
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
                        [0, 256],
                        [512, 768],
                        [1011, 1123],
                        [0, 256],
                        [777, 888],
                        [1024, 1536],
                        [0, 128],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],
                        [1024, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        ],
    )
    @parameterize(
        "ref_block_config",
        [
            {
                "swap_ab": False,
                "ref_block_size": None,
            },
            {
                "swap_ab": False,
                "ref_block_size": (64, 64),
            },
            {
                "swap_ab": True,
                "ref_block_size": (8, 64),
            },
            {
                "swap_ab": True,
                "ref_block_size": (16, 64),
            },
            {
                "swap_ab": True,
                "ref_block_size": (32, 64),
            },
            {
                "swap_ab": True,
                "ref_block_size": (64, 64),
            },
        ],
    )
    @parameterize("model_config", MODEL_CONFIGS)
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    def test_ffa_simple(
        self,
        attn_mask_config: dict[str, Any],
        ref_block_config: dict[str, Any],
        model_config: dict[str, Any],
        dtype: torch.dtype,
    ):
        # -----    switch env flags by FlagCombGenerator   ---- #
        flag_comb = next(self.flag_iterator)
        flag_comb_test_case = FlagCombGenerator.to_test_case(flag_comb)

        # extract config
        seqlen: int = attn_mask_config["seqlen"]
        seqlen_sink: int = attn_mask_config.get("seqlen_sink", 0)
        sink_layout: AttnSinkLayout = attn_mask_config.get("sink_layout", "sh")
        num_heads_q: int = model_config["num_heads_q"]
        q_ranges: AttnRanges = attn_mask_config["q_ranges"]
        k_ranges: AttnRanges = attn_mask_config["k_ranges"]
        attn_type_map: list[int] = attn_mask_config["attn_type_map"]
        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        assert len(q_ranges) == len(k_ranges) == len(attn_type_map), (
            "q_ranges, k_ranges and attn_type_map should have the same length"
            f", but got {len(q_ranges)=}, {len(k_ranges)=}, {len(attn_type_map)=}"
        )

        test_accumulation_inplace = bool(
            flag_comb.get("test_accumulation_inplace", False)
        )
        deterministic = bool(flag_comb.get("deterministic", False))
        auto_range_merge = bool(flag_comb.get("auto_range_merge", False))
        random_attn_type_map = bool(flag_comb.get("random_attn_type_map", False))
        swap_ab = ref_block_config["swap_ab"]
        ref_block_size = ref_block_config["ref_block_size"]

        if random_attn_type_map:
            # we now support attn type idx in {0, 1, 2, 3}
            attn_type_map = torch.randint(0, 4, (len(attn_type_map),)).tolist()

        test_case = (
            f"[RANK {self.rank}][test_ffa_simple]"
            f"[{attn_mask_config['name']}]"
            f"[{model_config['name']}]"
            f"[dtype={dtype}]"
            f"[swap_ab={swap_ab}]"
            f"[ref_block_size={ref_block_size}]"
            f"[has_sink={seqlen_sink > 0}]"
            f"[sink_layout={sink_layout}] x "
            f"{flag_comb_test_case}"
        )

        self.run_test_case(
            seqlen_q=seqlen,
            seqlen_kv=seqlen,
            seqlen_sink=seqlen_sink,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            dtype=dtype,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            auto_range_merge=auto_range_merge,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            sink_layout=sink_layout,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            test_case=test_case,
            err_ratio_dict={
                "dq_min_mismatch_thres": 5e-3,
                # FIXME: dsink ratios are fragile right now, need to be improved later
                "dsink_mismatch_thres_ratio": MISMATCH_THRES_RATIO * 1.5,
                "dsink_min_mismatch_thres": max(1 / (seqlen_sink * num_heads_q), 8e-2)
                if seqlen_sink > 0 and sink_layout == "sh"
                else 8e-2,
                "dsink_min_norm_rtol": 0.015,
                "dsink_norm_rtol_ratio": NORM_RTOL_RATIO * 2,
                "dsink_atol": 2e-4 if sink_layout == "sh" else EPSILON,
                "dsink_rtol": 0.1,
            },
        )

    @with_run_in_mp
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
            # FIXME: ut failed for the following 2 test cases, maybe due to very small qk ranges.
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
    @parameterize(
        "ref_block_config",
        [
            {
                "swap_ab": False,
                "ref_block_size": None,
            },
            {
                "swap_ab": False,
                "ref_block_size": (64, 64),
            },
            {
                "swap_ab": True,
                "ref_block_size": (8, 64),
            },
            {
                "swap_ab": True,
                "ref_block_size": (16, 64),
            },
            {
                "swap_ab": True,
                "ref_block_size": (32, 64),
            },
            {
                "swap_ab": True,
                "ref_block_size": (64, 64),
            },
        ],
    )
    @parameterize(
        "num_pairs", [10, 100, 1000]
    )  # the max num of qk range pairs to generate
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize(
        "attn_type", [0, 1, 2, 3, 4]
    )  # 0 - 3 means attn type are all 0/1/2/3, 4 means random attn type.
    def test_ffa_random(
        self,
        model_config: dict[str, Any],
        generate_config: dict[str, Any],
        ref_block_config: dict[str, Any],
        num_pairs: int,
        dtype: torch.dtype,
        attn_type: int,
    ):
        """in this test, we generate q,k range randomly and as complicate as possible"""
        # -----    switch env flags by FlagCombGenerator   ---- #
        flag_comb = next(self.flag_iterator)
        flag_comb_test_case = FlagCombGenerator.to_test_case(flag_comb)

        # extract config
        total_seqlen_q: int = generate_config["total_seqlen_q"]
        total_seqlen_k: int = generate_config["total_seqlen_k"]
        min_len_q: int = generate_config["min_len_q"]
        min_len_k: int = generate_config["min_len_k"]
        max_len_q: int = generate_config["max_len_q"]
        max_len_k: int = generate_config["min_len_k"]

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
        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        assert len(q_ranges) == len(k_ranges) == len(attn_type_map), (
            "q_ranges, k_ranges and attn_type_map should have the same length"
            f", but got {len(q_ranges)=}, {len(k_ranges)=}, {len(attn_type_map)=}"
        )

        swap_ab = ref_block_config["swap_ab"]
        ref_block_size = ref_block_config["ref_block_size"]
        test_accumulation_inplace = bool(
            flag_comb.get("test_accumulation_inplace", False)
        )
        deterministic = bool(flag_comb.get("deterministic", False))
        auto_range_merge = bool(flag_comb.get("auto_range_merge", False))

        test_case = (
            f"[RANK {self.rank}][test_ffa_random]"
            f"[{model_config['name']}]"
            f"[{generate_config['name']}]"
            f"[num_pairs={num_pairs}]"
            f"[dtype={dtype}]"
            f"[attn_type_map=[{attn_type}] x {q_ranges.size}]"
            f"[swap_ab={swap_ab}]"
            f"[ref_block_size={ref_block_size}] x "
            f"{flag_comb_test_case}"
        )

        self.run_test_case(
            seqlen_q=total_seqlen_q,
            seqlen_kv=total_seqlen_k,
            seqlen_sink=0,  # pass testing attn sink for now
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            dtype=dtype,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            auto_range_merge=auto_range_merge,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            test_case=test_case,
            sink_layout="sh",
            err_ratio_dict={
                "dq_mismatch_thres_ratio": MISMATCH_THRES_RATIO * 1.5,
                "dq_min_mismatch_thres": 0.025,
                "dk_mismatch_thres_ratio": MISMATCH_THRES_RATIO * 1.5,
                "dk_min_mismatch_thres": 0.025,
                "dv_norm_rtol_ratio": NORM_RTOL_RATIO * 1.5,
            },
        )

    @parameterize("sink_layout", ["sh", "ssh"])  # ["sh", "ssh", "shd"])
    def test_ffa_compiled(self, sink_layout: AttnSinkLayout):
        s, s_sink = 2048, 4
        hq, hk, d = 6, 3, 128

        q = torch.randn(s, hq, d, dtype=torch.bfloat16, device=self.device)
        k = torch.randn(s, hk, d, dtype=torch.bfloat16, device=self.device)
        v = torch.randn_like(k)
        do = torch.randn_like(q)
        match sink_layout:
            case "sh":
                sink = torch.randn(s_sink, hq, dtype=torch.float32, device=self.device)
            case "ssh":
                sink = torch.randn(
                    s, s_sink, hq, dtype=torch.float32, device=self.device
                )
            case "shd":
                raise NotImplementedError(
                    f"sink_layout {sink_layout} is not supported yet"
                )
            case _:
                raise ValueError(f"Invalid sink_layout {sink_layout}")

        [x.requires_grad_(True) for x in (q, k, v, sink)]

        q_ranges = AttnRanges.from_ranges([[0, s // 2], [s // 2, s]])
        k_ranges = AttnRanges.from_ranges([[0, s // 2], [s // 2, s]])
        attn_type_map = [0, 1]

        compiled_ffa_func = torch.compile(fullgraph=True)(flex_flash_attn_func)

        o, lse = compiled_ffa_func(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges.to_tensor(self.device),
            k_ranges=k_ranges.to_tensor(self.device),
            attn_type_map=torch.tensor(
                attn_type_map, dtype=torch.int32, device=self.device
            ),
            sink=sink,
            sink_layout=sink_layout,
            # FIXME: compiling does not support auto_range_merge
            # due to custom unique_consecutive_pairs kernel with dynamic output shape
            auto_range_merge=False,
        )
        o.backward(do)
        dq, dk, dv, dsink = q.grad, k.grad, v.grad, sink.grad

        self.assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=s,
            total_seqlen_k=s,
            total_q=q,
            total_k=k,
            total_v=v,
            total_sink=sink,
            total_out=o,
            total_lse=lse,
            grad_total_q=dq,
            grad_total_k=dk,
            grad_total_v=dv,
            grad_total_sink=dsink,
            grad_total_out=do,
            dtype=torch.bfloat16,
            sink_layout=sink_layout,
            test_case=("[test_ffa_compiled]" f"[sink_layout={sink_layout}]"),
        )


if __name__ == "__main__":
    run_tests()
