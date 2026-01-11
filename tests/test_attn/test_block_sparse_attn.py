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

from typing import Any, Optional, Tuple

import pytest
import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.functional import flex_flash_attn_func
from magi_attention.functional.flex_flash_attn import (
    _flex_flash_attn_backward,
    _flex_flash_attn_forward,
    merge_ranges,
)
from magi_attention.functional.utils import correct_attn_fwd_result

# from magi_attention.testing import parameterize
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import (  # ref_attn_func,
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.utils.sparse_utils import (
    choose_ref_block,
    flatten_block_mask_to_kv_shape,
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask_triton,
    generate_ranges_from_topk_indices_triton,
    generate_ranges_from_var_block_mask,
    generate_variable_block_sparse_pattern,
    get_sdpa_mask_from_block_sparse_mask,
    get_sdpa_mask_from_topk_indices,
    get_sdpa_mask_from_var_block_mask,
)


class TestBlockSparseAttn(DistTestBase):
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
        return 3000  # Increase timeout for JIT compilation

    def check_deterministic(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        q_ranges_tensor,
        k_ranges_tensor,
        attn_type_map_tensor,
        auto_range_merge,
        ref_block_size,
        test_case,
        o_ref: torch.Tensor,
        dq_ref: torch.Tensor,
        dk_ref: torch.Tensor,
        dv_ref: torch.Tensor,
    ):
        # (Implementation is identical to the original)
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
            attn_type_map_tensor,
            auto_range_merge=auto_range_merge,
            deterministic=True,
            ref_block_size=ref_block_size,
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
        attn_type_map_tensor,
        auto_range_merge,
        deterministic,
        test_case,
    ):
        # (Implementation is identical to the original)
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
            fwd_q_ranges = bwd_q_ranges = q_ranges_tensor
            fwd_k_ranges = bwd_k_ranges = k_ranges_tensor
            fwd_attn_type_map = bwd_attn_type_map = attn_type_map_tensor
            merge_q_ranges = merge_k_ranges = None
            fwd_qk_map = bwd_kq_map = None
            fwd_unique_count = bwd_unique_count = None

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
            merge_q_ranges=merge_q_ranges,
            qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            ref_block_size=None,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_fwd_atomic_reduction=False,
            out_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
        )
        o_ref, lse_ref = correct_attn_fwd_result(
            out_list=[o, o_acc], lse_list=[lse, lse_acc]
        )
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
            merge_q_ranges=merge_q_ranges,
            qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            ref_block_size=None,
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
            None,  # dq
            None,  # dk
            None,  # dv
            None,  # dsink
            lse_ref,
            bwd_q_ranges,
            bwd_k_ranges,
            bwd_attn_type_map,
            merge_k_ranges,
            bwd_kq_map,
            bwd_unique_count,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,  # TODO: test when it's `True`
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
            None,  # sink
            "sh",  # sink_layout
            o_ref.to(q.dtype),
            dq_acc,
            dk_acc,
            dv_acc,
            None,  # dsink
            lse_ref,
            bwd_q_ranges,
            bwd_k_ranges,
            bwd_attn_type_map,
            merge_k_ranges,
            bwd_kq_map,
            bwd_unique_count,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,  # TODO: test when it's `True`
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

    def get_ffa_result(
        self,
        q,
        k,
        v,
        grad_output,
        block_mask,
        head_wise,
        block_size,
        nhq,
        nhk,
        pack_gqa,
        deterministic,
        test_accumulation_inplace,
        swap_ab,
        ref_block_size,
        sparse_load,
        test_case,
        err_msg_list,
        sparse_format="block_mask",
        uniform=True,
        block_row_sz=None,
        block_col_sz=None,
    ):
        # (Implementation is identical to the original)
        s = q.size(1)
        h1 = k.size(2)
        q = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=h1)
        assert nhq % nhk == 0
        """
        repeats = nhq // nhk
        if head_wise == "q":
            k = torch.repeat_interleave(k, repeats=repeats, dim=2)
            v = torch.repeat_interleave(v, repeats=repeats, dim=2)
        """
        # flatten kv head.
        k = rearrange(k, "b s h d -> (b h s) 1 d")
        v = rearrange(v, "b s h d -> (b h s) 1 d")
        q.retain_grad()
        k.retain_grad()
        v.retain_grad()
        q.grad, k.grad, v.grad = None, None, None
        # flat_block_sparse_mask = flatten_block_mask(block_mask, nhq, nhk)
        if uniform:
            q_block_size, k_block_size = block_size
            if sparse_format == "block_mask":
                (
                    q_ranges_tensor,
                    k_ranges_tensor,
                ) = generate_ranges_from_block_mask_triton(
                    block_mask, q_block_size, k_block_size
                )
            elif sparse_format == "topk":
                num_k_blocks = s // k_block_size
                (
                    q_ranges_tensor,
                    k_ranges_tensor,
                ) = generate_ranges_from_topk_indices_triton(
                    block_mask, q_block_size, k_block_size, num_k_blocks
                )
        else:
            flat_block_sparse_mask = flatten_block_mask_to_kv_shape(block_mask)
            q_ranges_tensor, k_ranges_tensor = generate_ranges_from_var_block_mask(
                flat_block_sparse_mask, block_row_sz, block_col_sz, nhq, nhk
            )
        attn_type_map_tensor = torch.zeros(
            len(q_ranges_tensor), dtype=torch.int32, device="cuda"
        )

        # FIXME: dout shape error when enable test_accumulation_inplace
        """
        if test_accumulation_inplace:
            # If test_accumulation_inplace is True, we will test the accumulation and return
            self.check_flex_flash_attn_accumulation(
                q=q,
                k=k,
                v=v,
                do=grad_output,
                q_ranges_tensor=q_ranges_tensor,
                k_ranges_tensor=k_ranges_tensor,
                attn_type_map_tensor=attn_type_map,
                auto_range_merge=True,
                deterministic=deterministic,
                test_case=test_case,
            )

        q.grad = None
        k.grad = None
        v.grad = None
        """
        qhead_per_khead = q.size(1) // k.size(1)
        ref_block_size = choose_ref_block(
            block_size,
            swap_ab=swap_ab,
            pack_gqa=pack_gqa,
            qhead_per_khead=qhead_per_khead,
            sparse_load=sparse_load,
        )

        o, lse = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map_tensor,
            auto_range_merge=True,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            sparse_load=sparse_load,
        )
        o = rearrange(o, "(b h1 s) h2 d -> b s (h1 h2) d", b=1, s=s, h1=h1)
        lse = rearrange(lse, "(h1 s) h2 -> s (h1 h2)", s=s, h1=h1)

        o.backward(grad_output)

        if deterministic:
            err_msg_list.append(
                self.check_deterministic(
                    q=q,
                    k=k,
                    v=v,
                    do=grad_output,
                    q_ranges_tensor=q_ranges_tensor,
                    k_ranges_tensor=k_ranges_tensor,
                    attn_type_map_tensor=attn_type_map_tensor,
                    auto_range_merge=True,
                    ref_block_size=ref_block_size,
                    test_case=test_case,
                    o_ref=o,
                    dq_ref=q.grad,
                    dk_ref=k.grad,
                    dv_ref=v.grad,
                )
            )

        return o, lse

    def get_sdpa_attn_ref(
        self,
        q,
        k,
        v,
        grad_output,
        seqlen,
        block_size,
        block_mask,
        sparse_format="block_mask",
        uniform=True,
        block_row_sz=None,
        block_col_sz=None,
        high_precision=False,
    ):
        # (Implementation is identical to the original)

        q = rearrange(q, "1 s h d -> s h d")  # shd
        k = rearrange(k, "1 s h d -> s h d")
        v = rearrange(v, "1 s h d -> s h d")
        if uniform:
            q_block_size, k_block_size = block_size
            if sparse_format == "block_mask":
                sdpa_mask_4d = get_sdpa_mask_from_block_sparse_mask(
                    block_mask, seqlen, seqlen, q_block_size, k_block_size, q.size(1)
                )
            elif sparse_format == "topk":
                sdpa_mask_4d = get_sdpa_mask_from_topk_indices(
                    block_mask, seqlen, seqlen, q_block_size, k_block_size, q.size(1)
                )
            else:
                raise ValueError("Not supported sparse format.")
        else:
            sdpa_mask_4d = get_sdpa_mask_from_var_block_mask(
                block_mask, seqlen, seqlen, block_row_sz, block_col_sz, q.size(1)
            )
        sdpa_mask = rearrange(
            sdpa_mask_4d, "1 h seqlen_q seqlen_k -> h seqlen_q seqlen_k"
        )

        o, lse = ref_attn_func(
            q=q,
            k=k,
            v=v,
            sink=None,
            mask=sdpa_mask,
            layout="thd",
            high_precision=high_precision,
            backend="sdpa",
            return_lse=True,
            sink_layout=None,
        )

        o = rearrange(o, "s h d -> 1 s h d")
        lse = rearrange(lse, "1 seqlen h -> seqlen h")
        o.backward(grad_output)

        return o, lse

    def assert_close_to_torch_ref(
        self,
        dtype,
        q,
        k,
        v,
        grad_output,
        seqlen,
        block_size,
        block_mask,
        head_wise,
        sparse_format,
        nhq,
        nhk,
        pack_gqa,
        deterministic,
        test_accumulation_inplace,
        swap_ab: bool,
        ref_block_size: tuple[int, int],
        sparse_load,
        test_case,
        sparsity_ratio,
        uniform=True,
        block_row_sz=None,
        block_col_sz=None,
        err_ratio_dict: dict[str, float] = {},
    ):
        # (Implementation is identical to the original)
        high_precision_torch_out_ref, high_precision_lse_ref = self.get_sdpa_attn_ref(
            q,
            k,
            v,
            grad_output,
            seqlen,
            block_size,
            block_mask,
            sparse_format=sparse_format,
            uniform=uniform,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            high_precision=True,
        )
        high_precision_dq_ref, high_precision_dk_ref, high_precision_dv_ref = (
            q.grad,
            k.grad,
            v.grad,
        )

        q.grad, k.grad, v.grad = None, None, None
        low_precision_torch_out_ref, low_precision_lse_ref = self.get_sdpa_attn_ref(
            q,
            k,
            v,
            grad_output,
            seqlen,
            block_size,
            block_mask,
            sparse_format=sparse_format,
            uniform=uniform,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            high_precision=False,
        )
        low_precision_dq_ref, low_precision_dk_ref, low_precision_dv_ref = (
            q.grad,
            k.grad,
            v.grad,
        )

        q.grad, k.grad, v.grad = None, None, None
        err_msg_list: list[str] = []

        ffa_out, ffa_lse = self.get_ffa_result(
            q,
            k,
            v,
            grad_output,
            block_mask,
            head_wise,
            block_size,
            nhq,
            nhk,
            pack_gqa,
            deterministic,
            test_accumulation_inplace,
            swap_ab,
            ref_block_size,
            sparse_load,
            test_case,
            err_msg_list,
            sparse_format=sparse_format,
            uniform=uniform,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
        )
        ffa_dq, ffa_dk, ffa_dv = q.grad, k.grad, v.grad

        #  -------  test with torch ref ------- #
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

        # -----   assert close for fwd out   ---- #
        # norm_rtol_ratio = 2.0
        out_norm = calc_inf_norm(ffa_out, high_precision_torch_out_ref)
        out_ref_norm = calc_inf_norm(
            low_precision_torch_out_ref, high_precision_torch_out_ref
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
            actual=low_precision_torch_out_ref,
            expected=high_precision_torch_out_ref,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=o_mismatch_thres_ratio,
            min_mismatch_thres=o_min_mismatch_thres,
            max_mismatch_thres=o_max_mismatch_thres,
        )
        try:
            assert_close(
                ffa_out,
                high_precision_torch_out_ref,
                atol=o_atol,
                rtol=o_rtol,
                mismatch_threshold=o_thres,
                test_case=f"{test_case} => o",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for fwd lse   ---- #

        lse_norm = calc_inf_norm(ffa_lse, high_precision_lse_ref)
        lse_ref_norm = calc_inf_norm(low_precision_lse_ref, high_precision_lse_ref)
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
            actual=low_precision_lse_ref,
            expected=high_precision_lse_ref,
            atol=lse_atol,
            rtol=lse_rtol,
            mismatch_thres_ratio=lse_mismatch_thres_ratio,
            min_mismatch_thres=lse_min_mismatch_thres,
            max_mismatch_thres=lse_max_mismatch_thres,
        )
        try:
            assert_close(
                ffa_lse,
                high_precision_lse_ref,
                atol=lse_atol,
                rtol=lse_rtol,
                mismatch_threshold=lse_thres,
                test_case=f"{test_case} => lse",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        dq_norm = calc_inf_norm(ffa_dq, high_precision_dq_ref)
        dq_ref_norm = calc_inf_norm(low_precision_dq_ref, high_precision_dq_ref)

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
            actual=low_precision_dq_ref,
            expected=high_precision_dq_ref,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_thres_ratio=dq_mismatch_thres_ratio,
            min_mismatch_thres=dq_min_mismatch_thres,
            max_mismatch_thres=dq_max_mismatch_thres,
        )
        try:
            assert_close(
                ffa_dq,
                high_precision_dq_ref,
                atol=dq_atol,
                rtol=dq_rtol,
                mismatch_threshold=dq_thres,
                test_case=f"{test_case} => dq",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        dk_norm = calc_inf_norm(ffa_dk, high_precision_dk_ref)
        dk_ref_norm = calc_inf_norm(low_precision_dk_ref, high_precision_dk_ref)

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
            actual=low_precision_dk_ref,
            expected=high_precision_dk_ref,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_thres_ratio=dk_mismatch_thres_ratio,
            min_mismatch_thres=dk_min_mismatch_thres,
            max_mismatch_thres=dk_max_mismatch_thres,
        )
        try:
            assert_close(
                ffa_dk,
                high_precision_dk_ref,
                atol=dk_atol,
                rtol=dk_rtol,
                mismatch_threshold=dk_thres,
                test_case=f"{test_case} => dk",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        dv_norm = calc_inf_norm(ffa_dv, high_precision_dv_ref)
        dv_ref_norm = calc_inf_norm(low_precision_dv_ref, high_precision_dv_ref)

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
            actual=low_precision_dv_ref,
            expected=high_precision_dv_ref,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_thres_ratio=dv_mismatch_thres_ratio,
            min_mismatch_thres=dv_min_mismatch_thres,
            max_mismatch_thres=dv_max_mismatch_thres,
        )
        try:
            assert_close(
                ffa_dv,
                low_precision_dv_ref,
                atol=dv_atol,
                rtol=dv_rtol,
                mismatch_threshold=dv_thres,
                test_case=f"{test_case} => dv",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

    def _generate_sparse_pattern(
        self,
        test_type: str,
        num_heads_q: int,
        num_heads_kv: int,
        seqlen: int,
        sparsity_ratio: float,
        sparsity_granularity: str,
        sparse_format: str,
        block_size: Optional[Tuple[int, int]] = None,
        average_block_size: Optional[Tuple[int, int]] = None,
        min_block_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[
        torch.Tensor, Tuple[int, int], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Helper function to generate either uniform or variable block sparse patterns.

        Returns:
            A tuple containing:
            - block_mask (torch.Tensor): The generated sparse mask.
            - block_sizes (Tuple[int, int]): Q block size and K/V block size.
            - block_row_sz (torch.Tensor or None): Row block sizes for variable patterns.
            - block_col_sz (torch.Tensor or None): Column block sizes for variable patterns.
        """
        if test_type == "uniform":
            assert (
                block_size is not None
            ), "`block_size` is required for 'uniform' test type."

            q_block_size, k_block_size = block_size
            num_q_blocks = seqlen // q_block_size
            num_kv_blocks = seqlen // k_block_size
            block_mask, _ = generate_block_sparse_pattern(
                num_q_heads=num_heads_q,
                num_kv_heads=num_heads_kv,
                num_q_blocks=num_q_blocks,
                num_kv_blocks=num_kv_blocks,
                sparsity=sparsity_ratio,
                mode=sparsity_granularity,
                sparse_format=sparse_format,
                device="cuda",
            )
            return block_mask, block_size, None, None
        elif test_type == "variable":
            assert (
                average_block_size is not None
            ), "`average_block_size` is required for 'variable' test type."
            assert (
                min_block_size is not None
            ), "`min_block_size` is required for 'variable' test type."

            q_avg_block_size, k_avg_block_size = average_block_size
            min_q_block_size, min_k_block_size = min_block_size
            num_q_blocks = seqlen // q_avg_block_size
            num_kv_blocks = seqlen // k_avg_block_size
            (
                block_mask,
                block_row_sz,
                block_col_sz,
            ) = generate_variable_block_sparse_pattern(
                num_q_heads=num_heads_q,
                num_kv_heads=num_heads_kv,
                seqlen_q=seqlen,
                seqlen_k=seqlen,
                num_q_blocks=num_q_blocks,
                num_kv_blocks=num_kv_blocks,
                min_q_block_size=min_q_block_size,
                min_kv_block_size=min_k_block_size,
                sparsity=sparsity_ratio,
                mode=sparsity_granularity,
                device="cuda",
            )
            return block_mask, average_block_size, block_row_sz, block_col_sz
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

    @pytest.mark.slow
    @with_run_in_mp
    @parameterize(
        "model_config",
        [
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
            # {
            #    "name": "mha_nh1_hd64",
            #    "num_heads_q": 1,
            #    "num_heads_kv": 1,
            #    "head_dim": 64,
            # },
            # {
            #    "name": "gqa_nhq4_nhkv2_hd64",
            #    "num_heads_q": 4,
            #    "num_heads_kv": 2,
            #    "head_dim": 64,
            # },
        ],
    )
    @parameterize("seqlen", [2048])
    @parameterize(
        "block_config",
        [
            # Uniform blocks with same Q/K block size
            {
                "type": "uniform",
                "q_size": 64,
                "k_size": 64,
                "swap_ab": False,
                "ref_block_size": (64, 64),
            },
            {
                "type": "uniform",
                "q_size": 128,
                "k_size": 128,
                "swap_ab": False,
                "ref_block_size": (128, 128),
            },
            {
                "type": "uniform",
                "q_size": 64,
                "k_size": 64,
                "swap_ab": True,
                "ref_block_size": (64, 64),
            },
            {
                "type": "uniform",
                "q_size": 128,
                "k_size": 128,
                "swap_ab": True,
                "ref_block_size": (64, 64),
            },
            # Small Q block sizes
            {
                "type": "uniform",
                "q_size": 32,
                "k_size": 64,
                "swap_ab": True,
                "ref_block_size": (32, 64),
            },
            {
                "type": "uniform",
                "q_size": 16,
                "k_size": 64,
                "swap_ab": False,
                "ref_block_size": (64, 64),
            },
            # Small K block sizes
            {
                "type": "uniform",
                "q_size": 64,
                "k_size": 8,
                "swap_ab": False,
                "ref_block_size": (64, 64),
            },
            # Variable blocks
            {
                "type": "variable",
                "q_size": 64,
                "k_size": 64,
                "min_q_size": 16,
                "min_k_size": 16,
            },
            {
                "type": "variable",
                "q_size": 128,
                "k_size": 128,
                "min_q_size": 16,
                "min_k_size": 16,
            },
        ],
    )
    @parameterize("sparsity_ratio", [0.1, 0.5, 1.0])
    @parameterize("sparsity_granularity", ["per_kv_head"])
    @parameterize("sparse_format", ["block_mask", "topk"])
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("attn_type", [0])  # For now, we only test full mask for block sparse.
    @parameterize("pack_gqa", [False, True])
    @parameterize("sparse_load", [False, True])
    @parameterize(
        "deterministic", [False]
    )  # we do not support deterministic now if auto_rangemerge is true
    @parameterize("test_accumulation_inplace", [False])
    def test_block_sparse_attn(
        self,
        model_config: dict[str, Any],
        seqlen: int,
        block_config,
        sparsity_ratio: float,
        sparsity_granularity: str,
        sparse_format: str,
        dtype: torch.dtype,
        attn_type: int,
        pack_gqa: bool,
        sparse_load: bool,
        deterministic: bool,
        test_accumulation_inplace: bool,
    ):
        auto_range_merge = True
        # FIXME: auto_range_merge and deterministic can't be True at the same time
        if auto_range_merge and deterministic:
            return

        test_type = block_config["type"]
        if test_type == "variable" and sparse_format == "topk":
            # for variable block sparse pattern, it can't be described by topk indices data structure
            return

        # FIXME: support packGQA in sparse load setting
        if pack_gqa and sparse_load:
            return

        q_block_size = block_config["q_size"]
        k_block_size = block_config["k_size"]

        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        swap_ab = block_config.get("swap_ab", False)
        ref_block_size = block_config.get("ref_block_size", None)

        # swap_ab and sparse_load can't be True at the same time
        # since they target different settings
        if swap_ab and sparse_load:
            return

        # Prepare inputs
        if test_type == "uniform":
            block_size = (q_block_size, k_block_size)
            average_block_size = None
            min_block_size = None
        else:  # variable
            block_size = None
            average_block_size = (q_block_size, k_block_size)
            min_q_block_size = block_config["min_q_size"]
            min_k_block_size = block_config["min_k_size"]
            min_block_size = (min_q_block_size, min_k_block_size)

        # Generate the appropriate sparse pattern using the helper
        (
            block_mask,
            block_sizes,
            block_row_sz,
            block_col_sz,
        ) = self._generate_sparse_pattern(
            test_type=test_type,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            seqlen=seqlen,
            sparsity_ratio=sparsity_ratio,
            sparsity_granularity=sparsity_granularity,
            sparse_format=sparse_format,
            block_size=block_size,
            average_block_size=average_block_size,
            min_block_size=min_block_size,
        )

        # Construct a descriptive test case name
        q_bs, k_bs = block_sizes
        block_info = (
            f"block_size=({q_bs},{k_bs})"
            if test_type == "uniform"
            else f"avg_block_size=({q_bs},{k_bs})"
        )
        test_case = (
            f"[RANK {self.rank}][test_block_sparse_attn]"
            f"[{model_config['name']}]"
            f"[{test_type}]"
            f"[{block_info}]"
            f"[swap_ab={swap_ab}]"
            f"[ref_block_size={ref_block_size}]"
            f"[sparsity_granularity={sparsity_granularity}]"
            f"[sparsity_ratio={sparsity_ratio}]"
            f"[sparse_format={sparse_format}]"
            f"[dtype={dtype}]"
            f"[attn_type={attn_type}]"
            f"[pack_gqa={pack_gqa}]"
            f"[sparse_load={sparse_load}]"
            f"[auto_range_merge={auto_range_merge}]"
            f"[deterministic={deterministic}]"
            f"[test_accumulation_inplace={test_accumulation_inplace}]"
        )
        print(f"[RANK {self.rank}]: {test_case=}")
        # ----- Construct q, k, vdata ----- #
        q = torch.randn(
            (1, seqlen, num_heads_q, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (1, seqlen, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (1, seqlen, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)

        # we may custom set this dict in feature.
        # err_ratio_dict = {
        # "dq_min_mismatch_thres": 5e-3,
        # }

        # Execute the test and assertions
        self.assert_close_to_torch_ref(
            dtype=dtype,
            q=q,
            k=k,
            v=v,
            grad_output=do,
            seqlen=seqlen,
            block_size=block_sizes,
            block_mask=block_mask,
            head_wise=sparsity_granularity,
            sparse_format=sparse_format,
            nhq=num_heads_q,
            nhk=num_heads_kv,
            pack_gqa=pack_gqa,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            sparse_load=sparse_load,
            test_case=test_case,
            sparsity_ratio=sparsity_ratio,
            uniform=(test_type == "uniform"),
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            err_ratio_dict={},
        )

    # NOTE: this simple test is for github ci.
    @with_run_in_mp
    @parameterize(
        "model_config",
        [
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
        ],
    )
    @parameterize("seqlen", [2048])
    @parameterize(
        "block_config",
        [
            # Uniform blocks with same Q/K block size
            {
                "type": "uniform",
                "q_size": 64,
                "k_size": 64,
                "swap_ab": True,
                "ref_block_size": (64, 64),
            },
            {
                "type": "uniform",
                "q_size": 128,
                "k_size": 128,
                "swap_ab": True,
                "ref_block_size": (64, 64),
            },
            # Small Q block sizes
            {
                "type": "uniform",
                "q_size": 32,
                "k_size": 64,
                "swap_ab": True,
                "ref_block_size": (32, 64),
            },
            # Small K block sizes
            {
                "type": "uniform",
                "q_size": 64,
                "k_size": 8,
                "swap_ab": True,
                "ref_block_size": (64, 64),
            },
        ],
    )
    @parameterize("sparsity_ratio", [0.1, 0.5, 1.0])
    @parameterize("sparsity_granularity", ["per_kv_head"])
    @parameterize("sparse_format", ["block_mask", "topk"])
    @parameterize("dtype", [torch.bfloat16])
    @parameterize("attn_type", [0])  # For now, we only test full mask for block sparse.
    @parameterize("pack_gqa", [True])
    @parameterize("sparse_load", [False, True])
    @parameterize(
        "deterministic", [False]
    )  # we do not support deterministic now if auto_rangemerge is true
    @parameterize("test_accumulation_inplace", [False])
    def test_simple_block_sparse_attn(
        self,
        model_config: dict[str, Any],
        seqlen: int,
        block_config,
        sparsity_ratio: float,
        sparsity_granularity: str,
        sparse_format: str,
        dtype: torch.dtype,
        attn_type: int,
        pack_gqa: bool,
        sparse_load: bool,
        deterministic: bool,
        test_accumulation_inplace: bool,
    ):
        auto_range_merge = True
        # FIXME: auto_range_merge and deterministic can't be True at the same time
        if auto_range_merge and deterministic:
            return
        # FIXME: support packGQA in sparse load setting
        if pack_gqa and sparse_load:
            return

        test_type = block_config["type"]
        q_block_size = block_config["q_size"]
        k_block_size = block_config["k_size"]

        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        swap_ab = block_config.get("swap_ab", False)
        ref_block_size = block_config.get("ref_block_size", None)

        # swap_ab and sparse_load can't be True at the same time
        # since they target different settings
        if swap_ab and sparse_load:
            return

        # Prepare inputs
        if test_type == "uniform":
            block_size = (q_block_size, k_block_size)
            average_block_size = None
            min_block_size = None
        else:  # variable
            block_size = None
            average_block_size = (q_block_size, k_block_size)
            min_q_block_size = block_config["min_q_size"]
            min_k_block_size = block_config["min_k_size"]
            min_block_size = (min_q_block_size, min_k_block_size)

        # Generate the appropriate sparse pattern using the helper
        (
            block_mask,
            block_sizes,
            block_row_sz,
            block_col_sz,
        ) = self._generate_sparse_pattern(
            test_type=test_type,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            seqlen=seqlen,
            sparsity_ratio=sparsity_ratio,
            sparsity_granularity=sparsity_granularity,
            sparse_format=sparse_format,
            block_size=block_size,
            average_block_size=average_block_size,
            min_block_size=min_block_size,
        )

        # Construct a descriptive test case name
        q_bs, k_bs = block_sizes
        block_info = (
            f"block_size=({q_bs},{k_bs})"
            if test_type == "uniform"
            else f"avg_block_size=({q_bs},{k_bs})"
        )
        test_case = (
            f"[{model_config['name']}]"
            f"[{test_type}]"
            f"[{block_info}]"
            f"[swap_ab={swap_ab}]"
            f"[ref_block_size={ref_block_size}]"
            f"[sparsity_granularity={sparsity_granularity}]"
            f"[sparsity_ratio={sparsity_ratio}]"
            f"[sparse_format={sparse_format}]"
            f"[dtype={dtype}]"
            f"[attn_type={attn_type}]"
            f"[pack_gqa={pack_gqa}]"
            f"[sparse_load={sparse_load}]"
            f"[auto_range_merge={auto_range_merge}]"
            f"[deterministic={deterministic}]"
            f"[test_accumulation_inplace={test_accumulation_inplace}]"
        )

        # ----- Construct q, k, vdata ----- #
        q = torch.randn(
            (1, seqlen, num_heads_q, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (1, seqlen, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (1, seqlen, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)

        # we may custom set this dict in feature.
        # err_ratio_dict = {
        # "dq_min_mismatch_thres": 5e-3,
        # }

        # Execute the test and assertions
        self.assert_close_to_torch_ref(
            dtype=dtype,
            q=q,
            k=k,
            v=v,
            grad_output=do,
            seqlen=seqlen,
            block_size=block_sizes,
            block_mask=block_mask,
            head_wise=sparsity_granularity,
            sparse_format=sparse_format,
            nhq=num_heads_q,
            nhk=num_heads_kv,
            pack_gqa=pack_gqa,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            sparse_load=sparse_load,
            test_case=test_case,
            sparsity_ratio=sparsity_ratio,
            uniform=(test_type == "uniform"),
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            err_ratio_dict={},
        )


if __name__ == "__main__":
    run_tests()
