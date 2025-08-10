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


import unittest
from typing import Any
from unittest import TestCase

import torch
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention as sdpa_func

from magi_attention.functional import flex_flash_attn_func
from magi_attention.functional.dist_attn import result_correction
from magi_attention.functional.flex_flash_attn import (
    _flex_flash_attn_backward,
    _flex_flash_attn_forward,
    merge_ranges,
)
from magi_attention.testing import parameterize
from magi_attention.testing.precision import assert_close, calc_inf_norm
from magi_attention.utils import (
    flatten_head_mask,
    flatten_kvhead_mask,
    generate_headwise_4D_block_sparse_pattern,
    generate_kv_headwise_4D_block_sparse_pattern,
    generate_ranges_from_block_mask,
    generate_ranges_from_var_block_mask,
    get_random_variable_block_mask,
    get_sdpa_mask_from_block_sparse_mask,
    get_sdpa_mask_from_var_block_mask,
    is_list_value_any,
)


class TestFlexFlashAttn(TestCase):
    @property
    def seed(self):
        return 42

    @property
    def device(self):
        return torch.cuda.current_device()

    def setUp(self):
        torch.manual_seed(self.seed)

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
        deterministic,
        test_accumulation_inplace,
        test_case,
        err_msg_list,
        uniform=True,
        block_row_sz=None,
        block_col_sz=None,
    ):
        s, h = q.size(1), q.size(2)
        q = rearrange(
            q, "b s h d -> (b h s) 1 d"
        )  # flatten as (head dimension, seq dimension)

        assert nhq % nhk == 0

        repeats = nhq // nhk
        if head_wise == "q":
            k = torch.repeat_interleave(
                k, repeats=repeats, dim=2
            )  # we need to flatten k, v along head dimension for GQA setting.
            v = torch.repeat_interleave(v, repeats=repeats, dim=2)

        k = rearrange(k, "b s h d -> (b h s) 1 d")
        v = rearrange(v, "b s h d -> (b h s) 1 d")

        if head_wise == "q":
            flat_block_sparse_mask = flatten_head_mask(block_mask)
        else:
            flat_block_sparse_mask = flatten_kvhead_mask(block_mask, nhq, nhk)

        if uniform:
            q_ranges_tensor, k_ranges_tensor = generate_ranges_from_block_mask(
                flat_block_sparse_mask, block_size, block_size
            )
        else:
            q_ranges_tensor, k_ranges_tensor = generate_ranges_from_var_block_mask(
                flat_block_sparse_mask, block_row_sz, block_col_sz
            )

        attn_type_map = torch.zeros(
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
                max_seqlen_q=block_size,
                max_seqlen_k=block_size,
                attn_type_map_tensor=attn_type_map,
                auto_range_merge=True,
                deterministic=deterministic,
                test_case=test_case,
            )

        q.grad = None
        k.grad = None
        v.grad = None
        """

        o, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map,
            max_seqlen_q=block_size,
            max_seqlen_k=block_size,
            auto_range_merge=True,  # we should enable auto_range_merge for block sparse mask.
        )

        o = rearrange(o, "(b h s) 1 d -> b s h d", b=1, s=s, h=h)
        o.backward(grad_output)

        return o

    def get_sdpa_attn_ref(
        self,
        q,
        k,
        v,
        grad_output,
        seqlen,
        block_size,
        block_mask,
        uniform=True,
        block_row_sz=None,
        block_col_sz=None,
        high_precision=False,
    ):
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        if uniform:
            sdpa_mask_4d = get_sdpa_mask_from_block_sparse_mask(
                block_mask, seqlen, seqlen, block_size, block_size
            )
        else:
            sdpa_mask_4d = get_sdpa_mask_from_var_block_mask(
                block_mask, seqlen, seqlen, block_row_sz, block_col_sz
            )

        if high_precision:
            o = sdpa_func(
                q.to(torch.float64),
                k.to(torch.float64),
                v.to(torch.float64),
                attn_mask=sdpa_mask_4d,
                is_causal=False,
                enable_gqa=True,
            )
        else:
            o = sdpa_func(
                q,
                k,
                v,
                attn_mask=sdpa_mask_4d,
                is_causal=False,
                enable_gqa=True,
            )

        o = rearrange(o, "b h s d -> b s h d")
        o = o.to(q.dtype)
        o.backward(grad_output)

        return o

    def assert_close_to_torch_ref(
        self,
        q,
        k,
        v,
        grad_output,
        seqlen,
        block_size,
        block_mask,
        head_wise,
        nhq,
        nhk,
        deterministic,
        test_accumulation_inplace,
        test_case,
        uniform=True,
        block_row_sz=None,
        block_col_sz=None,
    ):
        high_precision_torch_out_ref = self.get_sdpa_attn_ref(
            q,
            k,
            v,
            grad_output,
            seqlen,
            block_size,
            block_mask,
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

        low_precision_torch_out_ref = self.get_sdpa_attn_ref(
            q,
            k,
            v,
            grad_output,
            seqlen,
            block_size,
            block_mask,
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

        err_msg_list = []
        ffa_out = self.get_ffa_result(
            q,
            k,
            v,
            grad_output,
            block_mask,
            head_wise,
            block_size,
            nhq,
            nhk,
            deterministic,
            test_accumulation_inplace,
            test_case,
            err_msg_list,
            uniform=uniform,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
        )
        ffa_dq, ffa_dk, ffa_dv = q.grad, k.grad, v.grad

        norm_rtol_ratio = 2.0

        out_norm = calc_inf_norm(ffa_out, high_precision_torch_out_ref)
        out_ref_norm = calc_inf_norm(
            low_precision_torch_out_ref, high_precision_torch_out_ref
        )

        try:
            self.assertLessEqual(
                out_norm,
                norm_rtol_ratio * out_ref_norm,
                msg=f"For {test_case=}: {out_norm=} should be no greater than {norm_rtol_ratio}x of {out_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        dq_norm = calc_inf_norm(ffa_dq, high_precision_dq_ref)
        dq_ref_norm = calc_inf_norm(low_precision_dq_ref, high_precision_dq_ref)

        try:
            self.assertLessEqual(
                dq_norm,
                norm_rtol_ratio * dq_ref_norm,
                msg=f"For {test_case=}: {dq_norm=} should be no greater than {norm_rtol_ratio}x of {dq_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        dk_norm = calc_inf_norm(ffa_dk, high_precision_dk_ref)
        dk_ref_norm = calc_inf_norm(low_precision_dk_ref, high_precision_dk_ref)

        try:
            self.assertLessEqual(
                dk_norm,
                norm_rtol_ratio * dk_ref_norm,
                msg=f"For {test_case=}: {dk_norm=} should be no greater than {norm_rtol_ratio}x of {dk_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        dv_norm = calc_inf_norm(ffa_dv, high_precision_dv_ref)
        dv_ref_norm = calc_inf_norm(low_precision_dv_ref, high_precision_dv_ref)

        try:
            self.assertLessEqual(
                dv_norm,
                norm_rtol_ratio * dv_ref_norm,
                msg=f"For {test_case=}: {dv_norm=} should be no greater than {norm_rtol_ratio}x of {dv_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

    def calc_inf_norm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        return (a.float() - b.float()).norm(p=float("inf")).item()

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
        ],
    )
    @parameterize("seqlen", [2048])
    @parameterize("block_size", [64, 128])
    @parameterize("sparsity_ratio", [0.1, 0.5, 1.0])
    @parameterize(
        "sparsity_granularity", ["per_query_head", "per_kv_head"]
    )  # generate sparse attn per query head or kv head.
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("attn_type", [0])  # for now, we only test full mask.
    @parameterize(
        "auto_range_merge", [True]
    )  # for sparse attn, we set auto_range_merge to True by default
    @parameterize("deterministic", [False, True])
    @parameterize("test_accumulation_inplace", [False, True])
    def test_block_sparse_attn(
        self,
        model_config: dict[str, Any],
        seqlen: int,
        block_size: int,
        sparsity_ratio: float,
        sparsity_granularity: str,
        dtype: torch.dtype,
        attn_type: int,
        auto_range_merge: bool,
        deterministic: bool,
        test_accumulation_inplace: bool,
    ):
        # FIXME: auto_range_merge and deterministic can't be True at the same time
        # due to some unresolved bug to be fixed as soon as possible
        if auto_range_merge and deterministic:
            return

        # we test per query head and per kvhead both.
        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]

        num_q_blocks = seqlen // block_size
        num_kv_blocks = seqlen // block_size

        # --- generate block mask and q, k range --- #
        if sparsity_granularity == "per_query_head":
            block_mask, scores = generate_headwise_4D_block_sparse_pattern(
                num_heads_q, num_q_blocks, num_kv_blocks, sparsity_ratio, device="cuda"
            )

            flat_block_sparse_mask = flatten_head_mask(block_mask)
            q_ranges_tensor, k_ranges_tensor = generate_ranges_from_block_mask(
                flat_block_sparse_mask, block_size, block_size
            )

        else:
            block_mask, scores = generate_kv_headwise_4D_block_sparse_pattern(
                num_heads_kv, num_q_blocks, num_kv_blocks, sparsity_ratio, device="cuda"
            )
            num_groups = num_heads_q // num_heads_kv
            block_mask = block_mask.repeat_interleave(num_groups, dim=1)

            flat_block_sparse_mask = flatten_kvhead_mask(
                block_mask, num_heads_q, num_heads_kv
            )
            q_ranges_tensor, k_ranges_tensor = generate_ranges_from_block_mask(
                flat_block_sparse_mask, block_size, block_size
            )

        attn_type_map = [attn_type] * len(q_ranges_tensor)

        assert len(q_ranges_tensor) == len(k_ranges_tensor) == len(attn_type_map), (
            "q_ranges, k_ranges and attn_type_map should have the same length"
            f", but got {len(q_ranges_tensor)=}, {len(k_ranges_tensor)=}, {len(attn_type_map)=}"
        )

        test_case = (
            f"[{model_config['name']}]"
            f"[block_size={block_size}]"
            f"[q_range={q_ranges_tensor}]"
            f"[k_range={k_ranges_tensor}]"
            f"[sparsity_granularity={sparsity_granularity}]"
            f"[dtype={dtype}]"
            f"[attn_type_map={attn_type_map}]"
            f"[auto_range_merge={auto_range_merge}]"
            f"[deterministic={deterministic}]"
            f"[test_accumulation_inplace={test_accumulation_inplace}"
        )

        print(f"{test_case=}")
        # we assume q and k has the same seqlen.
        total_seqlen_q = seqlen
        total_seqlen_k = seqlen
        # FIXME: for square bi-causal mask, i.e. when only the main diagonal is valid
        # ffa bwd kernel encounters with some precision issue with dq/dk,
        # thus we skip here and will fix it asap
        if is_list_value_any(attn_type_map, 3):
            return

        # construct data
        q = torch.randn(
            (1, total_seqlen_q, num_heads_q, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (1, total_seqlen_k, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (1, total_seqlen_k, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)

        self.assert_close_to_torch_ref(
            q,
            k,
            v,
            do,
            seqlen,
            block_size,
            block_mask,
            sparsity_granularity,
            num_heads_q,
            num_heads_kv,
            deterministic,
            test_accumulation_inplace,
            test_case,
        )

    @parameterize(
        "model_config",
        [
            {
                "name": "mha_nh8_hd128",
                "num_heads_q": 8,
                "num_heads_kv": 8,
                "head_dim": 128,
            },
            # {
            #    "name": "gqa_nhq16_nhkv4_hd128",
            #    "num_heads_q": 16,
            #    "num_heads_kv": 4,
            #    "head_dim": 128,
            # },
            {
                "name": "mha_nh1_hd64",
                "num_heads_q": 1,
                "num_heads_kv": 1,
                "head_dim": 64,
            },
            # {
            #    "name": "gqa_nhq4_nhkv2_hd64",
            #    "num_heads_q": 4,
            #    "num_heads_kv": 2,
            #    "head_dim": 64,
            # },
        ],
    )
    @parameterize("seqlen", [2048])
    @parameterize("average_block_size", [64])
    @parameterize("min_block_size", [32])
    @parameterize("max_block_size", [128])
    @parameterize("sparsity_ratio", [0.1, 0.5, 1.0])
    @parameterize(
        "sparsity_granularity", ["per_query_head", "per_kv_head"]
    )  # generate sparse attn per query head or kv head.
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("attn_type", [0])  # for now, we only test full mask.
    @parameterize(
        "auto_range_merge", [True]
    )  # for sparse attn, we set auto_range_merge to True by default
    @parameterize("deterministic", [False, True])
    @parameterize("test_accumulation_inplace", [False, True])
    def test_var_block_sparse_attn(
        self,
        model_config: dict[str, Any],
        seqlen: int,
        average_block_size: int,
        min_block_size: int,
        max_block_size: int,
        sparsity_ratio: float,
        sparsity_granularity: str,
        dtype: torch.dtype,
        attn_type: int,
        auto_range_merge: bool,
        deterministic: bool,
        test_accumulation_inplace: bool,
    ):
        # FIXME: auto_range_merge and deterministic can't be True at the same time
        # due to some unresolved bug to be fixed as soon as possible
        if auto_range_merge and deterministic:
            return

        if sparsity_granularity == "per_kv_head":
            return

        # we test per query head and per kvhead both.
        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]

        num_q_blocks = seqlen // average_block_size
        num_kv_blocks = seqlen // average_block_size

        # --- generate block mask and q, k range --- #
        if sparsity_granularity == "per_query_head":
            block_mask, block_row_sz, block_col_sz = get_random_variable_block_mask(
                seqlen,
                seqlen,
                num_q_blocks,
                num_kv_blocks,
                num_heads_q,
                min_q_block_size=min_block_size,
                min_kv_block_size=min_block_size,
                sparsity_ratio=sparsity_ratio,
                bsz=1,
                device="cuda",
            )

            flat_block_sparse_mask = flatten_head_mask(block_mask)
            q_ranges_tensor, k_ranges_tensor = generate_ranges_from_var_block_mask(
                flat_block_sparse_mask, block_row_sz, block_col_sz
            )

        else:
            block_mask, scores = generate_kv_headwise_4D_block_sparse_pattern(
                num_heads_kv, num_q_blocks, num_kv_blocks, sparsity_ratio, device="cuda"
            )
            num_groups = num_heads_q // num_heads_kv
            block_mask = block_mask.repeat_interleave(num_groups, dim=1)

            flat_block_sparse_mask = flatten_kvhead_mask(
                block_mask, num_heads_q, num_heads_kv
            )
            q_ranges_tensor, k_ranges_tensor = generate_ranges_from_block_mask(
                flat_block_sparse_mask, average_block_size, average_block_size
            )

        attn_type_map = [attn_type] * len(q_ranges_tensor)

        assert len(q_ranges_tensor) == len(k_ranges_tensor) == len(attn_type_map), (
            "q_ranges, k_ranges and attn_type_map should have the same length"
            f", but got {len(q_ranges_tensor)=}, {len(k_ranges_tensor)=}, {len(attn_type_map)=}"
        )

        test_case = (
            f"[{model_config['name']}]"
            f"[average_block_size={average_block_size}]"
            f"[q_range={q_ranges_tensor}]"
            f"[k_range={k_ranges_tensor}]"
            f"[sparsity_granularity={sparsity_granularity}]"
            f"[dtype={dtype}]"
            f"[attn_type_map={attn_type_map}]"
            f"[auto_range_merge={auto_range_merge}]"
            f"[deterministic={deterministic}]"
            f"[test_accumulation_inplace={test_accumulation_inplace}"
        )

        print(f"{test_case=}")
        # we assume q and k has the same seqlen.
        total_seqlen_q = seqlen
        total_seqlen_k = seqlen
        # FIXME: for square bi-causal mask, i.e. when only the main diagonal is valid
        # ffa bwd kernel encounters with some precision issue with dq/dk,
        # thus we skip here and will fix it asap
        if is_list_value_any(attn_type_map, 3):
            return

        # construct data
        q = torch.randn(
            (1, total_seqlen_q, num_heads_q, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (1, total_seqlen_k, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (1, total_seqlen_k, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)

        self.assert_close_to_torch_ref(
            q,
            k,
            v,
            do,
            seqlen,
            average_block_size,
            block_mask,
            sparsity_granularity,
            num_heads_q,
            num_heads_kv,
            deterministic,
            test_accumulation_inplace,
            test_case,
            uniform=False,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
        )


if __name__ == "__main__":
    unittest.main()
