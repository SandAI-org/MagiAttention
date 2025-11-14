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

from typing import Any, Optional, Tuple

import pytest
import torch
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention as sdpa_func
from torch.testing._internal.common_utils import run_tests

from magi_attention.functional import flex_flash_attn_func
from magi_attention.functional.flex_flash_attn import (
    _flex_flash_attn_backward,
    _flex_flash_attn_forward,
    merge_ranges,
)
from magi_attention.functional.utils import correct_attn_fwd_result
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import assert_close, calc_inf_norm
from magi_attention.testing.utils import switch_ffa_verbose_jit_build_decorator
from magi_attention.utils.sparse_utils import (
    choose_ref_block,
    flatten_block_mask,
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask,
    generate_ranges_from_var_block_mask,
    generate_variable_block_sparse_pattern,
    get_sdpa_mask_from_block_sparse_mask,
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
        return 600  # Increase timeout for JIT compilation

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
            None,  # sink
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
        # (Implementation is identical to the original)
        s, h = q.size(1), q.size(2)
        q = rearrange(q, "b s h d -> (b h s) 1 d")
        assert nhq % nhk == 0

        repeats = nhq // nhk
        if head_wise == "q":
            k = torch.repeat_interleave(k, repeats=repeats, dim=2)
            v = torch.repeat_interleave(v, repeats=repeats, dim=2)

        k = rearrange(k, "b s h d -> (b h s) 1 d")
        v = rearrange(v, "b s h d -> (b h s) 1 d")
        q.retain_grad()
        k.retain_grad()
        v.retain_grad()
        q.grad, k.grad, v.grad = None, None, None

        flat_block_sparse_mask = flatten_block_mask(block_mask, nhq, nhk)

        if uniform:
            q_block_size, k_block_size = block_size
            q_ranges_tensor, k_ranges_tensor = generate_ranges_from_block_mask(
                flat_block_sparse_mask, q_block_size, k_block_size
            )
        else:
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

        ref_block_size = choose_ref_block(block_size)
        o, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map_tensor,
            auto_range_merge=True,
            ref_block_size=ref_block_size,
        )

        o = rearrange(o, "(b h s) 1 d -> b s h d", b=1, s=s, h=h)
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
        # (Implementation is identical to the original)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")
        if uniform:
            q_block_size, k_block_size = block_size
            sdpa_mask_4d = get_sdpa_mask_from_block_sparse_mask(
                block_mask, seqlen, seqlen, q_block_size, k_block_size
            )
        else:
            sdpa_mask_4d = get_sdpa_mask_from_var_block_mask(
                block_mask, seqlen, seqlen, block_row_sz, block_col_sz
            )

        o_tensor = q.to(torch.float64) if high_precision else q
        k_tensor = k.to(torch.float64) if high_precision else k
        v_tensor = v.to(torch.float64) if high_precision else v

        o = sdpa_func(
            o_tensor,
            k_tensor,
            v_tensor,
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
        # (Implementation is identical to the original)
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

    def _generate_sparse_pattern(
        self,
        test_type: str,
        num_heads_q: int,
        num_heads_kv: int,
        seqlen: int,
        sparsity_ratio: float,
        sparsity_granularity: str,
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

    @switch_ffa_verbose_jit_build_decorator
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
    @parameterize(
        "block_config",
        [
            # Uniform blocks with same Q/K block size
            {"type": "uniform", "q_size": 64, "k_size": 64},
            {"type": "uniform", "q_size": 128, "k_size": 128},
            # Small Q block sizes
            {"type": "uniform", "q_size": 32, "k_size": 64},
            {"type": "uniform", "q_size": 16, "k_size": 64},
            {"type": "uniform", "q_size": 8, "k_size": 64},
            # Small K block sizes
            {"type": "uniform", "q_size": 64, "k_size": 32},
            {"type": "uniform", "q_size": 64, "k_size": 16},
            {"type": "uniform", "q_size": 64, "k_size": 8},
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
    @parameterize("sparsity_granularity", ["per_q_head", "per_kv_head"])
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("attn_type", [0])  # For now, we only test full mask.
    @parameterize("deterministic", [True, False])
    @parameterize("test_accumulation_inplace", [False])
    def test_block_sparse_attn(
        self,
        model_config: dict[str, Any],
        seqlen: int,
        block_config,
        sparsity_ratio: float,
        sparsity_granularity: str,
        dtype: torch.dtype,
        attn_type: int,
        deterministic: bool,
        test_accumulation_inplace: bool,
    ):
        auto_range_merge = True
        # FIXME: auto_range_merge and deterministic can't be True at the same time
        if auto_range_merge and deterministic:
            return

        test_type = block_config["type"]
        q_block_size = block_config["q_size"]
        k_block_size = block_config["k_size"]

        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]

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
            f"[sparsity_granularity={sparsity_granularity}]"
            f"[dtype={dtype}]"
            f"[attn_type={attn_type}]"
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

        # Execute the test and assertions
        self.assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            grad_output=do,
            seqlen=seqlen,
            block_size=block_sizes,
            block_mask=block_mask,
            head_wise=sparsity_granularity,
            nhq=num_heads_q,
            nhk=num_heads_kv,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            test_case=test_case,
            uniform=(test_type == "uniform"),
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
        )


if __name__ == "__main__":
    run_tests()
