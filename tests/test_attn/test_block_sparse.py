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

import os
import time
import unittest
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import (
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.utils.sparse_utils import (
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask_triton,
    get_sdpa_mask_from_block_sparse_mask,
)


class _BlockSparseTestHelper(unittest.TestCase):
    @property
    def device(self):
        return torch.cuda.current_device()

    def check_deterministic(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        q_ranges_tensor,
        k_ranges_tensor,
        attn_type_map_tensor,
        range_merge,
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
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            max_seqlen_q=None,
            attn_type_map=attn_type_map_tensor,
            range_merge=range_merge,
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
        swap_ab,
        ref_block_size,
        block_sparse,
        swap_bwd_qk_loop,
        test_case,
        err_msg_list,
        sparse_format="block_mask",
        uniform=True,
        block_row_sz=None,
        block_col_sz=None,
        max_seqlen_q=None,
    ):
        s = q.size(1)
        h1 = k.size(2)
        q = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=h1)
        assert nhq % nhk == 0
        # flatten kv head.
        k = rearrange(k, "b s h d -> (b h s) 1 d")
        v = rearrange(v, "b s h d -> (b h s) 1 d")
        q.retain_grad()
        k.retain_grad()
        v.retain_grad()
        q.grad, k.grad, v.grad = None, None, None
        q_block_size, sparse_k_block_size = block_size
        (
            q_ranges_tensor,
            k_ranges_tensor,
        ) = generate_ranges_from_block_mask_triton(
            block_mask, q_block_size, sparse_k_block_size
        )
        attn_type_map_tensor = torch.zeros(
            len(q_ranges_tensor), dtype=torch.int32, device="cuda"
        )

        o, meta = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            max_seqlen_q=max_seqlen_q,
            attn_type_map=attn_type_map_tensor,
            range_merge=True,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            block_sparse=block_sparse,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
        )
        torch.cuda.synchronize()

        lse = meta.lse
        o = rearrange(o, "(b h1 s) h2 d -> b s (h1 h2) d", b=1, s=s, h1=h1)
        lse = rearrange(lse, "(h1 s) h2 -> s (h1 h2)", s=s, h1=h1)
        o.backward(grad_output)
        torch.cuda.synchronize()

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
                    range_merge=True,
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
        q = rearrange(q, "1 s h d -> s h d")  # shd
        k = rearrange(k, "1 s h d -> s h d")
        v = rearrange(v, "1 s h d -> s h d")
        q_block_size, sparse_k_block_size = block_size
        sdpa_mask_4d = get_sdpa_mask_from_block_sparse_mask(
            block_mask, seqlen, seqlen, q_block_size, sparse_k_block_size, q.size(1)
        )
        sdpa_mask = rearrange(
            sdpa_mask_4d, "1 h seqlen_q seqlen_k -> h seqlen_q seqlen_k"
        )
        o, meta = ref_attn_func(
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
        lse = meta.lse
        torch.cuda.synchronize()

        o = rearrange(o, "s h d -> 1 s h d")
        lse = rearrange(lse, "1 seqlen h -> seqlen h")
        o.backward(grad_output)
        torch.cuda.synchronize()

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
        swap_ab: bool,
        ref_block_size: tuple[int, int],
        block_sparse,
        swap_bwd_qk_loop,
        test_case,
        sparsity_ratio,
        uniform=True,
        block_row_sz=None,
        block_col_sz=None,
        err_ratio_dict: dict[str, float] = {},
        max_seqlen_q=None,
    ):
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
            swap_ab,
            ref_block_size,
            block_sparse,
            swap_bwd_qk_loop,
            test_case,
            err_msg_list,
            sparse_format=sparse_format,
            uniform=uniform,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            max_seqlen_q=max_seqlen_q,
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
    ) -> Tuple[
        torch.Tensor, Tuple[int, int], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Helper function to generate uniform block sparse patterns.

        Returns:
            A tuple containing:
            - block_mask (torch.Tensor): The generated sparse mask.
            - block_sizes (Tuple[int, int]): Q block size and K/V block size.
            - block_row_sz (None): Reserved for compatibility.
            - block_col_sz (None): Reserved for compatibility.
        """
        if test_type == "uniform":
            assert (
                block_size is not None
            ), "`block_size` is required for 'uniform' test type."

            q_block_size, sparse_k_block_size = block_size
            num_q_blocks = seqlen // q_block_size
            num_kv_blocks = seqlen // sparse_k_block_size
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
        else:
            raise ValueError(f"Unknown test_type: {test_type}")


class TestBlockSparseSimple(unittest.TestCase):
    """GQA LoopQ + pack_gqa=False block-sparse tests.

    These exercise the InnerLoopQ path with GQA where pack_gqa is NOT
    auto-enabled — a supported edge case under active development.
    """

    @property
    def device(self):
        return torch.cuda.current_device()

    @classmethod
    def precompile_kernel_specs(cls):
        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}
        for cfg in cls.LOOPQ_GQA_CONFIGS:
            kbs = cfg["k_size"]
            rbs = cfg.get("ref_block_size", (128, 128))
            common = dict(
                pack_gqa=True,
                pack_gqa_factor=4,
                sparse_k_block_size=kbs,
                block_sparse=True,
                range_merge=True,
            )
            add_ffa_spec(
                specs,
                direction="fwd",
                ref_block_size=rbs,
                disable_atomic=True,
                **common,
            )
            add_ffa_spec(
                specs,
                direction="bwd",
                ref_block_size=rbs,
                disable_atomic=True,
                bwd_dkv_bf16=True,
                **common,
            )
        return specs

    LOOPQ_GQA_CONFIGS = [
        {
            "name": "block_sparse_loopq_q64k64",
            "q_size": 64,
            "k_size": 64,
            "ref_block_size": (64, 128),
            "err_ratio_dict": {"dq_min_norm_rtol": 0.05},
        },
        {
            "name": "block_sparse_loopq_q128k1",
            "q_size": 128,
            "k_size": 1,
            "ref_block_size": (128, 128),
            "err_ratio_dict": {"dq_min_norm_rtol": 0.05},
        },
    ]

    @parameterize("cfg", LOOPQ_GQA_CONFIGS)
    def test_block_sparse_loopq_gqa_no_packgqa(self, cfg):
        """BlockSparse LoopQ with GQA (NHQ=16, NHK=4) and pack_gqa=False."""
        torch.manual_seed(42)
        device = self.device
        seqlen = 2048
        dtype = torch.bfloat16
        num_heads_q, num_heads_kv, head_dim = 16, 4, 128

        helper = _BlockSparseTestHelper.__new__(_BlockSparseTestHelper)
        block_size = (cfg["q_size"], cfg["k_size"])
        (
            block_mask,
            block_sizes,
            block_row_sz,
            block_col_sz,
        ) = helper._generate_sparse_pattern(
            test_type="uniform",
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            seqlen=seqlen,
            sparsity_ratio=0.5,
            sparsity_granularity="per_kv_head",
            sparse_format="block_mask",
            block_size=block_size,
        )

        q = torch.randn(
            1,
            seqlen,
            num_heads_q,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            1,
            seqlen,
            num_heads_kv,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            1,
            seqlen,
            num_heads_kv,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        do = torch.randn_like(q)

        test_case = f"[block_sparse_loopq_gqa][{cfg['name']}]"
        print(f"\n>>> {test_case} START", flush=True)
        t0 = time.time()
        helper.assert_close_to_torch_ref(
            dtype=dtype,
            q=q,
            k=k,
            v=v,
            grad_output=do,
            seqlen=seqlen,
            block_size=block_sizes,
            block_mask=block_mask,
            head_wise="per_kv_head",
            sparse_format="block_mask",
            nhq=num_heads_q,
            nhk=num_heads_kv,
            pack_gqa=False,
            deterministic=False,
            swap_ab=False,
            ref_block_size=cfg["ref_block_size"],
            block_sparse=True,
            swap_bwd_qk_loop=False,
            test_case=test_case,
            sparsity_ratio=0.5,
            uniform=True,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            max_seqlen_q=cfg["q_size"],
            err_ratio_dict=cfg.get("err_ratio_dict", {}),
        )
        print(f">>> {test_case} PASSED  ({time.time() - t0:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════════════
# TestBlockSparseSweep — Classic CI sweep
# ═══════════════════════════════════════════════════════════


class TestBlockSparseSweep(DistTestBase):
    """BlockSparse Classic sweep — CI gate.

    Fixed compile params: NHQ=128, NHK=1(MQA), D=128, q_block=1, kbs=128, PackGQA=True.
    Varies runtime: seqlen × sparsity × swap_bwd_qk_loop(LoopK/LoopQ).
    """

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 600

    # parameter space shared by @parameterize and precompile_kernel_specs
    _PARAM_SPACE: dict[str, list] = dict(
        q_seqlen=[512, 1000, 16384],
        kv_seqlen=[512, 1000, 16384],
        sparsity=[0.2],
        swap_bwd_qk_loop=[False, True],
    )

    @classmethod
    def precompile_kernel_specs(cls):
        """Standard precompile interface — see magi_attention/testing/precompile.py.

        All classic-sweep combos share the same compile-time params
        (MQA128 + kbs=128); only the BWD loop direction changes kernels.
        The dense reference adds the non-sparse ARM+PackGQA128 kernels.
        """
        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}
        # dense reference: block_sparse=False, ARM, PackGQA128, swap=True
        add_ffa_spec(
            specs,
            direction="fwd",
            pack_gqa=True,
            pack_gqa_factor=128,
            range_merge=True,
        )
        add_ffa_spec(
            specs,
            direction="bwd",
            pack_gqa=True,
            pack_gqa_factor=128,
            range_merge=True,
            bwd_inner_loop_k=True,
        )
        # sparse FWD (auto-flags: disable_fwd_atomic, ref forced (128,128))
        add_ffa_spec(
            specs,
            direction="fwd",
            ref_block_size=(128, 128),
            disable_atomic=True,
            pack_gqa=True,
            pack_gqa_factor=128,
            block_sparse=True,
            range_merge=True,
            sparse_k_block_size=128,
        )
        for swap in cls._PARAM_SPACE["swap_bwd_qk_loop"]:
            add_ffa_spec(
                specs,
                direction="bwd",
                disable_atomic=not swap,  # LoopQ + PackGQA → dkv-atomic disabled
                disable_dq_atomic=swap,  # LoopK → dq-atomic disabled
                pack_gqa=True,
                pack_gqa_factor=128,
                block_sparse=True,
                range_merge=True,
                bwd_inner_loop_k=swap,
                sparse_k_block_size=128,
                bwd_dq_bf16=swap,
            )
        return specs

    @with_run_in_mp
    @parameterize("q_seqlen", _PARAM_SPACE["q_seqlen"])
    @parameterize("kv_seqlen", _PARAM_SPACE["kv_seqlen"])
    @parameterize("sparsity", _PARAM_SPACE["sparsity"])
    @parameterize("swap_bwd_qk_loop", _PARAM_SPACE["swap_bwd_qk_loop"])
    def test_block_sparse_mqa_sweep(
        self, q_seqlen, kv_seqlen, sparsity, swap_bwd_qk_loop
    ):
        from magi_attention.utils.sparse_utils import (
            generate_ranges_from_block_mask_triton,
        )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        device = self.device
        nhq, nhk, head_dim = 128, 1, 128
        dtype = torch.bfloat16
        kbs = 128

        n_q_blocks = q_seqlen
        n_k_blocks = kv_seqlen // kbs
        n_attend = max(1, int(n_k_blocks * (1.0 - sparsity)))

        sel = torch.rand(n_q_blocks, n_k_blocks, device=device).argsort(dim=1)[
            :, :n_attend
        ]
        block_mask = torch.zeros(
            1, nhk, n_q_blocks, n_k_blocks, dtype=torch.bool, device=device
        )
        block_mask[0, 0].scatter_(1, sel, True)
        q_ranges, k_ranges = generate_ranges_from_block_mask_triton(block_mask, 1, kbs)
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=device)

        q0 = torch.randn(q_seqlen, nhq, head_dim, device=device, dtype=dtype)
        k0 = torch.randn(kv_seqlen, nhk, head_dim, device=device, dtype=dtype)
        v0 = torch.randn(kv_seqlen, nhk, head_dim, device=device, dtype=dtype)
        do = torch.randn(q_seqlen, nhq, head_dim, device=device, dtype=dtype)

        def run(block_sparse, swap):
            q = q0.clone().requires_grad_(True)
            k = k0.clone().requires_grad_(True)
            v = v0.clone().requires_grad_(True)
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                block_sparse=block_sparse,
                range_merge=True,
                pack_gqa=True,
                swap_bwd_qk_loop=swap,
            )
            out.backward(do)
            return out.detach(), q.grad, k.grad, v.grad

        ref = run(block_sparse=False, swap=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        got = run(block_sparse=True, swap=swap_bwd_qk_loop)
        loop_name = "loopk" if swap_bwd_qk_loop else "loopq"
        tol = 2e-2
        for name, a, b in zip(("out", "dq", "dk", "dv"), got, ref):
            err = (
                (a.float() - b.float()).abs().max()
                / b.float().abs().max().clamp_min(1e-6)
            ).item()
            assert (
                err < tol
            ), f"sweep[Sq={q_seqlen},Skv={kv_seqlen},sp={sparsity},{loop_name}] {name} max_rel_err={err:.3e} >= {tol}"


# ═══════════════════════════════════════════════════════════
# TestBlockSparseComprehensiveSweep — comprehensive coverage (CI)
# ═══════════════════════════════════════════════════════════


class TestBlockSparseComprehensiveSweep(DistTestBase):
    """BlockSparse Comprehensive sweep — CI.

    Cross-product of GQA config × block size × inner env variants.
    """

    # parameter space shared by @parameterize and precompile_kernel_specs
    _PARAM_SPACE: dict[str, list] = dict(
        nhq_nhk=[(8, 8), (16, 4), (128, 1), (1, 1), (4, 2)],
        head_dim=[64, 128],
        q_size=[1, 16, 64, 128],
        k_size=[1, 8, 64, 128],
        sparsity_ratio=[0.5],
        inner_dir=["true", "false"],
        inner_load_mode=["tma", "cpasync"],
        inner_store_mode=["tma", "tma1d", "atomicadd"],
    )

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    @classmethod
    def precompile_kernel_specs(cls):
        """Standard precompile interface — see magi_attention/testing/precompile.py.

        FWD + BWD InnerLoopK for each (pack_f × hd × k_size × inner modes) combo.
        q_size and sparsity_ratio are runtime-only (don't affect compilation).
        """
        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}
        seen_pack_f: set = set()
        for nhq, nhk in cls._PARAM_SPACE["nhq_nhk"]:
            pack_f = nhq // nhk
            if pack_f in seen_pack_f:
                continue
            seen_pack_f.add(pack_f)
            for hd in cls._PARAM_SPACE["head_dim"]:
                for k_size in cls._PARAM_SPACE["k_size"]:
                    common = dict(
                        head_dim=hd,
                        pack_gqa=True,
                        pack_gqa_factor=pack_f,
                        block_sparse=True,
                        range_merge=True,
                        sparse_k_block_size=k_size,
                    )
                    for inner_dir in cls._PARAM_SPACE["inner_dir"]:
                        for inner_load in cls._PARAM_SPACE["inner_load_mode"]:
                            env_fwd = {
                                "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN": inner_dir,
                                "MAGI_ATTENTION_FFA_INNER_LOAD_MODE": inner_load,
                            }
                            add_ffa_spec(
                                specs,
                                direction="fwd",
                                env=env_fwd,
                                disable_atomic=True,
                                ref_block_size=(128, 128),
                                **common,
                            )
                            for inner_store in cls._PARAM_SPACE["inner_store_mode"]:
                                env_bwd = {
                                    "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN": inner_dir,
                                    "MAGI_ATTENTION_FFA_INNER_LOAD_MODE": inner_load,
                                    "MAGI_ATTENTION_FFA_INNER_STORE_MODE": inner_store,
                                }
                                add_ffa_spec(
                                    specs,
                                    direction="bwd",
                                    env=env_bwd,
                                    disable_dq_atomic=True,
                                    bwd_inner_loop_k=True,
                                    bwd_dq_bf16=True,
                                    **common,
                                )
        return specs

    @with_run_in_mp
    @parameterize("nhq_nhk", _PARAM_SPACE["nhq_nhk"])
    @parameterize("head_dim", _PARAM_SPACE["head_dim"])
    @parameterize("q_size", _PARAM_SPACE["q_size"])
    @parameterize("k_size", _PARAM_SPACE["k_size"])
    @parameterize("sparsity_ratio", _PARAM_SPACE["sparsity_ratio"])
    @parameterize("inner_dir", _PARAM_SPACE["inner_dir"])
    @parameterize("inner_load_mode", _PARAM_SPACE["inner_load_mode"])
    @parameterize("inner_store_mode", _PARAM_SPACE["inner_store_mode"])
    def test_block_sparse_comprehensive_sweep(
        self,
        nhq_nhk,
        head_dim,
        q_size,
        k_size,
        sparsity_ratio,
        inner_dir,
        inner_load_mode,
        inner_store_mode,
    ):
        nhq, nhk = nhq_nhk
        hd = head_dim

        torch.manual_seed(42)
        seqlen = 2048
        dtype = torch.bfloat16

        helper = _BlockSparseTestHelper.__new__(_BlockSparseTestHelper)

        block_size = (q_size, k_size)
        num_q_blocks = seqlen // q_size
        num_kv_blocks = seqlen // k_size
        block_mask, _ = generate_block_sparse_pattern(
            num_q_heads=nhq,
            num_kv_heads=nhk,
            num_q_blocks=num_q_blocks,
            num_kv_blocks=num_kv_blocks,
            sparsity=sparsity_ratio,
            mode="per_kv_head",
            sparse_format="block_mask",
            device="cuda",
        )

        q = torch.randn(
            1, seqlen, nhq, hd, dtype=dtype, device=self.device, requires_grad=True
        )
        k = torch.randn(
            1, seqlen, nhk, hd, dtype=dtype, device=self.device, requires_grad=True
        )
        v = torch.randn(
            1, seqlen, nhk, hd, dtype=dtype, device=self.device, requires_grad=True
        )
        do = torch.randn_like(q)

        test_case = (
            f"[comprehensive][nhq={nhq},nhk={nhk},hd={hd},q={q_size},k={k_size}]"
            f"[sp={sparsity_ratio}][dir={inner_dir},load={inner_load_mode},store={inner_store_mode}]"
        )
        inner_env = {
            "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN": inner_dir,
            "MAGI_ATTENTION_FFA_INNER_LOAD_MODE": inner_load_mode,
            "MAGI_ATTENTION_FFA_INNER_STORE_MODE": inner_store_mode,
        }
        for key, val in inner_env.items():
            os.environ[key] = val
        try:
            helper.assert_close_to_torch_ref(
                dtype=dtype,
                q=q,
                k=k,
                v=v,
                grad_output=do,
                seqlen=seqlen,
                block_size=block_size,
                block_mask=block_mask,
                head_wise="per_kv_head",
                sparse_format="block_mask",
                nhq=nhq,
                nhk=nhk,
                pack_gqa=True,
                deterministic=False,
                swap_ab=False,
                ref_block_size=(64, 128),
                block_sparse=True,
                swap_bwd_qk_loop=True,
                test_case=test_case,
                sparsity_ratio=sparsity_ratio,
                uniform=True,
                max_seqlen_q=q_size,
            )
        finally:
            for key in inner_env:
                os.environ.pop(key, None)


if __name__ == "__main__":
    run_tests()
