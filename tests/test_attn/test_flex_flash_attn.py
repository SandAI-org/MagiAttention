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

import magi_attention.testing
from magi_attention.common import AttnRanges
from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.testing.precision import (
    EPSILON,
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
                "seqlen": 1000,
                "q_ranges": AttnRanges.from_ranges([[0, 100], [1, 101]] * 10),
                "k_ranges": AttnRanges.from_ranges(
                    [[i * 50, (i + 1) * 50] for i in range(20)]
                ),
                "attn_type_map": [0, 1] * 10,
            },
        ],
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
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("random_attn_type_map", [False, True])
    @parameterize("auto_range_merge", [False, True])
    @parameterize("fwd_deterministic", [False, True])
    def test_flex_attn(
        self,
        attn_mask_config: dict[str, Any],
        model_config: dict[str, Any],
        dtype: torch.dtype,
        random_attn_type_map: bool,
        auto_range_merge: bool = False,
        fwd_deterministic: bool = False,
    ):
        # TODO: auto_range_merge == True that are not supported when attn_type_map different yet, thus skip here
        if auto_range_merge and fwd_deterministic:
            return
        if auto_range_merge and random_attn_type_map:
            return
        if auto_range_merge and attn_mask_config["name"] == "deterministic_sample":
            return

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

        # FIXME: for square bi-causal mask, i.e. when only the main diagonal is valid
        # ffa bwd kernel encounters with some precision issue with dq/dk,
        # thus we skip here and will fix it asap
        if is_list_value_any(attn_type_map, 3):
            return

        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        test_case = (
            f"[{attn_mask_config['name']}]"
            f"[{model_config['name']}]"
            f"[dtype={dtype}]"
            f"[random_attn_type_map={random_attn_type_map}]"
            f"[auto_range_merge={auto_range_merge}]"
            f"[fwd_deterministic={fwd_deterministic}]"
        )

        # construct data
        q = torch.randn(
            (seqlen, num_heads_q, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (seqlen, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (seqlen, num_heads_kv, head_dim),
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

        # run ffa forward
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
            deterministic=fwd_deterministic,
        )
        o.backward(do)

        if fwd_deterministic:
            o2, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges_tensor,
                k_ranges_tensor,
                max_seqlen_q,
                max_seqlen_k,
                attn_type_map_tensor,
                auto_range_merge=auto_range_merge,
                deterministic=fwd_deterministic,
            )
            try:
                assert torch.equal(o, o2)
            except Exception as e:
                raise AssertionError("\n\n".join([str(e)]))

        # compare with reference
        self.assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=seqlen,
            total_seqlen_k=seqlen,
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

        # -----   init error message list   ---- #

        err_msg_list: list[str] = []

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


if __name__ == "__main__":
    unittest.main()
