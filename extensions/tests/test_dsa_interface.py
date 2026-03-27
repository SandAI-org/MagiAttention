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

import unittest
from typing import Any
from unittest import TestCase

import torch
from magi_attn_extensions.dsa_interface import dsa_attn_func

from magi_attention.testing import parameterize
from magi_attention.testing.precision import (
    EPSILON,
    MISMATCH_THRES_RATIO,
    assert_close,
    extract_mismatch_threshold,
)

from .dsa_ref_attn import dsa_ref_attn_func


class TestDSASparseInterface(TestCase):
    def setUp(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    @property
    def seed(self):
        return 42

    @property
    def device(self):
        return torch.cuda.current_device()

    @parameterize(
        "attn_config",
        [
            {
                "sq": 128,
                "skv": 128,
                "nhq": 8,
                "nhkv": 2,
                "hd": 64,
                "topk": 16,
            },
            {
                "sq": 256,
                "skv": 512,
                "nhq": 16,
                "nhkv": 4,
                "hd": 128,
                "topk": 32,
            },
        ],
    )
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("backend", ["flex", "ffa", "sdpa"])
    def test_sparse_flex_vs_ref(
        self,
        attn_config: dict[str, Any],
        dtype: torch.dtype,
        backend: str,
    ):
        sq, skv = attn_config["sq"], attn_config["skv"]
        nhq, nhkv, hd = attn_config["nhq"], attn_config["nhkv"], attn_config["hd"]
        topk = attn_config["topk"]

        q = torch.randn((sq, nhq, hd), dtype=dtype, device=self.device)
        k = torch.randn((skv, nhkv, hd), dtype=dtype, device=self.device)
        v = torch.randn((skv, nhkv, hd), dtype=dtype, device=self.device)

        # construct random index_map: (nhkv, sq, topk)
        # each Q token corresponds to topk K tokens for each KV head
        index_map = torch.stack(
            [
                torch.stack(
                    [torch.randperm(skv, device=self.device)[:topk] for _ in range(sq)]
                )
                for _ in range(nhkv)
            ]
        ).to(torch.int32)

        softmax_scale = hd**-0.5

        test_case = f"Sparse Attention [{dtype=}, {backend=} {attn_config=}]"

        # run Torch reference implementation (as baseline)
        o_ref, lse_ref = dsa_ref_attn_func(
            q=q,
            k=k,
            v=v,
            index_map=index_map,
            softmax_scale=softmax_scale,
            high_precision=False,
        )

        # run Torch reference implementation (high precision)
        o_ref_hp, lse_ref_hp = dsa_ref_attn_func(
            q=q,
            k=k,
            v=v,
            index_map=index_map,
            softmax_scale=softmax_scale,
            high_precision=True,
        )

        # run Flex Attention backend
        o_flex, lse_flex = dsa_attn_func(
            q=q,
            k=k,
            v=v,
            index_map=index_map,
            softmax_scale=softmax_scale,
            backend=backend,
        )

        o_rtol = 0.05
        o_atol = EPSILON
        lse_rtol = 0.001
        lse_atol = EPSILON

        o_thres = extract_mismatch_threshold(
            actual=o_ref,
            expected=o_ref_hp,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=MISMATCH_THRES_RATIO,
        )
        assert_close(
            o_flex,
            o_ref_hp,
            rtol=o_rtol,
            atol=o_atol,
            mismatch_threshold=o_thres,
            test_case=f"Sparse Attention Output Mismatch [{test_case=}]",
        )
        lse_thres = extract_mismatch_threshold(
            actual=lse_ref,
            expected=lse_ref_hp,
            atol=lse_atol,
            rtol=lse_rtol,
            mismatch_thres_ratio=MISMATCH_THRES_RATIO,
        )
        assert_close(
            lse_flex,
            lse_ref_hp,
            rtol=lse_rtol,
            atol=lse_atol,
            mismatch_threshold=lse_thres,
            test_case=f"Sparse Attention LSE Mismatch [{test_case=}]",
        )


if __name__ == "__main__":
    unittest.main()
