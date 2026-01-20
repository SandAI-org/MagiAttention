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

import itertools
from typing import Any

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnSinkLayout
from magi_attention.functional.utils import correct_attn_lse, correct_attn_out, correct_attn_out_lse
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import EPSILON, assert_close
from magi_attention.utils import make_attn_mask_from_ffa_args, max_fp_dtype


class TestRefAttnFunc(DistTestBase):
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
        return 400

    @property
    def dtype(self) -> torch.dtype:
        return torch.float64

    def _assert_close_to_each_other(
        self,
        out_list: list[torch.Tensor],
        lse_list: list[torch.Tensor],
        dq_list: list[torch.Tensor],
        dk_list: list[torch.Tensor],
        dv_list: list[torch.Tensor],
        dsink_list: list[torch.Tensor],
        test_case: str = "",
    ) -> None:
        # -----   customize tolerance threshold  ---- #

        o_atol = EPSILON
        o_rtol = EPSILON

        lse_atol = EPSILON
        lse_rtol = EPSILON

        dq_atol = EPSILON
        dq_rtol = EPSILON

        dk_atol = EPSILON
        dk_rtol = EPSILON

        dv_atol = EPSILON
        dv_rtol = EPSILON

        dsink_atol = EPSILON
        dsink_rtol = EPSILON

        # -----   init error message list   ---- #

        err_msg_list: list[str] = []

        # -----   assert close for fwd out   ---- #

        out0 = out_list[0]
        for out in out_list[1:]:
            try:
                assert_close(
                    out,
                    out0,
                    atol=o_atol,
                    rtol=o_rtol,
                    test_case=f"{test_case} => o",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   assert close for fwd lse   ---- #

        lse0 = lse_list[0]
        for lse in lse_list:
            try:
                assert_close(
                    lse,
                    lse0,
                    atol=lse_atol,
                    rtol=lse_rtol,
                    test_case=f"{test_case} => lse",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   assert close for bwd dq   ---- #

        dq0 = dq_list[0]
        for dq in dq_list:
            try:
                assert_close(
                    dq,
                    dq0,
                    atol=dq_atol,
                    rtol=dq_rtol,
                    test_case=f"{test_case} => dq",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   assert close for bwd dk   ---- #

        dk0 = dk_list[0]
        for dk in dk_list:
            try:
                assert_close(
                    dk,
                    dk0,
                    atol=dk_atol,
                    rtol=dk_rtol,
                    test_case=f"{test_case} => dk",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   assert close for bwd dv   ---- #

        dv0 = dv_list[0]
        for dv in dv_list:
            try:
                assert_close(
                    dv,
                    dv0,
                    atol=dv_atol,
                    rtol=dv_rtol,
                    test_case=f"{test_case} => dv",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   assert close for bwd dsink   ---- #

        if dsink_list:
            dsink0 = dsink_list[0]
            for dsink in dsink_list:
                try:
                    assert_close(
                        dsink,
                        dsink0,
                        atol=dsink_atol,
                        rtol=dsink_rtol,
                        test_case=f"{test_case} => dsink",
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
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_type_map: list[int],
        sink_layout: AttnSinkLayout,
        test_case: str,
    ) -> None:
        has_sink = seqlen_sink > 0

        # construct data
        q = torch.randn(
            (seqlen_q, num_heads_q, head_dim),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (seqlen_kv, num_heads_kv, head_dim),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (seqlen_kv, num_heads_kv, head_dim),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)
        if has_sink:
            match sink_layout:
                case "sh":
                    sink = torch.randn(
                        (seqlen_sink, num_heads_q),
                        dtype=max_fp_dtype(torch.float32, self.dtype),
                        device=self.device,
                        requires_grad=True,
                    )
                case "ssh":
                    sink = torch.randn(
                        (seqlen_q, seqlen_sink, num_heads_q),
                        dtype=max_fp_dtype(torch.float32, self.dtype),
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

        # -----   build attn mask   ---- #

        mask = make_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=seqlen_q,
            total_seqlen_k=seqlen_kv,
            device=self.device,
        )

        # -----   run ref attn func   ---- #

        out_list, lse_list = [], []
        dq_list, dk_list, dv_list, dsink_list = [], [], [], []

        for backend, online_softmax in itertools.product(
            ("sdpa", "torch"), (False, True)
        ):
            # skip for sdpa backend
            if backend == "sdpa":
                if online_softmax:
                    continue
                if has_sink:
                    continue

            out, lse = ref_attn_func(
                q=q,
                k=k,
                v=v,
                sink=sink,
                mask=mask,
                layout="thd",
                sink_layout=sink_layout,
                backend=backend,
                high_precision=True,
                return_lse=True,
                online_softmax=online_softmax,
            )
            out_list.append(out)
            lse_list.append(lse)

            out.backward(do)
            dq_list.append(q.grad)
            dk_list.append(k.grad)
            dv_list.append(v.grad)
            dsink_list.append(sink.grad) if has_sink else ()
            q.grad, k.grad, v.grad = None, None, None

        # -----   assert close to each other   ---- #

        self._assert_close_to_each_other(
            out_list=out_list,
            lse_list=lse_list,
            dq_list=dq_list,
            dk_list=dk_list,
            dv_list=dv_list,
            dsink_list=dsink_list,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize(
        "attn_mask_config",
        [
            {
                "name": "full_2k",
                "seqlen": 2048,
                "seqlen_sink": 1,
                "sink_layout": "ssh",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
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
        "model_config",
        [
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
        ],
    )
    @parameterize("random_attn_type_map", [False, True])
    def test_ref_attn_func(
        self,
        attn_mask_config: dict[str, Any],
        model_config: dict[str, Any],
        random_attn_type_map: bool,
    ):
        # extract config
        seqlen: int = attn_mask_config["seqlen"]
        seqlen_sink: int = attn_mask_config.get("seqlen_sink", 0)
        sink_layout: AttnSinkLayout = attn_mask_config.get("sink_layout", "sh")
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

        if random_attn_type_map:
            # we now support attn type idx in {0, 1, 2, 3}
            attn_type_map = torch.randint(0, 4, (len(attn_type_map),)).tolist()

        test_case = (
            "[test_ref_attn_func]"
            f"[{attn_mask_config['name']}]"
            f"[{model_config['name']}]"
            f"[dtype={self.dtype}]"
            f"[random_attn_type_map={random_attn_type_map}]"
            f"[has_sink={seqlen_sink > 0}]"
            f"[sink_layout={sink_layout}]"
        )

        self.run_test_case(
            seqlen_q=seqlen,
            seqlen_kv=seqlen,
            seqlen_sink=seqlen_sink,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            sink_layout=sink_layout,
            test_case=test_case,
        )

    @parameterize("seqlen", [1024, 2049, 4090])
    @parameterize("num_heads", [1, 3, 8])
    @parameterize("head_dim", [64, 128])
    @parameterize("inplace", [False, True])
    @parameterize("contiguous", [True, False])
    def test_correct_attn_out_lse(
        self,
        seqlen: int,
        num_heads: int,
        head_dim: int,
        inplace: bool,
        contiguous: bool,
    ):
        out_to_correct = torch.randn(
            (num_heads, seqlen, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        lse_to_correct = torch.randn(
            (num_heads, seqlen),
            dtype=self.dtype,
            device=self.device,
        )
        corrected_out = torch.randn(
            (num_heads, seqlen, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        corrected_lse = torch.randn(
            (num_heads, seqlen),
            dtype=self.dtype,
            device=self.device,
        )
        
        if contiguous:
            out_to_correct = out_to_correct.contiguous()
            lse_to_correct = lse_to_correct.contiguous()
            corrected_out = corrected_out.contiguous()
            corrected_lse = corrected_lse.contiguous()
            
        out_to_correct_ref = out_to_correct.clone()
        lse_to_correct_ref = lse_to_correct.clone()
        corrected_out_ref = corrected_out.clone()
        corrected_lse_ref = corrected_lse.clone()
            
        out_ref, lse_ref = self._correct_attn_out_lse_ref(
            out_to_correct=out_to_correct_ref,
            lse_to_correct=lse_to_correct_ref,
            corrected_out=corrected_out_ref,
            corrected_lse=corrected_lse_ref,
            inplace=inplace,
        )
        
        out_test, lse_test = correct_attn_out_lse(
            out1=corrected_out,
            lse1=corrected_lse,
            out2=out_to_correct,
            lse2=lse_to_correct,
            inplace=inplace,
        )
        
        torch.testing.assert_close(out_ref, out_test)
        torch.testing.assert_close(lse_ref, lse_test)
        
        if inplace:
            assert corrected_out.data_ptr() == out_test.data_ptr()
            assert corrected_lse.data_ptr() == lse_test.data_ptr()
        
    @classmethod
    def _correct_attn_out_lse_ref(
        self,
        out_to_correct: torch.Tensor,
        lse_to_correct: torch.Tensor,
        corrected_out: torch.Tensor,
        corrected_lse: torch.Tensor,
        inplace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last_lse = corrected_lse.clone() if inplace else corrected_lse
        corrected_lse = correct_attn_lse(
            lse1=corrected_lse,
            lse2=lse_to_correct,
            inplace=inplace,
        )
        corrected_out = correct_attn_out(
            out1=corrected_out,
            lse1=last_lse,
            out2=out_to_correct,
            lse2=lse_to_correct,
            lse=corrected_lse,
            inplace=inplace,
        )
        
        return corrected_out, corrected_lse


if __name__ == "__main__":
    run_tests()
