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

# mypy: disable-error-code="union-attr"
import unittest
from typing import Any
from unittest import TestCase

import torch
from einops import rearrange, repeat

# isort: split
from magi_attention.api.functools import (
    infer_attn_mask_from_cu_seqlens,
    infer_varlen_mask_from_batch,
)
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnSinkLayout
from magi_attention.testing import parameterize
from magi_attention.testing.precision import (
    EPSILON,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
    ref_attn_func,
)
from magi_attention.utils import get_attn_mask_from_ffa_args

# isort: split
from extensions.fa2_interface_with_sink import (
    fa2_func_with_sink,
    fa2_kvpacked_func_with_sink,
    fa2_qkvpacked_func_with_sink,
    fa2_varlen_func_with_sink,
    fa2_varlen_kvpacked_func_with_sink,
    fa2_varlen_qkvpacked_func_with_sink,
)
from extensions.fa3_interface_with_sink import (
    fa3_func_with_sink,
    fa3_qkvpacked_func_with_sink,
    fa3_varlen_func_with_sink,
)


class TestFAInterfaceWithSink(TestCase):
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
        "mode",
        [
            "batch",
            "varlen",
            "qkvpacked",
            "kvpacked",
            "varlen_qkvpacked",
            "varlen_kvpacked",
        ],
    )
    @parameterize("sink_layout", ["sh"])  # ["sh", "ssh", "shd"])
    @parameterize(
        "attn_config",
        [
            {
                "batch_size": 1,
                "sq": 2048,
                "sk": 2048,
                "s_sink": 1,
                "nhq": 8,
                "nhk": 4,
                "hd": 64,
            },
            {
                "batch_size": 2,
                "sq": 1024,
                "sk": 1024,
                "s_sink": 2,
                "nhq": 8,
                "nhk": 8,
                "hd": 128,
            },
        ],
    )
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("causal", [False, True])
    def test_fa2_interface_with_sink(
        self,
        mode: str,
        sink_layout: AttnSinkLayout,
        attn_config: dict[str, Any],
        dtype: torch.dtype,
        causal: bool,
    ):
        b = attn_config["batch_size"]
        sq, sk, s_sink = attn_config["sq"], attn_config["sk"], attn_config["s_sink"]
        nhq, nhk, hd = attn_config["nhq"], attn_config["nhk"], attn_config["hd"]
        has_sink = s_sink > 0

        # construct data
        q = torch.randn(
            (b * sq, nhq, hd),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (b * sk, nhk, hd),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (b * sk, nhk, hd),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)
        sink = (
            self.init_sink_tensor(b * sq, s_sink, nhq, hd, sink_layout)
            if has_sink
            else None
        )

        # construct mask
        cu_seqlens_q, cu_seqlens_k = infer_varlen_mask_from_batch(b, sq)
        q_ranges, k_ranges, attn_type_map, *rest = infer_attn_mask_from_cu_seqlens(
            cu_seqlens_q, cu_seqlens_k, causal=causal
        )
        attn_type_map = [t.to_int_type() for t in attn_type_map]

        # run FA2 with sink
        match mode:
            case "batch":
                q_, k_, v_, do_ = [
                    rearrange(x, "(b s) h d -> b s h d", b=b) for x in (q, k, v, do)
                ]
                fa2_out = fa2_func_with_sink(
                    q=q_,
                    k=k_,
                    v=v_,
                    sink=sink,
                    causal=causal,
                    # NOTE: FA2 only supports returning lse when dropout_p > 0
                    return_attn_probs=False,
                )
                fa2_out.backward(do_)
                fa2_out = rearrange(fa2_out, "b s h d -> (b s) h d")
            case "varlen":
                fa2_out = fa2_varlen_func_with_sink(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=sq,
                    max_seqlen_k=sk,
                    sink=sink,
                    causal=causal,
                    # NOTE: FA2 only supports returning lse when dropout_p > 0
                    return_attn_probs=False,
                )
                fa2_out.backward(do)
            case "qkvpacked":
                q_, k_, v_, do_ = [
                    rearrange(x, "(b s) h d -> b s h d", b=b) for x in (q, k, v, do)
                ]
                # NOTE: FA2 does not support GQA/MQA for qkvpacked API
                rep_times = nhq // nhk
                k_, v_ = [
                    repeat(x, "b s h d -> b s (h r) d", r=rep_times) for x in (k_, v_)
                ]
                qkv = torch.stack([q_, k_, v_], dim=-3)  # stack before num_heads dim
                fa2_out = fa2_qkvpacked_func_with_sink(
                    qkv=qkv,
                    sink=sink,
                    causal=causal,
                    # NOTE: FA2 only supports returning lse when dropout_p > 0
                    return_attn_probs=False,
                )
                fa2_out.backward(do_)
                fa2_out = rearrange(fa2_out, "b s h d -> (b s) h d")
            case "kvpacked":
                q_, k_, v_, do_ = [
                    rearrange(x, "(b s) h d -> b s h d", b=b) for x in (q, k, v, do)
                ]
                kv = torch.stack([k_, v_], dim=-3)  # stack before num_heads dim
                fa2_out = fa2_kvpacked_func_with_sink(
                    q=q_,
                    kv=kv,
                    sink=sink,
                    causal=causal,
                    # NOTE: FA2 only supports returning lse when dropout_p > 0
                    return_attn_probs=False,
                )
                fa2_out.backward(do_)
                fa2_out = rearrange(fa2_out, "b s h d -> (b s) h d")
            case "varlen_qkvpacked":
                # NOTE: FA2 does not support GQA/MQA for varlen_qkvpacked API
                rep_times = nhq // nhk
                k_, v_ = [repeat(x, "s h d -> s (h r) d", r=rep_times) for x in (k, v)]
                qkv = torch.stack([q, k_, v_], dim=-3)  # stack before num_heads dim
                fa2_out = fa2_varlen_qkvpacked_func_with_sink(
                    qkv=qkv,
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=sq,
                    sink=sink,
                    causal=causal,
                    # NOTE: FA2 only supports returning lse when dropout_p > 0
                    return_attn_probs=False,
                )
                fa2_out.backward(do)
            case "varlen_kvpacked":
                kv = torch.stack([k, v], dim=-3)  # stack before num_heads dim
                fa2_out = fa2_varlen_kvpacked_func_with_sink(
                    q=q,
                    kv=kv,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=sq,
                    max_seqlen_k=sk,
                    sink=sink,
                    causal=causal,
                    # NOTE: FA2 only supports returning lse when dropout_p > 0
                    return_attn_probs=False,
                )
                fa2_out.backward(do)
            case _:
                raise NotImplementedError(f"Unsupported mode: {mode}")

        # fetch gradients
        fa2_dq, fa2_dk, fa2_dv = q.grad, k.grad, v.grad
        fa2_dsink = sink.grad if has_sink else None
        q.grad, k.grad, v.grad = None, None, None
        if has_sink:
            sink.grad = None

        # check
        self.assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=b * sq,
            total_seqlen_k=b * sk,
            total_q=q,
            total_k=k,
            total_v=v,
            total_sink=sink,
            total_out=fa2_out,
            total_lse=None,
            grad_total_q=fa2_dq,
            grad_total_k=fa2_dk,
            grad_total_v=fa2_dv,
            grad_total_sink=fa2_dsink,
            grad_total_out=do,
            dtype=dtype,
            sink_layout=sink_layout,
            test_case=(
                f"fa2_interface_with_sink_[{mode=}]x"
                f"[{attn_config=}]x[{dtype=}]x[{causal=}]"
            ),
        )

    @parameterize("mode", ["batch", "varlen", "qkvpacked"])
    @parameterize("sink_layout", ["sh"])  # ["sh", "ssh", "shd"])
    @parameterize(
        "attn_config",
        [
            {
                "batch_size": 1,
                "sq": 2048,
                "sk": 2048,
                "s_sink": 1,
                "nhq": 8,
                "nhk": 4,
                "hd": 64,
            },
            {
                "batch_size": 2,
                "sq": 1024,
                "sk": 1024,
                "s_sink": 2,
                "nhq": 8,
                "nhk": 8,
                "hd": 128,
            },
        ],
    )
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("causal", [False, True])
    def test_fa3_interface_with_sink(
        self,
        mode: str,
        sink_layout: AttnSinkLayout,
        attn_config: dict[str, Any],
        dtype: torch.dtype,
        causal: bool,
    ):
        b = attn_config["batch_size"]
        sq, sk, s_sink = attn_config["sq"], attn_config["sk"], attn_config["s_sink"]
        nhq, nhk, hd = attn_config["nhq"], attn_config["nhk"], attn_config["hd"]
        has_sink = s_sink > 0

        # construct data
        q = torch.randn(
            (b * sq, nhq, hd),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (b * sk, nhk, hd),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (b * sk, nhk, hd),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)
        sink = (
            self.init_sink_tensor(b * sq, s_sink, nhq, hd, sink_layout)
            if has_sink
            else None
        )
        if has_sink:
            assert isinstance(sink, torch.Tensor)

        # construct mask
        cu_seqlens_q, cu_seqlens_k = infer_varlen_mask_from_batch(b, sq)
        q_ranges, k_ranges, attn_type_map, *rest = infer_attn_mask_from_cu_seqlens(
            cu_seqlens_q, cu_seqlens_k, causal=causal
        )
        attn_type_map = [t.to_int_type() for t in attn_type_map]

        # run FA3 with sink
        match mode:
            case "batch":
                q_, k_, v_, do_ = [
                    rearrange(x, "(b s) h d -> b s h d", b=b) for x in (q, k, v, do)
                ]
                fa3_out, fa3_lse = fa3_func_with_sink(
                    q=q_,
                    k=k_,
                    v=v_,
                    sink=sink,
                    causal=causal,
                    return_attn_probs=True,
                )
                fa3_out.backward(do_)
                fa3_out = rearrange(fa3_out, "b s h d -> (b s) h d")
                fa3_lse = rearrange(fa3_lse, "b h s -> (b s) h")
            case "varlen":
                fa3_out, fa3_lse = fa3_varlen_func_with_sink(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=sq,
                    max_seqlen_k=sk,
                    sink=sink,
                    causal=causal,
                    return_attn_probs=True,
                )
                fa3_out.backward(do)
                fa3_lse = rearrange(fa3_lse, "h s -> s h")
            case "qkvpacked":
                q_, k_, v_, do_ = [
                    rearrange(x, "(b s) h d -> b s h d", b=b) for x in (q, k, v, do)
                ]
                qkv = torch.cat([q_, k_, v_], dim=-2)  # concat at num_heads dim
                fa3_out, fa3_lse = fa3_qkvpacked_func_with_sink(
                    qkv=qkv,
                    sink=sink,
                    causal=causal,
                    num_heads_q=nhq,
                    return_attn_probs=True,
                )
                fa3_out.backward(do_)
                fa3_out = rearrange(fa3_out, "b s h d -> (b s) h d")
                fa3_lse = rearrange(fa3_lse, "b h s -> (b s) h")
            case _:
                raise NotImplementedError(f"Unsupported mode: {mode}")

        # fetch gradients
        fa3_dq, fa3_dk, fa3_dv = q.grad, k.grad, v.grad
        fa3_dsink = sink.grad if has_sink else None
        q.grad, k.grad, v.grad = None, None, None
        if has_sink:
            sink.grad = None

        # check
        self.assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=b * sq,
            total_seqlen_k=b * sk,
            total_q=q,
            total_k=k,
            total_v=v,
            total_sink=sink,
            total_out=fa3_out,
            total_lse=fa3_lse,
            grad_total_q=fa3_dq,
            grad_total_k=fa3_dk,
            grad_total_v=fa3_dv,
            grad_total_sink=fa3_dsink,
            grad_total_out=do,
            dtype=dtype,
            sink_layout=sink_layout,
            test_case=(
                f"fa3_interface_with_sink_[{mode=}]x"
                f"[{attn_config=}]x[{dtype=}]x[{causal=}]"
            ),
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
        total_lse: torch.Tensor | None,
        grad_total_q: torch.Tensor,
        grad_total_k: torch.Tensor,
        grad_total_v: torch.Tensor,
        grad_total_sink: torch.Tensor | None,
        grad_total_out: torch.Tensor,
        dtype: torch.dtype,
        sink_layout: AttnSinkLayout = "sh",
        test_case: str = "",
    ) -> None:
        # -----   customize tolerance / threshold  ---- #

        if total_sink is not None:
            assert isinstance(total_sink, torch.Tensor)
            has_sink = True
        else:
            has_sink = False

        o_atol = EPSILON
        o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        o_norm_rtol_ratio = NORM_RTOL_RATIO
        o_mismatch_thres_ratio = MISMATCH_THRES_RATIO

        lse_atol = EPSILON
        lse_rtol = 0.001
        lse_norm_rtol_ratio = NORM_RTOL_RATIO
        lse_mismatch_thres_ratio = MISMATCH_THRES_RATIO

        dq_atol = EPSILON
        dq_rtol = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)
        dq_norm_rtol_ratio = NORM_RTOL_RATIO
        dq_mismatch_thres_ratio = MISMATCH_THRES_RATIO

        dk_atol = EPSILON
        dk_rtol = {torch.bfloat16: 0.15, torch.float16: 0.08}.get(dtype, 0.08)
        dk_norm_rtol_ratio = NORM_RTOL_RATIO
        dk_mismatch_thres_ratio = MISMATCH_THRES_RATIO

        dv_atol = EPSILON
        dv_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        dv_norm_rtol_ratio = NORM_RTOL_RATIO
        dv_mismatch_thres_ratio = MISMATCH_THRES_RATIO

        dsink_atol = EPSILON
        dsink_rtol = 0.05
        dsink_norm_rtol_ratio = NORM_RTOL_RATIO * 2
        dsink_mismatch_thres_ratio = MISMATCH_THRES_RATIO * 1.5
        dsink_min_mismatch_thres_ratio = (
            1 / (total_sink.numel()) if total_sink is not None else 0
        )

        # -----   build attn mask   ---- #

        mask = get_attn_mask_from_ffa_args(
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
            high_precision=True,
            backend="torch" if has_sink else "sdpa",
            return_lse=True,
            sink_layout=sink_layout,
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
            backend="torch" if has_sink else "sdpa",
            high_precision=False,
            return_lse=True,
            sink_layout=sink_layout,
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
                o_norm_rtol_ratio * out_ref_norm,
                msg=(
                    f"For {test_case=}: {out_norm=} should be no greater than "
                    f"({o_norm_rtol_ratio} x {out_ref_norm=}",
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
        )
        try:
            assert_close(
                total_out,
                total_out_ref_high_precision,
                atol=o_atol,
                rtol=o_rtol,
                mismatch_threshold=o_thres,
                test_case=f"{test_case} => o",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for fwd lse   ---- #

        if total_lse is not None:
            # fa style with Linf norm
            lse_norm = calc_inf_norm(total_lse, total_lse_ref_high_precision)
            lse_ref_norm = calc_inf_norm(
                total_lse_ref_low_precision, total_lse_ref_high_precision
            )
            try:
                self.assertLessEqual(
                    lse_norm,
                    lse_norm_rtol_ratio * lse_ref_norm,
                    msg=(
                        f"For {test_case=}: {lse_norm=} should be no greater than "
                        f"{lse_norm_rtol_ratio} x {lse_ref_norm=}"
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
            )
            try:
                assert_close(
                    total_lse,
                    total_lse_ref_high_precision,
                    atol=lse_atol,
                    rtol=lse_rtol,
                    mismatch_threshold=lse_thres,
                    test_case=f"{test_case} => lse",
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
                dq_norm_rtol_ratio * dq_ref_norm,
                msg=(
                    f"For {test_case=}: {dq_norm=} should be no greater than "
                    f"{dq_norm_rtol_ratio} x {dq_ref_norm=}"
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
        )
        try:
            assert_close(
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
                dk_norm_rtol_ratio * dk_ref_norm,
                msg=(
                    f"For {test_case=}: {dk_norm=} should be no greater than "
                    f"{dk_norm_rtol_ratio} x {dk_ref_norm=}"
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
        )
        try:
            assert_close(
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
                dv_norm_rtol_ratio * dv_ref_norm,
                msg=(
                    f"For {test_case=}: {dv_norm=} should be no greater than "
                    f"{dv_norm_rtol_ratio} x {dv_ref_norm=}"
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
        )
        try:
            assert_close(
                grad_total_v,
                grad_total_v_ref_high_precision,
                atol=dv_atol,
                rtol=dv_rtol,
                mismatch_threshold=dv_thres,
                test_case=f"{test_case} => dv",
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
                    dsink_norm_rtol_ratio * dsink_ref_norm,
                    msg=(
                        f"For {test_case=}: {dsink_norm=} should be no greater than "
                        f"{dsink_norm_rtol_ratio} x {dsink_ref_norm=}"
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
                min_mismatch_thres=dsink_min_mismatch_thres_ratio,
            )
            try:
                assert_close(
                    grad_total_sink,
                    grad_total_sink_ref_high_precision,
                    atol=dsink_atol,
                    rtol=dsink_rtol,
                    mismatch_threshold=dsink_thres,
                    test_case=f"{test_case} => dsink",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   raise error if any error occurs   ---- #

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

    def init_sink_tensor(
        self,
        sq: int,
        s_sink: int,
        nhq: int,
        hd: int,
        sink_layout: AttnSinkLayout = "sh",
    ) -> torch.Tensor:
        match sink_layout:
            case "sh":
                sink = torch.randn(
                    (s_sink, nhq),
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
            case "ssh":
                sink = torch.randn(
                    (sq, s_sink, nhq),
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

        return sink


if __name__ == "__main__":
    unittest.main()
