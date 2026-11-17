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
from unittest import TestCase, mock

import torch

from magi_attention.common import AttnRanges
from magi_attention.common.enum import (
    MagiAttentionKernelBackend,
    MagiAttentionPrecision,
)
from magi_attention.dist_attn_runtime_mgr import check_flag_comb
from magi_attention.functional import dist_attn
from magi_attention.functional.dist_attn import DistAttnFunc
from magi_attention.functional.dist_attn_runtime import DistAttnRuntime
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.testing import assert_equal, parameterize
from magi_attention.testing.utils import switch_env


class TestRuntimeForwardChecks(TestCase):
    def test_invalid_options_fail_before_communication_or_kernel_setup(self):
        cases = (
            ("FA4", dict(backend=MagiAttentionKernelBackend.FA4, max_logits=True)),
            ("attn sink", dict(flatten=True, sink=torch.empty(0))),
            ("return_max_logits", dict(flatten=True, max_logits=True)),
            ("overlap_degree", dict(no_overlap=True, degree=2)),
            ("qo_comm", dict(no_overlap=True, qo=True)),
        )
        for reason, options in cases:
            runtime = object.__new__(DistAttnRuntime)
            runtime.no_overlap = options.get("no_overlap", False)
            runtime.skip_comm = False
            runtime.overlap_degree = options.get("degree", 1)
            with (
                self.subTest(reason=reason),
                mock.patch.object(
                    DistAttnRuntime,
                    "kernel_backend",
                    new_callable=mock.PropertyMock,
                    return_value=options.get(
                        "backend", MagiAttentionKernelBackend.CUTEDSL
                    ),
                ),
                mock.patch.object(
                    DistAttnRuntime,
                    "flatten_head_groups",
                    new_callable=mock.PropertyMock,
                    return_value=options.get("flatten", False),
                ),
                mock.patch.object(
                    DistAttnRuntime,
                    "enable_qo_comm",
                    new_callable=mock.PropertyMock,
                    return_value=options.get("qo", False),
                ),
                mock.patch.object(dist_attn, "create_kernel_barrier") as barrier,
                mock.patch.object(
                    DistAttnFunc, "_no_overlap_forward"
                ) as no_overlap_forward,
                mock.patch.object(runtime, "get_curr_q_kv_and_fetch_next") as fetch,
            ):
                with self.assertRaisesRegex(AssertionError, reason):
                    DistAttnFunc.forward(
                        None,
                        None,
                        None,
                        None,
                        options.get("sink"),
                        runtime,
                        return_max_logits=options.get("max_logits", False),
                    )
                barrier.assert_not_called()
                no_overlap_forward.assert_not_called()
                fetch.assert_not_called()

    def test_supported_options_and_single_rank_no_overlap_are_allowed(self):
        runtime = object.__new__(DistAttnRuntime)
        runtime.no_overlap, runtime.skip_comm, runtime.overlap_degree = True, True, 0
        with (
            mock.patch.object(
                DistAttnRuntime,
                "kernel_backend",
                new_callable=mock.PropertyMock,
                return_value=MagiAttentionKernelBackend.CUTEDSL,
            ),
            mock.patch.object(
                DistAttnRuntime,
                "flatten_head_groups",
                new_callable=mock.PropertyMock,
                return_value=False,
            ),
        ):
            runtime._check_forward_args(None, return_max_logits=True)


class TestRuntimeNvtxLabels(TestCase):
    def test_attention_labels_preserve_full_reproducible_ranges(self):
        arg = object.__new__(AttnArg)
        arg.q_ranges = AttnRanges.from_ranges([[0, 16], [32, 48]])
        arg.k_ranges = AttnRanges.from_ranges([[0, 64], [0, 64]])
        arg.q_ranges_bwd = AttnRanges.from_ranges([[32, 48], [0, 16]])
        arg.k_ranges_bwd = AttnRanges.from_ranges([[0, 64], [0, 64]])
        arg.total_area = 2048
        self.assertIn(
            "attn/fwd | area=2,048 | ranges=2 | q_ranges=[0:16,32:48]",
            arg.get_nvtx_event_name(),
        )
        self.assertIn(
            "attn/bwd | area=2,048 | ranges=2 | q_ranges=[32:48,0:16]",
            arg.get_nvtx_event_name(is_bwd=True),
        )
        arg.q_ranges = AttnRanges.from_ranges([[i, i + 1] for i in range(20)])
        message = arg.get_nvtx_event_name()
        fields = dict(field.split("=", 1) for field in message.split(" | ")[1:])
        self.assertEqual(
            [
                list(map(int, bounds.split(":")))
                for bounds in fields["q_ranges"][1:-1].split(",")
            ],
            [[i, i + 1] for i in range(20)],
        )
        self.assertEqual(
            [
                list(map(int, bounds.split(":")))
                for bounds in fields["k_ranges"][1:-1].split(",")
            ],
            [[0, 64], [0, 64]],
        )

    def test_collective_label_counts_tuple_payload_and_lse(self):
        runtime = object.__new__(DistAttnRuntime)
        # Different K/V widths exercise tuple payload accounting.
        k = torch.empty(16, 2, 8, dtype=torch.bfloat16)
        v = torch.empty(16, 2, 4, dtype=torch.bfloat16)
        lse = torch.empty(16, 2, dtype=torch.float32)
        label = runtime._grpcoll_nvtx_event_name(
            "group_cast",
            input=(k, v),
            output=(k, v),
            comm_tokens=4,
            rdma_tokens=2,
            input_lse=lse,
            output_lse=lse,
        )
        self.assertIn("bytes=224 | rdma=112", label)
        self.assertIn("16x2x8:bfloat16, 16x2x4:bfloat16", label)
        self.assertIn("lse=[16x2:float32", label)
        fused = torch.empty(48, 2, 8, dtype=torch.bfloat16)
        self.assertIn(
            "bytes=128",
            runtime._grpcoll_nvtx_event_name(
                "group_reduce",
                input=fused,
                output=fused,
                comm_tokens=4,
            ),
        )


class TestAttentionPrecision(TestCase):
    def test_casts_preserve_gradients_output_dtype_and_metadata(self):
        for precision in (
            None,
            MagiAttentionPrecision.BF16,
            MagiAttentionPrecision.FP16,
            MagiAttentionPrecision.FP64,
        ):
            with self.subTest(precision=precision):
                inputs = [
                    torch.ones(2, 1, 4, dtype=torch.float32, requires_grad=True)
                    for _ in range(3)
                ]
                sink = torch.zeros(2, dtype=torch.float32)
                lse = torch.zeros(2, dtype=torch.float32)
                max_logits = torch.ones(1, dtype=torch.float32)
                runtime = mock.Mock(kernel_backend=MagiAttentionKernelBackend.SDPA)
                compute_dtype = (
                    torch.float32 if precision is None else precision.to_torch_dtype()
                )

                def apply(q, k, v, passed_sink, *args):
                    self.assertEqual((q.dtype, k.dtype, v.dtype), (compute_dtype,) * 3)
                    self.assertIs(passed_sink, sink)
                    if precision is None:
                        for value, original in zip((q, k, v), inputs):
                            self.assertIs(value, original)
                    return q + k + v, lse, max_logits

                with (
                    mock.patch.object(
                        dist_attn.env.general, "precision", return_value=precision
                    ),
                    mock.patch.object(DistAttnFunc, "apply", side_effect=apply),
                ):
                    out, meta = dist_attn.dist_attn_func(
                        *inputs, runtime, sink=sink, return_max_logits=True
                    )
                self.assertEqual(out.dtype, torch.float32)
                self.assertIs(meta.lse, lse)
                self.assertIs(meta.max_logits, max_logits)
                assert_equal(out, torch.full_like(inputs[0], 3))
                for grad in torch.autograd.grad(out.sum(), inputs):
                    assert_equal(grad, torch.ones_like(inputs[0]))

    def test_invalid_precision_fails_before_apply(self):
        value = torch.ones(2, 1, 4, dtype=torch.float16)
        runtime = mock.Mock(kernel_backend=MagiAttentionKernelBackend.FFA)
        with (
            mock.patch.object(
                dist_attn.env.general,
                "precision",
                return_value=MagiAttentionPrecision.FP32,
            ),
            mock.patch.object(DistAttnFunc, "apply") as apply,
        ):
            with self.assertRaisesRegex(AssertionError, "not supported"):
                dist_attn.dist_attn_func(value, value, value, runtime)
            apply.assert_not_called()

    def test_attention_exception_propagates_without_output_or_global_state_changes(
        self,
    ):
        value = torch.ones(2, 1, 4, dtype=torch.float32)
        default_dtype = torch.get_default_dtype()
        runtime = mock.Mock(kernel_backend=MagiAttentionKernelBackend.SDPA)
        with (
            mock.patch.object(
                dist_attn.env.general,
                "precision",
                return_value=MagiAttentionPrecision.BF16,
            ),
            mock.patch.object(
                DistAttnFunc, "apply", side_effect=RuntimeError("attention failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "attention failed"):
                dist_attn.dist_attn_func(value, value, value, runtime)
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(torch.get_default_dtype(), default_dtype)


class TestCheckFlagComb(unittest.TestCase):
    """Environment flag combinations that ``check_flag_comb`` validates before
    a dist-attn runtime is built; no device work."""

    _BASE_FLAGS = {
        "MAGI_ATTENTION_SDPA_BACKEND": "0",
        "MAGI_ATTENTION_FA4_BACKEND": "0",
        "MAGI_ATTENTION_DETERMINISTIC_MODE": "0",
        "MAGI_ATTENTION_HIERARCHICAL_COMM": "0",
        "MAGI_ATTENTION_NATIVE_GRPCOLL": "0",
        "MAGI_ATTENTION_QO_COMM": "0",
        "MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE": "0",
        "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE": "0",
        "MAGI_ATTENTION_CATGQA": "0",
        "MAGI_ATTENTION_BWD_PACK_FACTOR": "1",
    }

    @parameterize("backend", ["ffa", "sdpa", "sdpa_ol", "fa4", "cutedsl"])
    def test_bwd_pack_factor_requires_cutedsl(self, backend: str):
        flags = {**self._BASE_FLAGS, "MAGI_ATTENTION_KERNEL_BACKEND": backend}
        with switch_env({**flags, "MAGI_ATTENTION_BWD_PACK_FACTOR": "1"}):
            check_flag_comb()
        with switch_env({**flags, "MAGI_ATTENTION_BWD_PACK_FACTOR": "4"}):
            if backend == "cutedsl":
                check_flag_comb()
            else:
                with self.assertRaisesRegex(
                    AssertionError, "MAGI_ATTENTION_BWD_PACK_FACTOR > 1 requires"
                ):
                    check_flag_comb()

    def test_bwd_pack_factor_rejects_cat_gqa(self):
        flags = {
            **self._BASE_FLAGS,
            "MAGI_ATTENTION_KERNEL_BACKEND": "cutedsl",
            "MAGI_ATTENTION_CATGQA": "1",
        }
        with switch_env({**flags, "MAGI_ATTENTION_BWD_PACK_FACTOR": "1"}):
            check_flag_comb()
        with switch_env({**flags, "MAGI_ATTENTION_BWD_PACK_FACTOR": "2"}):
            with self.assertRaisesRegex(AssertionError, "MAGI_ATTENTION_CATGQA=1"):
                check_flag_comb()


if __name__ == "__main__":
    unittest.main()
