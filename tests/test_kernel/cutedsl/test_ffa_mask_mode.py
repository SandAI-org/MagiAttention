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
from unittest import mock

import torch

from magi_attention.kernel.cutedsl.ffa_utils import (
    MT_MAP,
    MaskMode,
    normalize_mask_type_spec,
    normalize_mask_types,
    validate_per_range_mask_feature_support,
)
from magi_attention.kernel.cutedsl.flex_flash_attn import FlexFlashAttnFunc


class TestFFAMaskMode(unittest.TestCase):
    def test_static_modes(self):
        full_default = normalize_mask_type_spec(None)
        self.assertEqual(full_default.mode, MaskMode.STATIC_FULL)
        self.assertEqual(full_default.static_mask_type, MT_MAP.full)
        self.assertIsNone(full_default.per_range_mask_types)

        full = normalize_mask_type_spec(MT_MAP.full)
        self.assertEqual(full.mode, MaskMode.STATIC_FULL)
        self.assertEqual(full.static_mask_type, MT_MAP.full)

        causal = normalize_mask_type_spec(MT_MAP.causal)
        self.assertEqual(causal.mode, MaskMode.STATIC_CAUSAL)
        self.assertEqual(causal.static_mask_type, MT_MAP.causal)

    def test_static_compatibility_wrapper(self):
        self.assertEqual(normalize_mask_types(None), MT_MAP.full)
        self.assertEqual(normalize_mask_types(MT_MAP.full), MT_MAP.full)
        self.assertEqual(normalize_mask_types(MT_MAP.causal), MT_MAP.causal)

    def test_invalid_scalar_inputs(self):
        with self.assertRaises(ValueError):
            normalize_mask_type_spec(4)
        with self.assertRaises(ValueError):
            normalize_mask_type_spec(-1)

    def test_scalar_inv_bi_causal_rejected(self):
        # Valid mask types, but only the per-range runtime path implements them;
        # a scalar must not fall through to the Full specialization.
        for mask_type in (MT_MAP.inv_causal, MT_MAP.bi_causal):
            with self.assertRaisesRegex(NotImplementedError, "per-range"):
                normalize_mask_type_spec(mask_type)

    def test_per_range_requires_cuda(self):
        mask_types = torch.tensor([MT_MAP.full, MT_MAP.causal], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            normalize_mask_type_spec(mask_types, is_varlen=True)

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            normalize_mask_type_spec(None, batch_size=-1)

    def test_feature_support_rejects_unsupported_combos(self):
        mask_types = torch.tensor([MT_MAP.full, MT_MAP.causal], dtype=torch.int32)
        # CPU tensor never reaches feature checks; build a fake per-range spec via mock.
        spec = mock.Mock()
        spec.is_per_range = True

        with self.assertRaisesRegex(NotImplementedError, "only supported on SM100"):
            validate_per_range_mask_feature_support(spec, major_arch=9)
        with self.assertRaisesRegex(NotImplementedError, "local"):
            validate_per_range_mask_feature_support(
                spec, major_arch=10, is_local=True
            )
        with self.assertRaisesRegex(NotImplementedError, "mask_mod"):
            validate_per_range_mask_feature_support(
                spec, major_arch=10, has_mask_mod=True
            )
        with self.assertRaisesRegex(NotImplementedError, "block sparsity"):
            validate_per_range_mask_feature_support(
                spec, major_arch=10, has_block_sparse=True
            )
        with self.assertRaisesRegex(NotImplementedError, "score_mod"):
            validate_per_range_mask_feature_support(
                spec, major_arch=10, has_score_mod=True
            )
        with self.assertRaisesRegex(NotImplementedError, "softcap"):
            validate_per_range_mask_feature_support(
                spec, major_arch=10, has_softcap=True
            )

        # Static modes are always accepted.
        static = normalize_mask_type_spec(MT_MAP.causal)
        validate_per_range_mask_feature_support(
            static,
            major_arch=8,
            is_local=True,
            has_mask_mod=True,
            has_block_sparse=True,
            has_score_mod=True,
            has_softcap=True,
        )
        _ = mask_types  # silence unused in CPU-only path

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_per_range_mode_and_validation(self):
        mask_types = torch.tensor(
            [MT_MAP.full, MT_MAP.causal, MT_MAP.full],
            device="cuda",
            dtype=torch.int32,
        )
        spec = normalize_mask_type_spec(
            mask_types,
            batch_size=3,
            is_varlen=True,
        )
        self.assertEqual(spec.mode, MaskMode.PER_RANGE)
        self.assertIs(spec.per_range_mask_types, mask_types)
        self.assertIsNone(spec.static_mask_type)

        validate_per_range_mask_feature_support(spec, major_arch=10)

        with self.assertRaisesRegex(ValueError, "only with q/k ranges"):
            normalize_mask_type_spec(mask_types, is_varlen=False)
        with self.assertRaisesRegex(ValueError, "length must match"):
            normalize_mask_type_spec(mask_types, batch_size=2, is_varlen=True)
        with self.assertRaisesRegex(NotImplementedError, "flex_flash_attn_func"):
            normalize_mask_types(mask_types)

        wrong_dtype = mask_types.to(torch.int64)
        with self.assertRaisesRegex(TypeError, "dtype torch.int32"):
            normalize_mask_type_spec(wrong_dtype, is_varlen=True)

        wrong_shape = mask_types[:, None]
        with self.assertRaisesRegex(ValueError, r"shape \[num_ranges\]"):
            normalize_mask_type_spec(wrong_shape, is_varlen=True)

        noncontiguous = torch.empty(6, device="cuda", dtype=torch.int32)[::2]
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            normalize_mask_type_spec(noncontiguous, is_varlen=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_autograd_per_range_fwd_bwd_launches(self):
        q_ranges = torch.tensor(
            [[0, 64], [64, 128]], device="cuda", dtype=torch.int32
        )
        k_ranges = q_ranges.clone()
        mask_types = torch.tensor(
            [MT_MAP.full, MT_MAP.causal], device="cuda", dtype=torch.int32
        )
        q = torch.randn(
            128, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            128, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            128, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        major = torch.cuda.get_device_capability()[0]
        if major not in (10, 11):
            self.skipTest("Per-range mask_types requires SM100/SM110")

        out, lse = FlexFlashAttnFunc.apply(
            q,
            k,
            v,
            q_ranges,
            k_ranges,
            mask_types,
            64,
            64,
            None,
            0.0,
            None,
            "sh",
            None,
            False,
            None,
        )
        self.assertEqual(out.shape, q.shape)
        out.sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_normalize_num_ranges_alias(self):
        mask_types = torch.tensor(
            [MT_MAP.full, MT_MAP.causal], device="cuda", dtype=torch.int32
        )
        spec = normalize_mask_type_spec(mask_types, num_ranges=2, is_varlen=True)
        self.assertTrue(spec.is_per_range)

        with self.assertRaisesRegex(ValueError, "disagree"):
            normalize_mask_type_spec(
                mask_types, num_ranges=2, batch_size=3, is_varlen=True
            )


if __name__ == "__main__":
    unittest.main()
