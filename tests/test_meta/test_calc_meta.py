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
from magi_attention.meta.collection import calc_meta
from magi_attention.meta.collection.calc_meta import FA4AttnArg
from magi_attention.testing import assert_equal


class TestFA4AttnArg(TestCase):
    def test_hstu_conversion_trims_tail_and_preserves_empty_first_interval(self):
        arg = object.__new__(FA4AttnArg)
        arg.k_ranges = AttnRanges.from_ranges([[1, 4]])
        arg.seqlen_q, arg.seqlen_k = 2, 4
        arg.tile_m, arg.tile_m_bwd = 2, 4
        arg.ffa_fwd_args_dict = dict(q_ranges=None, k_ranges=None, attn_type_map=None)
        for source, expected_rows in (
            (torch.tensor([[0, 0], [1, 1], [4, 4], [0, 0]]), 3),
            (torch.zeros(3, 2, dtype=torch.int32), 1),
        ):
            with (
                self.subTest(expected_rows=expected_rows),
                mock.patch.object(
                    calc_meta, "magi_to_hstu_cuda", create=True
                ) as extension,
            ):
                extension.magi_to_hstu.return_value = source
                result = arg._make_hstu_func()
            self.assertEqual(arg.n_func, expected_rows)
            self.assertEqual(result.shape, (1, 1, expected_rows, 10))
            assert_equal(result[0, 0, :, :2], source[:expected_rows])
            self.assertEqual(extension.magi_to_hstu.call_args.kwargs["n_max_func"], 3)

    def test_hstu_interval_encoding(self):
        # First interval [0,F0), followed by [F1,F2); no compiler extension needed.
        func = torch.tensor([[2, 0], [4, 1], [6, 3]])
        expected = torch.tensor(
            [
                [True, True, False, False, True, True],
                [False, True, True, False, False, False],
            ]
        )
        assert_equal(FA4AttnArg.func_to_mask(func, 6), expected)


if __name__ == "__main__":
    unittest.main()
