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
from unittest import TestCase

import torch

from magi_attention.common.range_op.utils import (
    _calc_cu_range_sizes,
    _calc_out2inp_range_map,
    _calc_ranges_row_map,
)


class TestRangeOpUtils(TestCase):
    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    def test_calc_cu_range_sizes(self):
        # ---------    normal ranges     --------- #

        ranges = torch.tensor(
            [[0, 2], [3, 6], [2, 3]],
            dtype=torch.int64,
            device=self.device,
        )
        cu_range_sizes_ref = torch.tensor(
            [0, 2, 5, 6], dtype=torch.int64, device=self.device
        )
        total_size_ref = 6
        cu_range_sizes, total_size = _calc_cu_range_sizes(ranges, device=self.device)

        self.assertEqual(total_size, total_size_ref)
        self.assertTrue(torch.equal(cu_range_sizes, cu_range_sizes_ref))

        # ---------    empty ranges    --------- #

        ranges = torch.tensor(
            [],
            dtype=torch.int64,
            device=self.device,
        )
        cu_range_sizes_ref = torch.tensor([0], dtype=torch.int64, device=self.device)
        total_size_ref = 0
        cu_range_sizes, total_size = _calc_cu_range_sizes(ranges, device=self.device)

        self.assertEqual(total_size, total_size_ref)
        self.assertTrue(torch.equal(cu_range_sizes, cu_range_sizes_ref))

    def test_calc_ranges_row_map(self):
        # ---------    normal ranges     --------- #

        ranges = torch.tensor(
            [[0, 2], [3, 6], [2, 3]],
            dtype=torch.int64,
            device=self.device,
        )
        total_size = 6
        row_map_ref = torch.tensor(
            [0, 0, 1, 1, 1, 2], dtype=torch.int64, device=self.device
        )

        row_map = _calc_ranges_row_map(ranges, total_size=total_size)

        self.assertTrue(torch.equal(row_map, row_map_ref))

        # ---------    emoty ranges     --------- #

        ranges = torch.tensor(
            [],
            dtype=torch.int64,
            device=self.device,
        )
        total_size = 0
        row_map_ref = torch.tensor([], dtype=torch.int64, device=self.device)

        row_map = _calc_ranges_row_map(ranges, total_size=total_size)

        self.assertTrue(torch.equal(row_map, row_map_ref))

    def test_calc_out2inp_range_map(self):
        # ---------    normal output ranges     --------- #

        output_ranges = torch.tensor(
            [[0, 2], [3, 6], [2, 3], [0, 2]],
            dtype=torch.int64,
            device=self.device,
        )
        out2inp_range_map_ref = torch.tensor(
            [
                [0, 3],
                [2, -1],
                [1, -1],
            ],
            dtype=torch.int64,
            device=self.device,
        )
        unique_ordered_out_ranges_ref = torch.tensor(
            [[0, 2], [2, 3], [3, 6]], dtype=torch.int64, device=self.device
        )
        max_inp_indices_size_ref = 2

        (
            out2inp_range_map,
            unique_ordered_out_ranges,
            max_inp_indices_size,
        ) = _calc_out2inp_range_map(output_ranges, device=self.device)

        self.assertTrue(torch.equal(out2inp_range_map, out2inp_range_map_ref))
        self.assertTrue(
            torch.equal(unique_ordered_out_ranges, unique_ordered_out_ranges_ref)
        )
        self.assertEqual(max_inp_indices_size, max_inp_indices_size_ref)

        # ---------    empty output ranges     --------- #

        output_ranges = torch.tensor(
            [],
            dtype=torch.int64,
            device=self.device,
        )
        out2inp_range_map_ref = torch.tensor([], dtype=torch.int64, device=self.device)
        unique_ordered_out_ranges_ref = torch.tensor(
            [], dtype=torch.int64, device=self.device
        )
        max_inp_indices_size_ref = 0

        (
            out2inp_range_map,
            unique_ordered_out_ranges,
            max_inp_indices_size,
        ) = _calc_out2inp_range_map(output_ranges, device=self.device)

        self.assertTrue(torch.equal(out2inp_range_map, out2inp_range_map_ref))
        self.assertTrue(
            torch.equal(unique_ordered_out_ranges, unique_ordered_out_ranges_ref)
        )
        self.assertEqual(max_inp_indices_size, max_inp_indices_size_ref)


if __name__ == "__main__":
    unittest.main()
