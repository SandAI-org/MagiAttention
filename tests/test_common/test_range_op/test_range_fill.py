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

from magi_attention.common.range_op import range_fill_


def range_fill_ref(
    out: torch.Tensor,
    ranges: torch.Tensor,
    val: float,
    dim: int = 0,
) -> torch.Tensor:
    # Return directly if ranges or tensor is empty
    if ranges.shape[0] == 0 or out.numel() == 0:
        return out

    # Handle the case when dim is not 0
    if dim != 0:
        out = out.transpose(0, dim).contiguous()
    else:
        out = out.contiguous()

    # Iterate through each range and fill with the specified value
    for start, end in ranges:
        out[start:end].fill_(val)

    # If transposed earlier, transpose back
    if dim != 0:
        out = out.transpose(0, dim)

    return out


class TestRangeFill(TestCase):
    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_range_fill(self):
        """Test range_fill_ function by comparing with reference implementation"""

        # --- Test case 1: Basic functionality --- #

        input_tensor = torch.zeros(10, 5, device=self.device)
        ranges = torch.tensor([[0, 3], [5, 8]], dtype=torch.int64, device=self.device)
        val = 1.0

        self.compare_implementations(
            input_tensor,
            ranges,
            val,
            test_case="Basic functionality",
        )

        # --- Test case 2: Empty tensor handling --- #

        empty_input = torch.empty(0, 5, device=self.device)
        empty_ranges = torch.empty(0, 2, dtype=torch.int64, device=self.device)

        self.compare_implementations(
            empty_input,
            empty_ranges,
            val,
            0,
            test_case="Empty tensor handling",
        )

        # --- Test case 3: Different dimension (dim=1) --- #

        input_tensor = torch.zeros(5, 10, 3, device=self.device)
        ranges = torch.tensor([[0, 3], [5, 8]], dtype=torch.int64, device=self.device)

        self.compare_implementations(
            input_tensor,
            ranges,
            val,
            dim=1,
            test_case="Different dimension (dim=1)",
        )

        # --- Test case 4: Large tensors --- #

        large_input = torch.zeros(100, 20, device=self.device)
        large_ranges = torch.tensor(
            [[0, 30], [40, 80]], dtype=torch.int64, device=self.device
        )

        self.compare_implementations(
            large_input,
            large_ranges,
            val,
            test_case="Large tensors",
        )

        # --- Test case 5: Edge case - single range --- #

        single_range_input = torch.zeros(10, 5, device=self.device)
        single_range = torch.tensor([[3, 7]], dtype=torch.int64, device=self.device)

        self.compare_implementations(
            single_range_input,
            single_range,
            val,
            test_case="Edge case - single range",
        )

        # --- Test case 6: Multi-dimensional tensors --- #

        multi_dim_input = torch.zeros(10, 5, 8, 4, device=self.device)

        self.compare_implementations(
            multi_dim_input,
            ranges,
            val,
            dim=0,
            test_case="Multi-dimensional tensors (dim=0)",
        )
        self.compare_implementations(
            multi_dim_input,
            ranges,
            val,
            dim=2,
            test_case="Multi-dimensional tensors (dim=2)",
        )

        # --- Test case 7: Non-contiguous memory layout --- #

        non_contiguous_input = torch.zeros(10, 5, device=self.device).transpose(0, 1)
        assert not non_contiguous_input.is_contiguous()

        self.compare_implementations(
            non_contiguous_input,
            ranges,
            val,
            dim=1,
            test_case="Non-contiguous memory layout",
        )

        # --- Test case 8: Various data types --- #

        for dtype in [torch.float16, torch.float32, torch.int32, torch.int64]:
            typed_input = torch.zeros(10, 5, device=self.device).to(dtype)
            if dtype.is_floating_point:
                self.compare_implementations(
                    typed_input,
                    ranges,
                    val,
                    test_case=f"Various data types ({dtype=})",
                )

        # --- Test case 9: Random data large-scale testing --- #

        torch.manual_seed(self.seed)
        for idx in range(5):
            # Randomly generate inputs
            input_size = torch.randint(20, 50, (1,)).item()
            feature_size = torch.randint(5, 15, (1,)).item()
            input_tensor = torch.zeros(input_size, feature_size, device=self.device)

            # Randomly generate ranges
            num_ranges = torch.randint(1, 10, (1,)).item()
            ranges_list = []
            sizes_list = [0]

            for _ in range(num_ranges):
                start = torch.randint(0, input_size - 5, (1,)).item()
                end = torch.randint(
                    start + 1, min(start + 10, input_size) + 1, (1,)
                ).item()
                ranges_list.append([start, end])
                sizes_list.append(sizes_list[-1] + (end - start))

            ranges = torch.tensor(ranges_list, dtype=torch.int64, device=self.device)

            # Test different fill values
            for val in [0.0, 1.0, -1.0, 3.14, 42.0]:
                self.compare_implementations(
                    input_tensor.clone(),
                    ranges,
                    val,
                    test_case=f"Random data large-scale testing ({idx=})",
                )

    @staticmethod
    def compare_implementations(
        input_tensor,
        ranges,
        val,
        dim=0,
        test_case: str = "",
    ):
        # Copy input tensors for comparison
        input_copy1 = input_tensor.clone()
        input_copy2 = input_tensor.clone()

        # Call the original implementation
        result = range_fill_(
            input=input_copy1,
            ranges=ranges,
            val=val,
            dim=dim,
        )

        # Call the reference implementation
        expected = range_fill_ref(
            out=input_copy2,
            ranges=ranges,
            val=val,
            dim=dim,
        )

        # Verify results match
        try:
            assert torch.equal(result, expected)
        except AssertionError as e:
            raise AssertionError(
                f"Test case: {test_case} failed with error: {e}\nwhere {result=}\n{expected=}\n"
            ) from e


if __name__ == "__main__":
    unittest.main()
