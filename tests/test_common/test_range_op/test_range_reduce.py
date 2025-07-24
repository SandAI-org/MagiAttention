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
from unittest import TestCase

import torch

from magi_attention.common.range_op import range_reduce


def range_reduce_ref(
    input: torch.Tensor,
    output: torch.Tensor,
    input_ranges: torch.Tensor,
    output_ranges: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """sum-reduce a2a output to output
    as a post-processing func for group_reduce_collective
    """

    # Handle the case when dim is not 0
    if dim != 0:
        input = input.transpose(0, dim).contiguous()
        output = output.transpose(0, dim).contiguous()
    else:
        input = input.contiguous()
        output = output.contiguous()

    for (out_start, out_end), (in_start, in_end) in zip(output_ranges, input_ranges):
        output[out_start:out_end] += input[in_start:in_end]

    # If transposed earlier, transpose back
    if dim != 0:
        output = output.transpose(0, dim)

    return output


class TestRangeReduce(TestCase):
    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_range_reduce(self):
        """Test range_reduce function by comparing with reference implementation"""

        # --- Test case 1: Basic functionality --- #

        input_tensor = torch.randn(10, 5, device=self.device)
        output_tensor = torch.randn(8, 5, device=self.device)
        input_ranges = torch.tensor(
            [[0, 2], [2, 3], [3, 6], [7, 8], [9, 10]],
            dtype=torch.int32,
            device=self.device,
        )
        output_ranges = torch.tensor(
            [[5, 7], [4, 5], [0, 3], [4, 5], [7, 8]],
            dtype=torch.int32,
            device=self.device,
        )
        cu_range_sizes = torch.tensor(
            [0, 2, 3, 6, 7], dtype=torch.int32, device=self.device
        )
        total_size = 8

        self.compare_implementations(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            test_case="Basic functionality",
        )

        # --- Test case 2: Empty tensor handling --- #

        empty_input = torch.empty(0, 5, device=self.device)
        empty_output = torch.empty(0, 5, device=self.device)
        empty_ranges = torch.empty(0, 2, dtype=torch.int32, device=self.device)
        empty_cu_sizes = torch.empty(0, dtype=torch.int32, device=self.device)

        self.compare_implementations(
            empty_input,
            empty_output,
            empty_ranges,
            empty_ranges,
            empty_cu_sizes,
            total_size=0,
            test_case="Empty tensor handling",
        )

        # --- Test case 3: Different dimension --- #

        input_tensor = torch.randn(5, 10, 3, device=self.device)
        output_tensor = torch.randn(5, 8, 3, device=self.device)
        input_ranges = torch.tensor(
            [[0, 3], [5, 8]], dtype=torch.int32, device=self.device
        )
        output_ranges = torch.tensor(
            [[0, 3], [4, 7]], dtype=torch.int32, device=self.device
        )
        cu_range_sizes = torch.tensor([0, 3], dtype=torch.int32, device=self.device)
        total_size = 6

        self.compare_implementations(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=1,
            test_case="Different dimension (dim=1)",
        )

        # --- Test case 4: Large tensors --- #

        large_input = torch.randn(100, 20, device=self.device)
        large_output = torch.randn(70, 20, device=self.device)
        large_input_ranges = torch.tensor(
            [[0, 30], [40, 80]], dtype=torch.int32, device=self.device
        )
        large_output_ranges = torch.tensor(
            [[0, 30], [30, 70]], dtype=torch.int32, device=self.device
        )
        large_cu_sizes = torch.tensor([0, 30], dtype=torch.int32, device=self.device)
        large_total_size = 70

        self.compare_implementations(
            large_input,
            large_output,
            large_input_ranges,
            large_output_ranges,
            large_cu_sizes,
            large_total_size,
            test_case="Large tensors",
        )

        # --- Test case 5: Edge case - single range --- #

        single_range_input = torch.randn(10, 5, device=self.device)
        single_range_output = torch.randn(4, 5, device=self.device)
        single_input_range = torch.tensor(
            [[3, 7]], dtype=torch.int32, device=self.device
        )
        single_output_range = torch.tensor(
            [[0, 4]], dtype=torch.int32, device=self.device
        )
        single_cu_size = torch.tensor([0], dtype=torch.int32, device=self.device)

        self.compare_implementations(
            single_range_input,
            single_range_output,
            single_input_range,
            single_output_range,
            single_cu_size,
            total_size=4,
            test_case="Edge case - single range",
        )

        # --- Test case 6: Multi-dimensional tensors --- #

        multi_dim_input = torch.randn(10, 5, 8, 4, device=self.device)
        multi_dim_output = torch.randn(8, 5, 8, 4, device=self.device)

        self.compare_implementations(
            multi_dim_input,
            multi_dim_output,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=0,
            test_case="Multi-dimensional tensors (dim=0)",
        )

        multi_dim_output2 = torch.randn(10, 5, 12, 4, device=self.device)
        self.compare_implementations(
            multi_dim_input,
            multi_dim_output2,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=2,
            test_case="Multi-dimensional tensors (dim=2)",
        )

        # --- Test case 7: Non-contiguous memory layout --- #

        non_contiguous_input = torch.randn(10, 5, device=self.device).transpose(0, 1)
        non_contiguous_output = torch.randn(5, 8, device=self.device)
        assert not non_contiguous_input.is_contiguous()

        self.compare_implementations(
            non_contiguous_input,
            non_contiguous_output,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=1,
            test_case="Non-contiguous memory layout",
        )

        # --- Test case 8: Various data types --- #

        for dtype in [torch.float16, torch.float32, torch.int32, torch.int64]:
            typed_input = torch.randn(10, 5, device=self.device).to(dtype)
            typed_output = torch.randn(8, 5, device=self.device).to(dtype)
            if dtype.is_floating_point:
                self.compare_implementations(
                    typed_input,
                    typed_output,
                    input_ranges,
                    output_ranges,
                    cu_range_sizes,
                    total_size,
                    test_case=f"Various data types ({dtype=})",
                )

    @staticmethod
    def compare_implementations(
        input_tensor,
        output_tensor,
        input_ranges,
        output_ranges,
        cu_range_sizes,
        total_size,
        dim=0,
        test_case: str = "",
    ):
        # Copy output tensors for comparison
        output_copy1 = output_tensor.clone()
        output_copy2 = output_tensor.clone()

        # Call the original implementation
        result = range_reduce(
            input=input_tensor,
            output=output_copy1,
            input_ranges=input_ranges,
            output_ranges=output_ranges,
            cu_range_sizes=cu_range_sizes,
            total_size=total_size,
            dim=dim,
        )

        # Call the reference implementation
        expected = range_reduce_ref(
            input=input_tensor,
            output=output_copy2,
            input_ranges=input_ranges,
            output_ranges=output_ranges,
            dim=dim,
        )

        # Verify results match
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
