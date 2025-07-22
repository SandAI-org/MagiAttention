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
    def test_range_reduce(self):
        """Test range_reduce function by comparing with reference implementation"""

        # Helper function to compare implementations
        def compare_implementations(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=0,
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
            assert torch.equal(
                result, expected
            ), f"Expected {result=} to be equal to {expected=}"
            return result, expected

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test case 1: Basic functionality
        input_tensor = torch.randn(10, 5, device=device)
        output_tensor = torch.randn(8, 5, device=device)
        input_ranges = torch.tensor([[0, 3], [5, 8]], dtype=torch.int32, device=device)
        output_ranges = torch.tensor([[0, 3], [4, 7]], dtype=torch.int32, device=device)
        cu_range_sizes = torch.tensor([0, 3], dtype=torch.int32, device=device)
        total_size = 6

        compare_implementations(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
        )

        # Test case 2: Empty tensor handling
        empty_input = torch.empty(0, 5, device=device)
        empty_output = torch.empty(0, 5, device=device)
        empty_ranges = torch.empty(0, 2, dtype=torch.int32, device=device)
        empty_cu_sizes = torch.empty(0, dtype=torch.int32, device=device)

        compare_implementations(
            empty_input, empty_output, empty_ranges, empty_ranges, empty_cu_sizes, 0, 0
        )

        # Test case 3: Different dimension (dim=1)
        input_tensor = torch.randn(5, 10, 3, device=device)
        output_tensor = torch.randn(5, 8, 3, device=device)
        input_ranges = torch.tensor([[0, 3], [5, 8]], dtype=torch.int32, device=device)
        output_ranges = torch.tensor([[0, 3], [4, 7]], dtype=torch.int32, device=device)
        cu_range_sizes = torch.tensor([0, 3], dtype=torch.int32, device=device)
        total_size = 6

        compare_implementations(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=1,
        )

        # Test case 4: Large tensors
        large_input = torch.randn(100, 20, device=device)
        large_output = torch.randn(70, 20, device=device)
        large_input_ranges = torch.tensor(
            [[0, 30], [40, 80]], dtype=torch.int32, device=device
        )
        large_output_ranges = torch.tensor(
            [[0, 30], [30, 70]], dtype=torch.int32, device=device
        )
        large_cu_sizes = torch.tensor([0, 30], dtype=torch.int32, device=device)
        large_total_size = 70

        compare_implementations(
            large_input,
            large_output,
            large_input_ranges,
            large_output_ranges,
            large_cu_sizes,
            large_total_size,
        )

        # Test case 5: Edge case - single range
        single_range_input = torch.randn(10, 5, device=device)
        single_range_output = torch.randn(4, 5, device=device)
        single_input_range = torch.tensor([[3, 7]], dtype=torch.int32, device=device)
        single_output_range = torch.tensor([[0, 4]], dtype=torch.int32, device=device)
        single_cu_size = torch.tensor([0], dtype=torch.int32, device=device)

        compare_implementations(
            single_range_input,
            single_range_output,
            single_input_range,
            single_output_range,
            single_cu_size,
            4,
        )

        # Test case 6: Multi-dimensional tensors
        multi_dim_input = torch.randn(10, 5, 8, 4, device=device)
        multi_dim_output = torch.randn(8, 5, 8, 4, device=device)

        compare_implementations(
            multi_dim_input,
            multi_dim_output,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=0,
        )

        multi_dim_output2 = torch.randn(10, 5, 12, 4, device=device)
        compare_implementations(
            multi_dim_input,
            multi_dim_output2,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=2,
        )

        # Test case 7: Non-contiguous memory layout
        non_contiguous_input = torch.randn(10, 5, device=device).transpose(0, 1)
        non_contiguous_output = torch.randn(5, 8, device=device)
        assert not non_contiguous_input.is_contiguous()

        compare_implementations(
            non_contiguous_input,
            non_contiguous_output,
            input_ranges,
            output_ranges,
            cu_range_sizes,
            total_size,
            dim=1,
        )

        # Test case 8: Various data types
        for dtype in [torch.float16, torch.float32, torch.int32, torch.int64]:
            typed_input = torch.randn(10, 5, device=device).to(dtype)
            typed_output = torch.randn(8, 5, device=device).to(dtype)
            if dtype.is_floating_point:
                compare_implementations(
                    typed_input,
                    typed_output,
                    input_ranges,
                    output_ranges,
                    cu_range_sizes,
                    total_size,
                )

        # Test case 9: Random data large-scale testing
        for _ in range(5):
            # Randomly generate inputs
            input_size = torch.randint(20, 50, (1,)).item()
            total_size = torch.randint(15, 40, (1,)).item()
            feature_size = torch.randint(5, 15, (1,)).item()
            input_tensor = torch.randn(input_size, feature_size, device=device)
            output_tensor = torch.randn(total_size, feature_size, device=device)

            # Randomly generate ranges
            num_ranges = torch.randint(100, 1000, (1,)).item()
            input_ranges_list = []
            output_ranges_list = []
            sizes_list = [0]
            output_pos = 0

            for _ in range(num_ranges):
                # Input range
                in_start = torch.randint(0, input_size - 5, (1,)).item()
                in_end = torch.randint(
                    in_start + 1, min(in_start + 10, input_size) + 1, (1,)
                ).item()
                range_size = in_end - in_start

                # Output range
                out_start = torch.randint(0, total_size - range_size, (1,)).item()
                out_end = out_start + range_size
                output_pos += range_size
                input_ranges_list.append([in_start, in_end])
                output_ranges_list.append([out_start, out_end])
                sizes_list.append(sizes_list[-1] + range_size)

            input_ranges = torch.tensor(
                input_ranges_list, dtype=torch.int32, device=device
            )
            output_ranges = torch.tensor(
                output_ranges_list, dtype=torch.int32, device=device
            )
            cu_range_sizes = torch.tensor(
                sizes_list[:-1], dtype=torch.int32, device=device
            )


if __name__ == "__main__":
    unittest.main()
