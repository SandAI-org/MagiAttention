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
from itertools import accumulate
from unittest import TestCase

import torch

from magi_attention.common.range_op import range_gather
from magi_attention.common.range_op._range_gather import RangeGatherKernelBackend
from magi_attention.testing import parameterize


def range_gather_ref(
    input: torch.Tensor,
    ranges: torch.Tensor,
    dim: int = 0,
):
    # Calculate cumulative range sizes and total size
    ranges_sizes = [0] + (ranges[:, 1] - ranges[:, 0]).tolist()
    cu_range_sizes = list(accumulate(ranges_sizes))
    total_size = cu_range_sizes[-1]
    cu_range_sizes = torch.tensor(
        cu_range_sizes[:-1], dtype=torch.int32, device=input.device
    )

    # Create output tensor buffer
    output_shape = list(input.shape)
    output_shape[dim] = total_size
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Return directly if empty tensor
    if ranges.shape[0] == 0 or input.numel() == 0:
        return output

    # Handle the case when dim is not 0
    if dim != 0:
        input = input.transpose(0, dim).contiguous()
        output = output.transpose(0, dim).contiguous()
    else:
        input = input.contiguous()
        output = output.contiguous()

    # Iterate through each range, copy input data to output
    for i, (start, end) in enumerate(ranges):
        out_start = cu_range_sizes[i].item()
        range_size = end.item() - start.item()
        output[out_start : out_start + range_size] = input[start:end]

    # If transposed earlier, transpose back
    if dim != 0:
        output = output.transpose(0, dim)

    return output


class TestRangeGather(TestCase):
    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @parameterize("kernel_backend", ["per_row", "per_range"])
    def test_range_gather(self, kernel_backend: RangeGatherKernelBackend):
        """Test range_gather function by comparing with reference implementation"""

        # --- Test case 1: Basic functionality --- #

        input_tensor = torch.randn(10, 5, device=self.device)
        ranges = torch.tensor(
            [[0, 3], [8, 9], [9, 10], [5, 8], [9, 10]],
            dtype=torch.int32,
            device=self.device,
        )

        self.compare_implementations(
            input_tensor,
            ranges,
            kernel_backend=kernel_backend,
            test_case="Basic functionality",
        )

        # --- Test case 2: Empty tensor handling --- #

        empty_input = torch.empty(0, 5, device=self.device)
        empty_ranges = torch.empty(0, 2, dtype=torch.int32, device=self.device)

        self.compare_implementations(
            empty_input,
            empty_ranges,
            0,
            kernel_backend=kernel_backend,
            test_case="Empty tensor handling",
        )

        # --- Test case 3: Different dimensions --- #

        input_tensor = torch.randn(5, 10, 3, device=self.device)
        ranges = torch.tensor([[0, 3], [5, 8]], dtype=torch.int32, device=self.device)

        self.compare_implementations(
            input_tensor,
            ranges,
            dim=1,
            kernel_backend=kernel_backend,
            test_case="Different dimensions",
        )

        # --- Test case 4: Large tensors --- #

        large_input = torch.randn(100, 20, device=self.device)
        large_ranges = torch.tensor(
            [[0, 30], [40, 80]], dtype=torch.int32, device=self.device
        )

        self.compare_implementations(
            large_input,
            large_ranges,
            kernel_backend=kernel_backend,
            test_case="Large tensors",
        )

        # --- Test case 5: Edge case - single range --- #

        single_range_input = torch.randn(10, 5, device=self.device)
        single_range = torch.tensor([[3, 7]], dtype=torch.int32, device=self.device)

        self.compare_implementations(
            single_range_input,
            single_range,
            kernel_backend=kernel_backend,
            test_case="Edge case - single range",
        )

        # --- Test case 6: Multi-dimensional tensors --- #

        multi_dim_input = torch.randn(10, 5, 8, 4, device=self.device)

        self.compare_implementations(
            multi_dim_input,
            ranges,
            dim=0,
            kernel_backend=kernel_backend,
            test_case="Multi-dimensional tensors (dim=0)",
        )
        self.compare_implementations(
            multi_dim_input,
            ranges,
            dim=2,
            test_case="Multi-dimensional tensors (dim=2)",
        )

        # --- Test case 7: Non-contiguous memory layout --- #

        non_contiguous_input = torch.randn(10, 5, device=self.device).transpose(0, 1)
        assert not non_contiguous_input.is_contiguous()

        self.compare_implementations(
            non_contiguous_input,
            ranges,
            dim=1,
            kernel_backend=kernel_backend,
            test_case="Non-contiguous memory layout",
        )

        # --- Test case 8: Various data types --- #

        for dtype in [torch.float16, torch.float32, torch.int32, torch.int64]:
            typed_input = torch.randn(10, 5, device=self.device).to(dtype)
            if dtype.is_floating_point:
                self.compare_implementations(
                    typed_input,
                    ranges,
                    kernel_backend=kernel_backend,
                    test_case=f"Various data types ({dtype=})",
                )

        # --- Test case 9: Random data large-scale testing --- #

        torch.manual_seed(self.seed)
        for idx in range(5):
            # Randomly generate input
            input_size = torch.randint(20, 50, (1,)).item()
            feature_size = torch.randint(5, 15, (1,)).item()
            input_tensor = torch.randn(input_size, feature_size, device=self.device)

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

            ranges = torch.tensor(ranges_list, dtype=torch.int32, device=self.device)

            self.compare_implementations(
                input_tensor,
                ranges,
                kernel_backend=kernel_backend,
                test_case=f"Random data large-scale testing ({idx=})",
            )

    @staticmethod
    def compare_implementations(
        input_tensor,
        ranges,
        dim=0,
        kernel_backend: RangeGatherKernelBackend | None = None,
        test_case: str = "",
    ):
        # Call the original implementation
        result = range_gather(
            input=input_tensor,
            ranges=ranges,
            dim=dim,
            range_gather_kernel_backend=kernel_backend,
        )

        # Call the reference implementation
        expected = range_gather_ref(
            input=input_tensor,
            ranges=ranges,
            dim=dim,
        )

        # Verify results match
        try:
            torch.equal(result, expected)
        except AssertionError as e:
            raise AssertionError(
                f"Test case: {test_case} failed with error: {e}\nwhere {result=}\n{expected=}\n"
            )


if __name__ == "__main__":
    unittest.main()
