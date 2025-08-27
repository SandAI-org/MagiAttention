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
from collections import defaultdict
from typing import Literal
from unittest import TestCase

import torch

from magi_attention.common.range_op import range_reduce
from magi_attention.functional.utils import correct_attn_lse, correct_attn_out
from magi_attention.testing import parameterize


def range_reduce_ref(
    input: torch.Tensor,
    output: torch.Tensor,
    input_ranges: torch.Tensor,
    output_ranges: torch.Tensor,
    dim: int = 0,
    reduce_op: Literal["sum", "avg", "lse"] = "sum",
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Reference implementation of range reduce

    Args:
        input (torch.Tensor): Source tensor to reduce from
        output (torch.Tensor): Destination tensor to reduce into
        input_ranges (torch.Tensor): Tensor of [start, end] ranges in the input
        output_ranges (torch.Tensor): Tensor of [start, end] ranges in the output
        dim (int, optional): Dimension along which to perform the reduction. Default is 0.
        reduce_op (Literal["sum", "avg", "lse"]): the reduce operation to use. Defaults to "sum"
            - "sum": sum reduction
            - "avg": average reduction
            - "lse": log-sum-exp weighted average reduction, with lse correction
        input_lse (torch.Tensor | None, optional): Log-sum-exp tensor for input. Defaults to None.
        output_lse (torch.Tensor | None, optional): Log-sum-exp tensor for output. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor] | torch.Tensor: The output tensor with the corrected lse
        after reduction if reduce_op is "lse", otherwise only the output tensor after reduction

        NOTE: for simplicity, this reference function does not guarantee in-place reduction
    """

    is_lse_reduce = reduce_op == "lse"
    if is_lse_reduce:
        assert (
            input_lse is not None and output_lse is not None
        ), "input_lse and output_lse must be provided when reduce_op is 'lse'"
        assert (
            input_lse.dtype == output_lse.dtype == torch.float32
        ), "input_lse and output_lse must be float32 when reduce_op is 'lse'"

    # Handle the case when dim is not 0
    if dim != 0:
        input = input.transpose(0, dim).contiguous()
        output = output.transpose(0, dim).contiguous()
        if is_lse_reduce:
            input_lse = input_lse.transpose(0, dim).contiguous()  # type: ignore[union-attr]
            output_lse = output_lse.transpose(0, dim).contiguous()  # type: ignore[union-attr]
    else:
        input = input.contiguous()
        output = output.contiguous()

    output_ranges = output_ranges.tolist()
    input_ranges = input_ranges.tolist()

    match reduce_op:
        case "sum":
            for (out_start, out_end), (in_start, in_end) in zip(
                output_ranges, input_ranges
            ):
                output[out_start:out_end] += input[in_start:in_end]
        case "avg":
            out_range_cnt_map: dict[tuple[int, int], int] = defaultdict(int)
            for (out_start, out_end), (in_start, in_end) in zip(
                output_ranges, input_ranges
            ):
                output[out_start:out_end] += input[in_start:in_end]
                out_range_cnt_map[(out_start, out_end)] += 1

            for (out_start, out_end), cnt in out_range_cnt_map.items():
                output[out_start:out_end] /= cnt
        case "lse":
            # transpose from [s, nh] to [nh, s]
            input_lse = input_lse.transpose(-1, -2)  # type: ignore[union-attr]
            output_lse = output_lse.transpose(-1, -2)  # type: ignore[union-attr]

            for (out_start, out_end), (in_start, in_end) in zip(
                output_ranges, input_ranges
            ):
                cur_lse = input_lse[in_start:in_end]
                old_lse_acc = output_lse[out_start:out_end]
                new_lse_acc = correct_attn_lse(
                    lse1=old_lse_acc,
                    lse2=cur_lse,
                )
                output_lse[out_start:out_end].copy_(new_lse_acc)

                cur_out = input[in_start:in_end]
                old_out_acc = output[out_start:out_end]
                new_out_acc = correct_attn_out(
                    out1=old_out_acc,
                    lse1=old_lse_acc,
                    out2=cur_out,
                    lse2=cur_lse,
                    lse=new_lse_acc,
                )
                output[out_start:out_end].copy_(new_out_acc)

            # transpose from [nh, s] back to [s, nh]
            output_lse = output_lse.transpose(-1, -2)
        case _:
            raise ValueError(f"Invalid reduce_op: {reduce_op}")

    # If transposed earlier, transpose back
    if dim != 0:
        output = output.transpose(0, dim)
        if is_lse_reduce:
            output_lse = output_lse.transpose(0, dim)  # type: ignore[union-attr]

    if is_lse_reduce:
        return output, output_lse

    return output


class TestRangeReduce(TestCase):
    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @parameterize("reduce_op", ["sum", "avg"])
    @parameterize("deterministic", [False, True])
    def test_normal_range_reduce(self, reduce_op, deterministic):
        """Test range_reduce function with normal reduction"""

        # --- Test case 1: Basic functionality --- #

        input_tensor = torch.randn(10, 5, device=self.device)
        output_tensor = torch.randn(8, 5, device=self.device)
        if reduce_op == "avg":
            output_tensor.zero_()
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

        self.compare_normal_range_reudce(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            deterministic=deterministic,
            reduce_op=reduce_op,
            test_case="Basic functionality",
        )

        # --- Test case 2: Empty tensor handling --- #

        empty_input = torch.empty(0, 5, device=self.device)
        empty_output = torch.empty(0, 5, device=self.device)
        empty_ranges = torch.empty(0, 2, dtype=torch.int32, device=self.device)

        self.compare_normal_range_reudce(
            empty_input,
            empty_output,
            empty_ranges,
            empty_ranges,
            deterministic=deterministic,
            reduce_op=reduce_op,
            test_case="Empty tensor handling",
        )

        # --- Test case 3: Different dimension --- #

        input_tensor = torch.randn(5, 10, 3, device=self.device)
        output_tensor = torch.randn(5, 8, 3, device=self.device)
        if reduce_op == "avg":
            output_tensor.zero_()
        input_ranges = torch.tensor(
            [[0, 3], [5, 8]], dtype=torch.int32, device=self.device
        )
        output_ranges = torch.tensor(
            [[0, 3], [4, 7]], dtype=torch.int32, device=self.device
        )

        self.compare_normal_range_reudce(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            dim=1,
            deterministic=deterministic,
            reduce_op=reduce_op,
            test_case="Different dimension (dim=1)",
        )

        # --- Test case 4: Large tensors --- #

        large_input = torch.randn(100, 20, device=self.device)
        large_output = torch.randn(70, 20, device=self.device)
        if reduce_op == "avg":
            large_output.zero_()
        large_input_ranges = torch.tensor(
            [[0, 30], [40, 80]], dtype=torch.int32, device=self.device
        )
        large_output_ranges = torch.tensor(
            [[0, 30], [30, 70]], dtype=torch.int32, device=self.device
        )

        self.compare_normal_range_reudce(
            large_input,
            large_output,
            large_input_ranges,
            large_output_ranges,
            deterministic=deterministic,
            reduce_op=reduce_op,
            test_case="Large tensors",
        )

        # --- Test case 5: Edge case - single range --- #

        single_range_input = torch.randn(10, 5, device=self.device)
        single_range_output = torch.randn(4, 5, device=self.device)
        if reduce_op == "avg":
            single_range_output.zero_()
        single_input_range = torch.tensor(
            [[3, 7]], dtype=torch.int32, device=self.device
        )
        single_output_range = torch.tensor(
            [[0, 4]], dtype=torch.int32, device=self.device
        )

        self.compare_normal_range_reudce(
            single_range_input,
            single_range_output,
            single_input_range,
            single_output_range,
            deterministic=deterministic,
            reduce_op=reduce_op,
            test_case="Edge case - single range",
        )

        # --- Test case 6: Multi-dimensional tensors --- #

        multi_dim_input = torch.randn(10, 5, 8, 4, device=self.device)
        multi_dim_output = torch.randn(8, 5, 8, 4, device=self.device)
        if reduce_op == "avg":
            multi_dim_output.zero_()

        self.compare_normal_range_reudce(
            multi_dim_input,
            multi_dim_output,
            input_ranges,
            output_ranges,
            dim=0,
            deterministic=deterministic,
            reduce_op=reduce_op,
            test_case="Multi-dimensional tensors (dim=0)",
        )

        multi_dim_output2 = torch.randn(10, 5, 12, 4, device=self.device)
        if reduce_op == "avg":
            multi_dim_output2.zero_()

        self.compare_normal_range_reudce(
            multi_dim_input,
            multi_dim_output2,
            input_ranges,
            output_ranges,
            dim=2,
            deterministic=deterministic,
            reduce_op=reduce_op,
            test_case="Multi-dimensional tensors (dim=2)",
        )

        # --- Test case 7: Non-contiguous memory layout --- #

        non_contiguous_input = torch.randn(10, 5, device=self.device).transpose(0, 1)
        non_contiguous_output = torch.randn(5, 8, device=self.device)
        assert not non_contiguous_input.is_contiguous()
        if reduce_op == "avg":
            non_contiguous_output.zero_()

        self.compare_normal_range_reudce(
            non_contiguous_input,
            non_contiguous_output,
            input_ranges,
            output_ranges,
            dim=1,
            deterministic=deterministic,
            reduce_op=reduce_op,
            test_case="Non-contiguous memory layout",
        )

        # --- Test case 8: Various data types --- #

        for dtype in [torch.float16, torch.float32, torch.int32, torch.int64]:
            typed_input = torch.randn(10, 5, device=self.device).to(dtype)
            typed_output = torch.randn(8, 5, device=self.device).to(dtype)
            if reduce_op == "avg":
                typed_output.zero_()
            if dtype.is_floating_point:
                self.compare_normal_range_reudce(
                    typed_input,
                    typed_output,
                    input_ranges,
                    output_ranges,
                    deterministic=deterministic,
                    reduce_op=reduce_op,
                    test_case=f"Various data types ({dtype=})",
                )

    @staticmethod
    def compare_normal_range_reudce(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        input_ranges: torch.Tensor,
        output_ranges: torch.Tensor,
        dim=0,
        deterministic=False,
        reduce_op="sum",
        test_case: str = "",
    ):
        assert reduce_op != "lse", "this func does not support lse-reduce"

        # Copy output tensors for comparison
        output1 = output_tensor.clone()
        output2 = output_tensor.clone()

        # Call the original implementation
        result = range_reduce(
            input=input_tensor,
            output=output1,
            input_ranges=input_ranges,
            output_ranges=output_ranges,
            dim=dim,
            deterministic=deterministic,
            reduce_op=reduce_op,
        )
        assert output1.data_ptr() == result.data_ptr(), "Not in-place reduction"  # type: ignore[union-attr]

        # Call the reference implementation
        expected = range_reduce_ref(
            input=input_tensor,
            output=output2,
            input_ranges=input_ranges,
            output_ranges=output_ranges,
            dim=dim,
            reduce_op=reduce_op,
        )

        # Verify results match
        try:
            torch.testing.assert_close(result, expected)
        except AssertionError as e:
            deter_str = "deterministic" if deterministic else "non-deterministic"
            raise AssertionError(
                f"Test case: {test_case} failed with reduce op {reduce_op} "
                f"in {deter_str} mode: {e}\nwhere {result=}\n{expected=}\n"
            )

    def test_lse_range_reduce(self):
        """Test range_reduce function with lse reduction"""

        # --- Test case 1: Basic functionality --- #

        input_tensor = torch.randn(10, 5, 3, dtype=torch.bfloat16, device=self.device)
        output_tensor = torch.randn(8, 5, 3, dtype=torch.bfloat16, device=self.device)
        input_lse = torch.randn(10, 5, dtype=torch.float32, device=self.device)
        output_lse = torch.randn(8, 5, dtype=torch.float32, device=self.device)
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

        self.compare_lse_range_reudce(
            input_tensor,
            output_tensor,
            input_lse,
            output_lse,
            input_ranges,
            output_ranges,
            dim=0,
            test_case="Basic functionality with lse reduce",
        )

    @staticmethod
    def compare_lse_range_reudce(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        input_lse: torch.Tensor,
        output_lse: torch.Tensor,
        input_ranges: torch.Tensor,
        output_ranges: torch.Tensor,
        dim=0,
        test_case: str = "",
    ):
        # Copy output tensors for comparison
        output1 = output_tensor.clone()
        output2 = output_tensor.clone()
        output_lse1 = output_lse.clone()
        output_lse2 = output_lse.clone()

        # Call the original implementation
        out, lse = range_reduce(
            input=input_tensor,
            output=output1,
            input_ranges=input_ranges,
            output_ranges=output_ranges,
            dim=dim,
            deterministic=True,
            reduce_op="lse",
            input_lse=input_lse,
            output_lse=output_lse1,
        )
        assert output1.data_ptr() == out.data_ptr(), "Not in-place reduction"
        assert output_lse1.data_ptr() == lse.data_ptr(), "Not in-place reduction"

        # Call the reference implementation
        out_ref, lse_ref = range_reduce_ref(
            input=input_tensor,
            output=output2,
            input_ranges=input_ranges,
            output_ranges=output_ranges,
            dim=dim,
            reduce_op="lse",
            input_lse=input_lse,
            output_lse=output_lse2,
        )

        # Verify results match
        try:
            torch.testing.assert_close(out, out_ref)
        except AssertionError as e:
            raise AssertionError(
                f"Test case: {test_case} failed for out: {e}\nwhere {out=}\n{out_ref=}\n"
            )
        try:
            torch.testing.assert_close(lse, lse_ref)
        except AssertionError as e:
            raise AssertionError(
                f"Test case: {test_case} failed for lse: {e}\nwhere {lse=}\n{lse_ref=}\n"
            )


if __name__ == "__main__":
    unittest.main()
