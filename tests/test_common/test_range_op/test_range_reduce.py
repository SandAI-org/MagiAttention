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
from unittest import TestCase

import torch

from magi_attention.common.enum import GroupReduceOp, OutMaybeWithLSE
from magi_attention.common.range_op import range_reduce
from magi_attention.functional.utils import correct_attn_lse, correct_attn_out
from magi_attention.testing import parameterize
from magi_attention.utils import is_fp_dtype_at_least, max_fp_dtype


def range_reduce_ref(
    input: torch.Tensor,
    output: torch.Tensor,
    input_ranges: torch.Tensor,
    output_ranges: torch.Tensor,
    dim: int = 0,
    reduce_op: GroupReduceOp = "sum",
    reduce_dtype: torch.dtype | None = None,
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
) -> OutMaybeWithLSE:
    """Reference implementation of range reduce

    Args:
        input (torch.Tensor): Source tensor to reduce from
        output (torch.Tensor): Destination tensor to reduce into
        input_ranges (torch.Tensor): Tensor of [start, end] ranges in the input
        output_ranges (torch.Tensor): Tensor of [start, end] ranges in the output
        dim (int, optional): Dimension along which to perform the reduction. Default is 0.
        reduce_op (ReduceOp): the reduce operation to use. Defaults to "sum"
            - "sum": sum reduction
            - "avg": average reduction
            - "lse": log-sum-exp weighted average reduction, with lse correction
        reduce_dtype (torch.dtype): the dtype for the reduction.
            Defaults to ``None`` to use the maximum precision of the input/output's dtype and fp32
        acc_reduce (bool): whether to accumulate the reduction to the given output buffer. Defaults to ``True``.
        input_lse (torch.Tensor | None, optional): Log-sum-exp tensor for input. Defaults to None.
        output_lse (torch.Tensor | None, optional): Log-sum-exp tensor for output. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor] | torch.Tensor: The output tensor with the corrected lse
        after reduction if reduce_op is "lse", otherwise only the output tensor after reduction

        NOTE: for simplicity, this reference function does not guarantee in-place reduction
    """

    reduce_dtype = reduce_dtype or max_fp_dtype(
        input.dtype, output.dtype, torch.float32
    )

    is_lse_reduce = reduce_op == "lse"
    if is_lse_reduce:
        assert (
            input_lse is not None and output_lse is not None
        ), "lse reduction requires input_lse and output_lse"
        assert is_fp_dtype_at_least(input_lse, torch.float32) and is_fp_dtype_at_least(
            output_lse, torch.float32
        ), "lse reduction requires input_lse and output_lse at least float32"
        assert input_lse.ndim == output_lse.ndim == 2, (
            "lse reduction requires input and output must be 2D tensors "
            "with the shape: [seqlen, nheads]"
        )
        assert input.ndim == output.ndim == 3, (
            "lse reduction requires input and output must be 3D tensors "
            "with the shape: [seqlen, nheads, head_dim]"
        )

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

    # Cast to reduce_dtype
    orig_output_dtype = output.dtype
    input = input.to(reduce_dtype)
    output = output.to(reduce_dtype)
    if is_lse_reduce:
        orig_output_lse_dtype = output_lse.dtype  # type: ignore[union-attr]
        input_lse = input_lse.to(reduce_dtype)  # type: ignore[union-attr]
        output_lse = output_lse.to(reduce_dtype)  # type: ignore[union-attr]

    output_ranges = output_ranges.tolist()
    input_ranges = input_ranges.tolist()

    match reduce_op:
        case "sum":
            for (out_start, out_end), (in_start, in_end) in zip(
                output_ranges, input_ranges
            ):
                output[out_start:out_end] += input[in_start:in_end]
        case "avg":
            out_range_cnt_map: dict[tuple[int, int], int] = (
                defaultdict(lambda: 1) if acc_reduce else defaultdict(int)
            )
            for (out_start, out_end), (in_start, in_end) in zip(
                output_ranges, input_ranges
            ):
                # if not acc_reduce, just store at the first time
                if not acc_reduce and out_range_cnt_map[(out_start, out_end)] == 0:
                    output[out_start:out_end] = input[in_start:in_end]
                else:
                    output[out_start:out_end] += input[in_start:in_end]
                out_range_cnt_map[(out_start, out_end)] += 1

            for (out_start, out_end), cnt in out_range_cnt_map.items():
                # if not acc_reduce and has no old value, set to 0
                if not acc_reduce and cnt == 0:
                    output[out_start:out_end] = 0
                elif cnt > 1:
                    output[out_start:out_end] /= cnt
        case "lse":
            out_range_vis_map: dict[tuple[int, int], bool] = defaultdict(bool)
            for (out_start, out_end), (in_start, in_end) in zip(
                output_ranges, input_ranges
            ):
                cur_lse = input_lse[in_start:in_end]  # type: ignore[index]
                # if not acc_reduce, just store at the first time
                if not acc_reduce and not out_range_vis_map[(out_start, out_end)]:
                    new_lse_acc = cur_lse  # type: ignore[index]
                else:
                    old_lse_acc = output_lse[out_start:out_end].clone()  # type: ignore[index]
                    new_lse_acc = correct_attn_lse(
                        lse1=old_lse_acc,
                        lse2=cur_lse,
                    )
                output_lse[out_start:out_end].copy_(new_lse_acc)  # type: ignore[index]

                cur_out = input[in_start:in_end]
                # if not acc_reduce, just store at the first time
                if not acc_reduce and not out_range_vis_map[(out_start, out_end)]:
                    new_out_acc = cur_out
                else:
                    old_out_acc = output[out_start:out_end].clone()
                    new_out_acc = correct_attn_out(
                        out1=old_out_acc,
                        lse1=old_lse_acc,
                        out2=cur_out,
                        lse2=cur_lse,
                        lse=new_lse_acc,
                    )
                output[out_start:out_end].copy_(new_out_acc)
                out_range_vis_map[(out_start, out_end)] = True
        case _:
            raise ValueError(f"Invalid reduce_op: {reduce_op}")

    # If transposed earlier, transpose back
    if dim != 0:
        output = output.transpose(0, dim)
        if is_lse_reduce:
            output_lse = output_lse.transpose(0, dim)  # type: ignore[union-attr]

    # Cast back to original dtype
    output = output.to(orig_output_dtype)
    if is_lse_reduce:
        output_lse = output_lse.to(orig_output_lse_dtype)  # type: ignore[union-attr]
        return output, output_lse

    return output


class TestRangeReduce(TestCase):
    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @parameterize("reduce_op", ["sum", "avg"])
    @parameterize("reduce_dtype", [None, torch.float32, torch.float64])
    @parameterize("acc_reduce", [True])  # TODO: support acc_reduce=False
    @parameterize("deterministic", [False, True])
    def test_normal_range_reduce(
        self, reduce_op, reduce_dtype, acc_reduce, deterministic
    ):
        """Test range_reduce function with normal reduction"""

        if not deterministic and reduce_dtype is not None:
            # non-deterministic mode only supports None
            return

        # --- Test case 1: Basic functionality --- #

        input_tensor = torch.randn(10, 5, dtype=self.dtype, device=self.device)
        output_tensor = torch.randn(8, 5, dtype=self.dtype, device=self.device)
        if reduce_op == "avg":
            output_tensor.zero_()
        input_ranges = torch.tensor(
            [[0, 2], [2, 3], [3, 6], [7, 8], [9, 10]],
            dtype=torch.int64,
            device=self.device,
        )
        output_ranges = torch.tensor(
            [[5, 7], [4, 5], [0, 3], [4, 5], [7, 8]],
            dtype=torch.int64,
            device=self.device,
        )

        self.compare_normal_range_reudce(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            deterministic=deterministic,
            reduce_op=reduce_op,
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            test_case="Basic functionality",
        )

        # --- Test case 2: Empty tensor handling --- #

        empty_input = torch.empty(0, 5, dtype=self.dtype, device=self.device)
        empty_output = torch.empty(0, 5, dtype=self.dtype, device=self.device)
        empty_ranges = torch.empty(0, 2, dtype=torch.int64, device=self.device)

        self.compare_normal_range_reudce(
            empty_input,
            empty_output,
            empty_ranges,
            empty_ranges,
            deterministic=deterministic,
            reduce_op=reduce_op,
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            test_case="Empty tensor handling",
        )

        # --- Test case 3: Different dimension --- #

        input_tensor = torch.randn(5, 10, 3, dtype=self.dtype, device=self.device)
        output_tensor = torch.randn(5, 8, 3, dtype=self.dtype, device=self.device)
        if reduce_op == "avg":
            output_tensor.zero_()
        input_ranges = torch.tensor(
            [[0, 3], [5, 8]], dtype=torch.int64, device=self.device
        )
        output_ranges = torch.tensor(
            [[0, 3], [4, 7]], dtype=torch.int64, device=self.device
        )

        self.compare_normal_range_reudce(
            input_tensor,
            output_tensor,
            input_ranges,
            output_ranges,
            dim=1,
            deterministic=deterministic,
            reduce_op=reduce_op,
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            test_case="Different dimension (dim=1)",
        )

        # --- Test case 4: Large tensors --- #

        large_input = torch.randn(100, 20, dtype=self.dtype, device=self.device)
        large_output = torch.randn(70, 20, dtype=self.dtype, device=self.device)
        if reduce_op == "avg":
            large_output.zero_()
        large_input_ranges = torch.tensor(
            [[0, 30], [40, 80]], dtype=torch.int64, device=self.device
        )
        large_output_ranges = torch.tensor(
            [[0, 30], [30, 70]], dtype=torch.int64, device=self.device
        )

        self.compare_normal_range_reudce(
            large_input,
            large_output,
            large_input_ranges,
            large_output_ranges,
            deterministic=deterministic,
            reduce_op=reduce_op,
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            test_case="Large tensors",
        )

        # --- Test case 5: Edge case - single range --- #

        single_range_input = torch.randn(10, 5, dtype=self.dtype, device=self.device)
        single_range_output = torch.randn(4, 5, dtype=self.dtype, device=self.device)
        if reduce_op == "avg":
            single_range_output.zero_()
        single_input_range = torch.tensor(
            [[3, 7]], dtype=torch.int64, device=self.device
        )
        single_output_range = torch.tensor(
            [[0, 4]], dtype=torch.int64, device=self.device
        )

        self.compare_normal_range_reudce(
            single_range_input,
            single_range_output,
            single_input_range,
            single_output_range,
            deterministic=deterministic,
            reduce_op=reduce_op,
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            test_case="Edge case - single range",
        )

        # --- Test case 6: Multi-dimensional tensors --- #

        multi_dim_input = torch.randn(10, 5, 8, 4, dtype=self.dtype, device=self.device)
        multi_dim_output = torch.randn(8, 5, 8, 4, dtype=self.dtype, device=self.device)
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
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            test_case="Multi-dimensional tensors (dim=0)",
        )

        multi_dim_output2 = torch.randn(
            10, 5, 12, 4, dtype=self.dtype, device=self.device
        )
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
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            test_case="Multi-dimensional tensors (dim=2)",
        )

        # --- Test case 7: Non-contiguous memory layout --- #

        non_contiguous_input = torch.randn(
            10, 5, dtype=self.dtype, device=self.device
        ).transpose(0, 1)
        non_contiguous_output = torch.randn(5, 8, dtype=self.dtype, device=self.device)
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
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
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
                    reduce_dtype=reduce_dtype,
                    acc_reduce=acc_reduce,
                    test_case=f"Various data types ({dtype=})",
                )

    @staticmethod
    def compare_normal_range_reudce(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        input_ranges: torch.Tensor,
        output_ranges: torch.Tensor,
        dim: int = 0,
        deterministic: bool = False,
        reduce_op: GroupReduceOp = "sum",
        reduce_dtype: torch.dtype | None = None,
        acc_reduce: bool = True,
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
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
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
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
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

    @parameterize("acc_reduce", [True])  # TODO: support acc_reduce=False
    def test_lse_range_reduce(self, acc_reduce):
        """Test range_reduce function with lse reduction"""

        # --- Test case 1: Basic functionality --- #

        input_tensor = torch.randn(10, 5, 3, dtype=self.dtype, device=self.device)
        output_tensor = torch.randn(8, 5, 3, dtype=self.dtype, device=self.device)
        input_lse = torch.randn(10, 5, dtype=torch.float32, device=self.device)
        output_lse = torch.randn(8, 5, dtype=torch.float32, device=self.device)
        input_ranges = torch.tensor(
            [[0, 2], [2, 3], [3, 6], [7, 8], [9, 10]],
            dtype=torch.int64,
            device=self.device,
        )
        output_ranges = torch.tensor(
            [[5, 7], [4, 5], [0, 3], [4, 5], [7, 8]],
            dtype=torch.int64,
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
            reduce_dtype=torch.float32,
            acc_reduce=acc_reduce,
            test_case="Basic functionality with bf16 input/output and fp32 lse",
        )

        # --- Test case 2: Single Ranges with fp64 --- #

        input_tensor = torch.randn(512, 2, 64, dtype=torch.float64, device=self.device)
        output_tensor = torch.randn(512, 2, 64, dtype=torch.float64, device=self.device)
        input_lse = torch.randn(512, 2, dtype=torch.float64, device=self.device)
        output_lse = torch.randn(512, 2, dtype=torch.float64, device=self.device)
        input_ranges = torch.tensor(
            [[0, 512]],
            dtype=torch.int64,
            device=self.device,
        )
        output_ranges = torch.tensor(
            [[0, 512]],
            dtype=torch.int64,
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
            reduce_dtype=None,
            acc_reduce=acc_reduce,
            test_case="Single Ranges with fp64 input/output and fp64 lse",
        )

        # --- Test case 3: Double Ranges with all -inf lse --- #

        input_tensor = torch.randn(
            (256, 3, 64), dtype=torch.float64, device=self.device
        )
        output_tensor = torch.randn(
            (128, 3, 64), dtype=torch.float64, device=self.device
        )
        input_lse = torch.full(
            (256, 3), fill_value=float("-inf"), dtype=torch.float64, device=self.device
        )
        output_lse = torch.full(
            (128, 3), fill_value=float("-inf"), dtype=torch.float64, device=self.device
        )
        input_ranges = torch.tensor(
            [[0, 128], [128, 256]],
            dtype=torch.int64,
            device=self.device,
        )
        output_ranges = torch.tensor(
            [[0, 128], [0, 128]],
            dtype=torch.int64,
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
            reduce_dtype=None,
            acc_reduce=acc_reduce,
            test_case="Double Ranges with all -inf lse",
        )

        # --- Test case 4: Incomplete Single Ranges with half -inf lse --- #

        input_tensor = torch.randn(
            (128, 6, 64), dtype=torch.float64, device=self.device
        )
        input_tensor[64:].zero_()
        output_tensor = torch.zeros(
            (256, 6, 64), dtype=torch.float64, device=self.device
        )
        input_lse = torch.randn((128, 6), dtype=torch.float64, device=self.device)
        input_lse[64:].fill_(float("-inf"))
        output_lse = torch.full(
            (256, 6), fill_value=float("-inf"), dtype=torch.float64, device=self.device
        )
        input_ranges = torch.tensor(
            [[0, 128]],
            dtype=torch.int64,
            device=self.device,
        )
        output_ranges = torch.tensor(
            [[0, 128]],
            dtype=torch.int64,
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
            reduce_dtype=None,
            acc_reduce=acc_reduce,
            test_case="Incomplete Single Ranges with half -inf lse",
        )

    @staticmethod
    def compare_lse_range_reudce(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        input_lse: torch.Tensor,
        output_lse: torch.Tensor,
        input_ranges: torch.Tensor,
        output_ranges: torch.Tensor,
        dim: int = 0,
        reduce_dtype: torch.dtype | None = None,
        acc_reduce: bool = True,
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
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            input_lse=input_lse,
            output_lse=output_lse1,
        )

        # check in-place
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
            reduce_dtype=reduce_dtype,
            acc_reduce=acc_reduce,
            input_lse=input_lse,
            output_lse=output_lse2,
        )

        # Verify results match
        err_msg_list: list[str] = []
        try:
            torch.testing.assert_close(out, out_ref)
        except AssertionError as e:
            err_msg_list.append(
                f"Test case: {test_case} failed for out: {e}\nwhere {out=}\n{out_ref=}\n"
            )
        try:
            torch.testing.assert_close(lse, lse_ref)
        except AssertionError as e:
            err_msg_list.append(
                f"Test case: {test_case} failed for lse: {e}\nwhere {lse=}\n{lse_ref=}\n"
            )

        if err_msg_list:
            raise AssertionError("\n".join(err_msg_list))


if __name__ == "__main__":
    unittest.main()
