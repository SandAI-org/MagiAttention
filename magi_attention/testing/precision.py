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

import os
import re

import torch
import torch.distributed as dist
from packaging import version

from magi_attention.functional.utils import safe_subtract
from magi_attention.utils.dtype import max_fp_dtype

if version.parse(torch.__version__) > version.parse("2.4"):
    # NOTE: in testing, we should explicitly allow bf16/fp16 reduction for sdpa
    # by setting `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`
    # due to the new feature since torch2.5:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

# usage: to avoid division by zero in numerical calculation and assert-close testing
EPSILON = 1e-8

PRINT_NO_MISMATCH = "MAGI_ATTENTION_TEST_PRINT_NO_MISMATCH"

# NOTE: an experimental value from fa/ffa/magi_attention testing
MISMATCH_THRES_RATIO: float = 2.0
MAX_MISMATCH_THRES: float = 0.75
NORM_RTOL_RATIO: float = 2.0

# IB spec: https://nvdam.widen.net/s/dps8txlsrf/infiniband-ndr-400g-architecture-datasheet-1620877-r4
IB_BANDWIDTH = 50e9  # 500 GB/s, single-end

# H100 spec: https://www.nvidia.com/en-us/data-center/h100/
H100_TFLOPS_16 = 989.5e12  # 989 teraFLOPS
H100_MATMUL_MFU = 0.7
H100_NVLINK_BANDWIDTH = 450e9  # 450 GB/s, single-end
H100_NVLINK_A2A_BWU = 0.6

# H800 spec: https://chaoqing-i.com/upload/20231128/NVIDIA%20H800%20GPU%20Datasheet.pdf
H800_TFLOPS_16 = 989.5e12  # 989 teraFLOPS
H800_NVLINK_BANDWIDTH = 200e9  # 200 GB/s, single-end
H800_NVLINK_A2A_BWU = 0.6


def extract_mismatch_info(error_msg: str) -> tuple[int, int, float]:
    match = re.search(r"Mismatched elements: (\d+) / (\d+)", error_msg)

    if match:
        mismatched_elements = int(match.group(1))
        total_elements = int(match.group(2))
        mismatch_ratio = mismatched_elements / total_elements
        return mismatched_elements, total_elements, mismatch_ratio
    else:
        raise ValueError(f"Could not find mismatch elements in {error_msg=}")


@torch.no_grad
def extract_mismatch_threshold(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    mismatch_thres_ratio: float = 1.0,
    min_mismatch_thres: float = 0.0,
    max_mismatch_thres: float = 1.0,
) -> float:
    mismatch_threshold = 0.0
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as e:
        error_msg = str(e)
        _, _, mismatch_threshold = extract_mismatch_info(error_msg)

    # scale it by `mismatch_thres_ratio`,
    # and clamp it in [min_mismatch_thres, max_mismatch_thres]
    return min(
        max(mismatch_threshold * mismatch_thres_ratio, min_mismatch_thres),
        max_mismatch_thres,
    )


def _side_by_side(str_a: str, str_b: str, col_width: int = 60, sep: str = " | ") -> str:
    """Format two multi-line strings into a two-column side-by-side layout."""
    lines_a = str_a.splitlines()
    lines_b = str_b.splitlines()
    n = max(len(lines_a), len(lines_b))
    lines_a += [""] * (n - len(lines_a))
    lines_b += [""] * (n - len(lines_b))
    return "\n".join(la.ljust(col_width) + sep + lb for la, lb in zip(lines_a, lines_b))


def print_tensor_by_cols(
    a: torch.Tensor,
    b: torch.Tensor,
    label_a: str = "Tensor A",
    label_b: str = "Tensor B",
    return_print_str_only: bool = False,
) -> str | None:
    """Print (or return) two tensors side-by-side for easy visual comparison.

    Args:
        a: first tensor.
        b: second tensor.
        label_a: header label for the left column.
        label_b: header label for the right column.
        return_print_str_only: if True, return the formatted string instead of printing it.

    Returns:
        The formatted string when ``return_print_str_only=True``, otherwise ``None``.
    """
    str_a = f"{label_a}:\n{a}"
    str_b = f"{label_b}:\n{b}"
    col_width = max(max(len(line) for line in str_a.splitlines()), len(label_a))
    result = _side_by_side(str_a, str_b, col_width=col_width)
    if return_print_str_only:
        return result
    print(result)
    return None


def _format_greatest_diff_values(
    error_msg: str, a: torch.Tensor, b: torch.Tensor
) -> str:
    """Parse greatest abs/rel diff indices from a torch assert_close error message
    and return a string showing the actual tensor values at those positions."""
    lines = []
    for label, pattern in [
        ("abs", r"Greatest absolute difference: (.+?) at index \(([^)]+)\)"),
        ("rel", r"Greatest relative difference: (.+?) at index \(([^)]+)\)"),
    ]:
        m = re.search(pattern, error_msg)
        if not m:
            continue
        diff_val = m.group(1)
        idx_raw = m.group(2)
        # index may be a single int (1-D tensor) or comma-separated ints
        # filter out empty strings caused by trailing commas e.g. "(2,)" -> "2,"
        idx: tuple[int, ...] = tuple(
            int(x.strip()) for x in idx_raw.split(",") if x.strip()
        )
        try:
            val_a = a[idx].item()
            val_b = b[idx].item()
            lines.append(
                f"  Greatest {label} diff = {diff_val} at index {idx}:"
                f"  a={val_a},  b={val_b}"
            )
        except Exception:
            pass
    return "\n".join(lines) if lines else "(could not extract index info)"


@torch.no_grad
def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mismatch_threshold: float = 0,
    test_case: str = "",
    allow_none: bool = True,
    print_tensor_when_mismatch: bool = True,
    print_rank: int = 0,
    print_no_mismatch: bool = True,
) -> None:
    """Assert that two tensors are close within given tolerances,
    with a mismatch threshold to allow some degree of mismatch.

    Args:
        a (torch.Tensor): tensor a.
        b (torch.Tensor): tensor b.
        atol (float, optional): absolute tolerance. Defaults to ``1e-5``.
        rtol (float, optional): relative tolerance. Defaults to ``1e-5``.
        mismatch_threshold (float, optional): allowed mismatch threshold. Defaults to ``0``.
        test_case (str, optional): test case description. Defaults to "".

        allow_none (bool, optional): if ``True``, allow a or b to be None
            (and skip assert_close in that case).
        print_tensor_when_mismatch (bool, optional): if ``True``, print the two tensors
            side-by-side when mismatch exceeds threshold. Defaults to ``True``.
        print_rank (int, optional): rank to print from. Defaults to ``0``.
            And set to ``-1`` to print from all ranks.
        print_no_mismatch (bool, optional): if ``True``, print a message when there is no mismatch.
            Defaults to ``True``.

            NOTE: Set ``MAGI_ATTENTION_TEST_PRINT_NO_MISMATCH=0`` to force disable printing no-mismatch messages,
            mainly used to reduce logging noise in CI.
    """
    assert (
        0 <= mismatch_threshold <= 1
    ), f"{mismatch_threshold=} must be between 0 and 1"

    if not allow_none:
        assert a is not None, f"{test_case=}: Tensor a is None"
        assert b is not None, f"{test_case=}: Tensor b is None"

    if dist.is_initialized():
        rank = dist.get_rank()
        is_this_print_rank = print_rank == -1 or rank == print_rank
    else:
        is_this_print_rank = True

    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        no_mismatch_info = f"[{test_case}]: has no mismatch"
        if is_this_print_rank:
            print_no_mismatch = (
                os.environ.get(PRINT_NO_MISMATCH, "1") == "1" and print_no_mismatch
            )
            if print_no_mismatch:
                print(no_mismatch_info)
    except AssertionError as e:
        error_msg = str(e)
        mismatched_elements, total_elements, mismatch_ratio = extract_mismatch_info(
            error_msg
        )

        mismatch_info = (
            f"[{test_case}]: mismatch_ratio = {mismatched_elements} / {total_elements} "
            f"= {mismatch_ratio * 100:.4f} % | mismatch_threshold={mismatch_threshold * 100:.2f} %"
        )

        if mismatch_ratio <= mismatch_threshold:
            if is_this_print_rank:
                print(mismatch_info)
            return
        else:
            diff_values_info = _format_greatest_diff_values(error_msg, a, b)
            error_details = (
                f"\n>>>>>>>  Torch Error Message: \n\n{error_msg}\n\n"
                f">>>>>>>  Greatest Diff Values: \n\n{diff_values_info}\n\n"
                f">>>>>>>  Mismatch Detailed Info: \n\n{mismatch_info}\n\n"
            )
            if print_tensor_when_mismatch:
                side_by_side = print_tensor_by_cols(a, b, return_print_str_only=True)
                error_details += f">>>>> Tensors (A | B):\n{side_by_side}\n\n"
            raise type(e)(error_details) from None


@torch.no_grad
def assert_equal(
    a: torch.Tensor,
    b: torch.Tensor,
    test_case: str = "",
    allow_none: bool = True,
    print_tensor_when_mismatch: bool = True,
    print_rank: int = 0,
    print_no_mismatch: bool = True,
) -> None:
    """Assert that two tensors are exactly equal (bit-for-bit).

    This is the exact-equality counterpart of :func:`assert_close` with ``atol=rtol=mismatch_threshold=0``.

    Prefer this over a bare ``assert torch.equal(a, b)``: ``torch.equal`` is a weaker
    check. Per the torch docs, ``torch.equal`` treats tensors containing NaNs as never
    equal (so it can spuriously fail on legitimately-identical NaN payloads), and it does
    NOT differentiate the dtypes of the two tensors during comparison. Building on
    ``torch.testing.assert_close`` (with ``equal_nan``/dtype checking) gives stricter,
    more meaningful equality semantics.

    Args:
        a (torch.Tensor): tensor a.
        b (torch.Tensor): tensor b.
        test_case (str, optional): test case description. Defaults to "".
        allow_none (bool, optional): if ``True``, allow a or b to be None
            (and skip the check in that case).
        print_tensor_when_mismatch (bool, optional): if ``True``, print the two tensors
            side-by-side on mismatch. Defaults to ``True``.
        print_rank (int, optional): rank to print from. Defaults to ``0``.
            And set to ``-1`` to print from all ranks.
        print_no_mismatch (bool, optional): if ``True``, print a message when there is no mismatch.
            Defaults to ``True``.

            NOTE: Set ``MAGI_ATTENTION_TEST_PRINT_NO_MISMATCH=0`` to force disable printing no-mismatch messages,
            mainly used to reduce logging noise in CI.
    """
    assert_close(
        a,
        b,
        atol=0,
        rtol=0,
        mismatch_threshold=0,
        test_case=test_case,
        allow_none=allow_none,
        print_tensor_when_mismatch=print_tensor_when_mismatch,
        print_rank=print_rank,
        print_no_mismatch=print_no_mismatch,
    )


@torch.no_grad
def calc_inf_norm(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    dtype = max_fp_dtype(a.dtype, b.dtype, torch.float32)
    return safe_subtract(a.to(dtype), b.to(dtype)).norm(p=float("inf")).item()
