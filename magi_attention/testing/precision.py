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

import re

import torch
from packaging import version

from magi_attention.functional.utils import safe_subtract
from magi_attention.utils import max_fp_dtype

if version.parse(torch.__version__) > version.parse("2.4"):
    # NOTE: in testing, we should explicitly allow bf16/fp16 reduction for sdpa
    # by setting `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`
    # due to the new feature since torch2.5:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

# usage: to avoid division by zero in numerical calculation and assert-close testing
EPSILON = 1e-8

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


@torch.no_grad
def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mismatch_threshold: float = 0,
    test_case: str = "",
) -> None:
    assert (
        0 <= mismatch_threshold <= 1
    ), f"{mismatch_threshold=} must be between 0 and 1"
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        no_mismatch_info = f"[{test_case}]: has no mismatch"
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(no_mismatch_info)
        else:
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
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    print(mismatch_info)
            else:
                print(mismatch_info)
            return
        else:
            raise type(e)(
                f"\n>>>>>>>  Torch Error Message: \n\n{error_msg}\n\n"
                f">>>>>>>  Mismatch Detailed Info: \n\n{mismatch_info}\n\n"
            ) from e


@torch.no_grad
def calc_inf_norm(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    dtype = max_fp_dtype(a.dtype, b.dtype, torch.float32)
    return safe_subtract(a.to(dtype), b.to(dtype)).norm(p=float("inf")).item()
