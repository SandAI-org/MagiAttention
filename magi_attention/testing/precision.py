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
from typing import overload

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch.nn.attention import SDPBackend, sdpa_kernel

from magi_attention.functional.utils import safe_softmax, safe_subtract
from magi_attention.utils import max_fp_dtype, to_higher_fp_dtype

if version.parse(torch.__version__) > version.parse("2.4"):
    # NOTE: in testing, we should explicitly allow bf16/fp16 reduction for sdpa
    # by setting `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`
    # due to the new feature since torch2.5:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

# usage: to avoid division by zero in numerical calculation and assert-close testing
EPSILON = 1e-8

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


def _extract_mismatch_info(error_msg: str) -> tuple[int, int, float]:
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
) -> float:
    mismatch_threshold = 0.0
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as e:
        error_msg = str(e)
        _, _, mismatch_threshold = _extract_mismatch_info(error_msg)

    # scale it by `mismatch_thres_ratio`, and clamp it in [0, 1]
    return min(max(mismatch_threshold * mismatch_thres_ratio, 0.0), 1.0)


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
        mismatched_elements, total_elements, mismatch_ratio = _extract_mismatch_info(
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


@overload
def _attn_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    v: None,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, float]:
    ...


@overload
def _attn_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    ...


def _attn_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | None,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, float]:
    # prepare softmax scale
    softmax_scale = q.size(-1) ** (-0.5) if softmax_scale is None else softmax_scale
    assert softmax_scale is not None  # mypy

    # repeat k,v heads
    nhq, nhk = q.size(-2), k.size(-2)
    assert nhq % nhk == 0
    rep_nhk = nhq // nhk
    q = rearrange(q, "s hq d -> hq s d")  # Q
    k = repeat(k, "s hk d -> (hk rep) d s", rep=rep_nhk)  # K.T
    if v is not None:
        v = repeat(v, "s hk d -> (hk rep) s d", rep=rep_nhk)  # V

    # prepare bias
    # where bias.shape = [1, sq, sk]
    bias = torch.zeros_like(mask, dtype=q.dtype, device=q.device)
    bias.masked_fill_(mask.logical_not(), float("-inf")).unsqueeze_(0)

    return q, k, v, bias, softmax_scale


@torch.no_grad
def calc_attn_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
):
    (q, k, _, bias, softmax_scale) = _attn_pre_process(
        q=q,
        k=k,
        v=None,
        mask=mask,
        softmax_scale=softmax_scale,
    )

    # calculate lse
    lse = (
        # apply `S = Q x K.T * scale + bias`
        # where S.shape = [nhq, sq, sk]
        to_higher_fp_dtype(
            q @ k * softmax_scale + bias,
            lowest_precision=torch.float32,
        )
        # apply row-wise lse `LSE = logsumexp(S, dim=-1)`
        # where LSE.shape = [nhq, sq]
        .logsumexp(dim=-1)
        # transpose and make contiguous
        # where LSE.shape = [sq, nhq]
        .t().contiguous()
    )

    return lse


def _ref_attn_sdpa_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if return_lse:
        lse = calc_attn_lse(
            q,
            k,
            mask,
            softmax_scale,
        )
    else:
        lse = None

    q = rearrange(q, "t h d -> 1 h t d")
    k = rearrange(k, "t h d -> 1 h t d")
    v = rearrange(v, "t h d -> 1 h t d")

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            enable_gqa=True,
            scale=softmax_scale,
        )

    out = rearrange(out, "1 h t d -> t h d")

    return out, lse


def _ref_attn_torch_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    (q, k, v, bias, softmax_scale) = _attn_pre_process(
        q=q,
        k=k,
        v=v,
        mask=mask,
        softmax_scale=softmax_scale,
    )

    # apply `S = Q x K.T * scale + bias`
    # where S.shape = [nhq, sq, sk]
    s = to_higher_fp_dtype(
        q @ k * softmax_scale + bias,
        lowest_precision=torch.float32,
    )

    if return_lse:
        # apply row-wise lse `LSE = logsumexp(S, dim=-1)`
        # where LSE.shape = [nhq, sq]
        # and then transpose and make contiguous
        # where LSE.shape = [sq, nhq]
        with torch.no_grad():
            lse = s.logsumexp(dim=-1).t().contiguous()
    else:
        lse = None

    # apply row-wise softmax `P = softmax(S, dim=-1)`
    # where P.shape = [nhq, sq, sk]
    p = safe_softmax(s, dim=-1).to(v.dtype)

    # apply `O = P x V`
    # where O.shape = [nhq, sq, d]
    out = p @ v

    # rearrange and make contiguous
    # where O.shape = [sq, nhq, d]
    out = rearrange(out, "nhq sq d -> sq nhq d").contiguous()

    return out, lse


def ref_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    layout: str = "thd",
    backend: str = "sdpa",
    high_precision: bool = False,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert layout in ("thd",), f"Unsupported layout: {layout}"
    assert softcap == 0.0, "non-zero softcap is not supported by now"

    # maybe cast input to high precision
    org_dtype = q.dtype
    lse_dtype = max_fp_dtype(org_dtype, torch.float32)
    if high_precision:  # use fp64 as ground-truth
        q = q.to(torch.float64)
        k = k.to(torch.float64)
        v = v.to(torch.float64)

    # apply reference attention with specified backend
    match backend:
        case "sdpa":
            assert sink is None, "sink is not supported for sdpa backend by now"
            out, lse = _ref_attn_sdpa_impl(
                q=q,
                k=k,
                v=v,
                mask=mask,
                softmax_scale=softmax_scale,
                return_lse=return_lse,
            )
        case "torch":
            out, lse = _ref_attn_torch_impl(
                q=q,
                k=k,
                v=v,
                sink=sink,
                mask=mask,
                softmax_scale=softmax_scale,
                return_lse=return_lse,
            )
        case _:
            raise NotImplementedError(f"Unsupported backend: {backend}")

    # maybe cast output back to original dtype
    out = out.to(org_dtype)
    if return_lse:
        assert lse is not None  # mypy
        lse = lse.to(lse_dtype)

    return out, lse
