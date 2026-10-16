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

"""Shared configs, input builders and reference checks for the CuTeDSL FFA tests."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import torch

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.testing import ref_attn_func
from magi_attention.testing.dist_common import DistTestBase
from magi_attention.testing.precision import (
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_ROUNDING_NOISE_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    calc_rounding_noise_norm,
    extract_mismatch_threshold,
)
from magi_attention.testing.utils import switch_envvars
from magi_attention.utils import make_attn_mask_from_ffa_args
from magi_attention.utils.arch import is_ampere
from magi_attention.utils.general import set_random_seed, str2seed

Ranges = list[list[int]]

# ─────────────────────────────────────────────────────────────────────────────
# Architecture and kernel path
# ─────────────────────────────────────────────────────────────────────────────

# Major arches whose kernels consume q/k ranges natively: overlapping q ranges
# (atomic O merge), per-range mask types, RangeMerge, sink, non-input out dtype.
SM100_FAMILY_MAJOR_ARCHS = (10, 11)
SM100_FAMILY_REQUIRED = "requires the SM100/SM110 CuTeDSL kernels"


def is_sm100_family() -> bool:
    return (
        torch.cuda.is_available() and get_device_arch()[1] in SM100_FAMILY_MAJOR_ARCHS
    )


class KernelPath(Enum):
    """Which arch's kernels serve the call."""

    NATIVE = "native"  # the kernels of the visible device
    SM80 = "sm80"  # the SM80 kernels, whose SASS also runs on newer GPUs


# On Ampere the native kernels already are the SM80 ones.
KERNEL_PATHS = (
    [KernelPath.NATIVE]
    if torch.cuda.is_available() and is_ampere()
    else [KernelPath.NATIVE, KernelPath.SM80]
)

_KERNEL_ARCH_ENVVAR = "MAGI_ATTENTION_FFA_CUTEDSL_ARCH"


@contextmanager
def use_kernel_path(path: KernelPath) -> Iterator[None]:
    """Serve the calls inside the context with the kernels of ``path``.

    ``get_device_arch`` is lru-cached, so the cache is cleared whenever the
    arch override changes.
    """
    if path is KernelPath.NATIVE:
        yield
        return
    switch_back = switch_envvars(
        [_KERNEL_ARCH_ENVVAR], enable_value_dict={_KERNEL_ARCH_ENVVAR: "sm_80"}
    )
    get_device_arch.cache_clear()
    try:
        yield
    finally:
        switch_back()
        get_device_arch.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# Test base and shapes
# ─────────────────────────────────────────────────────────────────────────────

# A cold run JIT-compiles every kernel variant of the sweep.
DIST_TEST_TIMEOUT_SECONDS = 3600


class CuteDSLDistTestBase(DistTestBase):
    """Spread independent parameter combinations over all visible GPUs."""

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @property
    def timeout(self) -> int:
        return DIST_TEST_TIMEOUT_SECONDS

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()


@dataclass(frozen=True)
class HeadConfig:
    name: str
    num_heads_q: int
    num_heads_kv: int
    head_dim: int


# Divisible by every kv head count below.
_NUM_HEADS_Q = 6

MHA_HD64 = HeadConfig("mha_hd64", _NUM_HEADS_Q, _NUM_HEADS_Q, 64)
MHA_HD128 = HeadConfig("mha_hd128", _NUM_HEADS_Q, _NUM_HEADS_Q, 128)
GQA_HD128 = HeadConfig("gqa_hd128", _NUM_HEADS_Q, _NUM_HEADS_Q // 2, 128)
MQA_HD64 = HeadConfig("mqa_hd64", _NUM_HEADS_Q, 1, 64)
MQA_HD128 = HeadConfig("mqa_hd128", _NUM_HEADS_Q, 1, 128)

INPUT_DTYPES = [torch.bfloat16, torch.float16]


def seed_test_case(test_case: str) -> None:
    """Seed every RNG from the test-case id, so a case reproduces on its own."""
    set_random_seed(str2seed(test_case))


def make_qkv(
    q_tokens_shape: tuple[int, ...],
    k_tokens_shape: tuple[int, ...],
    heads: HeadConfig,
    dtype: torch.dtype,
    device: int | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random leaf q/k/v of shape ``(*tokens_shape, num_heads, head_dim)``."""

    def rand(tokens_shape: tuple[int, ...], num_heads: int) -> torch.Tensor:
        shape = (*tokens_shape, num_heads, heads.head_dim)
        return torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    return (
        rand(q_tokens_shape, heads.num_heads_q),
        rand(k_tokens_shape, heads.num_heads_kv),
        rand(k_tokens_shape, heads.num_heads_kv),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ranges and mask types
# ─────────────────────────────────────────────────────────────────────────────


def ranges_from_lengths(lengths: Sequence[int], start: int = 0) -> Ranges:
    """Consecutive ``[start, end)`` ranges with the given lengths."""
    ranges = []
    for length in lengths:
        ranges.append([start, start + length])
        start += length
    return ranges


def to_ranges_tensor(ranges: Ranges, device: int | str) -> torch.Tensor:
    return torch.tensor(ranges, dtype=torch.int32, device=device)


def mask_types_arg(attn_type_map: list[int], device: int | str) -> int | torch.Tensor:
    """Pass full/causal shared by every range as the scalar of the statically
    specialized kernels, anything else as the per-range ``int32[R]`` tensor."""
    first = attn_type_map[0]
    if first in (MT_MAP.full, MT_MAP.causal) and all(
        mask_type == first for mask_type in attn_type_map
    ):
        return first
    return torch.tensor(attn_type_map, dtype=torch.int32, device=device)


def cycle_mask_types(pattern: Sequence[int], num_ranges: int) -> list[int]:
    """Repeat ``pattern`` over ``num_ranges`` relations."""
    return [pattern[i % len(pattern)] for i in range(num_ranges)]


# ─────────────────────────────────────────────────────────────────────────────
# Reference comparison
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttnTensors:
    """Forward output and gradients of one call, in the packed thd layout."""

    out: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor

    def named(self) -> list[tuple[str, torch.Tensor]]:
        return [("out", self.out), ("dq", self.dq), ("dk", self.dk), ("dv", self.dv)]


# Relative tolerance of each compared tensor, by input dtype.
_RTOL = {
    "out": {torch.bfloat16: 0.05, torch.float16: 0.05},
    "dq": {torch.bfloat16: 0.3, torch.float16: 0.2},
    "dk": {torch.bfloat16: 0.15, torch.float16: 0.08},
    "dv": {torch.bfloat16: 0.05, torch.float16: 0.05},
}

# Floor of the allowed mismatch ratio. Masked-out gradient positions hold
# ~1e-7 noise where the reference is exactly 0: an infinite relative diff that
# inflates the ratio on small shapes. The Linf-norm gate stays the primary one.
_MIN_MISMATCH_THRES = {
    "out": 5e-3,
    "dq": 1e-2,
    "dk": 1e-2,
    "dv": 5e-3,
}


def _torch_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    sink: torch.Tensor | None,
    mask: torch.Tensor,
    high_precision: bool,
) -> AttnTensors:
    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out, _ = ref_attn_func(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        mask=mask,
        sink=None if sink is None else sink.detach(),
        layout="thd",
        # The sdpa reference has no sink support.
        backend="sdpa" if sink is None else "torch",
        high_precision=high_precision,
    )
    dq, dk, dv = torch.autograd.grad(out, (q_ref, k_ref, v_ref), do.to(out.dtype))
    return AttnTensors(out=out, dq=dq, dk=dk, dv=dv)


def _check_one(
    name: str,
    actual: torch.Tensor,
    ref_hi: torch.Tensor,
    ref_lo: torch.Tensor,
    dtype: torch.dtype,
    test_case: str,
) -> list[str]:
    """Bound the error by the low-precision reference's own error: an Linf
    norm gate (flash-attn style) plus an rtol mismatch ratio (torch style).

    The Linf gate also allows the rounding noise of ``ref_hi`` in its stored
    dtype, as ``test_pipeline`` and the FlashAttention tests do.
    """
    actual = actual.to(ref_hi.dtype)
    rtol = _RTOL[name][dtype]
    errors = []
    norm = calc_inf_norm(actual, ref_hi)
    ref_norm = calc_inf_norm(ref_lo, ref_hi)
    noise_norm = calc_rounding_noise_norm(ref_hi)
    if norm > NORM_RTOL_RATIO * ref_norm + NORM_ROUNDING_NOISE_RATIO * noise_norm:
        errors.append(
            f"{test_case} => {name}: {norm=} > {NORM_RTOL_RATIO} x {ref_norm=} "
            f"+ {NORM_ROUNDING_NOISE_RATIO} x {noise_norm=}"
        )
    mismatch_threshold = extract_mismatch_threshold(
        actual=ref_lo,
        expected=ref_hi,
        atol=EPSILON,
        rtol=rtol,
        mismatch_thres_ratio=MISMATCH_THRES_RATIO,
        min_mismatch_thres=_MIN_MISMATCH_THRES[name],
        max_mismatch_thres=MAX_MISMATCH_THRES,
    )
    try:
        assert_close(
            actual,
            ref_hi,
            atol=EPSILON,
            rtol=rtol,
            mismatch_threshold=mismatch_threshold,
            test_case=f"{test_case} => {name}",
            print_rank=-1,
        )
    except AssertionError as e:
        errors.append(str(e))
    return errors


def assert_close_to_torch_ref(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    actual: AttnTensors,
    q_ranges: Ranges,
    k_ranges: Ranges,
    attn_type_map: list[int],
    test_case: str,
    sink: torch.Tensor | None = None,
) -> None:
    """Compare a call against fp64 and input-dtype torch references.

    All tensors use the packed thd layout; ``q_ranges``/``k_ranges`` and
    ``attn_type_map`` describe the mask in FFA form.
    """
    mask = make_attn_mask_from_ffa_args(
        q_ranges=AttnRanges.from_ranges(q_ranges),
        k_ranges=AttnRanges.from_ranges(k_ranges),
        attn_type_map=attn_type_map,
        total_seqlen_q=q.shape[0],
        total_seqlen_k=k.shape[0],
        device=q.device,
    )
    references = [
        _torch_reference(
            q=q, k=k, v=v, do=do, sink=sink, mask=mask, high_precision=high_precision
        )
        for high_precision in (True, False)
    ]
    ref_hi, ref_lo = (dict(ref.named()) for ref in references)
    errors = []
    for name, value in actual.named():
        errors += _check_one(
            name, value, ref_hi[name], ref_lo[name], q.dtype, test_case
        )
    if errors:
        raise AssertionError("\n\n".join(errors))


def torch_reference_out_lse(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: Ranges,
    k_ranges: Ranges,
    sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64-computed fp32 ``(out, lse)`` of full-mask relations, thd layout.

    Rows without any key get ``out = 0`` and the sink-only (or ``-inf``) LSE.
    """
    mask = make_attn_mask_from_ffa_args(
        q_ranges=AttnRanges.from_ranges(q_ranges),
        k_ranges=AttnRanges.from_ranges(k_ranges),
        attn_type_map=[MT_MAP.full] * len(q_ranges),
        total_seqlen_q=q.shape[0],
        total_seqlen_k=k.shape[0],
        device=q.device,
    )
    out, meta = ref_attn_func(
        q=q.detach().float(),
        k=k.detach().float(),
        v=v.detach().float(),
        mask=mask,
        sink=sink,
        layout="thd",
        backend="torch",
        high_precision=True,
        return_lse=True,
    )
    assert meta.lse is not None
    return out, meta.lse
