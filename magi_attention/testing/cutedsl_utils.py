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

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from unittest import mock

import torch

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.testing import ref_attn_func
from magi_attention.testing.dist_common import DistTestBase
from magi_attention.testing.precision import (
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_ROUNDING_NOISE_RATIO,
    NORM_RTOL_RATIO,
    ROUNDING_NOISE_PERTURBATION,
    assert_close,
    assert_equal,
    calc_inf_norm,
    calc_rounding_noise_norm,
    extract_mismatch_threshold,
)
from magi_attention.utils import make_attn_mask_from_ffa_args
from magi_attention.utils.general import set_random_seed, str2seed

Ranges = list[list[int]]
Relations = list[tuple[list[int], list[int], int]]

# Major arches whose kernels consume q/k ranges natively: overlapping q ranges
# (atomic O merge), per-range mask types, RangeMerge, sink, non-input out dtype.
SM100_FAMILY_MAJOR_ARCHS = (10, 11)
SM100_FAMILY_REQUIRED = "requires the SM100/SM110 CuTeDSL kernels"


def is_sm100_family() -> bool:
    return (
        torch.cuda.is_available() and get_device_arch()[1] in SM100_FAMILY_MAJOR_ARCHS
    )


# A cold run JIT-compiles every kernel variant of the sweep.
DIST_TEST_TIMEOUT_SECONDS = 3600


class CuteDSLDistTestBase(DistTestBase):
    """Spread independent parameter combinations over all visible GPUs."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        super().setUp()

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
MQA_HD128 = HeadConfig("mqa_hd128", _NUM_HEADS_Q, 1, 128)


def seed_test_case(test_case: str) -> None:
    """Seed every RNG from the test-case id, so a case reproduces on its own."""
    set_random_seed(str2seed(test_case))


@contextmanager
def record_fwd_kernels() -> Iterator[list[FFAFwdSm100]]:
    """Record the SM100 fwd kernel objects built within the context under an
    empty fwd compile cache, as :func:`record_bwd_kernels` does for the bwd."""
    built: list[FFAFwdSm100] = []
    init = FFAFwdSm100.__init__

    def record(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    with mock.patch.object(FFAFwdSm100, "__init__", record), mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ):
        yield built


@contextmanager
def record_bwd_kernels() -> Iterator[list[FFABwdSm100]]:
    """Record the SM100 bwd kernel objects built within the context.

    The specialization is only visible on the kernel object, which a JIT cache
    hit never builds, so the context also swaps in empty fwd and bwd compile
    caches.
    """
    built: list[FFABwdSm100] = []
    init = FFABwdSm100.__init__

    def record(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    with mock.patch.object(FFABwdSm100, "__init__", record), mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ), mock.patch.object(_flex_flash_attn_bwd, "compile_cache", JITCache()):
        yield built


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

# fp32 rounding leaves up to ~7 fp32 eps of max|ref_hi| where ref_hi is exactly 0.
_MISMATCH_ATOL_RATIO = 16 * torch.finfo(torch.float32).eps


def _torch_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    sink: torch.Tensor | None,
    mask: torch.Tensor,
    high_precision: bool,
    softcap: float,
    softmax_scale: float | None,
) -> AttnTensors:
    q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out, _ = ref_attn_func(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        mask=mask,
        sink=sink,
        softcap=softcap,
        softmax_scale=softmax_scale,
        layout="thd",
        # The sdpa reference supports neither sink nor softcap.
        backend="sdpa" if sink is None and softcap == 0.0 else "torch",
        high_precision=high_precision,
    )
    grads = torch.autograd.grad(out, (q_ref, k_ref, v_ref), do.to(out.dtype))
    return AttnTensors(out, *grads)


def _check_one(
    name: str,
    actual: torch.Tensor,
    ref_hi: torch.Tensor,
    ref_lo: torch.Tensor,
    dtype: torch.dtype,
    test_case: str,
) -> None:
    """Check maximum error and the fraction of elements exceeding tolerance."""
    actual = actual.to(ref_hi.dtype)
    rtol = _RTOL[name][dtype]
    norm = calc_inf_norm(actual, ref_hi)
    ref_norm = calc_inf_norm(ref_lo, ref_hi)
    noise_norm = calc_rounding_noise_norm(
        ref_hi, perturbation=ROUNDING_NOISE_PERTURBATION
    )
    bound = NORM_RTOL_RATIO * ref_norm + NORM_ROUNDING_NOISE_RATIO * noise_norm
    assert norm <= bound, f"{test_case} => {name}: {norm=}, {bound=}"
    finite_ref = ref_hi[torch.isfinite(ref_hi)]
    ref_scale = finite_ref.abs().max().item() if finite_ref.numel() else 0.0
    atol = _MISMATCH_ATOL_RATIO * ref_scale
    mismatch_threshold = extract_mismatch_threshold(
        actual=ref_lo,
        expected=ref_hi,
        atol=atol,
        rtol=rtol,
        mismatch_thres_ratio=MISMATCH_THRES_RATIO,
        max_mismatch_thres=MAX_MISMATCH_THRES,
    )
    assert_close(
        actual,
        ref_hi,
        atol=atol,
        rtol=rtol,
        mismatch_threshold=mismatch_threshold,
        test_case=f"{test_case} => {name}",
        print_rank=-1,
    )


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
    softcap: float = 0.0,
    softmax_scale: float | None = None,
) -> None:
    """Compare a call against fp64 and input-dtype torch references.

    All tensors use the packed thd layout; ``q_ranges``/``k_ranges`` and
    ``attn_type_map`` describe the mask in FFA form. ``softcap`` and
    ``softmax_scale`` take the meaning of the kernel arguments.
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
            q=q,
            k=k,
            v=v,
            do=do,
            sink=sink,
            mask=mask,
            high_precision=high_precision,
            softcap=softcap,
            softmax_scale=softmax_scale,
        )
        for high_precision in (True, False)
    ]
    ref_hi, ref_lo = (dict(ref.named()) for ref in references)
    for name, value in actual.named():
        _check_one(name, value, ref_hi[name], ref_lo[name], q.dtype, test_case)


def torch_reference_out_lse(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: Ranges,
    k_ranges: Ranges,
    sink: torch.Tensor | None = None,
    attn_type_map: list[int] | None = None,
    softcap: float = 0.0,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64-computed fp32 ``(out, lse)``, thd layout.

    ``attn_type_map`` defaults to a full mask for every relation. Rows without
    any key get ``out = 0`` and the sink-only (or ``-inf``) LSE.
    """
    mask = make_attn_mask_from_ffa_args(
        q_ranges=AttnRanges.from_ranges(q_ranges),
        k_ranges=AttnRanges.from_ranges(k_ranges),
        attn_type_map=(
            [MT_MAP.full] * len(q_ranges) if attn_type_map is None else attn_type_map
        ),
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
        softcap=softcap,
        softmax_scale=softmax_scale,
        layout="thd",
        backend="torch",
        high_precision=True,
        return_lse=True,
    )
    assert meta.lse is not None
    return out, meta.lse


def dsink_reference(
    out: torch.Tensor, dout: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    """fp64 ``dsink[s, h] = -sum_q sum_d(O * dO) * exp(sink[s, h] - LSE[q, h])``
    over thd ``out``/``dout`` and ``[total_q, num_heads]`` ``lse``.

    Rows with LSE = -inf have no keys and contribute nothing.
    """
    has_keys = lse != -math.inf
    delta = (
        out.double().masked_fill(~has_keys[..., None], 0)
        * dout.double().masked_fill(~has_keys[..., None], 0)
    ).sum(dim=-1)
    probability = torch.exp(sink.double()[None] - lse.double()[:, None])
    probability = probability.masked_fill(~has_keys[:, None], 0)
    return -(delta[:, None] * probability).sum(dim=0).float()


DSINK_ATOL = 2e-4
DSINK_RTOL = 0.1


def assert_cuda_graph_matches_eager(
    fn: Callable[..., Sequence[torch.Tensor]],
    capture_inputs: Sequence[torch.Tensor],
    replay_inputs_list: Sequence[Sequence[torch.Tensor]],
    *,
    check: Callable[[int, Sequence[torch.Tensor], Sequence[torch.Tensor]], None]
    | None = None,
    test_case: str = "",
) -> list[tuple[torch.Tensor, ...]]:
    """Capture ``fn`` on a cloned static buffer set and verify every replay."""
    static = [t.clone() for t in capture_inputs]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn(*static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn(*static)

    replayed: list[tuple[torch.Tensor, ...]] = []
    for step, inputs in enumerate(replay_inputs_list):
        for dst, src in zip(static, inputs, strict=True):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        got = tuple(t.clone() for t in static_out)
        eager = tuple(fn(*inputs))
        torch.cuda.synchronize()
        if check is not None:
            check(step, got, eager)
        else:
            for idx, (g, w) in enumerate(zip(got, eager, strict=True)):
                assert_equal(
                    g,
                    w,
                    test_case=f"{test_case}[replay={step}][{idx}]"
                    if test_case
                    else f"tensor[{idx}]",
                    print_rank=-1,
                    print_no_mismatch=False,
                )
        replayed.append(got)
    return replayed


def overlapping_relations(
    len_qs: Sequence[int], coverage: int, mask_types: Sequence[int]
) -> tuple[Relations, int, int]:
    """Per q length, ``coverage`` relations over shifted, unaligned q ranges,
    each against its own K range; returns (relations, total_q, total_k)."""
    relations: Relations = []
    q_cursor, k_cursor, i = 0, 0, 0
    for len_q in len_qs:
        q_start = q_cursor + 3
        for c in range(coverage):
            len_k = len_q + 37
            relations.append(
                (
                    [q_start + c, q_start + c + len_q],
                    [k_cursor, k_cursor + len_k],
                    mask_types[i % len(mask_types)],
                )
            )
            k_cursor += len_k
            i += 1
        q_cursor = q_start + coverage + len_q
    return relations, q_cursor + 5, k_cursor
