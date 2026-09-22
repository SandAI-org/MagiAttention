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

"""Agile harness for ``ffa_bwd_sm100_inner_loop_k``.

# DEVIATION: this harness lives at the repo root instead of tests/
# Reason: prompt.md asks for MagiAttention/test.py during kernel bring-up
# Tracking: move under tests/test_attn once the inner-loop-K kernel replaces the hack

Forward always uses the existing cutedsl ``flex_flash_attn_func``. Backward
always uses ``ffa_bwd_sm100_inner_loop_k``, so commenting out the hack in that
module switches this script onto the new kernel.

Usage:
    python test.py smoke
    python test.py correct
    python test.py bench
"""

import sys
from collections.abc import Callable
from functools import partial

import torch

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.kernel.cutedsl.ffa_bwd_sm100_inner_loop_k import (
    ffa_bwd_sm100_inner_loop_k,
)
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP
from magi_attention.kernel.cutedsl.flex_flash_attn import flex_flash_attn_func
from magi_attention.testing.precision import (
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.testing.ref_attn import ref_attn_func
from magi_attention.utils.general import make_attn_mask_from_ffa_args

NUM_HEADS_Q = 64
NUM_HEADS_KV = 8

# varlen_full_1k from tests/test_attn/test_flex_flash_attn.py
_VARLEN_FULL_1K: list[list[int]] = [
    [0, 366],
    [366, 391],
    [391, 471],
    [471, 835],
    [835, 984],
    [984, 1005],
    [1005, 1017],
    [1017, 1020],
    [1020, 1023],
    [1023, 1024],
]

_GRAD_RTOL = {
    "dq": {torch.bfloat16: 0.3, torch.float16: 0.2},
    "dk": {torch.bfloat16: 0.15, torch.float16: 0.08},
    "dv": {torch.bfloat16: 0.05, torch.float16: 0.05},
}
_GRAD_RTOL_DEFAULT = {"dq": 0.2, "dk": 0.08, "dv": 0.05}


def _ref_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float,
    high_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    out, _meta = ref_attn_func(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        mask=mask,
        softmax_scale=softmax_scale,
        layout="thd",
        backend="sdpa",
        high_precision=high_precision,
        return_lse=True,
    )
    out.backward(do)
    assert q_ref.grad is not None and k_ref.grad is not None and v_ref.grad is not None
    return q_ref.grad, k_ref.grad, v_ref.grad


def _assert_grad(
    name: str,
    actual: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    dtype: torch.dtype,
    test_case: str,
) -> None:
    rtol = _GRAD_RTOL[name].get(dtype, _GRAD_RTOL_DEFAULT[name])
    atol = EPSILON
    actual_norm = calc_inf_norm(actual, high)
    ref_norm = calc_inf_norm(low, high)
    limit = max(0.0, NORM_RTOL_RATIO * ref_norm)
    if actual_norm > limit:
        raise AssertionError(
            f"{test_case} => {name}: Linf {actual_norm} > "
            f"{NORM_RTOL_RATIO} x low-vs-high Linf {ref_norm}"
        )
    mismatch = extract_mismatch_threshold(
        actual=low,
        expected=high,
        atol=atol,
        rtol=rtol,
        mismatch_thres_ratio=MISMATCH_THRES_RATIO,
        min_mismatch_thres=0.0,
        max_mismatch_thres=MAX_MISMATCH_THRES,
    )
    assert_close(
        actual,
        high,
        atol=atol,
        rtol=rtol,
        mismatch_threshold=mismatch,
        test_case=f"{test_case} => {name}",
    )


def _run_case(
    name: str,
    seqlen: int,
    ranges: list[list[int]],
    dtype: torch.dtype,
    head_dim: int = 128,
) -> None:
    """Forward with the existing kernel, backward with the new entry, check grads."""
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(seqlen, NUM_HEADS_Q, head_dim, device=device, dtype=dtype)
    k = torch.randn(seqlen, NUM_HEADS_KV, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen, NUM_HEADS_KV, head_dim, device=device, dtype=dtype)
    do = torch.randn_like(q)
    q_ranges = AttnRanges.from_ranges(ranges)
    k_ranges = AttnRanges.from_ranges(ranges)
    attn_type_map = [0] * len(q_ranges)
    q_ranges_tensor = q_ranges.to_tensor(device)
    k_ranges_tensor = k_ranges.to_tensor(device)
    softmax_scale = head_dim**-0.5

    out, meta = flex_flash_attn_func(
        q,
        k,
        v,
        q_ranges=q_ranges_tensor,
        k_ranges=k_ranges_tensor,
        mask_types=MT_MAP.full,
        softmax_scale=softmax_scale,
    )
    assert meta.lse is not None

    dq, dk, dv = ffa_bwd_sm100_inner_loop_k(
        q,
        k,
        v,
        out,
        do,
        meta.lse,
        q_ranges_tensor,
        k_ranges_tensor,
        mask_types=MT_MAP.full,
        softmax_scale=softmax_scale,
        tile_m=128,
        tile_n=64,
        head_dim=head_dim,
    )

    mask = make_attn_mask_from_ffa_args(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        total_seqlen_q=seqlen,
        total_seqlen_k=seqlen,
        device=device,
    )
    dq_hi, dk_hi, dv_hi = _ref_grads(
        q, k, v, do, mask, softmax_scale, high_precision=True
    )
    dq_lo, dk_lo, dv_lo = _ref_grads(
        q, k, v, do, mask, softmax_scale, high_precision=False
    )
    test_case = f"{name} dtype={dtype}"
    _assert_grad("dq", dq, dq_hi, dq_lo, dtype, test_case)
    _assert_grad("dk", dk, dk_hi, dk_lo, dtype, test_case)
    _assert_grad("dv", dv, dv_hi, dv_lo, dtype, test_case)
    print(f"PASS {test_case}")


def smoke() -> None:
    """Full 1024 x 1024, one range, bf16. Short functional check."""
    _run_case(
        name="smoke_full_1024",
        seqlen=1024,
        ranges=[[0, 1024]],
        dtype=torch.bfloat16,
    )


def correct() -> None:
    """Full and varlen-full masks, bf16 and fp16, against the fp64 reference."""
    cases: list[tuple[str, int, list[list[int]]]] = [
        ("full_1024", 1024, [[0, 1024]]),
        ("varlen_full_1k", 1024, _VARLEN_FULL_1K),
    ]
    for dtype in (torch.bfloat16, torch.float16):
        for name, seqlen, ranges in cases:
            _run_case(name=name, seqlen=seqlen, ranges=ranges, dtype=dtype)


def _time_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench() -> None:
    """Time backward only and print TFLOPS.

    Numbers come from the HACK_NON_SWAP_LOOP delegate (existing K-outer
    FFABwdSm100), not from the inner-loop-K kernel.
    """
    # Local import: baselines.utils pulls flex_attention, which smoke/correct do not need.
    from exps.attn.baselines.utils import calculate_attn_flops

    device = "cuda"
    head_dim = 128
    dtype = torch.bfloat16
    softmax_scale = head_dim**-0.5
    warmup = 5
    iters = 10
    print(
        "HACK_NON_SWAP_LOOP is active: TFLOPS are existing K-outer FFABwdSm100, "
        f"not inner-loop-K. heads_q={NUM_HEADS_Q} heads_kv={NUM_HEADS_KV} "
        f"head_dim={head_dim} dtype={dtype}"
    )
    print(f"{'seqlen':>8} {'ms':>10} {'TFLOPS':>10}")
    for seqlen in [2048 * i for i in range(1, 8)]:
        torch.manual_seed(0)
        q = torch.randn(seqlen, NUM_HEADS_Q, head_dim, device=device, dtype=dtype)
        k = torch.randn(seqlen, NUM_HEADS_KV, head_dim, device=device, dtype=dtype)
        v = torch.randn(seqlen, NUM_HEADS_KV, head_dim, device=device, dtype=dtype)
        do = torch.randn_like(q)
        q_ranges = AttnRanges.from_ranges([[0, seqlen]])
        k_ranges = AttnRanges.from_ranges([[0, seqlen]])
        q_ranges_tensor = q_ranges.to_tensor(device)
        k_ranges_tensor = k_ranges.to_tensor(device)
        out, meta = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            mask_types=MT_MAP.full,
            softmax_scale=softmax_scale,
        )
        assert meta.lse is not None
        run_bwd = partial(
            ffa_bwd_sm100_inner_loop_k,
            q,
            k,
            v,
            out,
            do,
            meta.lse,
            q_ranges_tensor,
            k_ranges_tensor,
            mask_types=MT_MAP.full,
            softmax_scale=softmax_scale,
            tile_m=128,
            tile_n=64,
            head_dim=head_dim,
        )
        ms = _time_ms(run_bwd, warmup=warmup, iters=iters)
        flops = calculate_attn_flops(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[AttnMaskType.FULL],
            total_seqlen_q=seqlen,
            num_heads_q=NUM_HEADS_Q,
            head_dim=head_dim,
        )["bwd"]
        tflops = flops / ms * 1e-9
        print(f"{seqlen:8d} {ms:10.3f} {tflops:10.2f}")


_MODES = {
    "smoke": smoke,
    "correct": correct,
    "bench": bench,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _MODES:
        print("usage: python test.py smoke|correct|bench", file=sys.stderr)
        raise SystemExit(2)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    _MODES[sys.argv[1]]()


if __name__ == "__main__":
    main()
