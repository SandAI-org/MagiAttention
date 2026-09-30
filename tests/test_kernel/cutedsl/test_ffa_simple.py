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

"""Smoke-test suite for the forked cutedsl kernel.

Covers the simplest training-relevant cases so that after each round of
changes we can quickly verify correctness has not regressed:

  * Non-varlen fwd+bwd: full / causal  x  MHA / GQA / MQA
  * Varlen (packed cu_seqlens) fwd+bwd: full / causal  x  MHA / GQA / MQA

Run:
    pytest tests/test_kernel/cutedsl/test_ffa_simple.py -v
"""

import math
import random
from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import cutlass
import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.functional.utils import sink_bwd
from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.ffa_bwd_dsink import bwd_dsink
from magi_attention.kernel.cutedsl.ffa_bwd_postprocess import (
    FFABwdPostProcess,
    bwd_postprocess,
    bwd_postprocess_rowmajor,
)
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import (
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.testing.utils import switch_envvars
from magi_attention.utils import make_attn_mask_from_ffa_args
from magi_attention.utils.arch import is_ampere
from magi_attention.utils.dtype import to_cute_dtype

# ─────────────────────────────────────────────────────────────────────────────
# SM80 kernel selection
# ─────────────────────────────────────────────────────────────────────────────
#
# The SM80 kernel path is selected via the MAGI_ATTENTION_FFA_CUTEDSL_ARCH override
# rather than the real device capability, so it can be exercised on newer GPUs
# (the compiled SM80 SASS runs fine on sm90/sm100). get_device_arch() is
# lru_cached, so we must clear the cache whenever we toggle the override.


@contextmanager
def _record_bwd_postprocess() -> Iterator[list[FFABwdPostProcess]]:
    """Record every :class:`FFABwdPostProcess` built, bypassing the JIT cache."""
    built: list[FFABwdPostProcess] = []
    init = FFABwdPostProcess.__init__

    def record(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    with mock.patch.object(bwd_postprocess, "compile_cache", {}), mock.patch.object(
        FFABwdPostProcess, "__init__", record
    ):
        yield built


@contextmanager
def _maybe_force_sm80(enabled: bool):
    """Force the FFA kernel path to SM80 within the context when ``enabled``."""
    if not enabled:
        yield
        return

    switch_back = switch_envvars(
        ["MAGI_ATTENTION_FFA_CUTEDSL_ARCH"],
        enable_value_dict={"MAGI_ATTENTION_FFA_CUTEDSL_ARCH": "sm_80"},
    )
    get_device_arch.cache_clear()
    try:
        yield
    finally:
        switch_back()
        get_device_arch.cache_clear()


# per-tensor relative tolerance (fa-style), keyed by dtype where it matters
_RTOL = {
    "o": {torch.bfloat16: 0.05, torch.float16: 0.05},
    "dq": {torch.bfloat16: 0.3, torch.float16: 0.2},
    "dk": {torch.bfloat16: 0.15, torch.float16: 0.08},
    "dv": {torch.bfloat16: 0.05, torch.float16: 0.05},
    "dsink": {torch.bfloat16: 0.15, torch.float16: 0.15},
}

# per-tensor fa-style Linf-norm ratio against the low-precision reference.
# NOTES: dsink reduces bf16/fp16 out*dout over every query row, so its rounding
# error is a global sum rather than per-element and needs twice the headroom
_NORM_RTOL_RATIO = {
    "o": NORM_RTOL_RATIO,
    "dq": NORM_RTOL_RATIO,
    "dk": NORM_RTOL_RATIO,
    "dv": NORM_RTOL_RATIO,
    "dsink": NORM_RTOL_RATIO * 2,
}

# per-tensor lower bound on the allowed mismatch ratio. The kernel writes tiny
# (~1e-7) fp noise into masked-out gradient positions where the sdpa reference
# is exactly 0, which shows up as an "inf" relative diff and inflates the
# mismatch ratio on the small smoke-test shapes. A small floor absorbs this
# without weakening the primary Linf-norm gate (mirrors the reference test's
# ``err_ratio_dict`` idiom).
_MIN_MISMATCH_THRES = {
    "o": 5e-3,
    "dq": 1e-2,
    "dk": 1e-2,
    "dv": 5e-3,
    "dsink": 5e-3,
}


class TestFfaSimple(DistTestBase):
    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @property
    def timeout(self) -> int:
        return 3600

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    # ─────────────────────────────────────────────────────────────────────
    # reference comparison (torch high/low precision) over a packed thd layout
    # ─────────────────────────────────────────────────────────────────────

    def _compare(
        self,
        name: str,
        actual: torch.Tensor,
        ref_hi: torch.Tensor,
        ref_lo: torch.Tensor,
        rtol: float,
        test_case: str,
        err_msg_list: list[str],
    ) -> None:
        # fa style with Linf norm
        norm = calc_inf_norm(actual, ref_hi)
        ref_norm = calc_inf_norm(ref_lo, ref_hi)
        try:
            self.assertLessEqual(
                norm,
                _NORM_RTOL_RATIO[name] * ref_norm,
                msg=(
                    f"For {test_case=}: {name} {norm=} should be no greater than "
                    f"{_NORM_RTOL_RATIO[name]} x {ref_norm=}"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        thres = extract_mismatch_threshold(
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
                mismatch_threshold=thres,
                test_case=f"{test_case} => {name}",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

    def assert_close_to_torch_ref(
        self,
        *,
        q_thd: torch.Tensor,
        k_thd: torch.Tensor,
        v_thd: torch.Tensor,
        do_thd: torch.Tensor,
        out_thd: torch.Tensor,
        dq_thd: torch.Tensor,
        dk_thd: torch.Tensor,
        dv_thd: torch.Tensor,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_type_map: list[int],
        total_seqlen_q: int,
        total_seqlen_k: int,
        dtype: torch.dtype,
        test_case: str,
        sink: torch.Tensor | None = None,
        dsink_thd: torch.Tensor | None = None,
    ) -> None:
        """Compare the kernel out/dq/dk/dv[/dsink] against a torch reference (thd layout).

        The reference is run twice (fp64 high precision + fp16/bf16 low
        precision) so we can derive fa-style norm bounds and torch-style
        mismatch thresholds, then assert closeness for each tensor.
        """
        mask = make_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=q_thd.device,
        )

        def _ref(high_precision: bool):
            q_ref = q_thd.clone().detach().requires_grad_()
            k_ref = k_thd.clone().detach().requires_grad_()
            v_ref = v_thd.clone().detach().requires_grad_()
            sink_ref = None if sink is None else sink.clone().detach().requires_grad_()
            inputs = [t for t in (q_ref, k_ref, v_ref, sink_ref) if t is not None]
            out_ref, _ = ref_attn_func(
                q=q_ref,
                k=k_ref,
                v=v_ref,
                mask=mask,
                sink=sink_ref,
                sink_layout="sh",
                layout="thd",
                backend="sdpa" if sink is None else "torch",
                high_precision=high_precision,
            )
            return out_ref, *torch.autograd.grad(out_ref, inputs, do_thd)

        names = ["o", "dq", "dk", "dv"]
        actuals = [out_thd, dq_thd, dk_thd, dv_thd]
        if sink is not None:
            names.append("dsink")
            actuals.append(dsink_thd)
        refs_hi = _ref(high_precision=True)
        refs_lo = _ref(high_precision=False)

        err_msg_list: list[str] = []
        for name, actual, ref_hi, ref_lo in zip(names, actuals, refs_hi, refs_lo):
            self._compare(
                name=name,
                actual=actual,
                ref_hi=ref_hi,
                ref_lo=ref_lo,
                rtol=_RTOL[name][dtype],
                test_case=test_case,
                err_msg_list=err_msg_list,
            )

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

    # ─────────────────────────────────────────────────────────────────────
    # Non-varlen (dense b,s,h,d): fwd + bwd
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("dtype", [torch.bfloat16, torch.float16])
    @parameterize("mha_type", ["mha", "gqa", "mqa"])
    @parameterize("mask_types", [MT_MAP.full, MT_MAP.causal])
    @parameterize("d", [64, 128])
    @parameterize("force_sm80", [False, True])
    @parameterize("seqlens", [(256, 256), (1024, 1024), (203, 123)])
    def test_non_varlen_fwd_bwd(
        self, seqlens, force_sm80, d, mask_types, mha_type, dtype
    ):
        """Non-varlen flex_flash_attn_func: fwd + bwd for full/causal x MHA/GQA/MQA."""
        if force_sm80 and is_ampere():
            # kernel path is already SM80 on Ampere, no need to force it
            return

        seqlen_q, seqlen_k = seqlens
        device = self.device
        seed = self.seed + seqlen_q + seqlen_k + d + mask_types * 3
        torch.random.manual_seed(seed)
        random.seed(seed)

        batch_size = 4
        nheads = 6
        nheads_kv = {"mha": nheads, "gqa": 3, "mqa": 1}[mha_type]

        q = torch.randn(
            batch_size, seqlen_q, nheads, d, device=device, dtype=dtype
        ).requires_grad_()
        k = torch.randn(
            batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        v = torch.randn(
            batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()

        test_case = (
            f"[RANK {self.rank}][test_non_varlen_fwd_bwd]"
            f"[{force_sm80=}][{seqlen_q=}][{seqlen_k=}][{d=}]"
            f"[{mask_types=}][{mha_type=}][{dtype=}]"
        )

        with _maybe_force_sm80(force_sm80):
            out, _ = flex_flash_attn_func(q, k, v, mask_types=mask_types)
            g = torch.randn_like(out)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)

        # flatten (b, s, h, d) -> (b*s, h, d) and build block-diagonal ranges
        q_ranges = AttnRanges.from_ranges(
            [[i * seqlen_q, (i + 1) * seqlen_q] for i in range(batch_size)]
        )
        k_ranges = AttnRanges.from_ranges(
            [[i * seqlen_k, (i + 1) * seqlen_k] for i in range(batch_size)]
        )
        self.assert_close_to_torch_ref(
            q_thd=rearrange(q.detach(), "b s h d -> (b s) h d"),
            k_thd=rearrange(k.detach(), "b s h d -> (b s) h d"),
            v_thd=rearrange(v.detach(), "b s h d -> (b s) h d"),
            do_thd=rearrange(g, "b s h d -> (b s) h d"),
            out_thd=rearrange(out, "b s h d -> (b s) h d"),
            dq_thd=rearrange(dq, "b s h d -> (b s) h d"),
            dk_thd=rearrange(dk, "b s h d -> (b s) h d"),
            dv_thd=rearrange(dv, "b s h d -> (b s) h d"),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=[mask_types] * batch_size,
            total_seqlen_q=batch_size * seqlen_q,
            total_seqlen_k=batch_size * seqlen_k,
            dtype=dtype,
            test_case=test_case,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Varlen (packed, q/k ranges): fwd + bwd
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("dtype", [torch.bfloat16, torch.float16])
    @parameterize("mha_type", ["mha", "gqa", "mqa"])
    @parameterize("mask_types", [MT_MAP.full, MT_MAP.causal])
    @parameterize("d", [64, 128])
    @parameterize("force_sm80", [False, True])
    @parameterize("seqlen", [128, 512, 1024])
    def test_varlen_fwd_bwd(self, seqlen, force_sm80, d, mask_types, mha_type, dtype):
        """Varlen flex_flash_attn_func (packed q/k ranges): fwd + bwd."""
        if force_sm80 and is_ampere():
            # kernel path is already SM80 on Ampere, no need to force it
            return

        device = self.device
        seed = self.seed + seqlen + d + mask_types * 5
        torch.random.manual_seed(seed)
        random.seed(seed)

        batch_size = 8
        nheads = 6
        nheads_kv = {"mha": nheads, "gqa": 3, "mqa": 1}[mha_type]

        q_v = torch.randn(
            batch_size * seqlen, nheads, d, device=device, dtype=dtype
        ).requires_grad_()
        k_v = torch.randn(
            batch_size * seqlen, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        v_v = torch.randn(
            batch_size * seqlen, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()

        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32
        )
        # q/k ranges equivalent to the cu_seqlens partition: [[0, s], [s, 2s], ...]
        q_ranges_t = torch.stack([cu_seqlens[:-1], cu_seqlens[1:]], dim=1)
        k_ranges_t = q_ranges_t.clone()

        test_case = (
            f"[RANK {self.rank}][test_varlen_fwd_bwd]"
            f"[{force_sm80=}][{seqlen=}][{d=}]"
            f"[{mask_types=}][{mha_type=}][{dtype=}]"
        )

        with _maybe_force_sm80(force_sm80):
            out_v, _ = flex_flash_attn_func(
                q_v,
                k_v,
                v_v,
                q_ranges=q_ranges_t,
                k_ranges=k_ranges_t,
                mask_types=mask_types,
                max_seqlen_q=seqlen,
                max_seqlen_k=seqlen,
                # The atomic fwd merge is SM100/SM110-only; the SM80 varlen
                # path collapses ranges to cu_seqlens with direct store.
                disable_fwd_atomic_reduction=force_sm80,
            )
            out_v = out_v.to(dtype)
            g = torch.randn_like(out_v)
            dq_v, dk_v, dv_v = torch.autograd.grad(out_v, (q_v, k_v, v_v), g)

        q_ranges = AttnRanges.from_ranges(
            [[i * seqlen, (i + 1) * seqlen] for i in range(batch_size)]
        )
        self.assert_close_to_torch_ref(
            q_thd=q_v.detach(),
            k_thd=k_v.detach(),
            v_thd=v_v.detach(),
            do_thd=g,
            out_thd=out_v,
            dq_thd=dq_v,
            dk_thd=dk_v,
            dv_thd=dv_v,
            q_ranges=q_ranges,
            k_ranges=q_ranges,
            attn_type_map=[mask_types] * batch_size,
            total_seqlen_q=batch_size * seqlen,
            total_seqlen_k=batch_size * seqlen,
            dtype=dtype,
            test_case=test_case,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Overlapping q_ranges: atomic merge with dtype-O (default) vs fp32-O
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("out_dtype", [None, torch.float32])
    def test_overlap_atomic_out_dtype(self, out_dtype):
        """Overlapping q_ranges atomic merge: dtype-O default vs fp32 lossless."""
        _, major_arch = get_device_arch()
        if major_arch not in (10, 11):
            return

        device = self.device
        dtype = torch.bfloat16
        d, nheads, total = 128, 4, 768
        seed = self.seed + d + (0 if out_dtype is None else 1)
        torch.random.manual_seed(seed)

        q = torch.randn(total, nheads, d, device=device, dtype=dtype).requires_grad_()
        k = torch.randn(total, nheads, d, device=device, dtype=dtype).requires_grad_()
        v = torch.randn(total, nheads, d, device=device, dtype=dtype).requires_grad_()

        # q[256:512] is covered by both relations -> exercised by the atomic
        # merge. k ranges are disjoint across relations (2D-disjoint contract:
        # the merge would otherwise double-count the shared k rows).
        q_ranges_t = torch.tensor(
            [[0, 512], [256, 768]], device=device, dtype=torch.int32
        )
        k_ranges_t = torch.tensor(
            [[0, 512], [512, 768]], device=device, dtype=torch.int32
        )
        test_case = f"[RANK {self.rank}][test_overlap_atomic_out_dtype][{out_dtype=}]"

        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_types=MT_MAP.full,
            max_seqlen_q=total,
            max_seqlen_k=total,
            out_dtype=out_dtype,
        )
        g = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)

        self.assert_close_to_torch_ref(
            q_thd=q.detach(),
            k_thd=k.detach(),
            v_thd=v.detach(),
            do_thd=g,
            out_thd=out.to(dtype),
            dq_thd=dq,
            dk_thd=dk,
            dv_thd=dv,
            q_ranges=AttnRanges.from_ranges([[0, 512], [256, 768]]),
            k_ranges=AttnRanges.from_ranges([[0, 512], [512, 768]]),
            attn_type_map=[MT_MAP.full, MT_MAP.full],
            total_seqlen_q=total,
            total_seqlen_k=total,
            dtype=dtype,
            test_case=test_case,
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Caller-provided dq/dk/dv buffers accumulate gradients from overlapping ranges
    # ─────────────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("overlap_side", ["q", "k"])
    def test_bwd_accumulates_caller_buffers_with_overlapping_q_or_k_ranges(
        self, overlap_side
    ):
        """Accumulate caller buffers once for overlapping q or k ranges.

        The self-allocated backward result is used only as the accumulation
        reference, not as an independent numerical reference.
        """
        _, major_arch = get_device_arch()
        if major_arch not in (10, 11):
            self.skipTest("caller-buffer accumulation requires SM100/SM110")

        device = self.device
        dtype = torch.bfloat16
        d, nheads, num_ranges, seg = 128, 4, 8, 256
        total = num_ranges * seg
        torch.random.manual_seed(self.seed + d + (overlap_side == "k"))

        q, k, v, do = (
            torch.randn(total, nheads, d, device=device, dtype=dtype) for _ in range(4)
        )
        wide = [[i * seg, min((i + 2) * seg, total)] for i in range(num_ranges)]
        narrow = [[i * seg, (i + 1) * seg] for i in range(num_ranges)]
        q_ranges, k_ranges = (wide, narrow) if overlap_side == "q" else (narrow, wide)
        ranges = dict(
            q_ranges=torch.tensor(q_ranges, device=device, dtype=torch.int32),
            k_ranges=torch.tensor(k_ranges, device=device, dtype=torch.int32),
            max_seqlen_q=2 * seg,
            max_seqlen_k=2 * seg,
        )
        out, lse = _flex_flash_attn_fwd(q, k, v, out_dtype=dtype, **ranges)

        fp32 = dict(dq_type=torch.float32, dk_type=torch.float32, dv_type=torch.float32)
        ref_grads = _flex_flash_attn_bwd(q, k, v, out, lse, do, **ranges, **fp32)

        init = [torch.randn_like(x, dtype=torch.float32) for x in (q, k, v)]
        dq, dk, dv = (x.clone() for x in init)
        grads = _flex_flash_attn_bwd(
            q, k, v, out, lse, do, dq=dq, dk=dk, dv=dv, **ranges, **fp32
        )

        for name, grad, buf, x0, ref in zip(
            ("dq", "dk", "dv"), grads, (dq, dk, dv), init, ref_grads
        ):
            assert grad is buf, f"{name} must be the caller buffer"
            torch.testing.assert_close(
                buf, x0 + ref, rtol=1e-4, atol=1e-4, msg=lambda m: f"{name}: {m}"
            )

    @with_run_in_mp
    @parameterize("range_merge", [False, True])
    def test_bwd_direct_dq_caller_buffer_keeps_hole_rows(self, range_merge):
        """Direct-path dQ adds onto a caller buffer and leaves hole rows unchanged.

        Without RangeMerge the dense-slot postprocess reads only tiles the
        preprocess clears, so ``dq_accum`` may start uninitialized; with it the
        row-major postprocess adds every physical row, so ``dq_accum`` must be
        zero outside q_ranges. ``torch.empty`` returns NaN-filled float tensors
        during the call, so a missing zero-fill shows up in the result.
        """
        _, major_arch = get_device_arch()
        if major_arch not in (10, 11):
            self.skipTest("caller-buffer accumulation requires SM100/SM110")
        device, dtype = self.device, torch.bfloat16
        total, nheads, d = 1024, 4, 128
        torch.random.manual_seed(self.seed + int(range_merge))

        q, k, v, do = (
            torch.randn(total, nheads, d, device=device, dtype=dtype) for _ in range(4)
        )
        # q rows [200, 300), [700, 800) and [900, 1024) are holes.
        ranges = dict(
            q_ranges=torch.tensor(
                [[0, 200], [300, 700], [800, 900]], device=device, dtype=torch.int32
            ),
            k_ranges=torch.tensor(
                [[0, 300], [300, 800], [800, 1024]], device=device, dtype=torch.int32
            ),
            max_seqlen_q=400,
            max_seqlen_k=500,
        )
        direct = dict(
            disable_fwd_atomic_reduction=True, disable_bwd_dkv_atomic_reduction=True
        )
        out, lse = _flex_flash_attn_fwd(
            q, k, v, out_dtype=dtype, disable_fwd_atomic_reduction=True, **ranges
        )
        fp32 = dict(dq_type=torch.float32, dk_type=torch.float32, dv_type=torch.float32)
        ref_dq = _flex_flash_attn_bwd(
            q, k, v, out, lse, do, range_merge=range_merge, **direct, **ranges, **fp32
        )[0]

        init = torch.randn(total, nheads, d, device=device)
        dq = init.clone()
        empty = torch.empty

        def nan_empty(*args, **kwargs):
            t = empty(*args, **kwargs)
            return t.fill_(float("nan")) if t.is_floating_point() else t

        with mock.patch("torch.empty", nan_empty), mock.patch(
            "magi_attention.kernel.cutedsl.flex_flash_attn.bwd_postprocess_rowmajor",
            wraps=bwd_postprocess_rowmajor,
        ) as rowmajor:
            grad = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                dq=dq,
                range_merge=range_merge,
                **direct,
                **ranges,
                **fp32,
            )[0]
        dq_rowmajor = any(call.args[1] is dq for call in rowmajor.call_args_list)
        assert dq_rowmajor == range_merge

        assert grad is dq, "dq must be the caller buffer"
        assert ref_dq[200:300].abs().max() == 0, "hole rows must have zero gradient"
        torch.testing.assert_close(dq, init + ref_dq, rtol=1e-5, atol=1e-5)

    @with_run_in_mp
    @parameterize(
        "case",
        [
            # (out dtype, fp32 partial, old buffer value): rounding the partial
            # to the out dtype before the add gives 0 and inf respectively.
            (torch.bfloat16, 1 + 2**-10, -1.0),
            (torch.float16, 65536.0, -65504.0),
            (torch.float32, 1 + 2**-20, -1.0),
        ],
    )
    @parameterize("hd_2cta", [(64, False), (128, False), (128, True)])
    @parameterize("layout", ["dense", "ranges"])
    def test_bwd_postprocess_accumulate_rounds_once(self, case, hd_2cta, layout):
        """An accumulating postprocess adds the fp32 partial before rounding.

        A uniform accumulator makes the expected output independent of the
        accumulator's tile layout, so the result is compared exactly.
        """
        arch, major_arch = get_device_arch()
        if major_arch not in (10, 11):
            self.skipTest("fp32 staging of the accumulating postprocess is SM100/SM110")
        out_dtype, partial, old = case
        head_dim, use_2cta = hd_2cta
        if use_2cta and major_arch != 10:
            self.skipTest("the 2-CTA postprocess is SM100 only")
        device, num_head, tile_m = self.device, 2, 128
        hdim_rounded = (head_dim + 31) // 32 * 32

        if layout == "dense":
            batch, seqlen = 2, 200  # partial last tile
            seqlen_rounded = (seqlen + tile_m - 1) // tile_m * tile_m
            accum = torch.full(
                (batch, num_head, seqlen_rounded * hdim_rounded), partial, device=device
            )
            out = torch.full(
                (batch, seqlen, num_head, head_dim), old, device=device, dtype=out_dtype
            )
            in_range = torch.ones(batch, seqlen, dtype=torch.bool, device=device)
            ranges = None
        else:
            total, range_list = 512, [[0, 200], [300, 450]]
            ranges = torch.tensor(range_list, dtype=torch.int32, device=device)
            slots = (total + tile_m - 1) // tile_m + len(range_list)
            accum = torch.full(
                (num_head, slots * tile_m * hdim_rounded), partial, device=device
            )
            out = torch.full(
                (total, num_head, head_dim), old, device=device, dtype=out_dtype
            )
            in_range = torch.zeros(total, dtype=torch.bool, device=device)
            for start, end in range_list:
                in_range[start:end] = True

        with _record_bwd_postprocess() as built:
            bwd_postprocess(
                accum,
                out,
                1.0,
                None,
                None,
                arch,
                to_cute_dtype(out_dtype),
                head_dim,
                tile_m,
                128,
                1,
                False,
                use_2cta_instrs=use_2cta,
                ranges=ranges,
                use_dense_dqacc_for_ranges=ranges is not None,
                accumulate=True,
            )
        assert len(built) == 1 and built[0].use_2cta_instrs == use_2cta
        assert built[0].stage_dtype is cutlass.Float32

        expected = (
            torch.tensor(partial) + torch.tensor(old, dtype=out_dtype).float()
        ).to(out_dtype)
        assert torch.all(out[in_range] == expected.to(device)), (
            f"{out_dtype=} {hd_2cta=} {layout=}: got "
            f"{out[in_range].float().unique().tolist()}, expected {expected.item()}"
        )
        assert torch.all(out[~in_range] == old), "rows outside ranges must keep old"

    @with_run_in_mp
    @parameterize("head_dim", [64, 128])
    @parameterize("mha_type", ["mha", "gqa"])
    def test_bwd_accumulates_low_precision_caller_buffers(self, head_dim, mha_type):
        """bf16 caller buffers get ``round(g + old)``, not ``round(round(g) + old)``.

        Each buffer starts at ``-round(g)``, so rounding the partial first
        would leave exactly zero. The atomic reduction order differs between
        the two backward runs, so the residual is compared with a tolerance
        relative to ``|g|``. MHA exercises dQ; GQA also exercises the reducing
        dK/dV postprocess.
        """
        _, major_arch = get_device_arch()
        if major_arch not in (10, 11):
            self.skipTest("caller-buffer accumulation requires SM100/SM110")
        device, dtype = self.device, torch.bfloat16
        batch, seqlen, nheads = 1, 1000, 4  # partial last tile
        nheads_kv = nheads if mha_type == "mha" else 1
        torch.random.manual_seed(self.seed + head_dim + (mha_type == "gqa"))

        q, do = (
            torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype)
            for _ in range(2)
        )
        k, v = (
            torch.randn(batch, seqlen, nheads_kv, head_dim, device=device, dtype=dtype)
            for _ in range(2)
        )
        out, lse = _flex_flash_attn_fwd(q, k, v)

        fp32 = dict(dq_type=torch.float32, dk_type=torch.float32, dv_type=torch.float32)
        ref_grads = _flex_flash_attn_bwd(q, k, v, out, lse, do, **fp32)
        names = ("dq", "dk", "dv") if mha_type == "gqa" else ("dq",)
        bufs = {n: -g.to(dtype) for n, g in zip(("dq", "dk", "dv"), ref_grads)}
        with _record_bwd_postprocess() as built:
            grads = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                **{n: bufs[n] for n in names},
                **{f"{n}_type": dtype for n in ("dq", "dk", "dv")},
            )
        # dQ builds first (later calls may hit its compile key); it takes the
        # 2-CTA postprocess at head_dim 128 on SM100.
        assert built and all(obj.accumulate for obj in built)
        assert all(obj.stage_dtype is cutlass.Float32 for obj in built)
        assert built[0].use_2cta_instrs == (head_dim == 128 and major_arch == 10)
        for name, grad, ref in zip(("dq", "dk", "dv"), grads, ref_grads):
            if name not in names:
                continue
            buf = bufs[name]
            assert grad is buf, f"{name} must be the caller buffer"
            expected = ref - ref.to(dtype).float()
            assert buf.count_nonzero() > buf.numel() // 2, f"{name}: residual lost"
            torch.testing.assert_close(
                buf.float(),
                expected,
                rtol=0,
                atol=5e-5 * ref.abs().max().item(),
                msg=lambda m: f"{name}: {m}",
            )

    # ─────────────────────────────────────────────────────────────────────
    # Learnable sink: fwd fold (direct store / atomic merge) + bwd dsink
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("mha_type", ["mha", "gqa"])
    @parameterize("d", [64, 128])
    @parameterize("overlap", [False, True])
    @parameterize("n_sink", [1, 4, 129])
    def test_sink_fwd_bwd(self, n_sink, overlap, d, mha_type):
        """``[n_sink, nhq]`` fp32 sink: out/lse fold on both fwd paths, dsink in bwd."""
        _, major_arch = get_device_arch()
        if major_arch not in (10, 11):
            self.skipTest("sink on the ranges path requires SM100/SM110")

        device = self.device
        dtype = torch.bfloat16
        nheads, total = 4, 768
        nheads_kv = {"mha": nheads, "gqa": 2}[mha_type]
        torch.random.manual_seed(self.seed + n_sink * 3 + d + int(overlap))

        q = torch.randn(total, nheads, d, device=device, dtype=dtype).requires_grad_()
        k = torch.randn(
            total, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        v = torch.randn(
            total, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        sink = torch.randn(
            n_sink, nheads, device=device, dtype=torch.float32
        ).requires_grad_()

        # NOTES: the atomic-merge path folds the sink in the postprocess and may
        # leave q[512:768] attending to the sinks only; the direct-store path
        # folds it in-kernel and requires every q row to be covered
        if overlap:
            q_ranges = [[0, 512], [256, 512]]
            k_ranges = [[0, 512], [512, 768]]
        else:
            q_ranges = [[0, 256], [256, 768]]
            k_ranges = [[0, 512], [512, 768]]
        q_ranges_t = torch.tensor(q_ranges, device=device, dtype=torch.int32)
        k_ranges_t = torch.tensor(k_ranges, device=device, dtype=torch.int32)
        test_case = (
            f"[RANK {self.rank}][test_sink_fwd_bwd]"
            f"[{n_sink=}][{overlap=}][{d=}][{mha_type=}]"
        )

        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_types=MT_MAP.full,
            max_seqlen_q=total,
            max_seqlen_k=total,
            sink=sink,
            disable_fwd_atomic_reduction=not overlap,
        )
        g = torch.randn_like(out)
        dq, dk, dv, dsink = torch.autograd.grad(out, (q, k, v, sink), g)

        self.assert_close_to_torch_ref(
            q_thd=q.detach(),
            k_thd=k.detach(),
            v_thd=v.detach(),
            do_thd=g,
            out_thd=out.to(dtype),
            dq_thd=dq,
            dk_thd=dk,
            dv_thd=dv,
            q_ranges=AttnRanges.from_ranges(q_ranges),
            k_ranges=AttnRanges.from_ranges(k_ranges),
            attn_type_map=[MT_MAP.full, MT_MAP.full],
            total_seqlen_q=total,
            total_seqlen_k=total,
            dtype=dtype,
            test_case=test_case,
            sink=sink.detach(),
            dsink_thd=dsink,
        )

    @with_run_in_mp
    @parameterize("total_q", [0, 1, 127, 128, 129])
    @parameterize("n_sink", [1, 129])
    def test_bwd_dsink_matches_reference(self, total_q, n_sink):
        """dsink for any sink count and partial query tiles, against fp64 torch.

        Row 1 has LSE -inf (a row no relation covers on the direct path, which
        contributes nothing), row 2 attends only to the sinks, and head 2's
        sinks are all -inf (zero gradient).
        """
        device, num_head, head_dim_v = self.device, 3, 128
        torch.random.manual_seed(self.seed + total_q + n_sink)
        sink = torch.randn(n_sink, num_head, device=device) * 3
        sink[:, 2] = float("-inf")
        out, dout = (
            torch.randn(
                total_q, num_head, head_dim_v, device=device, dtype=torch.bfloat16
            )
            for _ in range(2)
        )
        lse_sink = torch.logsumexp(sink, dim=0)
        lse = torch.logaddexp(
            torch.randn(total_q, num_head, device=device) * 3, lse_sink
        )
        if total_q > 2:
            lse[1] = float("-inf")
            lse[2] = lse_sink

        dsink = bwd_dsink(out, dout, lse, sink)

        # The reference folds a -inf row into exp(sink - inf) = 0.
        lse_ref = torch.where(torch.isneginf(lse), torch.inf, lse).double()
        dsink_ref = sink_bwd(sink.double(), lse_ref, out.double(), dout.double())
        assert dsink.shape == sink.shape and torch.isfinite(dsink).all()
        assert torch.all(dsink[:, 2] == 0)
        torch.testing.assert_close(dsink.double(), dsink_ref, rtol=1e-5, atol=1e-5)

    @with_run_in_mp
    @parameterize("disable_fwd_atomic_reduction", [False, True])
    def test_sink_with_empty_k(self, disable_fwd_atomic_reduction):
        """With no keys every row attends only to the sinks: O = 0, LSE = lse_sink."""
        _, major_arch = get_device_arch()
        if major_arch not in (10, 11):
            self.skipTest("sink on the ranges path requires SM100/SM110")
        device, dtype, nheads, d, total_q = self.device, torch.bfloat16, 4, 128, 64
        q = torch.randn(total_q, nheads, d, device=device, dtype=dtype)
        k = torch.empty(0, nheads, d, device=device, dtype=dtype)
        v = torch.empty_like(k)
        sink = torch.zeros(4, nheads, device=device)
        ranges = dict(
            q_ranges=torch.tensor([[0, total_q]], device=device, dtype=torch.int32),
            k_ranges=torch.tensor([[0, 0]], device=device, dtype=torch.int32),
            max_seqlen_q=total_q,
            max_seqlen_k=0,
            sink=sink,
            sink_layout="sh",
            disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
        )
        out, lse = _flex_flash_attn_fwd(q, k, v, **ranges)
        assert torch.all(out == 0)
        torch.testing.assert_close(
            lse, torch.full_like(lse, math.log(4)), rtol=0, atol=1e-6
        )

        dq, dk, dv, dsink = _flex_flash_attn_bwd(
            q, k, v, out.to(dtype), lse, torch.randn_like(q), **ranges
        )
        assert torch.all(dq == 0) and dk.shape == k.shape and dv.shape == v.shape
        assert dsink is not None and torch.all(dsink == 0)

    # ─────────────────────────────────────────────────────────────────────
    # Varlen opt-flag contract
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("mha_type", ["mha", "gqa"])
    @parameterize(
        "mask_types",
        [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal, "mixed"],
    )
    def test_varlen_opt_flags(self, mask_types, mha_type):
        """Ranges opt flags: direct store + coverage + dense dqacc/dkvacc.

        full / causal go in as the scalar (static kernel); inv_causal /
        bi_causal and "mixed" (all four types cycling over the ranges) go in
        as the int32[R] tensor (per-range kernel, 2-CTA at head_dim 128).
        """
        _, major_arch = get_device_arch()
        if major_arch not in (10, 11):
            return

        device = self.device
        dtype = torch.bfloat16
        # seqlen_q < seqlen_k keeps bi_causal a band instead of the diagonal,
        # where P == 1 exactly and the reference dq vanishes.
        seqlen_q, seqlen_k, batch_size, nheads, d = 256, 512, 8, 6, 128
        nheads_kv = {"mha": nheads, "gqa": 3}[mha_type]
        attn_type_map: list[int] = (
            [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal]
            * (batch_size // 4)
            if mask_types == "mixed"
            else [mask_types] * batch_size
        )
        seed = self.seed + seqlen_k + d + sum(attn_type_map) * 7
        torch.random.manual_seed(seed)
        random.seed(seed)

        q_v = torch.randn(
            batch_size * seqlen_q, nheads, d, device=device, dtype=dtype
        ).requires_grad_()
        k_v = torch.randn(
            batch_size * seqlen_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        v_v = torch.randn(
            batch_size * seqlen_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()

        def partition(seqlen: int) -> torch.Tensor:
            cu = torch.arange(
                0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32
            )
            return torch.stack([cu[:-1], cu[1:]], dim=1)

        q_ranges_t = partition(seqlen_q)
        k_ranges_t = partition(seqlen_k)
        mask_types_arg: int | torch.Tensor = (
            mask_types
            if mask_types in (MT_MAP.full, MT_MAP.causal)
            else torch.tensor(attn_type_map, device=device, dtype=torch.int32)
        )

        test_case = (
            f"[RANK {self.rank}][test_varlen_opt_flags][{mask_types=}][{mha_type=}]"
        )

        out_v, _ = flex_flash_attn_func(
            q_v,
            k_v,
            v_v,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_types=mask_types_arg,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            disable_fwd_atomic_reduction=True,
            # direct-store disjoint dKV is MHA-only (unique-writer contract)
            disable_bwd_dkv_atomic_reduction=(mha_type == "mha"),
        )
        g = torch.randn_like(out_v)
        dq_v, dk_v, dv_v = torch.autograd.grad(out_v, (q_v, k_v, v_v), g)

        self.assert_close_to_torch_ref(
            q_thd=q_v.detach(),
            k_thd=k_v.detach(),
            v_thd=v_v.detach(),
            do_thd=g,
            out_thd=out_v,
            dq_thd=dq_v,
            dk_thd=dk_v,
            dv_thd=dv_v,
            q_ranges=AttnRanges.from_ranges(q_ranges_t.tolist()),
            k_ranges=AttnRanges.from_ranges(k_ranges_t.tolist()),
            attn_type_map=attn_type_map,
            total_seqlen_q=batch_size * seqlen_q,
            total_seqlen_k=batch_size * seqlen_k,
            dtype=dtype,
            test_case=test_case,
        )


if __name__ == "__main__":
    run_tests()
