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

import random
from contextlib import contextmanager

import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
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

# ─────────────────────────────────────────────────────────────────────────────
# SM80 kernel selection
# ─────────────────────────────────────────────────────────────────────────────
#
# The SM80 kernel path is selected via the MAGI_ATTENTION_FFA_CUTEDSL_ARCH override
# rather than the real device capability, so it can be exercised on newer GPUs
# (the compiled SM80 SASS runs fine on sm90/sm100). get_device_arch() is
# lru_cached, so we must clear the cache whenever we toggle the override.


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
        return 600

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
                NORM_RTOL_RATIO * ref_norm,
                msg=(
                    f"For {test_case=}: {name} {norm=} should be no greater than "
                    f"{NORM_RTOL_RATIO} x {ref_norm=}"
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
    ) -> None:
        """Compare the kernel out/dq/dk/dv against a torch reference (thd layout).

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
            out_ref, _ = ref_attn_func(
                q=q_ref,
                k=k_ref,
                v=v_ref,
                mask=mask,
                layout="thd",
                backend="sdpa",
                high_precision=high_precision,
            )
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(
                out_ref, (q_ref, k_ref, v_ref), do_thd
            )
            return out_ref, dq_ref, dk_ref, dv_ref

        out_hi, dq_hi, dk_hi, dv_hi = _ref(high_precision=True)
        out_lo, dq_lo, dk_lo, dv_lo = _ref(high_precision=False)

        err_msg_list: list[str] = []
        for name, actual, ref_hi, ref_lo in [
            ("o", out_thd, out_hi, out_lo),
            ("dq", dq_thd, dq_hi, dq_lo),
            ("dk", dk_thd, dk_hi, dk_lo),
            ("dv", dv_thd, dv_hi, dv_lo),
        ]:
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
            )
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


if __name__ == "__main__":
    run_tests()
