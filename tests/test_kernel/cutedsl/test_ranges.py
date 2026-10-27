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
import unittest
from dataclasses import dataclass

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from magi_attention.kernel.cutedsl import MT_MAP, flex_flash_attn_func
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import (
    GQA_HD128,
    MHA_HD64,
    MHA_HD128,
    SM100_FAMILY_REQUIRED,
    AttnTensors,
    CuteDSLDistTestBase,
    HeadConfig,
    Ranges,
    assert_close_to_torch_ref,
    cycle_mask_types,
    is_sm100_family,
    make_qkv,
    mask_types_arg,
    ranges_from_lengths,
    seed_test_case,
    to_ranges_tensor,
    torch_reference_out_lse,
)
from magi_attention.testing.dist_common import with_run_in_mp

DTYPE = torch.bfloat16

# Relations of the batched cases. Each q range is shorter than its k range, so
# bi_causal stays a band: on the diagonal P == 1 and the reference dq vanishes.
NUM_RELATIONS = 8
Q_RANGE_LEN = 256
K_RANGE_LEN = 512

MASK_PATTERNS = {
    "full": [MT_MAP.full],
    "causal": [MT_MAP.causal],
    "inv_causal": [MT_MAP.inv_causal],
    "bi_causal": [MT_MAP.bi_causal],
    "mixed": [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal],
}

# Length unit of the overlap and accumulation geometries.
SPAN = 128

# Accumulation tensors: fp32 O buffer, the atomic-merge precision.
ACCUM_ATOL = ACCUM_RTOL = 1e-2
LSE_ATOL = LSE_RTOL = 1e-3


@dataclass(frozen=True)
class UniqueWriterCase:
    name: str
    heads: HeadConfig
    pattern: str
    disable_fwd_atomic_reduction: bool
    disable_bwd_dkv_atomic_reduction: bool


UNIQUE_WRITER_CASES = [
    UniqueWriterCase("direct_o", MHA_HD128, "mixed", True, False),
    UniqueWriterCase("direct_dkv", MHA_HD128, "mixed", False, True),
    UniqueWriterCase("direct_o_and_dkv", MHA_HD128, "full", True, True),
    # Direct dK/dV store is MHA-only; GQA keeps the dK/dV reduction.
    UniqueWriterCase("direct_o_gqa", GQA_HD128, "mixed", True, False),
]

# (heads, mask pattern) per kernel variant: one row each.
PER_RANGE_CASES = [
    (MHA_HD128, "bi_causal"),
    (GQA_HD128, "mixed"),
]
OVERLAP_CASES = [
    (MHA_HD128, MT_MAP.full),
    (GQA_HD128, MT_MAP.causal),
]


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRanges(CuteDSLDistTestBase):
    def _run_and_check(
        self,
        *,
        q_ranges: Ranges,
        k_ranges: Ranges,
        attn_type_map: list[int],
        heads: HeadConfig,
        test_case: str,
        disable_fwd_atomic_reduction: bool = False,
        disable_bwd_dkv_atomic_reduction: bool = False,
    ) -> None:
        seed_test_case(test_case)
        q, k, v = make_qkv(
            (max(end for _, end in q_ranges),),
            (max(end for _, end in k_ranges),),
            heads,
            DTYPE,
            self.device,
        )
        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=to_ranges_tensor(q_ranges, self.device),
            k_ranges=to_ranges_tensor(k_ranges, self.device),
            mask_types=mask_types_arg(attn_type_map, self.device),
            max_seqlen_q=max(end - start for start, end in q_ranges),
            max_seqlen_k=max(end - start for start, end in k_ranges),
            disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
            disable_bwd_dkv_atomic_reduction=disable_bwd_dkv_atomic_reduction,
        )
        do = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
        assert_close_to_torch_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("case", PER_RANGE_CASES)
    def test_per_range_mask_types(self, case: tuple[HeadConfig, str]):
        """Mask types beyond full/causal go through the per-range kernel."""
        heads, pattern = case
        self._run_and_check(
            q_ranges=ranges_from_lengths([Q_RANGE_LEN] * NUM_RELATIONS),
            k_ranges=ranges_from_lengths([K_RANGE_LEN] * NUM_RELATIONS),
            attn_type_map=cycle_mask_types(MASK_PATTERNS[pattern], NUM_RELATIONS),
            heads=heads,
            test_case=f"[test_per_range_mask_types][{pattern}][{heads.name}]",
        )

    @with_run_in_mp
    @parameterize("case", OVERLAP_CASES)
    def test_overlapping_q_ranges(self, case: tuple[HeadConfig, int]):
        """Two relations share q rows; the atomic O merge combines them.

        Their k ranges are disjoint, so every (q, k) pair is attended once.
        """
        heads, mask_type = case
        self._run_and_check(
            q_ranges=[[0, 4 * SPAN], [2 * SPAN, 6 * SPAN]],
            k_ranges=[[0, 4 * SPAN], [4 * SPAN, 6 * SPAN]],
            attn_type_map=[mask_type] * 2,
            heads=heads,
            test_case=f"[test_overlapping_q_ranges][{mask_type=}][{heads.name}]",
        )

    @with_run_in_mp
    @parameterize("case", UNIQUE_WRITER_CASES)
    def test_unique_writer_contracts(self, case: UniqueWriterCase):
        """Sorted disjoint ranges let O / dK,dV skip the atomic reduction."""
        self._run_and_check(
            q_ranges=ranges_from_lengths([Q_RANGE_LEN] * NUM_RELATIONS),
            k_ranges=ranges_from_lengths([K_RANGE_LEN] * NUM_RELATIONS),
            attn_type_map=cycle_mask_types(MASK_PATTERNS[case.pattern], NUM_RELATIONS),
            heads=case.heads,
            disable_fwd_atomic_reduction=case.disable_fwd_atomic_reduction,
            disable_bwd_dkv_atomic_reduction=case.disable_bwd_dkv_atomic_reduction,
            test_case=f"[test_unique_writer_contracts][{case.name}]",
        )


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRangeAccumulation(TestCase):
    """Atomic-merge forward calls accumulating into caller-owned O/LSE."""

    def setUp(self) -> None:
        self.device = torch.cuda.current_device()
        self.heads = MHA_HD128
        seed_test_case(self.id())

    def _accumulate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        q_ranges: Ranges,
        k_ranges: Ranges,
    ) -> None:
        _flex_flash_attn_fwd(
            q,
            k,
            v,
            out=out,
            lse=lse,
            q_ranges=to_ranges_tensor(q_ranges, self.device),
            k_ranges=to_ranges_tensor(k_ranges, self.device),
            mask_types=MT_MAP.full,
            max_seqlen_q=max(end - start for start, end in q_ranges),
            max_seqlen_k=max(end - start for start, end in k_ranges),
        )

    def _empty_state(self, total_q: int) -> tuple[torch.Tensor, torch.Tensor]:
        """O needs no initialization where LSE is -inf."""
        heads = self.heads
        out = torch.full(
            (total_q, heads.num_heads_q, heads.head_dim),
            math.nan,
            device=self.device,
            dtype=torch.float32,
        )
        lse = torch.full((total_q, heads.num_heads_q), -math.inf, device=self.device)
        return out, lse

    def test_calls_accumulate_to_their_union(self):
        """Three calls equal one call over the union of their relations.

        k ranges are disjoint across calls; the last q span stays uncovered
        and ends with O = 0, LSE = -inf.
        """
        calls = [
            ([[0, 2 * SPAN]], [[0, 2 * SPAN]]),
            ([[SPAN, 4 * SPAN]], [[2 * SPAN, 3 * SPAN]]),
            ([[3 * SPAN, 5 * SPAN]], [[3 * SPAN, 4 * SPAN]]),
        ]
        total_q, total_k = 6 * SPAN, 4 * SPAN
        q, k, v = make_qkv((total_q,), (total_k,), self.heads, DTYPE, self.device)
        out, lse = self._empty_state(total_q)

        for q_ranges, k_ranges in calls:
            self._accumulate(q, k, v, out, lse, q_ranges, k_ranges)

        out_ref, lse_ref = torch_reference_out_lse(
            q=q,
            k=k,
            v=v,
            q_ranges=[r for q_ranges, _ in calls for r in q_ranges],
            k_ranges=[r for _, k_ranges in calls for r in k_ranges],
        )
        torch.testing.assert_close(out, out_ref, atol=ACCUM_ATOL, rtol=ACCUM_RTOL)
        torch.testing.assert_close(lse, lse_ref, atol=LSE_ATOL, rtol=LSE_RTOL)

    def test_empty_k_call_keeps_the_accumulated_state(self):
        """A call without keys leaves accumulated rows and zeroes empty rows."""
        total_q = 4 * SPAN
        accumulated = slice(0, 2 * SPAN)
        empty = slice(2 * SPAN, total_q)
        q, k, v = make_qkv((total_q,), (0,), self.heads, DTYPE, self.device)
        out, lse = self._empty_state(total_q)
        out[accumulated] = torch.randn_like(out[accumulated])
        lse[accumulated] = torch.randn_like(lse[accumulated])
        out_prev, lse_prev = out.clone(), lse.clone()

        self._accumulate(q, k, v, out, lse, [[0, total_q]], [[0, 0]])

        self.assertTrue(torch.all(out[empty] == 0), "empty-row O is not zero")
        torch.testing.assert_close(
            out[accumulated], out_prev[accumulated], atol=0, rtol=0
        )
        torch.testing.assert_close(lse, lse_prev, atol=0, rtol=0)


# SM reservation of the overlap tests: the grid is capped at num_sm - SM_MARGIN.
SM_MARGIN = 4
# Runs of empty BiCausal tiles need a grid of two CTAs, so each CTA walks many
# tiles and the per-tile handoff phases cycle.
EMPTY_TILE_SEGMENTS = 16
EMPTY_TILE_LEN_Q = 256
EMPTY_TILE_LEN_K = 128


def _compiled_entries(kernel: object) -> int:
    """Number of compiled specializations the kernel entry point holds.

    ``_flex_flash_attn_fwd``/``_bwd`` attach their JIT cache at import time, so
    a test can count the specializations a call compiles.
    """
    return len(getattr(kernel, "compile_cache").cache)


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestPersistentSmMargin(CuteDSLDistTestBase):
    """``sm_margin`` caps the persistent grid without changing the result.

    A launch grid is not observable from a test, so the reservation is asserted
    on the compiled specialization: ``sm_margin > 0`` compiles the persistent
    kernel, and the margin value itself is a runtime argument of that kernel.
    """

    def _run_margin(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        *,
        q_ranges: Ranges,
        k_ranges: Ranges,
        mask_types: int | torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        sm_margin: int,
        direct: bool,
    ) -> tuple[torch.Tensor, ...]:
        out, meta = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=to_ranges_tensor(q_ranges, self.device),
            k_ranges=to_ranges_tensor(k_ranges, self.device),
            mask_types=mask_types,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            disable_fwd_atomic_reduction=direct,
            disable_bwd_dkv_atomic_reduction=direct,
            sm_margin=sm_margin,
        )
        assert meta.lse is not None
        grad_out = do.to(out.dtype)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_out)
        return out.detach(), meta.lse.detach(), dq, dk, dv, grad_out

    @with_run_in_mp
    @parameterize("heads", [MHA_HD64, MHA_HD128])
    def test_sm_margin_matches_full_grid(self, heads: HeadConfig):
        """The same ranges give the same tensors on the full and the capped grid."""
        test_case = f"[test_sm_margin_matches_full_grid][{heads.name}]"
        seed_test_case(test_case)
        q_ranges = ranges_from_lengths([Q_RANGE_LEN] * NUM_RELATIONS)
        k_ranges = ranges_from_lengths([K_RANGE_LEN] * NUM_RELATIONS)
        q, k, v = make_qkv(
            (NUM_RELATIONS * Q_RANGE_LEN,),
            (NUM_RELATIONS * K_RANGE_LEN,),
            heads,
            DTYPE,
            self.device,
        )
        do = torch.randn_like(q)

        runs, grew = {}, {}
        for margin in (0, SM_MARGIN, 2 * SM_MARGIN):
            fwd_before = _compiled_entries(_flex_flash_attn_fwd)
            bwd_before = _compiled_entries(_flex_flash_attn_bwd)
            runs[margin] = self._run_margin(
                q,
                k,
                v,
                do,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                mask_types=MT_MAP.full,
                max_seqlen_q=Q_RANGE_LEN,
                max_seqlen_k=K_RANGE_LEN,
                sm_margin=margin,
                direct=True,
            )
            grew[margin] = (
                _compiled_entries(_flex_flash_attn_fwd) - fwd_before,
                _compiled_entries(_flex_flash_attn_bwd) - bwd_before,
            )

        # Zero margin compiles the static range kernels; a nonzero margin
        # compiles the persistent ones; the margin value is a runtime argument,
        # so 4 and 8 share one compiled kernel.
        self.assertEqual(grew[0], (1, 1), f"{test_case} => full grid")
        self.assertEqual(grew[SM_MARGIN], (1, 1), f"{test_case} => persistent")
        self.assertEqual(grew[2 * SM_MARGIN], (0, 0), f"{test_case} => shared")

        out0, lse0, dq0, dk0, dv0, _ = runs[0]
        for margin in (SM_MARGIN, 2 * SM_MARGIN):
            out, lse, dq, dk, dv, grad_out = runs[margin]
            # Every output element has a single writer, so the tile order of
            # the schedule must not change a bit. dQ accumulates over the K
            # tiles, so its last bit may follow the tile order.
            torch.testing.assert_close(out, out0, rtol=0, atol=0)
            torch.testing.assert_close(lse, lse0, rtol=0, atol=0)
            for name, got, ref in (("dq", dq, dq0), ("dk", dk, dk0), ("dv", dv, dv0)):
                torch.testing.assert_close(
                    got,
                    ref,
                    msg=lambda m, n=name: f"{test_case} => {n}: {m}",
                )
            assert_close_to_torch_ref(
                q=q.detach(),
                k=k.detach(),
                v=v.detach(),
                do=grad_out,
                actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=[MT_MAP.full] * NUM_RELATIONS,
                test_case=f"{test_case}[{margin=}]",
            )

        # A dense call has no range kernel to cap its grid, so it rejects the
        # reservation instead of silently running on every SM.
        dense = torch.randn(1, 256, 2, 64, device=self.device, dtype=DTYPE)
        dense_lse = torch.zeros(1, 2, 256, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "sm_margin"):
            _flex_flash_attn_fwd(dense, dense, dense, sm_margin=SM_MARGIN)
        with self.assertRaisesRegex(NotImplementedError, "sm_margin"):
            _flex_flash_attn_bwd(
                dense, dense, dense, dense, dense_lse, dense, sm_margin=SM_MARGIN
            )

    @with_run_in_mp
    @parameterize("heads", [MHA_HD64, MHA_HD128])
    def test_sm_margin_empty_tile_runs(self, heads: HeadConfig):
        """Runs of empty BiCausal tiles keep the per-tile handoff phases aligned.

        A tile outside the band has no (q, k) pair, and every warp role skips it
        in the same way. A two-CTA grid makes each CTA walk many tiles, so a
        role that skips a different set of tiles desynchronizes its phase.
        """
        sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        if sm_count <= 2:
            self.skipTest("a two-CTA persistent grid needs more than two SMs")
        margin = sm_count - 2
        test_case = f"[test_sm_margin_empty_tile_runs][{heads.name}][{margin=}]"
        seed_test_case(test_case)
        self.assertGreater(
            EMPTY_TILE_SEGMENTS * heads.num_heads_q, 4 * (sm_count - margin)
        )
        pattern = [MT_MAP.full, MT_MAP.bi_causal, MT_MAP.bi_causal, MT_MAP.bi_causal]
        attn_type_map = cycle_mask_types(pattern, EMPTY_TILE_SEGMENTS)
        q_ranges = ranges_from_lengths([EMPTY_TILE_LEN_Q] * EMPTY_TILE_SEGMENTS)
        k_ranges = ranges_from_lengths([EMPTY_TILE_LEN_K] * EMPTY_TILE_SEGMENTS)
        q, k, v = make_qkv(
            (EMPTY_TILE_SEGMENTS * EMPTY_TILE_LEN_Q,),
            (EMPTY_TILE_SEGMENTS * EMPTY_TILE_LEN_K,),
            heads,
            DTYPE,
            self.device,
        )
        do = torch.randn_like(q)

        bwd_before = _compiled_entries(_flex_flash_attn_bwd)
        out, _, dq, dk, dv, grad_out = self._run_margin(
            q,
            k,
            v,
            do,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=mask_types_arg(attn_type_map, self.device),
            max_seqlen_q=EMPTY_TILE_LEN_Q,
            max_seqlen_k=EMPTY_TILE_LEN_K,
            sm_margin=margin,
            direct=False,
        )
        self.assertEqual(
            _compiled_entries(_flex_flash_attn_bwd) - bwd_before,
            1,
            f"{test_case} => persistent backward",
        )
        assert_close_to_torch_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=grad_out,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )


if __name__ == "__main__":
    run_tests()
