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

"""Learnable sink logits (``"sh"`` layout): public fwd + bwd including dsink,
folding the sink once into accumulated O/LSE, and the dsink reduction kernel."""

import math
import unittest
from dataclasses import dataclass

import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import MT_MAP, flex_flash_attn_func
from magi_attention.kernel.cutedsl.ffa_bwd_dsink import DSINK_TILE_M, bwd_dsink
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import with_run_in_mp
from tests.test_kernel.cutedsl.cutedsl_test_utils import (
    GQA_HD128,
    MHA_HD128,
    SM100_FAMILY_REQUIRED,
    AttnTensors,
    CuteDSLDistTestBase,
    HeadConfig,
    Ranges,
    assert_close_to_torch_ref,
    is_sm100_family,
    make_qkv,
    ranges_from_lengths,
    seed_test_case,
    to_ranges_tensor,
    torch_reference_out_lse,
)

HEAD_CONFIGS = [MHA_HD128, GQA_HD128]
DTYPE = torch.bfloat16
# 9 exceeds the 8-sink cap of the C++ backend.
NUM_SINKS = [1, 9]

# Length unit of the range geometries.
SPAN = 128


@dataclass(frozen=True)
class RangesGeometry:
    name: str
    total_q: int
    total_k: int
    q_ranges: Ranges
    k_ranges: Ranges


PARTITION = RangesGeometry(
    "partition",
    total_q=8 * SPAN,
    total_k=8 * SPAN,
    q_ranges=ranges_from_lengths([2 * SPAN] * 4),
    k_ranges=ranges_from_lengths([2 * SPAN] * 4),
)
# Shared q rows merge two relations; the sink still counts once per row.
OVERLAPPING_Q = RangesGeometry(
    "overlapping_q",
    total_q=6 * SPAN,
    total_k=6 * SPAN,
    q_ranges=[[0, 4 * SPAN], [2 * SPAN, 6 * SPAN]],
    k_ranges=[[0, 4 * SPAN], [4 * SPAN, 6 * SPAN]],
)
# Rows outside every q range attend to the sink only: O = 0.
UNCOVERED_Q = RangesGeometry(
    "uncovered_q",
    total_q=6 * SPAN,
    total_k=4 * SPAN,
    q_ranges=[[0, 2 * SPAN], [3 * SPAN, 5 * SPAN]],
    k_ranges=[[0, 2 * SPAN], [2 * SPAN, 4 * SPAN]],
)
RANGES_GEOMETRIES = [PARTITION, OVERLAPPING_Q, UNCOVERED_Q]

DENSE_BATCH_SIZE = 2
DENSE_SEQLEN = 4 * SPAN

FOLD_NUM_SINKS = 2
# Kernel-level forward checks against the fp64 reference.
OUT_TOL = {torch.float32: 1e-2, torch.bfloat16: 2e-2}
LSE_ATOL = LSE_RTOL = 1e-3
# Exact up to fp32 rounding: folding the sink only rescales stored O/LSE.
FOLD_ATOL = FOLD_RTOL = 1e-5


def make_sink(num_sinks: int, num_heads: int, device: int | str) -> torch.Tensor:
    return torch.randn(num_sinks, num_heads, device=device, requires_grad=True)


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


# Match the FFA backend's "sh" dsink atol/rtol in
# tests/test_attn/test_flex_flash_attn.py. Require every element to pass here.
# The reference uses this call's O/LSE; O and dQ/dK/dV cover end-to-end effects.
DSINK_ATOL = 2e-4
DSINK_RTOL = 0.1


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestSinkFwdBwd(CuteDSLDistTestBase):
    @with_run_in_mp
    @parameterize("heads", HEAD_CONFIGS)
    @parameterize("num_sinks", NUM_SINKS)
    def test_dense_sink_fwd_bwd(self, heads: HeadConfig, num_sinks: int):
        test_case = f"[test_dense_sink_fwd_bwd][{heads.name}][{num_sinks=}]"
        seed_test_case(test_case)
        tokens = (DENSE_BATCH_SIZE, DENSE_SEQLEN)
        q, k, v = make_qkv(tokens, tokens, heads, DTYPE, self.device)
        sink = make_sink(num_sinks, heads.num_heads_q, self.device)

        out, meta = flex_flash_attn_func(q, k, v, mask_types=MT_MAP.causal, sink=sink)
        do = torch.randn_like(out)
        dq, dk, dv, dsink = torch.autograd.grad(out, (q, k, v, sink), do)

        def to_thd(t: torch.Tensor) -> torch.Tensor:
            return rearrange(t, "b s h d -> (b s) h d")

        torch.testing.assert_close(
            dsink,
            dsink_reference(
                to_thd(out), to_thd(do), rearrange(meta.lse, "b h s -> (b s) h"), sink
            ),
            atol=DSINK_ATOL,
            rtol=DSINK_RTOL,
        )
        assert_close_to_torch_ref(
            q=to_thd(q.detach()),
            k=to_thd(k.detach()),
            v=to_thd(v.detach()),
            do=to_thd(do),
            sink=sink,
            actual=AttnTensors(
                out=to_thd(out), dq=to_thd(dq), dk=to_thd(dk), dv=to_thd(dv)
            ),
            q_ranges=ranges_from_lengths([DENSE_SEQLEN] * DENSE_BATCH_SIZE),
            k_ranges=ranges_from_lengths([DENSE_SEQLEN] * DENSE_BATCH_SIZE),
            attn_type_map=[MT_MAP.causal] * DENSE_BATCH_SIZE,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("geometry", RANGES_GEOMETRIES)
    @parameterize("heads", HEAD_CONFIGS)
    @parameterize("num_sinks", NUM_SINKS)
    def test_ranges_sink_fwd_bwd(
        self, geometry: RangesGeometry, heads: HeadConfig, num_sinks: int
    ):
        test_case = (
            f"[test_ranges_sink_fwd_bwd][{geometry.name}][{heads.name}][{num_sinks=}]"
        )
        seed_test_case(test_case)
        q, k, v = make_qkv(
            (geometry.total_q,), (geometry.total_k,), heads, DTYPE, self.device
        )
        sink = make_sink(num_sinks, heads.num_heads_q, self.device)

        out, meta = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=to_ranges_tensor(geometry.q_ranges, self.device),
            k_ranges=to_ranges_tensor(geometry.k_ranges, self.device),
            mask_types=MT_MAP.full,
            max_seqlen_q=max(end - start for start, end in geometry.q_ranges),
            max_seqlen_k=max(end - start for start, end in geometry.k_ranges),
            sink=sink,
        )
        do = torch.randn_like(out)
        dq, dk, dv, dsink = torch.autograd.grad(out, (q, k, v, sink), do)

        torch.testing.assert_close(
            dsink,
            dsink_reference(out, do, meta.lse, sink),
            atol=DSINK_ATOL,
            rtol=DSINK_RTOL,
        )
        assert_close_to_torch_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            sink=sink,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=geometry.q_ranges,
            k_ranges=geometry.k_ranges,
            attn_type_map=[MT_MAP.full] * len(geometry.q_ranges),
            test_case=test_case,
        )


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestSinkFolding(unittest.TestCase):
    """The atomic-merge forward folds the sink into caller-owned O/LSE."""

    def setUp(self) -> None:
        self.device = torch.cuda.current_device()
        self.heads = MHA_HD128
        seed_test_case(self.id())

    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        q_ranges: Ranges,
        k_ranges: Ranges,
        sink: torch.Tensor | None,
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
            sink=sink,
        )

    def _empty_state(
        self, total_q: int, out_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """NaN O checks that rows at LSE = -inf are never read."""
        out = torch.full(
            (total_q, self.heads.num_heads_q, self.heads.head_dim),
            math.nan,
            device=self.device,
            dtype=out_dtype,
        )
        lse = torch.full(
            (total_q, self.heads.num_heads_q), -math.inf, device=self.device
        )
        return out, lse

    def _sink(self) -> torch.Tensor:
        return torch.randn(FOLD_NUM_SINKS, self.heads.num_heads_q, device=self.device)

    @parameterize("out_dtype", [torch.float32, torch.bfloat16])
    def test_uncovered_rows_get_the_sink_only_lse(self, out_dtype: torch.dtype):
        geometry = UNCOVERED_Q
        q, k, v = make_qkv(
            (geometry.total_q,), (geometry.total_k,), self.heads, DTYPE, self.device
        )
        sink = self._sink()
        out, lse = self._empty_state(geometry.total_q, out_dtype)

        self._fwd(q, k, v, out, lse, geometry.q_ranges, geometry.k_ranges, sink)

        uncovered = torch.ones(geometry.total_q, dtype=torch.bool, device=self.device)
        for start, end in geometry.q_ranges:
            uncovered[start:end] = False
        self.assertTrue(torch.all(out[uncovered] == 0), "uncovered O is not zero")
        torch.testing.assert_close(
            lse[uncovered],
            torch.logsumexp(sink, dim=0).expand(int(uncovered.sum()), -1),
        )
        out_ref, lse_ref = torch_reference_out_lse(
            q=q,
            k=k,
            v=v,
            q_ranges=geometry.q_ranges,
            k_ranges=geometry.k_ranges,
            sink=sink,
        )
        tol = OUT_TOL[out_dtype]
        torch.testing.assert_close(out.float(), out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(lse, lse_ref, atol=LSE_ATOL, rtol=LSE_RTOL)

    @parameterize("sink_call", ["first", "last"])
    def test_sink_counts_once_across_accumulating_calls(self, sink_call: str):
        """Passing the sink to the first or the last call gives the same result."""
        # k ranges are disjoint across calls; the last q span stays uncovered.
        calls = [
            ([[0, SPAN]], [[0, 2 * SPAN]]),
            ([[SPAN // 2, 2 * SPAN]], [[2 * SPAN, 3 * SPAN]]),
            ([[3 * SPAN // 2, 5 * SPAN // 2]], [[3 * SPAN, 4 * SPAN]]),
        ]
        total_q, total_k = 3 * SPAN, 4 * SPAN
        sink_call_index = 0 if sink_call == "first" else len(calls) - 1
        q, k, v = make_qkv((total_q,), (total_k,), self.heads, DTYPE, self.device)
        sink = self._sink()
        out, lse = self._empty_state(total_q, torch.float32)

        for index, (q_ranges, k_ranges) in enumerate(calls):
            call_sink = sink if index == sink_call_index else None
            self._fwd(q, k, v, out, lse, q_ranges, k_ranges, call_sink)

        out_ref, lse_ref = torch_reference_out_lse(
            q=q,
            k=k,
            v=v,
            q_ranges=[r for q_ranges, _ in calls for r in q_ranges],
            k_ranges=[r for _, k_ranges in calls for r in k_ranges],
            sink=sink,
        )
        tol = OUT_TOL[torch.float32]
        torch.testing.assert_close(out, out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(lse, lse_ref, atol=LSE_ATOL, rtol=LSE_RTOL)

    def test_empty_k_call_folds_the_sink(self):
        """Without keys the call only folds the sink into the stored state."""
        total_q = 2 * SPAN
        accumulated = slice(0, SPAN)
        empty = slice(SPAN, total_q)
        q, k, v = make_qkv((total_q,), (0,), self.heads, DTYPE, self.device)
        sink = self._sink()
        out, lse = self._empty_state(total_q, torch.float32)
        out[accumulated] = torch.randn_like(out[accumulated])
        lse[accumulated] = torch.randn_like(lse[accumulated])
        out_prev, lse_prev = out.clone(), lse.clone()

        self._fwd(q, k, v, out, lse, [[0, total_q]], [[0, 0]], sink)

        lse_sink = torch.logsumexp(sink, dim=0)
        lse_ref = torch.logaddexp(lse_prev[accumulated], lse_sink)
        rescale = torch.exp(lse_prev[accumulated] - lse_ref)[..., None]
        torch.testing.assert_close(
            lse[accumulated], lse_ref, atol=FOLD_ATOL, rtol=FOLD_RTOL
        )
        torch.testing.assert_close(
            out[accumulated],
            out_prev[accumulated] * rescale,
            atol=FOLD_ATOL,
            rtol=FOLD_RTOL,
        )
        self.assertTrue(torch.all(out[empty] == 0), "empty-row O is not zero")
        torch.testing.assert_close(lse[empty], lse_sink.expand(SPAN, -1))


# ─────────────────────────────────────────────────────────────────────────────
# dsink reduction kernel
# ─────────────────────────────────────────────────────────────────────────────


def _dsink_inputs(
    total_q: int, num_sinks: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.cuda.current_device()
    num_heads, head_dim = 3, 64
    out, dout = (
        torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(2)
    )
    sink = torch.randn(num_sinks, num_heads, device=device)
    # The forward LSE includes the sinks.
    lse = torch.logaddexp(
        torch.logsumexp(sink, dim=0),
        torch.randn(total_q, num_heads, device=device),
    )
    return out, dout, lse, sink


class TestDSinkReduction(unittest.TestCase):
    def setUp(self) -> None:
        seed_test_case(self.id())

    # Single row, full tile, and tails across one or several tiles.
    @parameterize("total_q", [1, DSINK_TILE_M, DSINK_TILE_M + 1, 3 * DSINK_TILE_M + 1])
    @parameterize("num_sinks", [1, 9])  # 9 exceeds the C++ backend's 8-sink cap.
    @parameterize("dtype", [torch.bfloat16, torch.float16])
    def test_rows_and_sink_counts(
        self, total_q: int, num_sinks: int, dtype: torch.dtype
    ):
        inputs = _dsink_inputs(total_q, num_sinks, dtype)
        dsink = bwd_dsink(*inputs)
        self.assertEqual(dsink.dtype, torch.float32)
        torch.testing.assert_close(
            dsink, dsink_reference(*inputs), atol=DSINK_ATOL, rtol=DSINK_RTOL
        )

    @parameterize("shape", [(0, 9), (DSINK_TILE_M + 1, 0)])
    def test_empty_inputs(self, shape: tuple[int, int]):
        total_q, num_sinks = shape
        inputs = _dsink_inputs(total_q, num_sinks, torch.bfloat16)
        dsink = bwd_dsink(*inputs)
        torch.testing.assert_close(dsink, dsink_reference(*inputs), atol=0, rtol=0)

    @parameterize("case", ["extreme_logits", "disabled_sinks", "empty_rows"])
    @parameterize("dtype", [torch.bfloat16, torch.float16])
    def test_numerical_corner_cases(self, case: str, dtype: torch.dtype):
        out, dout, lse, sink = _dsink_inputs(3 * DSINK_TILE_M + 1, 9, dtype)
        if case == "extreme_logits":
            # Shift logits and LSE together: probabilities stay finite.
            shifts = torch.tensor([1000.0, -1000.0, 0.0], device=sink.device)
            sink += shifts
            lse += shifts
        elif case == "disabled_sinks":
            sink[:, -1] = -math.inf
            lse[:, -1] = torch.randn_like(lse[:, -1])
        else:
            # Empty rows must contribute zero even with NaN O/dO.
            lse[::7] = -math.inf
            out[::7] = math.nan
            dout[::7] = math.nan

        dsink = bwd_dsink(out, dout, lse, sink)
        torch.testing.assert_close(
            dsink,
            dsink_reference(out, dout, lse, sink),
            atol=DSINK_ATOL,
            rtol=DSINK_RTOL,
        )
        if case == "disabled_sinks":
            self.assertTrue(
                torch.all(dsink[:, -1] == 0), "all -inf sinks need zero grad"
            )

    # The larger case has more query tiles than reducer threads.
    @parameterize("total_q", [DSINK_TILE_M, 2 * DSINK_TILE_M**2 + 1])
    def test_overwrites_a_strided_buffer_on_every_call(self, total_q: int):
        """Repeated calls start fresh completion counts and give equal results."""
        num_sinks = 9
        inputs = _dsink_inputs(total_q, num_sinks, torch.bfloat16)
        expected = dsink_reference(*inputs)
        storage = torch.full(
            (num_sinks, 2 * expected.shape[1]), math.nan, device=expected.device
        )
        dsink = storage[:, ::2]
        first_result = None
        for _ in range(3):
            dsink.fill_(math.nan)
            result = bwd_dsink(*inputs, dsink=dsink)
            self.assertIs(result, dsink)
            torch.testing.assert_close(
                result, expected, atol=DSINK_ATOL, rtol=DSINK_RTOL
            )
            if first_result is None:
                first_result = result.clone()
            torch.testing.assert_close(result, first_result, atol=0, rtol=0)
        self.assertTrue(torch.all(torch.isnan(storage[:, 1::2])), "wrote off-stride")

    def test_graph_replay_resets_the_completion_counter(self):
        # Multiple query tiles exercise the completion counter on every replay.
        out, dout, lse, sink = _dsink_inputs(16 * DSINK_TILE_M + 1, 9, torch.bfloat16)
        expected = dsink_reference(out, dout, lse, sink)
        bwd_dsink(out, dout, lse, sink)  # compile outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = bwd_dsink(out, dout, lse, sink)
        for _ in range(3):
            # Flipping dO flips dsink, so a stale result cannot pass.
            dout.neg_()
            expected.neg_()
            result.fill_(math.nan)
            graph.replay()
            torch.testing.assert_close(
                result, expected, atol=DSINK_ATOL, rtol=DSINK_RTOL
            )


if __name__ == "__main__":
    run_tests()
