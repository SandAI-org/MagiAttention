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

"""Empty rows and the sink fold of the atomic-merge forward postprocess.

Rows no relation covers keep whatever the O buffer held; with a sink the
postprocess must still leave them at O = 0, LSE = lse_sink, since a later
call merging into the same O/LSE reads O back for every finite LSE.
"""

import math
from unittest import TestCase

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd
from magi_attention.testing import parameterize


def _ref_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: list[list[int]],
    k_ranges: list[list[int]],
    sink: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 (out, lse) of the relation union plus one copy of the sink."""
    total_q, num_head, head_dim = q.shape
    mask = torch.zeros(total_q, k.shape[0], dtype=torch.bool, device=q.device)
    for (qs, qe), (ks, ke) in zip(q_ranges, k_ranges):
        mask[qs:qe, ks:ke] = True
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(head_dim)
    s = s.masked_fill(~mask, -math.inf)
    if sink is not None:
        sink_s = sink.float().t()[:, None, :].expand(num_head, total_q, -1)
        s = torch.cat([s, sink_s], dim=-1)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None]).nan_to_num(0.0)[..., : k.shape[0]]
    out = torch.einsum("hqk,khd->qhd", p, v.float())
    return out, lse.t().contiguous()


class TestFfaFwdPostprocess(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the atomic-merge fwd requires SM100/SM110")
        torch.manual_seed(42)

    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        q_ranges: list[list[int]],
        k_ranges: list[list[int]],
        sink: torch.Tensor | None,
    ) -> None:
        _flex_flash_attn_fwd(
            q,
            k,
            v,
            out=out,
            lse=lse,
            q_ranges=torch.tensor(q_ranges, dtype=torch.int32, device=q.device),
            k_ranges=torch.tensor(k_ranges, dtype=torch.int32, device=q.device),
            mask_types=MT_MAP.full,
            max_seqlen_q=max(e - s for s, e in q_ranges),
            max_seqlen_k=max(e - s for s, e in k_ranges),
            sink=sink,
            disable_fwd_atomic_reduction=False,
        )

    @parameterize("out_dtype", [torch.float32, torch.bfloat16])
    @parameterize("fill", [math.nan, math.inf])
    def test_sink_zeroes_uncovered_rows_without_reading_o(self, fill, out_dtype):
        """Uncovered rows whose O holds NaN/Inf end at exactly O = 0 and
        LSE = lse_sink; covered rows match the reference."""
        device, num_head, head_dim = "cuda", 4, 128
        total_q = total_k = 512
        # Rows [64, 96) merge two relations; [160, 288) and [352, 512) are holes.
        q_ranges = [[0, 96], [64, 160], [288, 352]]
        k_ranges = [[0, 128], [128, 320], [256, 512]]
        q, k, v = (
            torch.randn(n, num_head, head_dim, device=device, dtype=torch.bfloat16)
            for n in (total_q, total_k, total_k)
        )
        sink = torch.randn(2, num_head, device=device, dtype=torch.float32)
        out = torch.full(
            (total_q, num_head, head_dim), fill, device=device, dtype=out_dtype
        )
        lse = torch.full((total_q, num_head), -math.inf, device=device)

        self._fwd(q, k, v, out, lse, q_ranges, k_ranges, sink)

        out_ref, lse_ref = _ref_fwd(q, k, v, q_ranges, k_ranges, sink)
        holes = torch.ones(total_q, dtype=torch.bool, device=device)
        for qs, qe in q_ranges:
            holes[qs:qe] = False
        self.assertTrue(torch.all(out[holes] == 0), "hole O is not exactly zero")
        torch.testing.assert_close(
            lse[holes], torch.logsumexp(sink, dim=0).expand(int(holes.sum()), -1)
        )
        tol = 1e-2 if out_dtype == torch.float32 else 2e-2
        torch.testing.assert_close(out.float(), out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

    @parameterize("sink_call", [0, 1, 2])
    def test_sink_once_in_any_call_of_an_accumulation(self, sink_call):
        """Three calls merge into one fp32 O/LSE state, the sink passed to only
        one of them: the result matches the relation union plus one sink.

        With ``sink_call=0`` the rows the first call leaves empty go through
        the sink fold first (O = 0, LSE = lse_sink) and get their real keys
        from later calls, which read that O back."""
        device, num_head, head_dim = "cuda", 4, 128
        total_q, total_k = 384, 512
        # Disjoint K ranges, so no (q, k) pair is counted twice; [320, 384) is
        # never covered.
        calls = [
            ([[0, 128]], [[0, 256]]),
            ([[64, 256]], [[256, 384]]),
            ([[192, 320]], [[384, 512]]),
        ]
        q = torch.randn(
            total_q, num_head, head_dim, device=device, dtype=torch.bfloat16
        )
        k, v = (
            torch.randn(
                total_k, num_head, head_dim, device=device, dtype=torch.bfloat16
            )
            for _ in range(2)
        )
        sink = torch.randn(2, num_head, device=device, dtype=torch.float32)
        out = torch.full(
            (total_q, num_head, head_dim), math.nan, device=device, dtype=torch.float32
        )
        lse = torch.full((total_q, num_head), -math.inf, device=device)

        for i, (q_ranges, k_ranges) in enumerate(calls):
            self._fwd(
                q,
                k,
                v,
                out,
                lse,
                q_ranges,
                k_ranges,
                sink if i == sink_call else None,
            )

        out_ref, lse_ref = _ref_fwd(
            q,
            k,
            v,
            [r for qr, _ in calls for r in qr],
            [r for _, kr in calls for r in kr],
            sink,
        )
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

    @parameterize("with_sink", [False, True])
    def test_empty_k_call_keeps_the_accumulated_state(self, with_sink):
        """An atomic call with no K adds no pair to the caller's O/LSE: rows
        with a finite LSE keep their state (the sink, if passed, is folded in
        once), and rows still at LSE = -inf end at O = 0."""
        device, num_head, head_dim, total_q = "cuda", 4, 128, 256
        q = torch.randn(
            total_q, num_head, head_dim, device=device, dtype=torch.bfloat16
        )
        k = v = torch.empty(0, num_head, head_dim, device=device, dtype=q.dtype)
        sink = (
            torch.randn(2, num_head, device=device, dtype=torch.float32)
            if with_sink
            else None
        )
        # Rows [0, 128) carry a previous result; rows [128, 256) are empty.
        out = torch.randn(total_q, num_head, head_dim, device=device)
        out[128:] = math.nan
        lse = torch.randn(total_q, num_head, device=device)
        lse[128:] = -math.inf
        out_prev, lse_prev = out.clone(), lse.clone()

        self._fwd(q, k, v, out, lse, [[0, total_q]], [[0, 0]], sink)

        self.assertTrue(torch.all(out[128:] == 0), "empty-row O is not zero")
        if sink is None:
            torch.testing.assert_close(out[:128], out_prev[:128], atol=0, rtol=0)
            torch.testing.assert_close(lse, lse_prev, atol=0, rtol=0)
            return
        lse_sink = torch.logsumexp(sink, dim=0)
        lse_ref = torch.logaddexp(lse_prev[:128], lse_sink)
        torch.testing.assert_close(lse[:128], lse_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            out[:128],
            out_prev[:128] * torch.exp(lse_prev[:128] - lse_ref)[..., None],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(lse[128:], lse_sink.expand(128, -1))


if __name__ == "__main__":
    run_tests()
