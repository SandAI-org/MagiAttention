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

"""Deterministic q/k ranges of the SM100/SM110 CuTeDSL FFA kernels.

With ``deterministic=True`` the range atomic forward merges the O/LSE
partials of overlapping relations in relation order (``range_chain``), on a
persistent grid whose clusters claim every tile from one zero-based counter.
O and LSE are then bitwise identical across runs and across ``sm_margin``.
The protocol itself is checked on the CPU in
``test_ffa_range_deterministic_protocol.py``.
"""

import math
from contextlib import contextmanager
from typing import Iterator
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import (
    MT_MAP,
    get_device_arch,
    validate_range_deterministic,
)
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd
from magi_attention.testing import parameterize

_HEAD_DIM = 128
_NUM_HEAD_KV = 2
_TILE_M = 128
# fp32 O of bf16 inputs against the fp32 reference of the same bf16 inputs.
_TOL = 1e-2

Relations = list[tuple[list[int], list[int], int]]


@contextmanager
def _record_fwd() -> Iterator[tuple[list[FFAFwdSm100], list[torch.Tensor]]]:
    """Record the SM100 fwd kernel objects built and the DYNAMIC tile counters
    allocated within the context.

    The kernel object is only built on a JIT cache miss, so the context swaps
    in an empty compile cache. The tile counter is the only ``[1]`` int32
    tensor the fwd host allocates.
    """
    built: list[FFAFwdSm100] = []
    counters: list[torch.Tensor] = []
    init = FFAFwdSm100.__init__
    zeros = torch.zeros

    def record_init(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    def record_zeros(*args, **kwargs):
        t = zeros(*args, **kwargs)
        if t.shape == (1,) and t.dtype == torch.int32:
            counters.append(t)
        return t

    with mock.patch.object(FFAFwdSm100, "__init__", record_init), mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ), mock.patch.object(torch, "zeros", record_zeros):
        yield built, counters


def _relation_mask(len_q: int, len_k: int, mask_type: int, device) -> torch.Tensor:
    i = torch.arange(len_q, device=device)[:, None]
    j = torch.arange(len_k, device=device)[None, :]
    mask = torch.ones(len_q, len_k, dtype=torch.bool, device=device)
    if mask_type in (MT_MAP.causal, MT_MAP.bi_causal):
        mask &= j <= i + (len_k - len_q)
    if mask_type in (MT_MAP.inv_causal, MT_MAP.bi_causal):
        mask &= j >= i
    return mask


def _ref_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    relations: Relations,
    sink: torch.Tensor | None = None,
    softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 (out, lse, max_logits) of the relation union plus one sink copy.

    Relations must not share a (q, k) pair, so the union is their sum.
    """
    total_q, num_head, head_dim = q.shape
    group = num_head // k.shape[1]
    mask = torch.zeros(total_q, k.shape[0], dtype=torch.bool, device=q.device)
    for (qs, qe), (ks, ke), mask_type in relations:
        if qe <= qs or ke <= ks:
            continue
        rel = _relation_mask(qe - qs, ke - ks, mask_type, q.device)
        assert not (mask[qs:qe, ks:ke] & rel).any(), "relations share a pair"
        mask[qs:qe, ks:ke] |= rel
    s = torch.einsum(
        "qhd,khd->hqk", q.float(), k.repeat_interleave(group, dim=1).float()
    ) / math.sqrt(head_dim)
    if softcap is not None:
        s = softcap * torch.tanh(s / softcap)
    s = s.masked_fill(~mask, -math.inf)
    max_logits = s.amax(dim=(1, 2))
    if sink is not None:
        sink_s = sink.float().t()[:, None, :].expand(num_head, total_q, -1)
        s = torch.cat([s, sink_s], dim=-1)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None]).nan_to_num(0.0)[..., : k.shape[0]]
    out = torch.einsum("hqk,khd->qhd", p, v.repeat_interleave(group, dim=1).float())
    return out, lse.t().contiguous(), max_logits


def _qkv(total_q: int, total_k: int, group: int, head_dim: int = _HEAD_DIM):
    num_head = _NUM_HEAD_KV * group
    q = torch.randn(total_q, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
    k, v = (
        torch.randn(
            total_k, _NUM_HEAD_KV, head_dim, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(2)
    )
    return q, k, v


def _overlapping_relations(
    len_qs: list[int], coverage: int, mask_types: list[int]
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


def _range_args(relations: Relations, device="cuda") -> dict:
    q_ranges = [r[0] for r in relations]
    k_ranges = [r[1] for r in relations]
    return dict(
        q_ranges=torch.tensor(q_ranges, dtype=torch.int32, device=device),
        k_ranges=torch.tensor(k_ranges, dtype=torch.int32, device=device),
        mask_types=torch.tensor(
            [r[2] for r in relations], dtype=torch.int32, device=device
        ),
        max_seqlen_q=max(max(e - s for s, e in q_ranges), 1),
        max_seqlen_k=max(max(e - s for s, e in k_ranges), 1),
    )


class TestFfaRangeDeterministicHost(TestCase):
    """Host validation; launches no kernel."""

    def test_validate_accepts_head_dim_128_within_budget(self):
        validate_range_deterministic(
            head_dim=128,
            head_dim_v=128,
            num_sm=148,
            sm_margin=146,
            cluster_size=2,
            max_tickets=1000,
        )

    def test_validate_rejects_other_head_dims(self):
        for head_dim, head_dim_v in ((64, 64), (192, 128), (128, 64)):
            with self.assertRaises(NotImplementedError):
                validate_range_deterministic(
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    num_sm=148,
                    sm_margin=0,
                    cluster_size=1,
                    max_tickets=1,
                )

    def test_validate_rejects_a_budget_below_one_cluster(self):
        with self.assertRaises(ValueError):
            validate_range_deterministic(
                head_dim=128,
                head_dim_v=128,
                num_sm=148,
                sm_margin=147,
                cluster_size=2,
                max_tickets=1,
            )

    def test_validate_rejects_a_tile_counter_overflow(self):
        with self.assertRaises(ValueError):
            validate_range_deterministic(
                head_dim=128,
                head_dim_v=128,
                num_sm=148,
                sm_margin=0,
                cluster_size=1,
                max_tickets=2**31 - 100,
            )


class TestFfaRangeDeterministicFwd(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("deterministic q/k ranges require SM100/SM110")
        torch.manual_seed(42)
        self.num_sm = torch.cuda.get_device_properties(0).multi_processor_count

    def _log_ran(self, test_case: str) -> None:
        print(f"ran {test_case}", flush=True)

    def _det_fwd(self, q, k, v, relations: Relations, **kwargs):
        return _flex_flash_attn_fwd(
            q,
            k,
            v,
            **_range_args(relations),
            disable_fwd_atomic_reduction=False,
            deterministic=True,
            **kwargs,
        )

    def _assert_bitwise_across_sm_margin(self, run) -> tuple:
        """``run(sm_margin)`` -> tensors; equal bit for bit on every legal
        budget of the 1-CTA grid, including a single CTA."""
        first = None
        for sm_margin in (0, 1, self.num_sm - 7, self.num_sm - 1, 0):
            result = tuple(t.clone() for t in run(sm_margin))
            if first is None:
                first = result
                continue
            for got, want in zip(result, first):
                self.assertTrue(
                    torch.equal(got, want), f"differs at sm_margin={sm_margin}"
                )
        assert first is not None
        return first

    @parameterize(
        "case",
        [
            # (out dtype, q heads per kv head)
            (torch.float32, 1),
            (torch.float32, 2),
            (torch.bfloat16, 2),
        ],
    )
    def test_fwd_bitwise_across_runs_and_sm_margin(self, case):
        """Four relations cover every row of each unaligned q range; O/LSE do
        not depend on the grid and match the reference."""
        out_dtype, group = case
        relations, total_q, total_k = _overlapping_relations(
            [129, 300, 1000, 77], coverage=4, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group)
        out, lse = self._assert_bitwise_across_sm_margin(
            lambda m: self._det_fwd(
                q, k, v, relations, out_dtype=out_dtype, sm_margin=m
            )
        )
        self.assertEqual(out.dtype, out_dtype)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out.float(), out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
        self._log_ran(f"bitwise {case}")

    def test_fwd_empty_keyless_and_protection_block_relations(self):
        """An empty q range, a relation without keys (its tiles arrive with
        no contribution), a one-row range, and [1, 130) whose last tile only
        has a row in block 1 while its lock span reaches block 2, which a
        later relation starting at 256 owns alone."""
        relations: Relations = [
            ([1, 130], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([256, 400], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ]
        q, k, v = _qkv(420, 900, group=2)
        out, lse = self._assert_bitwise_across_sm_margin(
            lambda m: self._det_fwd(q, k, v, relations, sm_margin=m)
        )
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        finite = torch.isfinite(lse_ref)
        self.assertTrue(torch.equal(finite, torch.isfinite(lse)))
        torch.testing.assert_close(lse[finite], lse_ref[finite], atol=1e-3, rtol=1e-3)
        self._log_ran("empty/keyless/protection")

    def test_fwd_caller_buffer_accumulation_with_sink(self):
        """Two calls merge into one caller-provided fp32 state, the second
        with a sink; the final state is bitwise independent of the grid."""
        calls: list[Relations] = [
            [([0, 140], [0, 300], MT_MAP.full), ([70, 260], [300, 500], MT_MAP.causal)],
            [
                ([3, 250], [500, 800], MT_MAP.full),
                ([128, 129], [800, 820], MT_MAP.full),
            ],
        ]
        q, k, v = _qkv(260, 820, group=2)
        num_head = q.shape[1]
        sink = torch.randn(2, num_head, device="cuda", dtype=torch.float32)

        def run(sm_margin):
            out = torch.zeros(260, num_head, _HEAD_DIM, device="cuda")
            lse = torch.full((260, num_head), -math.inf, device="cuda")
            for i, relations in enumerate(calls):
                self._det_fwd(
                    q,
                    k,
                    v,
                    relations,
                    out=out,
                    lse=lse,
                    sink=sink if i == 1 else None,
                    sm_margin=sm_margin,
                )
            return out, lse

        out, lse = self._assert_bitwise_across_sm_margin(run)
        out_ref, lse_ref, _ = _ref_fwd(
            q, k, v, [r for relations in calls for r in relations], sink
        )
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
        self._log_ran("caller buffer + sink")

    def test_fwd_softcap_and_max_logits(self):
        relations, total_q, total_k = _overlapping_relations(
            [200, 45], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group=2)
        q = (q.float() * 4).to(q.dtype)  # reach the saturating range of the cap

        def run(sm_margin):
            max_logits = torch.full((q.shape[1],), -math.inf, device="cuda")
            out, lse = self._det_fwd(
                q,
                k,
                v,
                relations,
                softcap=5.0,
                max_logits=max_logits,
                sm_margin=sm_margin,
            )
            return out, lse, max_logits

        out, lse, max_logits = self._assert_bitwise_across_sm_margin(run)
        out_ref, lse_ref, max_ref = _ref_fwd(q, k, v, relations, softcap=5.0)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(max_logits, max_ref, atol=1e-3, rtol=1e-4)
        self._log_ran("softcap + max_logits")

    @parameterize("sm_margin_from_end", [None, 1, 3])
    def test_fwd_tile_counter_claims_every_tile_from_zero(self, sm_margin_from_end):
        """Every cluster claims its first tile from the counter too, so the
        counter ends at the tile count plus one final claim per cluster. The
        launch is persistent even without an SM reservation."""
        relations, total_q, total_k = _overlapping_relations(
            [129, 600], coverage=3, mask_types=[MT_MAP.full]
        )
        q, k, v = _qkv(total_q, total_k, group=2)
        sm_margin = (
            0 if sm_margin_from_end is None else self.num_sm - sm_margin_from_end
        )
        with _record_fwd() as (built, counters):
            self._det_fwd(
                q, k, v, relations, out_dtype=torch.bfloat16, sm_margin=sm_margin
            )
        torch.cuda.synchronize()
        (kernel,) = built
        self.assertTrue(kernel.deterministic and kernel.is_persistent)
        self.assertEqual(kernel.cluster_shape_mn, (1, 1))
        tile_rows = kernel.cta_tiler[0]
        num_head = q.shape[1]
        num_tiles = num_head * sum(
            math.ceil((e - s) / tile_rows) for (s, e), _, _ in relations
        )
        # SingleTileVarlenScheduler.get_grid_shape under DYNAMIC.
        grid_bound = (
            (total_q + len(relations) * (tile_rows - 1)) // tile_rows * num_head
        )
        num_clusters = min(grid_bound, self.num_sm - sm_margin)
        (counter,) = counters
        self.assertEqual(int(counter.item()), num_tiles + num_clusters)
        self._log_ran(f"counter sm_margin={sm_margin}")

    def test_fwd_rejects_unsupported_combinations(self):
        relations, total_q, total_k = _overlapping_relations(
            [40], coverage=2, mask_types=[MT_MAP.full]
        )
        q64, k64, v64 = _qkv(total_q, total_k, group=2, head_dim=64)
        with self.assertRaises(NotImplementedError):
            self._det_fwd(q64, k64, v64, relations)
        q, k, v = _qkv(total_q, total_k, group=2)
        with self.assertRaises(NotImplementedError):
            self._det_fwd(q, k, v, relations, pack_gqa=True)
        with self.assertRaises(ValueError):
            self._det_fwd(q, k, v, relations, sm_margin=self.num_sm)

    def test_direct_store_fwd_ignores_deterministic(self):
        """The direct store has one writer per O row: no chain, no forced
        persistence."""
        relations: Relations = [([0, 100], [0, 100], MT_MAP.full)]
        q, k, v = _qkv(100, 100, group=1)
        with _record_fwd() as (built, _):
            _flex_flash_attn_fwd(
                q,
                k,
                v,
                **_range_args(relations),
                disable_fwd_atomic_reduction=True,
                deterministic=True,
            )
        (kernel,) = built
        self.assertFalse(kernel.deterministic)


if __name__ == "__main__":
    run_tests()
