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
partials of overlapping relations in relation order, and the backward merges
the dQ partials of every (relation, cluster K tile) and the reduced dK/dV
partials of every (relation, q head) in that order (``range_chain``), on a
persistent grid whose clusters claim every tile from one zero-based counter.
The results are then bitwise identical across runs and across ``sm_margin``.
The protocol itself is checked on the CPU in
``test_ffa_range_deterministic_protocol.py``.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator
from unittest import TestCase, mock

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import flex_flash_attn_func, range_chain
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import (
    MT_MAP,
    get_device_arch,
    validate_range_deterministic,
)
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.testing import parameterize
from tests.test_kernel.cutedsl.range_deterministic_probes import (
    odd_lane_conflict_records,
    record_visits,
)

_HEAD_DIM = 128
_NUM_HEAD_KV = 2
_TILE_M = 128
# fp32 O of bf16 inputs against the fp32 reference of the same bf16 inputs.
_TOL = 1e-2

Relations = list[tuple[list[int], list[int], int]]


@dataclass
class _Launch:
    """What :func:`_record_launch` saw: kernel objects built, DYNAMIC tile
    counters, and range-chain states ``(blocks, heads, 2)`` in allocation
    order (fwd: O/LSE; bwd: dQ, then dK/dV when reduced)."""

    built: list = field(default_factory=list)
    counters: list[torch.Tensor] = field(default_factory=list)
    chains: list[torch.Tensor] = field(default_factory=list)


@contextmanager
def _record_launch(
    kernel_cls: type[FFAFwdSm100] | type[FFABwdSm100],
    host_fn: Callable,
) -> Iterator[_Launch]:
    """Record the SM100 kernel objects of ``kernel_cls`` built and the state
    tensors allocated by ``host_fn`` within the context.

    The kernel object is only built on a JIT cache miss, so the context swaps
    in an empty compile cache. The tile counter is the only ``[1]`` int32
    tensor the hosts allocate, and the chain states the only 3-D int32 ones
    of trailing size 2.
    """
    launch = _Launch()
    init = kernel_cls.__init__
    zeros = torch.zeros

    def record_init(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        launch.built.append(obj)

    def record_zeros(*args, **kwargs):
        t = zeros(*args, **kwargs)
        if t.dtype == torch.int32 and t.shape == (1,):
            launch.counters.append(t)
        if t.dtype == torch.int32 and t.dim() == 3 and t.shape[-1] == 2:
            launch.chains.append(t)
        return t

    with mock.patch.object(kernel_cls, "__init__", record_init), mock.patch.object(
        host_fn, "compile_cache", JITCache()
    ), mock.patch.object(torch, "zeros", record_zeros):
        yield launch


def _grid_clusters(
    num_ranges: int,
    num_head: int,
    total_rows: int,
    tile: int,
    cluster: int,
    sm_budget: int,
) -> int:
    """Clusters of the persistent grid, as SingleTileVarlenScheduler.get_grid_shape."""
    blocks = (total_rows + num_ranges * (cluster * tile - 1)) // tile
    grid_ctas = min(
        blocks // cluster * cluster * num_head, sm_budget // cluster * cluster
    )
    return grid_ctas // cluster


def _num_units(unit_rows: list[int], num_head: int, tile: int, cluster: int) -> int:
    """Cluster units: ``tile * cluster`` rows of one range and head."""
    return num_head * sum(math.ceil(n / (tile * cluster)) for n in unit_rows)


def _expected_tile_counter(
    unit_rows: list[int],
    num_head: int,
    total_rows: int,
    tile: int,
    cluster: int,
    sm_budget: int,
) -> int:
    """Final DYNAMIC tile counter under claim_first_tile: one claim per cluster
    unit plus the final claim of every cluster."""
    return _num_units(unit_rows, num_head, tile, cluster) + _grid_clusters(
        len(unit_rows), num_head, total_rows, tile, cluster, sm_budget
    )


def _assert_chain_settled(
    chain: torch.Tensor,
    ranges: list[list[int]],
    block_size: int,
    last_writer: Callable[[int], int],
) -> None:
    """After a launch, every block of every head has published the last
    writer with rows there, ``last_writer(r)`` of the highest such range
    (0: no writer), and no arrival count is left: every event arrived exactly
    once and every (writer, block) published."""
    torch.cuda.synchronize()
    chain = chain.cpu()
    want = torch.zeros(chain.shape[0], dtype=torch.int32)
    for r, (start, end) in enumerate(ranges):
        writer = last_writer(r)
        if end > start and writer != 0:
            want[start // block_size : (end - 1) // block_size + 1] = writer
    assert torch.equal(chain[..., 1], torch.zeros_like(chain[..., 1])), (
        "arrival counts left: " f"{chain[..., 1].nonzero().tolist()[:8]}"
    )
    got = chain[..., 0]
    assert torch.equal(
        got, want[:, None].expand_as(got)
    ), f"published {got[:, 0].tolist()}, expected {want.tolist()}"


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


def _qkv(
    total_q: int,
    total_k: int,
    group: int,
    head_dim: int = _HEAD_DIM,
    num_head_kv: int = _NUM_HEAD_KV,
):
    num_head = num_head_kv * group
    q = torch.randn(total_q, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
    k, v = (
        torch.randn(total_k, num_head_kv, head_dim, device="cuda", dtype=torch.bfloat16)
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
            max_writer=2**31 - 1,
        )

    def test_validate_rejects_other_head_dims(self):
        for head_dim, head_dim_v in ((64, 64), (192, 128), (128, 64), (192, 192)):
            with self.assertRaises(NotImplementedError):
                validate_range_deterministic(
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    num_sm=148,
                    sm_margin=0,
                    cluster_size=1,
                    max_tickets=1,
                    max_writer=1,
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
                max_writer=1,
            )

    def test_validate_counts_the_final_claims_in_the_tile_counter_bound(self):
        """The counter ends at the tickets plus one final claim per cluster:
        74 2-CTA clusters on 148 SMs."""
        kwargs = dict(
            head_dim=128, head_dim_v=128, num_sm=148, sm_margin=0, max_writer=1
        )
        validate_range_deterministic(
            cluster_size=2, max_tickets=2**31 - 1 - 74, **kwargs
        )
        with self.assertRaises(ValueError):
            validate_range_deterministic(
                cluster_size=2, max_tickets=2**31 - 74, **kwargs
            )

    def test_validate_rejects_a_writer_number_overflow(self):
        kwargs = dict(
            head_dim=128,
            head_dim_v=128,
            num_sm=148,
            sm_margin=0,
            cluster_size=1,
            max_tickets=1,
        )
        validate_range_deterministic(max_writer=2**31 - 1, **kwargs)
        with self.assertRaises(ValueError):
            validate_range_deterministic(max_writer=2**31, **kwargs)

    def test_tile_counter_bound_counts_overlapping_ranges(self):
        """2^22 ranges all covering the same 4096 rows: the units are bounded
        per range by the longest range, not by the token count, and overflow
        the int32 tile counter in both directions (fwd: 16 q heads, 32 tiles
        of 128 rows; bwd: 32 q heads, 16 cluster K tiles of 256 rows)."""
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("deterministic q/k ranges require SM100/SM110")
        num_ranges, rows = 2**22, 4096
        ranges = torch.tensor([0, rows], dtype=torch.int32, device="cuda").expand(
            num_ranges, 2
        )
        args = dict(
            q_ranges=ranges.contiguous(),
            k_ranges=ranges.contiguous(),
            mask_types=0,
            max_seqlen_q=rows,
            max_seqlen_k=rows,
        )
        q, k, v = _qkv(rows, rows, group=1, num_head_kv=16)
        with self.assertRaisesRegex(ValueError, "tile counter"):
            _flex_flash_attn_fwd(
                q, k, v, **args, disable_fwd_atomic_reduction=False, deterministic=True
            )
        q, k, v = _qkv(rows, rows, group=1, num_head_kv=32)
        lse = torch.zeros(rows, q.shape[1], device="cuda")
        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ), self.assertRaisesRegex(ValueError, "tile counter"):
            _flex_flash_attn_bwd(
                q,
                k,
                v,
                q,
                lse,
                q,
                **args,
                disable_bwd_dkv_atomic_reduction=False,
                deterministic=True,
            )

    def test_range_merge_is_rejected_before_its_preprocessing(self):
        """deterministic + RangeMerge raises at the entry of the autograd
        function and of the raw backward, before the merge plan is built,
        and does not fall back to range_merge=False or deterministic=False."""
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device for the input tensors")
        relations: Relations = [
            ([0, 100], [0, 100], MT_MAP.full),
            ([100, 200], [100, 200], MT_MAP.full),
        ]
        q, k, v = _qkv(200, 200, group=1)
        lse = torch.zeros(200, q.shape[1], device="cuda")
        contracts = dict(
            disable_fwd_atomic_reduction=True, disable_bwd_dkv_atomic_reduction=True
        )
        with mock.patch(
            "magi_attention.kernel.cutedsl.flex_flash_attn._apply_range_merge",
            side_effect=AssertionError("RangeMerge preprocessing ran"),
        ) as apply_range_merge:
            with self.assertRaisesRegex(NotImplementedError, "RangeMerge"):
                flex_flash_attn_func(
                    q,
                    k,
                    v,
                    **_range_args(relations),
                    range_merge=True,
                    deterministic=True,
                    **contracts,
                )
            with self.assertRaisesRegex(NotImplementedError, "RangeMerge"):
                _flex_flash_attn_bwd(
                    q,
                    k,
                    v,
                    q,
                    lse,
                    q,
                    **_range_args(relations),
                    range_merge=True,
                    deterministic=True,
                    disable_bwd_dkv_atomic_reduction=True,
                )
        apply_range_merge.assert_not_called()


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

    def _assert_fwd_chain_settled(self, q, k, v, relations: Relations, **kwargs):
        """Relation r is writer r + 1 of every 128-row block its q range has
        a row in, empty and keyless relations included."""
        with _record_launch(FFAFwdSm100, _flex_flash_attn_fwd) as launch:
            self._det_fwd(q, k, v, relations, **kwargs)
        (chain,) = launch.chains
        _assert_chain_settled(
            chain, [r[0] for r in relations], _TILE_M, lambda r: r + 1
        )

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
        self._assert_fwd_chain_settled(q, k, v, relations)
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
        with _record_launch(FFAFwdSm100, _flex_flash_attn_fwd) as launch:
            self._det_fwd(
                q, k, v, relations, out_dtype=torch.bfloat16, sm_margin=sm_margin
            )
        torch.cuda.synchronize()
        (kernel,) = launch.built
        self.assertTrue(kernel.deterministic and kernel.is_persistent)
        self.assertEqual(kernel.cluster_shape_mn, (1, 1))
        (counter,) = launch.counters
        expected = _expected_tile_counter(
            [e - s for (s, e), _, _ in relations],
            num_head=q.shape[1],
            total_rows=total_q,
            tile=kernel.cta_tiler[0],
            cluster=1,
            sm_budget=self.num_sm - sm_margin,
        )
        self.assertEqual(int(counter.item()), expected)
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
        with _record_launch(FFAFwdSm100, _flex_flash_attn_fwd) as launch:
            _flex_flash_attn_fwd(
                q,
                k,
                v,
                **_range_args(relations),
                disable_fwd_atomic_reduction=True,
                deterministic=True,
            )
        (kernel,) = launch.built
        self.assertFalse(kernel.deterministic)


def _shared_k_relations() -> tuple[Relations, int, int]:
    """Relations sharing unaligned K rows (reduced dK/dV) and overlapping Q
    rows (merged O, dQ), without a repeated (q, k) pair: the A relations have
    disjoint Q over overlapping K, and every B relation overlaps two A
    relations in Q over K rows no A relation touches."""
    relations: Relations = []
    for i in range(4):
        relations.append(
            ([5 + 300 * i, 265 + 300 * i], [7 + 30 * i, 407 + 30 * i], MT_MAP.full)
        )
        relations.append(
            (
                [105 + 300 * i, 355 + 300 * i],
                [600 + 13 * i, 900 + 13 * i],
                MT_MAP.causal if i % 2 else MT_MAP.full,
            )
        )
    return relations, 1260, 944


def _assert_bwd_chains_settled(
    launch: _Launch, relations: Relations, group: int, cat_gqa: bool
) -> None:
    """dQ: writer r * S + n + 1 for cluster K tile n of relation r, with S the
    cluster K tiles of max_seqlen_k (the longest K range, see _range_args),
    on the tile_m-row Q blocks; relations without K tiles have no writer.
    dK/dV (when reduced): writer (r + 1) * G for the last q head of relation
    r, G = 1 under cat_gqa or MHA, on the cluster K-tile blocks."""
    (kernel,) = launch.built
    cluster_tile_k = kernel.tile_n * kernel.cta_group_size
    k_tiles = [math.ceil((ke - ks) / cluster_tile_k) for _, (ks, ke), _ in relations]
    stride = max(
        math.ceil(max(ke - ks for _, (ks, ke), _ in relations) / cluster_tile_k), 1
    )
    dq_chain, *dkv_chain = launch.chains
    _assert_chain_settled(
        dq_chain,
        [r[0] for r in relations],
        kernel.tile_m,
        lambda r: r * stride + k_tiles[r] if k_tiles[r] else 0,
    )
    if dkv_chain:
        dkv_stride = 1 if cat_gqa else group
        _assert_chain_settled(
            dkv_chain[0],
            [r[1] for r in relations],
            cluster_tile_k,
            lambda r: (r + 1) * dkv_stride,
        )


class TestFfaRangeDeterministicBwd(TestCase):
    """dQ of every (relation, cluster K tile) is chained per Q row block, and
    reduced dK/dV of every (relation, q head) per K row block."""

    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("deterministic q/k ranges require SM100/SM110")
        torch.manual_seed(42)
        self.num_sm = torch.cuda.get_device_properties(0).multi_processor_count

    def _log_ran(self, test_case: str) -> None:
        print(f"ran {test_case}", flush=True)

    def _fwd_bwd_inputs(self, relations: Relations, total_q, total_k, group):
        q, k, v = _qkv(total_q, total_k, group)
        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            **_range_args(relations),
            disable_fwd_atomic_reduction=False,
            deterministic=True,
        )
        return q, k, v, out.to(q.dtype), lse, torch.randn_like(q)

    def _det_bwd(self, q, k, v, out, lse, do, relations, direct_dkv=True, **kwargs):
        return _flex_flash_attn_bwd(
            q,
            k,
            v,
            out,
            lse,
            do,
            **_range_args(relations),
            disable_bwd_dkv_atomic_reduction=direct_dkv,
            deterministic=True,
            **kwargs,
        )[:3]

    def _assert_grads_bitwise_across_sm_margin(
        self, q, k, v, out, lse, do, relations, two_cta, **kwargs
    ):
        """Gradients on every legal budget of the cluster size are equal bit
        for bit; returns them after checking the reference."""
        cluster = 2 if two_cta else 1
        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0" if two_cta else "1"}
        first = None
        with mock.patch.dict("os.environ", env):
            for sm_margin in (
                0,
                1,
                self.num_sm - 4 * cluster,
                self.num_sm - cluster,
                0,
            ):
                with _record_launch(FFABwdSm100, _flex_flash_attn_bwd) as launch:
                    grads = self._det_bwd(
                        q, k, v, out, lse, do, relations, sm_margin=sm_margin, **kwargs
                    )
                (kernel,) = launch.built
                self.assertEqual(kernel.use_2cta_instrs, two_cta)
                self.assertTrue(kernel.is_persistent)
                _assert_bwd_chains_settled(
                    launch,
                    relations,
                    group=q.shape[1] // k.shape[1],
                    cat_gqa=kwargs.get("cat_gqa", False),
                )
                if first is None:
                    first = [g.clone() for g in grads]
                    continue
                for name, got, want in zip("qkv", grads, first):
                    self.assertTrue(
                        torch.equal(got, want),
                        f"d{name} differs at sm_margin={sm_margin}",
                    )
        assert first is not None
        refs = self._ref_grads(q, k, v, do, relations)
        for name, got, want in zip("qkv", first, refs):
            torch.testing.assert_close(
                got.float(), want, atol=3e-2, rtol=3e-2, msg=lambda m: f"d{name}: {m}"
            )
        return first

    def _ref_grads(self, q, k, v, do, relations):
        qkv = [t.detach().float().requires_grad_() for t in (q, k, v)]
        out_ref, _, _ = _ref_fwd(*qkv, relations)
        out_ref.backward(do.float())
        return [t.grad for t in qkv]

    @parameterize(
        "case",
        [
            # (q heads per kv head, cat_gqa, 2-CTA)
            (1, False, True),
            (1, False, False),
            (2, True, True),
            (2, True, False),
        ],
    )
    def test_bwd_dq_bitwise_across_sm_margin(self, case):
        """Three relations cover every row of each unaligned q range, half of
        them causal, so some Q tiles of a K tile are masked out and only
        chain; the dK/dV store is direct, one writer per K row."""
        group, cat_gqa, two_cta = case
        relations, total_q, total_k = _overlapping_relations(
            [129, 300, 1000, 77], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v, out, lse, do = self._fwd_bwd_inputs(relations, total_q, total_k, group)
        self._assert_grads_bitwise_across_sm_margin(
            q, k, v, out, lse, do, relations, two_cta, cat_gqa=cat_gqa
        )
        self._log_ran(f"bwd dq bitwise {case}")

    @parameterize(
        "case",
        [
            # (q heads per kv head, cat_gqa, 2-CTA)
            (1, False, True),
            (2, False, True),
            (2, False, False),
            (2, True, True),
        ],
    )
    def test_bwd_reduced_dkv_bitwise_across_sm_margin(self, case):
        """Relations share unaligned K rows, so dK/dV are reduced: per
        (relation, q head) without cat_gqa, per relation with it or MHA."""
        group, cat_gqa, two_cta = case
        relations, total_q, total_k = _shared_k_relations()
        q, k, v, out, lse, do = self._fwd_bwd_inputs(relations, total_q, total_k, group)
        self._assert_grads_bitwise_across_sm_margin(
            q,
            k,
            v,
            out,
            lse,
            do,
            relations,
            two_cta,
            direct_dkv=False,
            cat_gqa=cat_gqa,
        )
        self._log_ran(f"bwd dkv bitwise {case}")

    def test_bwd_empty_and_keyless_relations(self):
        """A relation without keys has no K tile and so no dQ writer; an empty
        q range has no Q block; a one-row range and a range starting at a
        block's last row still chain."""
        relations: Relations = [
            ([1, 130], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([127, 400], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ]
        q, k, v, out, lse, do = self._fwd_bwd_inputs(relations, 420, 900, group=1)
        first = None
        for sm_margin in (0, self.num_sm - 2, 0):
            grads = self._det_bwd(q, k, v, out, lse, do, relations, sm_margin=sm_margin)
            if first is None:
                first = [g.clone() for g in grads]
                continue
            for got, want in zip(grads, first):
                self.assertTrue(torch.equal(got, want))
        assert first is not None
        for name, got, want in zip(
            "qkv", first, self._ref_grads(q, k, v, do, relations)
        ):
            torch.testing.assert_close(
                got.float(), want, atol=3e-2, rtol=3e-2, msg=lambda m: f"d{name}: {m}"
            )
        self._log_ran("bwd empty/keyless")

    def test_bwd_tile_counter_claims_every_tile_from_zero(self):
        relations, total_q, total_k = _overlapping_relations(
            [129, 600], coverage=2, mask_types=[MT_MAP.full]
        )
        q, k, v, out, lse, do = self._fwd_bwd_inputs(relations, total_q, total_k, 1)
        sm_margin = self.num_sm - 6
        with _record_launch(FFABwdSm100, _flex_flash_attn_bwd) as launch:
            self._det_bwd(q, k, v, out, lse, do, relations, sm_margin=sm_margin)
        torch.cuda.synchronize()
        (kernel,) = launch.built
        (counter,) = launch.counters
        expected = _expected_tile_counter(
            [e - s for _, (s, e), _ in relations],
            num_head=q.shape[1],
            total_rows=total_k,
            tile=kernel.tile_n,
            cluster=kernel.cta_group_size,
            sm_budget=self.num_sm - sm_margin,
        )
        self.assertEqual(int(counter.item()), expected)

    def test_bwd_rejects_unsupported_combinations(self):
        relations, total_q, total_k = _overlapping_relations(
            [40], coverage=2, mask_types=[MT_MAP.full]
        )
        q, k, v, out, lse, do = self._fwd_bwd_inputs(relations, total_q, total_k, 1)
        q64, k64, v64 = _qkv(total_q, total_k, group=1, head_dim=64)
        with self.assertRaises(NotImplementedError):
            self._det_bwd(q64, k64, v64, out[..., :64], lse, do[..., :64], relations)
        # A 2-CTA cluster does not fit one SM, and the cluster size is kept.
        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ), self.assertRaises(ValueError):
            self._det_bwd(q, k, v, out, lse, do, relations, sm_margin=self.num_sm - 1)
        with self.assertRaisesRegex(NotImplementedError, "RangeMerge"):
            self._det_bwd(q, k, v, out, lse, do, relations, range_merge=True)

    def test_public_api_fwd_bwd_bitwise_with_sink_and_softcap(self):
        """flex_flash_attn_func(deterministic=True) on GQA with reduced dK/dV,
        a sink and a softcap: O, LSE and every gradient, dsink included, are
        bitwise equal across sm_margin."""
        relations, total_q, total_k = _shared_k_relations()
        q, k, v = _qkv(total_q, total_k, group=2)
        sink = torch.randn(2, q.shape[1], device="cuda", dtype=torch.float32)
        do = torch.randn_like(q)

        def run(sm_margin):
            leaves = [t.detach().clone().requires_grad_() for t in (q, k, v, sink)]
            out, meta = flex_flash_attn_func(
                *leaves[:3],
                **_range_args(relations),
                sink=leaves[3],
                softcap=8.0,
                deterministic=True,
                sm_margin=sm_margin,
            )
            out.backward(do)
            return [out.detach(), meta.lse, *(t.grad for t in leaves)]

        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ):
            first = run(0)
            for sm_margin in (1, self.num_sm - 2, 0):
                for name, got, want in zip(
                    ("out", "lse", "dq", "dk", "dv", "dsink"), run(sm_margin), first
                ):
                    self.assertTrue(
                        torch.equal(got, want),
                        f"{name} differs at sm_margin={sm_margin}",
                    )
        self._log_ran("public api")


class _ScanConflictsLauncher:
    """One warp running ``range_chain.scan_conflicts`` over every range into
    conflict-table row 0, with writer ``r + 1`` for range ``r``."""

    def __init__(self, block_size: int):
        self.block_size = block_size

    @cute.jit
    def __call__(
        self,
        mRanges: cute.Tensor,
        mConflict: cute.Tensor,
        num_ranges: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mRanges, mConflict, num_ranges).launch(
            grid=(1, 1, 1), block=(cute.arch.WARP_SIZE, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, mRanges: cute.Tensor, mConflict: cute.Tensor, num_ranges: Int32):
        range_chain.scan_conflicts(
            mRanges,
            mConflict,
            Int32(0),
            Int32(0),
            num_ranges,
            self.block_size,
            last_writer=lambda r: r + 1,
        )


class TestFfaRangeChainScan(TestCase):
    """The conflict scan records, per block, the last range with a row there,
    also when consecutive ranges share a block through different lanes and
    the stores of one range are delayed past those of the next."""

    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the range chain runs on SM100/SM110")

    def _scan(self, ranges: list[list[int]], block_size: int) -> torch.Tensor:
        num_blocks = max(e for _, e in ranges) // block_size + 2
        mranges = torch.tensor(ranges, dtype=torch.int32, device="cuda")
        conflict = torch.zeros(1, num_blocks, dtype=torch.int32, device="cuda")
        args = (
            from_dlpack(mranges),
            from_dlpack(conflict),
            Int32(len(ranges)),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        cute.compile(_ScanConflictsLauncher(block_size), *args)(*args)
        torch.cuda.synchronize()
        return conflict[0].cpu()

    @parameterize(
        "ranges",
        [
            # the shared block is lane 1's for r0 and lane 0's for r1
            [[0, 256], [128, 256]],
            # unaligned starts, three ranges over block 1
            [[5, 300], [130, 140], [0, 129]],
            # more than 32 blocks: block 33 is lane 1's second store for r0
            [[0, 40 * 128], [33 * 128 + 7, 35 * 128]],
            # an empty range between two that share blocks
            [[0, 384], [200, 200], [129, 384]],
        ],
    )
    def test_scan_keeps_the_last_range_per_block_under_delayed_lanes(self, ranges):
        block_size = 128
        want = torch.zeros(
            max(e for _, e in ranges) // block_size + 2, dtype=torch.int32
        )
        for r, (start, end) in enumerate(ranges):
            if end > start:
                want[start // block_size : (end - 1) // block_size + 1] = r + 1
        self.assertTrue(torch.equal(self._scan(ranges, block_size), want))
        with odd_lane_conflict_records(50_000):
            got = self._scan(ranges, block_size)
        self.assertTrue(torch.equal(got, want), f"{got.tolist()} != {want.tolist()}")


# Relations of one head with T cluster units around a grid of C clusters,
# T in {0, 1, C - 1, C, C + 1}; the overlapping ones also chain.
_FWD_UNIT_CLUSTERS = 8  # 1-CTA: two 128-row q stages, 256 rows per unit
_FWD_UNIT_RELATIONS: dict[int, Relations] = {
    0: [([40, 40], [0, 100], MT_MAP.full), ([90, 90], [100, 300], MT_MAP.full)],
    1: [([5, 105], [0, 100], MT_MAP.full)],
    7: [([0, 768], [0, 300], MT_MAP.full), ([100, 1100], [300, 700], MT_MAP.causal)],
    8: [([0, 768], [0, 300], MT_MAP.full), ([100, 1300], [300, 700], MT_MAP.causal)],
    9: [([0, 768], [0, 300], MT_MAP.full), ([100, 1500], [300, 700], MT_MAP.causal)],
}
_BWD_UNIT_CLUSTERS = 4  # 2-CTA: 256-row cluster K tiles
_BWD_UNIT_RELATIONS: dict[int, Relations] = {
    0: [([0, 300], [10, 10], MT_MAP.full), ([100, 400], [20, 20], MT_MAP.full)],
    1: [([0, 300], [0, 200], MT_MAP.full)],
    3: [([0, 300], [0, 512], MT_MAP.full), ([100, 400], [600, 800], MT_MAP.causal)],
    4: [([0, 300], [0, 512], MT_MAP.full), ([100, 400], [600, 1100], MT_MAP.causal)],
    5: [([0, 300], [0, 768], MT_MAP.full), ([100, 400], [800, 1300], MT_MAP.causal)],
}


class TestFfaRangeDeterministicClaims(TestCase):
    """Every valid DYNAMIC unit is claimed once and consumed by every warp of
    every CTA of one cluster, for unit counts around the cluster count, and
    the deterministic launch replays under CUDA Graphs."""

    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("deterministic q/k ranges require SM100/SM110")
        torch.manual_seed(42)
        self.num_sm = torch.cuda.get_device_properties(0).multi_processor_count

    def _assert_each_unit_visited_once(
        self, visits: torch.Tensor, clusters: torch.Tensor, num_units: int, cluster: int
    ) -> None:
        """Tiles ``unit * cluster + rank`` of the valid units are consumed by
        the same number of warps, on every rank by the same cluster; no other
        tile is valid."""
        visits, clusters = visits.cpu(), clusters.cpu()
        valid = num_units * cluster
        self.assertTrue(torch.equal(visits[valid:], torch.zeros_like(visits[valid:])))
        if num_units == 0:
            return
        warps = int(visits[0])
        self.assertGreater(warps, 0)
        self.assertTrue(
            torch.equal(visits[:valid], torch.full((valid,), warps, dtype=torch.int32)),
            f"warps per tile: {visits[:valid].tolist()}",
        )
        per_unit = clusters[:valid].view(num_units, cluster)
        self.assertTrue(torch.equal(per_unit % warps, torch.zeros_like(per_unit)))
        self.assertTrue(
            torch.equal(per_unit, per_unit[:, :1].expand_as(per_unit)),
            f"cluster sums per unit: {per_unit.tolist()}",
        )

    @parameterize("num_units", sorted(_FWD_UNIT_RELATIONS))
    def test_fwd_claims_each_unit_once(self, num_units):
        relations = _FWD_UNIT_RELATIONS[num_units]
        total_q, total_k = 2000, 700
        q, k, v = _qkv(total_q, total_k, group=1, num_head_kv=1)
        sm_margin = self.num_sm - _FWD_UNIT_CLUSTERS
        with record_visits(num_units + 64, 1) as (visits, clusters), _record_launch(
            FFAFwdSm100, _flex_flash_attn_fwd
        ) as launch:
            out, lse = _flex_flash_attn_fwd(
                q,
                k,
                v,
                **_range_args(relations),
                disable_fwd_atomic_reduction=False,
                deterministic=True,
                sm_margin=sm_margin,
            )
        torch.cuda.synchronize()
        (kernel,) = launch.built
        rows = [e - s for (s, e), _, _ in relations]
        self.assertEqual(_num_units(rows, 1, kernel.cta_tiler[0], 1), num_units)
        clusters_launched = _grid_clusters(
            len(relations), 1, total_q, kernel.cta_tiler[0], 1, _FWD_UNIT_CLUSTERS
        )
        self.assertEqual(clusters_launched, _FWD_UNIT_CLUSTERS)
        (counter,) = launch.counters
        self.assertEqual(int(counter.item()), num_units + clusters_launched)
        self._assert_each_unit_visited_once(visits, clusters, num_units, 1)
        (chain,) = launch.chains
        _assert_chain_settled(
            chain, [r[0] for r in relations], _TILE_M, lambda r: r + 1
        )
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        print(f"ran fwd claims T={num_units}", flush=True)

    @parameterize("num_units", sorted(_BWD_UNIT_RELATIONS))
    def test_bwd_2cta_claims_each_unit_once_on_both_peers(self, num_units):
        relations = _BWD_UNIT_RELATIONS[num_units]
        total_q, total_k = 400, 1400
        q, k, v = _qkv(total_q, total_k, group=1, num_head_kv=1)
        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            **_range_args(relations),
            disable_fwd_atomic_reduction=False,
            deterministic=True,
        )
        out, do = out.to(q.dtype), torch.randn_like(q)
        sm_margin = self.num_sm - 2 * _BWD_UNIT_CLUSTERS
        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ), record_visits(2 * (num_units + 16), 2) as (
            visits,
            clusters,
        ), _record_launch(
            FFABwdSm100, _flex_flash_attn_bwd
        ) as launch:
            grads = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                **_range_args(relations),
                disable_bwd_dkv_atomic_reduction=False,
                deterministic=True,
                sm_margin=sm_margin,
            )[:3]
        torch.cuda.synchronize()
        (kernel,) = launch.built
        self.assertTrue(kernel.use_2cta_instrs)
        rows = [e - s for _, (s, e), _ in relations]
        self.assertEqual(_num_units(rows, 1, kernel.tile_n, 2), num_units)
        clusters_launched = _grid_clusters(
            len(relations), 1, total_k, kernel.tile_n, 2, 2 * _BWD_UNIT_CLUSTERS
        )
        self.assertEqual(clusters_launched, _BWD_UNIT_CLUSTERS)
        (counter,) = launch.counters
        self.assertEqual(int(counter.item()), num_units + clusters_launched)
        self._assert_each_unit_visited_once(visits, clusters, num_units, 2)
        _assert_bwd_chains_settled(launch, relations, group=1, cat_gqa=False)
        qkv = [t.detach().float().requires_grad_() for t in (q, k, v)]
        _ref_fwd(*qkv, relations)[0].backward(do.float())
        for name, got, ref in zip("qkv", grads, qkv):
            torch.testing.assert_close(
                got.float(),
                ref.grad,
                atol=3e-2,
                rtol=3e-2,
                msg=lambda m: f"d{name}: {m}",
            )
        print(f"ran bwd claims T={num_units}", flush=True)

    def test_cuda_graph_replay_matches_eager_bitwise(self):
        """The chain, conflict and tile-counter states are re-zeroed inside
        the captured graph, so replays on new data equal eager runs bit for
        bit, forward and backward (2-CTA, GQA, reduced dK/dV)."""
        relations, total_q, total_k = _shared_k_relations()
        args = _range_args(relations)

        def fwd(q, k, v):
            return _flex_flash_attn_fwd(
                q, k, v, **args, disable_fwd_atomic_reduction=False, deterministic=True
            )

        def bwd(q, k, v, out, lse, do):
            return _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                **args,
                disable_bwd_dkv_atomic_reduction=False,
                deterministic=True,
            )[:3]

        def inputs():
            q, k, v = _qkv(total_q, total_k, group=2)
            out, lse = fwd(q, k, v)
            return q, k, v, out.to(q.dtype), lse, torch.randn_like(q)

        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ):
            data = [inputs(), inputs()]
            eager = [
                ([t.clone() for t in fwd(*d[:3])], [g.clone() for g in bwd(*d)])
                for d in data
            ]
            static = [t.clone() for t in data[0]]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                fwd(*static[:3])
                bwd(*static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_fwd = fwd(*static[:3])
                static_bwd = bwd(*static)
            for i in (1, 0, 1):
                for dst, src in zip(static, data[i]):
                    dst.copy_(src)
                graph.replay()
                torch.cuda.synchronize()
                want_fwd, want_bwd = eager[i]
                for name, got, want in zip(
                    ("out", "lse", "dq", "dk", "dv"),
                    [*static_fwd, *static_bwd],
                    [*want_fwd, *want_bwd],
                ):
                    self.assertTrue(
                        torch.equal(got, want), f"{name} of replay on data {i}"
                    )
        print("ran cuda graph replay", flush=True)


if __name__ == "__main__":
    run_tests()
