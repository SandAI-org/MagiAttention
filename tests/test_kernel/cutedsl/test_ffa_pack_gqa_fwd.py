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

"""GQA row packing on the SM100 CuTeDSL range atomic forward.

Packed row ``p`` of a relation starting at token ``q_start`` of kv head
``h_kv`` is the physical ``(q_start + p // G, h_kv * G + p % G)``. O, LSE,
range locks and max logits all use that physical identity, and the range
atomic path sizes q_stage by the Q rows the MMA actually sees:
``max_seqlen_q * G`` when packed, ``max_seqlen_q`` otherwise.
"""

import math
from contextlib import contextmanager
from typing import Iterator
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.functional.cutedsl_ffa import cutedsl_fwd
from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.utils import make_attn_mask_from_ffa_args

_TILE_M = 128
_HEAD_DIM = 128
_NUM_HEAD_KV = 2
# fp32 O of bf16 inputs against the fp32 reference of the same bf16 inputs.
_TOL = 1e-2

Relations = list[tuple[list[int], list[int], int]]


@contextmanager
def _record_fwd_kernels() -> Iterator[list[FFAFwdSm100]]:
    """Record the SM100 fwd kernel objects built within the context.

    The specialization is only visible on the kernel object, which a JIT cache
    hit never builds, so the context also swaps in an empty compile cache.
    """
    built: list[FFAFwdSm100] = []
    init = FFAFwdSm100.__init__

    def record(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    with mock.patch.object(FFAFwdSm100, "__init__", record), mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ):
        yield built


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 (out, lse, max_logits) of the relation union plus one sink copy.

    Relations must not share a (q, k) pair, so the union is their sum.
    """
    total_q, num_head, head_dim = q.shape
    group = num_head // k.shape[1]
    mask = torch.zeros(total_q, k.shape[0], dtype=torch.bool, device=q.device)
    for (qs, qe), (ks, ke), mask_type in relations:
        rel = _relation_mask(qe - qs, ke - ks, mask_type, q.device)
        assert not (mask[qs:qe, ks:ke] & rel).any(), "relations share a pair"
        mask[qs:qe, ks:ke] |= rel
    s = torch.einsum(
        "qhd,khd->hqk", q.float(), k.repeat_interleave(group, dim=1).float()
    ) / math.sqrt(head_dim)
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
    q_head_scale: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    head_dim: int = _HEAD_DIM,
):
    device = "cuda"
    num_head = _NUM_HEAD_KV * group
    q = torch.randn(total_q, num_head, head_dim, device=device, dtype=dtype)
    if q_head_scale:
        # Distinct per-head logit scales expose a head mix-up in reductions.
        scale = 1.0 + 0.25 * torch.arange(num_head, device=device)
        q = (q.float() * scale[None, :, None]).to(dtype)
    k, v = (
        torch.randn(total_k, _NUM_HEAD_KV, head_dim, device=device, dtype=dtype)
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
        max_seqlen_q=max(e - s for s, e in q_ranges),
        max_seqlen_k=max(e - s for s, e in k_ranges),
    )


class TestFfaPackGqaFwd(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the range atomic fwd requires SM100/SM110")
        torch.manual_seed(42)

    def _log_ran(self, test_case: str) -> None:
        print(f"ran {test_case}", flush=True)

    def _packed_fwd(self, q, k, v, relations: Relations, **kwargs):
        """Packed atomic fwd that also returns the kernel object it built."""
        with _record_fwd_kernels() as built:
            out, lse = _flex_flash_attn_fwd(
                q,
                k,
                v,
                **_range_args(relations),
                disable_fwd_atomic_reduction=False,
                **kwargs,
            )
        self.assertEqual(len(built), 1)
        return out, lse, built[0]

    # ─────────────────────────────────────────────────────────────────────
    # q_stage (the prerequisite fix)
    # ─────────────────────────────────────────────────────────────────────

    @parameterize(
        "case",
        [
            # (atomic, len_q, out_dtype, q_stage, 2-CTA, fp32-O K/V borrowing)
            # 128 unpacked rows fit one stage; the G-scaled count (512) would
            # pick two stages with the second one empty.
            (True, 128, torch.float32, 1, False, True),
            (True, 129, torch.float32, 2, False, True),
            # The direct-store path keeps the G-scaled count, so 512 rows
            # still select two stages and the 2-CTA kernel.
            (False, 128, torch.bfloat16, 2, True, False),
        ],
    )
    def test_unpacked_gqa_q_stage_counts_unpacked_rows(self, case):
        """The unpacked GQA range kernel's q_stage and the configs derived
        from it; without sm_margin a range launch is persistent iff fp32 O
        borrows K/V smem."""
        if get_device_arch()[1] != 10:
            self.skipTest("the q_stage rule is SM100-specific")
        atomic, len_q, out_dtype, q_stage, two_cta, borrow_kv = case
        device, dtype, head_dim = "cuda", torch.bfloat16, 128
        num_head, num_head_kv, num_rel, len_k = 16, 4, 8, 256
        q_ranges = [[i * len_q, (i + 1) * len_q] for i in range(num_rel)]
        k_ranges = [[i * len_k, (i + 1) * len_k] for i in range(num_rel)]
        q = torch.randn(num_rel * len_q, num_head, head_dim, device=device, dtype=dtype)
        k, v = (
            torch.randn(
                num_rel * len_k, num_head_kv, head_dim, device=device, dtype=dtype
            )
            for _ in range(2)
        )

        with _record_fwd_kernels() as built:
            out, _ = _flex_flash_attn_fwd(
                q,
                k,
                v,
                q_ranges=torch.tensor(q_ranges, dtype=torch.int32, device=device),
                k_ranges=torch.tensor(k_ranges, dtype=torch.int32, device=device),
                mask_types=MT_MAP.full,
                max_seqlen_q=len_q,
                max_seqlen_k=len_k,
                pack_gqa=False,
                disable_fwd_atomic_reduction=not atomic,
                out_dtype=out_dtype,
            )
        self.assertEqual(len(built), 1)
        kernel = built[0]
        self.assertFalse(kernel.pack_gqa)
        self.assertEqual(kernel.disable_fwd_atomic_reduction, not atomic)
        self.assertEqual(kernel.q_stage, q_stage)
        self.assertEqual(kernel.use_2cta_instrs, two_cta)
        self.assertEqual(kernel.sO_borrow_kv, borrow_kv)
        self.assertEqual(kernel.is_persistent, borrow_kv)

        mask = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_ranges),
            k_ranges=AttnRanges.from_ranges(k_ranges),
            attn_type_map=[MT_MAP.full] * num_rel,
            total_seqlen_q=q.shape[0],
            total_seqlen_k=k.shape[0],
            device=device,
        )
        out_ref, _ = ref_attn_func(
            q=q, k=k, v=v, mask=mask, layout="thd", high_precision=True
        )
        torch.testing.assert_close(out.float(), out_ref.float(), atol=2e-2, rtol=2e-2)

    # ─────────────────────────────────────────────────────────────────────
    # Numerics on the GPU
    # ─────────────────────────────────────────────────────────────────────

    @parameterize("group", [2, 4, 8, 16])
    def test_packed_stage_boundaries_match_reference(self, group):
        """q lengths around the stage tile width T = tile_m / G, over shifted
        unaligned q ranges covered by up to three relations with all four
        mask types: packed matches the reference and the unpacked kernel."""
        t = _TILE_M // group
        mask_types = [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal]
        # Only SM10x runs two Q stages; elsewhere the long case still spans
        # several work tiles of one stage.
        two_stages = get_device_arch()[1] == 10
        for len_qs, q_stage in (
            # Up to one stage tile of packed rows: a single stage.
            ([1, t - 1, t] if t > 1 else [1], 1),
            # Second stage, second work tile and a long unaligned range.
            ([t + 1, 2 * t + 1, 1000], 2 if two_stages else 1),
        ):
            relations, total_q, total_k = _overlapping_relations(
                len_qs, coverage=3, mask_types=mask_types
            )
            q, k, v = _qkv(total_q, total_k, group)
            out, lse, kernel = self._packed_fwd(
                q, k, v, relations, pack_gqa=True, out_dtype=torch.float32
            )
            self.assertTrue(kernel.pack_gqa)
            self.assertEqual(kernel.q_stage, q_stage)
            out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
            torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
            torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

            out_unpacked, lse_unpacked = _flex_flash_attn_fwd(
                q,
                k,
                v,
                **_range_args(relations),
                pack_gqa=False,
                disable_fwd_atomic_reduction=False,
                out_dtype=torch.float32,
            )
            torch.testing.assert_close(out, out_unpacked, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(lse, lse_unpacked, atol=1e-4, rtol=1e-4)
            self._log_ran(f"stage boundaries G={group} len_qs={len_qs}")

    def test_packed_bf16_out_matches_reference(self):
        relations, total_q, total_k = _overlapping_relations(
            [9, 33, 300], coverage=2, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group=4)
        out, lse, kernel = self._packed_fwd(
            q, k, v, relations, pack_gqa=True, out_dtype=torch.bfloat16
        )
        self.assertTrue(kernel.pack_gqa)
        self.assertEqual(out.dtype, torch.bfloat16)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out.float(), out_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

    @parameterize("case", [(torch.float16, 128), (torch.bfloat16, 64)])
    def test_packed_other_dtype_and_head_dim(self, case):
        """(input dtype, head_dim) besides the bf16 / d128 used elsewhere."""
        dtype, head_dim = case
        relations, total_q, total_k = _overlapping_relations(
            [7, 40, 260], coverage=2, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, 4, dtype=dtype, head_dim=head_dim)
        out, lse, kernel = self._packed_fwd(
            q, k, v, relations, pack_gqa=True, out_dtype=torch.float32
        )
        self.assertTrue(kernel.pack_gqa)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

    def test_packed_relation_without_any_pair(self):
        """A bi_causal relation with len_k < len_q has no attended pair for
        any of its (valid) q rows. Merged with a full relation over the same
        q range it changes nothing, in either relation order; alone it leaves
        O = 0 and LSE = max_logits = -inf, or LSE = lse_sink with a sink."""
        group, len_q = 4, 129
        full = ([3, 3 + len_q], [0, len_q], MT_MAP.full)
        empty = ([3, 3 + len_q], [len_q, len_q + 17], MT_MAP.bi_causal)
        total_q, total_k = 3 + len_q + 5, len_q + 17
        q, k, v = _qkv(total_q, total_k, group)
        num_head = _NUM_HEAD_KV * group

        out_ref, lse_ref, _ = _ref_fwd(q, k, v, [full])
        for relations in ([full, empty], [empty, full]):
            out, lse, kernel = self._packed_fwd(
                q, k, v, relations, pack_gqa=True, out_dtype=torch.float32
            )
            self.assertTrue(kernel.pack_gqa)
            torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
            torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

        max_logits = torch.full((num_head,), -math.inf, device="cuda")
        out, lse, _ = self._packed_fwd(
            q, k, v, [empty], pack_gqa=True, max_logits=max_logits
        )
        self.assertTrue(torch.all(out == 0))
        self.assertTrue(torch.all(lse == -math.inf))
        self.assertTrue(torch.all(max_logits == -math.inf))

        sink = torch.randn(2, num_head, device="cuda", dtype=torch.float32)
        out, lse, _ = self._packed_fwd(q, k, v, [empty], pack_gqa=True, sink=sink)
        self.assertTrue(torch.all(out == 0))
        torch.testing.assert_close(
            lse, torch.logsumexp(sink, dim=0).expand(total_q, -1)
        )

    def test_packed_lock_contention_under_few_resident_ctas(self):
        """16 relations write the same unaligned q range, spanning two lock
        blocks per stage tile, from a persistent grid of two CTAs that each
        loop over many tiles."""
        group, len_q = 4, 300
        relations: Relations = [
            ([5, 5 + len_q], [i * 200, (i + 1) * 200], MT_MAP.full) for i in range(16)
        ]
        q, k, v = _qkv(5 + len_q + 7, 16 * 200, group)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        num_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        for _ in range(10):
            out, lse, kernel = self._packed_fwd(
                q,
                k,
                v,
                relations,
                pack_gqa=True,
                out_dtype=torch.float32,
                sm_margin=num_sm - 2,
            )
            self.assertTrue(kernel.pack_gqa and kernel.is_persistent)
            # Two resident CTAs, each looping over at least three work tiles.
            work_rows = kernel.q_stage * _TILE_M
            num_tiles = (
                len(relations) * _NUM_HEAD_KV * math.ceil(len_q * group / work_rows)
            )
            self.assertGreaterEqual(num_tiles, 3 * 2)
            torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
            torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

    @parameterize("sink_call", [None, 0, 1, 2])
    def test_packed_caller_buffer_accumulation_with_one_sink(self, sink_call):
        """Three packed calls merge into one fp32 O/LSE state whose O starts
        as NaN; the sink, if any, goes to exactly one call. The first call
        leaves rows empty that later calls cover."""
        group = 8
        calls: list[Relations] = [
            [([0, 40], [0, 300], MT_MAP.full)],
            [([20, 150], [300, 500], MT_MAP.causal)],
            [([100, 230], [500, 800], MT_MAP.full), ([3, 9], [800, 820], MT_MAP.full)],
        ]
        total_q, total_k = 260, 820
        q, k, v = _qkv(total_q, total_k, group)
        num_head = _NUM_HEAD_KV * group
        sink = torch.randn(2, num_head, device="cuda", dtype=torch.float32)
        out = torch.full(
            (total_q, num_head, _HEAD_DIM), math.nan, device="cuda", dtype=torch.float32
        )
        lse = torch.full((total_q, num_head), -math.inf, device="cuda")
        for i, relations in enumerate(calls):
            self._packed_fwd(
                q,
                k,
                v,
                relations,
                out=out,
                lse=lse,
                pack_gqa=True,
                sink=sink if i == sink_call else None,
            )
        out_ref, lse_ref, _ = _ref_fwd(
            q,
            k,
            v,
            [r for relations in calls for r in relations],
            sink if sink_call is not None else None,
        )
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        finite = torch.isfinite(lse_ref)
        self.assertTrue(torch.equal(finite, torch.isfinite(lse)))
        torch.testing.assert_close(lse[finite], lse_ref[finite], atol=1e-3, rtol=1e-3)

    def test_packed_max_logits_reduce_by_physical_q_head(self):
        group = 8
        relations, total_q, total_k = _overlapping_relations(
            [5, 40, 260], coverage=2, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group, q_head_scale=True)
        num_head = _NUM_HEAD_KV * group
        max_logits = torch.full((num_head,), -math.inf, device="cuda")
        self._packed_fwd(q, k, v, relations, pack_gqa=True, max_logits=max_logits)
        _, _, max_logits_ref = _ref_fwd(q, k, v, relations)
        self.assertTrue(torch.isfinite(max_logits_ref).all())
        torch.testing.assert_close(max_logits, max_logits_ref, atol=1e-3, rtol=1e-4)

    def test_packed_cuda_graph_replay(self):
        """A captured packed fwd replays on new data: the lock array and the
        DYNAMIC tile counter are re-zeroed inside the graph."""
        group = 4
        relations, total_q, total_k = _overlapping_relations(
            [17, 70], coverage=2, mask_types=[MT_MAP.full]
        )
        q, k, v = _qkv(total_q, total_k, group)
        num_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        kwargs = dict(
            **_range_args(relations),
            pack_gqa=True,
            disable_fwd_atomic_reduction=False,
            out_dtype=torch.float32,
            sm_margin=num_sm - 4,
        )
        _flex_flash_attn_fwd(q, k, v, **kwargs)  # compile outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = _flex_flash_attn_fwd(q, k, v, **kwargs)
        for _ in range(3):
            new_q, new_k, new_v = _qkv(total_q, total_k, group)
            q.copy_(new_q)
            k.copy_(new_k)
            v.copy_(new_v)
            graph.replay()
            out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
            torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
            torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

    def test_packed_bwd_consumes_packed_fwd_lse(self):
        """The bwd (never packed) reads the physical LSE the packed fwd
        wrote: gradients match those of the unpacked fwd."""
        group = 4
        relations, total_q, total_k = _overlapping_relations(
            [33, 150], coverage=2, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group)
        do = torch.randn_like(q)
        grads = {}
        for pack in (False, True):
            qkv = [t.detach().clone().requires_grad_() for t in (q, k, v)]
            out, _ = flex_flash_attn_func(
                *qkv,
                **_range_args(relations),
                pack_gqa=pack,
                disable_fwd_atomic_reduction=False,
            )
            out.backward(do)
            grads[pack] = [t.grad.float() for t in qkv]
        for name, packed, unpacked in zip("qkv", grads[True], grads[False]):
            torch.testing.assert_close(
                packed, unpacked, atol=2e-2, rtol=2e-2, msg=lambda m: f"d{name}: {m}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Defaults, rejection and the dist entry
    # ─────────────────────────────────────────────────────────────────────

    def test_atomic_fwd_defaults_to_unpacked(self):
        relations, total_q, total_k = _overlapping_relations(
            [20], coverage=2, mask_types=[MT_MAP.full]
        )
        q, k, v = _qkv(total_q, total_k, group=4)
        _, _, kernel = self._packed_fwd(q, k, v, relations)
        self.assertFalse(kernel.pack_gqa)

    def test_packed_atomic_fwd_rejects_group_not_dividing_tile_m(self):
        relations, total_q, total_k = _overlapping_relations(
            [20], coverage=2, mask_types=[MT_MAP.full]
        )
        q, k, v = _qkv(total_q, total_k, group=3)
        with self.assertRaises(NotImplementedError):
            _flex_flash_attn_fwd(
                q,
                k,
                v,
                **_range_args(relations),
                pack_gqa=True,
                disable_fwd_atomic_reduction=False,
            )

    def test_cutedsl_fwd_forwards_pack_gqa(self):
        """The dist wrapper's explicit ``pack_gqa=True`` reaches the kernel
        and matches its default unpacked atomic forward."""
        group = 4
        relations, total_q, total_k = _overlapping_relations(
            [12, 140], coverage=2, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group)
        attn_arg = AttnArg(
            q_ranges=AttnRanges.from_ranges([r[0] for r in relations]),
            k_ranges=AttnRanges.from_ranges([r[1] for r in relations]),
            attn_type_map=[r[2] for r in relations],
        )
        results = {}
        for pack in (None, True):
            with _record_fwd_kernels() as built:
                results[pack] = cutedsl_fwd(
                    q, k, v, attn_arg, softmax_scale=None, softcap=0.0, pack_gqa=pack
                )
            self.assertEqual([b.pack_gqa for b in built], [pack is True])
        for packed, unpacked in zip(results[True][:2], results[None][:2]):
            torch.testing.assert_close(packed, unpacked, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
