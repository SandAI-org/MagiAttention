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
from typing import Callable, Sequence
from unittest import TestCase, mock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import flex_flash_attn_func, range_chain
from magi_attention.kernel.cutedsl.block_info import BlockInfo
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.range_merge import (
    plan_range_merge,
    plan_range_merge_bwd,
)
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import record_bwd_kernels, record_fwd_kernels
from tests.test_kernel.cutedsl.range_deterministic_probes import (
    DelayPoint,
    delay_injection,
    odd_lane_conflict_records,
    record_visits,
)

_HEAD_DIM = 128
_DET_HEAD_DIMS = [64, 128]
_NUM_HEAD_KV = 2
_TOL = 1e-2

Relations = list[tuple[list[int], list[int], int]]


def _assert_bitwise_equal(
    got: Sequence[torch.Tensor], want: Sequence[torch.Tensor]
) -> None:
    equal = [torch.equal(g, w) for g, w in zip(got, want, strict=True)]
    assert all(equal), f"bitwise equal per tensor: {equal}"


def _assert_grads_close(
    grads: Sequence[torch.Tensor], want: Sequence[torch.Tensor]
) -> None:
    dq, dk, dv = grads
    dq_want, dk_want, dv_want = want
    torch.testing.assert_close(
        dq.float(), dq_want.float(), atol=3e-2, rtol=3e-2, msg=lambda m: f"dq: {m}"
    )
    torch.testing.assert_close(
        dk.float(), dk_want.float(), atol=3e-2, rtol=3e-2, msg=lambda m: f"dk: {m}"
    )
    torch.testing.assert_close(
        dv.float(), dv_want.float(), atol=3e-2, rtol=3e-2, msg=lambda m: f"dv: {m}"
    )


def _replay(
    graph: torch.cuda.CUDAGraph,
    static_inputs: Sequence[torch.Tensor],
    inputs: Sequence[torch.Tensor],
) -> None:
    for dst, src in zip(static_inputs, inputs, strict=True):
        dst.copy_(src)
    graph.replay()


_torch_empty_like = torch.empty_like


def _nan_empty_like(*args, **kwargs) -> torch.Tensor:
    t = _torch_empty_like(*args, **kwargs)
    return t.fill_(float("nan")) if t.is_floating_point() else t


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
    s = s.masked_fill(~mask, -math.inf)
    max_logits = s.amax(dim=(1, 2))
    if sink is not None:
        sink_s = sink.float().t()[:, None, :].expand(num_head, total_q, -1)
        s = torch.cat([s, sink_s], dim=-1)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None]).nan_to_num(0.0)[..., : k.shape[0]]
    out = torch.einsum("hqk,khd->qhd", p, v.repeat_interleave(group, dim=1).float())
    return out, lse.t().contiguous(), max_logits


def _ref_grads(q, k, v, do, relations: Relations) -> list[torch.Tensor]:
    q32, k32, v32 = (t.detach().float().requires_grad_() for t in (q, k, v))
    out_ref, _, _ = _ref_fwd(q32, k32, v32, relations)
    out_ref.backward(do.float())
    return [q32.grad, k32.grad, v32.grad]


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


def _merged_q_relations() -> tuple[Relations, int, int]:
    relations: Relations = [
        ([131, 600], [100, 400], MT_MAP.full),
        ([0, 131], [0, 300], MT_MAP.full),
        ([601, 1000], [900, 1200], MT_MAP.causal),
        ([0, 131], [300, 500], MT_MAP.causal),
        ([600, 601], [0, 50], MT_MAP.full),
        ([131, 600], [600, 900], MT_MAP.inv_causal),
        ([0, 131], [530, 530], MT_MAP.full),
        ([0, 131], [500, 530], MT_MAP.full),
    ]
    return relations, 1000, 1200


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


class TestFFARangeDeterministicHost(TestCase):
    def test_tile_counter_bound_counts_overlapping_ranges(self):
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

    def test_debug_mode_rejects_max_seqlen_k_below_the_longest_k_range(self):
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("deterministic q/k ranges require SM100/SM110")
        relations: Relations = [
            ([0, 100], [0, 300], MT_MAP.full),
            ([50, 200], [300, 400], MT_MAP.causal),
        ]
        q, k, v = _qkv(200, 400, group=1)
        lse = torch.zeros(200, q.shape[1], device="cuda")
        args = _range_args(relations)
        self.assertEqual(args["max_seqlen_k"], 300)

        class KernelReached(Exception):
            pass

        def bwd(max_seqlen_k: int) -> None:
            _flex_flash_attn_bwd(
                q,
                k,
                v,
                q,
                lse,
                q,
                **{**args, "max_seqlen_k": max_seqlen_k},
                disable_bwd_dkv_atomic_reduction=False,
                deterministic=True,
            )

        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE": "1"}
        ), mock.patch.object(
            _flex_flash_attn_bwd, "compile_cache", JITCache()
        ), mock.patch.object(
            FFABwdSm100, "__init__", side_effect=KernelReached
        ):
            with self.assertRaisesRegex(
                ValueError, r"max_seqlen_k=299 < longest k range 300"
            ):
                bwd(299)
            with self.assertRaises(KernelReached):
                bwd(300)


class TestFFARangeDeterministicFwd(TestCase):
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
        first = run(0)
        _assert_bitwise_equal(run(1), first)
        _assert_bitwise_equal(run(self.num_sm - 7), first)
        _assert_bitwise_equal(run(self.num_sm - 1), first)
        _assert_bitwise_equal(run(0), first)
        return first

    @parameterize(
        "case",
        [
            (torch.float32, 1),
            (torch.float32, 2),
            (torch.bfloat16, 2),
        ],
    )
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_fwd_bitwise_across_runs_and_sm_margin(self, case, head_dim):
        out_dtype, group = case
        relations, total_q, total_k = _overlapping_relations(
            [129, 300, 1000, 77], coverage=4, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group, head_dim=head_dim)
        out, lse = self._assert_bitwise_across_sm_margin(
            lambda m: self._det_fwd(
                q, k, v, relations, out_dtype=out_dtype, sm_margin=m
            )
        )
        self.assertEqual(out.dtype, out_dtype)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out.float(), out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
        self._log_ran(f"bitwise {case} hd{head_dim}")

    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_fwd_empty_keyless_and_protection_block_relations(self, head_dim):
        relations: Relations = [
            ([1, 130], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([256, 400], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ]
        q, k, v = _qkv(420, 900, group=2, head_dim=head_dim)
        out, lse = self._assert_bitwise_across_sm_margin(
            lambda m: self._det_fwd(q, k, v, relations, sm_margin=m)
        )
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        finite = torch.isfinite(lse_ref)
        self.assertTrue(torch.equal(finite, torch.isfinite(lse)))
        torch.testing.assert_close(lse[finite], lse_ref[finite], atol=1e-3, rtol=1e-3)
        self._log_ran(f"empty/keyless/protection hd{head_dim}")

    def test_fwd_caller_buffer_accumulation_with_sink(self):
        first_call: Relations = [
            ([0, 140], [0, 300], MT_MAP.full),
            ([70, 260], [300, 500], MT_MAP.causal),
        ]
        second_call: Relations = [
            ([3, 250], [500, 800], MT_MAP.full),
            ([128, 129], [800, 820], MT_MAP.full),
        ]
        q, k, v = _qkv(260, 820, group=2)
        num_head = q.shape[1]
        sink = torch.randn(2, num_head, device="cuda", dtype=torch.float32)

        def run(sm_margin):
            out = torch.zeros(260, num_head, _HEAD_DIM, device="cuda")
            lse = torch.full((260, num_head), -math.inf, device="cuda")
            self._det_fwd(q, k, v, first_call, out=out, lse=lse, sm_margin=sm_margin)
            self._det_fwd(
                q, k, v, second_call, out=out, lse=lse, sink=sink, sm_margin=sm_margin
            )
            return out, lse

        out, lse = self._assert_bitwise_across_sm_margin(run)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, first_call + second_call, sink)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
        self._log_ran("caller buffer + sink")

    def test_fwd_max_logits(self):
        relations, total_q, total_k = _overlapping_relations(
            [200, 45], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group=2)

        def run(sm_margin):
            max_logits = torch.full((q.shape[1],), -math.inf, device="cuda")
            out, lse = self._det_fwd(
                q,
                k,
                v,
                relations,
                max_logits=max_logits,
                sm_margin=sm_margin,
            )
            return out, lse, max_logits

        out, lse, max_logits = self._assert_bitwise_across_sm_margin(run)
        out_ref, lse_ref, max_ref = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(max_logits, max_ref, atol=1e-3, rtol=1e-4)
        self._log_ran("max_logits")

    def test_fwd_launch_is_persistent_without_an_sm_reservation(self):
        relations, total_q, total_k = _overlapping_relations(
            [129, 600], coverage=3, mask_types=[MT_MAP.full]
        )
        q, k, v = _qkv(total_q, total_k, group=2)
        with record_fwd_kernels() as built:
            self._det_fwd(q, k, v, relations, out_dtype=torch.bfloat16, sm_margin=0)
        (kernel,) = built
        self.assertTrue(kernel.deterministic and kernel.is_persistent)
        self.assertEqual(kernel.cluster_shape_mn, (1, 1))
        self._log_ran("persistent without an sm reservation")

    @parameterize(
        "case",
        [
            (torch.float32, 1, _TOL),
            (torch.float32, 2, _TOL),
            (torch.bfloat16, 2, 2 * _TOL),
        ],
    )
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_fwd_range_merge_bitwise_across_runs_and_sm_margin(self, case, head_dim):
        out_dtype, group, tol = case
        relations, total_q, total_k = _merged_q_relations()
        q, k, v = _qkv(total_q, total_k, group, head_dim=head_dim)
        args = _range_args(relations)
        plan = plan_range_merge(args["q_ranges"], args["k_ranges"], args["mask_types"])

        def merge_fwd(sm_margin, range_merge=True):
            return _flex_flash_attn_fwd(
                q,
                k,
                v,
                **args,
                disable_fwd_atomic_reduction=True,
                range_merge=range_merge,
                deterministic=True,
                out_dtype=out_dtype,
                sm_margin=sm_margin,
            )

        with record_fwd_kernels() as built:
            merge_fwd(0)
        (kernel,) = built
        self.assertTrue(kernel.range_merge)
        self.assertFalse(kernel.deterministic)
        with record_fwd_kernels() as reserved_built:
            merge_fwd(self.num_sm - 7)
        (reserved_kernel,) = reserved_built
        self.assertTrue(reserved_kernel.range_merge)
        self.assertFalse(reserved_kernel.deterministic)
        self.assertTrue(reserved_kernel.is_persistent)
        out, lse = self._assert_bitwise_across_sm_margin(merge_fwd)
        _assert_bitwise_equal(merge_fwd(self.num_sm - 7, plan), (out, lse))
        self.assertEqual(out.dtype, out_dtype)
        out_atomic, lse_atomic = self._det_fwd(q, k, v, relations)
        torch.testing.assert_close(out.float(), out_atomic, atol=tol, rtol=tol)
        torch.testing.assert_close(lse, lse_atomic, atol=1e-3, rtol=1e-3)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out.float(), out_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
        self._log_ran(f"range merge fwd {case} hd{head_dim}")

    def _assert_bitwise_packed(self, q, k, v, relations: Relations) -> tuple:
        with record_fwd_kernels() as built:
            result = self._assert_bitwise_across_sm_margin(
                lambda m: self._det_fwd(q, k, v, relations, pack_gqa=True, sm_margin=m)
            )
        kernel = built[0]
        self.assertTrue(kernel.deterministic and kernel.pack_gqa)
        self.assertEqual(kernel.qhead_per_kvhead, q.shape[1] // k.shape[1])
        return result

    @parameterize("group", [2, 4])
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_packed_fwd_bitwise_across_runs_and_sm_margin(self, group, head_dim):
        relations, total_q, total_k = _overlapping_relations(
            [129, 300, 1000, 77], coverage=4, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = _qkv(total_q, total_k, group, head_dim=head_dim)
        out, lse = self._assert_bitwise_packed(q, k, v, relations)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
        out_unpacked, lse_unpacked = self._det_fwd(q, k, v, relations, pack_gqa=False)
        torch.testing.assert_close(out, out_unpacked, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(lse, lse_unpacked, atol=1e-4, rtol=1e-4)
        self._log_ran(f"packed bitwise G{group} hd{head_dim}")

    @parameterize("group", [2, 4])
    def test_packed_fwd_empty_keyless_and_block_boundary_relations(self, group):
        relations: Relations = [
            ([61, 200], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([64, 128], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ]
        q, k, v = _qkv(420, 900, group=group)
        out, lse = self._assert_bitwise_packed(q, k, v, relations)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        finite = torch.isfinite(lse_ref)
        self.assertTrue(torch.equal(finite, torch.isfinite(lse)))
        torch.testing.assert_close(lse[finite], lse_ref[finite], atol=1e-3, rtol=1e-3)
        self._log_ran(f"packed empty/keyless/boundary G{group}")

    def test_packed_fwd_cuda_graph_replay_matches_eager_bitwise(self):
        relations, total_q, total_k = _overlapping_relations(
            [129, 300], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        args = _range_args(relations)

        def fwd(q, k, v):
            return _flex_flash_attn_fwd(
                q,
                k,
                v,
                **args,
                disable_fwd_atomic_reduction=False,
                deterministic=True,
                pack_gqa=True,
            )

        data_0 = _qkv(total_q, total_k, group=4)
        data_1 = _qkv(total_q, total_k, group=4)
        eager_0 = fwd(*data_0)
        eager_1 = fwd(*data_1)
        static = [t.clone() for t in data_0]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fwd(*static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = fwd(*static)
        _replay(graph, static, data_1)
        _assert_bitwise_equal(static_out, eager_1)
        _replay(graph, static, data_0)
        _assert_bitwise_equal(static_out, eager_0)
        _replay(graph, static, data_1)
        _assert_bitwise_equal(static_out, eager_1)
        self._log_ran("packed cuda graph replay")


def _shared_k_relations() -> tuple[Relations, int, int]:
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


def _merged_k_relations() -> tuple[Relations, int, int]:
    relations: Relations = [
        ([600, 700], [300, 557], MT_MAP.causal),
        ([5, 305], [0, 300], MT_MAP.causal),
        ([0, 143], [557, 700], MT_MAP.full),
        ([1, 301], [700, 1000], MT_MAP.causal),
        ([4, 304], [0, 300], MT_MAP.inv_causal),
        ([0, 200], [1000, 1000], MT_MAP.full),
        ([100, 357], [300, 557], MT_MAP.full),
        ([50, 50], [557, 700], MT_MAP.full),
        ([0, 300], [700, 1000], MT_MAP.inv_causal),
        ([400, 450], [0, 300], MT_MAP.full),
        ([250, 600], [557, 700], MT_MAP.inv_causal),
    ]
    return relations, 700, 1000


def _merged_k_relations_heavy_middle() -> tuple[Relations, int, int]:
    relations, total_q, _ = _merged_k_relations()
    relations = [
        (q_range, [580, 700] if k_range == [557, 700] else k_range, mask_type)
        for q_range, k_range, mask_type in relations
    ] + [
        ([q_start, q_end], [300, 557], MT_MAP.full)
        for q_start, q_end in ((0, 50), (50, 100), (357, 480), (480, 600))
    ]
    return relations, total_q, 1100


def _merged_k_relations_2cta_adjacent(
    num_groups: int = 8, pairs_per_group: int = 4
) -> tuple[Relations, int, int]:
    len_k, base_k = 160, 256
    span = 64 * pairs_per_group + 128
    total_q = span * num_groups
    relations: Relations = [([0, total_q], [0, base_k], MT_MAP.full)]
    for g in range(num_groups):
        k_range = [base_k + len_k * g, base_k + len_k * (g + 1)]
        for p in range(pairs_per_group):
            q_start = span * g + 64 * p
            if pairs_per_group == 1:
                mask_type = MT_MAP.full
            elif p == 0:
                mask_type = MT_MAP.inv_causal
            elif p == pairs_per_group - 1:
                mask_type = MT_MAP.causal
            else:
                mask_type = MT_MAP.bi_causal
            relations.append(([q_start, q_start + 128], list(k_range), mask_type))
    return relations, total_q, base_k + len_k * num_groups


_DISJOINT_Q_RELATIONS: dict[bool, tuple[Relations, int]] = {
    True: (
        [
            ([3, 132], [0, 290], MT_MAP.full),
            ([140, 410], [290, 560], MT_MAP.causal),
            ([420, 420], [560, 620], MT_MAP.full),
            ([430, 690], [620, 620], MT_MAP.full),
            ([700, 965], [620, 900], MT_MAP.inv_causal),
            ([970, 1240], [900, 1170], MT_MAP.bi_causal),
        ],
        1180,
    ),
    False: (
        [
            ([3, 132], [10, 300], MT_MAP.full),
            ([140, 410], [150, 430], MT_MAP.causal),
            ([420, 420], [200, 260], MT_MAP.full),
            ([430, 690], [430, 430], MT_MAP.full),
            ([700, 965], [100, 380], MT_MAP.inv_causal),
            ([970, 1240], [250, 520], MT_MAP.bi_causal),
        ],
        550,
    ),
}


class TestFFARangeDeterministicBwd(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("deterministic q/k ranges require SM100/SM110")
        torch.manual_seed(42)
        self.num_sm = torch.cuda.get_device_properties(0).multi_processor_count

    def _log_ran(self, test_case: str) -> None:
        print(f"ran {test_case}", flush=True)

    def _fwd_bwd_inputs(
        self, relations: Relations, total_q, total_k, group, head_dim=_HEAD_DIM
    ):
        q, k, v = _qkv(total_q, total_k, group, head_dim=head_dim)
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
        cluster = 2 if two_cta else 1
        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0" if two_cta else "1"}

        def run(sm_margin: int):
            return self._det_bwd(
                q, k, v, out, lse, do, relations, sm_margin=sm_margin, **kwargs
            )

        with mock.patch.dict("os.environ", env), record_bwd_kernels() as built:
            first = run(0)
            _assert_bitwise_equal(run(1), first)
            _assert_bitwise_equal(run(self.num_sm - 4 * cluster), first)
            _assert_bitwise_equal(run(self.num_sm - cluster), first)
            _assert_bitwise_equal(run(0), first)
        kernel = built[0]
        self.assertEqual(kernel.use_2cta_instrs, two_cta)
        self.assertTrue(kernel.is_persistent)
        _assert_grads_close(first, _ref_grads(q, k, v, do, relations))
        return first

    @parameterize(
        "case",
        [
            (1, False, True, 128),
            (1, False, False, 128),
            (2, True, True, 128),
            (2, True, False, 128),
            (1, False, False, 64),
            (2, True, False, 64),
        ],
    )
    def test_bwd_dq_bitwise_across_sm_margin(self, case):
        group, cat_gqa, two_cta, head_dim = case
        relations, total_q, total_k = _overlapping_relations(
            [129, 300, 1000, 77], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v, out, lse, do = self._fwd_bwd_inputs(
            relations, total_q, total_k, group, head_dim
        )
        self._assert_grads_bitwise_across_sm_margin(
            q, k, v, out, lse, do, relations, two_cta, cat_gqa=cat_gqa
        )
        self._log_ran(f"bwd dq bitwise {case}")

    @parameterize(
        "case",
        [
            (1, False, True, 128),
            (2, False, True, 128),
            (2, False, False, 128),
            (2, True, True, 128),
            (1, False, False, 64),
            (2, False, False, 64),
            (2, True, False, 64),
        ],
    )
    def test_bwd_reduced_dkv_bitwise_across_sm_margin(self, case):
        group, cat_gqa, two_cta, head_dim = case
        relations, total_q, total_k = _shared_k_relations()
        q, k, v, out, lse, do = self._fwd_bwd_inputs(
            relations, total_q, total_k, group, head_dim
        )
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

    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_bwd_empty_and_keyless_relations(self, head_dim):
        relations: Relations = [
            ([1, 130], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([127, 400], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ]
        q, k, v, out, lse, do = self._fwd_bwd_inputs(
            relations, 420, 900, group=1, head_dim=head_dim
        )

        def run(sm_margin: int):
            return self._det_bwd(q, k, v, out, lse, do, relations, sm_margin=sm_margin)

        first = run(0)
        _assert_bitwise_equal(run(self.num_sm - 2), first)
        _assert_bitwise_equal(run(0), first)
        _assert_grads_close(first, _ref_grads(q, k, v, do, relations))
        self._log_ran(f"bwd empty/keyless hd{head_dim}")

    def test_bwd_reads_no_range_back_outside_debug_mode(self):
        relations, total_q, total_k = _overlapping_relations(
            [129, 300], coverage=2, mask_types=[MT_MAP.full]
        )
        q, k, v, out, lse, do = self._fwd_bwd_inputs(relations, total_q, total_k, 1)
        disjoint_rels: Relations = [
            ([3, 140], [0, 300], MT_MAP.full),
            ([145, 420], [300, 620], MT_MAP.causal),
        ]
        qd, kd, vd, outd, lsed, dod = self._fwd_bwd_inputs(disjoint_rels, 430, 620, 1)
        merge_rels, mq_tot, mk_tot = _merged_k_relations()
        qm, km, vm, outm, lsem, dom = self._fwd_bwd_inputs(
            merge_rels, mq_tot, mk_tot, 1
        )
        # Built up front: copying the host range lists to the device syncs.
        args = _range_args(relations)
        disjoint_args = _range_args(disjoint_rels)
        merge_args = _range_args(merge_rels)
        merge_plan = plan_range_merge_bwd(
            merge_args["q_ranges"], merge_args["k_ranges"], merge_args["mask_types"]
        )

        def bwd_chained() -> None:
            with mock.patch.dict(
                "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE": "0"}
            ):
                _flex_flash_attn_bwd(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    do,
                    **args,
                    disable_bwd_dkv_atomic_reduction=False,
                    deterministic=True,
                )

        def bwd_disjoint_dense() -> None:
            with mock.patch.dict(
                "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE": "0"}
            ):
                _flex_flash_attn_bwd(
                    qd,
                    kd,
                    vd,
                    outd,
                    lsed,
                    dod,
                    **disjoint_args,
                    disable_fwd_atomic_reduction=True,
                    disable_bwd_dkv_atomic_reduction=True,
                    deterministic=True,
                )

        def bwd_merged() -> None:
            with mock.patch.dict(
                "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE": "0"}
            ):
                _flex_flash_attn_bwd(
                    qm,
                    km,
                    vm,
                    outm,
                    lsem,
                    dom,
                    **merge_args,
                    range_merge=merge_plan,
                    disable_bwd_dkv_atomic_reduction=True,
                    deterministic=True,
                )

        bwd_chained()
        bwd_disjoint_dense()
        bwd_merged()
        torch.cuda.synchronize()
        prev_sync_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            bwd_chained()
            bwd_disjoint_dense()
            bwd_merged()
        finally:
            torch.cuda.set_sync_debug_mode(prev_sync_debug_mode)
        self._log_ran("bwd no sync outside debug mode")

    @parameterize(
        "case",
        [
            (1, False, 64, False),
            (2, True, 64, False),
            (1, False, 128, False),
            (2, True, 128, False),
            (1, False, 128, True),
            (2, True, 128, True),
        ],
    )
    def test_bwd_range_merge_bitwise_across_sm_margin(self, case):
        self._check_bwd_range_merge(case, *_merged_k_relations())

    @parameterize("case", [(1, False, 128, False), (2, True, 128, True)])
    def test_bwd_range_merge_heavy_middle_group_and_k_rows_in_no_group(self, case):
        relations, total_q, total_k = _merged_k_relations_heavy_middle()
        dq, dk, dv = self._check_bwd_range_merge(
            case, relations, total_q, total_k, empty_like=_nan_empty_like
        )
        self.assertTrue(torch.equal(dk[557:580], torch.zeros_like(dk[557:580])))
        self.assertTrue(torch.equal(dk[1000:1100], torch.zeros_like(dk[1000:1100])))
        self.assertTrue(torch.equal(dv[557:580], torch.zeros_like(dv[557:580])))
        self.assertTrue(torch.equal(dv[1000:1100], torch.zeros_like(dv[1000:1100])))

    def _check_bwd_range_merge(
        self,
        case,
        relations,
        total_q,
        total_k,
        empty_like: Callable[..., torch.Tensor] = torch.empty_like,
    ):
        group, cat_gqa, head_dim, two_cta = case
        cluster = 2 if two_cta else 1
        q, k, v, out, lse, do = self._fwd_bwd_inputs(
            relations, total_q, total_k, group, head_dim
        )
        args = _range_args(relations)
        plan = plan_range_merge_bwd(
            args["q_ranges"], args["k_ranges"], args["mask_types"]
        )

        def merge_bwd(sm_margin, range_merge=True):
            with mock.patch("torch.empty_like", empty_like):
                return self._det_bwd(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    do,
                    relations,
                    range_merge=range_merge,
                    cat_gqa=cat_gqa,
                    sm_margin=sm_margin,
                )

        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0" if two_cta else "1"}
        with mock.patch.dict("os.environ", env):
            with record_bwd_kernels() as built:
                first = merge_bwd(0)
            (kernel,) = built
            self.assertTrue(kernel.range_merge and kernel.deterministic)
            self.assertTrue(kernel.is_persistent)
            self.assertEqual(kernel.use_2cta_instrs, two_cta)
            self.assertEqual(kernel.dq_pair_handoff, two_cta)
            self.assertFalse(kernel.dKV_postprocess)
            _assert_bitwise_equal(merge_bwd(1), first)
            _assert_bitwise_equal(merge_bwd(self.num_sm - 4 * cluster), first)
            _assert_bitwise_equal(merge_bwd(self.num_sm - cluster), first)
            _assert_bitwise_equal(merge_bwd(0), first)
            _assert_bitwise_equal(merge_bwd(self.num_sm - 4 * cluster, plan), first)
            unmerged = self._det_bwd(
                q, k, v, out, lse, do, relations, direct_dkv=False, cat_gqa=cat_gqa
            )
        _assert_grads_close(first, unmerged)
        _assert_grads_close(first, _ref_grads(q, k, v, do, relations))
        self._log_ran(f"bwd range merge {case}")
        return first

    @parameterize("cat_gqa", [False, True])
    def test_bwd_range_merge_2cta_adjacent_pair_conflicts_bitwise(self, cat_gqa):
        relations, total_q, total_k = _merged_k_relations_2cta_adjacent()
        group = 2 if cat_gqa else 1
        q, k, v, out, lse, do = self._fwd_bwd_inputs(relations, total_q, total_k, group)

        def merge_bwd(sm_margin):
            return self._det_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                relations,
                range_merge=True,
                cat_gqa=cat_gqa,
                sm_margin=sm_margin,
            )

        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        with mock.patch.dict("os.environ", env):
            with record_bwd_kernels() as built:
                first = merge_bwd(0)
            (kernel,) = built
            self.assertTrue(kernel.use_2cta_instrs and kernel.dq_pair_handoff)
            with delay_injection(DelayPoint.CHAIN_WAIT, seed=7, max_ns=20_000):
                _assert_bitwise_equal(merge_bwd(0), first)
                _assert_bitwise_equal(merge_bwd(0), first)
                _assert_bitwise_equal(merge_bwd(0), first)
                _assert_bitwise_equal(merge_bwd(0), first)
                _assert_bitwise_equal(merge_bwd(0), first)
                _assert_bitwise_equal(merge_bwd(self.num_sm - 2), first)
                _assert_bitwise_equal(merge_bwd(self.num_sm - 2), first)
                _assert_bitwise_equal(merge_bwd(self.num_sm - 2), first)
                _assert_bitwise_equal(merge_bwd(self.num_sm - 2), first)
                _assert_bitwise_equal(merge_bwd(self.num_sm - 2), first)
        _assert_grads_close(first, _ref_grads(q, k, v, do, relations))
        self._log_ran(f"bwd range merge 2-CTA adjacent conflicts {cat_gqa=}")

    @parameterize(
        "case",
        [
            (64, False),
            (128, True),
        ],
    )
    def test_bwd_range_merge_cuda_graph_replay_matches_eager_bitwise(self, case):
        head_dim, two_cta = case
        relations, total_q, total_k = _merged_k_relations()
        args = _range_args(relations)
        plan = plan_range_merge_bwd(
            args["q_ranges"], args["k_ranges"], args["mask_types"]
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
                range_merge=plan,
                disable_bwd_dkv_atomic_reduction=True,
                cat_gqa=True,
                deterministic=True,
            )[:3]

        data_0 = self._fwd_bwd_inputs(relations, total_q, total_k, 2, head_dim=head_dim)
        data_1 = self._fwd_bwd_inputs(relations, total_q, total_k, 2, head_dim=head_dim)
        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0" if two_cta else "1"}
        with mock.patch.dict("os.environ", env), record_bwd_kernels() as built:
            eager_0 = bwd(*data_0)
            eager_1 = bwd(*data_1)
            static = [t.clone() for t in data_0]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                bwd(*static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_grads = bwd(*static)
            # Replay inside record_bwd_kernels so the swapped cache keeps the module.
            (kernel,) = built
            self.assertEqual(kernel.use_2cta_instrs, two_cta)
            self.assertTrue(kernel.range_merge and kernel.deterministic)
            _replay(graph, static, data_1)
            _assert_bitwise_equal(static_grads, eager_1)
            _replay(graph, static, data_0)
            _assert_bitwise_equal(static_grads, eager_0)
            _replay(graph, static, data_1)
            _assert_bitwise_equal(static_grads, eager_1)
        self._log_ran(f"bwd range merge cuda graph replay {case}")

    def test_public_api_fwd_bwd_bitwise_with_sink(self):
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
                deterministic=True,
                sm_margin=sm_margin,
            )
            out.backward(do)
            return [out.detach(), meta.lse, *(t.grad for t in leaves)]

        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ):
            first = run(0)
            _assert_bitwise_equal(run(1), first)
            _assert_bitwise_equal(run(self.num_sm - 2), first)
            _assert_bitwise_equal(run(0), first)
        self._log_ran("public api")

    @parameterize(
        "case",
        [
            (False, False, 64),
            (False, False, 128),
            (False, True, 128),
            (True, False, 128),
            (True, True, 128),
        ],
    )
    def test_bwd_straddling_masked_blocks_all_mask_types(self, case):
        range_merge, two_cta, head_dim = case
        cluster = 2 if two_cta else 1
        relations: Relations = [
            ([0, 390], [10, 270], MT_MAP.bi_causal),
            ([0, 390], [270, 530], MT_MAP.bi_causal),
            ([400, 790], [270, 530], MT_MAP.bi_causal),
            ([400, 520], [530, 600], MT_MAP.bi_causal),
            ([7, 267], [600, 860], MT_MAP.causal),
            ([6, 266], [600, 860], MT_MAP.inv_causal),
            ([19, 535], [860, 1140], MT_MAP.full),
        ]
        total_q, total_k = 800, 1150
        q, k, v, out, lse, do = self._fwd_bwd_inputs(
            relations, total_q, total_k, group=2, head_dim=head_dim
        )
        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0" if two_cta else "1"}

        def run(sm_margin: int):
            return self._det_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                relations,
                direct_dkv=range_merge,
                range_merge=range_merge,
                cat_gqa=True,
                sm_margin=sm_margin,
            )

        with mock.patch.dict("os.environ", env):
            first = run(0)
            _assert_bitwise_equal(run(1), first)
            _assert_bitwise_equal(run(self.num_sm - 2 * cluster), first)
            _assert_bitwise_equal(run(self.num_sm - cluster), first)
            with delay_injection(DelayPoint.CHAIN_WAIT, seed=11, max_ns=15_000):
                _assert_bitwise_equal(run(self.num_sm - 2 * cluster), first)
        _assert_grads_close(first, _ref_grads(q, k, v, do, relations))
        self._log_ran(f"bwd straddling masked {case}")

    @parameterize(
        "case",
        [
            (1, False, True, 128, True),
            (1, False, False, 128, False),
            (2, True, True, 128, False),
            (2, False, True, 128, False),
            (1, False, False, 64, True),
            (2, True, False, 64, False),
        ],
    )
    def test_bwd_disjoint_q_dense_dqacc_bitwise_across_sm_margin(self, case):
        group, cat_gqa, two_cta, head_dim, direct_dkv = case
        cluster = 2 if two_cta else 1
        relations, total_k = _DISJOINT_Q_RELATIONS[direct_dkv]
        total_q = 1260
        q, k, v = _qkv(total_q, total_k, group=group, head_dim=head_dim)
        args = _range_args(relations)
        out, lse = _flex_flash_attn_fwd(
            q, k, v, **args, disable_fwd_atomic_reduction=True, deterministic=True
        )
        out, do = out.to(q.dtype), torch.randn_like(q)
        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0" if two_cta else "1"}

        def run(sm_margin: int, dq_init: torch.Tensor | None = None):
            return _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                dq=dq_init,
                **args,
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=direct_dkv,
                cat_gqa=cat_gqa,
                deterministic=True,
                sm_margin=sm_margin,
            )[:3]

        with mock.patch.dict("os.environ", env):
            with record_bwd_kernels() as built:
                first = run(0)
            (kernel,) = built
            self.assertTrue(kernel.use_dense_dqacc_for_ranges)
            self.assertTrue(kernel.range_dq_semaphore)
            self.assertFalse(kernel.range_dq_chain)
            self.assertFalse(kernel.dQ_rowmajor_accum)
            _assert_bitwise_equal(run(1), first)
            _assert_bitwise_equal(run(self.num_sm - 2 * cluster), first)
            _assert_bitwise_equal(run(self.num_sm - cluster), first)
            with delay_injection(DelayPoint.START, seed=19, max_ns=20_000):
                _assert_bitwise_equal(run(0), first)
            dq_base = torch.randn(q.shape, device="cuda", dtype=torch.float32)
            dq_acc_0 = run(0, dq_init=dq_base.clone())[0]
            dq_acc_1 = run(self.num_sm - cluster, dq_init=dq_base.clone())[0]
            self.assertTrue(torch.equal(dq_acc_0, dq_acc_1))
            torch.testing.assert_close(
                dq_acc_0, dq_base + first[0], atol=1e-4, rtol=1e-4
            )
        _assert_grads_close(first, _ref_grads(q, k, v, do, relations))
        self._log_ran(f"bwd disjoint dense dqacc {case}")

    @parameterize("case", [(64, False), (128, True)])
    def test_bwd_disjoint_q_dense_dqacc_cuda_graph_replay(self, case):
        head_dim, two_cta = case
        relations: Relations = [
            ([5, 270], [0, 310], MT_MAP.causal),
            ([280, 550], [120, 420], MT_MAP.inv_causal),
            ([560, 830], [200, 510], MT_MAP.full),
        ]
        total_q, total_k = 850, 520
        args = _range_args(relations)

        def fwd_bwd_inputs():
            q, k, v = _qkv(total_q, total_k, group=2, head_dim=head_dim)
            out, lse = _flex_flash_attn_fwd(
                q, k, v, **args, disable_fwd_atomic_reduction=True, deterministic=True
            )
            return q, k, v, out.to(q.dtype), lse, torch.randn_like(q)

        def bwd(q, k, v, out, lse, do):
            return _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                **args,
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=False,
                cat_gqa=True,
                deterministic=True,
            )[:3]

        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0" if two_cta else "1"}
        with mock.patch.dict("os.environ", env), record_bwd_kernels() as built:
            data_0 = fwd_bwd_inputs()
            data_1 = fwd_bwd_inputs()
            eager_0 = bwd(*data_0)
            eager_1 = bwd(*data_1)
            static = [t.clone() for t in data_0]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                bwd(*static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_grads = bwd(*static)
            (kernel,) = built
            self.assertTrue(kernel.range_dq_semaphore)
            self.assertEqual(kernel.use_2cta_instrs, two_cta)
            _replay(graph, static, data_1)
            _assert_bitwise_equal(static_grads, eager_1)
            _replay(graph, static, data_0)
            _assert_bitwise_equal(static_grads, eager_0)
            _replay(graph, static, data_1)
            _assert_bitwise_equal(static_grads, eager_1)
        self._log_ran(f"bwd disjoint dense dqacc graph {case}")

    @parameterize("cat_gqa", [False, True])
    def test_bwd_range_merge_2cta_overlap_gated_pair_handoff(self, cat_gqa):
        relations: Relations = [
            ([0, 900], [0, 256], MT_MAP.full),
            ([0, 128], [256, 440], MT_MAP.full),
            ([200, 315], [256, 440], MT_MAP.causal),
            (
                [320, 515],
                [256, 440],
                MT_MAP.bi_causal,
            ),
            ([520, 648], [256, 440], MT_MAP.inv_causal),
            ([584, 712], [256, 440], MT_MAP.causal),
            ([768, 890], [256, 440], MT_MAP.full),
        ]
        total_q, total_k = 900, 450
        group = 2 if cat_gqa else 1
        q, k, v, out, lse, do = self._fwd_bwd_inputs(relations, total_q, total_k, group)

        def merge_bwd(sm_margin: int):
            return self._det_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                relations,
                range_merge=True,
                cat_gqa=cat_gqa,
                sm_margin=sm_margin,
            )

        env = {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        with mock.patch.dict("os.environ", env):
            with record_bwd_kernels() as built:
                first = merge_bwd(0)
            (kernel,) = built
            self.assertTrue(kernel.use_2cta_instrs and kernel.dq_pair_handoff)
            with delay_injection(DelayPoint.CHAIN_WAIT, seed=23, max_ns=20_000):
                _assert_bitwise_equal(merge_bwd(0), first)
                _assert_bitwise_equal(merge_bwd(self.num_sm - 2), first)
            with delay_injection(DelayPoint.PAIR_ARRIVE, seed=23, max_ns=20_000):
                _assert_bitwise_equal(merge_bwd(0), first)
                _assert_bitwise_equal(merge_bwd(self.num_sm - 2), first)
        _assert_grads_close(first, _ref_grads(q, k, v, do, relations))
        self._log_ran(f"bwd range merge 2-CTA overlap-gated handoff {cat_gqa=}")


class TestFFARangePersistentGrid(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("sm_margin requires SM100/SM110 q/k ranges")
        torch.manual_seed(42)
        self.num_sm = torch.cuda.get_device_properties(0).multi_processor_count

    def _budget_error(self, sm_margin: int, cluster_size: int):
        return self.assertRaisesRegex(
            ValueError,
            rf"\({self.num_sm} - {sm_margin}\).*cluster_size={cluster_size}",
        )

    def _bwd_inputs(self):
        relations, total_q, total_k = _shared_k_relations()
        q, k, v = _qkv(total_q, total_k, group=2)
        out, lse = _flex_flash_attn_fwd(
            q, k, v, **_range_args(relations), disable_fwd_atomic_reduction=False
        )
        return relations, q, k, v, out.to(q.dtype), lse, torch.randn_like(q)

    def _bwd(self, relations, q, k, v, out, lse, do, sm_margin: int):
        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ):
            return _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                **_range_args(relations),
                disable_bwd_dkv_atomic_reduction=False,
                sm_margin=sm_margin,
            )[:3]

    def test_bwd_2cta_rejects_a_budget_below_one_cluster(self):
        inputs = self._bwd_inputs()
        with record_bwd_kernels() as built, self._budget_error(self.num_sm - 1, 2):
            self._bwd(*inputs, sm_margin=self.num_sm - 1)
        self.assertEqual(built, [])
        with record_bwd_kernels() as built, self._budget_error(self.num_sm, 2):
            self._bwd(*inputs, sm_margin=self.num_sm)
        self.assertEqual(built, [])

    def test_bwd_2cta_runs_on_one_cluster(self):
        relations, q, k, v, out, lse, do = self._bwd_inputs()
        with record_bwd_kernels() as built:
            grads = self._bwd(relations, q, k, v, out, lse, do, self.num_sm - 2)
        (kernel,) = built
        self.assertTrue(kernel.use_2cta_instrs)
        self.assertTrue(kernel.is_persistent)
        qkv = [t.detach().float().requires_grad_() for t in (q, k, v)]
        out_ref, _, _ = _ref_fwd(*qkv, relations)
        out_ref.backward(do.float())
        for name, got, ref in zip("qkv", grads, qkv):
            torch.testing.assert_close(
                got.float(),
                ref.grad,
                atol=3e-2,
                rtol=3e-2,
                msg=lambda m: f"d{name}: {m}",
            )

    def test_fwd_rejects_a_budget_below_one_cluster(self):
        relations, total_q, total_k = _shared_k_relations()
        q, k, v = _qkv(total_q, total_k, group=2)
        with record_fwd_kernels() as built, self._budget_error(self.num_sm, 1):
            _flex_flash_attn_fwd(
                q,
                k,
                v,
                **_range_args(relations),
                disable_fwd_atomic_reduction=False,
                sm_margin=self.num_sm,
            )
        self.assertEqual(built, [])
        q, k, v = _qkv(1024, 1024, group=1)
        ranges = torch.tensor([[0, 512], [512, 1024]], dtype=torch.int32, device="cuda")
        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ), record_fwd_kernels() as built, self._budget_error(self.num_sm - 1, 2):
            _flex_flash_attn_fwd(
                q,
                k,
                v,
                q_ranges=ranges,
                k_ranges=ranges,
                mask_types=MT_MAP.full,
                max_seqlen_q=512,
                max_seqlen_k=512,
                disable_fwd_atomic_reduction=True,
                sm_margin=self.num_sm - 1,
            )
        self.assertEqual(built, [])


class _ScanConflictsLauncher:
    def __init__(self, block_size: int, rows_per_token: int):
        self.block_size = block_size
        self.rows_per_token = rows_per_token

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
            rows_per_token=self.rows_per_token,
        )


class TestFFARangeChainScan(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the range chain runs on SM100/SM110")

    def _scan(
        self, ranges: list[list[int]], block_size: int, rows_per_token: int = 1
    ) -> torch.Tensor:
        num_blocks = max(e for _, e in ranges) * rows_per_token // block_size + 2
        mranges = torch.tensor(ranges, dtype=torch.int32, device="cuda")
        conflict = torch.zeros(1, num_blocks, dtype=torch.int32, device="cuda")
        args = (
            from_dlpack(mranges),
            from_dlpack(conflict),
            Int32(len(ranges)),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        launcher = _ScanConflictsLauncher(block_size, rows_per_token)
        cute.compile(launcher, *args)(*args)
        torch.cuda.synchronize()
        return conflict[0].cpu()

    @parameterize(
        "ranges",
        [
            [[0, 256], [128, 256]],
            [[5, 300], [130, 140], [0, 129]],
            [[0, 40 * 128], [33 * 128 + 7, 35 * 128]],
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

    @parameterize("rows_per_token", [2, 4])
    def test_scan_scales_token_ranges_to_packed_row_blocks(self, rows_per_token):
        block_size, g = 128, rows_per_token
        ranges = [[61, 100], [32, 64], [0, 33], [70, 70], [90, 130]]
        want = torch.zeros(130 * g // block_size + 2, dtype=torch.int32)
        for r, (start, end) in enumerate(ranges):
            if end > start:
                want[start * g // block_size : (end * g - 1) // block_size + 1] = r + 1
        got = self._scan(ranges, block_size, rows_per_token=g)
        self.assertTrue(torch.equal(got, want), f"{got.tolist()} != {want.tolist()}")


_FWD_UNIT_CLUSTERS = 8
_FWD_UNIT_RELATIONS: dict[int, Relations] = {
    1: [([5, 105], [0, 100], MT_MAP.full)],
    7: [([0, 768], [0, 300], MT_MAP.full), ([100, 1100], [300, 700], MT_MAP.causal)],
    8: [([0, 768], [0, 300], MT_MAP.full), ([100, 1300], [300, 700], MT_MAP.causal)],
    9: [([0, 768], [0, 300], MT_MAP.full), ([100, 1500], [300, 700], MT_MAP.causal)],
}
_FWD_EMPTY_RELATIONS: Relations = [
    ([40, 40], [0, 100], MT_MAP.full),
    ([90, 90], [100, 300], MT_MAP.full),
]
_BWD_UNIT_CLUSTERS = 4
_BWD_UNIT_RELATIONS: dict[int, Relations] = {
    1: [([0, 300], [0, 200], MT_MAP.full)],
    3: [([0, 300], [0, 512], MT_MAP.full), ([100, 400], [600, 800], MT_MAP.causal)],
    4: [([0, 300], [0, 512], MT_MAP.full), ([100, 400], [600, 1100], MT_MAP.causal)],
    5: [([0, 300], [0, 768], MT_MAP.full), ([100, 400], [800, 1300], MT_MAP.causal)],
}
_BWD_EMPTY_RELATIONS: Relations = [
    ([0, 300], [10, 10], MT_MAP.full),
    ([100, 400], [20, 20], MT_MAP.full),
]


class TestFFARangeDeterministicClaims(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("deterministic q/k ranges require SM100/SM110")
        torch.manual_seed(42)
        self.num_sm = torch.cuda.get_device_properties(0).multi_processor_count

    def _assert_each_unit_visited_once(
        self, visits: torch.Tensor, clusters: torch.Tensor, num_units: int, cluster: int
    ) -> None:
        valid = num_units * cluster
        self.assertTrue(torch.equal(visits[valid:], torch.zeros_like(visits[valid:])))
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

    def _fwd_visits(
        self, relations: Relations, num_tiles: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = _qkv(2000, 700, group=1, num_head_kv=1)
        with record_visits(num_tiles, 1) as (visits, clusters):
            out, _ = _flex_flash_attn_fwd(
                q,
                k,
                v,
                **_range_args(relations),
                disable_fwd_atomic_reduction=False,
                deterministic=True,
                sm_margin=self.num_sm - _FWD_UNIT_CLUSTERS,
            )
        out_ref, _, _ = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out, out_ref, atol=_TOL, rtol=_TOL)
        return visits.cpu(), clusters.cpu()

    def _bwd_2cta_visits(
        self, relations: Relations, num_tiles: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = _qkv(400, 1400, group=1, num_head_kv=1)
        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            **_range_args(relations),
            disable_fwd_atomic_reduction=False,
            deterministic=True,
        )
        out, do = out.to(q.dtype), torch.randn_like(q)
        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ), record_visits(num_tiles, 2) as (
            visits,
            clusters,
        ), record_bwd_kernels() as built:
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
                sm_margin=self.num_sm - 2 * _BWD_UNIT_CLUSTERS,
            )[:3]
        (kernel,) = built
        self.assertTrue(kernel.use_2cta_instrs)
        _assert_grads_close(grads, _ref_grads(q, k, v, do, relations))
        return visits.cpu(), clusters.cpu()

    @parameterize("num_units", sorted(_FWD_UNIT_RELATIONS))
    def test_fwd_claims_each_unit_once(self, num_units):
        visits, clusters = self._fwd_visits(
            _FWD_UNIT_RELATIONS[num_units], num_units + 64
        )
        self._assert_each_unit_visited_once(visits, clusters, num_units, 1)
        print(f"ran fwd claims T={num_units}", flush=True)

    def test_fwd_claims_no_unit_of_empty_q_ranges(self):
        visits, _ = self._fwd_visits(_FWD_EMPTY_RELATIONS, 64)
        self.assertTrue(torch.equal(visits, torch.zeros_like(visits)))
        print("ran fwd claims T=0", flush=True)

    @parameterize("num_units", sorted(_BWD_UNIT_RELATIONS))
    def test_bwd_2cta_claims_each_unit_once_on_both_peers(self, num_units):
        visits, clusters = self._bwd_2cta_visits(
            _BWD_UNIT_RELATIONS[num_units], 2 * (num_units + 16)
        )
        self._assert_each_unit_visited_once(visits, clusters, num_units, 2)
        print(f"ran bwd claims T={num_units}", flush=True)

    def test_bwd_2cta_claims_no_unit_of_empty_k_ranges(self):
        visits, _ = self._bwd_2cta_visits(_BWD_EMPTY_RELATIONS, 32)
        self.assertTrue(torch.equal(visits, torch.zeros_like(visits)))
        print("ran bwd claims T=0", flush=True)

    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_cuda_graph_replay_matches_eager_bitwise(self, head_dim):
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
            q, k, v = _qkv(total_q, total_k, group=2, head_dim=head_dim)
            out, lse = fwd(q, k, v)
            return q, k, v, out.to(q.dtype), lse, torch.randn_like(q)

        with mock.patch.dict(
            "os.environ", {"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "0"}
        ):
            data_0 = inputs()
            data_1 = inputs()
            eager_0 = (*fwd(*data_0[:3]), *bwd(*data_0))
            eager_1 = (*fwd(*data_1[:3]), *bwd(*data_1))
            static = [t.clone() for t in data_0]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                fwd(*static[:3])
                bwd(*static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = (*fwd(*static[:3]), *bwd(*static))
            _replay(graph, static, data_1)
            _assert_bitwise_equal(static_out, eager_1)
            _replay(graph, static, data_0)
            _assert_bitwise_equal(static_out, eager_0)
            _replay(graph, static, data_1)
            _assert_bitwise_equal(static_out, eager_1)
        print(f"ran cuda graph replay hd{head_dim}", flush=True)


class _BlockInfoDualLauncher:
    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        is_causal: bool,
        is_local: bool,
        window_left: int | None,
        window_right: int | None,
        pack_factor: int,
        per_range: bool,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.is_causal = is_causal
        self.is_local = is_local
        self.window_left = window_left
        self.window_right = window_right
        self.pack_factor = pack_factor
        self.per_range = per_range

    @cute.jit
    def __call__(
        self,
        mSeqQ: cute.Tensor,
        mSeqK: cute.Tensor,
        mIdx: cute.Tensor,
        mAttn: cute.Tensor,
        mFwdLo: cute.Tensor,
        mFwdHi: cute.Tensor,
        mDualLo: cute.Tensor,
        mDualHi: cute.Tensor,
        count: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = (count + 127) // 128
        self.kernel(
            mSeqQ, mSeqK, mIdx, mAttn, mFwdLo, mFwdHi, mDualLo, mDualHi, count
        ).launch(grid=(grid_x, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        mSeqQ: cute.Tensor,
        mSeqK: cute.Tensor,
        mIdx: cute.Tensor,
        mAttn: cute.Tensor,
        mFwdLo: cute.Tensor,
        mFwdHi: cute.Tensor,
        mDualLo: cute.Tensor,
        mDualHi: cute.Tensor,
        count: Int32,
    ):
        wl = Int32(self.window_left) if self.window_left is not None else None
        wr = Int32(self.window_right) if self.window_right is not None else None
        info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            wl,
            wr,
            qhead_per_kvhead_packgqa=self.pack_factor,
        )
        i = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if i < count:
            sq = Int32(mSeqQ[i])
            sk = Int32(mSeqK[i])
            idx = Int32(mIdx[i])
            if cutlass.const_expr(self.per_range):
                at = Int32(mAttn[i])
                flo, fhi = info.get_m_block_min_max_per_range_seqlens(sq, sk, idx, at)
                dlo, dhi = info.get_n_block_min_max_for_m_block_per_range_seqlens(
                    sq, sk, idx, at
                )
            else:
                flo, fhi = info.get_m_block_min_max_seqlens(sq, sk, idx)
                dlo, dhi = info.get_n_block_min_max_for_m_block_seqlens(sq, sk, idx)
            mFwdLo[i] = flo
            mFwdHi[i] = fhi
            mDualLo[i] = dlo
            mDualHi[i] = dhi


class TestBlockInfoDualBounds(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("SM100/SM110 BlockInfo dual bounds test")

    def _eval_device(
        self,
        launcher: _BlockInfoDualLauncher,
        seqlen_q: torch.Tensor,
        seqlen_k: torch.Tensor,
        block: torch.Tensor,
        mask_type: torch.Tensor,
    ) -> torch.Tensor:
        count = seqlen_q.shape[0]
        sq, sk, idx, at = (
            t.to("cuda").contiguous() for t in (seqlen_q, seqlen_k, block, mask_type)
        )
        flo = torch.empty(count, dtype=torch.int32, device="cuda")
        fhi = torch.empty(count, dtype=torch.int32, device="cuda")
        dlo = torch.empty(count, dtype=torch.int32, device="cuda")
        dhi = torch.empty(count, dtype=torch.int32, device="cuda")
        args = (
            from_dlpack(sq),
            from_dlpack(sk),
            from_dlpack(idx),
            from_dlpack(at),
            from_dlpack(flo),
            from_dlpack(fhi),
            from_dlpack(dlo),
            from_dlpack(dhi),
            Int32(count),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        cute.compile(launcher, *args)(*args)
        return torch.stack([flo, fhi, dlo, dhi]).cpu()

    def _assert_dual_equivalence(
        self,
        launcher: _BlockInfoDualLauncher,
        seqlen_q: torch.Tensor,
        seqlen_k: torch.Tensor,
        mask_type: torch.Tensor,
    ) -> None:
        tile_m, tile_n = launcher.tile_m, launcher.tile_n
        num_m = (seqlen_q * launcher.pack_factor + tile_m - 1) // tile_m
        num_n = (seqlen_k + tile_n - 1) // tile_n
        num_blocks = int(torch.maximum(num_m, num_n).max())
        blocks = torch.arange(num_blocks, dtype=torch.int32)
        bounds = self._eval_device(
            launcher,
            seqlen_q.repeat_interleave(num_blocks),
            seqlen_k.repeat_interleave(num_blocks),
            blocks.repeat(seqlen_q.shape[0]),
            mask_type.repeat_interleave(num_blocks),
        )
        m_lo, m_hi, n_lo, n_hi = bounds.view(4, -1, num_blocks)
        # [C, m, n]: m-bounds indexed by n block, n-bounds indexed by m block.
        m = blocks[None, :, None]
        n = blocks[None, None, :]
        in_m = (m_lo[:, None, :] <= m) & (m < m_hi[:, None, :])
        in_n = (n_lo[:, :, None] <= n) & (n < n_hi[:, :, None])
        valid = (m < num_m[:, None, None]) & (n < num_n[:, None, None])
        bad = ((in_m != in_n) & valid).nonzero()
        combination = bad[:, 0]
        mismatches = torch.stack(
            [
                seqlen_q[combination],
                seqlen_k[combination],
                mask_type[combination],
                bad[:, 1],
                bad[:, 2],
            ],
            dim=1,
        )
        self.assertEqual(
            mismatches.shape[0],
            0,
            f"{vars(launcher)}: (seqlen_q, seqlen_k, mask_type, m, n) where the "
            f"bounds disagree: {mismatches[:8].tolist()}",
        )

    @parameterize("pack_factor", [1, 2, 4, 8])
    @parameterize("tile_n", [128, 256])
    def test_per_range_dual_equivalence_on_device(self, pack_factor, tile_n):
        seqlens = torch.tensor(
            [0, 1, 2, 7, 15, 16, 31, 32, 33, 64, 127, 128, 129, 195, 256, 257, 390],
            dtype=torch.int32,
        )
        mask_types = torch.tensor(
            [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal],
            dtype=torch.int32,
        )
        seqlen_q, seqlen_k, mask_type = torch.cartesian_prod(
            seqlens, seqlens, mask_types
        ).unbind(1)
        launcher = _BlockInfoDualLauncher(
            128, tile_n, True, False, None, None, pack_factor, per_range=True
        )
        self._assert_dual_equivalence(launcher, seqlen_q, seqlen_k, mask_type)

    @parameterize(
        "spec",
        [
            (False, False, None, None, 1, 256),
            (False, False, None, None, 4, 128),
            (True, False, None, None, 1, 256),
            (True, False, None, None, 2, 128),
            (True, False, None, None, 8, 256),
            (False, True, 0, 0, 1, 128),
            (False, True, 7, 19, 1, 128),
            (False, True, 2, 1, 1, 256),
            (False, True, 128, 0, 1, 256),
            (False, True, 0, 128, 1, 256),
            (False, True, None, 15, 1, 128),
            (False, True, 15, None, 1, 128),
        ],
    )
    def test_static_causal_and_local_dual_equivalence_on_device(self, spec):
        is_causal, is_local, wl, wr, pack_factor, tile_n = spec
        seqlens = torch.tensor(
            [1, 2, 7, 16, 33, 64, 127, 128, 129, 255, 256, 257, 390], dtype=torch.int32
        )
        seqlen_q, seqlen_k = torch.cartesian_prod(seqlens, seqlens).unbind(1)
        launcher = _BlockInfoDualLauncher(
            128, tile_n, is_causal, is_local, wl, wr, pack_factor, per_range=False
        )
        self._assert_dual_equivalence(
            launcher, seqlen_q, seqlen_k, torch.zeros_like(seqlen_q)
        )


if __name__ == "__main__":
    run_tests()
