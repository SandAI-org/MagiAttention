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

"""Range locks of the SM100 CuTeDSL range atomic forward.

A stage tile of a relation whose q range does not start on a multiple of
tile_m straddles two lock blocks and merges its O/LSE rows under both locks.
The masks here are a sink-plus-sliding-window decomposition. Every row from
``start`` on is covered by a relation against the sink keys (one K block per
tile) and one against a window of ``_WINDOW + 1`` keys over the same q range,
so their tiles contend for the same lock blocks. The correctness test moves
half of the sink keys into a relation that starts ``_SHIFT`` rows later; each
of its tiles overlaps two tiles of the others and shares one lock block with
each, so a merge that skips one of its two locks races on the shared rows.
"""

import math
import statistics
from contextlib import contextmanager
from typing import Iterator
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd
from magi_attention.testing import parameterize

_TILE_M = 128
_SINK = 16
_SHIFT = _TILE_M // 2
_WINDOW = 1024
# Row 1041 = 8 * 128 + 17 starts the overlapping q range inside a lock block;
# row 1152 = 9 * 128 starts it on a block boundary. Both keep the window keys
# clear of the sink keys (start - _WINDOW >= _SINK).
_UNALIGNED_START = 8 * _TILE_M + 17
_ALIGNED_START = 9 * _TILE_M
# fp32 O of bf16 inputs against the fp32 reference of the same bf16 inputs.
_TOL = 1e-2

Relations = list[tuple[int, int, int, int, int]]


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


def _sink_window_relations(
    seqlen: int, start: int, batch: int = 1, shift_half_sink: bool = False
) -> Relations:
    """(q_start, q_end, k_start, k_end, mask_type) of ``batch`` packed
    sequences: causal rows before ``start``, then the sink keys (full) and
    the window keys ``[q - _WINDOW, q]`` (bi_causal). With
    ``shift_half_sink`` the second half of the sink keys is a separate
    relation starting at ``start + _SHIFT``."""
    sink_end = _SINK // 2 if shift_half_sink else _SINK
    relations: Relations = []
    for b in range(batch):
        o = b * seqlen
        relations += [
            (o, o + start, o, o + start, MT_MAP.causal),
            (o + start, o + seqlen, o, o + sink_end, MT_MAP.full),
            (o + start, o + seqlen, o + start - _WINDOW, o + seqlen, MT_MAP.bi_causal),
        ]
        if shift_half_sink:
            relations.append(
                (o + start + _SHIFT, o + seqlen, o + sink_end, o + _SINK, MT_MAP.full)
            )
    return relations


def _range_args(relations: Relations, device="cuda") -> dict:
    return dict(
        q_ranges=torch.tensor(
            [r[0:2] for r in relations], dtype=torch.int32, device=device
        ),
        k_ranges=torch.tensor(
            [r[2:4] for r in relations], dtype=torch.int32, device=device
        ),
        mask_types=torch.tensor(
            [r[4] for r in relations], dtype=torch.int32, device=device
        ),
        max_seqlen_q=max(r[1] - r[0] for r in relations),
        max_seqlen_k=max(r[3] - r[2] for r in relations),
    )


def _ref_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, relations: Relations
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 (out, lse) of the relation union; relations share no (q, k) pair."""
    total_q, total_k = q.shape[0], k.shape[0]
    mask = torch.zeros(total_q, total_k, dtype=torch.bool, device=q.device)
    for qs, qe, ks, ke, mask_type in relations:
        i = torch.arange(qe - qs, device=q.device)[:, None]
        j = torch.arange(ke - ks, device=q.device)[None, :]
        rel = torch.ones(qe - qs, ke - ks, dtype=torch.bool, device=q.device)
        if mask_type in (MT_MAP.causal, MT_MAP.bi_causal):
            rel &= j <= i + (ke - ks) - (qe - qs)
        if mask_type in (MT_MAP.inv_causal, MT_MAP.bi_causal):
            rel &= j >= i
        assert not (mask[qs:qe, ks:ke] & rel).any(), "relations share a pair"
        mask[qs:qe, ks:ke] |= rel
    s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask, -math.inf)
    lse = torch.logsumexp(s, dim=-1)
    out = torch.einsum("hqk,khd->qhd", torch.exp(s - lse[..., None]), v.float())
    return out, lse.t().contiguous()


def _qkv(total: int, num_head: int, head_dim: int, head_dim_v: int):
    q, k = (
        torch.randn(total, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    v = torch.randn(total, num_head, head_dim_v, device="cuda", dtype=torch.bfloat16)
    return q, k, v


class TestFfaFwdRangeLocks(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] != 10:
            self.skipTest("the range lock protocol is in the SM100 fwd kernel")
        torch.manual_seed(42)

    def _fwd_kernel(self, q, k, v, relations: Relations):
        """Run the range atomic fwd; return (out, lse, the kernel object)."""
        with _record_fwd_kernels() as built:
            out, lse = _flex_flash_attn_fwd(
                q, k, v, **_range_args(relations), disable_fwd_atomic_reduction=False
            )
        self.assertEqual(len(built), 1)
        kernel = built[0]
        self.assertTrue(kernel.use_per_range_mask)
        self.assertFalse(kernel.disable_fwd_atomic_reduction)
        return out, lse, kernel

    @parameterize("head_dims", [(192, 128), (128, 128)])
    @parameterize("start", [_UNALIGNED_START, _ALIGNED_START])
    def test_overlapping_q_ranges_match_reference(self, head_dims, start):
        """Overlapping q ranges merge to the dense reference at hd192 and
        hd128, both on the non-persistent q/k-range launch. With the
        unaligned start every tile straddles two lock blocks; with the
        aligned start only the tiles of the shifted relation do."""
        head_dim, head_dim_v = head_dims
        seqlen = 4096
        relations = _sink_window_relations(seqlen, start, shift_half_sink=True)
        q, k, v = _qkv(seqlen, 4, head_dim, head_dim_v)
        out, lse, kernel = self._fwd_kernel(q, k, v, relations)
        self.assertFalse(kernel.is_persistent)
        out_ref, lse_ref = _ref_fwd(q, k, v, relations)
        torch.testing.assert_close(out.float(), out_ref, atol=_TOL, rtol=_TOL)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)

    def test_unaligned_merges_do_not_serialize(self):
        """hd192 with the unaligned start runs within 2x of the aligned start.

        The launch is non-persistent with one tile per CTA, and the per-range
        kernel issues m blocks in descending order, so if a tile waits for its
        upper lock while it holds the lower one, the tiles below it queue
        behind it and their merges run one at a time: the unaligned case then
        takes about 17x the aligned one on a B300 at 1400 MHz. Without that
        wait it takes about 1.5x, so 2.0 separates the two. Both cases do the
        same math up to 111 rows of the causal prefix.
        """
        num_sm = torch.cuda.get_device_properties(0).multi_processor_count
        if num_sm < 128:
            # The queue is as long as the number of co-resident tiles; the
            # factor of 17 is measured with 148 SMs.
            self.skipTest(f"needs at least 128 SMs, found {num_sm}")
        seqlen, batch = 32768, 4
        q, k, v = _qkv(seqlen * batch, 16, 192, 128)
        args = {
            start: _range_args(_sink_window_relations(seqlen, start, batch))
            for start in (_UNALIGNED_START, _ALIGNED_START)
        }
        with _record_fwd_kernels() as built:
            for start in args:
                _flex_flash_attn_fwd(
                    q, k, v, **args[start], disable_fwd_atomic_reduction=False
                )
        # One kernel serves both cases: the start only changes runtime data.
        self.assertEqual(len(built), 1)
        self.assertFalse(built[0].is_persistent)
        self.assertTrue(built[0].use_per_range_mask)

        times: dict[int, list[float]] = {start: [] for start in args}
        for _ in range(3):
            for start in args:
                _flex_flash_attn_fwd(
                    q, k, v, **args[start], disable_fwd_atomic_reduction=False
                )
        for _ in range(10):
            # Interleaved so clock and power drift hit both cases alike.
            for start in args:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                _flex_flash_attn_fwd(
                    q, k, v, **args[start], disable_fwd_atomic_reduction=False
                )
                end.record()
                end.synchronize()
                times[start].append(begin.elapsed_time(end))
        unaligned = statistics.median(times[_UNALIGNED_START])
        aligned = statistics.median(times[_ALIGNED_START])
        print(
            f"hd192 unaligned {unaligned:.3f} ms, aligned {aligned:.3f} ms, "
            f"ratio {unaligned / aligned:.2f}",
            flush=True,
        )
        self.assertLess(unaligned / aligned, 2.0)


if __name__ == "__main__":
    run_tests()
