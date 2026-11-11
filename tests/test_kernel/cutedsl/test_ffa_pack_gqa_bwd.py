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

from typing import Any
from unittest import TestCase

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import (
    MT_MAP,
    BwdGqaFold,
    get_device_arch,
    select_bwd_gqa_fold,
)
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_bwd
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import (
    AttnTensors,
    assert_close_to_torch_ref,
    record_bwd_kernels,
)
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp

# Relation lengths with range starts that are neither tile- nor
# group-aligned, Q tails of 1..tile_m-1 packed rows, and inv/bi-causal
# relations with len_q != len_k.
_Q_LENS = [1, 7, 31, 33, 129, 320, 200, 77]
_K_LENS = [5, 7, 40, 33, 100, 320, 257, 90]
_MASK_CYCLE = [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal]

# (num_head, num_head_kv, bwd_pack_factor, expected (pack_factor, head_loop))
_FOLD_CASES = {
    "full_pack2": (4, 2, 2, (2, False)),
    "full_pack4": (8, 2, 4, (4, False)),
    "full_pack8": (16, 2, 8, (8, False)),
    "pack4_loop2": (16, 2, 4, (4, True)),
    "pack2_loop3": (12, 2, 2, (2, True)),
}


def _contiguous_ranges(lengths: list[int]) -> list[list[int]]:
    ranges, start = [], 0
    for length in lengths:
        ranges.append([start, start + length])
        start += length
    return ranges


class TestFfaPackGqaBwd(DistTestBase):
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

    def _skip_unless_sm100(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("bwd_pack_factor > 1 requires SM100/SM110")

    def _log_ran(self, test_case: str) -> None:
        print(f"[RANK {self.rank}] ran {test_case}", flush=True)

    def _inputs(
        self,
        *,
        num_head: int,
        num_head_kv: int,
        d: int,
        q_ranges: list[list[int]],
        k_ranges: list[list[int]],
        mask_types: list[int] | int,
        seed: int,
    ):
        """Forward of a ranges problem; returns its bwd inputs and kwargs."""
        torch.random.manual_seed(seed)
        dtype, device = torch.bfloat16, self.device
        total_q = max(end for _, end in q_ranges)
        total_k = max(end for _, end in k_ranges)
        d_v = 128 if d == 192 else d
        q = torch.randn(total_q, num_head, d, device=device, dtype=dtype)
        k = torch.randn(total_k, num_head_kv, d, device=device, dtype=dtype)
        v = torch.randn(total_k, num_head_kv, d_v, device=device, dtype=dtype)
        kwargs = dict(
            q_ranges=torch.tensor(q_ranges, device=device, dtype=torch.int32),
            k_ranges=torch.tensor(k_ranges, device=device, dtype=torch.int32),
            mask_types=(
                mask_types
                if isinstance(mask_types, int)
                else torch.tensor(mask_types, device=device, dtype=torch.int32)
            ),
            max_seqlen_q=max(max(end - start for start, end in q_ranges), 1),
            max_seqlen_k=max(max(end - start for start, end in k_ranges), 1),
        )
        out, meta = flex_flash_attn_func(q, k, v, **kwargs)
        do = torch.randn_like(out)
        return q, k, v, out.to(dtype), meta.lse, do, kwargs

    def _assert_kernel(
        self,
        bwd_kernel: FFABwdSm100,
        test_case: str,
        *,
        fold: tuple[int, bool],
        d: int,
        direct: bool,
    ) -> None:
        """The fold, the 1-CTA (d=64) or 2-CTA (d=128/192) kernel, and dK/dV
        reduced through the postprocess unless the direct store is declared."""
        gqa_fold = bwd_kernel.gqa_fold
        self.assertEqual((gqa_fold.pack_factor, gqa_fold.head_loop), fold, test_case)
        self.assertEqual(bwd_kernel.cta_group_size, 1 if d == 64 else 2, test_case)
        self.assertEqual(bwd_kernel.dKV_postprocess, not direct, test_case)
        if fold[0] > 1:
            self.assertTrue(bwd_kernel.dQ_rowmajor_accum, test_case)
            self.assertFalse(bwd_kernel.use_dense_dqacc_for_ranges, test_case)

    # ─────────────────────────────────────────────────────────────────────
    # Ranges
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("direct", [False, True])
    @parameterize("fold_case", list(_FOLD_CASES))
    @parameterize("d", [64, 128, 192])
    def test_ranges_fold(self, d, fold_case, direct):
        """Full and partial packs on the 1-CTA (d=64) and 2-CTA (d=128/192)
        kernels, with all four per-range mask types, unaligned range starts
        and short Q tails, on the atomic and the direct dK/dV store."""
        self._skip_unless_sm100()
        num_head, num_head_kv, pack_factor, expected = _FOLD_CASES[fold_case]
        if d == 192 and fold_case == "full_pack8":
            # Same kernel shape as full_pack4 at d=192 apart from the pack
            # factor, which d=64/128 already cover.
            return
        q_ranges = _contiguous_ranges(_Q_LENS)
        k_ranges = _contiguous_ranges(_K_LENS)
        attn_type_map = [_MASK_CYCLE[i % 4] for i in range(len(_Q_LENS))]
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=num_head,
            num_head_kv=num_head_kv,
            d=d,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=attn_type_map,
            seed=self.seed + d + num_head + int(direct),
        )
        test_case = f"[test_ranges_fold][{d=}][{fold_case=}][{direct=}]"
        self._log_ran(test_case)
        with record_bwd_kernels() as built:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                bwd_pack_factor=pack_factor,
                disable_bwd_dkv_atomic_reduction=direct,
                **kwargs,
            )
        (bwd_kernel,) = built
        self._assert_kernel(bwd_kernel, test_case, fold=expected, d=d, direct=direct)
        self.assertTrue(bwd_kernel.use_per_range_mask, test_case)
        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out, dq, dk, dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("mask_type", [MT_MAP.full, MT_MAP.causal])
    @parameterize("d", [64, 128])
    def test_static_mask_fold(self, d, mask_type):
        """The static mask entry (one ``int`` mask type) bounds and masks
        packed rows through ``get_m_block_min_max`` and the dense transposed
        mask; ``len_q != len_k`` moves the causal diagonal off the tile grid."""
        self._skip_unless_sm100()
        q_ranges = _contiguous_ranges(_Q_LENS)
        k_ranges = _contiguous_ranges(_K_LENS)
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=8,
            num_head_kv=2,
            d=d,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=mask_type,
            seed=self.seed + d + mask_type,
        )
        test_case = f"[test_static_mask_fold][{d=}][{mask_type=}]"
        self._log_ran(test_case)
        with record_bwd_kernels() as built:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                bwd_pack_factor=4,
                disable_bwd_dkv_atomic_reduction=True,
                **kwargs,
            )
        (bwd_kernel,) = built
        self._assert_kernel(bwd_kernel, test_case, fold=(4, False), d=d, direct=True)
        self.assertFalse(bwd_kernel.use_per_range_mask, test_case)
        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out, dq, dk, dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=[mask_type] * len(q_ranges),
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("d", [64, 128])
    def test_overlap(self, d):
        """Overlapping q ranges (several relations reduce into one dQ row)
        and overlapping k ranges (atomic dK/dV) under the full pack. The
        (q, k) rectangles of the relations are pairwise disjoint, so the
        mask-union reference applies."""
        self._skip_unless_sm100()
        q_ranges = [[0, 300], [200, 600], [600, 1000], [0, 150], [999, 1300]]
        k_ranges = [[0, 300], [300, 700], [100, 500], [700, 1000], [900, 1300]]
        attn_type_map = [_MASK_CYCLE[i % 4] for i in range(len(q_ranges))]
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=8,
            num_head_kv=2,
            d=d,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=attn_type_map,
            seed=self.seed + d,
        )
        test_case = f"[test_overlap][{d=}]"
        self._log_ran(test_case)
        with record_bwd_kernels() as built:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q, k, v, out, lse, do, bwd_pack_factor=4, **kwargs
            )
        (bwd_kernel,) = built
        self._assert_kernel(bwd_kernel, test_case, fold=(4, False), d=d, direct=False)
        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out, dq, dk, dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("fold_case", ["full_pack4", "pack4_loop2"])
    @parameterize("d", [64, 128])
    def test_persistent_empty_tiles(self, d, fold_case):
        """``sm_margin = num_sm - 2`` leaves two SMs (two 1-CTA CTAs or one
        2-CTA cluster), and bi-causal relations with ``len_q > len_k`` give
        every CTA runs of K tiles without any Q block, whose direct dK/dV
        are zero-filled. Covers the 1-CTA (d=64) and 2-CTA (d=128) kernels."""
        self._skip_unless_sm100()
        num_head, num_head_kv, pack_factor, expected = _FOLD_CASES[fold_case]
        nseg, len_q, len_k = 24, 300, 150
        pattern = [MT_MAP.full, MT_MAP.bi_causal, MT_MAP.bi_causal, MT_MAP.causal]
        attn_type_map = [pattern[i % 4] for i in range(nseg)]
        q_ranges = _contiguous_ranges([len_q] * nseg)
        k_ranges = _contiguous_ranges([len_k] * nseg)
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=num_head,
            num_head_kv=num_head_kv,
            d=d,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=attn_type_map,
            seed=self.seed + d + num_head,
        )
        num_sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        test_case = f"[test_persistent_empty_tiles][{d=}][{fold_case=}]"
        self._log_ran(test_case)
        with record_bwd_kernels() as built:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                bwd_pack_factor=pack_factor,
                disable_bwd_dkv_atomic_reduction=True,
                sm_margin=num_sm - 2,
                **kwargs,
            )
        (bwd_kernel,) = built
        self._assert_kernel(bwd_kernel, test_case, fold=expected, d=d, direct=True)
        self.assertTrue(bwd_kernel.is_persistent, test_case)
        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out, dq, dk, dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("heads", [(8, 2), (12, 2)])
    @parameterize("d", [64, 128])
    def test_range_merge_fold(self, d, heads):
        """RangeMerge pairs, some of which leave a K tile without any Q
        block, under the full pack (8/2: pack 4) and the partial pack with
        the head loop (12/2: pack 2, three groups), both requested through
        ``flex_flash_attn_func``."""
        self._skip_unless_sm100()
        num_head, num_head_kv = heads
        pack_factor, expected = {
            (8, 2): (4, (4, False)),
            (12, 2): (2, (2, True)),
        }[heads]
        device, dtype = self.device, torch.bfloat16
        q_groups = [(0, 257), (257, 386), (386, 579)]
        k_groups = [(0, 129), (129, 386), (386, 579)]
        relation_ids = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
        q_ranges = [list(q_groups[qi]) for qi, _ in relation_ids]
        k_ranges = [list(k_groups[ki]) for _, ki in relation_ids]
        attn_type_map = [
            MT_MAP.bi_causal,
            MT_MAP.causal,
            MT_MAP.full,
            MT_MAP.full,
            MT_MAP.inv_causal,
            MT_MAP.full,
        ]
        torch.random.manual_seed(self.seed + d + num_head)
        q = torch.randn(579, num_head, d, device=device, dtype=dtype)
        k = torch.randn(579, num_head_kv, d, device=device, dtype=dtype)
        v = torch.randn_like(k)
        q, k, v = (t.requires_grad_() for t in (q, k, v))
        test_case = f"[test_range_merge_fold][{d=}][{heads=}]"
        self._log_ran(test_case)
        with record_bwd_kernels() as built:
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=torch.tensor(q_ranges, device=device, dtype=torch.int32),
                k_ranges=torch.tensor(k_ranges, device=device, dtype=torch.int32),
                mask_types=torch.tensor(
                    attn_type_map, device=device, dtype=torch.int32
                ),
                max_seqlen_q=257,
                max_seqlen_k=257,
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=True,
                range_merge=True,
                bwd_pack_factor=pack_factor,
            )
            do = torch.randn_like(out)
            grads = torch.autograd.grad(out, (q, k, v), do)
        (bwd_kernel,) = built
        self._assert_kernel(bwd_kernel, test_case, fold=expected, d=d, direct=True)
        self.assertTrue(bwd_kernel.range_merge, test_case)
        assert_close_to_torch_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            actual=AttnTensors(out.detach(), *grads),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("direct", [False, True])
    def test_empty_ranges(self, direct):
        """Relations with an empty q range, an empty k range, both empty, or
        an empty bi-causal band (``len_q > len_k``) next to non-empty ones.
        The empty-k relation shares its q rows with a non-empty relation, so
        every q row stays covered."""
        self._skip_unless_sm100()
        q_ranges = [[0, 100], [100, 100], [0, 77], [100, 250], [250, 250], [250, 314]]
        k_ranges = [[0, 90], [90, 140], [140, 140], [140, 220], [220, 220], [220, 284]]
        attn_type_map = [
            MT_MAP.causal,
            MT_MAP.full,
            MT_MAP.full,
            MT_MAP.bi_causal,
            MT_MAP.full,
            MT_MAP.inv_causal,
        ]
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=8,
            num_head_kv=2,
            d=128,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=attn_type_map,
            seed=self.seed + int(direct),
        )
        test_case = f"[test_empty_ranges][{direct=}]"
        self._log_ran(test_case)
        with record_bwd_kernels() as built:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                bwd_pack_factor=4,
                disable_bwd_dkv_atomic_reduction=direct,
                **kwargs,
            )
        (bwd_kernel,) = built
        self._assert_kernel(
            bwd_kernel, test_case, fold=(4, False), d=128, direct=direct
        )
        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out, dq, dk, dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("direct", [False, True])
    def test_caller_buffers(self, direct):
        """dQ always accumulates into a caller buffer; dK/dV accumulate on
        the atomic path and are overwritten by the direct store."""
        self._skip_unless_sm100()
        q_ranges = _contiguous_ranges(_Q_LENS)
        k_ranges = _contiguous_ranges(_K_LENS)
        attn_type_map = [_MASK_CYCLE[i % 4] for i in range(len(_Q_LENS))]
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=8,
            num_head_kv=2,
            d=128,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=attn_type_map,
            seed=self.seed + int(direct),
        )
        fp32 = dict(dq_type=torch.float32, dk_type=torch.float32, dv_type=torch.float32)

        def bwd(**bufs) -> tuple[torch.Tensor, ...]:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                bwd_pack_factor=4,
                disable_bwd_dkv_atomic_reduction=direct,
                **fp32,
                **kwargs,
                **bufs,
            )
            return dq, dk, dv

        with record_bwd_kernels() as built:
            fresh = bwd()
        (bwd_kernel,) = built
        test_case = f"[test_caller_buffers][{direct=}]"
        self._assert_kernel(
            bwd_kernel, test_case, fold=(4, False), d=128, direct=direct
        )
        self._log_ran(test_case)
        init = [torch.randn_like(g) for g in fresh]
        bufs = [t.clone() for t in init]
        got = bwd(dq=bufs[0], dk=bufs[1], dv=bufs[2])
        for name, g, buf, base, accumulates in zip(
            ("dq", "dk", "dv"),
            got,
            bufs,
            init,
            (True, not direct, not direct),
        ):
            self.assertIs(g, buf, f"{test_case} {name} must be the caller buffer")
            expected = fresh[("dq", "dk", "dv").index(name)]
            if accumulates:
                expected = expected + base
            torch.testing.assert_close(
                g,
                expected,
                rtol=0,
                atol=1e-4 * expected.abs().max().item(),
                msg=lambda m: f"{test_case} {name}: {m}",
            )

    @with_run_in_mp
    @parameterize("d", [64, 128])
    def test_full_pack_matches_unpacked(self, d):
        """``bwd_pack_factor=4`` and ``bwd_pack_factor=1`` reduce the same
        partial products in a different order, so their gradients agree to
        fp32 rounding of the reduction."""
        self._skip_unless_sm100()
        q_ranges = _contiguous_ranges(_Q_LENS)
        k_ranges = _contiguous_ranges(_K_LENS)
        attn_type_map = [_MASK_CYCLE[i % 4] for i in range(len(_Q_LENS))]
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=8,
            num_head_kv=2,
            d=d,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=attn_type_map,
            seed=self.seed + d,
        )
        fp32 = dict(dq_type=torch.float32, dk_type=torch.float32, dv_type=torch.float32)
        test_case = f"[test_full_pack_matches_unpacked][{d=}]"
        self._log_ran(test_case)
        grads = {}
        for pack_factor in (4, 1):
            with record_bwd_kernels() as built:
                dq, dk, dv, _ = _flex_flash_attn_bwd(
                    q, k, v, out, lse, do, bwd_pack_factor=pack_factor, **fp32, **kwargs
                )
            (bwd_kernel,) = built
            self._assert_kernel(
                bwd_kernel,
                f"{test_case}[{pack_factor=}]",
                fold=(pack_factor, False),
                d=d,
                direct=False,
            )
            grads[pack_factor] = (dq, dk, dv)
        for name, packed, unpacked in zip(("dq", "dk", "dv"), grads[4], grads[1]):
            torch.testing.assert_close(
                packed,
                unpacked,
                rtol=0,
                atol=1e-5 * unpacked.abs().max().item(),
                msg=lambda m: f"{test_case} {name}: {m}",
            )

    @with_run_in_mp
    @parameterize(
        "case",
        [
            # (d, num_head, num_head_kv, direct)
            (64, 32, 2, False),
            (128, 32, 2, True),
            (128, 64, 1, False),
            (64, 128, 1, True),
        ],
    )
    def test_wide_full_pack(self, case):
        """Full packs of 16, 64 and 128 q heads per kv head, up to the Q rows
        of one CTA (128 on the 1-CTA kernel at d=64, 64 on the 2-CTA kernel
        at d=128), where a CTA holds a single token."""
        self._skip_unless_sm100()
        d, num_head, num_head_kv, direct = case
        qhead_per_kvhead = num_head // num_head_kv
        q_ranges = _contiguous_ranges(_Q_LENS)
        k_ranges = _contiguous_ranges(_K_LENS)
        attn_type_map = [_MASK_CYCLE[i % 4] for i in range(len(_Q_LENS))]
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=num_head,
            num_head_kv=num_head_kv,
            d=d,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=attn_type_map,
            seed=self.seed + d + num_head,
        )
        test_case = f"[test_wide_full_pack][{case=}]"
        self._log_ran(test_case)
        with record_bwd_kernels() as built:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                bwd_pack_factor=qhead_per_kvhead,
                disable_bwd_dkv_atomic_reduction=direct,
                **kwargs,
            )
        (bwd_kernel,) = built
        self._assert_kernel(
            bwd_kernel, test_case, fold=(qhead_per_kvhead, False), d=d, direct=direct
        )
        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out, dq, dk, dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("variant", ["packed", "unpacked", "packed_persistent"])
    @parameterize("d", [64, 128])
    def test_sorted_q_dq_route(self, d, variant):
        """Sorted, disjoint q ranges (``disable_fwd_atomic_reduction``) select
        the per-range dQ slots for the unpacked backward only; a packed
        backward reduces dQ through the row-major dQacc. dQ accumulates onto a
        caller fp32 buffer. ``packed_persistent`` adds the remaining arguments
        of the dist backward: declared coverage and ``sm_margin > 0``."""
        self._skip_unless_sm100()
        q_ranges = _contiguous_ranges(_Q_LENS)
        k_ranges = _contiguous_ranges(_K_LENS)
        attn_type_map = [_MASK_CYCLE[i % 4] for i in range(len(_Q_LENS))]
        q, k, v, out, lse, do, kwargs = self._inputs(
            num_head=8,
            num_head_kv=2,
            d=d,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=attn_type_map,
            seed=self.seed + d + len(variant),
        )
        num_sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        variant_kwargs: dict[str, Any] = {
            "packed": dict(bwd_pack_factor=4),
            "unpacked": dict(bwd_pack_factor=1),
            "packed_persistent": dict(
                bwd_pack_factor=4,
                declared_q_full_coverage=True,
                declared_k_full_coverage=True,
                sm_margin=num_sm - 8,
            ),
        }[variant]
        expected = (1, False) if variant == "unpacked" else (4, False)
        test_case = f"[test_sorted_q_dq_route][{d=}][{variant=}]"
        self._log_ran(test_case)
        dq_init = torch.randn(q.shape, device=q.device, dtype=torch.float32)
        dq_buf = dq_init.clone()
        with record_bwd_kernels() as built:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                dq=dq_buf,
                dq_type=torch.float32,
                dk_type=torch.float32,
                dv_type=torch.float32,
                disable_fwd_atomic_reduction=True,
                **variant_kwargs,
                **kwargs,
            )
        (bwd_kernel,) = built
        self._assert_kernel(bwd_kernel, test_case, fold=expected, d=d, direct=False)
        self.assertEqual(
            bwd_kernel.use_dense_dqacc_for_ranges, variant == "unpacked", test_case
        )
        self.assertEqual(bwd_kernel.dQ_rowmajor_accum, variant != "unpacked", test_case)
        self.assertEqual(
            bwd_kernel.is_persistent, variant == "packed_persistent", test_case
        )
        self.assertIs(dq, dq_buf, test_case)
        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out, dq - dq_init, dk, dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    def test_rejections(self):
        """Requests that no fold can serve fail before any bwd kernel is
        built, including a pack factor passed through ``flex_flash_attn_func``."""
        self._skip_unless_sm100()
        device, dtype = self.device, torch.bfloat16
        q_ranges = _contiguous_ranges([96, 160])
        ranges = torch.tensor(q_ranges, device=device, dtype=torch.int32)
        self._log_ran("[test_rejections]")

        def ranges_bwd(num_head: int, **kwargs) -> None:
            q = torch.randn(256, num_head, 128, device=device, dtype=dtype)
            k = torch.randn(256, 2, 128, device=device, dtype=dtype)
            lse = torch.zeros(256, num_head, device=device, dtype=torch.float32)
            _flex_flash_attn_bwd(
                q,
                k,
                k,
                q,
                lse,
                q,
                q_ranges=ranges,
                k_ranges=ranges,
                max_seqlen_q=160,
                max_seqlen_k=160,
                **kwargs,
            )

        with record_bwd_kernels() as built:
            with self.assertRaisesRegex(ValueError, "cat_gqa=True"):
                ranges_bwd(8, bwd_pack_factor=4, cat_gqa=True)
            with self.assertRaisesRegex(ValueError, "positive divisor"):
                ranges_bwd(8, bwd_pack_factor=3)
            # 3 q heads per kv head cannot split the 64 Q rows of a 2-CTA CTA
            # (d=128) into whole tokens.
            with self.assertRaisesRegex(NotImplementedError, "divide the Q rows"):
                ranges_bwd(6, bwd_pack_factor=3)
            # GQA without a fold that covers every q head has several dK/dV
            # writers per kv head.
            with self.assertRaisesRegex(AssertionError, "unique dK/dV writer"):
                ranges_bwd(8, disable_bwd_dkv_atomic_reduction=True)
            q = torch.randn(1, 256, 8, 128, device=device, dtype=dtype)
            k = torch.randn(1, 256, 2, 128, device=device, dtype=dtype)
            lse = torch.zeros(1, 8, 256, device=device, dtype=torch.float32)
            with self.assertRaisesRegex(NotImplementedError, "requires q/k ranges"):
                _flex_flash_attn_bwd(q, k, k, q, lse, q, bwd_pack_factor=4)
            # The autograd function hands bwd_pack_factor to the backward.
            q = torch.randn(256, 8, 64, device=device, dtype=dtype)
            k = torch.randn(256, 2, 64, device=device, dtype=dtype)
            q, k = (t.requires_grad_() for t in (q, k))
            out, _ = flex_flash_attn_func(
                q,
                k,
                k,
                q_ranges=ranges,
                k_ranges=ranges,
                max_seqlen_q=160,
                max_seqlen_k=160,
                bwd_pack_factor=3,
            )
            with self.assertRaisesRegex(ValueError, "positive divisor"):
                torch.autograd.grad(out, (q, k), torch.randn_like(out))
        self.assertEqual(built, [], "a rejected request built a bwd kernel")


class TestSelectBwdGqaFold(TestCase):
    """Rules of ``select_bwd_gqa_fold``; no device work."""

    def _select(self, **overrides) -> BwdGqaFold:
        args: dict[str, Any] = dict(
            bwd_pack_factor=1,
            cat_gqa=False,
            major_arch=10,
            has_ranges=True,
            has_block_sparse=False,
            has_mask_mod=False,
            has_score_mod=False,
            has_softcap=False,
            deterministic=False,
            num_head=8,
            num_head_kv=2,
            tile_m=128,
            cta_group_size=1,
        )
        args.update(overrides)
        return select_bwd_gqa_fold(**args)

    def test_fold_fields(self):
        unfolded = BwdGqaFold(8)
        self.assertEqual((unfolded.kv_group, unfolded.groups_per_tile), (8, 1))
        self.assertFalse(unfolded.covers_all_qheads)
        cat = BwdGqaFold(8, 1, head_loop=True)
        self.assertEqual((cat.kv_group, cat.groups_per_tile), (8, 8))
        self.assertTrue(cat.covers_all_qheads)
        partial = BwdGqaFold(8, 2, head_loop=True)
        self.assertEqual((partial.kv_group, partial.groups_per_tile), (4, 4))
        self.assertTrue(partial.covers_all_qheads)
        full = BwdGqaFold(8, 8)
        self.assertEqual((full.kv_group, full.groups_per_tile), (1, 1))
        self.assertTrue(full.covers_all_qheads)
        with self.assertRaises(ValueError):
            BwdGqaFold(8, 3)
        with self.assertRaises(ValueError):
            BwdGqaFold(4, 4, head_loop=True)
        with self.assertRaisesRegex(ValueError, "needs head_loop"):
            BwdGqaFold(8, 2)

    def test_pack_factor_one_keeps_the_unpacked_backward(self):
        """``bwd_pack_factor=1`` is unfolded, or the head loop under
        ``cat_gqa``, also where a pack factor above 1 is unsupported."""
        self.assertEqual(self._select(), BwdGqaFold(4))
        self.assertEqual(self._select(cat_gqa=True), BwdGqaFold(4, 1, head_loop=True))
        self.assertEqual(
            self._select(cat_gqa=True, has_ranges=False),
            BwdGqaFold(4, 1, head_loop=True),
        )
        for scope in (
            dict(major_arch=9),
            dict(has_ranges=False),
            dict(has_block_sparse=True),
            dict(has_mask_mod=True),
            dict(has_score_mod=True),
            dict(has_softcap=True),
            dict(deterministic=True),
        ):
            self.assertEqual(self._select(**scope), BwdGqaFold(4), str(scope))

    def test_pack_factor_builds_full_or_looped_pack(self):
        """``pf = qhead_per_kvhead`` is the full pack; a smaller pf walks the
        remaining packed groups in the head loop. Both cover every q head."""
        for num_head, pack_factor, expected in (
            (4, 2, BwdGqaFold(2, 2)),
            (8, 4, BwdGqaFold(4, 4)),
            (8, 2, BwdGqaFold(4, 2, head_loop=True)),
            (16, 8, BwdGqaFold(8, 8)),
            (16, 4, BwdGqaFold(8, 4, head_loop=True)),
            (16, 2, BwdGqaFold(8, 2, head_loop=True)),
            (12, 2, BwdGqaFold(6, 2, head_loop=True)),
            (24, 4, BwdGqaFold(12, 4, head_loop=True)),
        ):
            for cta_group_size in (1, 2):
                case = f"{num_head=} {pack_factor=} {cta_group_size=}"
                fold = self._select(
                    num_head=num_head,
                    bwd_pack_factor=pack_factor,
                    cta_group_size=cta_group_size,
                )
                self.assertEqual(fold, expected, case)
                self.assertTrue(fold.covers_all_qheads, case)
        self.assertEqual(
            self._select(bwd_pack_factor=4, major_arch=11), BwdGqaFold(4, 4)
        )

    def test_mha(self):
        """MHA has one q head per kv head: only ``bwd_pack_factor=1`` is
        valid, and ``cat_gqa`` is ignored."""
        self.assertEqual(self._select(num_head_kv=8), BwdGqaFold(1))
        self.assertEqual(self._select(num_head_kv=8, cat_gqa=True), BwdGqaFold(1))
        with self.assertRaisesRegex(ValueError, r"qhead_per_kvhead \(1\)"):
            self._select(num_head_kv=8, bwd_pack_factor=2)

    def test_invalid_pack_factor(self):
        """A pack factor that is not a positive divisor of qhead_per_kvhead,
        or one above 1 together with cat_gqa, raises ValueError."""
        for pack_factor in (0, -4, 3, 8):
            with self.assertRaisesRegex(
                ValueError,
                r"bwd_pack_factor \(-?\d+\) must be a positive divisor of "
                r"qhead_per_kvhead \(4\)",
                msg=f"{pack_factor=}",
            ):
                self._select(bwd_pack_factor=pack_factor)
        for pack_factor in (2, 4):
            with self.assertRaisesRegex(ValueError, "cat_gqa=True"):
                self._select(bwd_pack_factor=pack_factor, cat_gqa=True)

    def test_rows_per_cta_limit(self):
        """The pack factor must divide the Q rows of one CTA: 128 on the
        1-CTA kernel, 64 on the 2-CTA kernel."""
        self.assertEqual(
            self._select(num_head=128, num_head_kv=1, bwd_pack_factor=128),
            BwdGqaFold(128, 128),
        )
        self.assertEqual(
            self._select(
                num_head=64, num_head_kv=1, bwd_pack_factor=64, cta_group_size=2
            ),
            BwdGqaFold(64, 64),
        )
        self.assertEqual(
            self._select(
                num_head=128, num_head_kv=1, bwd_pack_factor=64, cta_group_size=2
            ),
            BwdGqaFold(128, 64, head_loop=True),
        )
        for overrides, rows in (
            (dict(num_head=256, num_head_kv=1, bwd_pack_factor=256), 128),
            (
                dict(
                    num_head=128, num_head_kv=1, bwd_pack_factor=128, cta_group_size=2
                ),
                64,
            ),
            (dict(num_head=6, bwd_pack_factor=3), 128),
            (dict(num_head=6, bwd_pack_factor=3, cta_group_size=2), 64),
        ):
            with self.assertRaisesRegex(
                NotImplementedError,
                rf"must divide the Q rows per CTA \({rows} ",
                msg=str(overrides),
            ):
                self._select(**overrides)


if __name__ == "__main__":
    run_tests()
