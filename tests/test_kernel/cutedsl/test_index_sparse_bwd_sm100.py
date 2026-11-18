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

"""
Can ``ffa_bwd_sm100_index`` replace the sm90 IndexSparse backward?

Replays ``tests/test_functional/test_index_sparse.py`` on SM100: same configs, same
data (seed, q/k/v, ``build_index_sparse_indices``), same SDPA reference and
tolerances. The backward runs ``ffa_bwd_sm100_index`` through
``ffa_bwd_sm100_index_sm90_layout``, which takes the sm90 tensors unchanged.
It consumes out / lse of the fp32 gather ``index_sparse_ref_fwd``, itself
checked against SDPA with the sm90 forward tolerance, so a failure here comes
from the backward alone. ``test_index_sparse_fwd_sm100.py`` runs the SM100
forward through the same runner.

Every case ends as one of:
  PASS — dq / dk / dv match SDPA within the sm90 tolerances
  GAP  — the case needs an ``Sm100IndexGap`` (nothing is launched)
  FAIL — the comparison failed or the kernel raised

Each test method runs all of its cases, prints the table, then fails on any
FAIL. GAP cases are printed and do not fail the run: SM100 still lacks
``head_dim != 128`` and ``sparse_k_block_size > 1`` (see the TODO in
``run_index_sparse_sweep``), and CI only gates the supported subset.

Structure:
  TestIndexSparseBwdSm100Sweep              — sm90 Classic sweep: q_seqlen × kv_seqlen × topk
  TestIndexSparseBwdSm100ComprehensiveSweep — sm90 Comprehensive sweep: head_config × head_dim × kbs
  TestIndexSparseBwdSm100Geometry           — the adapter's multi-kv-head refold, q-head groups
                                              whose q blocks straddle the 128-row Q tile, q
                                              blocks of several tokens (sm90 q_block_size > 1)
  TestIndexSparseBwdSm100Options            — both V gather handoffs (gather_v_async), K-chunked
                                              passes (num_k_passes) and their defaults
  TestIndexSparseBwdSm100Comparison         — the case runner rejects a shifted dK

Not ported from the sm90 test:
  - inner_dir / inner_load_mode / inner_store_mode / swap_bwd_qk_loop: sm90
    kernel knobs that do not change the function computed; SM100 has one kernel.
  - the deterministic check: it never runs in the sm90 test (its condition
    ``swap_bwd_qk_loop is not True`` fails for every case that does not also
    set ``check_deterministic=False``).
  - TestIndexSparseViewTrick: forward-only checks of a user-side layout trick;
    every view-trick config there has 4 q heads per kv head after folding,
    which the comprehensive MQA4 / GQA cases already cover.

Run (from the repo root):
    pytest tests/test_kernel/cutedsl/test_index_sparse_bwd_sm100.py -s
"""

import itertools
import unittest
from functools import partial
from typing import Any

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.ffa_bwd_sm100_index import (
    gather_v_async_default,
    num_k_passes_default,
)
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.index_sparse import (
    BwdGrads,
    ffa_bwd_sm100_index_sm90_layout,
    run_index_sparse_config,
    run_index_sparse_sweep,
)
from magi_attention.testing.sparse import SEED
from magi_attention.utils.arch import is_sm

require_blackwell = unittest.skipUnless(is_sm(100, 103), "requires Blackwell SM100/103")


# lse of a forward under test against the fp32 gather: both accumulate in fp32
# from the same 16-bit inputs and differ only by the exp2 / log2 approximations.


# ═══════════════════════════════════════════════════════════
# Case runner
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# Sweep collection and report
# ═══════════════════════════════════════════════════════════


# TODO: Sm100IndexGap cases are printed above and do not fail CI.
# HEAD_DIM: FFAFwdSm100Index and FFABwdSm100Index fix head_dim at 128.
# SPARSE_K_BLOCK: the kernels gather single K tokens, not K blocks
# (sparse_k_block_size > 1).
# Q_TILE_FILL (forward only): one index row must fill one 128-row Q tile.
# Drop a Sm100IndexGap member once the kernels serve it so those cases
# compare numerically instead of reporting GAP.


# ═══════════════════════════════════════════════════════════
# TestIndexSparseBwdSm100Sweep — sm90 Classic sweep
# ═══════════════════════════════════════════════════════════


@require_blackwell
class TestIndexSparseBwdSm100Sweep(DistTestBase):
    """sm90 Classic sweep: q_seqlen × kv_seqlen × topk, MQA128, D=128, kbs=1."""

    Q_SEQLENS = [512, 1000, 8192]
    KV_SEQLENS = [512, 1000, 8192]
    TOPKS = [128, 256]

    @property
    def seed(self):
        return SEED

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    @with_run_in_mp
    def test_index_sparse_bwd_sm100_classic(self):
        cases = []
        for q_seqlen, kv_seqlen, topk in itertools.product(
            self.Q_SEQLENS, self.KV_SEQLENS, self.TOPKS
        ):
            if topk > kv_seqlen:
                continue
            config = {
                "B": 1,
                "S_q": q_seqlen,
                "S_kv": kv_seqlen,
                "NHQ": 128,
                "NHK": 1,
                "D": 128,
                "topk": topk,
            }
            cases.append((f"q{q_seqlen}_kv{kv_seqlen}_topk{topk}", config))
        run_index_sparse_sweep(self.device, "classic", cases)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseBwdSm100ComprehensiveSweep — sm90 Comprehensive sweep
# ═══════════════════════════════════════════════════════════


@require_blackwell
class TestIndexSparseBwdSm100ComprehensiveSweep(DistTestBase):
    """sm90 Comprehensive sweep: head_config × head_dim × kbs.

    The sm90 inner_* / LoopQ-LoopK axes are dropped (see module docstring),
    so each remaining combination runs once.
    """

    _PARAM_SPACE: dict[str, list] = dict(
        head_config=[
            (128, 1, True),  # MQA128
            (4, 1, True),  # MQA4
            (128, 2, True),  # GQA 128:2
            (32, 4, True),  # GQA 32:4
            (4, 4, True),  # MHA4
            (32, 32, True),  # MHA32
        ],
        head_dim=[64, 128],
        kbs=[1, 8, 128],
    )

    @classmethod
    def _iter_cases(cls):
        for head_config, hd, kbs in itertools.product(
            cls._PARAM_SPACE["head_config"],
            cls._PARAM_SPACE["head_dim"],
            cls._PARAM_SPACE["kbs"],
        ):
            nhq, nhk, pack_gqa = head_config
            # Same validity rule as the sm90 sweep: kbs > 1 only for NHK=1, PackGQA, D=128.
            if kbs > 1 and (nhk > 1 or not pack_gqa or hd != 128):
                continue
            if kbs <= 1:
                S, topk = 256, 128
            else:
                S = 1024
                topk = max(2, 128 // kbs)
            config: dict[str, Any] = {
                "B": 1,
                "S": S,
                "NHQ": nhq,
                "NHK": nhk,
                "D": hd,
                "topk": topk,
                "sparse_k_block_size": kbs,
            }
            if kbs > 1:
                config["max_topk"] = topk
            yield f"nhq{nhq}_nhk{nhk}_hd{hd}_kbs{kbs}", config

    @property
    def seed(self):
        return SEED

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    @with_run_in_mp
    def test_index_sparse_bwd_sm100_comprehensive(self):
        run_index_sparse_sweep(self.device, "comprehensive", list(self._iter_cases()))


# ═══════════════════════════════════════════════════════════
# TestIndexSparseBwdSm100Geometry — q / kv head layouts the sweeps do not reach
# ═══════════════════════════════════════════════════════════


@require_blackwell
class TestIndexSparseBwdSm100Geometry(unittest.TestCase):
    """The adapter's multi-kv-head refold and q blocks that do not fill one Q tile exactly."""

    @property
    def device(self):
        return torch.cuda.current_device()

    def test_refold_keeps_multiple_kv_heads(self):
        """num_heads_kv = 2 with 128 q heads each: the refold goes through a copy."""
        config = {
            "B": 1,
            "S": 256,
            "NHQ": 256,
            "NHK": 2,
            "D": 128,
            "topk": 128,
            "keep_kv_heads": True,
        }
        run_index_sparse_config(self.device, config)

    def test_refold_small_group_keeps_multiple_kv_heads(self):
        """num_heads_kv = 2 with 4 q heads each: 32 q blocks share one Q tile, via a copy."""
        config = {
            "B": 1,
            "S": 256,
            "NHQ": 8,
            "NHK": 2,
            "D": 128,
            "topk": 128,
            "keep_kv_heads": True,
        }
        run_index_sparse_config(self.device, config)

    def test_q_head_group_straddles_q_tile(self):
        """Groups that neither divide nor are multiples of 128: q blocks cross tile edges."""
        for nhq in (48, 192):
            with self.subTest(nhq=nhq):
                config = {
                    "B": 1,
                    "S": 256,
                    "NHQ": nhq,
                    "NHK": 1,
                    "D": 128,
                    "topk": 128,
                }
                run_index_sparse_config(self.device, config)

    def test_q_block_of_tokens_shares_indices(self):
        """sm90 q_block_size > 1: q_block_size * group rows share one index row.

        16 x 8 fills one 128-row Q tile exactly; 4 x 8 puts four q blocks in a tile.
        """
        for q_block_size, nhq, nhk, keep_kv_heads in (
            (16, 8, 1, False),
            (16, 16, 2, True),
            (4, 8, 1, False),
        ):
            with self.subTest(q_block_size=q_block_size, nhq=nhq, nhk=nhk):
                config = {
                    "B": 1,
                    "S": 256,
                    "NHQ": nhq,
                    "NHK": nhk,
                    "D": 128,
                    "topk": 128,
                    "q_block_size": q_block_size,
                    "keep_kv_heads": keep_kv_heads,
                }
                run_index_sparse_config(self.device, config)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseBwdSm100Options — gather_v_async and num_k_passes
# ═══════════════════════════════════════════════════════════


def _bwd_with_sorted_rows(q, k, v, out, do, lse, index_sparse_indices, **kwargs):
    """Sort every row's valid ids ascending (attention does not depend on their
    order), keeping the -1 tail, and pass the rows as sorted."""
    pad = torch.iinfo(torch.int32).max
    rows = torch.where(index_sparse_indices >= 0, index_sparse_indices, pad)
    rows = rows.sort(dim=-1).values
    rows = torch.where(rows == pad, -1, rows).int()
    return ffa_bwd_sm100_index_sm90_layout(
        q, k, v, out, do, lse, rows, index_sparse_rows_sorted=True, **kwargs
    )


@require_blackwell
class TestIndexSparseBwdSm100Options(unittest.TestCase):
    """``gather_v_async`` and ``num_k_passes``: each path against the reference, and their defaults."""

    @property
    def device(self):
        return torch.cuda.current_device()

    def test_gather_v_async_both_paths_match_reference(self):
        """Both V gather handoffs are exact, whichever the footprint would pick.

        The sweeps above are all small enough for the default to keep the
        synchronous V gather, so force each path here: one q block per Q tile,
        several q blocks per Q tile (segments), and -1 padded index rows whose
        valid length is not a multiple of the 64-wide K tile.
        """
        configs = {
            "mqa128": {"B": 1, "S": 256, "NHQ": 128, "NHK": 1, "D": 128, "topk": 128},
            "mqa4_segments": {
                "B": 1,
                "S": 256,
                "NHQ": 4,
                "NHK": 1,
                "D": 128,
                "topk": 128,
            },
            "mqa128_ragged": {
                "B": 2,
                "S_q": 256,
                "S_kv": 1000,
                "NHQ": 128,
                "NHK": 1,
                "D": 128,
                "topk": [64, 200],
                "max_topk": 256,
            },
        }
        for gather_v_async in (True, False):
            for name, config in configs.items():
                with self.subTest(gather_v_async=gather_v_async, config=name):
                    run_index_sparse_config(
                        self.device,
                        config,
                        bwd_fn=partial(
                            ffa_bwd_sm100_index_sm90_layout,
                            gather_v_async=gather_v_async,
                        ),
                    )

    def test_gather_v_async_default_follows_l2_footprint(self):
        """The default is on only while K/V rows + fp32 dK/dV rows of all kv heads span (L2/2, 4x L2]."""
        l2_bytes = 126 * 2**20
        # bf16 K/V rows + fp32 dK/dV rows at head_dim 128: 1.5 KiB per (token, kv head)
        row_bytes = 128 * (2 * 2 + 2 * 4)
        lower_rows = l2_bytes // 2 // row_bytes
        upper_rows = 4 * l2_bytes // row_bytes
        self.assertFalse(
            gather_v_async_default(lower_rows, 1, 128, 2, l2_bytes),
            "exactly half of L2 still keeps the synchronous V gather",
        )
        self.assertTrue(gather_v_async_default(lower_rows + 1, 1, 128, 2, l2_bytes))
        self.assertTrue(
            gather_v_async_default(upper_rows, 1, 128, 2, l2_bytes),
            "exactly 4x L2 still takes the async V gather",
        )
        self.assertFalse(gather_v_async_default(upper_rows + 1, 1, 128, 2, l2_bytes))
        # one kv head: 32k tokens fit half of L2, 64k-256k do not, 512k is past 4x L2
        self.assertFalse(gather_v_async_default(32768, 1, 128, 2, l2_bytes))
        self.assertTrue(gather_v_async_default(65536, 1, 128, 2, l2_bytes))
        self.assertTrue(gather_v_async_default(262144, 1, 128, 2, l2_bytes))
        self.assertFalse(gather_v_async_default(524288, 1, 128, 2, l2_bytes))
        # kv heads add up
        self.assertTrue(gather_v_async_default(32768, 2, 128, 2, l2_bytes))
        self.assertFalse(gather_v_async_default(262144, 2, 128, 2, l2_bytes))

    def test_k_chunked_passes_match_reference(self):
        """Splitting K into passes sums dQ across them in fp32 and keeps dK/dV exact.

        Pass counts that leave whole and ragged chunks, rows whose ids miss some
        chunks (topk 128 of 256 tokens in 3 chunks), several q blocks per Q tile,
        -1 padded rows and two kv heads kept apart.
        """
        configs = {
            "mqa128": {"B": 1, "S": 256, "NHQ": 128, "NHK": 1, "D": 128, "topk": 128},
            "mqa4_segments": {
                "B": 1,
                "S": 256,
                "NHQ": 4,
                "NHK": 1,
                "D": 128,
                "topk": 128,
            },
            "mqa128_ragged": {
                "B": 2,
                "S_q": 256,
                "S_kv": 1000,
                "NHQ": 128,
                "NHK": 1,
                "D": 128,
                "topk": [64, 200],
                "max_topk": 256,
            },
            "gqa_two_kv_heads": {
                "B": 1,
                "S": 256,
                "NHQ": 8,
                "NHK": 2,
                "D": 128,
                "topk": 128,
                "keep_kv_heads": True,
            },
        }
        for num_k_passes in (2, 3):
            for rows_sorted in (False, True):
                bwd_fn = partial(
                    (
                        _bwd_with_sorted_rows
                        if rows_sorted
                        else ffa_bwd_sm100_index_sm90_layout
                    ),
                    num_k_passes=num_k_passes,
                )
                for name, config in configs.items():
                    with self.subTest(
                        num_k_passes=num_k_passes, rows_sorted=rows_sorted, config=name
                    ):
                        run_index_sparse_config(self.device, config, bwd_fn=bwd_fn)

    def test_num_k_passes_default_follows_l2_footprint_and_topk(self):
        """Passes start past 2x L2, aim at 0.8x L2 chunks, and need 256 ids per row and pass."""
        l2_bytes = 126 * 2**20
        # bf16 K/V rows + fp32 dK/dV rows at head_dim 128: 1.5 KiB per (token, kv head)
        row_bytes = 128 * (2 * 2 + 2 * 4)
        two_l2_rows = 2 * l2_bytes // row_bytes
        self.assertEqual(
            num_k_passes_default(two_l2_rows, 1, 128, 2, 2048, l2_bytes), 1
        )
        self.assertEqual(
            num_k_passes_default(two_l2_rows + 1, 1, 128, 2, 2048, l2_bytes), 3
        )
        # one kv head: 128k tokens stay single pass, 256k / 512k take 4 / 8 passes
        self.assertEqual(num_k_passes_default(131072, 1, 128, 2, 2048, l2_bytes), 1)
        self.assertEqual(num_k_passes_default(262144, 1, 128, 2, 2048, l2_bytes), 4)
        self.assertEqual(num_k_passes_default(524288, 1, 128, 2, 2048, l2_bytes), 8)
        # kv heads add up
        self.assertEqual(num_k_passes_default(262144, 2, 128, 2, 4096, l2_bytes), 8)
        # fewer than 256 ids per row and pass: the passes would not pay off
        self.assertEqual(num_k_passes_default(524288, 1, 128, 2, 1024, l2_bytes), 1)
        self.assertEqual(num_k_passes_default(262144, 1, 128, 2, 1024, l2_bytes), 4)
        # at most 16 passes
        self.assertEqual(num_k_passes_default(2**21, 1, 128, 2, 8192, l2_bytes), 16)

    def test_num_k_passes_default_sorted_rows_take_half_l2_chunks(self):
        """Ascending rows: same 2x L2 start, 0.5x L2 chunks, same caps."""
        l2_bytes = 126 * 2**20
        two_l2_rows = 2 * l2_bytes // (128 * (2 * 2 + 2 * 4))
        passes = partial(num_k_passes_default, rows_sorted=True)
        self.assertEqual(passes(two_l2_rows, 1, 128, 2, 2048, l2_bytes), 1)
        self.assertEqual(passes(two_l2_rows + 1, 1, 128, 2, 2048, l2_bytes), 5)
        # one kv head: 192k / 256k / 384k tokens take 5 / 7 / 10 passes
        self.assertEqual(passes(196608, 1, 128, 2, 39296, l2_bytes), 5)
        self.assertEqual(passes(262144, 1, 128, 2, 52480, l2_bytes), 7)
        self.assertEqual(passes(393216, 1, 128, 2, 39296, l2_bytes), 10)
        # fewer than 256 ids per row and pass
        self.assertEqual(passes(393216, 1, 128, 2, 2048, l2_bytes), 1)
        self.assertEqual(passes(786432, 1, 128, 2, 78592, l2_bytes), 16)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseBwdSm100Comparison — the case runner rejects wrong gradients
# ═══════════════════════════════════════════════════════════


@require_blackwell
class TestIndexSparseBwdSm100Comparison(unittest.TestCase):
    """A PASS of ``run_index_sparse_config`` needs the right gradients: a shifted dK fails."""

    @property
    def device(self):
        return torch.cuda.current_device()

    def test_comparison_rejects_dk_on_wrong_rows(self):
        """A PASS must mean something: dK scattered one K row off has to fail."""

        def shifted_dk_bwd(*args, **kwargs) -> BwdGrads:
            dq, dk, dv = ffa_bwd_sm100_index_sm90_layout(*args, **kwargs)
            return dq, dk.roll(1, dims=0), dv

        config = {"B": 1, "S": 256, "NHQ": 128, "NHK": 1, "D": 128, "topk": 128}
        with self.assertRaises(AssertionError) as ctx:
            run_index_sparse_config(self.device, config, bwd_fn=shifted_dk_bwd)
        self.assertIn("=> dk", str(ctx.exception))


if __name__ == "__main__":
    run_tests()
