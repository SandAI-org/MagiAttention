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

import itertools
import unittest
from typing import Callable
from unittest import TestCase, mock

import torch

from magi_attention.common import AttnRanges
from magi_attention.meta.collection import calc_meta
from magi_attention.meta.collection.calc_meta import AttnArg, FA4AttnArg
from magi_attention.testing import assert_equal, parameterize
from magi_attention.testing.utils import switch_env, switch_envvar_context


class TestFA4AttnArg(TestCase):
    def test_hstu_conversion_trims_tail_and_preserves_empty_first_interval(self):
        arg = object.__new__(FA4AttnArg)
        arg.k_ranges = AttnRanges.from_ranges([[1, 4]])
        arg.seqlen_q, arg.seqlen_k = 2, 4
        arg.tile_m, arg.tile_m_bwd = 2, 4
        arg.ffa_fwd_args_dict = dict(q_ranges=None, k_ranges=None, attn_type_map=None)
        for source, expected_rows in (
            (torch.tensor([[0, 0], [1, 1], [4, 4], [0, 0]]), 3),
            (torch.zeros(3, 2, dtype=torch.int32), 1),
        ):
            with (
                self.subTest(expected_rows=expected_rows),
                mock.patch.object(
                    calc_meta, "magi_to_hstu_cuda", create=True
                ) as extension,
            ):
                extension.magi_to_hstu.return_value = source
                result = arg._make_hstu_func()
            self.assertEqual(arg.n_func, expected_rows)
            self.assertEqual(result.shape, (1, 1, expected_rows, 10))
            assert_equal(result[0, 0, :, :2], source[:expected_rows])
            self.assertEqual(extension.magi_to_hstu.call_args.kwargs["n_max_func"], 3)

    def test_hstu_interval_encoding(self):
        # First interval [0,F0), followed by [F1,F2); no compiler extension needed.
        func = torch.tensor([[2, 0], [4, 1], [6, 3]])
        expected = torch.tensor(
            [
                [True, True, False, False, True, True],
                [False, True, True, False, False, False],
            ]
        )
        assert_equal(FA4AttnArg.func_to_mask(func, 6), expected)


RANGE_MERGE_ENV = "MAGI_ATTENTION_RANGE_MERGE"


BWD_PACK_FACTOR_ENV = "MAGI_ATTENTION_BWD_PACK_FACTOR"


ROW_SETS: list[tuple[list[list[int]], list[list[int]]]] = [
    # shared k starts
    (
        [[256, 384], [128, 256], [0, 128]],
        [[256, 384], [0, 256], [0, 128]],
    ),
    # k rows form a partition and include an empty row
    (
        [[128, 256], [0, 128], [256, 384]],
        [[0, 128], [128, 128], [128, 256]],
    ),
    # k rows form a partition
    (
        [[0, 64], [64, 192], [192, 256], [256, 512]],
        [[256, 512], [0, 64], [192, 256], [64, 192]],
    ),
    # overlapping q rows
    (
        [[0, 128], [64, 192], [192, 256]],
        [[0, 64], [0, 64], [64, 128]],
    ),
    # holes in both q and k
    (
        [[0, 64], [128, 192]],
        [[0, 64], [128, 192]],
    ),
]


def _permute_rows(
    q_rows: list[list[int]], k_rows: list[list[int]]
) -> list[tuple[list[list[int]], list[list[int]]]]:
    return [
        ([q_rows[i] for i in order], [k_rows[i] for i in order])
        for order in itertools.permutations(range(len(q_rows)))
    ]


ROW_CASES = [
    row_case
    for q_rows, k_rows in ROW_SETS
    for row_case in _permute_rows(q_rows, k_rows)
]


def _make_attn_arg(
    q_rows: list[list[int]], k_rows: list[list[int]], range_merge: bool = True
) -> AttnArg:
    with switch_envvar_context(RANGE_MERGE_ENV, enable=range_merge):
        return AttnArg(
            q_ranges=AttnRanges.from_ranges(q_rows),
            k_ranges=AttnRanges.from_ranges(k_rows),
            attn_type_map=[0] * len(q_rows),
        )


def _end_if(ranges: AttnRanges, predicate: Callable[[int], bool]) -> int | None:
    return ranges.end if predicate(ranges.end) else None


def _fold_env(
    backend: str, cat_gqa: bool, pack_factor: int, range_merge: bool = False
) -> dict[str, str]:
    """Environment variables that decide AttnArg's direct dK/dV grant and the
    cutedsl backward GQA fold; the legacy backend toggles stay off so that
    MAGI_ATTENTION_KERNEL_BACKEND alone selects the backend."""
    return {
        "MAGI_ATTENTION_KERNEL_BACKEND": backend,
        "MAGI_ATTENTION_SDPA_BACKEND": "0",
        "MAGI_ATTENTION_FA4_BACKEND": "0",
        "MAGI_ATTENTION_CATGQA": "1" if cat_gqa else "0",
        BWD_PACK_FACTOR_ENV: str(pack_factor),
        RANGE_MERGE_ENV: "1" if range_merge else "0",
    }


class TestAttnArgBwdArgs(TestCase):
    """AttnArg's host range flags match the bwd range tensors it builds."""

    @parameterize("range_merge", [False, True])
    @parameterize("rows", ROW_CASES)
    def test_range_flags_match_tensors(self, range_merge: bool, rows):
        q_rows, k_rows = rows
        attn_arg = _make_attn_arg(q_rows, k_rows, range_merge)
        ffa_args = attn_arg.to_ffa_args(is_bwd=True)
        q_ranges = AttnRanges.from_ranges(ffa_args["q_ranges"].tolist())
        k_ranges = AttnRanges.from_ranges(ffa_args["k_ranges"].tolist())
        msg = f"{range_merge=} {q_rows=} {k_rows=}"

        self.assertEqual(attn_arg.bwd_args_q_sorted, q_ranges.is_sorted(), msg)
        self.assertEqual(
            attn_arg.bwd_args_q_coverage_end,
            _end_if(q_ranges, q_ranges.is_full_coverage),
            msg,
        )
        self.assertEqual(
            attn_arg.bwd_args_k_partition_end,
            _end_if(k_ranges, k_ranges.is_cu_seqlens),
            msg,
        )


class TestAttnArgFwdArgs(TestCase):
    """AttnArg's host range flags match the fwd range tensors it builds."""

    @parameterize("range_merge", [False, True])
    @parameterize("rows", ROW_CASES)
    def test_range_flags_match_tensors(self, range_merge: bool, rows):
        q_rows, k_rows = rows
        attn_arg = _make_attn_arg(q_rows, k_rows, range_merge)
        ffa_args = attn_arg.to_ffa_args(is_bwd=False)
        q_ranges = AttnRanges.from_ranges(ffa_args["q_ranges"].tolist())
        msg = f"{range_merge=} {q_rows=} {k_rows=}"

        self.assertEqual(attn_arg.fwd_args_q_sorted, q_ranges.is_sorted(), msg)
        self.assertEqual(
            attn_arg.fwd_args_q_coverage_end,
            _end_if(q_ranges, q_ranges.is_full_coverage),
            msg,
        )


class TestAttnArgDirectDkvGrant(TestCase):
    """AttnArg grants the direct dK/dV store (``disable_bwd_dkv_atomic_reduction``)
    to sorted, disjoint k ranges only under CatGQA, or under a bwd pack factor
    above 1 on the cutedsl backend, the only backend that reads
    MAGI_ATTENTION_BWD_PACK_FACTOR."""

    @staticmethod
    def _grant(
        q_rows: list[list[int]],
        k_rows: list[list[int]],
        *,
        backend: str,
        cat_gqa: bool = False,
        pack_factor: int = 1,
        range_merge: bool = False,
    ) -> bool:
        with switch_env(_fold_env(backend, cat_gqa, pack_factor, range_merge)):
            attn_arg = AttnArg(
                q_ranges=AttnRanges.from_ranges(q_rows),
                k_ranges=AttnRanges.from_ranges(k_rows),
                attn_type_map=[0] * len(q_rows),
            )
        return attn_arg.disable_bwd_dkv_atomic_reduction

    @parameterize("backend", ["cutedsl", "ffa", "sdpa", "sdpa_ol", "fa4"])
    def test_grant_by_backend(self, backend: str):
        rows = [[0, 128], [128, 256]]
        self.assertFalse(self._grant(rows, rows, backend=backend), backend)
        self.assertTrue(self._grant(rows, rows, backend=backend, cat_gqa=True), backend)
        self.assertEqual(
            self._grant(rows, rows, backend=backend, pack_factor=4),
            backend == "cutedsl",
            backend,
        )

    def test_pack_factor_keeps_the_k_range_conditions(self):
        q_rows = [[0, 128], [128, 256]]
        for k_rows in ([[0, 128], [64, 192]], [[128, 256], [0, 128]]):
            self.assertFalse(
                self._grant(q_rows, k_rows, backend="cutedsl", pack_factor=4),
                str(k_rows),
            )
        self.assertFalse(
            self._grant(
                q_rows, q_rows, backend="cutedsl", pack_factor=4, range_merge=True
            )
        )


if __name__ == "__main__":
    unittest.main()
