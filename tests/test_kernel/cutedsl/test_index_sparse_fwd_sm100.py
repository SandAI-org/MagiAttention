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
import math
import unittest
from functools import partial
from typing import Any

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_fwd_sm100_index import (
    FFAFwdSm100Index,
    ffa_fwd_sm100_index,
)
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.index_sparse import (
    Sm100IndexGap,
    Sm100IndexUnsupportedError,
    ffa_fwd_sm100_index_sm90_layout,
    index_sparse_ref_fwd,
)
from magi_attention.utils import set_random_seed
from magi_attention.utils.arch import is_sm10x
from tests.test_attn.sparse_test_utils import SEED
from tests.test_kernel.cutedsl.test_index_sparse_bwd_sm100 import (
    FwdOutLse,
    _run_index_sparse_config,
    _run_sweep,
)

# Forward only: every test of this file passes bwd_fn=None.
_run_fwd_config = partial(
    _run_index_sparse_config, fwd_fn=ffa_fwd_sm100_index_sm90_layout, bwd_fn=None
)
_run_fwd_sweep = partial(
    _run_sweep, fwd_fn=ffa_fwd_sm100_index_sm90_layout, bwd_fn=None
)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseFwdSm100Sweep — sm90 Classic sweep
# ═══════════════════════════════════════════════════════════


@unittest.skipUnless(is_sm10x(), "ffa_fwd_sm100_index needs an SM100 GPU")
class TestIndexSparseFwdSm100Sweep(DistTestBase):
    """sm90 Classic sweep: q_seqlen × kv_seqlen × topk, MQA128, D=128, kbs=1.

    Replays the configs of ``tests/test_attn/test_index_sparse.py`` through
    ``ffa_fwd_sm100_index_sm90_layout``: out against SDPA with the sm90 forward
    tolerance, lse against the fp32 gather ``index_sparse_ref_fwd``.
    """

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
    def test_index_sparse_fwd_sm100_classic(self):
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
        _run_fwd_sweep(self.device, "fwd classic", cases)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseFwdSm100ComprehensiveSweep — sm90 Comprehensive sweep
# ═══════════════════════════════════════════════════════════


@unittest.skipUnless(is_sm10x(), "ffa_fwd_sm100_index needs an SM100 GPU")
class TestIndexSparseFwdSm100ComprehensiveSweep(DistTestBase):
    """sm90 Comprehensive sweep: head_config × head_dim × kbs.

    Every case keeps its kv heads and uses ``q_block_size = 128 / G``, so one
    index row fills one 128-row Q tile, as ``ffa_fwd_sm100_index`` requires.
    The sm90 sweep flattens kv heads with ``q_block_size = 1``, which fills the
    tile for MQA128 only.
    """

    _PARAM_SPACE: dict[str, list] = dict(
        head_config=[
            (128, 1),  # MQA128
            (4, 1),  # MQA4
            (128, 2),  # GQA 128:2
            (32, 4),  # GQA 32:4
            (4, 4),  # MHA4
            (32, 32),  # MHA32
        ],
        head_dim=[64, 128],
        kbs=[1, 8, 128],
    )

    @classmethod
    def _iter_cases(cls):
        for (nhq, nhk), hd, kbs in itertools.product(
            cls._PARAM_SPACE["head_config"],
            cls._PARAM_SPACE["head_dim"],
            cls._PARAM_SPACE["kbs"],
        ):
            # Same validity rule as the sm90 sweep: kbs > 1 only for NHK=1, D=128.
            if kbs > 1 and (nhk > 1 or hd != 128):
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
                "q_block_size": FFAFwdSm100Index.tile_m // (nhq // nhk),
                "keep_kv_heads": True,
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
    def test_index_sparse_fwd_sm100_comprehensive(self):
        _run_fwd_sweep(self.device, "fwd comprehensive", list(self._iter_cases()))


# ═══════════════════════════════════════════════════════════
# TestIndexSparseFwdSm100Adapter — the sm90 layout adapter
# ═══════════════════════════════════════════════════════════


@unittest.skipUnless(is_sm10x(), "ffa_fwd_sm100_index needs an SM100 GPU")
class TestIndexSparseFwdSm100Adapter(unittest.TestCase):
    """``ffa_fwd_sm100_index_sm90_layout``: topk padding, the all-valid path and gap reporting."""

    @property
    def device(self):
        return torch.cuda.current_device()

    def test_topk_not_multiple_of_align_is_padded(self):
        """sm90 topk 130: the adapter appends -1 columns up to FFAFwdSm100Index.topk_align."""
        config = {"B": 1, "S": 256, "NHQ": 128, "NHK": 1, "D": 128, "topk": 130}
        _run_fwd_config(self.device, config)

    def test_assume_valid_ids_selects_fast_path(self):
        """Index rows without -1 and topk a multiple of 2 * tile_n: the all-valid kernel runs."""
        config = {
            "B": 1,
            "S": 512,
            "NHQ": 8,
            "NHK": 1,
            "D": 128,
            "topk": 2 * FFAFwdSm100Index.tile_n,
            "q_block_size": 16,
        }
        _run_index_sparse_config(
            self.device,
            config,
            fwd_fn=partial(ffa_fwd_sm100_index_sm90_layout, assume_valid_ids=True),
            bwd_fn=None,
        )
        # compile key: (dtype, group, capability, debug, ids_all_valid)
        keys = ffa_fwd_sm100_index.compile_cache.cache.keys()  # type: ignore[attr-defined]
        self.assertTrue(any(key[1] == 8 and key[-1] is True for key in keys))

    def test_assume_valid_ids_needs_aligned_topk(self):
        """Aligning topk would append -1 ids, which assume_valid_ids rules out."""
        device = self.device
        q = torch.randn(16, 8, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(64, 1, 128, device=device, dtype=torch.bfloat16)
        idx = torch.arange(126, device=device, dtype=torch.int32).view(1, 1, 126) % 64
        with self.assertRaises(AssertionError) as ctx:
            ffa_fwd_sm100_index_sm90_layout(
                q, k, k, idx, q_block_size=16, assume_valid_ids=True
            )
        self.assertIn("assume_valid_ids needs topk", str(ctx.exception))

    def test_q_tile_fill_gap_raises_before_launch(self):
        """MQA4 with q_block_size 1 fills 4 of 128 Q tile rows: a Q_TILE_FILL gap."""
        config = {"B": 1, "S": 256, "NHQ": 4, "NHK": 1, "D": 128, "topk": 128}
        with self.assertRaises(Sm100IndexUnsupportedError) as ctx:
            _run_fwd_config(self.device, config)
        self.assertEqual(ctx.exception.gaps, (Sm100IndexGap.Q_TILE_FILL,))


# ═══════════════════════════════════════════════════════════
# TestIndexSparseFwdSm100Comparison — the case runner rejects wrong outputs
# ═══════════════════════════════════════════════════════════


@unittest.skipUnless(is_sm10x(), "ffa_fwd_sm100_index needs an SM100 GPU")
class TestIndexSparseFwdSm100Comparison(unittest.TestCase):
    """A PASS of ``_run_index_sparse_config`` needs the right out and lse: shifted ones fail."""

    @property
    def device(self):
        return torch.cuda.current_device()

    def test_comparison_rejects_out_on_wrong_head(self):
        """A PASS must mean something: out moved to the next q head has to fail."""

        def shifted_out_fwd(*args, **kwargs) -> FwdOutLse:
            out, lse = ffa_fwd_sm100_index_sm90_layout(*args, **kwargs)
            return out.roll(1, dims=1), lse

        config = {
            "B": 1,
            "S": 256,
            "NHQ": 8,
            "NHK": 1,
            "D": 128,
            "topk": 128,
            "q_block_size": 16,
        }
        with self.assertRaises(AssertionError) as ctx:
            _run_index_sparse_config(
                self.device, config, fwd_fn=shifted_out_fwd, bwd_fn=None
            )
        self.assertIn("=> fwd_out", str(ctx.exception))

    def test_comparison_rejects_lse_on_wrong_head(self):
        """A PASS must mean something: lse moved to the next q head has to fail."""

        def shifted_lse_fwd(*args, **kwargs) -> FwdOutLse:
            out, lse = ffa_fwd_sm100_index_sm90_layout(*args, **kwargs)
            return out, lse.roll(1, dims=1)

        config = {
            "B": 1,
            "S": 256,
            "NHQ": 8,
            "NHK": 1,
            "D": 128,
            "topk": 128,
            "q_block_size": 16,
        }
        with self.assertRaises(AssertionError) as ctx:
            _run_index_sparse_config(
                self.device, config, fwd_fn=shifted_lse_fwd, bwd_fn=None
            )
        self.assertIn("=> lse", str(ctx.exception))


# ═══════════════════════════════════════════════════════════
# TestIndexSparseFwdSm100Numerics — kernel inputs the sm90 builder never makes
# ═══════════════════════════════════════════════════════════


def _random_indices(
    num_q_blocks: int, num_heads_kv: int, topk: int, total_k: int, device
) -> torch.Tensor:
    """Sorted distinct ids per row, each row truncated to a random length (tail -1)."""
    rows = []
    for _ in range(num_q_blocks * num_heads_kv):
        rows.append(torch.randperm(total_k, device=device)[:topk].sort().values)
    idx = torch.stack(rows).view(num_q_blocks, num_heads_kv, topk).int()
    lens = torch.randint(1, topk + 1, (num_q_blocks, num_heads_kv), device=device)
    pos = torch.arange(topk, device=device)[None, None, :]
    idx[pos >= lens[..., None]] = -1
    return idx


@unittest.skipUnless(is_sm10x(), "ffa_fwd_sm100_index needs an SM100 GPU")
class TestIndexSparseFwdSm100Numerics(unittest.TestCase):
    """``ffa_fwd_sm100_index`` called directly against the fp32 gather reference.

    Covers the kernel's index contract beyond what the sm90 builder produces:
    -1 padding of random length, rows and whole indices without a valid id,
    out-of-range ids, repeated ids, topk around tile boundaries, a ragged Q
    tail, preallocated outputs and compile-cache reuse across shapes.
    """

    OUT_ATOL = 2e-2
    LSE_ATOL = 1e-3

    def setUp(self):
        super().setUp()
        set_random_seed(SEED)

    def _check(
        self,
        tq,
        tk,
        nhq,
        nhk,
        topk,
        dtype,
        *,
        idx=None,
        preallocate=False,
        softmax_scale=None,
    ):
        set_random_seed(SEED)
        device = torch.cuda.current_device()
        group = nhq // nhk
        q_block_size = FFAFwdSm100Index.tile_m // group
        num_q_blocks = -(-tq // q_block_size)
        q = torch.randn(tq, nhq, 128, device=device, dtype=dtype)
        k = torch.randn(tk, nhk, 128, device=device, dtype=dtype)
        v = torch.randn(tk, nhk, 128, device=device, dtype=dtype)
        if idx is None:
            idx = _random_indices(num_q_blocks, nhk, topk, tk, device)
        scale = 1.0 / math.sqrt(128) if softmax_scale is None else softmax_scale
        # Every id outside [0, tk) is excluded from attention.
        ref_indices = idx.masked_fill((idx < 0) | (idx >= tk), -1)
        ref_out, ref_lse = index_sparse_ref_fwd(
            q, k, v, ref_indices, q_block_size=q_block_size, softmax_scale=scale
        )
        out_buffer = torch.full_like(q, float("nan")) if preallocate else None
        lse_buffer = torch.full_like(ref_lse, float("nan")) if preallocate else None
        out, lse = ffa_fwd_sm100_index(
            q,
            k,
            v,
            idx,
            softmax_scale=scale,
            q_block_size=q_block_size,
            out=out_buffer,
            lse=lse_buffer,
        )
        torch.cuda.synchronize()
        if preallocate:
            self.assertIs(out, out_buffer)
            self.assertIs(lse, lse_buffer)
        finite = torch.isfinite(ref_lse)
        case = f"tq={tq} tk={tk} nhq={nhq} nhk={nhk} topk={topk} {dtype}"
        self.assertTrue(
            torch.equal(torch.isneginf(lse), ~finite), f"{case}: empty-row lse"
        )
        self.assertTrue(
            bool((out.float()[~finite] == 0).all()), f"{case}: empty-row out"
        )
        if finite.any():
            err_out = (out.float() - ref_out.float()).abs()[finite].max().item()
            err_lse = (lse - ref_lse)[finite].abs().max().item()
            self.assertLess(err_out, self.OUT_ATOL, f"{case}: out err {err_out}")
            self.assertLess(err_lse, self.LSE_ATOL, f"{case}: lse err {err_lse}")

    @parameterize("shape", [(1024, 1024, 256), (1000, 1500, 200), (4096, 4096, 2048)])
    def test_gqa8_bf16(self, shape):
        tq, tk, topk = shape
        self._check(tq, tk, 64, 8, topk, torch.bfloat16)

    def test_gqa8_fp16_single_kv_head(self):
        self._check(2048, 4096, 8, 1, 512, torch.float16)

    def test_mha(self):
        """G = 1: q_block_size 128, one index row per Q tile of 128 tokens."""
        self._check(512, 512, 16, 16, 128, torch.bfloat16)

    def test_ragged_q_tail(self):
        """total_q not a multiple of the Q tile: the tail tile is masked."""
        self._check(16 * 7 + 5, 777, 32, 4, 96, torch.bfloat16)

    def test_rows_without_valid_ids(self):
        device = torch.cuda.current_device()
        tq, tk, nhq, nhk, topk = 512, 1024, 16, 2, 128
        idx = _random_indices(tq // 16, nhk, topk, tk, device)
        idx[0, :, :] = -1
        idx[7, 0, :] = -1
        self._check(tq, tk, nhq, nhk, topk, torch.bfloat16, idx=idx)

    @parameterize("topk", [4, 128, 260])
    def test_all_rows_without_valid_ids(self, topk):
        idx = torch.full(
            (4, 2, topk), -1, device=torch.cuda.current_device(), dtype=torch.int32
        )
        self._check(53, 521, 16, 2, topk, torch.bfloat16, idx=idx, preallocate=True)

    @parameterize("dtype", [torch.bfloat16, torch.float16])
    @parameterize("topk", [4, 124, 128, 132, 256, 260, 384])
    def test_topk_tile_boundaries(self, dtype, topk):
        idx = (
            torch.arange(topk, device=torch.cuda.current_device(), dtype=torch.int32)
            .expand(3, 2, topk)
            .clone()
        )
        idx[..., -1] = -1
        self._check(37, 521, 16, 2, topk, dtype, idx=idx)

    def test_out_of_range_ids_are_masked(self):
        tk, topk = 521, 260
        idx = (
            torch.arange(topk, device=torch.cuda.current_device(), dtype=torch.int32)
            .expand(3, 2, topk)
            .clone()
        )
        idx[..., -8:-4] = tk
        idx[..., -4:] = -1
        idx[0, 0] = tk + 1
        self._check(37, tk, 16, 2, topk, torch.bfloat16, idx=idx)

    def test_duplicate_ids_are_attended_twice(self):
        """Each occurrence of a K token contributes independently to attention."""
        device = torch.cuda.current_device()
        tq, tk, nhq, nhk, topk = 256, 256, 8, 1, 64
        idx = _random_indices(tq // 16, nhk, topk, tk, device)
        idx[:, :, 1] = idx[:, :, 0]
        self._check(tq, tk, nhq, nhk, topk, torch.bfloat16, idx=idx)

    @parameterize("dtype", [torch.bfloat16, torch.float16])
    def test_preallocated_outputs(self, dtype):
        self._check(37, 521, 16, 2, 260, dtype, preallocate=True)

    @parameterize("group", [1, 4, 8, 128])
    def test_cache_reuse_with_dynamic_shapes(self, group):
        cache = ffa_fwd_sm100_index.compile_cache  # type: ignore[attr-defined]
        self.assertIsInstance(cache, JITCache)
        self._check(129, 257, group, 1, 4, torch.bfloat16)
        compiled = cache.cache.copy()
        self._check(67, 521, group * 2, 2, 260, torch.bfloat16, softmax_scale=0.125)
        self.assertEqual(compiled.keys(), cache.cache.keys())
        for key, fn in compiled.items():
            self.assertIs(cache.cache[key], fn)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseFwdSm100EndToEnd — the SM100 forward feeds the SM100 backward
# ═══════════════════════════════════════════════════════════


@unittest.skipUnless(is_sm10x(), "ffa_fwd_sm100_index needs an SM100 GPU")
class TestIndexSparseFwdSm100EndToEnd(unittest.TestCase):
    """out / lse of ``ffa_fwd_sm100_index`` feed ``ffa_bwd_sm100_index``.

    out and dq / dk / dv are checked against SDPA, lse against the fp32 gather.
    """

    @property
    def device(self):
        return torch.cuda.current_device()

    def test_gqa8_one_kv_head(self):
        config = {
            "B": 1,
            "S": 512,
            "NHQ": 8,
            "NHK": 1,
            "D": 128,
            "topk": 128,
            "q_block_size": 16,
        }
        _run_index_sparse_config(
            self.device, config, fwd_fn=ffa_fwd_sm100_index_sm90_layout
        )

    def test_gqa8_multiple_kv_heads(self):
        config = {
            "B": 1,
            "S": 256,
            "NHQ": 64,
            "NHK": 8,
            "D": 128,
            "topk": 128,
            "q_block_size": 16,
            "keep_kv_heads": True,
        }
        _run_index_sparse_config(
            self.device, config, fwd_fn=ffa_fwd_sm100_index_sm90_layout
        )

    def test_mha_q_block_128(self):
        config = {
            "B": 1,
            "S": 512,
            "NHQ": 1,
            "NHK": 1,
            "D": 128,
            "topk": 128,
            "q_block_size": 128,
        }
        _run_index_sparse_config(
            self.device, config, fwd_fn=ffa_fwd_sm100_index_sm90_layout
        )


if __name__ == "__main__":
    run_tests()
