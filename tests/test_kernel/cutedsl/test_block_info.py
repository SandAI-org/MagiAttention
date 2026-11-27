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

from unittest import TestCase

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.block_info import BlockInfo
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.testing import parameterize


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
                dlo, dhi = info.get_n_block_min_max_per_range_seqlens(sq, sk, idx, at)
            else:
                flo, fhi = info.get_m_block_min_max_seqlens(sq, sk, idx)
                dlo, dhi = info.get_n_block_min_max_seqlens(sq, sk, idx)
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

    def test_empty_and_inverted_bounds_on_device(self):
        # Check *_seqlens normalization on both directions for negative n_idx_right,
        # inverted bi_causal, empty sequences, and oob indices.
        dual_per_range = _BlockInfoDualLauncher(
            128, 128, True, False, None, None, 1, per_range=True
        )
        per_range_cases = [
            (512, 128, 0, MT_MAP.causal),
            (512, 128, 0, MT_MAP.bi_causal),
            (512, 128, 1, MT_MAP.bi_causal),
            (512, 128, 3, MT_MAP.bi_causal),
            (512, 128, 4, MT_MAP.full),  # oob m_block and oob n_block
            (512, 128, -1, MT_MAP.full),
            (0, 128, 0, MT_MAP.causal),
            (128, 0, 0, MT_MAP.inv_causal),
        ]
        pr_out = self._eval_device(
            dual_per_range,
            torch.tensor([c[0] for c in per_range_cases], dtype=torch.int32),
            torch.tensor([c[1] for c in per_range_cases], dtype=torch.int32),
            torch.tensor([c[2] for c in per_range_cases], dtype=torch.int32),
            torch.tensor([c[3] for c in per_range_cases], dtype=torch.int32),
        )
        m_lo, m_hi, n_lo, n_hi = pr_out.tolist()
        self.assertEqual((n_lo[0], n_hi[0]), (0, 0))
        self.assertEqual((m_lo[1], m_hi[1]), (1, 1))
        self.assertEqual((n_lo[1], n_hi[1]), (0, 0))
        self.assertEqual((n_lo[2], n_hi[2]), (0, 0))
        self.assertEqual((n_lo[3], n_hi[3]), (1, 1))
        for k in range(4, len(per_range_cases)):
            self.assertEqual((m_lo[k], m_hi[k]), (0, 0))
            self.assertEqual((n_lo[k], n_hi[k]), (0, 0))
        for k in range(len(per_range_cases)):
            self.assertTrue(0 <= m_lo[k] <= m_hi[k])
            self.assertTrue(0 <= n_lo[k] <= n_hi[k])

        dual_static = _BlockInfoDualLauncher(
            128, 128, True, False, None, None, 1, per_range=False
        )
        st_out = self._eval_device(
            dual_static,
            torch.tensor([512, 512, 0, 128], dtype=torch.int32),
            torch.tensor([128, 128, 128, 0], dtype=torch.int32),
            torch.tensor([0, 4, 0, 0], dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
        )
        sm_lo, sm_hi, sn_lo, sn_hi = st_out.tolist()
        self.assertEqual((sn_lo[0], sn_hi[0]), (0, 0))
        self.assertEqual((sm_lo[0], sm_hi[0]), (3, 4))
        for k in range(1, 4):
            self.assertEqual((sm_lo[k], sm_hi[k]), (0, 0))
            self.assertEqual((sn_lo[k], sn_hi[k]), (0, 0))


if __name__ == "__main__":
    run_tests()
