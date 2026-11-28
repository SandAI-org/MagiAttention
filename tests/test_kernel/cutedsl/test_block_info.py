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
            sq, sk, idx = Int32(mSeqQ[i]), Int32(mSeqK[i]), Int32(mIdx[i])
            if cutlass.const_expr(self.per_range):
                at = Int32(mAttn[i])
                flo, fhi = info.get_m_block_min_max_seqlens(sq, sk, idx, at)
                dlo, dhi = info.get_n_block_min_max_seqlens(sq, sk, idx, at)
            else:
                flo, fhi = info.get_m_block_min_max_seqlens(sq, sk, idx)
                dlo, dhi = info.get_n_block_min_max_seqlens(sq, sk, idx)
            mFwdLo[i], mFwdHi[i], mDualLo[i], mDualHi[i] = flo, fhi, dlo, dhi


class TestBlockInfoDualBounds(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("SM100/SM110 BlockInfo dual bounds test")

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
        # Include -1 and num_blocks to verify out-of-bounds normalization to (0, 0).
        blocks = torch.arange(-1, num_blocks + 1, dtype=torch.int32)
        num_idx = blocks.shape[0]
        sq, sk, idx, at = (
            t.to("cuda").contiguous()
            for t in (
                seqlen_q.repeat_interleave(num_idx),
                seqlen_k.repeat_interleave(num_idx),
                blocks.repeat(seqlen_q.shape[0]),
                mask_type.repeat_interleave(num_idx),
            )
        )
        outs = [
            torch.empty(sq.shape[0], dtype=torch.int32, device="cuda") for _ in range(4)
        ]
        args = (
            *(from_dlpack(t) for t in (sq, sk, idx, at, *outs)),
            Int32(sq.shape[0]),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        cute.compile(launcher, *args)(*args)
        m_lo, m_hi, n_lo, n_hi = torch.stack(outs).cpu().view(4, -1, num_idx)

        nonempty = (seqlen_q[:, None] > 0) & (seqlen_k[:, None] > 0) & (blocks >= 0)
        m_cap = torch.where(nonempty & (blocks < num_n[:, None]), num_m[:, None], 0)
        n_cap = torch.where(nonempty & (blocks < num_m[:, None]), num_n[:, None], 0)
        self.assertTrue(bool(((0 <= m_lo) & (m_lo <= m_hi) & (m_hi <= m_cap)).all()))
        self.assertTrue(bool(((0 <= n_lo) & (n_lo <= n_hi) & (n_hi <= n_cap)).all()))

        grid = torch.arange(num_blocks, dtype=torch.int32)
        m, n = grid[None, :, None], grid[None, None, :]
        in_m = (m_lo[:, None, 1:-1] <= m) & (m < m_hi[:, None, 1:-1])
        in_n = (n_lo[:, 1:-1, None] <= n) & (n < n_hi[:, 1:-1, None])
        valid = (m < num_m[:, None, None]) & (n < num_n[:, None, None])
        self.assertFalse(bool(((in_m != in_n) & valid).any()), f"{vars(launcher)}")

    @parameterize("pack_factor", [1, 2, 4, 8])
    @parameterize("tile_n", [128, 256])
    def test_per_range_dual_equivalence_on_device(self, pack_factor, tile_n):
        seqlens = torch.tensor(
            [0, 1, 7, 16, 33, 64, 127, 128, 129, 256, 257, 390, 512], dtype=torch.int32
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
            (True, False, None, None, 8, 128),
            (False, True, 0, 0, 1, 128),
            (False, True, 7, 19, 1, 128),
            (False, True, 128, 0, 1, 256),
            (False, True, None, 15, 1, 128),
            (False, True, 15, None, 1, 128),
        ],
    )
    def test_static_causal_and_local_dual_equivalence_on_device(self, spec):
        is_causal, is_local, wl, wr, pack_factor, tile_n = spec
        seqlens = torch.tensor(
            [0, 1, 7, 33, 64, 127, 128, 129, 256, 390, 512], dtype=torch.int32
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
