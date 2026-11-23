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

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from .ffa_utils import MT_MAP
from .seqlen_info import SeqlenInfoQK, SeqlenInfoQKNewK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def _n_block_bounds(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        m_block: Int32,
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_k, self.tile_n)
        if const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_k - seqlen_q
            n_idx_right = (
                n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
            )
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        n_block_min = 0
        if const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_k - seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.tile_n, 0)
        return n_block_min, n_block_max

    @cute.jit
    def _n_block_bounds_per_range(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        m_block: Int32,
        attn_type: Int32,
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_k, self.tile_n)
        if attn_type == MT_MAP.causal or attn_type == MT_MAP.bi_causal:
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx_right = m_idx_max + seqlen_k - seqlen_q
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        n_block_min = Int32(0)
        if attn_type == MT_MAP.inv_causal or attn_type == MT_MAP.bi_causal:
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_block_min = (
                n_block_max if m_idx_min >= seqlen_k else m_idx_min // self.tile_n
            )
        return n_block_min, n_block_max

    @cute.jit
    def _m_block_bounds(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        n_block: Int32,
    ) -> Tuple[Int32, Int32]:
        pf = self.qhead_per_kvhead_packgqa
        if const_expr(pf > 1):
            assert not self.is_local, "packed bwd does not support local attention"
            m_block_max = cute.ceil_div(seqlen_q * pf, self.tile_m)
        else:
            m_block_max = cute.ceil_div(seqlen_q, self.tile_m)
        m_block_min = 0
        if const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            n_idx_min = n_block * self.tile_n
            m_idx = n_idx_min + seqlen_q - seqlen_k
            m_idx_right = (
                m_idx if const_expr(self.is_causal) else m_idx - self.window_size_right
            )
            if const_expr(pf > 1):
                m_idx_right = m_idx_right * pf
            m_block_min = max(m_block_min, m_idx_right // self.tile_m)
        if const_expr(self.is_local and self.window_size_left is not None):
            n_idx_max = (n_block + 1) * self.tile_n
            m_idx = n_idx_max + seqlen_q - seqlen_k
            m_idx_left = m_idx + self.window_size_left
            m_block_max = min(m_block_max, cute.ceil_div(m_idx_left, self.tile_m))
        return m_block_min, m_block_max

    @cute.jit
    def _m_block_bounds_per_range(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        n_block: Int32,
        attn_type: Int32,
    ) -> Tuple[Int32, Int32]:
        pf = self.qhead_per_kvhead_packgqa
        if const_expr(pf > 1):
            m_block_max = cute.ceil_div(seqlen_q * pf, self.tile_m)
        else:
            m_block_max = cute.ceil_div(seqlen_q, self.tile_m)
        if attn_type == MT_MAP.inv_causal or attn_type == MT_MAP.bi_causal:
            n_idx_max = min(seqlen_k, (n_block + 1) * self.tile_n)
            if const_expr(pf > 1):
                n_idx_max = n_idx_max * pf
            m_block_max = min(m_block_max, cute.ceil_div(n_idx_max, self.tile_m))
        m_block_min = Int32(0)
        if attn_type == MT_MAP.causal or attn_type == MT_MAP.bi_causal:
            n_idx_min = n_block * self.tile_n
            m_idx_right = n_idx_min + seqlen_q - seqlen_k
            if const_expr(pf > 1):
                m_idx_right = m_idx_right * pf
            m_block_min = max(m_block_min, m_idx_right // self.tile_m)
        return m_block_min, m_block_max

    @cute.jit
    def _split_kv_bounds(
        self,
        n_block_min: Int32,
        n_block_max: Int32,
        split_idx: Int32,
        num_splits: Int32,
    ) -> Tuple[Int32, Int32]:
        if cutlass.const_expr(self.is_split_kv):
            num_n_blocks_per_split = (
                Int32(0)
                if n_block_max <= n_block_min
                else (n_block_max - n_block_min + num_splits - 1) // num_splits
            )
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
        return n_block_min, n_block_max

    @staticmethod
    @cute.jit
    def _normalize_bounds(lo: Int32, hi: Int32) -> Tuple[Int32, Int32]:
        hi = cutlass.max(Int32(0), Int32(hi))
        lo = cutlass.min(hi, cutlass.max(Int32(0), Int32(lo)))
        return lo, hi

    @cute.jit
    def _m_block_upper(self, seqlen_q: Int32) -> Int32:
        pf = self.qhead_per_kvhead_packgqa
        if const_expr(pf > 1):
            return cute.ceil_div(seqlen_q * pf, self.tile_m)
        return cute.ceil_div(seqlen_q, self.tile_m)

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        n_block_min, n_block_max = self._n_block_bounds(
            seqlen_info.seqlen_q, seqlen_info.seqlen_k, m_block
        )
        return self._split_kv_bounds(n_block_min, n_block_max, split_idx, num_splits)

    @cute.jit
    def get_n_block_min_max_per_range(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        attn_type: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        """Runtime n-block bounds for per-range masks (fwd), all four types.

        Ignores compile-time ``is_causal`` / ``is_local``; uses ``attn_type``
        only. Local / sliding-window is not supported here.
        """
        n_block_min, n_block_max = self._n_block_bounds_per_range(
            seqlen_info.seqlen_q, seqlen_info.seqlen_k, m_block, attn_type
        )
        return self._split_kv_bounds(n_block_min, n_block_max, split_idx, num_splits)

    @cute.jit
    def get_m_block_min_max(
        self, seqlen_info: SeqlenInfoQK, n_block: Int32
    ) -> Tuple[Int32, Int32]:
        return self._m_block_bounds(seqlen_info.seqlen_q, seqlen_info.seqlen_k, n_block)

    @cute.jit
    def get_m_block_min_max_per_range(
        self,
        seqlen_info: SeqlenInfoQK,
        n_block: Int32,
        attn_type: Int32,
    ) -> Tuple[Int32, Int32]:
        """Runtime m-block bounds for per-range masks (bwd), all four types.

        Dual of :meth:`get_n_block_min_max_per_range`: ``has_causal`` narrows
        the lower bound here and the upper bound in fwd; ``has_inv`` is the
        other way round.
        """
        return self._m_block_bounds_per_range(
            seqlen_info.seqlen_q, seqlen_info.seqlen_k, n_block, attn_type
        )

    @cute.jit
    def get_n_block_k_new_min_max(
        self,
        seqlen_info: SeqlenInfoQKNewK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        """Get the block range for new K tokens (append KV).

        First computes the full n_block range via get_n_block_min_max, then maps
        those blocks into the new-K index space by subtracting seqlen_k_og.
        """
        n_block_min, n_block_max = self.get_n_block_min_max(
            seqlen_info,
            m_block,
            split_idx,
            num_splits,
        )
        idx_k_new_min = cutlass.max(
            n_block_min * self.tile_n - seqlen_info.seqlen_k_og, 0
        )
        idx_k_new_max = cutlass.min(
            n_block_max * self.tile_n - seqlen_info.seqlen_k_og,
            seqlen_info.seqlen_k_new,
        )
        n_block_new_min = idx_k_new_min // self.tile_n
        n_block_new_max = (
            cute.ceil_div(idx_k_new_max, self.tile_n)
            if idx_k_new_max > idx_k_new_min
            else n_block_new_min
        )
        return n_block_new_min, n_block_new_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        """If we have separate iterations with causal or local masking at the start, where do we stop"""
        m_idx_min = m_block * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.tile_n)

    @cute.jit
    def get_n_block_min_causal_local_mask_per_range(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
        n_block_max: Int32,
        attn_type: Int32,
    ) -> Int32:
        """Causal-mask segment bound for per-range masks (fwd softmax).

        Full / InvCausal return ``n_block_max`` (empty right segment); Causal /
        BiCausal match :meth:`get_n_block_min_causal_local_mask`.
        """
        # No early return: CuteDSL forbids return inside a dynamic `if`.
        result = n_block_max
        if attn_type == MT_MAP.causal or attn_type == MT_MAP.bi_causal:
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx_right = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            result = cutlass.max(n_block_min, n_idx_right // self.tile_n)
        return result

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        """If we have separate iterations with local masking at the end, where do we stop the non-masked iterations"""
        if const_expr(not self.is_local or self.window_size_left is None):
            return n_block_min
        else:
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            return cutlass.max(n_block_min, cute.ceil_div(n_idx_left, self.tile_n))

    @cute.jit
    def get_n_block_min_before_local_mask_per_range(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
        n_block_max: Int32,
        attn_type: Int32,
    ) -> Int32:
        """Left-segment bound for per-range masks (fwd softmax).

        Splits the iteration into an unmasked middle and a left band that the
        inv-causal diagonal cuts through. Full / Causal return ``n_block_min``
        so the left segment is empty.
        """
        # No early return: CuteDSL forbids return inside a dynamic `if`.
        result = n_block_min
        if attn_type == MT_MAP.inv_causal or attn_type == MT_MAP.bi_causal:
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            result = cutlass.min(n_block_max, cute.ceil_div(m_idx_max, self.tile_n))
        return result

    @cute.jit
    def get_n_block_max_for_m_block(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
    ) -> Int32:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(self.is_causal or self.window_size_right is not None):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx_right = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            if const_expr(self.window_size_right is not None):
                n_idx_right += self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        return n_block_max

    @cute.jit
    def get_m_block_min_max_seqlens(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        n_block: Int32,
    ) -> Tuple[Int32, Int32]:
        lo, hi = Int32(0), Int32(0)
        if seqlen_q > 0 and seqlen_k > 0:
            if n_block >= 0 and n_block < cute.ceil_div(seqlen_k, self.tile_n):
                lo, hi = self._normalize_bounds(
                    *self._m_block_bounds(seqlen_q, seqlen_k, n_block)
                )
        return lo, hi

    @cute.jit
    def get_m_block_min_max_per_range_seqlens(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        n_block: Int32,
        attn_type: Int32,
    ) -> Tuple[Int32, Int32]:
        lo, hi = Int32(0), Int32(0)
        if seqlen_q > 0 and seqlen_k > 0:
            if n_block >= 0 and n_block < cute.ceil_div(seqlen_k, self.tile_n):
                lo, hi = self._normalize_bounds(
                    *self._m_block_bounds_per_range(
                        seqlen_q, seqlen_k, n_block, attn_type
                    )
                )
        return lo, hi

    @cute.jit
    def get_n_block_min_max_seqlens(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        m_block: Int32,
    ) -> Tuple[Int32, Int32]:
        """Exact dual of get_m_block_min_max_seqlens (unsplit K range)."""
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            assert not self.is_local, "packed bwd does not support local attention"
        lo, hi = Int32(0), Int32(0)
        if seqlen_q > 0 and seqlen_k > 0:
            if m_block >= 0 and m_block < self._m_block_upper(seqlen_q):
                lo, hi = self._normalize_bounds(
                    *self._n_block_bounds(seqlen_q, seqlen_k, m_block)
                )
        return lo, hi

    @cute.jit
    def get_n_block_min_max_per_range_seqlens(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        m_block: Int32,
        attn_type: Int32,
    ) -> Tuple[Int32, Int32]:
        """Exact dual of get_m_block_min_max_per_range_seqlens (unsplit K range)."""
        lo, hi = Int32(0), Int32(0)
        if seqlen_q > 0 and seqlen_k > 0:
            if m_block >= 0 and m_block < self._m_block_upper(seqlen_q):
                lo, hi = self._normalize_bounds(
                    *self._n_block_bounds_per_range(
                        seqlen_q, seqlen_k, m_block, attn_type
                    )
                )
        return lo, hi
