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

"""SM90-specific runtime mask-type dispatch for per-range attention masking.

Supports all four mask types (full=0, causal=1, inv_causal=2, bi_causal=3)
determined at runtime per range via ``attn_type_map[batch_idx]``.

For FULL and CAUSAL, delegates to the existing ``AttentionMask.apply_mask``.
For INV_CAUSAL and BI_CAUSAL, implements R2P bitmask masking inline (SM90 fast
path using ``r2p_bitmask_above`` / ``r2p_bitmask_below``).
"""

from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from quack import layout_utils

from . import cutedsl_utils
from .mask import (
    AttentionMask,
    mask_r2p_lambda,
    r2p_bitmask_above,
    r2p_bitmask_below,
    sm90_col_to_r2p_idx,
)


@cute.jit
def _apply_inv_causal_or_bi_causal_mask_sm90(
    acc_S: cute.Tensor,
    attn_mask: AttentionMask,
    m_block: Int32,
    n_block: Int32,
    thr_mma: cute.TiledMma,
    mask_seqlen: cutlass.Constexpr[bool],
    is_bi_causal: bool,
) -> None:
    """Apply inv_causal (or bi_causal) mask using SM90 R2P bitmask fast path.

    - inv_causal: keep positions where col >= row + offset (above diagonal)
    - bi_causal: keep positions on the diagonal (col == row + offset)
    """
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=attn_mask.swap_AB)
    acc_shape = (attn_mask.tile_m, attn_mask.tile_n)
    cS = cute.make_identity_tensor(
        acc_shape if not attn_mask.swap_AB else acc_shape[::-1]
    )
    tScS_mn = layout_utils.reshape_acc_to_mn(
        thr_mma.partition_C(cS), transpose=attn_mask.swap_AB
    )
    t0ScS_mn = layout_utils.reshape_acc_to_mn(
        thr_mma.get_slice(0).partition_C(cS), transpose=attn_mask.swap_AB
    )
    COL = 1 if const_expr(not attn_mask.swap_AB) else 0
    thr_col_offset = tScS_mn[0][COL]

    seqlenk_col_limit = attn_mask.seqlen_k - n_block * attn_mask.tile_n - thr_col_offset

    inv_causal_row_offset = (
        attn_mask.seqlen_k
        - n_block * attn_mask.tile_n
        - attn_mask.seqlen_q
        - thr_col_offset
    )
    causal_row_offset = inv_causal_row_offset + 1

    if const_expr(not attn_mask.swap_AB):
        threads_per_row = thr_mma.tv_layout_C.shape[0][0]
        mma_m_idx = None
        if const_expr(attn_mask.qhead_per_kvhead_packgqa != 1):
            tidx = thr_mma.thr_idx
            mma_m_idx = (
                m_block * attn_mask.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
            ) // attn_mask.qhead_per_kvhead_packgqa

        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
            if const_expr(attn_mask.qhead_per_kvhead_packgqa == 1):
                row_idx = tScS_mn[r, 0][0] + m_block * attn_mask.tile_m
            else:
                row_idx = cutedsl_utils.shuffle_sync(
                    mma_m_idx, r % threads_per_row, width=threads_per_row
                )

            col_limit_left = row_idx + inv_causal_row_offset
            col_limit_left_r2p = sm90_col_to_r2p_idx(col_limit_left)

            if is_bi_causal:
                col_limit_right = row_idx + causal_row_offset
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                col_limit_right_r2p = sm90_col_to_r2p_idx(col_limit_right)
                mask_r2p_lambda(
                    acc_S_mn[r, None],
                    lambda s: r2p_bitmask_below(col_limit_right_r2p, s)
                    & r2p_bitmask_above(col_limit_left_r2p, s),
                    rank1=True,
                )
            else:
                if const_expr(mask_seqlen):
                    seqlenk_r2p = sm90_col_to_r2p_idx(seqlenk_col_limit)
                    mask_r2p_lambda(
                        acc_S_mn[r, None],
                        lambda s: r2p_bitmask_below(seqlenk_r2p, s)
                        & r2p_bitmask_above(col_limit_left_r2p, s),
                        rank1=True,
                    )
                else:
                    mask_r2p_lambda(
                        acc_S_mn[r, None],
                        lambda s: r2p_bitmask_above(col_limit_left_r2p, s),
                        rank1=True,
                    )
    else:
        assert attn_mask.qhead_per_kvhead_packgqa == 1
        ROW = 1 if const_expr(not attn_mask.swap_AB) else 0
        thr_row_offset = tScS_mn[0][ROW]
        seqlenq_row_limit = (
            attn_mask.seqlen_q - m_block * attn_mask.tile_m - thr_row_offset
        )
        for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
            col_idx = t0ScS_mn[0, c][COL] + thr_col_offset + n_block * attn_mask.tile_n
            row_limit_inv = col_idx + attn_mask.seqlen_q - attn_mask.seqlen_k
            if is_bi_causal:
                row_limit_causal = row_limit_inv + 1
                if const_expr(mask_seqlen):
                    row_limit_causal = cutlass.min(row_limit_causal, seqlenq_row_limit)
                for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                    row_val = t0ScS_mn[r, 0][ROW]
                    if row_val >= row_limit_causal or row_val < row_limit_inv:
                        acc_S_mn[r, c] = -Float32.inf
            else:
                for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                    row_val = t0ScS_mn[r, 0][ROW]
                    oob = row_val < row_limit_inv
                    if const_expr(mask_seqlen):
                        oob = oob | (t0ScS_mn[0, c][COL] >= seqlenk_col_limit)
                    if oob:
                        acc_S_mn[r, c] = -Float32.inf


@cute.jit
def apply_mask_with_runtime_type_sm90(
    acc_S: cute.Tensor,
    n_block: Int32,
    mask_seqlen: cutlass.Constexpr[bool],
    mask: AttentionMask,
    mask_type: Int32,
    batch_idx: Int32,
    head_idx: Int32,
    m_block: Int32,
    thr_mma: cute.TiledMma,
    mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
    aux_tensors: Optional[list] = None,
    fastdiv_mods=(None, None),
) -> None:
    """Runtime dispatch for per-range mask_type on SM90.

    Delegates FULL and CAUSAL to the existing AttentionMask.apply_mask, and
    handles INV_CAUSAL / BI_CAUSAL with dedicated SM90 R2P logic.
    """
    if mask_type == 0:
        mask.apply_mask(
            acc_S,
            batch_idx,
            head_idx,
            m_block,
            n_block,
            thr_mma,
            mask_seqlen=mask_seqlen,
            mask_causal=False,
            mask_local=False,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )
    elif mask_type == 1:
        mask.apply_mask(
            acc_S,
            batch_idx,
            head_idx,
            m_block,
            n_block,
            thr_mma,
            mask_seqlen=mask_seqlen,
            mask_causal=True,
            mask_local=False,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )
    elif mask_type == 2:
        _apply_inv_causal_or_bi_causal_mask_sm90(
            acc_S,
            mask,
            m_block,
            n_block,
            thr_mma,
            mask_seqlen=mask_seqlen,
            is_bi_causal=False,
        )
    else:
        _apply_inv_causal_or_bi_causal_mask_sm90(
            acc_S,
            mask,
            m_block,
            n_block,
            thr_mma,
            mask_seqlen=mask_seqlen,
            is_bi_causal=True,
        )
