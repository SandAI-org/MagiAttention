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


from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint8, const_expr

# isort: split
from quack.compile_utils import make_fake_tensor as fake_tensor

from . import range_chain
from .block_info import BlockInfo
from .cache_utils import get_jit_cache
from .cutedsl_utils import elem_pointer
from .ffa_utils import MT_MAP

_CKPT_THREADS = 128
_CKPT_WARPS = _CKPT_THREADS // 32


@dataclass(frozen=True)
class BwdCkptSpec:
    """Compile-time specialization of the backward checkpoint kernels; it must
    equal the tiling and mask configuration of the attention kernel.

    ``mask_type`` is the static mask of the attention kernel, None for a
    per-range mask.
    """

    tile_m: int
    cluster_tile_k: int
    mask_type: Optional[int]
    range_merge: bool

    @property
    def use_per_range_mask(self) -> bool:
        return self.mask_type is None

    @property
    def block_info(self) -> BlockInfo:
        return BlockInfo(
            self.tile_m,
            self.cluster_tile_k,
            self.mask_type is None or self.mask_type == MT_MAP.causal,
        )


def conflict_ckpt_shape(num_ranges: int, max_ckpts: int) -> tuple[int, int]:
    """Return ``(num_ckpts, ckpt_stride)`` with ``num_ckpts <= max_ckpts``.

    Checkpoint row ``c`` holds, per physical block, the highest writer among
    ranges ``[0, c * ckpt_stride)``, 0 if none. A chunk-max kernel reduces the
    writers of chunk ``c`` (``ckpt_stride`` ranges) into row ``c``; an exclusive
    prefix max along the rows then yields the checkpoints. The attention kernel
    reads row ``r // ckpt_stride`` and scans the earlier ranges of that chunk
    itself, so both sides must number writers identically.
    """
    assert max_ckpts > 0
    ckpt_stride = max(1, (num_ranges + max_ckpts - 1) // max_ckpts)
    num_ckpts = max(1, (num_ranges + ckpt_stride - 1) // ckpt_stride)
    return num_ckpts, ckpt_stride


class _ConflictPrefixMax:
    """In-place exclusive prefix max along axis 0 of ``(num_ckpts, num_blocks)``."""

    @cute.jit
    def __call__(
        self,
        mCkpt: cute.Tensor,
        num_ckpts: Int32,
        num_blocks: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(mCkpt, num_ckpts, num_blocks).launch(
            grid=(cute.ceil_div(num_blocks, _CKPT_THREADS), 1, 1),
            block=(_CKPT_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mCkpt: cute.Tensor, num_ckpts: Int32, num_blocks: Int32):
        b = Int32(cute.arch.block_idx()[0]) * _CKPT_THREADS + cute.arch.thread_idx()[0]
        if b < num_blocks:
            running = Int32(0)
            c = Int32(0)
            while c < num_ckpts:
                cur = Int32(mCkpt[c, b])
                mCkpt[c, b] = running
                running = cutlass.max(running, cur)
                c += 1


class _FwdConflictChunkMax:
    """Reduce the forward writer ``r + 1`` of each range into
    ``mCkpt[r // ckpt_stride, b]``."""

    def __init__(self, block_size: int, rows_per_token: int):
        self.block_size = block_size
        self.rows_per_token = rows_per_token

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mCkpt: cute.Tensor,
        num_ranges: Int32,
        ckpt_stride: Int32,
        num_ckpts: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(mQRanges, mCkpt, num_ranges, ckpt_stride, num_ckpts).launch(
            grid=(cute.ceil_div(num_ranges, _CKPT_WARPS), 1, 1),
            block=(_CKPT_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mCkpt: cute.Tensor,
        num_ranges: Int32,
        ckpt_stride: Int32,
        num_ckpts: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_in_cta = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        r = Int32(cute.arch.block_idx()[0]) * _CKPT_WARPS + warp_in_cta
        if r < num_ranges:
            c = r // ckpt_stride
            # The exclusive prefix max never reads the last chunk.
            if c < num_ckpts - 1:
                start = mQRanges[r, 0] * self.rows_per_token
                end = mQRanges[r, 1] * self.rows_per_token
                if end > start:
                    writer = r + 1
                    b = start // self.block_size + lane
                    last_b = (end - 1) // self.block_size
                    while b <= last_b:
                        if ckpt_stride == 1:
                            mCkpt[c, b] = writer
                        else:
                            cute.arch.atomic_max(
                                elem_pointer(mCkpt, (c, b)),
                                writer,
                                sem="relaxed",
                                scope="gpu",
                            )
                        b += cute.arch.WARP_SIZE


class _BwdDQConflictChunkMax:
    """Reduce the highest dQ writer of each pair into
    ``mCkpt[g // ckpt_stride, b]``, with ``g`` the range or RangeMerge group."""

    def __init__(self, spec: BwdCkptSpec):
        self.spec = spec

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mRowGroup: Optional[cute.Tensor],
        mCkpt: cute.Tensor,
        writer_stride: Int32,
        ckpt_stride: Int32,
        num_ckpts: Int32,
        num_pairs: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mQRanges,
            mKRanges,
            mMaskTypes,
            mRowGroup,
            mCkpt,
            writer_stride,
            ckpt_stride,
            num_ckpts,
            num_pairs,
        ).launch(
            grid=(cute.ceil_div(num_pairs, _CKPT_WARPS), 1, 1),
            block=(_CKPT_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mRowGroup: Optional[cute.Tensor],
        mCkpt: cute.Tensor,
        writer_stride: Int32,
        ckpt_stride: Int32,
        num_ckpts: Int32,
        num_pairs: Int32,
    ):
        spec = self.spec
        tidx = cute.arch.thread_idx()[0]
        warp_in_cta = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        row = Int32(cute.arch.block_idx()[0]) * _CKPT_WARPS + warp_in_cta
        if row < num_pairs:
            group_idx = row
            if const_expr(spec.range_merge):
                assert mRowGroup is not None
                group_idx = Int32(mRowGroup[row])
            c = group_idx // ckpt_stride
            if c < num_ckpts - 1:
                q_s = mQRanges[row, 0]
                q_e = mQRanges[row, 1]
                seqlen_q = q_e - q_s
                seqlen_k = mKRanges[group_idx, 1] - mKRanges[group_idx, 0]
                if seqlen_q > 0 and seqlen_k > 0:
                    attn_r = Int32(MT_MAP.full)
                    if const_expr(spec.use_per_range_mask):
                        assert mMaskTypes is not None
                        attn_r = Int32(mMaskTypes[row])
                    block_info = spec.block_info
                    b = q_s // spec.tile_m + lane
                    last_b = (q_e - 1) // spec.tile_m
                    while b <= last_b:
                        k_tile_end = range_chain.dq_pair_k_tile_end_on_block(
                            block_info,
                            seqlen_q,
                            seqlen_k,
                            q_s,
                            attn_r,
                            b,
                            spec.use_per_range_mask,
                        )
                        if k_tile_end > 0:
                            writer = group_idx * writer_stride + k_tile_end
                            if const_expr(not spec.range_merge) and ckpt_stride == 1:
                                mCkpt[c, b] = writer
                            else:
                                cute.arch.atomic_max(
                                    elem_pointer(mCkpt, (c, b)),
                                    writer,
                                    sem="relaxed",
                                    scope="gpu",
                                )
                        b += cute.arch.WARP_SIZE


class _BwdDKVConflictChunkMax:
    """Reduce the last dK/dV writer of each pair into
    ``mCkpt[g // ckpt_stride, kb]`` and, under RangeMerge, set
    ``mGroupAllKActive[g]`` when a pair of group ``g`` makes every K tile of
    the group active."""

    def __init__(self, spec: BwdCkptSpec):
        self.spec = spec

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mRowGroup: Optional[cute.Tensor],
        mCkpt: cute.Tensor,
        mGroupAllKActive: Optional[cute.Tensor],
        writer_stride: Int32,
        ckpt_stride: Int32,
        num_ckpts: Int32,
        num_pairs: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mQRanges,
            mKRanges,
            mMaskTypes,
            mRowGroup,
            mCkpt,
            mGroupAllKActive,
            writer_stride,
            ckpt_stride,
            num_ckpts,
            num_pairs,
        ).launch(
            grid=(cute.ceil_div(num_pairs, _CKPT_WARPS), 1, 1),
            block=(_CKPT_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mRowGroup: Optional[cute.Tensor],
        mCkpt: cute.Tensor,
        mGroupAllKActive: Optional[cute.Tensor],
        writer_stride: Int32,
        ckpt_stride: Int32,
        num_ckpts: Int32,
        num_pairs: Int32,
    ):
        spec = self.spec
        tidx = cute.arch.thread_idx()[0]
        warp_in_cta = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        row = Int32(cute.arch.block_idx()[0]) * _CKPT_WARPS + warp_in_cta
        if row < num_pairs:
            group_idx = row
            if const_expr(spec.range_merge):
                assert mRowGroup is not None
                group_idx = Int32(mRowGroup[row])
            k_s = mKRanges[group_idx, 0]
            k_e = mKRanges[group_idx, 1]
            seqlen_q = mQRanges[row, 1] - mQRanges[row, 0]
            seqlen_k = k_e - k_s
            if seqlen_q > 0 and seqlen_k > 0:
                attn_r = Int32(MT_MAP.full)
                if const_expr(spec.use_per_range_mask):
                    assert mMaskTypes is not None
                    attn_r = Int32(mMaskTypes[row])
                if const_expr(spec.range_merge):
                    assert mGroupAllKActive is not None
                    # Only bi_causal with seqlen_q > seqlen_k leaves a K tile
                    # of the pair without an unmasked Q tile.
                    if lane == 0 and (
                        attn_r != MT_MAP.bi_causal or seqlen_q <= seqlen_k
                    ):
                        mGroupAllKActive[group_idx] = Uint8(1)
                c = group_idx // ckpt_stride
                if c < num_ckpts - 1:
                    writer = (group_idx + 1) * writer_stride
                    block_info = spec.block_info
                    kb = k_s // spec.cluster_tile_k + lane
                    last_kb = (k_e - 1) // spec.cluster_tile_k
                    while kb <= last_kb:
                        if range_chain.dkv_pair_is_active_on_block(
                            block_info,
                            seqlen_q,
                            seqlen_k,
                            k_s,
                            attn_r,
                            kb,
                            spec.use_per_range_mask,
                        ):
                            if const_expr(not spec.range_merge) and ckpt_stride == 1:
                                mCkpt[c, kb] = writer
                            else:
                                cute.arch.atomic_max(
                                    elem_pointer(mCkpt, (c, kb)),
                                    writer,
                                    sem="relaxed",
                                    scope="gpu",
                                )
                        kb += cute.arch.WARP_SIZE


_PREFIX_MAX_CACHE = get_jit_cache("range_chain_prefix_max")
_FWD_CKPT_CACHE = get_jit_cache("range_chain_fwd_ckpt")
_BWD_DQ_CKPT_CACHE = get_jit_cache("range_chain_bwd_dq_ckpt")
_BWD_DKV_CKPT_CACHE = get_jit_cache("range_chain_bwd_dkv_ckpt")


def _fake_i32(*shape):
    return fake_tensor(Int32, shape, divisibility=1)


def _compile(launcher, *fake_args):
    return cute.compile(
        launcher,
        *fake_args,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _fake_bwd_range_args(spec: BwdCkptSpec) -> tuple:
    """Fake ``(mQRanges, mKRanges, mMaskTypes, mRowGroup, mCkpt)``."""
    sym = cute.sym_int
    return (
        _fake_i32(sym(), 2),
        _fake_i32(sym(), 2),
        _fake_i32(sym()) if spec.use_per_range_mask else None,
        _fake_i32(sym()) if spec.range_merge else None,
        _fake_i32(sym(), sym()),
    )


def _run_conflict_prefix_max(ckpt: torch.Tensor) -> None:
    num_ckpts, num_blocks = ckpt.shape
    key = ("prefix_max",)
    if key not in _PREFIX_MAX_CACHE:
        sym = cute.sym_int
        _PREFIX_MAX_CACHE[key] = _compile(
            _ConflictPrefixMax(), _fake_i32(sym(), sym()), Int32(0), Int32(0)
        )
    _PREFIX_MAX_CACHE[key](ckpt, num_ckpts, num_blocks)


def build_fwd_conflict_ckpt(
    q_ranges: torch.Tensor,
    num_blocks: int,
    max_ckpts: int,
    block_size: int,
    rows_per_token: int,
) -> tuple[torch.Tensor, int]:
    """Build the forward checkpoints ``(num_ckpts, num_blocks)`` on the device
    of ``q_ranges`` and return them with ``ckpt_stride``.

    A block holds ``block_size`` rows and a token of ``q_ranges`` spans
    ``rows_per_token`` rows.
    """
    num_ranges = q_ranges.shape[0]
    num_ckpts, ckpt_stride = conflict_ckpt_shape(num_ranges, max_ckpts)
    ckpt = torch.zeros(num_ckpts, num_blocks, dtype=torch.int32, device=q_ranges.device)
    if num_ckpts == 1:
        return ckpt, ckpt_stride
    key = (block_size, rows_per_token)
    if key not in _FWD_CKPT_CACHE:
        sym = cute.sym_int
        _FWD_CKPT_CACHE[key] = _compile(
            _FwdConflictChunkMax(block_size, rows_per_token),
            _fake_i32(sym(), 2),
            _fake_i32(sym(), sym()),
            Int32(0),
            Int32(0),
            Int32(0),
        )
    _FWD_CKPT_CACHE[key](q_ranges, ckpt, num_ranges, ckpt_stride, num_ckpts)
    _run_conflict_prefix_max(ckpt)
    return ckpt, ckpt_stride


def build_bwd_dq_conflict_ckpt(
    spec: BwdCkptSpec,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: Optional[torch.Tensor],
    row_group: Optional[torch.Tensor],
    num_blocks: int,
    max_ckpts: int,
    writer_stride: int,
) -> tuple[torch.Tensor, int]:
    """Build the backward dQ checkpoints ``(num_ckpts, num_blocks)`` and return
    them with ``ckpt_stride``.

    ``k_ranges`` has one row per chain range. Without RangeMerge ``q_ranges``
    pairs with it row by row; with RangeMerge ``row_group[i]`` is the
    ``k_ranges`` row of pair ``i``.
    """
    num_pairs = q_ranges.shape[0]
    num_ckpts, ckpt_stride = conflict_ckpt_shape(k_ranges.shape[0], max_ckpts)
    ckpt = torch.zeros(num_ckpts, num_blocks, dtype=torch.int32, device=q_ranges.device)
    if num_ckpts == 1 or num_pairs == 0:
        return ckpt, ckpt_stride
    key = (spec,)
    if key not in _BWD_DQ_CKPT_CACHE:
        _BWD_DQ_CKPT_CACHE[key] = _compile(
            _BwdDQConflictChunkMax(spec),
            *_fake_bwd_range_args(spec),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
        )
    _BWD_DQ_CKPT_CACHE[key](
        q_ranges,
        k_ranges,
        mask_types if spec.use_per_range_mask else None,
        row_group if spec.range_merge else None,
        ckpt,
        writer_stride,
        ckpt_stride,
        num_ckpts,
        num_pairs,
    )
    _run_conflict_prefix_max(ckpt)
    return ckpt, ckpt_stride


def build_bwd_dkv_conflict_ckpt_and_all_k_active(
    spec: BwdCkptSpec,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: Optional[torch.Tensor],
    row_group: Optional[torch.Tensor],
    num_blocks: int,
    max_ckpts: int,
    writer_stride: int,
) -> tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Build the backward dK/dV checkpoints ``(num_ckpts, num_blocks)`` and
    return them with ``ckpt_stride`` and ``group_all_k_active``.

    The ranges follow :func:`build_bwd_dq_conflict_ckpt`. Under RangeMerge
    ``group_all_k_active`` is uint8 ``[num_groups]``: 1 if a single pair of the
    group makes every K tile of the group active, so the attention kernel may
    skip its walk over the pairs of the group. It is ``None`` otherwise.
    """
    num_pairs = q_ranges.shape[0]
    num_groups = k_ranges.shape[0]
    num_ckpts, ckpt_stride = conflict_ckpt_shape(num_groups, max_ckpts)
    ckpt = torch.zeros(num_ckpts, num_blocks, dtype=torch.int32, device=q_ranges.device)
    group_all_k_active = (
        torch.zeros(num_groups, dtype=torch.uint8, device=q_ranges.device)
        if spec.range_merge
        else None
    )
    if num_pairs == 0 or (num_ckpts == 1 and not spec.range_merge):
        return ckpt, ckpt_stride, group_all_k_active
    key = (spec,)
    if key not in _BWD_DKV_CKPT_CACHE:
        sym = cute.sym_int
        _BWD_DKV_CKPT_CACHE[key] = _compile(
            _BwdDKVConflictChunkMax(spec),
            *_fake_bwd_range_args(spec),
            fake_tensor(Uint8, (sym(),), divisibility=1) if spec.range_merge else None,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
        )
    _BWD_DKV_CKPT_CACHE[key](
        q_ranges,
        k_ranges,
        mask_types if spec.use_per_range_mask else None,
        row_group if spec.range_merge else None,
        ckpt,
        group_all_k_active,
        writer_stride,
        ckpt_stride,
        num_ckpts,
        num_pairs,
    )
    if num_ckpts > 1:
        _run_conflict_prefix_max(ckpt)
    return ckpt, ckpt_stride, group_all_k_active
