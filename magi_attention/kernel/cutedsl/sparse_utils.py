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

# mypy: disable-error-code="arg-type,index"

"""
Block-sparsity utilities for FlexFlashAttention.

This module hosts both:
- Host-side normalization/validation of block-sparse metadata tensors, and
- Device-side (CUTE DSL) runtime helpers that produce/consume block-sparse
  loads inside the forward/backward attention kernels.
"""

import math
from enum import IntEnum
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from quack import copy_utils

from .cutedsl_utils import get_broadcast_dims, to_cute_tensor
from .named_barrier import NamedBarrierBwd
from .seqlen_info import SeqlenInfoQK

# ---------------------------------------------------------------------------
# Inner-loop load/store mode enums (mirrors csrc/flexible_flash_attention/inner_ldst_mode.hpp)
# ---------------------------------------------------------------------------


class InnerLoadMode(IntEnum):
    """Inner-loop KV load strategy for sparse attention.

    Tma:     2D TMA descriptor — used when tiles are physically contiguous (kbs >= n_block_size).
    CpAsync: cp.async per-row scatter (per-token loads for arbitrary token indices).
    """

    Tma = 0
    CpAsync = 2


class InnerStoreMode(IntEnum):
    """Inner-loop store strategy (BWD dK/dV accumulation).

    Tma:        2D TMA reduce-add from swizzled SMEM.
    Tma1d:      cp.reduce.async.bulk per-row from linear SMEM.
    AtomicAdd:  scalar atomicAdd from SMEM.
    BypassSmem: register atomicAdd to gmem (no SMEM buffer).
    """

    Tma = 0
    Tma1d = 1
    AtomicAdd = 2
    BypassSmem = 3


class OuterStoreMode(IntEnum):
    """Outer-loop store strategy (epilogue O/dQ/dKV write).

    Tma:  2D TMA store / reduce-add.
    Stg:  per-thread STS + STG.128 with residual guard.
    Tma1d: R2S to linear SMEM, then per-row cp.async.bulk S2G.
    """

    Tma = 0
    Stg = 1
    Tma1d = 2


def derive_outer_store_mode(
    cu_seqlens_q_present: bool = False,
    seqused_q_present: bool = False,
    pack_gqa: bool = False,
    m_block_size: int = 128,
    qhead_per_kvhead: int = 1,
) -> OuterStoreMode:
    """Derive OuterStoreMode from kernel configuration.

    Tma:   2D tile store when output is fully contiguous in 2D.
    Tma1d: Per-row bulk S2G when rows are contiguous but tile is not (varlen only,
           without PackGQA which makes row→gmem mapping non-trivial).
    Stg:   Fallback per-element store (varlen+PackGQA, or PackGQA misalignment).
    """
    if cu_seqlens_q_present or seqused_q_present:
        if pack_gqa and qhead_per_kvhead > 1:
            return OuterStoreMode.Stg
        return OuterStoreMode.Tma1d
    if pack_gqa and m_block_size % qhead_per_kvhead != 0:
        return OuterStoreMode.Stg
    return OuterStoreMode.Tma


# ---------------------------------------------------------------------------
# Block-sparse tensor data structures
# ---------------------------------------------------------------------------


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None = None
    full_block_idx: cute.Tensor | None = None
    cu_total_m_blocks: cute.Tensor | None = None
    cu_block_idx_offsets: cute.Tensor | None = None
    dq_write_order: cute.Tensor | None = None
    dq_write_order_full: cute.Tensor | None = None
    is_valid_total: cute.Tensor | None = None

    def __new_from_mlir_values__(self, values):
        new_fields = []
        idx = 0
        for original in self:
            if original is None:
                new_fields.append(None)
            else:
                new_fields.append(values[idx])
                idx += 1
        return BlockSparseTensors(*new_fields)


class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    cu_total_m_blocks: torch.Tensor | None = None
    cu_block_idx_offsets: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None
    dq_write_order: torch.Tensor | None = None
    dq_write_order_full: torch.Tensor | None = None
    is_valid_total: torch.Tensor | None = None
    spt: bool | None = None


# ---------------------------------------------------------------------------
# Host-side normalization/validation helpers
# ---------------------------------------------------------------------------


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_sparse_q_block_size(
    tensors: BlockSparseTensorsTorch | None,
    seqlen_q: int,
) -> int | None:
    """Return the Q sparse block size, or None when sparsity is unset or ambiguous."""
    if tensors is None:
        return None
    if tensors.block_size is not None:
        return tensors.block_size[0]
    num_m_blocks = tensors.mask_block_idx.shape[2]
    min_block_size = ceildiv(seqlen_q, num_m_blocks)
    max_block_size = (
        seqlen_q if num_m_blocks == 1 else (seqlen_q - 1) // (num_m_blocks - 1)
    )
    if min_block_size != max_block_size:
        return None
    return min_block_size


def _expand_sparsity_tensor(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    tensor_name: str,
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> torch.Tensor:
    """Check if we need to expand the tensor to expected shape, and do so if possible."""
    needs_expand = tensor.shape != expected_shape
    if not needs_expand:
        return tensor
    can_expand = all(
        map(lambda cur, tgt: cur == tgt or cur == 1, tensor.shape, expected_shape)
    )
    if not can_expand:
        context_clause = f" ({context})" if context else ""
        resolved_hint = hint() if callable(hint) else hint
        hint_clause = f" Hint: {resolved_hint}" if resolved_hint else ""
        raise ValueError(
            f"{tensor_name}{context_clause} with shape {tensor.shape} cannot be expanded to expected shape {expected_shape}."
            f"{hint_clause}"
        )
    return tensor.expand(*expected_shape)


def _check_and_expand_block(
    name: str,
    cnt: torch.Tensor | None,
    idx: torch.Tensor | None,
    expected_count_shape: Tuple[int, ...],
    expected_index_shape: Tuple[int, ...],
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if (cnt is None) != (idx is None):
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must both be provided or both be None"
        )
    if cnt is None or idx is None:
        return None, None
    if cnt.dtype != torch.int32 or idx.dtype != torch.int32:
        raise ValueError(f"{name}_block tensors must have dtype torch.int32")
    if cnt.device != idx.device:
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must be on the same device"
        )
    if not cnt.is_cuda or not idx.is_cuda:
        raise ValueError(f"{name}_block tensors must live on CUDA")
    expanded_cnt = _expand_sparsity_tensor(
        cnt, expected_count_shape, f"{name}_block_cnt", context, hint
    )
    # [Note] Allow Compact block sparse indices
    # Allow the last dimension (n_blocks) of idx to be <= expected, since
    # FA4 only accesses indices 0..cnt-1 per query tile. This enables compact
    # index tensors that avoid O(N^2) memory at long sequence lengths.
    if idx.ndim == 4 and idx.shape[3] <= expected_index_shape[3]:
        expected_index_shape = (*expected_index_shape[:3], idx.shape[3])
    expanded_idx = _expand_sparsity_tensor(
        idx, expected_index_shape, f"{name}_block_idx", context, hint
    )
    return expanded_cnt, expanded_idx


def _check_and_expand_metadata_tensor(
    name: str,
    tensor: torch.Tensor | None,
    expected_shape: Tuple[int, ...],
    context: str | None,
    hint: str | Callable[[], str] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as block sparse tensors")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must live on CUDA")
    return _expand_sparsity_tensor(tensor, expected_shape, name, context, hint)


def infer_block_sparse_expected_shapes(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    context: str,
    sparse_block_size_q: int | None = None,
    sparse_block_size_kv: int | None = None,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], int]:
    """Infer shapes and scaling for block-sparse tensors.

    Expectations:
    - mask_block_cnt is (B, H, M) and mask_block_idx is (B, H, M, N).
    - Batch/head dims may be 1 for broadcast, or match the requested sizes.
    - sparse_block_size_kv must match tile_n.
    - sparse_block_size_q must be a multiple of q_stage * tile_m.
    - If sparse_block_size_q is omitted and seqlen_q/num_m_blocks is ambiguous,
      the caller must provide block_size to disambiguate. TODO will make this required in a future PR.
    """
    base_m_block = q_stage * m_block_size
    base_n_block = n_block_size
    if sparse_block_size_kv is None:
        sparse_block_size_kv = base_n_block
    if sparse_block_size_kv != base_n_block:
        raise ValueError(
            f"Block sparse tensors{context} require BLOCK_SIZE_KV={base_n_block}."
        )
    if tensors.mask_block_idx is None:
        raise ValueError(
            "mask_block_cnt and mask_block_idx must be provided for block sparsity."
        )
    num_m_blocks = tensors.mask_block_idx.shape[2]

    if sparse_block_size_q is None:
        sparse_block_size_q = get_sparse_q_block_size(tensors, seqlen_q)
        if sparse_block_size_q is None and base_m_block != 1:
            raise ValueError(
                f"Block sparse tensors{context} require explicit sparse_block_size[0] "
                f"to disambiguate block size for seqlen_q={seqlen_q} and num_m_blocks={num_m_blocks}."
            )
        if sparse_block_size_q is None:
            sparse_block_size_q = ceildiv(seqlen_q, num_m_blocks)

    if sparse_block_size_q % base_m_block != 0:
        raise ValueError(
            f"Block sparse tensors{context} have block size {sparse_block_size_q}, "
            f"which must be a multiple of {base_m_block}."
        )

    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, sparse_block_size_kv)
    q_subtile_factor = sparse_block_size_q // base_m_block
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)

    mask_block_cnt = tensors.mask_block_cnt
    mask_block_idx = tensors.mask_block_idx
    if mask_block_cnt is None or mask_block_idx is None:
        raise ValueError(
            "mask_block_cnt and mask_block_idx must be provided for block sparsity."
        )
    if mask_block_cnt.ndim != 3 or mask_block_idx.ndim != 4:
        raise ValueError(
            f"Block sparse tensors{context} must have shapes (B, H, M) and (B, H, M, N)."
        )
    for dim_name, cur, tgt in (
        ("batch", mask_block_cnt.shape[0], expected_count_shape[0]),
        ("head", mask_block_cnt.shape[1], expected_count_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(
                f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1."
            )
    for dim_name, cur, tgt in (
        ("batch", mask_block_idx.shape[0], expected_index_shape[0]),
        ("head", mask_block_idx.shape[1], expected_index_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(
                f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1."
            )
    if mask_block_cnt.shape[2] != mask_block_idx.shape[2]:
        raise ValueError(
            f"Block sparse tensors{context} must share the same m-block dimension."
        )
    # [Note] Allow Compact block sparse indices: FA4 only accesses indices 0..cnt-1
    # per query tile, so idx.shape[3] can be <= expected_n_blocks.
    if mask_block_idx.shape[3] > expected_n_blocks:
        raise ValueError(
            f"Block sparse tensors{context} n-block dimension must be <= {expected_n_blocks}."
        )
    if expected_m_blocks != num_m_blocks:
        raise ValueError(
            f"Block sparse tensors{context} m-block dimension {num_m_blocks} does not match "
            f"sparse_block_size_q={sparse_block_size_q}. "
            f"Set BlockSparseTensorsTorch.block_size to match the BlockMask BLOCK_SIZE."
        )
    return expected_count_shape, expected_index_shape, q_subtile_factor


def get_block_sparse_expected_shapes_bwd(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    subtile_factor: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Return (expected_count_shape, expected_index_shape) for backward block sparse normalization.

    Backward uses Q-direction indexing (transposed from forward), where shapes are
    indexed by N-blocks first, then M-blocks. The sparse_block_size_q is determined
    by subtile_factor * m_block_size.
    """
    sparse_block_size_q = subtile_factor * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_n_blocks)
    expected_index_shape = (batch_size, num_head, expected_n_blocks, expected_m_blocks)
    return expected_count_shape, expected_index_shape


def normalize_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch,
    *,
    expected_count_shape: Tuple[int, ...],
    expected_index_shape: Tuple[int, ...],
    context: str | None = None,
    hint: str | Callable[[], str] | None = None,
) -> BlockSparseTensorsTorch:
    if tensors.mask_block_cnt is None or tensors.mask_block_idx is None:
        raise ValueError(
            "mask_block_cnt and mask_block_idx must be provided for block sparsity."
        )

    mask_cnt, mask_idx = _check_and_expand_block(
        "mask",
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if mask_cnt is None or mask_idx is None:
        raise ValueError(
            "mask_block_cnt and mask_block_idx must be provided for block sparsity."
        )

    full_cnt, full_idx = _check_and_expand_block(
        "full",
        tensors.full_block_cnt,
        tensors.full_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if full_cnt is not None and mask_cnt.device != full_cnt.device:
        raise ValueError("All block sparse tensors must be on the same device")

    dq_write_order = _check_and_expand_metadata_tensor(
        "dq_write_order",
        tensors.dq_write_order,
        tuple(mask_idx.shape),
        context,
        hint,
        mask_cnt.device,
    )
    dq_write_order_full = _check_and_expand_metadata_tensor(
        "dq_write_order_full",
        tensors.dq_write_order_full,
        tuple(full_idx.shape) if full_idx is not None else expected_index_shape,
        context,
        hint,
        mask_cnt.device,
    )
    spt = tensors.spt
    if spt is not None and not isinstance(spt, bool):
        raise ValueError("spt must be a bool when provided")
    if spt is not None and dq_write_order is None:
        raise ValueError("spt requires dq_write_order to be provided")

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        cu_total_m_blocks=tensors.cu_total_m_blocks,
        cu_block_idx_offsets=tensors.cu_block_idx_offsets,
        block_size=tensors.block_size,
        dq_write_order=dq_write_order,
        dq_write_order_full=dq_write_order_full,
        is_valid_total=tensors.is_valid_total,
        spt=spt,
    )


def is_block_sparsity_enabled(tensors: BlockSparseTensorsTorch) -> bool:
    return any(t is not None for t in (tensors.full_block_cnt, tensors.mask_block_cnt))


def get_block_sparse_broadcast_pattern(
    tensors: BlockSparseTensorsTorch,
) -> Tuple[Tuple[bool, ...], ...] | None:
    """Return broadcast pattern for block sparse tensors by checking actual strides.

    Returns a tuple of broadcast patterns (one per tensor) where each pattern
    is a tuple of bools indicating which dims have stride=0.
    This is used in compile keys to ensure kernels are recompiled when
    broadcast patterns change, since CuTe's mark_layout_dynamic() keeps
    stride=0 as static.

    The tensors should already be expanded/normalized before calling this function.

    Returns None if block sparsity is not enabled.
    """
    if not is_block_sparsity_enabled(tensors):
        return None

    patterns = []
    for tensor in (
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_idx,
        tensors.dq_write_order,
        tensors.dq_write_order_full,
    ):
        if tensor is not None:
            patterns.append(get_broadcast_dims(tensor))
        else:
            patterns.append(None)
    return tuple(patterns)


def normalize_block_sparse_config(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    q_stage: int,
) -> tuple[BlockSparseTensorsTorch, Tuple[Tuple[bool, ...], ...] | None, int]:
    """Validate the block-sparse config, infer expected shapes, and normalize.

    Handles both fixed-length (3D `[B, H, M]` / 4D `[B, H, M, N]`) and varlen
    (2D `[H, total_m_blocks]` / `[H, total_n_blocks]`) layouts. Varlen is
    detected by `tensors.cu_total_m_blocks is not None` and forces
    `q_subtile_factor == 1` (TODO: potentially remove this restriction).
    """
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = None, n_block_size
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity requires sparse_block_size[1]={n_block_size} to match tile_n."
        )
    if tensors.cu_total_m_blocks is not None:
        base_m_block = q_stage * m_block_size
        if sparse_block_size_q is not None and sparse_block_size_q != base_m_block:
            raise ValueError(
                f"Varlen block sparsity requires sparse_block_size[0]={base_m_block} "
                f"(= q_stage * tile_m); got {sparse_block_size_q}."
            )
        total_m_blocks = tensors.mask_block_cnt.shape[-1]
        total_n_blocks = tensors.mask_block_idx.shape[-1]
        expected_count_shape: tuple[int, ...] = (num_head, total_m_blocks)
        expected_index_shape: tuple[int, ...] = (num_head, total_n_blocks)
        q_subtile_factor = 1
    else:
        (
            expected_count_shape,
            expected_index_shape,
            q_subtile_factor,
        ) = infer_block_sparse_expected_shapes(
            tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            context="forward",
            sparse_block_size_q=sparse_block_size_q,
            sparse_block_size_kv=sparse_block_size_kv,
        )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
    )
    return (
        normalized_tensors,
        get_block_sparse_broadcast_pattern(normalized_tensors),
        q_subtile_factor,
    )


def normalize_block_sparse_config_bwd(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    subtile_factor: int,
) -> tuple[BlockSparseTensorsTorch, Tuple[Tuple[bool, ...], ...] | None]:
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = (
            subtile_factor * m_block_size,
            n_block_size,
        )
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_q != subtile_factor * m_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size_q={subtile_factor * m_block_size} "
            f"for subtile_factor={subtile_factor}."
        )
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size[1]={n_block_size} to match tile_n."
        )
    expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes_bwd(
        batch_size,
        num_head,
        seqlen_q,
        seqlen_k,
        m_block_size,
        n_block_size,
        subtile_factor,
    )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
        context="_flash_attn_bwd",
        hint=lambda: (
            f"Backward expects Q-direction block-sparse tensors (q_mask_cnt/q_mask_idx, "
            f"and optionally full_q_cnt/full_q_idx). Regenerate the backward BlockMask with "
            f"BLOCK_SIZE=({subtile_factor * m_block_size}, {n_block_size})."
        ),
    )
    return normalized_tensors, get_block_sparse_broadcast_pattern(normalized_tensors)


def to_cute_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch, enable_tvm_ffi: bool = True
) -> BlockSparseTensors | None:
    """Convert torch block sparsity tensors to CuTe tensors, optionally for tvm ffi"""
    if not is_block_sparsity_enabled(tensors):
        return None
    mask_block_cnt_tensor, mask_block_idx_tensor = [
        to_cute_tensor(
            t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi
        )
        for t in (tensors.mask_block_cnt, tensors.mask_block_idx)
    ]
    full_block_cnt_tensor, full_block_idx_tensor = [
        to_cute_tensor(
            t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi
        )
        if t is not None
        else None
        for t in (tensors.full_block_cnt, tensors.full_block_idx)
    ]
    cu_total_m_blocks_tensor, cu_block_idx_offsets_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=0, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (tensors.cu_total_m_blocks, tensors.cu_block_idx_offsets)
    ]
    dq_write_order_tensor, dq_write_order_full_tensor = [
        to_cute_tensor(
            t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi
        )
        if t is not None
        else None
        for t in (tensors.dq_write_order, tensors.dq_write_order_full)
    ]

    is_valid_total_tensor = (
        to_cute_tensor(
            tensors.is_valid_total,
            assumed_align=4,
            leading_dim=-1,
            enable_tvm_ffi=enable_tvm_ffi,
        )
        if tensors.is_valid_total is not None
        else None
    )

    return BlockSparseTensors(
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
        full_block_cnt_tensor,
        full_block_idx_tensor,
        cu_total_m_blocks_tensor,
        cu_block_idx_offsets_tensor,
        dq_write_order_tensor,
        dq_write_order_full_tensor,
        is_valid_total_tensor,
    )


def prepare_block_sparse_fwd(
    block_sparse_tensors: BlockSparseTensorsTorch | None,
    *,
    pack_gqa: bool,
    cu_seqlens_q: torch.Tensor | None,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int,
    tile_n: int,
    q_stage: int,
    qhead_per_kvhead: int = 1,
) -> tuple[BlockSparseTensorsTorch | None, object, int | None, bool]:
    """Host-side block-sparse preparation for the forward pass.

    Bundles the forward block-sparse host logic (pack_gqa adjustment, varlen
    validation, and config normalization) so the fwd entry point stays clean.

    Returns ``(normalized_tensors, broadcast_pattern, q_subtile_factor, pack_gqa)``.
    When ``block_sparse_tensors is None``, returns ``(None, None, None, pack_gqa)``.
    """
    if block_sparse_tensors is None:
        return None, None, None, pack_gqa

    head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
    packed_num_head = num_head // qhead_per_kvhead
    sparse_num_head = block_sparse_tensors.mask_block_cnt.shape[head_dim_idx]
    if pack_gqa and sparse_num_head not in (1, packed_num_head):
        pack_gqa = False
    if cu_seqlens_q is not None:
        assert (
            block_sparse_tensors.cu_total_m_blocks is not None
        ), "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."

    seqlen_q_norm = seqlen_q * qhead_per_kvhead if pack_gqa else seqlen_q
    num_head_norm = packed_num_head if pack_gqa else num_head
    (
        normalized_tensors,
        broadcast_pattern,
        q_subtile_factor,
    ) = normalize_block_sparse_config(
        block_sparse_tensors,
        batch_size=batch_size,
        num_head=num_head_norm,
        seqlen_q=seqlen_q_norm,
        seqlen_k=seqlen_k,
        block_size=(tile_m, tile_n),
        q_stage=q_stage,
    )
    return normalized_tensors, broadcast_pattern, q_subtile_factor, pack_gqa


def prepare_block_sparse_bwd(
    block_sparse_tensors: BlockSparseTensorsTorch | None,
    *,
    deterministic: bool,
    causal: bool,
    local: bool,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    subtile_factor: int,
) -> tuple[BlockSparseTensorsTorch | None, object, bool]:
    """Host-side block-sparse preparation for the backward pass.

    Bundles the backward block-sparse host logic (config normalization,
    deterministic validation, and the ``spt`` flag) so the bwd entry point stays
    clean. ``spt`` also covers the non-block-sparse case
    (``(causal or local) and deterministic``).

    Returns ``(normalized_tensors, broadcast_pattern, spt)``.
    """
    normalized_tensors = None
    broadcast_pattern = None
    if block_sparse_tensors is not None:
        (
            normalized_tensors,
            broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            subtile_factor=subtile_factor,
        )
        if deterministic:
            if normalized_tensors.dq_write_order is None:
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order in block_sparse_tensors"
                )
            if (
                normalized_tensors.full_block_cnt is not None
                and normalized_tensors.dq_write_order_full is None
            ):
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order_full when full blocks are present"
                )
            if normalized_tensors.spt is None:
                raise ValueError(
                    "deterministic block-sparse backward requires block_sparse_tensors.spt "
                    "to match dq_write_order direction"
                )

    if normalized_tensors is not None and normalized_tensors.spt is not None:
        spt = normalized_tensors.spt and deterministic
    else:
        spt = (causal or local) and deterministic

    return normalized_tensors, broadcast_pattern, spt


def prepare_block_sparse_bwd_loopk(
    block_sparse_tensors: BlockSparseTensorsTorch | None,
    *,
    causal: bool,
    local: bool,
    deterministic: bool,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    pack_gqa: bool = False,
    qhead_per_kvhead: int = 1,
) -> tuple[BlockSparseTensorsTorch | None, object, bool]:
    """Host-side block-sparse preparation for LoopK backward.

    LoopK iterates over K-blocks per Q-tile, so it uses **forward-direction**
    block-sparse tensors (indexed by M-blocks → N-block list), unlike LoopQ
    which uses the transposed Q-direction tensors.

    Returns ``(normalized_tensors, broadcast_pattern, spt)``.
    """
    normalized_tensors = None
    broadcast_pattern = None
    if block_sparse_tensors is not None:
        seqlen_q_eff = seqlen_q * qhead_per_kvhead if pack_gqa else seqlen_q
        expected_m_blocks = ceildiv(seqlen_q_eff, m_block_size)
        expected_n_blocks = ceildiv(seqlen_k, n_block_size)
        expected_count_shape = (batch_size, num_head, expected_m_blocks)
        expected_index_shape = (
            batch_size,
            num_head,
            expected_m_blocks,
            expected_n_blocks,
        )
        normalized_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=expected_count_shape,
            expected_index_shape=expected_index_shape,
            context="_flash_attn_bwd_loopk",
            hint=lambda: (
                f"LoopK backward expects forward-direction block-sparse tensors "
                f"(mask_block_cnt/mask_block_idx indexed by M-blocks). "
                f"Expected count shape {expected_count_shape}, index shape {expected_index_shape}."
            ),
        )
        broadcast_pattern = get_block_sparse_broadcast_pattern(normalized_tensors)

    spt = (causal or local) and deterministic
    return normalized_tensors, broadcast_pattern, spt


def index_sparse_indices_to_block_sparse(
    index_sparse_indices: torch.Tensor,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_kv_heads: int,
    m_block_size: int,
    n_block_size: int,
) -> BlockSparseTensorsTorch:
    """Convert token-level IndexSparse indices to block-level BlockSparseTensors.

    Produces **forward-direction** tensors (per M-block -> N-block list) suitable
    for BWD LoopK.

    Args:
        index_sparse_indices: (total_q, NHK, max_topk) int32 tensor.
            Values are logical KV token positions ``batch * S_kv + token_idx``.
            ``-1`` indicates padding.
    """
    device = index_sparse_indices.device
    total_q, NHK, max_topk = index_sparse_indices.shape
    assert NHK == num_kv_heads
    assert total_q == batch_size * seqlen_q

    M_blocks = ceildiv(seqlen_q, m_block_size)
    N_blocks = ceildiv(seqlen_k, n_block_size)

    valid = index_sparse_indices >= 0
    k_local = index_sparse_indices % seqlen_k
    k_block = (k_local // n_block_size).clamp(0, N_blocks - 1)

    q_idx = torch.arange(total_q, device=device)
    local_q = q_idx % seqlen_q
    m_block_per_q = local_q // m_block_size
    batch_per_q = q_idx // seqlen_q

    b_expand = batch_per_q[:, None, None].expand_as(k_block)
    m_expand = m_block_per_q[:, None, None].expand_as(k_block)
    h_expand = torch.arange(NHK, device=device)[None, :, None].expand_as(k_block)

    b_flat = b_expand[valid].long()
    h_flat = h_expand[valid].long()
    m_flat = m_expand[valid].long()
    k_flat = k_block[valid].long()

    presence = torch.zeros(
        batch_size, NHK, M_blocks, N_blocks, dtype=torch.bool, device=device
    )
    linear_idx = (
        b_flat * (NHK * M_blocks * N_blocks)
        + h_flat * (M_blocks * N_blocks)
        + m_flat * N_blocks
        + k_flat
    )
    presence.view(-1).scatter_(0, linear_idx, True)

    mask_block_cnt = presence.sum(dim=-1).int()
    max_cnt = int(mask_block_cnt.max().item())
    if max_cnt == 0:
        max_cnt = 1

    sorted_indices = presence.int().argsort(dim=-1, descending=True)
    mask_block_idx = sorted_indices[..., :max_cnt].contiguous().to(torch.int32)

    counts_expanded = mask_block_cnt.unsqueeze(-1)
    positions = torch.arange(max_cnt, device=device)
    mask_block_idx = mask_block_idx.where(
        positions < counts_expanded, torch.zeros_like(mask_block_idx)
    )

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
    )


def transpose_block_sparse_tensors(
    fwd_tensors: BlockSparseTensorsTorch,
    batch_size: int,
    num_kv_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    subtile_factor: int = 2,
) -> BlockSparseTensorsTorch:
    """Transpose forward-direction (M→N) block-sparse tensors to backward-direction (N→M).

    Forward: mask_block_cnt (B, NHK, M_blocks), mask_block_idx (B, NHK, M_blocks, max_n)
    Backward: mask_block_cnt (B, NHK, N_blocks), mask_block_idx (B, NHK, N_blocks, max_m_coarse)

    The backward path uses coarser Q blocks: sparse_q = subtile_factor * m_block_size.
    If any fine M-block in a coarse group has a presence entry for a given N-block,
    the coarse M-block is marked present.
    """
    import torch

    device = fwd_tensors.mask_block_cnt.device
    M_blocks_fine = ceildiv(seqlen_q, m_block_size)
    N_blocks = ceildiv(seqlen_k, n_block_size)
    sparse_q = subtile_factor * m_block_size
    M_blocks_coarse = ceildiv(seqlen_q, sparse_q)

    presence = torch.zeros(
        batch_size,
        num_kv_heads,
        M_blocks_fine,
        N_blocks,
        dtype=torch.bool,
        device=device,
    )

    # Include mask blocks in presence
    cnt = fwd_tensors.mask_block_cnt
    idx = fwd_tensors.mask_block_idx
    if cnt.any():
        max_n = idx.shape[-1]
        n_positions = torch.arange(max_n, device=device)
        valid = n_positions[None, None, None, :] < cnt.unsqueeze(-1)
        valid_idx = idx.clamp(0, N_blocks - 1)
        presence.scatter_(3, valid_idx.long() * valid.long(), valid)

    # Include full blocks in presence (previously missing)
    if fwd_tensors.full_block_cnt is not None and fwd_tensors.full_block_cnt.any():
        full_cnt = fwd_tensors.full_block_cnt
        full_idx = fwd_tensors.full_block_idx
        max_n_full = full_idx.shape[-1]
        n_pos_full = torch.arange(max_n_full, device=device)
        valid_full = n_pos_full[None, None, None, :] < full_cnt.unsqueeze(-1)
        valid_full_idx = full_idx.clamp(0, N_blocks - 1)
        presence.scatter_(3, valid_full_idx.long() * valid_full.long(), valid_full)

    if subtile_factor > 1 and M_blocks_fine > M_blocks_coarse:
        pad = M_blocks_coarse * subtile_factor - M_blocks_fine
        if pad > 0:
            presence = torch.nn.functional.pad(presence, (0, 0, 0, pad))
        presence = presence.view(
            batch_size, num_kv_heads, M_blocks_coarse, subtile_factor, N_blocks
        ).any(dim=3)
    elif M_blocks_fine == M_blocks_coarse:
        pass

    presence_t = presence.permute(0, 1, 3, 2).contiguous()

    bwd_cnt = presence_t.sum(dim=-1).int()
    max_m = max(int(bwd_cnt.max().item()), 1)
    sorted_m = presence_t.int().argsort(dim=-1, descending=True)
    bwd_idx = sorted_m[..., :max_m].contiguous().to(torch.int32)

    counts_expanded = bwd_cnt.unsqueeze(-1)
    positions = torch.arange(max_m, device=device)
    bwd_idx = bwd_idx.where(positions < counts_expanded, torch.zeros_like(bwd_idx))

    return BlockSparseTensorsTorch(
        mask_block_cnt=bwd_cnt,
        mask_block_idx=bwd_idx,
        block_size=(sparse_q, n_block_size),
    )


def block_sparse_call_tuple(
    normalized_tensors: BlockSparseTensorsTorch | None,
) -> tuple | None:
    """Flatten normalized block-sparse tensors into the kernel call-arg tuple.

    Shared by both the forward and backward call sites. Returns ``None`` when
    there are no block-sparse tensors.
    """
    if normalized_tensors is None:
        return None
    return (
        normalized_tensors.mask_block_cnt,
        normalized_tensors.mask_block_idx,
        normalized_tensors.full_block_cnt,
        normalized_tensors.full_block_idx,
        normalized_tensors.cu_total_m_blocks,
        normalized_tensors.cu_block_idx_offsets,
        normalized_tensors.dq_write_order,
        normalized_tensors.dq_write_order_full,
        normalized_tensors.is_valid_total,
    )


# ---------------------------------------------------------------------------
# Device-side (CUTE DSL) block-sparse runtime helpers
# ---------------------------------------------------------------------------
#
# Runtime execution helpers used by the CUTE DSL forward/backward kernels to
# produce and consume block-sparse K/V (and Q/dO) loads. Forked from the legacy
# block_sparse_utils module so the kernels no longer depend on it.


@cute.jit
def _get_curr_blocksparse_tensors_varlen(
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    """Varlen path: tensors are 2D [nheads, total_m_blocks] / [nheads, total_n_blocks]."""
    (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        *_,
    ) = blocksparse_tensors
    curr_m_block = seqlen_info.m_block_offset + m_block
    curr_block_idx_offset = (
        seqlen_info.block_idx_offset + m_block * seqlen_info.num_n_blocks
    )
    curr_mask_block_cnt = mask_block_cnt[head_idx, curr_m_block]
    curr_mask_block_idx = cute.domain_offset(
        curr_block_idx_offset, mask_block_idx[head_idx, None]
    )
    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[head_idx, curr_m_block]
        curr_full_block_idx = cute.domain_offset(
            curr_block_idx_offset, full_block_idx[head_idx, None]
        )
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None
    return (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    )


@cute.jit
def _get_curr_blocksparse_tensors(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    """Fixed-length path: tensors are 4D [batch, nheads, m_block, n_block]."""
    (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        *_,
    ) = blocksparse_tensors
    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block, None]
    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None
    return (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    )


@cute.jit
def get_curr_blocksparse_tensors(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    """Extract head, m_block, and batch-local blocksparsity data from blocksparse_tensors"""
    if const_expr(len(blocksparse_tensors.mask_block_cnt.shape) == 2):
        return _get_curr_blocksparse_tensors_varlen(
            head_idx, m_block, blocksparse_tensors, seqlen_info
        )
    return _get_curr_blocksparse_tensors(
        batch_idx, head_idx, m_block, blocksparse_tensors
    )


# NOTE [SM100 block-sparse empty tiles: mbarrier contract]
#
# For block-sparse SM100 forward, a given (m_block, stage) Q tile can have zero active
# KV blocks (total_block_cnt == 0). In that case there is no seqlen_kv iteration, so
# the softmax warp-group has no row stats to publish.
#
# The correction warp-group seeds fully-masked-row stats and runs the usual correction
# epilogue so output/LSE have well-defined values. Both warp-groups must still perform
# the softmax<->correction mbarrier handshake so phases advance correctly across
# empty->empty and empty->non-empty tile sequences.
#
# In the no-sink case, this corresponds to the usual fully-masked-row convention:
# output is zero and LSE is -inf.
#
# Barrier contract (each is `mbar_ptr + <offset> + stage`):
#
# Producer/consumer pairs:
# - `mbar_softmax_corr_full`    : softmax arrive        -> correction wait
# - `mbar_softmax_corr_empty`   : correction arrive     -> softmax wait
# - `mbar_P_full_O_rescaled`    : softmax arrive (+ correction arrive) -> MMA wait
# - `mbar_P_full_2`             : softmax arrive        -> MMA wait
# - `mbar_corr_epi_full_/empty` : correction <-> epilogue (only when epilogue is separate)
#
# Empty tile (`total_block_cnt == 0`):
# - Softmax: skips the seqlen_kv softmax path entirely (no P stores, no `mbar_P_full_*`).
#   It only arrives `mbar_softmax_corr_full` once per stage as a synthetic "no work" signal.
#   At the `softmax_loop` level, softmax unconditionally waits `mbar_softmax_corr_empty`
#   before each tile (when block-sparse) to drain a prior correction arrival and keep
#   phases aligned across non-empty -> empty transitions.
# - Correction: waits `mbar_softmax_corr_full`, seeds stats + runs `correction_epilogue(scale=0)`,
#   and arrives `mbar_softmax_corr_empty` (and `mbar_corr_epi_full_/empty` when applicable).
# - No `mbar_P_full_*` barriers are arrived (no P, no MMA O); only the softmax<->correction
#   (and correction<->epilogue) handshakes advance phases.
#
# Non-empty tile:
# - Softmax: runs `softmax_step` (produces P) and uses `mbar_softmax_corr_full/empty` to
#   publish row_max (during seqlen_kv) and final row stats (once per tile), and to advance phases;
#   arrives `mbar_P_full_*` when P is stored.
# - Correction: waits `mbar_softmax_corr_full`, may rescale/release O, arrives `mbar_softmax_corr_empty`
#   to ack/advance, and arrives `mbar_P_full_O_rescaled` when MMA can proceed.
#
# Backward (SM100):
# - Empty KV tile: for a given `n_block`, `total_m_block_cnt == 0` means no Q tiles contribute.
# - Both the load and compute loops guard all pipeline work on `process_tile`, so empty tiles
#   skip producer/consumer operations entirely (no per-tile mbarrier phase handshake like forward).
# - In the `not dKV_postprocess` path, dK/dV for empty KV tiles are explicitly written as zeros
#   even when `process_tile == False` (see `flash_bwd_sm100.py` `should_zero_dKV`).


@cute.jit
def load_block_list(
    block_indices: cute.Tensor,
    block_count,
    first_block_preloaded: cutlass.Constexpr,
    kv_producer_state,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
):
    """Iterate over the sparse blocks and load K, V into the pipeline.
    For the intra_wg_overlap case, we overlap the loads of K and V. And this
    means we need to pipeline the last V load from the partial block case,
    with the loads for the full blocks. Set first_block_preloaded when the
    caller has already issued the first K load for the list.

    Q is loaded separately on its own mbarrier before this function is called.

    Note:
        we iterate along the block_n indices in reverse.

    Returns:
        Updated kv_producer_state after processing the block list.

    """
    if block_count > 0:
        if const_expr(not intra_wg_overlap):
            for offset in cutlass.range(block_count):
                n_block = block_indices[block_count - 1 - offset]
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(src_idx=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
        else:
            n_block_first = block_indices[block_count - 1]
            if const_expr(not first_block_preloaded):
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_first, producer_state=kv_producer_state)

            for idx in cutlass.range(block_count - 1, unroll=1):
                n_block_prev = block_indices[block_count - 1 - idx]
                n_block = block_indices[block_count - 2 - idx]
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)

    return kv_producer_state


@cute.jit
def finish_overlap_v_load(
    block_indices: cute.Tensor,
    block_count,
    load_V,
    pipeline_v,
    kv_producer_state,
):
    """Load the final V block after overlapped K/V loads."""
    if block_count > 0:
        n_block_last = block_indices[0]
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block_last, producer_state=kv_producer_state)
        kv_producer_state.advance()

    return kv_producer_state


@cute.jit
def sparse_tensor_m_block(
    m_block,
    qhead_per_kvhead: cutlass.Constexpr[int],
    q_subtile_factor: cutlass.Constexpr[int],
):
    """Map packed m_block indices to block-sparse tensor indices."""
    block = m_block
    if const_expr(qhead_per_kvhead != 1):
        block = block // qhead_per_kvhead
    if const_expr(q_subtile_factor != 1):
        block = block // q_subtile_factor
    return block


@cute.jit
def produce_block_sparse_inner_iters(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info: SeqlenInfoQK,
    kv_producer_state,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """Iterate over the mask and full block lists for a single tile.

    Q is loaded separately on its own mbarrier before this function is called.

    The masked (partial) list may leave the last V load pending when intra-warp-group
    overlap is enabled. The first full block must consume that pending V while
    issuing its own K load on the next pipeline stage.

    In the intra-wg-overlap path, the last masked block leaves its V copy in flight
    while we advance the producer state to start the next full K. Either the full list
    overlaps that pending V load, or, if no full blocks exist, we explicitly drain it.

    Args:
        qhead_per_kvhead: Pack-GQA factor. When > 1, m_block is in packed space and
            must be converted to unpacked for sparse tensor indexing.
    """
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    ) = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen_info,
    )

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    if mask_empty:
        # No masked blocks: the full list owns the initial K load.
        kv_producer_state = load_block_list(
            curr_full_block_idx,
            curr_full_block_cnt,
            first_block_preloaded=False,
            kv_producer_state=kv_producer_state,
            load_K=load_K,
            load_V=load_V,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            intra_wg_overlap=intra_wg_overlap,
        )

        if const_expr(intra_wg_overlap) and curr_full_block_cnt > 0:
            kv_producer_state = finish_overlap_v_load(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_V,
                pipeline_v,
                kv_producer_state,
            )
    else:
        # Masked blocks present. When overlap is disabled this fully drains the list.
        kv_producer_state = load_block_list(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            first_block_preloaded=False,
            kv_producer_state=kv_producer_state,
            load_K=load_K,
            load_V=load_V,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            intra_wg_overlap=intra_wg_overlap,
        )

        if full_empty:
            if const_expr(intra_wg_overlap):
                kv_producer_state = finish_overlap_v_load(
                    curr_mask_block_idx,
                    curr_mask_block_cnt,
                    load_V,
                    pipeline_v,
                    kv_producer_state,
                )
        else:
            if const_expr(intra_wg_overlap):
                # Bridge the masked list to the full list by overlapping the pending masked V
                # with the first full K load.
                n_block_mask_last = curr_mask_block_idx[0]
                n_block_full_first = curr_full_block_idx[curr_full_block_cnt - 1]
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_full_first, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_mask_last, producer_state=kv_producer_state_prev)

                kv_producer_state = load_block_list(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    first_block_preloaded=True,
                    kv_producer_state=kv_producer_state,
                    load_K=load_K,
                    load_V=load_V,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    intra_wg_overlap=intra_wg_overlap,
                )

                kv_producer_state = finish_overlap_v_load(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    load_V,
                    pipeline_v,
                    kv_producer_state,
                )
            else:
                # Non-overlap path with both lists: run the full list normally.
                kv_producer_state = load_block_list(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    first_block_preloaded=False,
                    kv_producer_state=kv_producer_state,
                    load_K=load_K,
                    load_V=load_V,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    intra_wg_overlap=intra_wg_overlap,
                )

    return kv_producer_state


@cute.jit
def consume_block_sparse_inner_iters(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info,
    kv_consumer_state,
    mma_pv_fn,
    mma_one_n_block,
    process_first_half_block,
    process_last_half_block,
    mask_fn,
    score_mod_fn,
    O_should_accumulate,
    mask_mod,
    fastdiv_mods,
    intra_wg_overlap: cutlass.Constexpr,
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """Consume the mask and full block lists for a single tile on the consumer side.

    Mirrors `produce_block_sparse_inner_iters` so that the consumer pipeline uses
    the same sparse tensor indexing.

    Args:
        qhead_per_kvhead: Pack-GQA factor. When > 1, m_block is in packed space and
            must be converted to unpacked for sparse tensor indexing.
    """
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    ) = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen_info,
    )

    processed_any = curr_mask_block_cnt + curr_full_block_cnt > 0

    if const_expr(not intra_wg_overlap):
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            warp_scheduler_barrier_sync()
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=mask_n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=fastdiv_mods
                    if cutlass.const_expr(mask_mod is not None)
                    else None,
                ),
                is_first_n_block=True,
            )
            O_should_accumulate = True
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
            if curr_full_block_cnt == 0:
                warp_scheduler_barrier_arrive()

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                    is_first_n_block=True,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            warp_scheduler_barrier_arrive()
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            kv_consumer_state = process_first_half_block(
                n_block=mask_n_block,
                seqlen=seqlen_info,
                kv_consumer_state=kv_consumer_state,
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=fastdiv_mods
                    if cutlass.const_expr(mask_mod is not None)
                    else None,
                ),
                score_mod_fn=score_mod_fn,
                is_first_block=True,
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    seqlen=seqlen_info,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                kv_consumer_state = process_first_half_block(
                    n_block=full_n_block,
                    seqlen=seqlen_info,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    score_mod_fn=score_mod_fn,
                    is_first_block=True,
                )
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    seqlen=seqlen_info,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                )
                O_should_accumulate = True
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    seqlen=seqlen_info,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_mask_block_cnt + curr_full_block_cnt > 0:
            kv_consumer_state = process_last_half_block(
                kv_consumer_state=kv_consumer_state,
                zero_init=not O_should_accumulate,
            )
            O_should_accumulate = True

    return kv_consumer_state, O_should_accumulate, processed_any


@cute.jit
def load_block_list_sm100(
    block_indices: cute.Tensor,
    block_count,
    load_q_with_first: cutlass.Constexpr,
    q_stage: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
):
    """SM100 version of load_block_list (no intra_wg_overlap, no extra_tx_count)."""
    if block_count > 0:
        # First iteration: load Q alongside K if requested
        n_block_first = block_indices[block_count - 1]

        if const_expr(load_q_with_first):
            # SM100 loads Q0 and optionally Q1
            load_Q(block=0, stage=0)
            if const_expr(q_stage == 2):
                load_Q(block=1, stage=1)

        # SM100 doesn't use producer_acquire for pipeline_kv in load path
        # The pipeline barriers are handled inside load_KV
        load_K(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()
        load_V(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()

        # Remaining blocks
        for offset in cutlass.range(1, block_count):
            n_block = block_indices[block_count - 1 - offset]
            load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()
            load_V(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()

    return kv_producer_state


# SM100-specific tile processor using SM100 helpers
@cute.jit
def produce_block_sparse_inner_iters_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
    q_stage: cutlass.Constexpr,
    q_producer_phase: Int32,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    """SM100 entry point for sparse block iteration.

    SM100 uses PipelineTmaUmma which doesn't support extra_tx_count, so we use
    simplified block processing that just calls producer_acquire without extras.

    Args:
        m_block: which tile of m we are processing
        qhead_per_kvhead: Constexpr pack factor
    """
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    ) = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen_info,
    )

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    q_phase_flipped = False

    if mask_empty:
        # No masked blocks: process full list with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_full_block_idx,
            curr_full_block_cnt,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = not full_empty
    else:
        # Process masked blocks with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = True

        if not full_empty:
            # Process full blocks without Q loading
            kv_producer_state = load_block_list_sm100(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_q_with_first=False,
                q_stage=q_stage,
                kv_producer_state=kv_producer_state,
                load_Q=load_Q,
                load_K=load_K,
                load_V=load_V,
                pipeline_kv=pipeline_kv,
            )

    if q_phase_flipped:
        q_producer_phase ^= 1

    return kv_producer_state, q_producer_phase


@cute.jit
def get_total_block_count(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
    seqlen_info: SeqlenInfoQK,
):
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)
    mask_block_cnt, _, full_block_cnt, *_ = blocksparse_tensors

    if const_expr(len(mask_block_cnt.shape) == 2):
        # varlen path: tensors are [num_heads, total_m_block]
        curr_m = seqlen_info.m_block_offset + m_block_sparse
        total = mask_block_cnt[head_idx, curr_m]
        if const_expr(full_block_cnt is not None):
            total += full_block_cnt[head_idx, curr_m]
    else:
        # non-varlen: tensors are [batch, num_heads, m_block]
        total = mask_block_cnt[batch_idx, head_idx, m_block_sparse]
        if const_expr(full_block_cnt is not None):
            total += full_block_cnt[batch_idx, head_idx, m_block_sparse]

    return total


@cute.jit
def handle_block_sparse_empty_tile_correction_sm100(
    tidx: Int32,
    q_stage: cutlass.Constexpr,
    m_block_size: cutlass.Constexpr,
    qhead_per_kvhead,
    pack_gqa: cutlass.Constexpr,
    is_split_kv: cutlass.Constexpr,
    learnable_sink,
    mLSE,
    seqlen_info,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    split_idx: Int32,
    sScale: cute.Tensor,
    stats: list,
    correction_epilogue: Callable,
    thr_mma_pv: cute.core.ThrMma,
    tOtO: cute.Tensor,
    sO: cute.Tensor,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    pipeline_o_epi: cutlass.pipeline.PipelineAsync,
    sm_stats_consumer_phase: Int32,
    o_corr_consumer_phase: Int32,
    corr_epi_producer_phase: Int32,
    softmax_scale_log2: Float32,
    max_offset: Float32,
    max_offset_scale: Float32,
    mO_cur: Optional[cute.Tensor] = None,
    gO: Optional[cute.Tensor] = None,
    gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
):
    """Handle SM100 forward block-sparse tiles with no active KV blocks.

    This path is taken when `total_block_cnt == 0`. The softmax warp-group still
    arrives `mbar_softmax_corr_full` (synthetic "no work") so the correction
    warp-group can:

    - seed fully-masked-row stats (row_sum=1; row_max=-inf when tracked) for LSE
    - run `correction_epilogue` with `scale=0` so the output tile is written as zeros
      (independent of any prior tmem contents)
    - wait on `mbar_softmax_corr_full` and arrive `mbar_softmax_corr_empty`
      (and `mbar_corr_epi_*` when applicable) so phases stay aligned across tiles

    This helper intentionally does not touch `mbar_P_full_*` since no P is produced.
    See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
    """
    LOG2_E = Float32(math.log2(math.e))
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

    for stage in cutlass.range_constexpr(q_stage):
        row_sum_value = Float32(1.0)
        row_max_value = (
            -Float32.inf
            if const_expr(mLSE is not None or learnable_sink is not None)
            else None
        )
        if const_expr(learnable_sink is not None):
            sink_val = -Float32.inf
            if const_expr(not pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            elif tidx < m_block_size:
                q_head_idx = (
                    (q_stage * m_block + stage) * m_block_size + tidx
                ) % qhead_per_kvhead + head_idx * qhead_per_kvhead
                sink_val = Float32(learnable_sink[q_head_idx])
            if sink_val != -Float32.inf and (
                const_expr(not is_split_kv) or split_idx == 0
            ):
                if row_max_value == -Float32.inf:
                    row_max_value = sink_val * (LOG2_E / softmax_scale_log2)
                    row_sum_value = max_offset_scale
                else:
                    row_sum_value = row_sum_value + cute.math.exp2(
                        sink_val * LOG2_E
                        - row_max_value * softmax_scale_log2
                        + max_offset,
                        fastmath=True,
                    )
        if tidx < m_block_size:
            scale_row_idx = tidx + stage * m_block_size
            sScale[scale_row_idx] = row_sum_value
            if const_expr(mLSE is not None or learnable_sink is not None):
                sScale[scale_row_idx + q_stage * m_block_size] = row_max_value
        acc_flag = row_sum_value == Float32(0.0) or row_sum_value != row_sum_value
        stats[stage] = (row_sum_value, row_max_value, acc_flag)

        # See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
        pipeline_sm_stats.consumer_release_w_index(stage)

        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_acquire_w_index_phase(
                stage, corr_epi_producer_phase
            )

        gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
        correction_epilogue(
            thr_mma_pv,
            tOtO[None, None, None, stage],
            tidx,
            stage,
            m_block,
            seqlen_info.seqlen_q,
            Float32(
                0.0
            ),  # zero scale ensures empty tile writes zeros into staged outputs
            sO[None, None, stage],
            mO_cur,
            gO_stage,
            gmem_tiled_copy_O,
        )
        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_commit_w_index(stage)

    sm_stats_consumer_phase ^= 1
    corr_epi_producer_phase ^= 1

    return (
        sm_stats_consumer_phase,
        o_corr_consumer_phase,
        corr_epi_producer_phase,
    )


@cute.jit
def softmax_block_sparse_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    q_stage: cutlass.Constexpr,
    stage_idx: Int32,
    check_m_boundary: bool,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    ) = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen_info,
    )

    total_block_cnt = curr_mask_block_cnt + curr_full_block_cnt

    if total_block_cnt == 0:
        sm_stats_barrier.arrive_w_index(index=stage_idx * 4 + warp_idx)
    else:
        if curr_mask_block_cnt > 0:
            # NOTE: softmax consumes S-tiles in ascending block order (step k -> the
            # k-th block in curr_mask_block_idx), which is the REVERSE of the load
            # order. The per-step mask label must therefore also be ascending so that
            # mask_seqlen's col_limit (= seqlen_k - n_block*n_block_size) and mask_mod's
            # global-column computation land on the correct physical block. We apply
            # mask_seqlen=True on every mask step: full/interior blocks yield a col_limit
            # >= n_block_size (no-op), while the partial tail block (e.g. IndexSparse
            # padding, or the seqlen_k boundary) is masked correctly regardless of its
            # processing position. This mirrors SM90's block-identity-driven padding mask.
            mask_n_block = curr_mask_block_idx[0]
            (
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
                mask_n_block,
                is_first=True,
                mask_fn=partial(
                    mask_fn, mask_seqlen=True, check_q_boundary=check_m_boundary
                ),
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mask_n_block,
                    mask_fn=partial(
                        mask_fn, mask_seqlen=True, check_q_boundary=check_m_boundary
                    ),
                )

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=True,
                    mask_fn=partial(
                        mask_fn_none,
                        mask_seqlen=True,
                        check_q_boundary=check_m_boundary,
                    ),
                )
            else:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=False,
                    mask_fn=partial(
                        mask_fn_none,
                        mask_seqlen=True,
                        check_q_boundary=check_m_boundary,
                    ),
                )
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    mask_fn=partial(
                        mask_fn_none,
                        mask_seqlen=False,
                        check_q_boundary=check_m_boundary,
                    ),
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        total_block_cnt == 0,
    )


# ---------------------------------------------------------------------------
# Backward-specific block-sparse helpers (SM90 / SM100)
# ---------------------------------------------------------------------------
#
# In backward, iteration is transposed compared to forward:
# - Forward: outer loop over m_blocks (Q tiles), inner loop over n_blocks (KV tiles)
# - Backward: outer loop over n_blocks (KV tiles), inner loop over m_blocks (Q tiles)
#
# The backward block-sparse tensors use "Q direction" indexing:
# - q_block_cnt[batch, head, n_block] → count of m_blocks to process for this KV tile
# - q_block_idx[batch, head, n_block, :] → indices of m_blocks to process
#


@cute.jit
def get_total_q_block_count_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Count total tile iterations for given n_block (KV tile) in backward."""
    q_block_cnt, _, full_block_cnt, _, *_ = blocksparse_tensors
    total = q_block_cnt[batch_idx, head_idx, n_block]
    if const_expr(full_block_cnt is not None):
        total = total + full_block_cnt[batch_idx, head_idx, n_block]
    return total * subtile_factor


@cute.jit
def produce_block_sparse_q_loads_bwd_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    # Pipeline states (will be returned after advancing)
    producer_state_Q_LSE,
    producer_state_dO_dPsum,
    # Pipelines
    pipeline_Q,
    pipeline_LSE,
    pipeline_dO,
    pipeline_dPsum,
    # Load functions
    load_K,
    load_V,
    load_Q,
    load_dO,
    copy_stats,
    # Global tensors for LSE/dPsum
    gLSE,
    sLSE,
    gdPsum,
    sdPsum,
    # TMA copy bytes for extra_tx_count
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    # Flags for which loads to perform
    should_load_Q: cutlass.Constexpr,
    should_load_dO: cutlass.Constexpr,
    # Subtiling factor and bounds
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """SM100 backward block sparse loading with subtiling.

    Returns updated (producer_state_Q_LSE, producer_state_dO_dPsum).
    First iteration loads K/V alongside Q/dO; subsequent iterations load only Q/dO.
    """
    (
        curr_q_cnt,
        curr_q_idx,
        curr_full_cnt,
        curr_full_idx,
        loop_count,
    ) = get_block_sparse_iteration_info_bwd(
        blocksparse_tensors, batch_idx, head_idx, n_block, subtile_factor, m_block_max
    )

    for iter_idx in cutlass.range(loop_count, unroll=1):
        m_block, _ = get_m_block_from_iter_bwd(
            iter_idx,
            curr_q_cnt,
            curr_q_idx,
            curr_full_cnt,
            curr_full_idx,
            subtile_factor,
            m_block_max,
        )
        m_block_safe = m_block
        if m_block_max > 0:
            m_block_safe = cutlass.min(m_block, m_block_max - 1)

        if iter_idx == 0:
            # First block: load K/V alongside Q/dO
            if const_expr(should_load_Q):
                pipeline_Q.producer_acquire(
                    producer_state_Q_LSE, extra_tx_count=tma_copy_bytes_K
                )
                load_K(
                    tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE)
                )
                load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                pipeline_Q.producer_commit(producer_state_Q_LSE)
                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block_safe],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(
                            producer_state_Q_LSE
                        ),
                    )
                producer_state_Q_LSE.advance()
            if const_expr(should_load_dO):
                pipeline_dO.producer_acquire(
                    producer_state_dO_dPsum, extra_tx_count=tma_copy_bytes_V
                )
                load_V(
                    tma_bar_ptr=pipeline_dO.producer_get_barrier(
                        producer_state_dO_dPsum
                    )
                )
                load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                pipeline_dO.producer_commit(producer_state_dO_dPsum)
                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block_safe],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(
                            producer_state_dO_dPsum
                        ),
                    )
                producer_state_dO_dPsum.advance()
        else:
            # Subsequent blocks: just load Q/dO (K/V already loaded)
            if const_expr(should_load_Q):
                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                pipeline_Q.producer_commit(producer_state_Q_LSE)
                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block_safe],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(
                            producer_state_Q_LSE
                        ),
                    )
                producer_state_Q_LSE.advance()
            if const_expr(should_load_dO):
                pipeline_dO.producer_acquire(producer_state_dO_dPsum)
                load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                pipeline_dO.producer_commit(producer_state_dO_dPsum)
                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block_safe],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(
                            producer_state_dO_dPsum
                        ),
                    )
                producer_state_dO_dPsum.advance()

    return producer_state_Q_LSE, producer_state_dO_dPsum


@cute.jit
def get_block_sparse_iteration_info_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Extract block-sparse iteration info for backward pass.

    Returns (curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count).
    """
    q_cnt, q_idx, full_cnt, full_idx, *_ = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    sparse_block_count = curr_q_cnt
    if const_expr(full_cnt is not None):
        sparse_block_count = sparse_block_count + curr_full_cnt
    total_count = sparse_block_count * subtile_factor

    return curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count


@cute.jit
def get_m_block_from_iter_bwd(
    iter_idx,
    curr_q_cnt,
    curr_q_idx: cute.Tensor,
    curr_full_cnt,
    curr_full_idx: Optional[cute.Tensor],
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Derive m_block index and is_full_block flag from iteration index.

    Returns (m_block, is_full_block):
        - m_block: The actual Q-tile block index
        - is_full_block: True if this is a full block (no mask_mod needed)
    """
    sparse_iter_idx = iter_idx // subtile_factor
    subtile_offset = iter_idx % subtile_factor

    sparse_m_block = Int32(0)
    is_full_block = False
    if const_expr(curr_full_idx is not None):
        if sparse_iter_idx < curr_q_cnt:
            sparse_m_block = curr_q_idx[sparse_iter_idx]
        else:
            sparse_m_block = curr_full_idx[sparse_iter_idx - curr_q_cnt]
            is_full_block = True
    else:
        sparse_m_block = curr_q_idx[sparse_iter_idx]

    return sparse_m_block * subtile_factor + subtile_offset, is_full_block


@cute.jit
def _load_q_do_block_sm90(
    m_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    load_kv: bool,
):
    """Load one Q/dO block, optionally loading K/V on first iteration."""
    if load_kv:
        pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=tma_copy_bytes_K)
        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
    else:
        pipeline_Q.producer_acquire(producer_state_Q)
    load_Q(m_block, producer_state=producer_state_Q)
    load_LSE(m_block, producer_state=producer_state_Q)

    producer_state_dO_cur = (
        producer_state_dO if const_expr(not Q_stage_eq_dO_stage) else producer_state_Q
    )
    if load_kv:
        pipeline_dO.producer_acquire(
            producer_state_dO_cur, extra_tx_count=tma_copy_bytes_V
        )
        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
    else:
        pipeline_dO.producer_acquire(producer_state_dO_cur)
    load_dO(m_block, producer_state=producer_state_dO_cur)
    load_dPsum(m_block, producer_state=producer_state_dO_cur)

    producer_state_Q.advance()
    producer_state_dO.advance()
    return producer_state_Q, producer_state_dO


@cute.jit
def produce_block_sparse_q_loads_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
):
    """SM90 backward block sparse loading with separate partial/full loops.

    K/V are loaded with the first valid block. Iterates partial blocks first,
    then full blocks, matching consumer order.

    Returns updated (producer_state_Q, producer_state_dO).
    """
    q_cnt, q_idx, full_cnt, full_idx, *_ = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    kv_loaded = False

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                m_block,
                producer_state_Q,
                producer_state_dO,
                pipeline_Q,
                pipeline_dO,
                load_K,
                load_V,
                load_Q,
                load_dO,
                load_LSE,
                load_dPsum,
                tma_copy_bytes_K,
                tma_copy_bytes_V,
                Q_stage_eq_dO_stage,
                load_kv=not kv_loaded,
            )
            kv_loaded = True

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                    m_block,
                    producer_state_Q,
                    producer_state_dO,
                    pipeline_Q,
                    pipeline_dO,
                    load_K,
                    load_V,
                    load_Q,
                    load_dO,
                    load_LSE,
                    load_dPsum,
                    tma_copy_bytes_K,
                    tma_copy_bytes_V,
                    Q_stage_eq_dO_stage,
                    load_kv=not kv_loaded,
                )
                kv_loaded = True

    return producer_state_Q, producer_state_dO


@cute.jit
def consume_block_sparse_mma_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    consumer_state_Q,
    consumer_state_dO,
    mma_one_m_block_fn,
    mask,
    mask_mod,
    is_causal: cutlass.Constexpr,
    is_local: cutlass.Constexpr,
    thr_mma_SdP,
    score_mod_fn=None,
    score_mod_bwd_fn=None,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    aux_tensors=None,
    fastdiv_mods=(None, None),
):
    """SM90 backward block sparse MMA consumption with separate partial/full loops.

    Partial blocks are processed first (with mask_mod applied), then full blocks
    (without mask_mod). This ensures mask_mod is only applied where needed.

    Returns updated (consumer_state_Q, consumer_state_dO).
    """
    q_cnt, q_idx, full_cnt, full_idx, *_ = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    dKV_accumulate = False

    mask_fn_partial = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        n_block=n_block,
        thr_mma=thr_mma_SdP,
        mask_seqlen=True,
        mask_causal=is_causal,
        mask_local=is_local,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
    )

    mask_fn_full = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        n_block=n_block,
        thr_mma=thr_mma_SdP,
        mask_seqlen=True,
        mask_causal=is_causal,
        mask_local=is_local,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
    )

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                m_block,
                consumer_state_Q,
                consumer_state_dO,
                mask_fn=mask_fn_partial,
                score_mod_fn=score_mod_fn,
                score_mod_bwd_fn=score_mod_bwd_fn,
                dKV_accumulate=dKV_accumulate,
            )
            dKV_accumulate = True

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                    m_block,
                    consumer_state_Q,
                    consumer_state_dO,
                    mask_fn=mask_fn_full,
                    score_mod_fn=score_mod_fn,
                    score_mod_bwd_fn=score_mod_bwd_fn,
                    dKV_accumulate=dKV_accumulate,
                )
                dKV_accumulate = True

    return consumer_state_Q, consumer_state_dO


@cute.jit
def _store_one_dQaccum_sm90(
    m_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
):
    """Store dQaccum for a single m_block."""
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        cute.arch.cp_async_bulk_wait_group(
            num_dQ_warp_groups - 1 - warp_group_idx, read=True
        )
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
        with cute.arch.elect_one():
            copy_utils.cpasync_reduce_bulk_add_f32(
                sdQaccum[None, warp_group_idx].iterator,
                gdQaccum[(None, warp_group_idx), m_block].iterator,
                tma_copy_bytes_dQ,
            )
        cute.arch.cp_async_bulk_commit_group()


@cute.jit
def dQacc_store_block_sparse_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
):
    """SM90 backward block sparse dQaccum store with separate partial/full loops.

    Iterates partial blocks first, then full blocks, matching producer/consumer order.
    """
    q_cnt, q_idx, full_cnt, full_idx, *_ = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            _store_one_dQaccum_sm90(
                m_block,
                sdQaccum,
                gdQaccum,
                num_dQ_warp_groups,
                num_threads_per_warp_group,
                tma_copy_bytes_dQ,
            )

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                _store_one_dQaccum_sm90(
                    m_block,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                )


# ===========================================================================
# IndexSparse token-level tile preparation
# ===========================================================================


class IndexSparseTilesTorch(NamedTuple):
    """Token-level tile data for IndexSparse attention.

    tile_token_indices: (B, NHK, M_blocks, max_tokens_per_tile) int32
        Maps each (batch, kv_head, q_tile, local_row) to a global K token index.
        When inner_load_mode=Tma, each 128-token chunk is a physically contiguous
        block and the first element of each chunk encodes the block offset.
    scheduling_bst: BlockSparseTensorsTorch
        Block-sparse scheduling tensors derived from the token indices.
    is_valid_total: torch.Tensor
        (B, NHK, M_blocks) int32 — number of unique valid tokens per Q-tile.
    inner_load_mode: InnerLoadMode
        Load strategy: Tma (block-aligned) or CpAsync (per-token scatter).
    inner_store_mode: InnerStoreMode
        BWD dK/dV store strategy: Tma (TMA reduce-add), Tma1d (bulk scatter), etc.
    """

    tile_token_indices: torch.Tensor
    scheduling_bst: "BlockSparseTensorsTorch"
    is_valid_total: torch.Tensor
    inner_load_mode: InnerLoadMode = InnerLoadMode.CpAsync
    inner_store_mode: InnerStoreMode = InnerStoreMode.Tma1d


def prepare_index_sparse_tiles(
    index_sparse_indices: torch.Tensor,
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_kv_heads: int,
    num_q_heads: int,
    m_block_size: int = 128,
    n_block_size: int = 128,
    pack_gqa: bool = False,
    sparse_k_block_size: int = 1,
) -> IndexSparseTilesTorch:
    """Convert raw index_sparse_indices to tile-based IndexSparse tensors.

    Args:
        index_sparse_indices: (B, NHQ, SQ, TOPK) int32 — per-query token indices into K.
        Other args define the tiling geometry.

    Returns:
        IndexSparseTilesTorch with scheduling BST and tile_token_indices.
    """
    device = index_sparse_indices.device
    B = batch_size
    NHQ = num_q_heads
    NHK = num_kv_heads
    qhead_per_kvhead = NHQ // NHK
    TOPK = index_sparse_indices.shape[-1]

    # Determine M_blocks (Q tiles)
    if pack_gqa:
        M_blocks = ceildiv(seqlen_q * qhead_per_kvhead, m_block_size)
    else:
        M_blocks = ceildiv(seqlen_q, m_block_size)

    # For each Q-tile, collect the union of all token indices across queries in that tile
    # and pad to a multiple of n_block_size.
    max_tokens_per_tile = ceildiv(TOPK, n_block_size) * n_block_size

    # Fill with -1 sentinel (like SM90 CUTLASS): invalid positions load from
    # token 0 safely, and the kernel masks them via is_valid_total / seqlen_k.
    tile_token_indices = torch.full(
        (B, NHK, M_blocks, max_tokens_per_tile), -1, dtype=torch.int32, device=device
    )
    is_valid_total = torch.zeros((B, NHK, M_blocks), dtype=torch.int32, device=device)

    # Build tile_token_indices: for each Q-tile, take TOPK indices from the first query
    # in that tile (all queries in same tile share same K set for token-level IS).
    for b in range(B):
        for h_kv in range(NHK):
            for m_blk in range(M_blocks):
                if pack_gqa:
                    # PackGQA: m_block covers qhead_per_kvhead * m_block_size queries
                    q_start = m_blk * m_block_size // qhead_per_kvhead
                    h_q = (
                        m_blk * m_block_size % qhead_per_kvhead
                        + h_kv * qhead_per_kvhead
                    )
                    if q_start >= seqlen_q:
                        continue
                    h_q = min(h_q, NHQ - 1)
                else:
                    q_start = m_blk * m_block_size
                    h_q = h_kv * qhead_per_kvhead  # Use first Q head in group
                    if q_start >= seqlen_q:
                        continue

                # Take indices from the representative query
                indices = index_sparse_indices[b, h_q, q_start, :TOPK]
                valid_count = min(TOPK, seqlen_k)
                tile_token_indices[b, h_kv, m_blk, :valid_count] = indices[:valid_count]
                is_valid_total[b, h_kv, m_blk] = valid_count

    # Build scheduling BST: K-iteration count per Q-tile
    k_iter_count = ceildiv(TOPK, n_block_size)
    mask_block_cnt = torch.full(
        (B, NHK, M_blocks), k_iter_count, dtype=torch.int32, device=device
    )
    # Sequential block indices: 0, 1, 2, ...
    mask_block_idx = (
        torch.arange(k_iter_count, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, NHK, M_blocks, k_iter_count)
        .contiguous()
    )

    # For IS-TMA (kbs>=128), classify fully-valid blocks as 'full' to skip
    # per-tile softmax masking. Only the tail block needs masking if TOPK is
    # not a multiple of n_block_size (padding with -1 sentinels).
    has_tail_mask = (TOPK % n_block_size) != 0
    if sparse_k_block_size >= n_block_size and k_iter_count > 0:
        # IS-TMA: most blocks are full, at most 1 mask block for the tail
        n_full = k_iter_count - (1 if has_tail_mask else 0)
        n_mask = k_iter_count - n_full
        full_block_cnt = torch.full(
            (B, NHK, M_blocks), n_full, dtype=torch.int32, device=device
        )
        full_block_idx = (
            torch.arange(n_full, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, NHK, M_blocks, n_full)
            .contiguous()
        )
        mask_block_cnt = torch.full(
            (B, NHK, M_blocks), n_mask, dtype=torch.int32, device=device
        )
        if n_mask > 0:
            mask_block_idx = torch.full(
                (B, NHK, M_blocks, 1),
                k_iter_count - 1,
                dtype=torch.int32,
                device=device,
            )
        else:
            mask_block_idx = torch.zeros(
                (B, NHK, M_blocks, 1),
                dtype=torch.int32,
                device=device,
            )
    else:
        # IS-scatter or zero-block: all blocks are mask blocks (original behavior)
        full_block_cnt = torch.zeros(
            (B, NHK, M_blocks),
            dtype=torch.int32,
            device=device,
        )
        full_block_idx = torch.zeros(
            (B, NHK, M_blocks, 1),
            dtype=torch.int32,
            device=device,
        )
    scheduling_bst = BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(m_block_size, n_block_size),
        is_valid_total=is_valid_total,
    )

    # Select inner load mode based on sparse_k_block_size:
    # When kbs >= n_block_size (128), each tile_token_indices chunk is a physically
    # contiguous block → TMA can be used instead of per-token scatter.
    inner_load_mode = (
        InnerLoadMode.Tma
        if sparse_k_block_size >= n_block_size
        else InnerLoadMode.CpAsync
    )

    return IndexSparseTilesTorch(
        tile_token_indices=tile_token_indices,
        scheduling_bst=scheduling_bst,
        is_valid_total=is_valid_total,
        inner_load_mode=inner_load_mode,
        inner_store_mode=(
            InnerStoreMode.Tma
            if inner_load_mode == InnerLoadMode.Tma
            else InnerStoreMode.Tma1d
        ),
    )
