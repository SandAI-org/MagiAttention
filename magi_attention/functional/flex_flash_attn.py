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

from contextlib import contextmanager

import torch
from packaging import version

from magi_attention import env
from magi_attention.common import AttnForwardMeta
from magi_attention.common.enum import AttnSinkLayout, MagiAttentionKernelBackend
from magi_attention.common.ranges import AttnRanges
from magi_attention.env.general import is_profile_mode_enable, is_sanity_check_enable
from magi_attention.meta.collection.calc_meta import FA4AttnArg
from magi_attention.utils import nvtx

from ._flex_flash_attn_jit import _snapshot_env, get_ffa_jit_mod
from .fa4 import fa4_bwd, fa4_fwd, is_fa4_installed

is_magi_attn_ext_installed = False
try:
    from magi_attention import magi_attn_ext  # type: ignore[attr-defined]

    is_magi_attn_ext_installed = True
except ImportError:
    pass


# copied from https://github.com/Dao-AILab/flash-attention/blob/v2.8.2/flash_attn/flash_attn_interface.py#L56-73
is_torch_compile_supported = version.parse(torch.__version__) >= version.parse("2.4.0")
if is_torch_compile_supported:
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:

    def noop_custom_op_wrapper(
        name, fn=None, /, *, mutates_args, device_types=None, schema=None
    ):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper

# sparse_k_block_size is a compile-time constant for JIT kernel selection.
# Passed via module-level var to avoid adding it to the custom_op schema
# (PyTorch custom_op has issues with extra int params shifting mutated tensor indices).
_ffa_sparse_k_block_size: int = 1

profile_mode = is_profile_mode_enable()
if profile_mode:
    assert (
        is_magi_attn_ext_installed
    ), "magi_attn_ext must be installed when profile mode is enabled."


# -------------------       helpers   ------------------- #


def maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def merge_ranges(
    outer_ranges: torch.Tensor, inner_ranges: torch.Tensor, attn_type_map: torch.Tensor
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Sorts and deduplicates range tensors that represent Q-K attention block pairs.

    Args:
        outer_ranges (torch.Tensor): A tensor of shape `[num_ranges, 2]`,
            representing the outer ranges (e.g., Q-block ranges).
        inner_ranges (torch.Tensor): A tensor of shape `[num_ranges, 2]`,
            where each row is paired with the corresponding row in
            `outer_ranges`, representing the inner ranges (e.g., K-block ranges).
        attn_type_map (torch.Tensor): A tensor of shape `[num_ranges]`,
            representing the attn_type of each range.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A tuple containing five tensors:
        - merge_outer_ranges (torch.Tensor): The consecutive and unique ranges
          extracted from `outer_ranges`. Shape: `[num_ranges, 2]`.
        - sorted_outer_ranges (torch.Tensor): The original `outer_ranges` tensor
          after being sorted. Shape: `[num_ranges, 2]`.
        - sorted_inner_ranges (torch.Tensor): The original `inner_ranges` tensor,
          reordered to match the sorting of `outer_ranges`. Shape: `[num_ranges, 2]`.
        - range_map (torch.Tensor): The inverse index map. A tensor of shape
          `[num_ranges]`, where the value `range_map[i]` is the index of
          `sorted_outer_ranges[i]` in the `merge_outer_ranges` tensor.
        - unique_count (torch.Tensor): A scalar tensor containing a single
          integer, representing the number of unique ranges in `merge_outer_ranges`.

    Example:
        >>> outer_ranges = torch.tensor([[20, 30], [10, 20], [10, 20], [20, 30]], device='cuda', type=torch.int32)
        >>> inner_ranges = torch.tensor([[100, 110], [120, 130], [140, 150], [160, 170]], device='cuda', type=torch.int32)
        >>> attn_type_map = torch.tensor([0, 1, 0, 0], device='cuda', type=torch.int32)
        >>> (
        ...     merge_outer_ranges,
        ...     sorted_outer_ranges,
        ...     sorted_inner_ranges,
        ...     sorted_attn_type_map,
        ...     range_map,
        ...     unique_count,
        ... ) = merge_ranges(outer_ranges, inner_ranges, attn_type_map)
        >>> print("Unique Merged Outer Ranges:", merge_outer_ranges)
        Unique Merged Outer Ranges:
         tensor([[10, 20],
                [20, 30],
                [0, 0],
                [0, 0]])
        >>> print("Sorted Outer Ranges:", sorted_outer_ranges)
        Sorted Outer Ranges:
         tensor([[10, 20],
                [10, 20],
                [20, 30],
                [20, 30]])
        >>> print("Sorted Inner Ranges (paired with sorted outer):", sorted_inner_ranges)
        Sorted Inner Ranges (paired with sorted outer):
         tensor([[120, 130],
                [140, 150],
                [100, 110],
                [160, 170]])
        >>> print("Sorted Attention Type Map:", sorted_attn_type_map)
        Sorted Attention Type Map:
         tensor([1, 0, 0, 0])
        >>> print("Range Map (inverse indices):", range_map)
        Range Map (inverse indices):
         tensor([0, 2, 0, 0])
        >>> print("Unique Count:", unique_count)
        Unique Count:
         tensor(2, dtype=torch.int32)
    """
    assert (
        is_magi_attn_ext_installed
    ), "magi_attn_ext must be installed to use merge_ranges function."

    sorted_idx, is_sorted = magi_attn_ext.argsort_ranges(outer_ranges)
    # Reorder q/k ranges and attn_type_map in a single kernel based on the sorted index.
    (
        sorted_outer_ranges,
        sorted_inner_ranges,
        sorted_attn_type_map,
    ) = magi_attn_ext.reorder_ranges_and_attn_type_maps(
        outer_ranges, inner_ranges, attn_type_map, sorted_idx, is_sorted
    )

    if attn_type_map is None:
        sorted_attn_type_map = None

    (
        merge_outer_ranges,
        range_map,
        unique_count,
    ) = magi_attn_ext.unique_consecutive_pairs(sorted_outer_ranges)

    return (
        merge_outer_ranges,
        sorted_outer_ranges,
        sorted_inner_ranges,
        sorted_attn_type_map,
        range_map,
        unique_count,
    )


@contextmanager
def maybe_profile_ffa_ctx(event_name: str):
    if profile_mode:
        magi_attn_ext.start_event(event_name)

    yield

    if profile_mode:
        magi_attn_ext.stop_event(event_name)


# -------------------       ffa forward   ------------------- #


@_torch_custom_op_wrapper(
    "flex_flash_attn::_flex_flash_attn_forward_compilable",
    # NOTE: had better NOT use "out" in args since it is a reserved special arg for torch.compile
    mutates_args=("out_", "lse", "max_logits"),
    device_types="cuda",
)
def _flex_flash_attn_forward_compilable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: str,
    out_: torch.Tensor,
    lse: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    softmax_scale: float,
    softcap: float,
    out_type: torch.dtype | None,
    disable_fwd_atomic_reduction: bool,
    deterministic: bool,
    sm_margin: int,
    kblock_m: int | None,
    kblock_n: int | None,
    max_outer_range_width: int | None,
    range_merge: bool,
    merge_q_ranges: torch.Tensor | None,
    fwd_qk_map: torch.Tensor | None,
    fwd_unique_count: torch.Tensor | None,
    swap_ab: bool,
    pack_gqa: bool,
    block_sparse: bool,
    index_sparse: bool,
    index_sparse_indices: torch.Tensor | None,
    inner_indices_cnt: int,
    return_max_logits: bool,
    max_logits: torch.Tensor | None,
) -> None:
    """torch.ops.flex_flash_attn._flex_flash_attn_forward_compilable"""
    mod = get_ffa_jit_mod(
        direction="fwd",
        head_dim=q.shape[-1],
        compute_dtype=q.dtype,
        output_dtype=out_type
        or (q.dtype if disable_fwd_atomic_reduction else torch.float32),
        softcap=softcap > 0.0,
        disable_atomic_reduction=disable_fwd_atomic_reduction,
        disable_dq_atomic_reduction=False,
        deterministic=deterministic,
        # NOTE: since torch compile does not support tuple args,
        # we make a detour to reconstruct ref_block_size here
        ref_block_size=(kblock_m, kblock_n)
        if kblock_m is not None and kblock_n is not None
        else None,
        range_merge=range_merge,
        swap_ab=swap_ab,
        pack_gqa=pack_gqa,
        cat_gqa=False,
        pack_gqa_factor=q.size(1) // k.size(1),
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        profile_mode=profile_mode,
        return_max_logits=return_max_logits,
        sparse_k_block_size=_ffa_sparse_k_block_size,
        _env_snapshot=_snapshot_env(),
    )
    # Call for side effects: out_, lse, max_logits are mutated in place (mutates_args).
    mod.fwd(
        q,
        k,
        v,
        sink,
        out_,
        lse,
        max_logits,
        q_ranges,
        k_ranges,
        attn_type_map,
        max_outer_range_width,
        # for range merge
        merge_q_ranges,
        fwd_qk_map,
        fwd_unique_count,
        # for IndexSparse direct path
        index_sparse_indices,
        inner_indices_cnt,
        _ffa_sparse_k_block_size if index_sparse else 1,
        # for others
        softmax_scale,
        softcap,
        out_type,
        sink_layout,
        sm_margin,
    )


@_torch_register_fake_wrapper("flex_flash_attn::_flex_flash_attn_forward_compilable")
def _flex_flash_attn_forward_compilable_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: str,
    out_: torch.Tensor,
    lse: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    softmax_scale: float,
    softcap: float,
    out_type: torch.dtype | None,
    disable_fwd_atomic_reduction: bool,
    deterministic: bool,
    sm_margin: int,
    kblock_m: int | None,
    kblock_n: int | None,
    max_outer_range_width: int | None,
    range_merge: bool,
    merge_q_ranges: torch.Tensor | None,
    qk_map: torch.Tensor | None,
    fwd_unique_count: torch.Tensor | None,
    swap_ab: bool,
    pack_gqa: bool,
    block_sparse: bool,
    index_sparse: bool,
    index_sparse_indices: torch.Tensor | None,
    inner_indices_cnt: int,
    return_max_logits: bool,
    max_logits: torch.Tensor | None,
) -> None:
    pass


@nvtx.instrument_nvtx
def _flex_flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: AttnSinkLayout,
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    softmax_scale: float,
    softcap: float,
    out_type: torch.dtype | None,
    disable_fwd_atomic_reduction: bool,
    deterministic: bool,
    sm_margin: int,
    ref_block_size: tuple[int, int] | None = None,
    max_outer_range_width: int | None = None,
    range_merge: bool = False,
    merge_q_ranges: torch.Tensor | None = None,
    fwd_qk_map: torch.Tensor | None = None,
    fwd_unique_count: torch.Tensor | None = None,
    swap_ab: bool = False,
    pack_gqa: bool = False,
    block_sparse: bool = False,
    index_sparse: bool = False,
    index_sparse_indices: torch.Tensor | None = None,
    inner_indices_cnt: int = 0,
    sparse_k_block_size: int = 1,
    return_max_logits: bool = False,
    max_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, AttnForwardMeta]:
    if profile_mode:  # NOTE: stop_event is called inside the kernel
        magi_attn_ext.start_event("fwd_prepare")

    # make all input tensors contiguous before initializing output buffers
    q, k, v, sink, q_ranges, k_ranges = [
        maybe_contiguous(x) for x in (q, k, v, sink, q_ranges, k_ranges)
    ]

    out = (
        torch.empty_like(
            q,
            dtype=out_type
            or (q.dtype if disable_fwd_atomic_reduction else torch.float32),
            device=q.device,
        )
        if out is None
        else out
    )
    lse = (
        torch.full(
            (q.size(0), q.size(1)),
            fill_value=float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        if lse is None
        else lse
    )
    if return_max_logits and max_logits is None:
        max_logits = torch.full(
            (q.size(1),),
            fill_value=float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    if return_max_logits:
        assert q.size(1) <= 128, (
            f"num_qheads ({q.size(1)}) must be <= 128 because the epilogue shmem "
            "for max_logits reduction is fixed at 128 in C++ code. You can increase "
            "the shmem size by increasing the `smem_max_logits` in `epilogue_fwd.hpp`."
        )

    if ref_block_size is not None:
        kblock_m, kblock_n = ref_block_size
    else:
        kblock_m = None
        kblock_n = None

    # NOTE: we can not directly compile `_flex_flash_attn_forward`
    # since torch.compile does not allow returning the mutated args (out, lse)
    if not torch.compiler.is_compiling():
        global _ffa_sparse_k_block_size
        _ffa_sparse_k_block_size = sparse_k_block_size
    _flex_flash_attn_forward_compilable(
        q=q,
        k=k,
        v=v,
        sink=sink,
        sink_layout=sink_layout,
        out_=out,
        lse=lse,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=softmax_scale,
        softcap=softcap,
        out_type=out_type,
        disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
        deterministic=deterministic,
        sm_margin=sm_margin,
        kblock_m=kblock_m,
        kblock_n=kblock_n,
        max_outer_range_width=max_outer_range_width,
        range_merge=range_merge,
        merge_q_ranges=merge_q_ranges,
        fwd_qk_map=fwd_qk_map,
        fwd_unique_count=fwd_unique_count,
        swap_ab=swap_ab,
        pack_gqa=pack_gqa,
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        index_sparse_indices=index_sparse_indices,
        inner_indices_cnt=inner_indices_cnt,
        return_max_logits=return_max_logits,
        max_logits=max_logits,
    )

    return out, AttnForwardMeta(lse=lse, max_logits=max_logits)


# -------------------       ffa backward   ------------------- #


@_torch_custom_op_wrapper(
    "flex_flash_attn::_flex_flash_attn_backward_compilable",
    mutates_args=("dq", "dk", "dv", "dsink"),
    device_types="cuda",
)
def _flex_flash_attn_backward_compilable(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: str,
    out_: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dsink: torch.Tensor | None,
    q_ranges: torch.Tensor | None,
    k_ranges: torch.Tensor | None,
    attn_type_map: torch.Tensor | None,
    softmax_scale: float,
    softcap: float,
    dq_type: torch.dtype | None,
    dk_type: torch.dtype | None,
    dv_type: torch.dtype | None,
    disable_bwd_dkv_atomic_reduction: bool,
    disable_bwd_dq_atomic_reduction: bool,
    deterministic: bool,
    sm_margin: int,
    range_merge: bool,
    merge_k_ranges: torch.Tensor | None,
    bwd_kq_map: torch.Tensor | None,
    bwd_unique_count: torch.Tensor | None,
    bwd_inner_loop_k: bool,
    pack_gqa: bool,
    cat_gqa: bool,
    block_sparse: bool,
    index_sparse: bool,
    index_sparse_indices: torch.Tensor | None,
    inner_indices_cnt: int,
    kv_covered_mask: torch.Tensor | None,
) -> None:
    """torch.ops.flex_flash_attn._flex_flash_attn_backward_compilable"""
    mod = get_ffa_jit_mod(
        direction="bwd",
        head_dim=q.shape[-1],
        compute_dtype=q.dtype,
        output_dtype=None,
        softcap=softcap > 0.0,
        disable_atomic_reduction=disable_bwd_dkv_atomic_reduction,
        disable_dq_atomic_reduction=disable_bwd_dq_atomic_reduction,
        pack_gqa=pack_gqa,
        cat_gqa=cat_gqa,
        pack_gqa_factor=q.size(1) // k.size(1),
        deterministic=deterministic,
        range_merge=range_merge,
        bwd_inner_loop_k=bwd_inner_loop_k,
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        profile_mode=profile_mode,
        dq_dtype=dq_type
        or (q.dtype if disable_bwd_dq_atomic_reduction else torch.float32),
        dkv_dtype=dk_type
        or (k.dtype if disable_bwd_dkv_atomic_reduction else torch.float32),
        sparse_k_block_size=_ffa_sparse_k_block_size,
        _env_snapshot=_snapshot_env(),
    )

    (
        dq,
        dk,
        dv,
        # NOTE: when sink is not given
        # a new zero-sized empty dsink will be returned for convenience
        # no matter whether dsink buffer is given
        dsink,
    ) = mod.bwd(
        dout,
        q,
        k,
        v,
        sink,
        out_,
        dq,
        dk,
        dv,
        dsink,
        lse,
        q_ranges,
        k_ranges,
        attn_type_map,
        # for range merge
        merge_k_ranges,
        bwd_kq_map,
        bwd_unique_count,
        # for index attn
        index_sparse_indices,
        inner_indices_cnt,
        _ffa_sparse_k_block_size,
        # for others
        softmax_scale,
        softcap,
        dq_type,
        dk_type,
        dv_type,
        sink_layout,
        sm_margin,
        # coverage mask for IndexSparse postprocess
        kv_covered_mask,
    )


@_torch_register_fake_wrapper("flex_flash_attn::_flex_flash_attn_backward_compilable")
def _flex_flash_attn_backward_compilable_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: str,
    out_: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dsink: torch.Tensor | None,
    q_ranges: torch.Tensor | None,
    k_ranges: torch.Tensor | None,
    attn_type_map: torch.Tensor | None,
    softmax_scale: float,
    softcap: float,
    dq_type: torch.dtype | None,
    dk_type: torch.dtype | None,
    dv_type: torch.dtype | None,
    disable_bwd_dkv_atomic_reduction: bool,
    disable_bwd_dq_atomic_reduction: bool,
    deterministic: bool,
    sm_margin: int,
    range_merge: bool,
    merge_k_ranges: torch.Tensor | None,
    bwd_kq_map: torch.Tensor | None,
    bwd_unique_count: torch.Tensor | None,
    bwd_inner_loop_k: bool,
    pack_gqa: bool,
    cat_gqa: bool,
    block_sparse: bool,
    index_sparse: bool,
    index_sparse_indices: torch.Tensor | None,
    inner_indices_cnt: int,
    kv_covered_mask: torch.Tensor | None,
) -> None:
    pass


@nvtx.instrument_nvtx
def _flex_flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: AttnSinkLayout,
    out: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    dsink: torch.Tensor | None,
    q_ranges: torch.Tensor | None,
    k_ranges: torch.Tensor | None,
    attn_type_map: torch.Tensor | None,
    softmax_scale: float,
    softcap: float,
    dq_type: torch.dtype | None,
    dk_type: torch.dtype | None,
    dv_type: torch.dtype | None,
    disable_bwd_dkv_atomic_reduction: bool,
    disable_bwd_dq_atomic_reduction: bool,
    deterministic: bool,
    sm_margin: int = 0,
    range_merge: bool = False,
    merge_k_ranges: torch.Tensor | None = None,
    bwd_kq_map: torch.Tensor | None = None,
    bwd_unique_count: torch.Tensor | None = None,
    bwd_inner_loop_k: bool = False,
    pack_gqa: bool = False,
    cat_gqa: bool = False,
    block_sparse: bool = False,
    index_sparse: bool = False,
    index_sparse_indices: torch.Tensor | None = None,
    inner_indices_cnt: int = 0,
    sparse_k_block_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if profile_mode:  # NOTE: stop_event is called inside the kernel
        magi_attn_ext.start_event("bwd_prepare")

    # make all input tensors contiguous before initializing output buffers
    # NOTE: in backward, torch.compiler allows neither making nor checking contiguity
    # so we just skip here, but check inside the kernel
    if not torch.compiler.is_compiling():
        dout, q, k, v, sink, out, q_ranges, k_ranges, merge_k_ranges, bwd_kq_map = [
            maybe_contiguous(x)
            for x in (
                dout,
                q,
                k,
                v,
                sink,
                out,
                q_ranges,
                k_ranges,
                merge_k_ranges,
                bwd_kq_map,
            )
        ]

    # InnerLoopK + disable_bwd_dq_atomic_reduction: dQ is outer, epilogue uses per-element
    # direct store (one CTA per Q block) → empty_like is safe, no zero-init needed.
    # Otherwise dQ is inner → mainloop uses TMA_REDUCE_ADD → must be zeros.
    dq = (
        (
            torch.empty_like(q, dtype=dq_type or q.dtype)
            if disable_bwd_dq_atomic_reduction and bwd_inner_loop_k
            else torch.zeros_like(q, dtype=dq_type or torch.float32)
        )
        if dq is None
        else dq
    )

    clear_dkv = dk is None and dv is None
    kv_covered_mask = None
    if clear_dkv:
        # skip clear dk and dv if no reduction
        if disable_bwd_dkv_atomic_reduction:
            if index_sparse and index_sparse_indices is not None:
                dk = torch.empty_like(k, dtype=dk_type or k.dtype)
                dv = torch.empty_like(v, dtype=dv_type or v.dtype)
                # Coverage mask based on K-block structure: a block is covered
                # when it has >= 1 valid Q entry (inverted index value >= 0).
                total_k = k.size(0)
                num_k_blocks = index_sparse_indices.size(0)
                kbs_eff = total_k // num_k_blocks
                per_block_covered = (
                    index_sparse_indices.reshape(num_k_blocks, -1) >= 0
                ).any(dim=1)
                kv_covered_mask = per_block_covered.repeat_interleave(kbs_eff)
            else:
                dk = torch.empty_like(k, dtype=dk_type or k.dtype)
                dv = torch.empty_like(v, dtype=dv_type or v.dtype)
        else:
            dk = torch.zeros_like(k, dtype=dk_type or torch.float32)
            dv = torch.zeros_like(v, dtype=dv_type or torch.float32)
    else:
        dk = dk
        dv = dv

    dsink = (
        (torch.zeros_like(sink, dtype=torch.float32) if dsink is None else dsink)
        if sink is not None
        else None
    )

    # NOTE: we can not directly compile `_flex_flash_attn_backward`
    # since torch.compile does not allow returning the mutated args (dq, dk, dv, dsink)
    if not torch.compiler.is_compiling():
        global _ffa_sparse_k_block_size
        _ffa_sparse_k_block_size = sparse_k_block_size
    _flex_flash_attn_backward_compilable(
        dout=dout,
        q=q,
        k=k,
        v=v,
        sink=sink,
        sink_layout=sink_layout,
        out_=out,
        lse=lse,
        dq=dq,
        dk=dk,
        dv=dv,
        dsink=dsink,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=softmax_scale,
        softcap=softcap,
        dq_type=dq_type,
        dk_type=dk_type,
        dv_type=dv_type,
        disable_bwd_dkv_atomic_reduction=disable_bwd_dkv_atomic_reduction,
        disable_bwd_dq_atomic_reduction=disable_bwd_dq_atomic_reduction,
        deterministic=deterministic,
        sm_margin=sm_margin,
        range_merge=range_merge,
        merge_k_ranges=merge_k_ranges,
        bwd_kq_map=bwd_kq_map,
        bwd_unique_count=bwd_unique_count,
        bwd_inner_loop_k=bwd_inner_loop_k,
        pack_gqa=pack_gqa,
        cat_gqa=cat_gqa,
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        index_sparse_indices=index_sparse_indices,
        inner_indices_cnt=inner_indices_cnt,
        kv_covered_mask=kv_covered_mask,
    )

    return dq, dk, dv, dsink


# -------------------       ffa autograd   ------------------- #


class FlexFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor | None,
        sink_layout: AttnSinkLayout,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor | None,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
        deterministic: bool = False,
        sm_margin: int = 0,
        disable_fwd_atomic_reduction: bool = False,
        disable_bwd_dkv_atomic_reduction: bool = False,
        disable_bwd_dq_atomic_reduction: bool = False,
        ref_block_size: tuple[int, int] | None = None,
        max_outer_range_width: int | None = None,
        range_merge: bool = False,
        swap_ab: bool = False,
        pack_gqa: bool = False,
        cat_gqa: bool = False,
        block_sparse: bool = False,
        index_sparse: bool = False,
        bwd_inner_loop_k: bool | None = None,
        return_max_logits: bool = False,
        index_sparse_indices: torch.Tensor | None = None,
        inner_indices_cnt: int = 0,
        sparse_k_block_size: int = 1,
    ):
        softmax_scale = (
            q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        )

        assert not (
            block_sparse and index_sparse_indices is not None
        ), "block_sparse and index_sparse_indices are mutually exclusive."

        if q_ranges is not None:
            assert q_ranges.size(0) == k_ranges.size(0), (
                f"q_ranges and k_ranges must have the same number of ranges, "
                f"but got {q_ranges.size(0)} and {k_ranges.size(0)} respectively."
            )

            if attn_type_map is not None:
                assert attn_type_map.size(0) == q_ranges.size(0), (
                    f"attn_type_map must have the same number of ranges as q_ranges, "
                    f"but got {attn_type_map.size(0)} and {q_ranges.size(0)} respectively."
                )

        if block_sparse:
            range_merge = True

        if disable_bwd_dkv_atomic_reduction and bwd_inner_loop_k is True:
            raise RuntimeError(
                "When disable_bwd_dkv_atomic_reduction is true, bwd_inner_loop_k must not be True."
            )

        # ---- FA4 backend fast path ---- #
        if env.general.kernel_backend() == MagiAttentionKernelBackend.FA4:
            q_ranges_list = q_ranges.cpu().tolist()
            k_ranges_list = k_ranges.cpu().tolist()
            attn_type_map_list = (
                [0] * len(q_ranges_list)
                if attn_type_map is None
                else attn_type_map.cpu().tolist()
            )

            fa4_attn_arg = FA4AttnArg(
                q_ranges=AttnRanges.from_ranges(q_ranges_list),
                k_ranges=AttnRanges.from_ranges(k_ranges_list),
                attn_type_map=attn_type_map_list,
                seqlen_q=q.shape[0],
                seqlen_k=k.shape[0],
                headdim=q.shape[-1],
            )

            out, lse = fa4_fwd(
                q=q,
                k=k,
                v=v,
                sink=None,
                attn_arg=fa4_attn_arg,
                softmax_scale=softmax_scale,
                softcap=softcap,
            )
            out = out.to(q.dtype)

            ctx.save_for_backward(q, k, v, out, lse, q_ranges, k_ranges, attn_type_map)
            ctx.softmax_scale = softmax_scale
            ctx.softcap = softcap
            ctx.use_fa4_backend = True
            ctx.fa4_attn_arg = fa4_attn_arg

            return out, lse, None

        # ---- FFA (native) backend ---- #
        ctx.use_fa4_backend = False

        if range_merge:
            with maybe_profile_ffa_ctx("fwd_range_merge"):
                (
                    merge_q_ranges,
                    fwd_q_ranges,
                    fwd_k_ranges,
                    fwd_attn_type_map,
                    fwd_qk_map,
                    fwd_unique_count,
                ) = merge_ranges(q_ranges, k_ranges, attn_type_map=attn_type_map)

            if block_sparse:
                # Sparse paths always use (128, 128) tile — swap_ab is forbidden.
                # TODO: tune tile size for sparse TFLOPS (kBlockM=64 with small q_block_size)
                ref_block_size = (128, 128)
        else:
            fwd_q_ranges = q_ranges
            fwd_k_ranges = k_ranges
            fwd_attn_type_map = attn_type_map
            merge_q_ranges = None
            fwd_qk_map = None
            fwd_unique_count = None

        if not torch.compiler.is_compiling():
            global _ffa_sparse_k_block_size
            _ffa_sparse_k_block_size = sparse_k_block_size

        out, meta = _flex_flash_attn_forward(
            q=q,
            k=k,
            v=v,
            sink=sink,
            sink_layout=sink_layout,
            out=None,
            lse=None,
            q_ranges=fwd_q_ranges,
            k_ranges=fwd_k_ranges,
            attn_type_map=fwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=softcap,
            out_type=q.dtype
            if disable_fwd_atomic_reduction
            else torch.float32,  # out_type
            disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
            deterministic=deterministic,
            sm_margin=sm_margin,
            # optional args below mainly for sparse attn
            ref_block_size=ref_block_size,
            max_outer_range_width=max_outer_range_width,
            range_merge=range_merge,
            merge_q_ranges=merge_q_ranges,
            fwd_qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            swap_ab=swap_ab,
            pack_gqa=pack_gqa,
            block_sparse=block_sparse,
            index_sparse=index_sparse,
            index_sparse_indices=index_sparse_indices,
            inner_indices_cnt=inner_indices_cnt,
            sparse_k_block_size=sparse_k_block_size,
            return_max_logits=return_max_logits,
            max_logits=None,
        )
        lse = meta.lse
        max_logits = meta.max_logits

        # Cast output to the same dtype as q
        with maybe_profile_ffa_ctx("fwd_cast"):
            out = out.to(q.dtype)

        save_merge_info = (bwd_inner_loop_k is True) and range_merge

        tensors_to_save = [
            # 1. Base Tensors
            q,
            k,
            v,
            sink,
            out,
            lse,
            # 2. Range Merge Tensors
            q_ranges if not save_merge_info else fwd_q_ranges,
            k_ranges if not save_merge_info else fwd_k_ranges,
            attn_type_map if not save_merge_info else fwd_attn_type_map,
            merge_q_ranges if save_merge_info else None,
            fwd_qk_map if save_merge_info else None,
            fwd_unique_count if save_merge_info else None,
            # 3. IndexSparse Tensors
            index_sparse_indices if index_sparse else None,
        ]

        ctx.save_for_backward(*tensors_to_save)

        ctx.sink_layout = sink_layout
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.ref_block_size = ref_block_size
        ctx.range_merge = range_merge
        ctx.swap_ab = swap_ab
        ctx.block_sparse = block_sparse
        ctx.index_sparse = index_sparse
        ctx.inner_indices_cnt = inner_indices_cnt
        ctx.sparse_k_block_size = sparse_k_block_size
        ctx.bwd_inner_loop_k = bwd_inner_loop_k
        ctx.disable_bwd_dkv_atomic_reduction = disable_bwd_dkv_atomic_reduction
        ctx.disable_bwd_dq_atomic_reduction = disable_bwd_dq_atomic_reduction
        ctx.pack_gqa = pack_gqa
        ctx.cat_gqa = cat_gqa

        return out, lse, max_logits

    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args):  # pragma: no cover
        # ---- FA4 backend backward ---- #
        if ctx.use_fa4_backend:
            q, k, v, out, lse, q_ranges, k_ranges, attn_type_map = ctx.saved_tensors
            dq, dk, dv, _ = fa4_bwd(
                do=dout,
                q=q,
                k=k,
                v=v,
                sink=None,
                o=out,
                lse=lse,
                attn_arg=ctx.fa4_attn_arg,
                softmax_scale=ctx.softmax_scale,
                softcap=ctx.softcap,
            )
            dq = dq.to(q.dtype)
            dk = dk.to(k.dtype)
            dv = dv.to(v.dtype)
            return (
                dq,  # q
                dk,  # k
                dv,  # v
                None,  # sink
                None,  # sink_layout
                None,  # q_ranges
                None,  # k_ranges
                None,  # attn_type_map
                None,  # softmax_scale
                None,  # softcap
                None,  # deterministic
                None,  # sm_margin
                None,  # disable_fwd_atomic_reduction
                None,  # disable_bwd_dkv_atomic_reduction
                None,  # disable_bwd_dq_atomic_reduction
                None,  # ref_block_size
                None,  # max_outer_range_width
                None,  # range_merge
                None,  # swap_ab
                None,  # pack_gqa
                None,  # cat_gqa
                None,  # block_sparse
                None,  # index_sparse
                None,  # bwd_inner_loop_k
                None,  # return_max_logits
                None,  # index_sparse_indices
                None,  # inner_indices_cnt
                None,  # sparse_k_block_size
            )

        # ---- FFA (native) backend backward ---- #

        (
            # 1. Base Tensors
            q,
            k,
            v,
            sink,
            out,
            lse,
            # 2. Range Merge Tensors,
            q_ranges,
            k_ranges,
            attn_type_map,
            merge_q_ranges,
            fwd_qk_map,
            fwd_unique_count,
            # 3. IndexSparse Tensors
            index_sparse_indices,
        ) = ctx.saved_tensors

        bwd_inner_loop_k = (
            ctx.bwd_inner_loop_k if ctx.bwd_inner_loop_k is not None else False
        )

        if ctx.disable_bwd_dkv_atomic_reduction and bwd_inner_loop_k:
            raise RuntimeError(
                "disable_bwd_dkv_atomic_reduction is incompatible with bwd_inner_loop_k=True (InnerLoopK)."
            )

        if ctx.index_sparse:
            assert (
                q_ranges is None and k_ranges is None
            ), "IndexSparse BWD does not use q_ranges/k_ranges; they should be None"

            bwd_q_ranges = None
            bwd_k_ranges = None
            bwd_attn_type_map = None
            merge_k_ranges = None
            bwd_kq_map = None
            bwd_unique_count = None
            bwd_range_merge = False

            if not bwd_inner_loop_k:
                # IndexSparse BWD InnerLoopQ: outer=K block, inner=Q from inner_indices
                _loopq_kbs = ctx.sparse_k_block_size
                nhk = k.size(1)
                seqlen_k = v.size(0)

                from magi_attention.utils.sparse_utils import (
                    invert_index_sparse_indices,
                )

                if _loopq_kbs > 1 and nhk != 1:
                    raise NotImplementedError(
                        f"IndexSparse BWD InnerLoopQ with sparse_k_block_size>1 requires nhk=1, "
                        f"got nhk={nhk}. NHK>1 + kbs>1 has a flat-layout mismatch (P8-BUG-NHK)."
                    )

                # NOTE: invert_index_sparse_indices contains a GPU→CPU sync
                # (counts.max().item()) to determine the padded inner_topk.
                # This is acceptable for now as the inversion is O(nnz) and the
                # sync cost is amortized by the BWD kernel runtime.
                _inner_indices, _inner_topk = invert_index_sparse_indices(
                    index_sparse_indices,
                    seqlen_k=seqlen_k,
                    sparse_k_block_size=_loopq_kbs,
                    pad_multiple=64,
                )
                index_sparse_indices = _inner_indices.contiguous()
                ctx.inner_indices_cnt = _inner_topk

            # else: IndexSparse BWD InnerLoopK — use forward's topk_indices directly
        elif ctx.range_merge:
            bwd_range_merge = True
            with maybe_profile_ffa_ctx("bwd_range_merge"):
                if bwd_inner_loop_k:
                    if merge_q_ranges is not None:
                        # Reuse the forward range merge results directly
                        (
                            bwd_q_ranges,
                            bwd_k_ranges,
                            bwd_attn_type_map,
                            merge_k_ranges,
                            bwd_kq_map,
                            bwd_unique_count,
                        ) = (
                            q_ranges,
                            k_ranges,
                            attn_type_map,
                            merge_q_ranges,
                            fwd_qk_map,
                            fwd_unique_count,
                        )
                    else:
                        # InnerLoopK: outer loop is Q (m_blocks), merge by Q ranges
                        (
                            merge_k_ranges,
                            bwd_q_ranges,
                            bwd_k_ranges,
                            bwd_attn_type_map,
                            bwd_kq_map,
                            bwd_unique_count,
                        ) = merge_ranges(
                            q_ranges, k_ranges, attn_type_map=attn_type_map
                        )
                else:
                    # InnerLoopQ: outer loop is K (n_blocks), merge by K ranges
                    (
                        merge_k_ranges,
                        bwd_k_ranges,
                        bwd_q_ranges,
                        bwd_attn_type_map,
                        bwd_kq_map,
                        bwd_unique_count,
                    ) = merge_ranges(k_ranges, q_ranges, attn_type_map=attn_type_map)
        else:
            bwd_range_merge = False
            bwd_q_ranges, bwd_k_ranges, bwd_attn_type_map = (
                q_ranges,
                k_ranges,
                attn_type_map,
            )
            merge_k_ranges, bwd_kq_map, bwd_unique_count = None, None, None

        if not torch.compiler.is_compiling():
            global _ffa_sparse_k_block_size
            _ffa_sparse_k_block_size = ctx.sparse_k_block_size

        dq, dk, dv, dsink = _flex_flash_attn_backward(
            dout=dout,
            q=q,
            k=k,
            v=v,
            sink=sink,
            sink_layout=ctx.sink_layout,
            out=out,
            lse=lse,
            dq=None,
            dk=None,
            dv=None,
            dsink=None,
            q_ranges=bwd_q_ranges,
            k_ranges=bwd_k_ranges,
            attn_type_map=bwd_attn_type_map,
            softmax_scale=ctx.softmax_scale,
            softcap=ctx.softcap,
            dq_type=q.dtype if ctx.disable_bwd_dq_atomic_reduction else torch.float32,
            dk_type=k.dtype if ctx.disable_bwd_dkv_atomic_reduction else torch.float32,
            dv_type=v.dtype if ctx.disable_bwd_dkv_atomic_reduction else torch.float32,
            disable_bwd_dkv_atomic_reduction=ctx.disable_bwd_dkv_atomic_reduction,
            disable_bwd_dq_atomic_reduction=ctx.disable_bwd_dq_atomic_reduction,
            deterministic=ctx.deterministic,
            sm_margin=ctx.sm_margin,
            range_merge=bwd_range_merge,
            merge_k_ranges=merge_k_ranges,
            bwd_kq_map=bwd_kq_map,
            bwd_unique_count=bwd_unique_count,
            bwd_inner_loop_k=bwd_inner_loop_k,
            pack_gqa=ctx.pack_gqa,
            cat_gqa=ctx.cat_gqa,
            block_sparse=ctx.block_sparse,
            index_sparse=ctx.index_sparse,
            index_sparse_indices=index_sparse_indices,
            inner_indices_cnt=ctx.inner_indices_cnt,
            sparse_k_block_size=ctx.sparse_k_block_size,
        )

        # Cast gradients to the same dtype as inputs
        with maybe_profile_ffa_ctx("bwd_cast"):
            dq = dq.to(q.dtype)
            dk = dk.to(k.dtype)
            dv = dv.to(v.dtype)
            if sink is not None:
                assert dsink is not None  # mypy
                dsink = dsink.to(sink.dtype)

        return (
            dq,  # q
            dk,  # k
            dv,  # v
            dsink,  # sink
            None,  # sink_layout
            None,  # q_ranges
            None,  # k_ranges
            None,  # attn_type_map
            None,  # softmax_scale
            None,  # softcap
            None,  # deterministic
            None,  # sm_margin
            None,  # disable_fwd_atomic_reduction
            None,  # disable_bwd_dkv_atomic_reduction
            None,  # disable_bwd_dq_atomic_reduction
            None,  # ref_block_size
            None,  # max_outer_range_width
            None,  # range_merge
            None,  # swap_ab
            None,  # pack_gqa
            None,  # cat_gqa
            None,  # block_sparse
            None,  # index_sparse
            None,  # bwd_inner_loop_k
            None,  # return_max_logits
            None,  # index_sparse_indices
            None,  # inner_indices_cnt
            None,  # sparse_k_block_size
        )


# -------------------       ffa interface   ------------------- #


def flex_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor | None = None,
    k_ranges: torch.Tensor | None = None,
    attn_type_map: torch.Tensor | None = None,
    *,
    index_sparse_indices: torch.Tensor | None = None,
    q_block_size: int = 1,
    sparse_k_block_size: int = 1,
    sink: torch.Tensor | None = None,
    sink_layout: AttnSinkLayout = "sh",
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    disable_fwd_atomic_reduction: bool = False,
    disable_bwd_dkv_atomic_reduction: bool = False,
    disable_bwd_dq_atomic_reduction: bool = False,
    ref_block_size: tuple[int, int] | None = None,
    max_seqlen_q: int | None = None,
    range_merge: bool = False,
    swap_ab: bool = False,
    pack_gqa: bool = False,
    cat_gqa: bool = False,
    block_sparse: bool = False,
    index_sparse: bool = False,
    swap_bwd_qk_loop: bool | None = None,
    return_max_logits: bool = False,
) -> tuple[torch.Tensor, AttnForwardMeta]:
    """
    An interface similar to flash attention that doesn't require distributed environment, dispatch or undispatch.
    Directly call magi_attn_kernel to get attention output and lse. This is faster when you don't need context parallel.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.

        q_ranges (torch.Tensor, optional): Query ranges tensor to represent the attn mask.
            Mutually exclusive with ``index_sparse_indices``.
        k_ranges (torch.Tensor, optional): Key ranges tensor to represent the attn mask.
            Must be provided together with ``q_ranges``.

        index_sparse_indices (torch.Tensor, optional): IndexSparse token indices.
            Shape: ``(total_q, num_kv_heads, max_topk)``, dtype=int32.
            Values are **logical** KV token positions: ``batch_idx * S_kv + token_idx``.
            The kernel internally converts to physical row via ``pos * NHK + kv_head``.
            Use ``-1`` for padding (must be contiguous at the tail of each row).
            Mutually exclusive with ``q_ranges``.
            The kernel scans trailing ``-1`` entries to determine loop count and
            invalid count internally — no Python-side preprocessing is needed.
            ``max_topk`` (last dim) must be a multiple of tile_size (128, or 64 if swap_ab).
            The mask representation theoretically supports block-level KV (``sparse_k_block_size > 1``)
            but currently only ``sparse_k_block_size=1`` (token-level) is implemented.
        q_block_size (int, optional): Q block size. Defaults to ``1``.
            Currently only ``1`` (per-token Q granularity) is supported.
        sparse_k_block_size (int, optional): K block size for inner-loop contiguity. Defaults to ``1``.
            Must be a positive power of 2 (1, 2, 4, 8, 16, 32, 64, 128, 256, ...).
            For ``index_sparse``: specifies the granularity of K indices (each index covers
            ``sparse_k_block_size`` tokens). When ``sparse_k_block_size >= 128``, inner KV loads use TMA 2D;
            when ``sparse_k_block_size < 128``, inner KV loads use CpAsync scatter with per-token
            gather. Both paths are correct; TMA 2D is faster for large contiguous blocks.
            For ``block_sparse``: auto-derived from the uniform k_range size if left at
            default. When k_range_size >= 128, inner loads use TMA 2D for best perf.
            Not compatible with ``swap_ab``.

        max_seqlen_q (int | None, optional): Maximum sequence length for query. Defaults to ``None``.
            If provided, enables optimization for tile_scheduler. Most recommended to set this when using
            range_merge(for block sparse attention).
        attn_type_map (torch.Tensor, optional): Attention type map tensor with dtype=torch.int32,
            Defaults to ``None`` to apply full attention for all ranges.
            The values specify the attention type for each token:

                - 0: full attention
                - 1: causal attention
                - 2: inverse causal attention
                - 3: bidirectional causal attention

            More information about the attention type map can be found in the ``Note`` below.

        sink (torch.Tensor, optional): Learnable sink token tensor.
            Defaults to ``None`` to not apply attention sink.
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh". Available Options: "sh", "ssh".

        softmax_scale (float, optional): Softmax scale.
            Defaults to ``None`` to use: ``1/sqrt(head_dim)``.
        softcap (float, optional): Softcap. Defaults to ``0.0``.

        deterministic (bool, optional): Whether to use deterministic attention. Defaults to ``False``.

        sm_margin (int, optional): The amount of SMs reserved out,
            useful when considering overlapping with other kernels such as communication kernels.
            Defaults to ``0`` to use all available SMs.

        disable_fwd_atomic_reduction (bool, optional):
            Whether to disable forward atomic reduction. Defaults to ``False``.

                If you can ensure ``q_ranges`` is non-overlapped,
                you can set this to ``True`` for better performance.
                The "overlap" term among ``q_ranges`` is defined as:
                if any two ``q_range`` in ``q_ranges`` have non-empty intersection, then it is overlapped.
                For example, ``q_ranges`` = ``[[0, 15], [10, 20], [20, 30]]`` is overlapped
                since ``q_range1`` = ``[0, 15]`` and ``q_range2`` = ``[10, 20]`` intersect,
                while `` q_ranges`` = ``[[0, 15], [15, 20], [20, 30]]`` then is non-overlapped.

        disable_bwd_dkv_atomic_reduction (bool, optional):
            Whether to disable backward dK/dV atomic reduction. Defaults to ``False``.

                If you can ensure ``k_ranges`` (used in backward) is non-overlapped and sorted,
                you can set this to ``True`` for better performance.
                The "overlap" term among ``k_ranges`` is defined as:
                if any two ``k_range`` in ``k_ranges`` have non-empty intersection, then it is overlapped.
                For example, ``k_ranges`` = ``[[0, 15], [10, 20], [20, 30]]`` is overlapped
                since ``k_range1`` = ``[0, 15]`` and ``k_range2`` = ``[10, 20]`` intersect,
                while ``k_ranges`` = ``[[0, 15], [15, 20], [20, 30]]`` then is non-overlapped.
                **Note:** This flag can only be enabled with MHA or catGQA.

        disable_bwd_dq_atomic_reduction (bool, optional):
            Whether to disable backward dQ atomic reduction for BWD InnerLoopK
            (``swap_bwd_qk_loop=True``). Defaults to ``False``.

                When enabled, dQ uses the input dtype (bf16/fp16) instead of float32,
                skips zero-initialization (``empty_like``), and the epilogue uses per-element
                direct store instead of TMA atomic reduce-add. Safe when outer Q ranges
                are non-overlapping (guaranteed by BlockSparse/IndexSparse with RangeMerge).
                Auto-set to ``True`` for sparse scenarios with ``swap_bwd_qk_loop=True``.

        ref_block_size (tuple[int, int], optional):
            Reference block size (M, N) for kernel selection.
            Defaults to ``None`` to use the internal heuristic.
            **Note:** This flag is useful for sparse attention scenarios but still under development.

        max_seqlen_q (int | None, optional):
            Maximum sequence length for query. Defaults to ``None``.
            If provided, enables optimization for forward tile_scheduler,
            especially for block sparse attention scenarios.

        range_merge (bool, optional):
            Whether to automatically merge k_ranges for the same q_range. Defaults to ``False``.
            **Note:** This flag is useful for sparse attention scenarios but still under development.

        swap_ab (bool, optional): Whether to swap the order of A and B operands for the matmul operation
            (i.e. transpose `C=A x B^T` to `C^T= B x A^T`) in attention forward passes. Defaults to ``False``.
            **Not compatible with sparse attention** (``block_sparse`` or ``index_sparse``).
            Only applicable to dense attention with very small Q sequences.

        pack_gqa (bool, optional):
            Whether to group query heads sharing the same KV head into a single computation block tile for small
            seqlen_q scenarios. This method significantly improves the computational efficiency
            of block sparse attention when seqlen_q is small. Defaults to ``False``.
            **Note:** kblockm must be divisible by pack_gqa_factor (num_qhead // num_khead).
            For backward pass, this flag is only enabled when swap_bwd_qk_loop is True.

        cat_gqa (bool, optional):
            Whether to concatenate multiple Q heads sharing the same KV head,
            to optimize the backward performance under GQA settings. Defaults to ``False``.

        block_sparse (bool, optional):
            Whether to enable sparse load mode for optimizing performance when k_range size is small (< 64).
            Automatically enables ``range_merge``. Defaults to ``False``.
            Mutually exclusive with ``index_sparse_indices``.
            When enabled, ``sparse_k_block_size`` is auto-set to ``tile_N`` (128)
            so that inner KV loads use TMA 2D — without this, the kernel falls back to CpAsync
            scatter load with ~40% performance loss.
            **Contract:** all k_ranges (and, for ``swap_bwd_qk_loop=False`` backward, all q_ranges)
            must have one uniform size — the kernel relies on it for O(1) cursor seeks.
            Block-mask generated ranges satisfy this by construction; it is verified when
            ``MAGI_ATTENTION_SANITY_CHECK=1``.

        index_sparse (bool, optional):
            Whether to enable the IndexSparse kernel path, where the kernel directly reads
            ``index_sparse_indices`` instead of using q/k ranges. Automatically set to ``True``
            when ``index_sparse_indices`` is provided. Defaults to ``False``.

        swap_bwd_qk_loop (bool | None, optional): Controls backward double-loop order.
            ``False`` = LoopQ (outer K, inner Q), ``True`` = LoopK (outer Q, inner K).
            ``None`` (default) = auto-infer: IndexSparse with ``sparse_k_block_size < 128``
            uses LoopK (bounded inner K loop from forward indices),
            otherwise LoopQ.  LoopK requires ``pack_gqa_factor >= 128``; when
            not met, falls back to LoopQ automatically.

        return_max_logits (bool, optional): Whether to return the maximum attention logits,
            according to the Muon QK-Clip technique introduced in Kimi K2: https://arxiv.org/pdf/2507.20534.pdf.
            Defaults to ``False``.

    Returns:
        tuple[torch.Tensor, AttnForwardMeta]:
            - out (torch.Tensor): Attention output tensor
            - meta (AttnForwardMeta): Meta information of the attention forward pass,
                for now, including lse (torch.Tensor) with dtype=torch.float32,
                and max_logits (torch.Tensor) with dtype=torch.float32,
                if ``return_max_logits`` is ``True``, otherwise ``None``.

    Shape:
        - q: (num_tokens_q, num_heads_q, head_dim)
        - k: (num_tokens_kv, num_heads_kv, head_dim)
        - v: (num_tokens_kv, num_heads_kv, head_dim)
        - sink:
            - if sink_layout == "sh": (num_tokens_sink, num_heads_q)
            - if sink_layout == "ssh": (num_tokens_q, num_tokens_sink, num_heads_q)
        - q_ranges: (num_ranges, 2)
        - k_ranges: (num_ranges, 2)
        - attn_type_map: (num_ranges,)
        - out: (num_tokens_q, num_heads_q, head_dim)
        - lse: (num_tokens_q, num_heads_q)
        - max_logits: (num_heads_q,)

    Note:
        The ``attn_type_map`` explains the semantics of different attention mask types.
        In addition to the descriptions below, see our blog for a visual explanation:
        https://sandai-org.github.io/MagiAttention/blog/#flex-flash-attn

        1. Full attention:
            If seqlen_q = 5 and seqlen_k = 2::

                1 1
                1 1
                1 1
                1 1
                1 1

            If seqlen_q = 2 and seqlen_k = 5::

                1 1 1 1 1
                1 1 1 1 1

            If seqlen_q = 5 and seqlen_k = 5::

                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1

        2. Causal attention (bottom-right aligned):
            If seqlen_q = 5 and seqlen_k = 2::

                0 0
                0 0
                0 0
                1 0
                1 1

            If seqlen_q = 2 and seqlen_k = 5::

                1 1 1 1 0
                1 1 1 1 1

            If seqlen_q = 5 and seqlen_k = 5::

                1 0 0 0 0
                1 1 0 0 0
                1 1 1 0 0
                1 1 1 1 0
                1 1 1 1 1

        3. Inverse causal attention (top-left aligned):
            If seqlen_q = 5 and seqlen_k = 2::

                1 1
                0 1
                0 0
                0 0
                0 0

            If seqlen_q = 2 and seqlen_k = 5::

                1 1 1 1 1
                0 1 1 1 1

            If seqlen_q = 5 and seqlen_k = 5::

                1 1 1 1 1
                0 1 1 1 1
                0 0 1 1 1
                0 0 0 1 1
                0 0 0 0 1

        4. Bidirectional causal attention (intersection of causal and inverse causal):
            This is the element-wise AND of causal and inverse causal masks.

            If seqlen_q = 5 and seqlen_k = 2::

                0 0
                0 0
                0 0
                0 0
                0 0

            If seqlen_q = 2 and seqlen_k = 5::

                1 1 1 1 0
                0 1 1 1 1

            If seqlen_q = 5 and seqlen_k = 5::

                1 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0
                0 0 0 1 0
                0 0 0 0 1
    """

    bwd_inner_loop_k = swap_bwd_qk_loop
    max_outer_range_width = max_seqlen_q

    # ── Sparse mask input validation ──
    # Auto-infer index_sparse from tensor presence.
    if index_sparse_indices is not None:
        index_sparse = True
    _has_ranges = q_ranges is not None
    _num_sparse_inputs = int(_has_ranges) + int(index_sparse)
    assert _num_sparse_inputs == 1, (
        "Exactly one of (q_ranges + k_ranges) or index_sparse_indices must be provided. "
        f"Got: q_ranges={'set' if _has_ranges else 'None'}, "
        f"index_sparse_indices={'set' if index_sparse else 'None'}"
    )
    assert not (
        block_sparse and index_sparse
    ), "block_sparse and index_sparse_indices are mutually exclusive."
    assert not (
        swap_ab and (block_sparse or index_sparse)
    ), "swap_ab is not supported with sparse attention (block_sparse or index_sparse)."
    if _has_ranges:
        assert k_ranges is not None, "k_ranges must be provided together with q_ranges"

    if block_sparse:
        assert not swap_ab, (
            "swap_ab is not supported with block_sparse — sparse paths use scatter/TMA "
            "load modes that are incompatible with the swapped matmul layout."
        )
        # BlockSparse uses the same sparse_k_block_size mechanism as IndexSparse to
        # tell the kernel that inner KV tiles are contiguous (KBlockSize >= kBlockN
        # → TMA 2D load). When sparse_k_block_size is left at default (1), derive it
        # from the actual k_ranges uniform size. If the k_range size >= kBlockN,
        # the kernel uses TMA 2D; otherwise it falls back to CpAsync scatter.
        assert k_ranges is not None, "block_sparse requires k_ranges"
        if sparse_k_block_size <= 1:
            k_sizes = k_ranges[:, 1] - k_ranges[:, 0]
            _bs_k_size = k_sizes[0].item()
            sparse_k_block_size = _bs_k_size

        assert (
            sparse_k_block_size >= 1
        ), f"block_sparse: sparse_k_block_size must be >= 1, got {sparse_k_block_size}"

        if is_sanity_check_enable():
            assert q_ranges is not None
            k_sizes = k_ranges[:, 1] - k_ranges[:, 0]
            if not (k_sizes == k_sizes[0]).all().item():
                import warnings

                warnings.warn(
                    f"block_sparse: non-uniform k_ranges detected "
                    f"(sizes in [{k_sizes.min().item()}, {k_sizes.max().item()}]). "
                    f"range_merge will normalize them to ref_block_size.",
                    stacklevel=2,
                )

        if max_outer_range_width is None:
            max_outer_range_width = q_block_size

    # ── index_sparse_indices direct path: kernel reads indices directly ──
    if index_sparse:
        assert index_sparse_indices is not None
        assert index_sparse_indices.dim() == 3, (
            f"index_sparse_indices must be 3D (total_q, num_kv_heads, max_topk), "
            f"got shape {index_sparse_indices.shape}"
        )
        assert q_block_size == 1, (
            "Currently only q_block_size=1 (per-token Q granularity) is supported "
            f"for index_sparse_indices input, got q_block_size={q_block_size}"
        )
        tile_size = 128
        assert (
            sparse_k_block_size >= 1
            and (sparse_k_block_size & (sparse_k_block_size - 1)) == 0
        ), f"sparse_k_block_size must be a positive power of 2, got {sparse_k_block_size}"
        if sparse_k_block_size > 128:
            assert (
                sparse_k_block_size % 128 == 0
            ), f"sparse_k_block_size > 128 must be a multiple of 128, got {sparse_k_block_size}"
        total_q_idx, nhk_idx, max_topk_per_head = index_sparse_indices.shape
        if sparse_k_block_size > 1:
            effective_topk = max_topk_per_head * sparse_k_block_size
            assert effective_topk % tile_size == 0, (
                f"effective topk (max_topk_per_head={max_topk_per_head} * sparse_k_block_size={sparse_k_block_size} "
                f"= {effective_topk}) must be a multiple of tile_size={tile_size}."
            )
        else:
            assert max_topk_per_head % tile_size == 0, (
                f"index_sparse_indices last dim (max_topk_per_head={max_topk_per_head}) must be a multiple "
                f"of tile_size={tile_size}. Pad with -1 if needed."
            )
        # Keep 3D: kernel uses nhk (from shape_K) and bidh_kv to index
        # directly into dim-1 — no flatten/unflatten needed.
        index_sparse_indices = index_sparse_indices.contiguous()

        # IndexSparse uses indices, not ranges — assert ranges are not provided
        assert q_ranges is None and k_ranges is None, (
            "IndexSparse path requires index_sparse_indices only; "
            "q_ranges/k_ranges must not be provided simultaneously."
        )

        range_merge = False
        if max_outer_range_width is None:
            max_outer_range_width = q_block_size
        # IndexSparse: fixed (128, tile_size) tile — q_block_size=1 means PackGQA fills M dim.
        # TODO: tune kBlockM for non-PackGQA or small-head scenarios
        ref_block_size = (128, tile_size)

    assert not (
        bwd_inner_loop_k is True and deterministic
    ), "Deterministic mode is not supported when bwd_inner_loop_k is True."

    if env.general.kernel_backend() == MagiAttentionKernelBackend.FA4:
        assert is_fa4_installed, (
            "FA4 backend is enabled (MAGI_ATTENTION_FA4_BACKEND=1), "
            "but FlashAttn4 is not installed."
        )
        _FA4_UNSUPPORTED = {
            "sink": sink is not None,
            "deterministic": deterministic,
            "sm_margin": sm_margin != 0,
            "disable_fwd_atomic_reduction": disable_fwd_atomic_reduction,
            "disable_bwd_dkv_atomic_reduction": disable_bwd_dkv_atomic_reduction,
            "ref_block_size": ref_block_size is not None,
            "max_seqlen_q": max_outer_range_width is not None,
            "range_merge": range_merge,
            "swap_ab": swap_ab,
            "pack_gqa": pack_gqa,
            "cat_gqa": cat_gqa,
            "block_sparse": block_sparse,
            "index_sparse": index_sparse,
            "bwd_inner_loop_k": bwd_inner_loop_k,
            "return_max_logits": return_max_logits,
        }
        bad = [name for name, active in _FA4_UNSUPPORTED.items() if active]
        assert not bad, (
            f"FA4 backend does not support the following features: {bad}. "
            f"Please disable them or switch off the FA4 backend "
            f"(unset MAGI_ATTENTION_FA4_BACKEND)."
        )

    # Per-head topk width (dim-2 of the 3D tensor), NOT nhk * topk_per_head.
    inner_indices_cnt = (
        index_sparse_indices.shape[2]
        if index_sparse and index_sparse_indices is not None
        else 0
    )

    # ── Auto-set sparse flags ──
    if block_sparse:
        range_merge = True

    _is_sparse = block_sparse or index_sparse
    if _is_sparse:
        disable_fwd_atomic_reduction = True

        _is_mha = q.size(1) == k.size(1)

        # BlockSparse: fwd_qk_map from merge_ranges assumes PackGQA layout.
        # Without pack_gqa=True, the tile→range lookup produces wrong K ranges
        # → incorrect attention output.  IndexSparse doesn't use qk_map (uses
        # explicit per-token indices) so pack_gqa=False is safe there.
        if block_sparse and not pack_gqa and not cat_gqa and k.size(1) == 1:
            pack_gqa = True

        _gqa_safe = _is_mha or pack_gqa or cat_gqa

        # IndexSparse auto-infer BWD loop mode: when kbs < kBlockN (128),
        # inner K loads use CpAsync scatter → LoopQ's inverted-index inner
        # loop degenerates (each K block touched by ~all Q tokens).
        # LoopK keeps the inner loop bounded (topk / kBlockN iterations).
        if index_sparse and bwd_inner_loop_k is None and sparse_k_block_size < 128:
            bwd_inner_loop_k = True

        # IndexSparse + LoopK: each Q-tile (M=128) must cover exactly one
        # original seq position's packed heads.  When pack_gqa_factor < 128,
        # one Q-tile spans multiple seq positions with different index patterns
        # → BWD dQ is incorrect.  Fall back to LoopQ.
        if index_sparse and bwd_inner_loop_k is True and pack_gqa:
            _pack_f = q.size(1) // k.size(1)
            if _pack_f < 128:
                bwd_inner_loop_k = None

        # BWD InnerLoopQ (bwd_inner_loop_k != True): dKV is outer accumulation.
        # Safe only when GQA heads are packed (no cross-CTA dKV overlap).
        if _is_sparse and bwd_inner_loop_k is not True and _gqa_safe:
            disable_bwd_dkv_atomic_reduction = True

        # BWD InnerLoopK (bwd_inner_loop_k == True): dQ is outer accumulation.
        # Each CTA owns a unique Q block — safe regardless of GQA config.
        if bwd_inner_loop_k is True:
            disable_bwd_dq_atomic_reduction = True

    out, lse, max_logits = FlexFlashAttnFunc.apply(
        q,
        k,
        v,
        sink,
        sink_layout,
        q_ranges,
        k_ranges,
        attn_type_map,
        softmax_scale,
        softcap,
        deterministic,
        sm_margin,
        disable_fwd_atomic_reduction,
        disable_bwd_dkv_atomic_reduction,
        disable_bwd_dq_atomic_reduction,
        ref_block_size,
        max_outer_range_width,
        range_merge,
        swap_ab,
        pack_gqa,
        cat_gqa,
        block_sparse,
        index_sparse,
        bwd_inner_loop_k,
        return_max_logits,
        # for IndexSparse direct path
        index_sparse_indices,
        inner_indices_cnt,
        sparse_k_block_size,
    )
    return out, AttnForwardMeta(lse=lse, max_logits=max_logits)
