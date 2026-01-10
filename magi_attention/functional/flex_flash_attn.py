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

import os
from contextlib import contextmanager

import torch
from packaging import version

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.utils import nvtx

from ._flex_flash_attn_jit import get_ffa_jit_mod

# isort: off
# We need to import the CUDA kernels after importing torch
is_ffa_utils_installed = False
try:
    from magi_attention import flexible_flash_attention_utils_cuda as ffa_utils  # type: ignore[attr-defined]

    is_ffa_utils_installed = True
except ImportError:
    pass

# isort: on


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


profile_mode = os.environ.get("MAGI_ATTENTION_PROFILE_MODE", "0") == "1"


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
    assert is_ffa_utils_installed, (
        "The `flexible_flash_attention_utils_cuda` "
        "extension module is not installed."
    )

    sorted_idx = torch.argsort(outer_ranges[:, 0], dim=0, stable=True)
    sorted_outer_ranges = outer_ranges[sorted_idx]
    sorted_inner_ranges = inner_ranges[sorted_idx]
    if attn_type_map is None:
        sorted_attn_type_map = None
    else:
        sorted_attn_type_map = attn_type_map[sorted_idx]
    (
        merge_outer_ranges,
        range_map,
        unique_count,
    ) = ffa_utils.unique_consecutive_pairs(sorted_outer_ranges)

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
        ffa_utils.start_event(event_name)

    yield

    if profile_mode:
        ffa_utils.stop_event(event_name)


# -------------------       ffa forward   ------------------- #


@_torch_custom_op_wrapper(
    "flex_flash_attn::_flex_flash_attn_forward_compilable",
    # NOTE: had better NOT use "out" in args since it is a reserved special arg for torch.compile
    mutates_args=("out_", "lse"),
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
    attn_type_map: torch.Tensor,
    softmax_scale: float,
    softcap: float,
    disable_fwd_atomic_reduction: bool,
    out_type: torch.dtype | None,
    deterministic: bool,
    sm_margin: int,
    auto_range_merge: bool,
    merge_q_ranges: torch.Tensor | None,
    qk_map: torch.Tensor | None,
    fwd_unique_count: torch.Tensor | None,
    kblock_m: int | None,
    kblock_n: int | None,
    swap_ab: bool,
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
        deterministic=deterministic,
        profile_mode=profile_mode,
        ref_block_size=(kblock_m, kblock_n)
        if kblock_m is not None and kblock_n is not None
        else None,
        auto_range_merge=auto_range_merge,
        swap_ab=swap_ab,
    )

    out_, lse = mod.fwd(
        q,
        k,
        v,
        sink,
        out_,
        lse,
        q_ranges,
        k_ranges,
        attn_type_map,
        merge_q_ranges,
        qk_map,
        fwd_unique_count,
        softmax_scale,
        softcap,
        disable_fwd_atomic_reduction,
        out_type,
        sink_layout,
        deterministic,
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
    attn_type_map: torch.Tensor,
    softmax_scale: float,
    softcap: float,
    disable_fwd_atomic_reduction: bool,
    out_type: torch.dtype | None,
    deterministic: bool,
    sm_margin: int,
    auto_range_merge: bool,
    merge_q_ranges: torch.Tensor | None,
    qk_map: torch.Tensor | None,
    fwd_unique_count: torch.Tensor | None,
    kblock_m: int | None,
    kblock_n: int | None,
    swap_ab: bool,
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
    attn_type_map: torch.Tensor,
    softmax_scale: float,
    softcap: float,
    disable_fwd_atomic_reduction: bool,
    out_type: torch.dtype | None,
    deterministic: bool,
    sm_margin: int,
    auto_range_merge: bool = False,
    merge_q_ranges: torch.Tensor | None = None,
    qk_map: torch.Tensor | None = None,
    fwd_unique_count: torch.Tensor | None = None,
    ref_block_size: tuple[int, int] | None = None,
    swap_ab: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if profile_mode:  # NOTE: stop_event is called inside the kernel
        ffa_utils.start_event("fwd_prepare")

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

    if ref_block_size is not None:
        kblock_m, kblock_n = ref_block_size
    else:
        kblock_m = None
        kblock_n = None

    # NOTE: we can not directly compile `_flex_flash_attn_forward`
    # since torch.compile does not allow returning the mutated args (out, lse)
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
        disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
        out_type=out_type,
        deterministic=deterministic,
        sm_margin=sm_margin,
        auto_range_merge=auto_range_merge,
        merge_q_ranges=merge_q_ranges,
        qk_map=qk_map,
        fwd_unique_count=fwd_unique_count,
        kblock_m=kblock_m,
        kblock_n=kblock_n,
        swap_ab=swap_ab,
    )

    return out, lse


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
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    softmax_scale: float,
    softcap: float,
    disable_bwd_dkv_atomic_reduction: bool,
    dq_type: torch.dtype | None,
    dk_type: torch.dtype | None,
    dv_type: torch.dtype | None,
    deterministic: bool,
    sm_margin: int,
    auto_range_merge: bool,
    merge_k_ranges: torch.Tensor | None,
    bwd_kq_map: torch.Tensor | None,
    bwd_unique_count: torch.Tensor | None,
    swap_bwd_qk_loop: bool,
) -> None:
    """torch.ops.flex_flash_attn._flex_flash_attn_backward_compilable"""
    # DE-BUG
    import os

    swap_qk_loop_debug = os.environ.get("SWAP_QK_LOOP_DEBUG", "0") == "1"

    mod = get_ffa_jit_mod(
        direction="bwd",
        head_dim=q.shape[-1],
        compute_dtype=q.dtype,
        output_dtype=dk_type
        or (k.dtype if disable_bwd_dkv_atomic_reduction else torch.float32),
        softcap=softcap > 0.0,
        disable_atomic_reduction=disable_bwd_dkv_atomic_reduction,
        deterministic=deterministic,
        profile_mode=profile_mode,
        auto_range_merge=auto_range_merge,
        # DE-BUG
        swap_bwd_qk_loop=True if swap_qk_loop_debug else swap_bwd_qk_loop,
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
        merge_k_ranges,
        bwd_kq_map,
        bwd_unique_count,
        softmax_scale,
        softcap,
        disable_bwd_dkv_atomic_reduction,
        dq_type,
        dk_type,
        dv_type,
        sink_layout,
        deterministic,
        sm_margin,
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
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    softmax_scale: float,
    softcap: float,
    disable_bwd_dkv_atomic_reduction: bool,
    dq_type: torch.dtype | None,
    dk_type: torch.dtype | None,
    dv_type: torch.dtype | None,
    deterministic: bool,
    sm_margin: int,
    auto_range_merge: bool,
    merge_k_ranges: torch.Tensor | None,
    bwd_kq_map: torch.Tensor | None,
    bwd_unique_count: torch.Tensor | None,
    swap_bwd_qk_loop: bool,
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
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    softmax_scale: float,
    softcap: float,
    disable_bwd_dkv_atomic_reduction: bool,
    dq_type: torch.dtype | None,
    dk_type: torch.dtype | None,
    dv_type: torch.dtype | None,
    deterministic: bool,
    sm_margin: int,
    auto_range_merge: bool = False,
    merge_k_ranges: torch.Tensor | None = None,
    bwd_kq_map: torch.Tensor | None = None,
    bwd_unique_count: torch.Tensor | None = None,
    swap_bwd_qk_loop: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if profile_mode:  # NOTE: stop_event is called inside the kernel
        ffa_utils.start_event("bwd_prepare")

    # make all input tensors contiguous before initializing output buffers
    # NOTE: in backward, torch.compiler allows neither making nor checking contiguity
    # so we just skip here, but check inside the kernel
    if not torch.compiler.is_compiling():
        dout, q, k, v, sink, out, q_ranges, k_ranges = [
            maybe_contiguous(x) for x in (dout, q, k, v, sink, out, q_ranges, k_ranges)
        ]

    dq = torch.zeros_like(q, dtype=dq_type or torch.float32) if dq is None else dq
    dk = torch.zeros_like(k, dtype=dk_type or torch.float32) if dk is None else dk
    dv = torch.zeros_like(v, dtype=dv_type or torch.float32) if dv is None else dv
    dsink = (
        (torch.zeros_like(sink, dtype=torch.float32) if dsink is None else dsink)
        if sink is not None
        else None
    )

    # NOTE: we can not directly compile `_flex_flash_attn_backward`
    # since torch.compile does not allow returning the mutated args (dq, dk, dv, dsink)
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
        disable_bwd_dkv_atomic_reduction=disable_bwd_dkv_atomic_reduction,
        dq_type=dq_type,
        dk_type=dk_type,
        dv_type=dv_type,
        deterministic=deterministic,
        sm_margin=sm_margin,
        auto_range_merge=auto_range_merge,
        merge_k_ranges=merge_k_ranges,
        bwd_kq_map=bwd_kq_map,
        bwd_unique_count=bwd_unique_count,
        swap_bwd_qk_loop=swap_bwd_qk_loop,
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
        auto_range_merge: bool = False,
        ref_block_size: tuple[int, int] | None = None,
        swap_ab: bool = False,
        swap_bwd_qk_loop: bool = False,
    ):
        softmax_scale = (
            q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        )

        if auto_range_merge:
            with maybe_profile_ffa_ctx("fwd_range_merge"):
                (
                    merge_q_ranges,
                    fwd_q_ranges,
                    fwd_k_ranges,
                    fwd_attn_type_map,
                    fwd_qk_map,
                    fwd_unique_count,
                ) = merge_ranges(q_ranges, k_ranges, attn_type_map=attn_type_map)
        else:
            fwd_q_ranges = q_ranges
            fwd_k_ranges = k_ranges
            fwd_attn_type_map = attn_type_map
            merge_q_ranges = None
            fwd_qk_map = None
            fwd_unique_count = None

        out, lse = _flex_flash_attn_forward(
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
            disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
            out_type=q.dtype
            if disable_fwd_atomic_reduction
            else torch.float32,  # out_type
            deterministic=deterministic,
            sm_margin=sm_margin,
            # optional args below mainly for sparse attn
            auto_range_merge=auto_range_merge,
            merge_q_ranges=merge_q_ranges,
            qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            ref_block_size=ref_block_size,
            swap_ab=swap_ab,
        )

        # Cast output to the same dtype as q
        with maybe_profile_ffa_ctx("fwd_cast"):
            out = out.to(q.dtype)

        ctx.save_for_backward(
            q, k, v, sink, out, lse, q_ranges, k_ranges, attn_type_map
        )

        ctx.sink_layout = sink_layout
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.auto_range_merge = auto_range_merge
        ctx.ref_block_size = ref_block_size
        ctx.swap_ab = swap_ab
        ctx.swap_bwd_qk_loop = swap_bwd_qk_loop

        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args):  # pragma: no cover
        q, k, v, sink, out, lse, q_ranges, k_ranges, attn_type_map = ctx.saved_tensors

        if ctx.auto_range_merge:
            with maybe_profile_ffa_ctx("bwd_range_merge"):
                (
                    merge_k_ranges,
                    bwd_k_ranges,
                    bwd_q_ranges,
                    bwd_attn_type_map,
                    bwd_kq_map,
                    bwd_unique_count,
                ) = merge_ranges(k_ranges, q_ranges, attn_type_map=attn_type_map)
        else:
            bwd_q_ranges, bwd_k_ranges, bwd_attn_type_map = (
                q_ranges,
                k_ranges,
                attn_type_map,
            )
            merge_k_ranges, bwd_kq_map, bwd_unique_count = None, None, None

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
            disable_bwd_dkv_atomic_reduction=False,
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=ctx.deterministic,
            sm_margin=ctx.sm_margin,
            # optional args below mainly for sparse attn
            auto_range_merge=ctx.auto_range_merge,
            merge_k_ranges=merge_k_ranges,
            bwd_kq_map=bwd_kq_map,
            bwd_unique_count=bwd_unique_count,
            swap_bwd_qk_loop=ctx.swap_bwd_qk_loop,
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
            None,  # auto_range_merge
            None,  # ref_block_size
            None,  # swap_ab
            None,  # swap_bwd_qk_loop
        )


# -------------------       ffa interface   ------------------- #


def flex_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None = None,
    *,
    sink: torch.Tensor | None = None,
    sink_layout: AttnSinkLayout = "sh",
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    disable_fwd_atomic_reduction: bool = False,
    auto_range_merge: bool = False,
    ref_block_size: tuple[int, int] | None = None,
    swap_ab: bool = False,
    swap_bwd_qk_loop: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    An interface similar to flash attention that doesn't require distributed environment, dispatch or undispatch.
    Directly call magi_attn_kernel to get attention output and lse. This is faster when you don't need context parallel.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.

        q_ranges (torch.Tensor): Query ranges tensor to represent the attn mask.
        k_ranges (torch.Tensor): Key ranges tensor to represent the attn mask.
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

        auto_range_merge (bool, optional):
            Whether to automatically merge k_ranges for the same q_range. Defaults to ``False``.
            **Note:** This flag is useful for sparse attention scenarios but still under development.

        ref_block_size (tuple[int, int], optional):
            Reference block size (M, N) for kernel selection.
            Defaults to ``None`` to use the internal heuristic.
            **Note:** This flag is useful for sparse attention scenarios but still under development.

        swap_ab (bool, optional): Whether to swap the order of A and B operands for the matmul operation
            (i.e. transpose `C=A x B^T` to `C^T= B x A^T`) in attention forward/backward passes. Defaults to ``False``.
            **Note:** This flag is useful for sparse attention scenarios but still under development.

        swap_bwd_qk_loop (bool, optional): Whether to swap the order of Q and K double-loops
            (i.e. from the default `K for outer-loop and Q for inner-loop` to `Q for outer-loop and K for inner-loop`)
            in the attention backward pass. Defaults to ``False``.
            **Note:** This flag is useful for sparse attention scenarios but still under development.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - out (torch.Tensor): Attention output tensor
            - lse (torch.Tensor): Log-sum-exp values with dtype=torch.float32.

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

    assert not (auto_range_merge and deterministic), (
        "auto_range_merge and deterministic can't be True at the same time, "
        "due to some unresolved bug to be fixed as soon as possible."
    )

    return FlexFlashAttnFunc.apply(
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
        auto_range_merge,
        ref_block_size,
        swap_ab,
        swap_bwd_qk_loop,
    )
