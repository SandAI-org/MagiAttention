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


import torch

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.collection.calc_meta import AttnArg, FA4AttnArg

_has_cutlass_backend = False
_has_cute_backend = False
try:
    # CUTLASS package layout (ffa_fa3).
    from flash_attn_cute.ffa_fa3.flash_attn_interface import (
        _flash_attn_backward as _flash_attn_backward_cutlass,
    )
    from flash_attn_cute.ffa_fa3.flash_attn_interface import (
        _flash_attn_forward as _flash_attn_forward_cutlass,
    )

    _has_cutlass_backend = True
except ImportError:
    pass

try:
    # Original DSL interface.
    from flash_attn_cute.interface import _flash_attn_bwd, _flash_attn_fwd

    _has_cute_backend = True
except ImportError:
    pass

is_fa4_installed = _has_cutlass_backend or _has_cute_backend


def _use_cutlass_on_current_device() -> bool:
    """
    Runtime check for backend routing on the active CUDA device.
    """
    if not _has_cutlass_backend or not torch.cuda.is_available():
        return False
    cc_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    return cc_major in (8, 9)


if _has_cute_backend and not _use_cutlass_on_current_device():
    from .fa4_utils import load_precompiled_ffa_fa4

    load_precompiled_ffa_fa4()


def _should_use_cutlass_backend(device: torch.device) -> bool:
    """Use ffa_fa3 on sm80/sm90 when available."""
    cc_major, _ = torch.cuda.get_device_capability(device)
    if cc_major in (8, 9):
        if not _has_cutlass_backend:
            raise RuntimeError(
                "Detected sm80/sm90 GPU, but cutlass backend (flash_attn_cute.ffa_fa3) "
                "is not available."
            )
        return True
    return False


def _extract_linear_sparse_tensors(block_sparse):
    """
    Convert cutlass block sparse input into flat 6 tensors.
    Supports LinearBlockSparseTensors object or tuple of 6 tensors.
    """

    def _ensure_cnt_3d(t: torch.Tensor | None, name: str) -> torch.Tensor | None:
        if t is None:
            return None
        if t.dim() == 3:
            return t
        if t.dim() == 2:
            # [B, num_m_blocks] -> [B, 1, num_m_blocks] for broadcasting
            return t.unsqueeze(1)
        if t.dim() == 1:
            # [num_m_blocks] -> [1, 1, num_m_blocks] for broadcasting
            return t.unsqueeze(0).unsqueeze(0)
        raise RuntimeError(f"{name} must be 1D/2D/3D, got shape={tuple(t.shape)}")

    if block_sparse is None:
        return (None, None, None, None, None, None)
    if hasattr(block_sparse, "mask_block_cnt"):
        mask_cnt = _ensure_cnt_3d(block_sparse.mask_block_cnt, "mask_block_cnt")
        full_cnt = _ensure_cnt_3d(block_sparse.full_block_cnt, "full_block_cnt")
        return (
            mask_cnt,
            block_sparse.mask_block_offset,
            block_sparse.mask_block_idx,
            full_cnt,
            block_sparse.full_block_offset,
            block_sparse.full_block_idx,
        )
    (
        mask_cnt,
        mask_offset,
        mask_idx,
        full_cnt,
        full_offset,
        full_idx,
    ) = block_sparse
    return (
        _ensure_cnt_3d(mask_cnt, "mask_block_cnt"),
        mask_offset,
        mask_idx,
        _ensure_cnt_3d(full_cnt, "full_block_cnt"),
        full_offset,
        full_idx,
    )


@torch.no_grad()
def fa4_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    attn_arg: AttnArg,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sink_layout: AttnSinkLayout = "sh",
) -> tuple[torch.Tensor, torch.Tensor]:
    assert is_fa4_installed, "FlashAttn4 is not installed"
    assert isinstance(attn_arg, FA4AttnArg), "FA4 is only supported for FA4AttnArg"

    # Get FA4 arguments
    fa4_args = attn_arg.to_fa4_args(is_bwd=False)

    # Rearrange q,k,v: (s, h, d) -> (1, s, h, d)
    q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    if _should_use_cutlass_backend(q.device):
        (
            q2k_mask_cnt,
            q2k_mask_offset,
            q2k_mask_idx,
            q2k_full_cnt,
            q2k_full_offset,
            q2k_full_idx,
        ) = _extract_linear_sparse_tensors(fa4_args["linear_k_block_sparse_mask"])
        (
            k2q_mask_cnt,
            k2q_mask_offset,
            k2q_mask_idx,
            k2q_full_cnt,
            k2q_full_offset,
            k2q_full_idx,
        ) = _extract_linear_sparse_tensors(fa4_args["linear_q_block_sparse_mask"])
        out, lse, *_ = _flash_attn_forward_cutlass(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=softcap,
            num_splits=1,
            pack_gqa=False,
            sm_margin=0,
            arbitrary_func=fa4_args["aux_tensors"][0],
            # Q2K block sparse for forward
            block_sparse_mask_cnt=q2k_mask_cnt,
            block_sparse_mask_offset=q2k_mask_offset,
            block_sparse_mask_idx=q2k_mask_idx,
            block_sparse_full_cnt=q2k_full_cnt,
            block_sparse_full_offset=q2k_full_offset,
            block_sparse_full_idx=q2k_full_idx,
            # K2Q block sparse saved for backward
            k2q_block_sparse_mask_cnt=k2q_mask_cnt,
            k2q_block_sparse_mask_offset=k2q_mask_offset,
            k2q_block_sparse_mask_idx=k2q_mask_idx,
            k2q_block_sparse_full_cnt=k2q_full_cnt,
            k2q_block_sparse_full_offset=k2q_full_offset,
            k2q_block_sparse_full_idx=k2q_full_idx,
        )
    else:
        if not _has_cute_backend:
            raise RuntimeError(
                "FA4 CUDA backend is not available for this architecture. "
                "Need flash_attn_cute.interface for non-sm80/sm90 devices."
            )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            arbitrary=True,  # NOTE: to enable arbitrary mask functionality
            window_size_left=None,
            window_size_right=None,
            learnable_sink=sink,
            softcap=softcap,
            num_splits=1,
            pack_gqa=False,
            mask_mod=None,
            return_lse=True,
            block_sparse_tensors=fa4_args["linear_k_block_sparse_mask"],
            aux_tensors=fa4_args["aux_tensors"],
        )

    # Rearrange out: (1, s, h, d) -> (s, h, d)
    out = out.squeeze(0)
    # Rearrange lse: (1, h, s) -> (s, h)
    lse = lse.squeeze(0).mT

    return out, lse


@torch.no_grad()
def fa4_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_arg: AttnArg,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sink_layout: AttnSinkLayout = "sh",
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert is_fa4_installed, "FA4 backend is not installed"
    assert sink is None, "FA4 backend does not support learnable sink"
    assert isinstance(attn_arg, FA4AttnArg), "FA4 is only supported for FA4AttnArg"

    fa4_args = attn_arg.to_fa4_args(is_bwd=True)

    # Rearrange q,k,v,o,do: (s, h, d) -> (1, s, h, d)
    q, k, v, o, do = [x.unsqueeze(0) for x in (q, k, v, o, do)]

    # Rearrange lse: (s, h) -> (1, h, s)
    lse = lse.mT.unsqueeze(0).contiguous()

    if _should_use_cutlass_backend(q.device):
        (
            k2q_mask_cnt,
            k2q_mask_offset,
            k2q_mask_idx,
            k2q_full_cnt,
            k2q_full_offset,
            k2q_full_idx,
        ) = _extract_linear_sparse_tensors(fa4_args["linear_q_block_sparse_mask"])
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        _flash_attn_backward_cutlass(
            dout=do,
            q=q,
            k=k,
            v=v,
            out=o,
            softmax_lse=lse,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            sequed_q=None,
            sequed_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=softmax_scale,
            is_causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=softcap,
            deterministic=deterministic,
            sm_margin=0,
            arbitrary_func=fa4_args["aux_tensors"][0],
            block_sparse_mask_cnt=k2q_mask_cnt,
            block_sparse_mask_offset=k2q_mask_offset,
            block_sparse_mask_idx=k2q_mask_idx,
            block_sparse_full_cnt=k2q_full_cnt,
            block_sparse_full_offset=k2q_full_offset,
            block_sparse_full_idx=k2q_full_idx,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
    else:
        if not _has_cute_backend:
            raise RuntimeError(
                "FA4 CUDA backend is not available for this architecture. "
                "Need flash_attn_cute.interface for non-sm80/sm90 devices."
            )
        dq, dk, dv = _flash_attn_bwd(
            q=q,
            k=k,
            v=v,
            out=o,
            dout=do,
            lse=lse,
            softmax_scale=softmax_scale,
            causal=False,
            arbitrary=True,  # NOTE: to enable arbitrary mask functionality
            softcap=softcap,
            block_sparse_tensors=fa4_args["linear_q_block_sparse_mask"],
            aux_tensors=fa4_args["aux_tensors"],
            deterministic=deterministic,
        )
    dsink = None

    # Rearrange dq,dk,dv: (1, s, h, d) -> (s, h, d)
    dq, dk, dv = dq.squeeze(0), dk.squeeze(0), dv.squeeze(0)

    return dq, dk, dv, dsink


class FA4AttnFunc(torch.autograd.Function):
    """Autograd function for FA4 backend with arbitrary mask support.

    Uses FA4AttnArg from calc_meta.py to build FA4 args.
    """

    # Cache for reusing FA4AttnArg across calls
    _cached_fa4_attn_arg = None

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor | None,
        softmax_scale: float | None,
        softcap: float,
        deterministic: bool = False,
        reuse_attn_arg: bool = False,
    ):
        softmax_scale = (
            q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        )

        # Reuse cached FA4AttnArg if available and requested
        if reuse_attn_arg and FA4AttnFunc._cached_fa4_attn_arg is not None:
            fa4_attn_arg = FA4AttnFunc._cached_fa4_attn_arg  # type: ignore[unreachable]
        else:
            seqlen_q = q.shape[0]
            seqlen_k = k.shape[0]

            # Build AttnRanges from tensor (required by FA4AttnArg interface)
            q_ranges_list = q_ranges.cpu().tolist()
            k_ranges_list = k_ranges.cpu().tolist()

            # Build attn_type_map list
            if attn_type_map is None:
                attn_type_map_list = [0] * len(q_ranges_list)
            else:
                attn_type_map_list = attn_type_map.cpu().tolist()

            fa4_attn_arg = FA4AttnArg(
                q_ranges=AttnRanges.from_ranges(q_ranges_list),
                k_ranges=AttnRanges.from_ranges(k_ranges_list),
                attn_type_map=attn_type_map_list,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                headdim=q.shape[-1],
            )
            # Cache for future reuse
            FA4AttnFunc._cached_fa4_attn_arg = fa4_attn_arg

        out, lse = fa4_fwd(
            q=q,
            k=k,
            v=v,
            sink=None,
            attn_arg=fa4_attn_arg,
            softmax_scale=softmax_scale,
            softcap=softcap,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse, q_ranges, k_ranges, attn_type_map)
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.fa4_attn_arg = fa4_attn_arg

        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args):
        q, k, v, out, lse, q_ranges, k_ranges, attn_type_map = ctx.saved_tensors

        # Call fa4_bwd
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
            deterministic=ctx.deterministic,
        )

        # Return gradients for each input (None for non-tensor args)
        return dq, dk, dv, None, None, None, None, None, None, None


def ffa_fa4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None = None,
    *,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    deterministic: bool = False,
    reuse_attn_arg: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FA4 backend version of flex_flash_attn_func for benchmarking.

    Similar interface to flex_flash_attn_func but uses FA4 backend with arbitrary mask support.
    Supports both forward and backward passes.

    Args:
        q (torch.Tensor): Query tensor with shape (seqlen_q, num_heads_q, head_dim).
        k (torch.Tensor): Key tensor with shape (seqlen_k, num_heads_k, head_dim).
        v (torch.Tensor): Value tensor with shape (seqlen_k, num_heads_k, head_dim).
        q_ranges (torch.Tensor): Query ranges tensor with shape (num_ranges, 2).
        k_ranges (torch.Tensor): Key ranges tensor with shape (num_ranges, 2).
        attn_type_map (torch.Tensor, optional): Attention type map tensor.
        softmax_scale (float, optional): Softmax scale.
        softcap (float): Softcap value.
        deterministic (bool): If True, use deterministic backward pass.
        reuse_attn_arg (bool): If True, reuse the cached FA4AttnArg from previous call.
            Set to False for warmup/first call, then True for subsequent calls
            to measure only kernel time without FA4AttnArg creation overhead.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (out, lse)
    """
    return FA4AttnFunc.apply(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        attn_type_map,
        softmax_scale,
        softcap,
        deterministic,
        reuse_attn_arg,
    )
