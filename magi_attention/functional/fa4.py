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

is_fa4_installed = False
try:
    from flash_attn_cute.interface import _flash_attn_bwd, _flash_attn_fwd

    is_fa4_installed = True
except ImportError:
    pass

if is_fa4_installed:
    from .fa4_utils import load_precompiled_ffa_fa4

    load_precompiled_ffa_fa4()


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
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        arbitrary=True,  # NOTE: to eanble arbitrary mask functionality
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
    assert sink is None, "FA4 backend does not support leanable sink"
    assert isinstance(attn_arg, FA4AttnArg), "FA4 is only supported for FA4AttnArg"

    fa4_args = attn_arg.to_fa4_args(is_bwd=True)

    # Rearrange q,k,v,o,do: (s, h, d) -> (1, s, h, d)
    q, k, v, o, do = [x.unsqueeze(0) for x in (q, k, v, o, do)]

    # Rearrange lse: (s, h) -> (1, h, s)
    lse = lse.mT.unsqueeze(0).contiguous()

    dq, dk, dv = _flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=o,
        dout=do,
        lse=lse,
        softmax_scale=softmax_scale,
        causal=False,
        arbitrary=True,  # NOTE: to eanble arbitrary mask functionality
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

            # Create FA4AttnArg (reuses _transfer_ffa_args_to_fa4_args from calc_meta.py)
            fa4_attn_arg = FA4AttnArg(
                q_ranges=AttnRanges.from_ranges(q_ranges_list),
                k_ranges=AttnRanges.from_ranges(k_ranges_list),
                attn_type_map=attn_type_map_list,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
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
        )

        # Return gradients for each input (None for non-tensor args)
        return dq, dk, dv, None, None, None, None, None, None


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
        reuse_attn_arg,
    )
