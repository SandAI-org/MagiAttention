from typing import cast

import torch
from einops import reduce, rearrange

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.meta.collection.calc_meta import AttnArg, FA4AttnArg

is_fa4_installed = False
try:
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.interface import _flash_attn_bwd
    
    is_fa4_installed = True
except ImportError:
    pass


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
    fa4_args = cast(FA4AttnArg, attn_arg).to_fa4_args(is_bwd=False)
    
    # Rearrange q,k,v
    q, k, v = [
        rearrange(x, "s h d -> 1 s h d")
        for x in (q, k, v)
    ]

    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        arbitrary=True, # NOTE: to eanble arbitrary mask functionality
        window_size_left=None,
        window_size_right=None,
        learnable_sink=sink,
        softcap=softcap,
        num_splits=1,
        pack_gqa=False,
        mask_mod=None,
        block_sparse_tensors=fa4_args["linear_k_block_sparse_mask"],
        aux_tensors=fa4_args["aux_tensors"],
    )
    
    # Rearrange out, lse
    out = rearrange(out, "1 s h d -> s h d")
    lse = rearrange(lse, "1 h s -> s h")
    
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
    
    fa4_args = cast(FA4AttnArg, attn_arg).to_fa4_args(is_bwd=True)
    
    # Rearrange q,k,v,o,do
    q, k, v, o, do = [
        rearrange(x, "s h d -> 1 s h d")
        for x in (q, k, v, o, do)
    ]
    
    # Rearange lse
    lse = rearrange(lse, "s h -> 1 h s").contiguous()
    
    dq, dk, dv = _flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=o,
        dout=do,
        lse=lse,
        softmax_scale=softmax_scale,
        causal=False,
        arbitrary=True, # NOTE: to eanble arbitrary mask functionality
        softcap=softcap,
        block_sparse_tensors=fa4_args["linear_q_block_sparse_mask"],
        aux_tensors=fa4_args["aux_tensors"],
        deterministic=deterministic,
    )
    dsink = None
    
    # Rearrange dq,dk,dv
    dq, dk, dv = [
        rearrange(x, "1 s h d -> s h d")
        for x in (dq, dk, dv)
    ]
    
    return dq, dk, dv, dsink