import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.interface import _flash_attn_bwd

import torch
from einops import reduce

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.utils import make_attn_mask_from_ffa_args, to_higher_fp_dtype



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
    assert sink is None, "FA4 backend does not support leanable sink"
    raise NotImplementedError


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert sink is None, "FA4 backend does not support leanable sink"
    raise NotImplementedError