# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import math

import torch
from einops import reduce

from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.utils._utils import get_attn_mask_from_ffa_args

from .utils import safe_subtract

__all__ = [
    "sdpa_fwd",
    "sdpa_bwd",
]


# ------------------        sdpa fwd       ------------------ #


def sdpa_fwd_qkv_rearrange(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # reshape qkv to [1, num_heads, num_tokens, head_dim]
    q, k, v = [e.transpose(0, 1).unsqueeze(0).contiguous() for e in (q, k, v)]

    return q, k, v


def sdpa_fwd_out_lse_rearrange(
    out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # reshape out to [num_tokens, num_heads, head_dim]
    out = out.squeeze(0).transpose(0, 1).contiguous()
    # reshape lse to [num_tokens, num_heads]
    lse = lse.squeeze(0).transpose(0, 1).contiguous()

    return out, lse


def sdpa_fwd_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    sq, sk = q.size(-2), k.size(-2)
    softmax_scale = (
        1 / math.sqrt(q.size(-1)) if softmax_scale is None else softmax_scale
    )
    attn_bias = torch.zeros(sq, sk, dtype=q.dtype, device=q.device)
    nhq, nhk = q.size(-3), k.size(-3)
    rep_times = nhq // nhk

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(sq, sk, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if rep_times > 1:
        k = k.repeat_interleave(rep_times, -3)
        v = v.repeat_interleave(rep_times, -3)

    return q, k, v, attn_bias, softmax_scale, rep_times


def sdpa_fwd_calc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weight = q @ k.transpose(-2, -1) * softmax_scale
    attn_weight += attn_bias

    # NOTE: this lse is numerically stabilized according to
    # https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch-logsumexp
    # and `-inf` will result in `-inf` correctly as well
    lse = torch.logsumexp(attn_weight, dim=-1, keepdim=True)

    # BUG: pytorch softmax cannot assure the sum to 1 when dtype is float64
    # attn_weight = torch.softmax(attn_weight, dim=-1)

    # NOTE: two -inf subtraction will result in nan, but we need -inf
    # attn_weight = torch.exp(attn_weight - lse)
    attn_weight = torch.exp(safe_subtract(attn_weight, lse))

    out = attn_weight @ v

    return out, lse.squeeze(-1)


def _sdpa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v, attn_bias, softmax_scale, _ = sdpa_fwd_preprocess(
        q, k, v, attn_mask, is_causal, softmax_scale
    )

    out, lse = sdpa_fwd_calc(q, k, v, attn_bias, softmax_scale)

    return out, lse


@torch.no_grad()
def sdpa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_arg: AttnArg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SDPA forward function

    Args:
        q (torch.Tensor): [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]
        k (torch.Tensor): [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]
        v (torch.Tensor): [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]

        attn_arg (AttnArg): attention arguments for ffa

    Returns:
        torch.Tensor: out with shape [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]

        torch.Tensor: lse with shape [num_tokens_q, num_heads_q]
            or [batch_size, num_heads_q, num_tokens_q]
    """
    rearrange = len(q.shape) == 3  # from [t, nh, hd] to [1, nh, t, hd]

    if rearrange:
        q, k, v = sdpa_fwd_qkv_rearrange(q, k, v)

    # construct attn_mask from ranges
    attn_mask = get_attn_mask_from_ffa_args(
        q_ranges=attn_arg.q_ranges,
        k_ranges=attn_arg.k_ranges,
        attn_type_map=attn_arg.attn_type_map,
        total_seqlen_q=q.shape[-2],
        total_seqlen_k=k.shape[-2],
        device=torch.cuda.current_device(),
    )

    out, lse = _sdpa_fwd(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=False,
        softmax_scale=q.shape[-1] ** -0.5,
    )

    if rearrange:
        out, lse = sdpa_fwd_out_lse_rearrange(out, lse)

    return out, lse


# ------------------        sdpa bwd       ------------------ #


def sdpa_bwd_qkvodo_lse_rearrange(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    # reshape q, k, v, do to [1, num_heads, num_tokens, head_dim]
    # and lse to [1, num_heads, num_tokens]
    q, k, v, o, do, lse = [
        e.transpose(0, 1).unsqueeze(0).contiguous() for e in (q, k, v, o, do, lse)
    ]

    return q, k, v, o, do, lse


def sdpa_bwd_dqdkdv_rearrange(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # reshape dq,dk,dv to [num_tokens, num_heads, head_dim]
    dq, dk, dv = [e.squeeze(0).transpose(0, 1).contiguous() for e in (dq, dk, dv)]

    return dq, dk, dv


def sdpa_bwd_recalc_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    attn_bias: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    attn_weight = q @ k.transpose(-2, -1) * softmax_scale
    attn_weight += attn_bias

    # NOTE: two -inf subtraction will result in nan, but we need -inf
    # attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    attn_weight = torch.exp(safe_subtract(attn_weight, lse.unsqueeze(-1)))

    return attn_weight


def sdpa_bwd_calc(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    attn_weight: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dv = attn_weight.transpose(-2, -1) @ do
    grad_weight = do @ v.transpose(-2, -1)
    # grad_weight = softmax_bwd(grad_weight, attn_weight) * softmax_scale
    grad_weight = attn_weight * (grad_weight - delta) * softmax_scale
    dq = grad_weight @ k
    dk = grad_weight.transpose(-2, -1) @ q

    return dq, dk, dv


def sdpa_bwd_preprocess(
    do: torch.Tensor,
    o: torch.Tensor,
) -> torch.Tensor:
    # shape: [b, nh, sq, 1]
    delta = (do * o).sum(-1, keepdim=True)
    return delta


def sdpa_bwd_postprocess(
    dk: torch.Tensor,
    dv: torch.Tensor,
    rep_times: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rep_times > 1:
        dk = reduce(
            dk,
            "... (nhk rep_times) s hd -> ... nhk s hd",
            reduction="sum",
            rep_times=rep_times,
        )
        dv = reduce(
            dv,
            "... (nhk rep_times) s hd -> ... nhk s hd",
            reduction="sum",
            rep_times=rep_times,
        )

    return dk, dv


def _sdpa_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, attn_bias, softmax_scale, rep_times = sdpa_fwd_preprocess(
        q, k, v, attn_mask, is_causal, softmax_scale
    )

    delta = sdpa_bwd_preprocess(do, o)

    attn_weight = sdpa_bwd_recalc_fwd(q, k, lse, attn_bias, softmax_scale)

    dq, dk, dv = sdpa_bwd_calc(do, q, k, v, delta, attn_weight, softmax_scale)

    dk, dv = sdpa_bwd_postprocess(dk, dv, rep_times)

    return dq, dk, dv


@torch.no_grad()
def sdpa_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_arg: AttnArg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDPA backward function

    Args:
        do (torch.Tensor): [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]
        q (torch.Tensor): [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]
        k (torch.Tensor): [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]
        v (torch.Tensor): [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]
        o (torch.Tensor): [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]
        lse (torch.Tensor): [num_tokens_q, num_heads_q]
            or [batch_size, num_heads_q, num_tokens_q]
        attn_arg (AttnArg): attention arguments for ffa

    Returns:
        torch.Tensor: dq with shape [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]

        torch.Tensor: dk with shape [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]

        torch.Tensor: dv with shape [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]
    """
    rearrange = len(q.shape) == 3  # from [t, nh, hd] to [1, nh, t, hd]

    if rearrange:
        q, k, v, o, do, lse = sdpa_bwd_qkvodo_lse_rearrange(q, k, v, o, do, lse)

    # construct attn_mask from ranges
    attn_mask = get_attn_mask_from_ffa_args(
        q_ranges=attn_arg.q_ranges_bwd,
        k_ranges=attn_arg.k_ranges_bwd,
        attn_type_map=attn_arg.attn_type_map_bwd,
        total_seqlen_q=q.shape[-2],
        total_seqlen_k=k.shape[-2],
        device=torch.cuda.current_device(),
    )

    dq, dk, dv = _sdpa_bwd(
        do=do,
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        attn_mask=attn_mask,
        is_causal=False,
        softmax_scale=q.shape[-1] ** -0.5,
    )

    if rearrange:
        dq, dk, dv = sdpa_bwd_dqdkdv_rearrange(dq, dk, dv)

    return dq, dk, dv
