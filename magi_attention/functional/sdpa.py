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


import torch
from einops import reduce

from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.utils import get_attn_mask_from_ffa_args, to_higher_fp_dtype

from .utils import correct_attn_out_lse_with_sink, safe_softmax, sink_bwd, softmax_bwd

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
    softmax_scale = q.size(-1) ** (-0.5) if softmax_scale is None else softmax_scale
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
    attn_weight = to_higher_fp_dtype(
        q @ k.transpose(-2, -1) * softmax_scale,
        lowest_precision=torch.float32,
    )
    attn_weight += attn_bias

    # NOTE: this lse is numerically stabilized according to
    # https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch-logsumexp
    # and `-inf` will result in `-inf` correctly as well
    lse = torch.logsumexp(attn_weight, dim=-1, keepdim=True)

    # NOTE: pytorch softmax has many limitations and bugs
    # thus we use our own safe_softmax with lse involved
    attn_weight = safe_softmax(attn_weight, lse).to(v.dtype)

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
    sink: torch.Tensor | None,
    attn_arg: AttnArg,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SDPA forward function

    Args:
        q (torch.Tensor): [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]
        k (torch.Tensor): [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]
        v (torch.Tensor): [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]
        sink (torch.Tensor, optional): [num_tokens_sink, num_heads_q]
            Defaults to ``None`` to not apply attention sink.

        attn_arg (AttnArg): attention arguments for ffa

        softmax_scale (float, optional): softmax scale.
            Defaults to None to use default value: 1/sqrt(head_dim)
        softcap (float, optional): softcap. Defaults to 0.

    Returns:
        torch.Tensor: out with shape [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]

        torch.Tensor: lse with shape [num_tokens_q, num_heads_q]
            or [batch_size, num_heads_q, num_tokens_q]
    """
    assert softcap == 0.0, "non-zero softcap is not supported by now"

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
        softmax_scale=softmax_scale,
    )

    if rearrange:
        out, lse = sdpa_fwd_out_lse_rearrange(out, lse)

    if sink is not None:
        assert rearrange
        out, lse = correct_attn_out_lse_with_sink(
            out=out,
            lse=lse,
            sink=sink,
            inplace=True,
        )

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
    attn_weight = to_higher_fp_dtype(
        q @ k.transpose(-2, -1) * softmax_scale,
        lowest_precision=torch.float32,
    )
    attn_weight += attn_bias

    # NOTE: pytorch softmax has many limitations and bugs
    # thus we use our own safe_softmax with lse involved
    attn_weight = safe_softmax(attn_weight, lse.unsqueeze(-1)).to(q.dtype)

    return attn_weight


def sdpa_bwd_calc(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor | None,
    attn_weight: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dv = attn_weight.transpose(-2, -1) @ do
    grad_weight = do @ v.transpose(-2, -1)
    if delta is None:
        grad_weight = softmax_bwd(grad_weight, attn_weight) * softmax_scale
    else:  # an easier way to compute grad_weight if delta is given
        grad_weight = attn_weight * (grad_weight - delta) * softmax_scale
    dq = grad_weight @ k
    dk = grad_weight.transpose(-2, -1) @ q

    return dq, dk, dv


def sdpa_bwd_preprocess(
    do: torch.Tensor,
    o: torch.Tensor,
) -> torch.Tensor:
    """Calculate delta from o and do
    to avoid massive dot product of p and dp
    where for each row i:
    delta_i := p_i.T x dp_i
        = p_i.T x (v x do_i)
        = (p_i.T x v) x do_i
        = o_i.T x do_i
    """
    # shape: [b, nh, sq, 1]
    delta = (o * do).sum(-1, keepdim=True)
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
    sink: torch.Tensor | None,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_arg: AttnArg,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        sink (torch.Tensor, optional): [num_tokens_sink, num_heads_q]
            Defaults to ``None`` to not calculate dsink.
        o (torch.Tensor): [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]
        lse (torch.Tensor): [num_tokens_q, num_heads_q]
            or [batch_size, num_heads_q, num_tokens_q]
        attn_arg (AttnArg): attention arguments for ffa

        softmax_scale (float, optional): softmax scale.
            Defaults to None to use default value: 1/sqrt(head_dim)
        softcap (float, optional): softcap. Defaults to 0.

    Returns:
        torch.Tensor: dq with shape [num_tokens_q, num_heads_q, head_dim]
            or [batch_size, num_heads_q, num_tokens_q, head_dim]

        torch.Tensor: dk with shape [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]

        torch.Tensor: dv with shape [num_tokens_kv, num_heads_kv, head_dim]
            or [batch_size, num_heads_kv, num_tokens_kv, head_dim]

        torch.Tensor or None: dsink with shape [num_tokens_sink, num_heads_q]
            or None if sink is None
    """
    assert softcap == 0.0, "non-zero softcap is not supported by now"

    rearrange = len(q.shape) == 3  # from [t, nh, hd] to [1, nh, t, hd]

    if sink is not None:
        assert rearrange
        dsink = sink_bwd(
            sink=sink,
            lse=lse,
            o=o,
            do=do,
        )
    else:
        dsink = None

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
        softmax_scale=softmax_scale,
    )

    if rearrange:
        dq, dk, dv = sdpa_bwd_dqdkdv_rearrange(dq, dk, dv)

    return dq, dk, dv, dsink
