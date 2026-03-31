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

"""
Online (block-wise) SDPA backend.

Uses tiled online softmax to avoid materializing the full [sq, sk] attention
weight matrix, trading compute for memory.  The backward pass recomputes P
block-by-block from the saved LSE, identical to the offline SDPA backward.
"""

import torch

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.common.forward_meta import AttnForwardMeta
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.utils import (
    make_attn_mask_from_ffa_args,
    make_slice_mask_from_ffa_attn_type,
    max_fp_dtype,
    to_higher_fp_dtype,
)

from .sdpa import sdpa_bwd_dqdkdv_rearrange, sdpa_bwd_qkvodo_lse_rearrange
from .utils import (
    correct_attn_out_lse,
    correct_attn_out_lse_with_sink,
    safe_softmax,
    sink_bwd,
)

__all__ = [
    "sdpa_online_fwd",
    "sdpa_online_bwd",
]

BLOCK_Q: int = 1024
BLOCK_K: int = 1024


# ------------------   block-level bias helper   ------------------ #


def _make_block_attn_bias(
    attn_arg: AttnArg,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a small ``[q_end-q_start, k_end-k_start]`` attn-bias tensor
    from *attn_arg* without materializing the full ``[sq, sk]`` mask.

    Positions that are masked get ``-inf``; unmasked positions get ``0``.
    """
    bq_len = q_end - q_start
    bk_len = k_end - k_start
    mask = torch.zeros((bq_len, bk_len), dtype=torch.bool, device=device)

    for q_range, k_range, attn_type_idx in zip(
        attn_arg.q_ranges, attn_arg.k_ranges, attn_arg.attn_type_map
    ):
        qr_s, qr_e = q_range.start, q_range.end
        kr_s, kr_e = k_range.start, k_range.end

        iq_s = max(qr_s, q_start)
        iq_e = min(qr_e, q_end)
        ik_s = max(kr_s, k_start)
        ik_e = min(kr_e, k_end)

        if iq_s >= iq_e or ik_s >= ik_e:
            continue

        slice_mask = make_slice_mask_from_ffa_attn_type(
            seqlen_q=qr_e - qr_s,
            seqlen_k=kr_e - kr_s,
            attn_type_idx=attn_type_idx,
            device=device,
        )

        rq_s = iq_s - qr_s
        rq_e = iq_e - qr_s
        rk_s = ik_s - kr_s
        rk_e = ik_e - kr_s

        bq_s = iq_s - q_start
        bq_e = iq_e - q_start
        bk_s = ik_s - k_start
        bk_e = ik_e - k_start

        mask[bq_s:bq_e, bk_s:bk_e] = slice_mask[rq_s:rq_e, rk_s:rk_e]

    bias = torch.zeros((bq_len, bk_len), dtype=dtype, device=device)
    bias.masked_fill_(mask.logical_not(), float("-inf"))
    return bias


# ------------------   sdpa_online fwd   ------------------ #


@torch.no_grad
def sdpa_online_fwd_calc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_arg: AttnArg,
    softmax_scale: float,
    return_max_logits: bool = False,
    block_q: int = BLOCK_Q,
    block_k: int = BLOCK_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Block-wise online softmax forward.

    Works directly in ``[sq/sk, nhq, hd]`` layout (no batch dimension).

    Shapes:
        q:  [sq, nhq, hd]
        k:  [sk, nhq, hd]  (already GQA-expanded)
        v:  [sk, nhq, hd]
    """
    sq, nhq, hd = q.shape
    sk = k.size(0)

    lse_dtype = max_fp_dtype(q.dtype, torch.float32)

    out = torch.zeros(sq, nhq, hd, dtype=q.dtype, device=q.device)
    lse = torch.full(
        (sq, nhq), fill_value=float("-inf"), dtype=lse_dtype, device=q.device
    )

    if return_max_logits:
        max_logits = torch.full(
            (nhq,), fill_value=float("-inf"), dtype=lse_dtype, device=q.device
        )
    else:
        max_logits = None

    # transpose to [nhq, sq/sk, hd] for batched matmul
    q_t = q.transpose(0, 1)  # [nhq, sq, hd]
    k_t = k.transpose(0, 1)  # [nhq, sk, hd]
    v_t = v.transpose(0, 1)  # [nhq, sk, hd]

    for q_start in range(0, sq, block_q):
        q_end = min(q_start + block_q, sq)
        bq = q_t[:, q_start:q_end]  # [nhq, bq_len, hd]
        bout = out[q_start:q_end]  # [bq_len, nhq, hd]
        blse = lse[q_start:q_end]  # [bq_len, nhq]

        for k_start in range(0, sk, block_k):
            k_end = min(k_start + block_k, sk)
            bk = k_t[:, k_start:k_end]  # [nhq, bk_len, hd]
            bv = v_t[:, k_start:k_end]  # [nhq, bk_len, hd]

            bbias = _make_block_attn_bias(
                attn_arg,
                q_start,
                q_end,
                k_start,
                k_end,
                dtype=q.dtype,
                device=q.device,
            )  # [bq_len, bk_len]

            # S = Q @ K^T * scale + bias   shape: [nhq, bq_len, bk_len]
            bs = to_higher_fp_dtype(
                bq @ bk.transpose(-2, -1) * softmax_scale,
                lowest_precision=torch.float32,
            )
            bs += bbias  # broadcasts [bq_len, bk_len] -> [nhq, bq_len, bk_len]

            if return_max_logits:
                block_max = bs.view(nhq, -1).max(dim=-1).values
                torch.maximum(max_logits, block_max, out=max_logits)

            blse_ = bs.logsumexp(dim=-1)  # [nhq, bq_len]
            bp = safe_softmax(bs, blse_.unsqueeze(-1)).to(
                q.dtype
            )  # [nhq, bq_len, bk_len]
            bout_ = (
                (bp @ bv).transpose(0, 1).contiguous()
            )  # [nhq, bq_len, hd] -> [bq_len, nhq, hd]
            blse_ = blse_.transpose(0, 1).contiguous()  # [nhq, bq_len] -> [bq_len, nhq]

            # online merge via triton kernel (handles -inf correctly)
            correct_attn_out_lse(
                out1=bout,
                lse1=blse,
                out2=bout_,
                lse2=blse_,
                inplace=True,
            )

    return out, lse, max_logits


@torch.no_grad()
def sdpa_online_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    attn_arg: AttnArg,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sink_layout: AttnSinkLayout = "sh",
    return_max_logits: bool = False,
) -> tuple[torch.Tensor, AttnForwardMeta]:
    """Online (block-wise) SDPA forward function.

    Same interface as ``sdpa_fwd`` but uses tiled online softmax to avoid
    materializing the full ``[sq, sk]`` attention weight matrix.

    Input/output layout: ``[num_tokens, num_heads, head_dim]`` (3-D only).
    """
    assert softcap == 0.0, "non-zero softcap is not supported by now"
    assert len(q.shape) == 3, "sdpa_online_fwd only supports 3-D [t, nh, hd] layout"

    softmax_scale = q.size(-1) ** (-0.5) if softmax_scale is None else softmax_scale

    # GQA expansion
    nhq, nhk = q.size(1), k.size(1)
    rep_times = nhq // nhk
    if rep_times > 1:
        k = k.repeat_interleave(rep_times, dim=1)
        v = v.repeat_interleave(rep_times, dim=1)

    out, lse, max_logits = sdpa_online_fwd_calc(
        q, k, v, attn_arg, softmax_scale, return_max_logits
    )

    if sink is not None:
        out, lse = correct_attn_out_lse_with_sink(
            out=out,
            lse=lse,
            sink=sink,
            sink_layout=sink_layout,
            inplace=True,
        )

    return out, AttnForwardMeta(lse=lse, max_logits=max_logits)


# ------------------   sdpa_online bwd   ------------------ #
# The backward pass is identical to the offline SDPA backward because it
# already recomputes P block-by-block from the saved LSE.
# Backward is less memory-sensitive so we still materialize the full mask.

from .sdpa import _sdpa_bwd as _sdpa_online_bwd  # noqa: E402


@torch.no_grad()
def sdpa_online_bwd(
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
    """Online SDPA backward function.

    The backward computation is identical to the offline SDPA backward
    (``sdpa_bwd``) since it always recomputes P block-by-block from the
    saved LSE.  This function exists for API symmetry with
    ``sdpa_online_fwd``.

    Args / Returns: same as ``sdpa_bwd``.
    """
    assert softcap == 0.0, "non-zero softcap is not supported by now"
    assert len(q.shape) == 3, "sdpa_online_bwd only supports 3-D [t, nh, hd] layout"

    if sink is not None:
        dsink = sink_bwd(
            sink=sink,
            lse=lse,
            o=o,
            do=do,
            sink_layout=sink_layout,
        )
    else:
        dsink = None

    q, k, v, o, do, lse = sdpa_bwd_qkvodo_lse_rearrange(q, k, v, o, do, lse)

    attn_mask = make_attn_mask_from_ffa_args(
        q_ranges=attn_arg.q_ranges_bwd,
        k_ranges=attn_arg.k_ranges_bwd,
        attn_type_map=attn_arg.attn_type_map_bwd,
        total_seqlen_q=q.shape[-2],
        total_seqlen_k=k.shape[-2],
        device=torch.cuda.current_device(),
    )

    dq, dk, dv = _sdpa_online_bwd(
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

    dq, dk, dv = sdpa_bwd_dqdkdv_rearrange(dq, dk, dv)

    return dq, dk, dv, dsink
