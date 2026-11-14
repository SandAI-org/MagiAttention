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

"""Thin dist-attn wrappers for the CuteDSL FFA kernel.

The dist runtime's ``DistAttnFunc`` manually interleaves per-stage partial
attention and gradient merges, so it cannot reuse
:func:`magi_attention.kernel.cutedsl.flex_flash_attn_func` (an autograd
Function). These two functions call the raw
``_flex_flash_attn_fwd`` / ``_flex_flash_attn_bwd`` kernels directly. Tensor
contracts: q/k ranges and mask type map as produced by
``AttnArg.to_ffa_args()``; ``lse`` is fp32 ``(total_q, nhq)`` on both the
forward output and the backward input; ``sink`` is the dist contract's
``[n_sink, nhq]`` fp32 with layout "sh", and ``cutedsl_bwd`` returns the fp32
partial ``dsink`` of the same shape (``None`` without ``sink``).

Limitation: deterministic mode is rejected by ``dist_attn_runtime_mgr`` for
the CUTEDSL backend, because ``validate_range_feature_support`` rejects
deterministic backward with ranges.
"""

import torch

from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.meta.collection.calc_meta import AttnArg

__all__ = ["cutedsl_fwd", "cutedsl_bwd"]


def cutedsl_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_arg: AttnArg,
    softmax_scale: float | None,
    softcap: float,
    sink: torch.Tensor | None = None,
    sm_margin: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward wrapper: returns (out, lse) with lse in the dist contract's (sq, nhq) layout."""

    ffa_args = attn_arg.to_ffa_args(is_bwd=False)
    if not ffa_args:
        raise RuntimeError("cutedsl_fwd called with skip_attn_fwd=True")

    # TODO: Pre-initializing uncovered rows is a workaround; initialize their
    # output and LSE in the forward postprocess instead to avoid extra memory traffic.
    direct_store = attn_arg.disable_fwd_atomic_reduction
    out_buffer = lse_buffer = None
    if direct_store and attn_arg.bwd_args_q_coverage_end != q.shape[0]:
        # Direct stores leave uncovered Q rows untouched. Preserve the atomic
        # path's zero output and sink-only (or empty) normalization there.
        out_buffer = torch.zeros(
            (q.shape[0], q.shape[1], v.shape[-1]),
            dtype=torch.float32,
            device=q.device,
        )
        if sink is None:
            lse_buffer = torch.full(
                q.shape[:2], float("-inf"), dtype=torch.float32, device=q.device
            )
        else:
            sink_lse = sink[0] if sink.shape[0] == 1 else torch.logsumexp(sink, dim=0)
            lse_buffer = (
                sink_lse.float()
                .expand(q.shape[0], -1)
                .clone(memory_format=torch.contiguous_format)
            )

    out, lse = _flex_flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        q_ranges=ffa_args["q_ranges"],
        k_ranges=ffa_args["k_ranges"],
        max_seqlen_q=attn_arg.q_ranges.max_seqlen,
        max_seqlen_k=attn_arg.k_ranges.max_seqlen,
        softmax_scale=softmax_scale,
        softcap=softcap if softcap and softcap > 0 else None,
        sink=sink,
        sink_layout="sh",
        # Stage merges require FP32 partial outputs on both store paths.
        disable_fwd_atomic_reduction=direct_store,
        out_dtype=torch.float32,
        out=out_buffer,
        lse=lse_buffer,
        # range_merge: the dist layer's pre-merged relation IR is already the
        # input to AttnArg (merge happens in calc_meta), so the kernel sees the
        # plain per-range problem — no in-kernel merge needed.
        range_merge=False,
        mask_types=ffa_args["attn_type_map"],
        sm_margin=sm_margin,
    )

    return out, lse


def cutedsl_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_arg: AttnArg,
    softmax_scale: float | None,
    softcap: float,
    dq_acc: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    sm_margin: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Partial dq/dk/dv in fp32 for one dist-attn stage, plus dsink if ``sink`` is given.

    With ``dq_acc`` (fp32, shape of ``q``) the kernel adds this stage's dq onto
    it in place and the returned dq is ``dq_acc`` itself.
    """
    ffa_args = attn_arg.to_ffa_args(is_bwd=True)
    if not ffa_args:
        raise RuntimeError("cutedsl_bwd called with skip_attn_bwd=True")

    # Range flags of the bwd range tensors, computed by AttnArg on the host.
    q_full_coverage = attn_arg.bwd_args_q_coverage_end == q.shape[0]
    k_full_coverage = attn_arg.bwd_args_k_partition_end == k.shape[0]

    # If fwd went down the atomic path it returns fp32 out; the bwd kernel
    # consumes the input dtype, so cast back.
    if o.dtype != q.dtype:
        o = o.to(q.dtype)
        do = do.to(q.dtype)

    dq, dk, dv, dsink = _flex_flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=o,
        lse=lse,
        dout=do,
        dq=dq_acc,
        sink=sink,
        sink_layout="sh",
        q_ranges=ffa_args["q_ranges"],
        k_ranges=ffa_args["k_ranges"],
        max_seqlen_q=attn_arg.q_ranges.max_seqlen,
        max_seqlen_k=attn_arg.k_ranges.max_seqlen,
        softmax_scale=softmax_scale,
        softcap=softcap,
        deterministic=False,  # the kernel raises NotImplementedError for ranges + deterministic
        declared_q_full_coverage=q_full_coverage,
        declared_k_full_coverage=k_full_coverage,
        disable_fwd_atomic_reduction=attn_arg.disable_fwd_atomic_reduction
        and attn_arg.bwd_args_q_sorted,
        disable_bwd_dkv_atomic_reduction=attn_arg.disable_bwd_dkv_atomic_reduction,
        range_merge=False,
        mask_types=ffa_args["attn_type_map"],
        # Partial dq/dk/dv feed the fp32 dist reduce, so they must not round
        # through the input dtype.
        dq_type=torch.float32,
        dk_type=torch.float32,
        dv_type=torch.float32,
        sm_margin=sm_margin,
    )
    return dq, dk, dv, dsink
