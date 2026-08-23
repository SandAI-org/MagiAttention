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
``_flex_flash_attn_fwd`` / ``_flex_flash_attn_bwd`` kernels directly and adapt
the LSE layout; all other contracts (q/k ranges tensors, mask type map) match
``AttnArg.to_ffa_args()`` verbatim.

Limitations (guarded by asserts and by flag filtering at the dist
level):

- sink is not supported (CUTEDSL kernel requires per-head scalar bf16 sink,
  while the dist contract is ``[n_sink, nhq]`` fp32);
- ``deterministic=True`` with ranges is rejected by the kernel
  (``NotImplementedError``);
- head_dim > 128 is not supported on the kernel's fast path used here.
"""

from __future__ import annotations

import torch

from magi_attention.kernel.cutedsl.ffa_utils import MaskMode
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.meta.collection.calc_meta import AttnArg

__all__ = ["cutedsl_fwd", "cutedsl_bwd"]


def _per_range_mask_args(attn_type_map: torch.Tensor) -> dict:
    """Translate an attn_type_map tensor to the kernel's per-range mask mode."""
    return {
        "mask_mode": MaskMode.PER_RANGE,
        "mask_types_tensor": attn_type_map,
    }


def cutedsl_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    attn_arg: AttnArg,
    softmax_scale: float | None,
    softcap: float,
    sink_layout: str = "sh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward wrapper: returns (out, lse) with lse in the dist contract's (sq, nhq) layout."""
    assert sink is None, "CUTEDSL backend does not support attention sink"
    assert sink_layout == "sh", f"unsupported sink layout: {sink_layout}"

    ffa_args = attn_arg.to_ffa_args(is_bwd=False)
    if not ffa_args:
        raise RuntimeError("cutedsl_fwd called with skip_attn_fwd=True")

    out, lse_nhtq = _flex_flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        q_ranges=ffa_args["q_ranges"],
        k_ranges=ffa_args["k_ranges"],
        max_seqlen_q=attn_arg.q_ranges.max_seqlen,
        max_seqlen_k=attn_arg.k_ranges.max_seqlen,
        softmax_scale=softmax_scale,
        softcap=softcap if softcap and softcap > 0 else None,
        # Force fp32 partial out for the dist multi-stage merge. The dist
        # overlap path rescales partial (out, lse) pairs in fp32
        # (correct_attn_out_lse); a bf16/fp16 partial underflows that merge.
        # The atomic path defaults O to the input dtype, so ask for fp32 here.
        disable_fwd_atomic_reduction=False,
        out_dtype=torch.float32,
        # range_merge: the dist layer's pre-merged relation IR is already the
        # input to AttnArg (merge happens in calc_meta), so the kernel sees the
        # plain per-range problem — no in-kernel merge needed.
        range_merge=False,
        **_per_range_mask_args(ffa_args["attn_type_map"]),
    )

    # Kernel returns lse as (nh, tq); dist contract is (sq, nhq).
    return out, lse_nhtq.mT.contiguous()


def cutedsl_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_arg: AttnArg,
    softmax_scale: float | None,
    softcap: float,
    sink_layout: str = "sh",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Backward wrapper: partial_dsink is always None since sink is unsupported."""
    assert sink is None, "CUTEDSL backend does not support attention sink"
    assert sink_layout == "sh", f"unsupported sink layout: {sink_layout}"

    ffa_args = attn_arg.to_ffa_args(is_bwd=True)
    if not ffa_args:
        raise RuntimeError("cutedsl_bwd called with skip_attn_bwd=True")

    # If fwd went down the atomic path it returns fp32 out; the bwd kernel
    # consumes the input dtype, so cast back.
    if o.dtype != q.dtype:
        o = o.to(q.dtype)
        do = do.to(q.dtype)

    dq, dk, dv = _flex_flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=o,
        # Dist contract stores lse as (sq, nhq); kernel expects (nh, tq).
        lse=lse.mT.contiguous(),
        dout=do,
        q_ranges=ffa_args["q_ranges"],
        k_ranges=ffa_args["k_ranges"],
        max_seqlen_q=attn_arg.q_ranges.max_seqlen,
        max_seqlen_k=attn_arg.k_ranges.max_seqlen,
        # Host-side ranges are available here, so the exact scheduler grid
        # totals cost nothing. Rows must match ffa_args["k_ranges"],
        # which comes from the bwd-transformed k_ranges_bwd.
        k_tile_hints=(
            sum((r.seqlen + 127) // 128 for r in attn_arg.k_ranges_bwd),
            sum(((r.seqlen + 127) // 128 + 1) // 2 for r in attn_arg.k_ranges_bwd),
        ),
        softmax_scale=softmax_scale,
        softcap=softcap,
        deterministic=False,  # the kernel raises NotImplementedError for ranges + deterministic
        disable_fwd_atomic_reduction=attn_arg.disable_fwd_atomic_reduction,
        disable_bwd_dkv_atomic_reduction=attn_arg.disable_bwd_dkv_atomic_reduction,
        range_merge=False,
        **_per_range_mask_args(ffa_args["attn_type_map"]),
    )
    # The dist bwd hp-reduce path (``bwd_hp_reduce`` flag) expects partial
    # dq/dk/dv in fp32; the kernel returns input dtype, so cast up here.
    return dq.float(), dk.float(), dv.float(), None
