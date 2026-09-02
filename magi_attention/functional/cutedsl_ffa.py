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
forward output and the backward input.

Limitations (guarded by asserts and by flag filtering at the dist
level):

- sink is not supported (CUTEDSL kernel requires per-head scalar bf16 sink,
  while the dist contract is ``[n_sink, nhq]`` fp32);
- ``deterministic=True`` with ranges is rejected by the kernel
  (``NotImplementedError``);
- head_dim > 128 is not supported on the kernel's fast path used here.
"""

from __future__ import annotations

import weakref

import torch

from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.meta.collection.calc_meta import AttnArg

__all__ = ["cutedsl_fwd", "cutedsl_bwd"]

# Coverage declarations for the bwd kernel, derived exactly on the host.
# The two sides consume different contracts (do not infer one from the other):
# - declared_q_full_coverage only skips dQ hole zeroing -> union coverage
#   suffices, overlapping q rows are fine (is_full_coverage merges first).
# - declared_k_full_coverage additionally feeds k_ranges_sorted_disjoint in
#   the kernel host (scheduler grid bound Sum(len) <= total_k), so it may only
#   be set for a sorted partition -> is_cu_seqlens, never is_full_coverage.
#
# The natural home for these two bools is AttnArg.__post_init__, next to the
# bwd ranges they are derived from; until they move there, this cache stands in.
#
# Cached per AttnArg: the runtime manager reuses args across steps and the
# derivation sorts all ranges. Keyed by id() because AttnArg is unhashable;
# the finalizer evicts the entry before the id can be reused.
_bwd_coverage_cache: dict[int, tuple[int, int, bool, bool]] = {}


def _bwd_full_coverage(
    attn_arg: AttnArg, total_q: int, total_k: int
) -> tuple[bool, bool]:
    key = id(attn_arg)
    hit = _bwd_coverage_cache.get(key)
    if hit is not None and hit[0] == total_q and hit[1] == total_k:
        return hit[2], hit[3]
    if hit is None:
        weakref.finalize(attn_arg, _bwd_coverage_cache.pop, key, None)
    q_cov = attn_arg.q_ranges_bwd.is_full_coverage(total_q)
    k_cov = attn_arg.k_ranges_bwd.is_cu_seqlens(total_k)
    _bwd_coverage_cache[key] = (total_q, total_k, q_cov, k_cov)
    return q_cov, k_cov


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
        mask_types=ffa_args["attn_type_map"],
    )

    return out, lse


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

    q_full_coverage, k_full_coverage = _bwd_full_coverage(
        attn_arg, q.shape[0], k.shape[0]
    )

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
        lse=lse,
        dout=do,
        q_ranges=ffa_args["q_ranges"],
        k_ranges=ffa_args["k_ranges"],
        max_seqlen_q=attn_arg.q_ranges.max_seqlen,
        max_seqlen_k=attn_arg.k_ranges.max_seqlen,
        softmax_scale=softmax_scale,
        softcap=softcap,
        deterministic=False,  # the kernel raises NotImplementedError for ranges + deterministic
        declared_q_full_coverage=q_full_coverage,
        declared_k_full_coverage=k_full_coverage,
        disable_fwd_atomic_reduction=attn_arg.disable_fwd_atomic_reduction,
        disable_bwd_dkv_atomic_reduction=attn_arg.disable_bwd_dkv_atomic_reduction,
        range_merge=False,
        mask_types=ffa_args["attn_type_map"],
    )
    # The dist bwd hp-reduce path (``bwd_hp_reduce`` flag) expects partial
    # dq/dk/dv in fp32; the kernel returns input dtype, so cast up here.
    return dq.float(), dk.float(), dv.float(), None
