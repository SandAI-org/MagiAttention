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

"""Packed-layout Triton DSA attention kernel.

Public inputs are already packed by the caller (DSAClusterIndex in DistDSA):
    q          [C * nh_kv, q_tile_padded, D]
    k / v      [nh_kv * skv, 1, D]
    index_map  [1, C * nh_kv, topk]

Returns packed out [C * nh_kv, q_tile_padded, D] and log2 LSE.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ATOMIC_DTYPE = os.environ.get("DSA_TRITON_BWD_ATOMIC_DTYPE", "bf16").strip()
assert _ATOMIC_DTYPE in ("bf16", "fp32"), (
    "DSA_TRITON_BWD_ATOMIC_DTYPE must be bf16 or fp32, " f"got {_ATOMIC_DTYPE}"
)
_MAX_TRITON_LINEAR_ELEMENTS = 2**31 - 1


def _check_triton_indexable(*named_tensors: tuple[str, torch.Tensor]) -> None:
    too_large = [
        (name, tuple(tensor.shape), tensor.numel())
        for name, tensor in named_tensors
        if tensor.numel() > _MAX_TRITON_LINEAR_ELEMENTS
    ]
    if not too_large:
        return
    detail = ", ".join(
        f"{name}: shape={shape}, numel={numel}" for name, shape, numel in too_large
    )
    raise ValueError(
        "DSA Triton packed attention tensor is too large for the current "
        "32-bit-offset kernel path. "
        f"{detail}. Use context parallelism, increase DSA cluster size, or use "
        "a backend that supports this shape."
    )


def _bwd_autotune_configs():
    return [
        triton.Config({"BK": bk}, num_warps=num_warps, num_stages=num_stages)
        for bk in (32, 64, 128, 256)
        for num_warps in (4, 8)
        for num_stages in (1, 2, 3)
    ]


# ---------------------------------------------------------------------------
# Packed kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_gqa_packed_fwd_kernel(
    q,
    k,
    v,
    index_map,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_is,
    stride_it,
    stride_os,
    stride_oh,
    stride_od,
    stride_ls,
    stride_lh,
    SKV: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    H: tl.constexpr,
    BK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_d = offs_d < D

    q_ptrs = (
        q
        + pid_s * stride_qs
        + offs_h[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_blk = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0)

    max_log = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, DP], dtype=tl.float32)
    log_scale: tl.constexpr = sm_scale * 1.44269504

    for ck in range(tl.cdiv(K, BK)):
        t_offs = ck * BK + offs_t
        t_mask = t_offs < K
        idx = tl.load(
            index_map + pid_s * stride_is + t_offs * stride_it, mask=t_mask, other=-1
        )
        valid = t_mask & (idx >= 0) & (idx < SKV)

        k_blk = tl.load(
            k + idx[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=valid[:, None] & mask_d[None, :],
            other=0.0,
        )
        qk = tl.dot(q_blk, tl.trans(k_blk), out_dtype=tl.float32) * log_scale
        qk = tl.where(valid[None, :], qk, float("-inf"))

        cand_max = tl.max(qk, axis=1)
        new_max = tl.maximum(max_log, cand_max)
        exp_qk = tl.math.exp2(qk - new_max[:, None]).to(tl.float32)
        exp_qk = tl.where(valid[None, :], exp_qk, 0.0)
        alpha = tl.math.exp2(max_log - new_max).to(tl.float32)
        alpha = tl.where(max_log == -float("inf"), 0.0, alpha)
        sum_exp = sum_exp * alpha + tl.sum(exp_qk, axis=1)
        acc = acc * alpha[:, None]

        v_blk = tl.load(
            v + idx[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=valid[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc = tl.dot(exp_qk.to(v_blk.dtype), v_blk, acc, out_dtype=tl.float32)
        max_log = new_max

    out_vals = acc / sum_exp[:, None]
    out_ptrs = (
        output
        + pid_s * stride_os
        + offs_h[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_vals.to(output.dtype.element_ty), mask=mask_d[None, :])
    tl.store(
        lse + pid_s * stride_ls + offs_h * stride_lh, max_log + tl.math.log2(sum_exp)
    )


def _sparse_gqa_packed_bwd_kernel_impl(
    q,
    k,
    v,
    index_map,
    do,
    delta,
    lse,
    dq,
    dk,
    dv,
    sm_scale: tl.constexpr,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_is,
    stride_it,
    stride_dos,
    stride_doh,
    stride_dod,
    stride_ds,
    stride_dh,
    stride_ls,
    stride_lh,
    stride_dqs,
    stride_dqh,
    stride_dqd,
    stride_dks,
    stride_dkh,
    stride_dkd,
    stride_dvs,
    stride_dvh,
    stride_dvd,
    SKV: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    H: tl.constexpr,
    BK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_d = offs_d < D

    q_blk = tl.load(
        q
        + pid_s * stride_qs
        + offs_h[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_d[None, :],
        other=0.0,
    )
    do_blk = tl.load(
        do
        + pid_s * stride_dos
        + offs_h[:, None] * stride_doh
        + offs_d[None, :] * stride_dod,
        mask=mask_d[None, :],
        other=0.0,
    )
    lse_row = tl.load(lse + pid_s * stride_ls + offs_h * stride_lh).to(tl.float32)
    delta_row = tl.load(delta + pid_s * stride_ds + offs_h * stride_dh).to(tl.float32)

    dq_acc = tl.zeros([H, DP], dtype=tl.float32)
    log_scale: tl.constexpr = sm_scale * 1.44269504
    ln2: tl.constexpr = 0.6931471805599453

    for ck in range(tl.cdiv(K, BK)):
        t_offs = ck * BK + offs_t
        t_mask = t_offs < K
        idx = tl.load(
            index_map + pid_s * stride_is + t_offs * stride_it, mask=t_mask, other=-1
        )
        valid = t_mask & (idx >= 0) & (idx < SKV)

        k_blk = tl.load(
            k + idx[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=valid[:, None] & mask_d[None, :],
            other=0.0,
        )
        v_blk = tl.load(
            v + idx[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=valid[:, None] & mask_d[None, :],
            other=0.0,
        )

        x = tl.dot(q_blk, tl.trans(k_blk), out_dtype=tl.float32) * log_scale
        x = tl.where(valid[None, :], x, float("-inf"))
        p = tl.where(
            valid[None, :], tl.math.exp2(x - lse_row[:, None]).to(tl.float32), 0.0
        )

        dp = tl.dot(do_blk, tl.trans(v_blk), out_dtype=tl.float32)
        dx = ln2 * p * (dp - delta_row[:, None])
        dx = tl.where(valid[None, :], dx, 0.0)
        dx_in = dx.to(q_blk.dtype)
        p_in = p.to(do_blk.dtype)

        dq_acc += tl.dot(dx_in, k_blk, out_dtype=tl.float32) * log_scale
        dk_blk = tl.dot(tl.trans(dx_in), q_blk, out_dtype=tl.float32) * log_scale
        dv_blk = tl.dot(tl.trans(p_in), do_blk, out_dtype=tl.float32)

        dk_ptrs = dk + idx[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dv + idx[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        tl.atomic_add(dk_ptrs, dk_blk, mask=valid[:, None] & mask_d[None, :])
        tl.atomic_add(dv_ptrs, dv_blk, mask=valid[:, None] & mask_d[None, :])

    dq_ptrs = (
        dq
        + pid_s * stride_dqs
        + offs_h[:, None] * stride_dqh
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq_acc.to(dq.dtype.element_ty), mask=mask_d[None, :])


_sparse_gqa_packed_bwd_kernel = triton.jit(_sparse_gqa_packed_bwd_kernel_impl)
_sparse_gqa_packed_bwd_kernel_autotuned = triton.autotune(
    configs=_bwd_autotune_configs(),
    key=["K", "D", "H"],
    reset_to_zero=["dk", "dv"],
)(triton.jit(_sparse_gqa_packed_bwd_kernel_impl))


def _sparse_gqa_packed_fwd(q, k, v, index_map, sm_scale, BK):
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert index_map.is_contiguous()
    assert index_map.shape[0] == 1 and k.shape[1] == v.shape[1] == 1
    _check_triton_indexable(
        ("q", q),
        ("k", k),
        ("v", v),
        ("index_map", index_map),
    )
    SQ, H, D = q.shape
    SKV, _, D_k = k.shape
    _, _, D_v = v.shape
    _, SQ_i, K = index_map.shape
    assert SQ_i == SQ and D_k == D_v == D

    DP = triton.next_power_of_2(D)
    output = torch.empty_like(q)
    lse = torch.empty((SQ, H), device=q.device, dtype=torch.float32)
    _sparse_gqa_packed_fwd_kernel[(SQ,)](
        q,
        k,
        v,
        index_map,
        sm_scale,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        index_map.stride(1),
        index_map.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        SKV,
        K,
        D,
        DP,
        H,
        BK,
    )
    return output, lse


def _sparse_gqa_packed_bwd(
    q, k, v, index_map, do, delta, lse, sm_scale, BK, autotune: bool
):
    assert all(t.is_contiguous() for t in [q, k, v, index_map, do, delta, lse])
    SQ, H, D = q.shape
    SKV, _, _ = k.shape
    _, SQ_i, K = index_map.shape
    assert SQ_i == SQ and do.shape == q.shape and delta.shape == lse.shape == (SQ, H)

    DP = triton.next_power_of_2(D)
    dq = torch.empty_like(q)
    atomic_dtype = torch.bfloat16 if _ATOMIC_DTYPE == "bf16" else torch.float32
    dk = torch.zeros_like(k, dtype=atomic_dtype)
    dv = torch.zeros_like(v, dtype=atomic_dtype)
    kernel_args = (
        q,
        k,
        v,
        index_map,
        do,
        delta,
        lse,
        dq,
        dk,
        dv,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        index_map.stride(1),
        index_map.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        lse.stride(0),
        lse.stride(1),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        SKV,
        K,
        D,
        DP,
        H,
    )
    if autotune:
        _sparse_gqa_packed_bwd_kernel_autotuned[(SQ,)](*kernel_args)
    else:
        _sparse_gqa_packed_bwd_kernel[(SQ,)](*kernel_args, BK)
    return dq, dk.to(k.dtype), dv.to(v.dtype)


class SparseGQAPackedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, index_map, sm_scale, BK_FWD, BK_BWD, BWD_AUTOTUNE):
        out, lse = _sparse_gqa_packed_fwd(q, k, v, index_map, sm_scale, BK_FWD)
        ctx.save_for_backward(q, k, v, index_map, out, lse)
        ctx.sm_scale = sm_scale
        ctx.BK_BWD = BK_BWD
        ctx.BWD_AUTOTUNE = BWD_AUTOTUNE
        return out, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, index_map, out, lse = ctx.saved_tensors
        delta = (out.float() * do.float()).sum(dim=-1)
        if dlse is not None:
            delta = delta - dlse.float() * 1.4426950408889634
        dq, dk, dv = _sparse_gqa_packed_bwd(
            q,
            k,
            v,
            index_map,
            do.contiguous(),
            delta.contiguous(),
            lse,
            ctx.sm_scale,
            ctx.BK_BWD,
            ctx.BWD_AUTOTUNE,
        )
        return dq, dk, dv, None, None, None, None, None


def sparse_gqa_packed_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    sm_scale: float | None = None,
    BK_FWD: int = 64,
    BK_BWD: int = 64,
    BWD_AUTOTUNE: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed-layout DSA Triton attention.  Returns log2 LSE."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return SparseGQAPackedFn.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        index_map.contiguous(),
        sm_scale,
        BK_FWD,
        BK_BWD,
        BWD_AUTOTUNE,
    )
