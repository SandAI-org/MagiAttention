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
"""DSA-specialized Triton backward for index-map sparse attention.

Adapted from Ethan's MagiAttention DSA v4 backward.  It recomputes the
probabilities from a saved LSE (natural log) and streams K/V rows by
``index_map``, so any forward that produces ``out`` and ``lse`` (e.g. FFA
IndexSparse) can be paired with it.

Layout:
    q / dout   [sq, nhq, hd]
    k / v      [skv, nhkv, hd]
    index_map  [nhkv, sq, topk]   row offsets local to each KV head
    lse_saved  [nhkv, sq, nhq // nhkv]
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ATOMIC_DTYPE = os.environ.get(
    "DSA_FFA_TRITON_BWD_ATOMIC_DTYPE",
    os.environ.get("DSA_BWD_V4_DTYPE", "bf16"),
).strip()
assert _ATOMIC_DTYPE in ("bf16", "fp32"), (
    "DSA_FFA_TRITON_BWD_ATOMIC_DTYPE must be bf16 or fp32, " f"got {_ATOMIC_DTYPE}"
)


def _autotune_configs():
    configs = []
    for block_kv in (64, 128, 256):
        for num_warps in (4, 8):
            for num_stages in (1, 2, 3):
                configs.append(
                    triton.Config(
                        {"BLOCK_KV": block_kv},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=_autotune_configs(),
    key=["GROUP_SIZE", "GROUP_TILE", "HD"],
    reset_to_zero=["DK_BUF_ptr", "DV_BUF_ptr"],
)
@triton.jit
def _ffa_triton_bwd_kernel_v4(
    Q_ptr,
    K_ptr,
    V_ptr,
    IDX_ptr,
    DO_ptr,
    D_PTR,
    LSE_ptr,
    DQ_ptr,
    DK_BUF_ptr,
    DV_BUF_ptr,
    sq,
    skv,
    topk,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_i_h,
    stride_i_s,
    stride_i_t,
    stride_D_s,
    stride_D_h,
    stride_lse_h,
    stride_lse_s,
    stride_lse_g,
    stride_dkb_s,
    stride_dkb_h,
    stride_dkb_d,
    stride_dvb_s,
    stride_dvb_h,
    stride_dvb_d,
    softmax_scale,
    GROUP_SIZE: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    HD: tl.constexpr,
    USE_BF16_ATOMIC: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_gt = tl.program_id(2)

    g_start = pid_gt * GROUP_TILE
    h_start = pid_h * GROUP_SIZE + g_start
    offs_g = tl.arange(0, GROUP_TILE)
    offs_d = tl.arange(0, HD)
    offs_t = tl.arange(0, BLOCK_KV)
    g_mask = (g_start + offs_g) < GROUP_SIZE

    q_ptrs = (
        Q_ptr
        + pid_s * stride_q_s
        + (h_start + offs_g[:, None]) * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    do_ptrs = (
        DO_ptr
        + pid_s * stride_q_s
        + (h_start + offs_g[:, None]) * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=g_mask[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=g_mask[:, None], other=0.0)

    d_ptrs = D_PTR + pid_s * stride_D_s + (h_start + offs_g) * stride_D_h
    # D is sum(dO * O) - dLSE.  Subtracting dLSE here makes the usual
    # p * (dP - D) formula include the lse-output gradient.
    D = tl.load(d_ptrs, mask=g_mask, other=0.0)

    lse_ptrs = (
        LSE_ptr
        + pid_h * stride_lse_h
        + pid_s * stride_lse_s
        + (g_start + offs_g) * stride_lse_g
    )
    lse = tl.load(lse_ptrs, mask=g_mask, other=0.0)

    dq_acc = tl.zeros([GROUP_TILE, HD], dtype=tl.float32)

    for b in range(tl.cdiv(topk, BLOCK_KV)):
        t_offs = b * BLOCK_KV + offs_t
        t_mask = t_offs < topk
        idx_ptrs = (
            IDX_ptr + pid_h * stride_i_h + pid_s * stride_i_s + t_offs * stride_i_t
        )
        idx = tl.load(idx_ptrs, mask=t_mask, other=0)

        k_ptrs = (
            K_ptr
            + idx[:, None] * stride_k_s
            + pid_h * stride_k_h
            + offs_d[None, :] * stride_k_d
        )
        v_ptrs = (
            V_ptr
            + idx[:, None] * stride_v_s
            + pid_h * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=t_mask[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        scores = tl.where(t_mask[None, :], scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        p = tl.where(t_mask[None, :], p, 0.0)

        p_cast = p.to(do.dtype)
        dv_block = tl.dot(tl.trans(p_cast), do)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp.to(tl.float32) - D[:, None])
        ds_cast = ds.to(q.dtype)

        dq_acc += tl.dot(ds_cast, k).to(tl.float32) * softmax_scale
        dk_block = tl.dot(tl.trans(ds_cast), q) * softmax_scale

        dkb_ptrs = (
            DK_BUF_ptr
            + idx[:, None] * stride_dkb_s
            + pid_h * stride_dkb_h
            + offs_d[None, :] * stride_dkb_d
        )
        dvb_ptrs = (
            DV_BUF_ptr
            + idx[:, None] * stride_dvb_s
            + pid_h * stride_dvb_h
            + offs_d[None, :] * stride_dvb_d
        )
        if USE_BF16_ATOMIC:
            tl.atomic_add(dkb_ptrs, dk_block.to(tl.bfloat16), mask=t_mask[:, None])
            tl.atomic_add(dvb_ptrs, dv_block.to(tl.bfloat16), mask=t_mask[:, None])
        else:
            tl.atomic_add(dkb_ptrs, dk_block.to(tl.float32), mask=t_mask[:, None])
            tl.atomic_add(dvb_ptrs, dv_block.to(tl.float32), mask=t_mask[:, None])

    dq_ptrs = (
        DQ_ptr
        + pid_s * stride_q_s
        + (h_start + offs_g[:, None]) * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    tl.store(dq_ptrs, dq_acc.to(q.dtype), mask=g_mask[:, None])


def dsa_sparse_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    out: torch.Tensor,
    lse_saved: torch.Tensor,
    dout: torch.Tensor,
    dlse: torch.Tensor | None,
    softmax_scale: float,
    group_tile: int | None = None,
    use_bf16_atomic: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(dq, dk, dv)``.  ``dlse`` (if given) is folded into ``delta``.

    ``nhq // nhkv`` must be a power of 2 and at least 16.
    """
    sq, nhq, head_dim = q.shape
    skv, nhkv, _ = k.shape
    topk = index_map.shape[-1]
    assert nhq % nhkv == 0
    group_size = nhq // nhkv
    assert (
        v.shape[-1] == head_dim
    ), "FFA Triton backward currently requires head_dim_v == head_dim"
    assert group_size >= 16 and (group_size & (group_size - 1)) == 0
    if group_tile is None:
        group_tile = group_size
    assert (group_tile & (group_tile - 1)) == 0 and 16 <= group_tile <= group_size

    if use_bf16_atomic is None:
        use_bf16_atomic = _ATOMIC_DTYPE == "bf16"
    atomic_dtype = torch.bfloat16 if use_bf16_atomic else torch.float32

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    dout_c = dout.contiguous()
    idx_c = index_map.to(torch.int32).contiguous()
    lse_c = lse_saved.contiguous()
    assert lse_c.shape == (nhkv, sq, group_size)

    delta = (dout_c.to(torch.float32) * out.to(torch.float32)).sum(dim=-1)
    if dlse is not None:
        delta = delta - dlse.to(torch.float32)
    delta = delta.contiguous()

    dq = torch.empty_like(q_c)
    dk_buf = torch.zeros_like(k_c, dtype=atomic_dtype)
    dv_buf = torch.zeros_like(v_c, dtype=atomic_dtype)

    grid = (nhkv, sq, triton.cdiv(group_size, group_tile))
    _ffa_triton_bwd_kernel_v4[grid](
        q_c,
        k_c,
        v_c,
        idx_c,
        dout_c,
        delta,
        lse_c,
        dq,
        dk_buf,
        dv_buf,
        sq,
        skv,
        topk,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        v_c.stride(0),
        v_c.stride(1),
        v_c.stride(2),
        idx_c.stride(0),
        idx_c.stride(1),
        idx_c.stride(2),
        delta.stride(0),
        delta.stride(1),
        lse_c.stride(0),
        lse_c.stride(1),
        lse_c.stride(2),
        dk_buf.stride(0),
        dk_buf.stride(1),
        dk_buf.stride(2),
        dv_buf.stride(0),
        dv_buf.stride(1),
        dv_buf.stride(2),
        softmax_scale=softmax_scale,
        GROUP_SIZE=group_size,
        GROUP_TILE=group_tile,
        HD=head_dim,
        USE_BF16_ATOMIC=use_bf16_atomic,
    )
    return dq, dk_buf.to(k.dtype), dv_buf.to(v.dtype)
