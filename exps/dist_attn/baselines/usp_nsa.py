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

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformer_engine.pytorch.distributed import reduce_scatter_along_first_dim

from exps.dist_attn.baselines.nsa import (
    compute_blockq_p_slc,
    compute_gqa_p_slc,
    compute_p_slc,
)
from exps.dist_attn.baselines.shard import ParallelMode
from exps.dist_attn.baselines.utils_cp import (
    _varlen_all2all_after_attn,
    _varlen_all2all_before_attn,
)
from magi_attention.comm.functional import all_gather_fwd_scatter_bwd
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.functional.flex_flash_attn import (
    _flex_flash_attn_backward,
    _flex_flash_attn_forward,
)
from magi_attention.utils.nvtx import add_nvtx_event, instrument_nvtx


@dataclass
class RunTimeMeta:
    q_ranges_tensor: torch.Tensor
    k_ranges_tensor: torch.Tensor
    attn_type_map_tensor: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


# all-gather with dispatch to contiguous
def gather_with_reorder_before_attn(
    input: torch.Tensor,
    total_indices: torch.Tensor,
    cp_group,
):
    other_shape = input.shape[1:]
    output = all_gather_fwd_scatter_bwd(input, cp_group, dim=0).contiguous()
    if total_indices is not None:
        output = torch.gather(
            output, dim=0, index=total_indices[:, None, None].expand(-1, *other_shape)
        )

    return output


# contiguous to dispatch
def reorder_before_reduce_scatter(
    input: torch.Tensor,
    total_indices: torch.Tensor,
):
    if total_indices is None:
        return input
    other_shape = input.shape[1:]
    output = torch.gather(
        input, dim=0, index=total_indices[:, None, None].expand(-1, *other_shape)
    )

    return output


def generate_topk_ranges_kv(topk_index_kv: list[list[int]], stride, block_size_kv):
    idx_arr = np.array(topk_index_kv).flatten()
    st = idx_arr * stride
    ed = st + block_size_kv
    return list(zip(st.tolist(), ed.tolist()))


def generate_topk_ranges_q(block_num_q: int, block_size_q: int, topk: int):
    # (block_num_q,)
    base = np.arange(block_num_q) * block_size_q
    # (block_num_q * topk,)
    starts = np.repeat(base, topk)
    ends = starts + block_size_q
    return list(zip(starts.tolist(), ends.tolist()))


def generate_topk_ranges_kv_tensor(
    topk_index_kv: torch.Tensor, stride: int, block_size_kv: int, seqlen_k: int
):
    # (block_num_q*topk,)
    heads_k = topk_index_kv.shape[0]
    idx_heads = topk_index_kv.reshape(heads_k, -1)
    starts = idx_heads * stride
    off_heads = torch.arange(
        start=0,
        end=heads_k * seqlen_k,
        step=seqlen_k,
        dtype=topk_index_kv.dtype,
        device=idx_heads.device,
    )
    starts = starts + off_heads[:, None]
    ends = starts + block_size_kv
    return torch.stack([starts.flatten(), ends.flatten()], dim=1).to(torch.int32)


def generate_topk_ranges_q_tensor(
    block_num_q: int, block_size_q: int, topk: int, device
):
    # (block_num_q,)
    base = torch.arange(block_num_q, dtype=torch.int32, device=device) * block_size_q
    # (block_num_q * topk,)
    starts = torch.repeat_interleave(base, topk)
    ends = starts + block_size_q
    return torch.stack([starts, ends], dim=1)


# (window_left segment has same val, window_right segment has same val) → full → 0
# (window_left segment has same val, window_right segment val increases) → causal → 1
# (window_left val increases, window_right segment has same val) → inverse causal → 2
# (window_left val increases, window_right val increases) → bidirectional causal → 3
def shard_qkv_range_for_sliding_window(
    q_range: AttnRange,
    k_range: AttnRange,
    window_size_left: int,
    window_size_right: int,
    base_offset_q: int,
    device,
):
    # (sq,)
    # [ left, right ]
    window_tri = torch.arange(
        start=q_range.start, end=q_range.end, dtype=torch.int32, device=device
    )
    window_left = torch.clamp(window_tri - window_size_left, min=k_range.start)
    window_right = torch.clamp(window_tri + window_size_right, max=k_range.end - 1)

    left_diff = torch.diff(window_left)
    right_diff = torch.diff(window_right)
    # 0-3
    mask_type = left_diff * 2 + right_diff
    mask_type = torch.cat([mask_type[:1], mask_type])
    # Mark change
    change_flag = torch.diff(mask_type) != 0
    # TODO: cuda symchronize
    segment_starts = change_flag.nonzero(as_tuple=True)[0] + 1
    segment_starts = F.pad(segment_starts, (1, 0), value=0)
    segment_types = mask_type[segment_starts]

    # segment ranges
    segment_ends = F.pad(segment_starts[1:], (0, 1), value=len(mask_type))
    # [ left, right )
    segment_ranges_q = (
        torch.stack([segment_starts, segment_ends], dim=1) + base_offset_q
    )

    # left_ranges_kv = window_left[segment_ranges_q[:, 0]]
    # right_ranges_kv = window_right[segment_ranges_q[:, 1]]
    left_ranges_kv = window_left[segment_starts]
    right_ranges_kv = window_right[segment_ends - 1]
    # [ left, right )
    segment_ranges_kv = torch.stack([left_ranges_kv, right_ranges_kv + 1], dim=1)

    return (
        segment_ranges_q.to(torch.int32),
        segment_ranges_kv.to(torch.int32),
        segment_types.to(torch.int32),
    )


def extract_blocks_varlen(input: torch.Tensor, l: int, d: int):  # noqa: E741
    seqlen, heads, dim = input.shape
    num_blocks = (seqlen - l) // d + 1
    device = input.device
    start_indices = torch.arange(0, num_blocks * d, d, device=device)
    offsets = torch.arange(l, device=device)
    # [num_blocks,l]
    gather_indices = start_indices[:, None] + offsets[None, :]
    # s,l.h,d
    input_expand = input.unsqueeze(1).expand(-1, l, -1, -1)
    # n,l,h,d
    gather_indices = (
        gather_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, heads, dim)
    )
    blocks = torch.gather(input_expand, dim=0, index=gather_indices)
    return blocks


# q,k,v dispatch is contiguous
class FFATopkAGAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,  # local non-cmp
        k,
        v,
        causal,  # False
        softmax_scale,
        topk_index_kv,  # Tensor [q_heads, q_seqlen, k], global index
        block_size_kv,
        block_size_q,
        stride,
        deterministic,
        cp_group,
        total_gather_indices=None,
        total_scatter_indices=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k_ag = gather_with_reorder_before_attn(k, total_gather_indices, cp_group)
        v_ag = gather_with_reorder_before_attn(v, total_gather_indices, cp_group)

        seqlen_q, heads_q, hidden_dim = q.shape
        seqlen_k, heads_k = k_ag.shape[0], k_ag.shape[1]
        H = heads_q // heads_k
        # [heads_k, q_block_num, k]
        cmp_topk_index_kv = topk_index_kv[::H, ::block_size_q, :]
        device = q.device

        # construct meta args
        block_num_q_per_head = cmp_topk_index_kv.shape[1]
        topk = cmp_topk_index_kv.shape[2]
        max_seqlen_q = block_size_q
        max_seqlen_k = block_size_kv

        # attn_type_map (torch.Tensor): Attention type map with dtype=torch.int32.
        #     0: full attention
        #     1: causal attention

        if causal:
            attn_type_map_tensor = torch.ones(
                block_num_q_per_head * heads_k * topk, dtype=torch.int32, device=device
            )
        else:
            attn_type_map_tensor = torch.zeros(
                block_num_q_per_head * heads_k * topk, dtype=torch.int32, device=device
            )

        # [h*t, H, d]
        flatten_q = q.view(seqlen_q, -1, H, hidden_dim).transpose(0, 1).contiguous()
        flatten_q = flatten_q.view(-1, H, hidden_dim)
        # [h*t, 1, d]
        flatten_k_ag = k_ag.unsqueeze(2).transpose(0, 1).contiguous()
        flatten_k_ag = flatten_k_ag.view(-1, 1, hidden_dim)
        flatten_v_ag = v_ag.unsqueeze(2).transpose(0, 1).contiguous()
        flatten_v_ag = flatten_v_ag.view(-1, 1, hidden_dim)

        q_ranges_tensor = generate_topk_ranges_q_tensor(
            block_num_q_per_head * heads_k, block_size_q, topk, device
        )
        k_ranges_tensor = generate_topk_ranges_kv_tensor(
            cmp_topk_index_kv, stride, block_size_kv, seqlen_k
        )

        ffa_forward_args = [
            None,  # merge_q_ranges
            None,  # fwd_qk_map
            None,  # fwd_unique_count
            None,  # ref_block_size
            softmax_scale,
            0.0,  # softcap
            False,  # disable_fwd_atomic_reduction
            q.dtype,  # out_type
            deterministic,
            0,  # sm_margin
        ]

        # run ffa forward
        flatten_out, flatten_softmax_lse = _flex_flash_attn_forward(
            flatten_q,
            flatten_k_ag,
            flatten_v_ag,
            None,  # out
            None,  # lse
            q_ranges_tensor,
            k_ranges_tensor,
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map_tensor,
            *ffa_forward_args,
        )

        out = flatten_out.view(-1, seqlen_q, H, hidden_dim).transpose(0, 1).contiguous()
        out = out.view(seqlen_q, -1, hidden_dim)
        softmax_lse = (
            flatten_softmax_lse.view(H, -1, seqlen_q).transpose(0, 1).contiguous()
        )
        softmax_lse = softmax_lse.view(-1, seqlen_q)

        ctx.save_for_backward(q, k, v, flatten_out, flatten_softmax_lse)
        ctx.total_gather_indices = total_gather_indices
        ctx.total_scatter_indices = total_scatter_indices
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.q_ranges_tensor = q_ranges_tensor
        ctx.k_ranges_tensor = k_ranges_tensor
        ctx.cp_group = cp_group
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.causal = causal

        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, out, softmax_lse) = saved_tensors[:5]

        k_ag = gather_with_reorder_before_attn(
            k, ctx.total_gather_indices, ctx.cp_group
        )
        v_ag = gather_with_reorder_before_attn(
            v, ctx.total_gather_indices, ctx.cp_group
        )

        seqlen_q, heads_q, hidden_dim = q.shape
        seqlen_k, heads_k = k_ag.shape[0], k_ag.shape[1]
        H = heads_q // heads_k
        # [h*t, H, d]
        flatten_q = q.view(seqlen_q, -1, H, hidden_dim).transpose(0, 1).contiguous()
        flatten_q = flatten_q.view(-1, H, hidden_dim)
        flatten_dout = (
            dout.view(seqlen_q, -1, H, hidden_dim).transpose(0, 1).contiguous()
        )
        flatten_dout = flatten_dout.view(-1, H, hidden_dim)
        # [h*t, 1, d]
        flatten_k_ag = k_ag.unsqueeze(2).transpose(0, 1).contiguous()
        flatten_k_ag = flatten_k_ag.view(-1, 1, hidden_dim)
        flatten_v_ag = v_ag.unsqueeze(2).transpose(0, 1).contiguous()
        flatten_v_ag = flatten_v_ag.view(-1, 1, hidden_dim)

        device = q.device
        attn_block_num = ctx.q_ranges_tensor.shape[0]
        if ctx.causal:
            attn_type_map_tensor = torch.ones(
                attn_block_num, dtype=torch.int32, device=device
            )
        else:
            attn_type_map_tensor = torch.zeros(
                attn_block_num, dtype=torch.int32, device=device
            )

        heads_q = q.shape[1]
        heads_k = k.shape[1]
        H = heads_q // heads_k

        ffa_backward_args = [
            0.0,  # softcap
            False,  # disable_bwd_dkv_atomic_reduction
            torch.float32,  # dq_type
            torch.float32,  # dk_type
            torch.float32,  # dv_type
            ctx.deterministic,
            0,  # sm_margin
        ]

        flatten_dq, flatten_dk, flatten_dv, _ = _flex_flash_attn_backward(
            flatten_dout,
            flatten_q,
            flatten_k_ag,
            flatten_v_ag,
            out,
            None,  # dq
            None,  # dk
            None,  # dv
            softmax_lse,
            ctx.q_ranges_tensor,
            ctx.k_ranges_tensor,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            attn_type_map_tensor,
            None,  # merge_k_ranges,
            None,  # bwd_kq_map,
            None,  # bwd_unique_count,
            ctx.softmax_scale,
            *ffa_backward_args,
        )

        # [t,h,d]
        dq = flatten_dq.view(-1, seqlen_q, H, hidden_dim).transpose(0, 1).contiguous()
        dq = dq.view(seqlen_q, -1, hidden_dim)
        dk = flatten_dk.view(-1, seqlen_k, 1, hidden_dim).transpose(0, 1).contiguous()
        dk = dk.view(seqlen_k, -1, hidden_dim)
        dv = flatten_dv.view(-1, seqlen_k, 1, hidden_dim).transpose(0, 1).contiguous()
        dv = dv.view(seqlen_k, -1, hidden_dim)

        dk = reorder_before_reduce_scatter(dk, ctx.total_scatter_indices)
        dv = reorder_before_reduce_scatter(dv, ctx.total_scatter_indices)

        dk_part, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv_part, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        return (
            dq,
            dk_part,
            dv_part,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FFAWinAGAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        causal,  # False
        # q_ranges,   # global List[List[int]] [b, 2]
        # k_ranges,
        softmax_scale,
        # window_size_left,
        # window_size_right,
        deterministic,
        cp_group,
        runtime_meta,
        total_gather_indices=None,
        total_scatter_indices=None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k_ag = gather_with_reorder_before_attn(k, total_gather_indices, cp_group)
        v_ag = gather_with_reorder_before_attn(v, total_gather_indices, cp_group)

        ffa_forward_args = [
            None,  # merge_q_ranges
            None,  # fwd_qk_map
            None,  # fwd_unique_count
            None,  # ref_block_size
            softmax_scale,
            0.0,  # softcap
            False,  # disable_fwd_atomic_reduction
            q.dtype,  # out_type
            deterministic,
            0,  # sm_margin
        ]

        # run ffa forward
        out, softmax_lse = _flex_flash_attn_forward(
            q,
            k_ag,
            v_ag,
            None,  # out
            None,  # lse
            runtime_meta.q_ranges_tensor,
            runtime_meta.k_ranges_tensor,
            runtime_meta.max_seqlen_q,
            runtime_meta.max_seqlen_k,
            runtime_meta.attn_type_map_tensor,
            *ffa_forward_args,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.total_gather_indices = total_gather_indices
        ctx.total_scatter_indices = total_scatter_indices
        ctx.attn_type_map_tensor = runtime_meta.attn_type_map_tensor
        ctx.max_seqlen_q = runtime_meta.max_seqlen_q
        ctx.max_seqlen_k = runtime_meta.max_seqlen_k
        ctx.q_ranges_tensor = runtime_meta.q_ranges_tensor
        ctx.k_ranges_tensor = runtime_meta.k_ranges_tensor
        ctx.cp_group = cp_group
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.causal = causal

        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, out, softmax_lse) = saved_tensors[:5]

        k_ag = gather_with_reorder_before_attn(
            k, ctx.total_gather_indices, ctx.cp_group
        )
        v_ag = gather_with_reorder_before_attn(
            v, ctx.total_gather_indices, ctx.cp_group
        )

        ffa_backward_args = [
            0.0,  # softcap
            False,  # disable_bwd_dkv_atomic_reduction
            torch.float32,  # dq_type
            torch.float32,  # dk_type
            torch.float32,  # dv_type
            ctx.deterministic,
            0,  # sm_margin
        ]

        dq, dk, dv, _ = _flex_flash_attn_backward(
            dout,
            q,
            k_ag,
            v_ag,
            out,
            None,  # dq
            None,  # dk
            None,  # dv
            softmax_lse,
            ctx.q_ranges_tensor,
            ctx.k_ranges_tensor,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.attn_type_map_tensor,
            None,  # merge_k_ranges,
            None,  # bwd_kq_map,
            None,  # bwd_unique_count,
            ctx.softmax_scale,
            *ffa_backward_args,
        )

        dk = reorder_before_reduce_scatter(dk, ctx.total_scatter_indices)
        dv = reorder_before_reduce_scatter(dv, ctx.total_scatter_indices)

        dk_part, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv_part, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        return (
            dq,
            dk_part,
            dv_part,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            # None,
            # None,
            # None,
            # None,
        )


class FFACmpAGAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_cmp,  # local cmp
        k_cmp,
        v_cmp,
        causal,  # False
        q_ranges,  # global List[List[int]] [b, 2]
        k_ranges_cmp,
        softmax_scale,
        block_size_q,
        slc_top_k,
        blk_size_cmp,
        blk_size_slc,
        stride,
        num_blocks_slc,
        deterministic,
        cp_group,
        runtime_meta,
        total_gather_indices=None,
        total_scatter_indices=None,
    ):
        if softmax_scale is None:
            softmax_scale = q_cmp.shape[-1] ** (-0.5)

        k_ag_cmp = gather_with_reorder_before_attn(
            k_cmp, total_gather_indices, cp_group
        )
        v_ag_cmp = gather_with_reorder_before_attn(
            v_cmp, total_gather_indices, cp_group
        )

        # compute P_cmp
        # seqlen_k == num_block_cmp
        seqlen_q, heads_q = q_cmp.shape[0], q_cmp.shape[1]
        seqlen_k, heads_k = k_ag_cmp.shape[0], k_ag_cmp.shape[1]
        H = heads_q // heads_k
        k_ag_mha = k_ag_cmp.repeat_interleave(H, dim=1)

        device = q_cmp.device
        P_cmp = torch.full(
            (heads_q, seqlen_q, seqlen_k),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        for q_range, k_range in zip(q_ranges, k_ranges_cmp):
            start_q, end_q = q_range.start, q_range.end
            start_k, end_k = k_range.start, k_range.end
            q_part = q_cmp[start_q:end_q, :, :]
            k_part = k_ag_mha[start_k:end_k, :, :]

            attn_cmp_part = torch.einsum("shd,thd->hst", q_part, k_part)
            attn_cmp_part = attn_cmp_part.to(torch.float32) * softmax_scale
            # h,sq,sk
            P_cmp[:, start_q:end_q, start_k:end_k] = attn_cmp_part

        # compute P_slc
        P_cmp = F.softmax(P_cmp, dim=-1).to(q_cmp.dtype)

        if blk_size_cmp == blk_size_slc == stride:
            P_slc = P_cmp
        else:
            # TODO: bug fix
            P_slc = compute_p_slc(
                P_cmp, blk_size_slc, blk_size_cmp, stride, num_blocks_slc
            )

        # deal q_block_size
        P_slc = compute_blockq_p_slc(q_cmp, P_slc, block_size_q)
        # deal GQA
        P_slc = compute_gqa_p_slc(P_slc, heads_k)

        _, P_slc_idx = torch.topk(P_slc, dim=-1, k=slc_top_k)

        device = q_cmp.device
        # construct meta args
        if causal:
            attn_type_map_tensor = torch.ones(
                q_ranges.size, dtype=torch.int32, device=device
            )
        else:
            attn_type_map_tensor = torch.zeros(
                q_ranges.size, dtype=torch.int32, device=device
            )

        ffa_forward_args = [
            None,  # merge_q_ranges
            None,  # fwd_qk_map
            None,  # fwd_unique_count
            None,  # ref_block_size
            softmax_scale,
            0.0,  # softcap
            False,  # disable_fwd_atomic_reduction
            q_cmp.dtype,  # out_type
            deterministic,
            0,  # sm_margin
        ]

        # run ffa forward
        out, softmax_lse = _flex_flash_attn_forward(
            q_cmp,
            k_ag_cmp,
            v_ag_cmp,
            None,  # out
            None,  # lse
            runtime_meta.q_ranges_tensor,
            runtime_meta.k_ranges_tensor,
            runtime_meta.max_seqlen_q,
            runtime_meta.max_seqlen_k,
            attn_type_map_tensor,
            *ffa_forward_args,
        )

        ctx.save_for_backward(q_cmp, k_cmp, v_cmp, out, softmax_lse)
        ctx.total_gather_indices = total_gather_indices
        ctx.total_scatter_indices = total_scatter_indices
        ctx.max_seqlen_q = runtime_meta.max_seqlen_q
        ctx.max_seqlen_k = runtime_meta.max_seqlen_k
        ctx.q_ranges_tensor = runtime_meta.q_ranges_tensor
        ctx.k_ranges_tensor = runtime_meta.k_ranges_tensor
        ctx.cp_group = cp_group
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.causal = causal

        return out, softmax_lse, P_slc_idx

    @staticmethod
    def backward(ctx, dout, *args):
        (*saved_tensors,) = ctx.saved_tensors
        (q_cmp, k_cmp, v_cmp, out, softmax_lse) = saved_tensors[:5]

        k_ag_cmp = gather_with_reorder_before_attn(
            k_cmp, ctx.total_gather_indices, ctx.cp_group
        )
        v_ag_cmp = gather_with_reorder_before_attn(
            v_cmp, ctx.total_gather_indices, ctx.cp_group
        )

        device = q_cmp.device
        attn_block_num = ctx.q_ranges_tensor.shape[0]
        if ctx.causal:
            attn_type_map_tensor = torch.ones(
                attn_block_num, dtype=torch.int32, device=device
            )
        else:
            attn_type_map_tensor = torch.zeros(
                attn_block_num, dtype=torch.int32, device=device
            )

        ffa_backward_args = [
            0.0,  # softcap
            False,  # disable_bwd_dkv_atomic_reduction
            torch.float32,  # dq_type
            torch.float32,  # dk_type
            torch.float32,  # dv_type
            ctx.deterministic,
            0,  # sm_margin
        ]

        dq, dk, dv, _ = _flex_flash_attn_backward(
            dout,
            q_cmp,
            k_ag_cmp,
            v_ag_cmp,
            out,
            None,  # dq
            None,  # dk
            None,  # dv
            softmax_lse,
            ctx.q_ranges_tensor,
            ctx.k_ranges_tensor,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            attn_type_map_tensor,
            None,  # merge_k_ranges,
            None,  # bwd_kq_map,
            None,  # bwd_unique_count,
            ctx.softmax_scale,
            *ffa_backward_args,
        )

        dk = reorder_before_reduce_scatter(dk, ctx.total_scatter_indices)
        dv = reorder_before_reduce_scatter(dv, ctx.total_scatter_indices)

        dk_part, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv_part, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        return (
            dq,
            dk_part,
            dv_part,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# first dispatch ring，then ulysess
class USPAllGatherNSA:
    def __init__(
        self,
        cp_process_group: Dict,
        l_cmp,
        l_slc,
        slc_topk,
        stride,
        block_size_q,
        hidden_dim,
        dtype,
        device,
    ):
        self.pg_p2p = cp_process_group[ParallelMode.RING]
        self.pg_a2a = cp_process_group[ParallelMode.ULYSESS]

        self.l_cmp = l_cmp
        self.l_slc = l_slc
        self.slc_topk = slc_topk
        self.stride = stride
        self.block_size_q = block_size_q
        self.hidden_dim = hidden_dim

        self.num_blocks_slc = 0

        # local ranges
        self.shard_range_q: Optional[AttnRanges] = None
        self.local_ranges_q: Optional[AttnRanges] = None
        self.ranges_k: Optional[AttnRanges] = None
        self.ranges_k_cmp: Optional[AttnRanges] = None

        self.global_ranges_q: Optional[AttnRanges] = None

        # runtime meta
        self.win_runtime_meta = None
        self.cmp_runtime_meta = None

        # cmp mlp layer
        self.cmp_linear_k = torch.nn.Linear(l_cmp, 1, dtype=dtype, device=device)
        self.cmp_linear_v = torch.nn.Linear(l_cmp, 1, dtype=dtype, device=device)
        # cmp/slc/win
        self.gate_proj = torch.nn.Linear(hidden_dim, 3, dtype=dtype, device=device)

    @instrument_nvtx
    def pre_compute_attn_runtime_meta(
        self,
        ranges_q: AttnRanges,
        ranges_k: AttnRanges,
        window_size_left: int,
        window_size_right: int,
        device,
    ):
        # compute total_gather_indices,
        # compute total_scatter_indices,

        # check
        if not (self.l_cmp == self.l_slc == self.stride):
            assert self.l_slc > self.l_cmp, "l_slc must be greater than l_cmp"
            assert self.l_slc % self.stride == 0, "l_slc must be divisible by d"
            assert self.l_cmp % self.stride == 0, "l_cmp must be divisible by d"
            self.num_blocks_slc = (ranges_k[-1].end - self.l_slc) // self.stride + 1
        else:
            self.num_blocks_slc = (ranges_k[-1].end - self.l_cmp) // self.stride + 1

        # dispatch ranges
        self.dispatch_ranges(ranges_q, ranges_k)
        self.cmp_runtime_meta = RunTimeMeta(  # type: ignore[assignment]
            q_ranges_tensor=self.local_ranges_q.to_tensor(device),  # type: ignore[union-attr]
            k_ranges_tensor=self.ranges_k_cmp.to_tensor(device),  # type: ignore[union-attr]
            attn_type_map_tensor=None,
            max_seqlen_q=self.local_ranges_q.max_seqlen,  # type: ignore[union-attr]
            max_seqlen_k=self.ranges_k_cmp.max_seqlen,  # type: ignore[union-attr]
        )

        # print(f"{self.local_ranges_q=}")
        # print(f"{self.global_ranges_q=}")
        # print(f"{self.ranges_k_cmp=}")
        # print(f"{self.ranges_k=}")

        # compute runtime meta for FFAWinAGAttnFunc
        q_ranges_lst_win = []
        k_ranges_lst_win = []
        attn_type_map_lst_win = []
        base_offset_q = 0
        for q_range, k_range in zip(self.global_ranges_q, self.ranges_k):  # type: ignore[arg-type]
            seqlen_q = q_range.end - q_range.start
            (
                segment_q_ranges,
                segment_k_ranges,
                segment_attn_type,
            ) = shard_qkv_range_for_sliding_window(
                q_range,
                k_range,
                window_size_left,
                window_size_right,
                base_offset_q,
                device,
            )
            base_offset_q += seqlen_q
            q_ranges_lst_win.append(segment_q_ranges)
            k_ranges_lst_win.append(segment_k_ranges)
            attn_type_map_lst_win.append(segment_attn_type)

        q_ranges_tensor_win = torch.cat(q_ranges_lst_win, dim=0)
        k_ranges_tensor_win = torch.cat(k_ranges_lst_win, dim=0)
        attn_type_map_tensor_win = torch.cat(attn_type_map_lst_win, dim=0)

        # TODO: cuda symchronize
        max_seqlen_q_win = (
            (q_ranges_tensor_win[:, 1] - q_ranges_tensor_win[:, 0]).max().item()
        )
        max_seqlen_k_win = (
            (k_ranges_tensor_win[:, 1] - k_ranges_tensor_win[:, 0]).max().item()
        )

        # print(f"pre: {q_ranges_tensor_win=}")
        # print(f"pre: {k_ranges_tensor_win=}")
        # print(f"pre: {attn_type_map_tensor_win=}")
        # print(f"pre: {max_seqlen_q_win=},{max_seqlen_k_win=}")

        self.win_runtime_meta = RunTimeMeta(  # type: ignore[assignment]
            q_ranges_tensor=q_ranges_tensor_win,
            k_ranges_tensor=k_ranges_tensor_win,
            attn_type_map_tensor=attn_type_map_tensor_win,
            max_seqlen_q=max_seqlen_q_win,
            max_seqlen_k=max_seqlen_k_win,
        )

    def dispatch_ranges(
        self,
        ranges_q: AttnRanges,
        ranges_k: AttnRanges,
    ):
        self.local_ranges_q = AttnRanges()
        self.global_ranges_q = AttnRanges()
        self.ranges_k_cmp = AttnRanges()
        self.ranges_k = AttnRanges()

        for range_q, range_k in zip(ranges_q, ranges_k):
            if (
                range_q.start >= self.shard_range_q.end  # type: ignore[union-attr]
                or range_q.end <= self.shard_range_q.start  # type: ignore[union-attr]
            ):
                continue
            else:
                # q ranges meta
                truncate_range = range_q.truncate(
                    self.shard_range_q.start, self.shard_range_q.end  # type: ignore[union-attr]
                )
                self.global_ranges_q.append(
                    AttnRange(truncate_range.start, truncate_range.end)
                )
                truncate_range.start = truncate_range.start - self.shard_range_q.start  # type: ignore[union-attr]
                truncate_range.end = truncate_range.end - self.shard_range_q.start  # type: ignore[union-attr]
                self.local_ranges_q.append(truncate_range)

                # k ranges meta
                self.ranges_k.append(range_k)
                seqlen = range_k.seqlen
                seqlen_cmp = (seqlen - self.l_cmp) // self.stride + 1
                start_cmp = (range_k.start - self.l_cmp) // self.stride + 1
                cmp_range = AttnRange(start_cmp, start_cmp + seqlen_cmp)
                self.ranges_k_cmp.append(cmp_range)

    @instrument_nvtx
    def dispatch(
        self,
        x_global: torch.Tensor,
        ranges: AttnRanges,
    ):
        cp_size_p2p = dist.get_world_size(self.pg_p2p)
        cp_size_a2a = dist.get_world_size(self.pg_a2a)
        cp_rank_p2p = dist.get_rank(self.pg_p2p)
        cp_rank_a2a = dist.get_rank(self.pg_a2a)

        # ring dispatch
        x_shard = torch.chunk(x_global, cp_size_p2p, dim=0)[cp_rank_p2p].contiguous()

        # self dispatch q [start, end]
        chunk_size_ring = x_shard.shape[0]
        local_left_start = chunk_size_ring * cp_rank_p2p
        local_left_end = chunk_size_ring * (cp_rank_p2p + 1)
        self.shard_range_q = AttnRange(local_left_start, local_left_end)  # type: ignore[assignment]

        # ulysess dispatch
        x_local = torch.chunk(x_shard, cp_size_a2a, dim=0)[cp_rank_a2a].contiguous()

        return x_local

    @instrument_nvtx
    def undispatch(
        self,
        x_local: torch.Tensor,
    ):
        x_shard = all_gather_fwd_scatter_bwd(x_local, self.pg_a2a, dim=0).contiguous()

        x_global = all_gather_fwd_scatter_bwd(x_shard, self.pg_p2p, dim=0).contiguous()

        return x_global

    @instrument_nvtx
    def apply_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float,
        # window_size_left: int,
        # window_size_right: int,
        deterministic: bool,
    ):
        # all2all comm
        # t,h,d
        q_layer = _varlen_all2all_before_attn(q, self.pg_a2a)
        k_layer = _varlen_all2all_before_attn(k, self.pg_a2a)
        v_layer = _varlen_all2all_before_attn(v, self.pg_a2a)

        with add_nvtx_event("compress kv blocks"):
            # n,l,h,d
            k_cmp_blocks = extract_blocks_varlen(k_layer, self.l_cmp, self.stride)
            v_cmp_blocks = extract_blocks_varlen(v_layer, self.l_cmp, self.stride)
            # n,h,d
            k_cmp_layer = self.cmp_linear_k(k_cmp_blocks.permute(0, 2, 3, 1)).squeeze(
                -1
            )
            v_cmp_layer = self.cmp_linear_v(v_cmp_blocks.permute(0, 2, 3, 1)).squeeze(
                -1
            )

        with add_nvtx_event("FFACmpAGAttnFunc"):
            out_cmp, lse_cmp, P_slc_idx = FFACmpAGAttnFunc.apply(
                q_layer,
                k_cmp_layer,
                v_cmp_layer,
                # TODO
                False,  # causal
                self.local_ranges_q,
                self.ranges_k_cmp,
                softmax_scale,
                self.block_size_q,
                self.slc_topk,
                self.l_cmp,
                self.l_slc,
                self.stride,
                self.num_blocks_slc,
                deterministic,
                self.pg_p2p,
                self.cmp_runtime_meta,
                None,
                None,
            )

        # print(f"{out_cmp.shape=},{P_slc_idx.shape=}")

        with add_nvtx_event("FFATopkAGAttnFunc"):
            out_slc, lse_slc = FFATopkAGAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                # TODO
                False,  # causal
                softmax_scale,
                P_slc_idx,
                self.l_slc,
                self.block_size_q,
                self.stride,
                deterministic,
                self.pg_p2p,
                None,
                None,
            )

        # print(f"{out_slc.shape=}")

        with add_nvtx_event("FFAWinAGAttnFunc"):
            out_win, lse_win = FFAWinAGAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                # TODO
                False,  # causal
                # self.local_ranges_q,
                # self.ranges_k,
                softmax_scale,
                # window_size_left,
                # window_size_right,
                deterministic,
                self.pg_p2p,
                self.win_runtime_meta,
                None,
                None,
            )
        # out_slc = torch.zeros_like(out_cmp)
        # out_win = torch.zeros_like(out_cmp)

        # print(f"{out_win.shape=}")

        # t,h,3
        gate = self.gate_proj(q_layer)
        gate_score = F.sigmoid(gate)

        # t,3,h,d
        out_stack = torch.stack([out_cmp, out_slc, out_win], dim=1)
        output = torch.einsum("thc,tchd->thd", gate_score, out_stack)

        # all2all comm
        out_layer = _varlen_all2all_after_attn(output, self.pg_a2a)

        return out_layer, None
