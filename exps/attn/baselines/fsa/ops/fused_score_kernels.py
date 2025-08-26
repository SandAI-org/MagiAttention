# This file provides a fused implementation of computing attention score for selected attention indices.
# TODO: this implementation may incur illegal memory access issues, will be fixed.
import math

import torch
import triton
import triton.language as tl

from nsa_ref.ops.utils import is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()


@triton.jit
def fused_score_kernel(
    q_ptr,  # q_len x h x d
    k_ptr,  # k_len x h x d
    lse_ptr,  # h x n
    bs_ptr,  # h x n x nb
    offs_ptr,  # BO
    kernel_size,
    kernel_stride,
    num_offs,  # BO
    num_k_blocks,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,  # which is also num_q_heads
    HEAD_DIM,
    # sm_scale
    sm_scale,
    max_blocks,
    pad_len,
    block_size,
    block_stride,
    init_blocks,
    local_blocks,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_lh,
    stride_ln,
    stride_bsh,
    stride_bsq,
    stride_bsnb,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_bkh = tl.program_id(0)
    pid_b = pid_bkh // NUM_KV_HEADS
    pid_kh = pid_bkh % NUM_KV_HEADS
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)  # the blocks id of k
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    k_start += pid_k * BLOCK_SIZE_K * num_k_blocks
    if pid_q * BLOCK_SIZE_Q >= q_len or pid_k * BLOCK_SIZE_K >= k_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_kh * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + q_start * stride_ln + pid_kh * stride_lh,
        shape=(q_len, 1),
        strides=(stride_ln, stride_lh),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1),
    )
    # load q and lse
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")

    for j in range(num_k_blocks):
        k_start_j = k_start + j * BLOCK_SIZE_K
        if k_start_j < k_len:
            off_d = tl.arange(0, BLOCK_SIZE_D)
            off_q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
            # k offsets
            off_k = (k_start_j + tl.arange(0, BLOCK_SIZE_K)) * block_stride - pad_len
            k_ptrs = k_ptr + pid_kh * stride_kh + off_k[None, :] * stride_kn + off_d[:, None] * stride_kd
            causal_mask = off_q[:, None] >= (off_k * kernel_stride + kernel_size - 1)[None, :]

            # init block score
            bs = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
            for i in range(num_offs):
                k = tl.load(k_ptrs, mask=causal_mask, other=0)
                w = tl.load(offs_ptr + i, mask=i < num_offs, other=0)
                # compute qk
                qk = tl.dot(q, k) * qk_scale
                # compute score and apply weight
                bs += w * tl.where(causal_mask, tl.exp2(qk - lse), 0)

                # increment pointers
                off_k += 1
                k_ptrs = k_ptr + pid_kh * stride_kh + off_k[None, :] * stride_kn + off_d[:, None] * stride_kd
                causal_mask = off_q[:, None] >= (off_k * kernel_stride + kernel_size - 1)[None, :]

            # init mask and local mask
            off_bq = off_q // block_size
            off_bk = tl.arange(0, BLOCK_SIZE_K)
            bs = tl.where(
                (
                    (off_bq[:, None] >= k_start_j + off_bk[None, :])
                    & (off_bq[:, None] < k_start_j + off_bk[None, :] + local_blocks)
                )
                | (off_bk[None, :] < init_blocks - k_start_j),
                float("inf"),
                bs,
            )

            # save output
            bs_ptrs = (
                bs_ptr
                + pid_kh.to(tl.int64) * stride_bsh
                + q_start * stride_bsq
                + k_start_j * stride_bsnb
                + off_q[:, None] * stride_bsq
                + off_bk[None, :] * stride_bsnb
            )

            tl.store(
                bs_ptrs,
                bs.to(bs_ptr.dtype.element_ty),
                mask=(off_q < q_len)[:, None] & (off_bk < max_blocks - k_start_j)[None, :],
            )


def _fused_attention_score_and_transform(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    lse: torch.Tensor,  # [num_q_heads, total_query_len]
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    init_blocks: int = 1,
    local_blocks: int = 2,
    align_baseline: bool = False,
) -> torch.Tensor:

    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    max_blocks = math.ceil(max_seqlen_q / block_size)
    # init block score
    block_scores = torch.zeros(
        num_k_heads,
        q_len,
        max_blocks,
        dtype=torch.float32 if align_baseline else torch.bfloat16,
        device=q.device,
    )
    offs = (
        torch.arange(kernel_size // kernel_stride, device=q.device)[:, None]
        + torch.arange(block_size // kernel_stride, device=q.device)[None, :]
    ).view(-1)

    offs = torch.histc(offs, bins=offs.max() + 1, min=0, max=offs.max())

    num_offs = int(offs.shape[0])
    for i in range(cu_seqlens_q.shape[0] - 1):
        q_seq = q[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        k_seq = k[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        lse_seq = lse[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        block_scores_seq = block_scores[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]

        _fused_attention_score_and_transform_per_seq(
            q_seq,
            k_seq,
            lse_seq,
            block_scores_seq,
            kernel_size,
            kernel_stride,
            block_size,
            offs,
            num_offs,
            cu_seqlens_q[i: i + 2] - cu_seqlens_q[i],
            cu_seqlens_k[i: i + 2] - cu_seqlens_k[i],
            cu_seqlens_q[i + 1] - cu_seqlens_q[i],
            cu_seqlens_k[i + 1] - cu_seqlens_k[i],
            sm_scale,
            init_blocks,
            local_blocks,
        )
        block_scores[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]] = block_scores_seq
    return block_scores


@torch.inference_mode()
def _fused_attention_score_and_transform_per_seq(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    lse: torch.Tensor,  # [num_q_heads, total_query_len]
    block_score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    offs,
    num_offs,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert lse.dtype == torch.float32  # lse here is log2(sum(exp(qk*scale))), not log(sum(exp(qk*scale)))
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert q_len > k_len
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)

    max_blocks = math.ceil(max_seqlen_q / block_size)

    pad_len = kernel_size // kernel_stride - 1
    max_blocks = math.ceil(max_seqlen_q / block_size)

    BLOCK_SIZE_K = min(128, triton.next_power_of_2(max_blocks))
    # ensure qk is valid on triton
    BLOCK_SIZE_K = max(BLOCK_SIZE_K, 16)
    BLOCK_SIZE_Q = 128

    # launch kernel
    num_k_blocks = 1
    grid = lambda META: (
        batch_size * num_k_heads,
        triton.cdiv(max_seqlen_q, BLOCK_SIZE_Q),
        triton.cdiv(max_blocks, BLOCK_SIZE_K * num_k_blocks),
    )
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    fused_score_kernel[grid](
            q,
            k,
            lse,
            block_score,
            offs,
            kernel_size,
            kernel_stride,
            num_offs,
            num_k_blocks,
            cu_seqlens_q,
            cu_seqlens_k,
            num_k_heads,
            head_dim,
            sm_scale,
            max_blocks,
            pad_len,
            block_size,
            block_size // kernel_stride,
            init_blocks,
            local_blocks,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            lse.stride(0),
            lse.stride(1),
            block_score.stride(0),
            block_score.stride(1),
            block_score.stride(2),
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=8,
            num_stages=3,
        )
