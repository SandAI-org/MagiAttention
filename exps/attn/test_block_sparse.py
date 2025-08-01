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
from baselines.attn_impl import sdpa_func
from baselines.utils import (
    flatten_head_mask,
    flatten_kvhead_mask,
    generate_headwise_4D_block_sparse_pattern,
    generate_kv_headwise_4D_block_sparse_pattern,
    generate_ranges_from_mask,
    get_sdpa_mask_from_block_sparse_mask,
)
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func

BLOCK_M = 64
BLOCK_N = 64

SEQLEN = 4096
BATCH_SIZE = 1

NUM_HEADS_Q = 48
NUM_HEADS_KV = 12

HEAD_DIM = 8
SPARSITY_RATIO = 0.1
DTYPE = torch.bfloat16

ATTN_IMPL = "ffa"


def get_sdpa_attn_ref(q, k, v, grad_output, block_mask, high_precision=False):
    q = rearrange(q, "b s h d -> b h s d")
    k = rearrange(k, "b s h d -> b h s d")
    v = rearrange(v, "b s h d -> b h s d")

    sdpa_mask_4d = get_sdpa_mask_from_block_sparse_mask(
        block_mask, SEQLEN, SEQLEN, BLOCK_M, BLOCK_N
    )
    print(f"{sdpa_mask_4d.shape=}")
    if high_precision:
        o = sdpa_func(
            q.to(torch.float64),
            k.to(torch.float64),
            v.to(torch.float64),
            attn_mask=sdpa_mask_4d,
            is_causal=False,
            enable_gqa=True,
        )
    else:
        o = sdpa_func(
            q,
            k,
            v,
            attn_mask=sdpa_mask_4d,
            is_causal=False,
            enable_gqa=True,
        )

    o = rearrange(o, "b h s d -> b s h d")
    o = o.to(q.dtype)
    o.backward(grad_output)

    return o


def get_ffa_result(q, k, v, grad_output, block_mask, head_wise):
    s, h = q.size(1), q.size(2)
    q = rearrange(
        q, "b s h d -> (b h s) 1 d"
    )  # flatten as (head dimension, seq dimension)

    assert NUM_HEADS_Q % NUM_HEADS_KV == 0

    repeats = NUM_HEADS_Q // NUM_HEADS_KV
    if head_wise == "q":
        k = torch.repeat_interleave(
            k, repeats=repeats, dim=2
        )  # we need to flatten k, v along head dimension for GQA setting.
        v = torch.repeat_interleave(v, repeats=repeats, dim=2)

    k = rearrange(k, "b s h d -> (b h s) 1 d")
    v = rearrange(v, "b s h d -> (b h s) 1 d")

    if head_wise == "q":
        flat_block_sparse_mask = flatten_head_mask(block_mask)
    else:
        flat_block_sparse_mask = flatten_kvhead_mask(
            block_mask, NUM_HEADS_Q, NUM_HEADS_KV
        )

    q_ranges, k_ranges = generate_ranges_from_mask(
        flat_block_sparse_mask, BLOCK_M, BLOCK_N
    )

    attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

    o, _ = flex_flash_attn_func(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        max_seqlen_q=BLOCK_M,
        max_seqlen_k=BLOCK_N,
        auto_range_merge=True,  # we should enable auto_range_merge for block sparse mask.
    )

    o = rearrange(o, "(b h s) 1 d -> b s h d", b=1, s=s, h=h)
    o.backward(grad_output)

    return o


# assert attn_impl with sdpa


def calc_inf_norm(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return (a.float() - b.float()).norm(p=float("inf")).item()


def assertLessEqual(a, b, msg=None):
    """Raises a ValueError if a is not less than or equal to b."""
    if not a <= b:
        raise ValueError(msg)  # You can use ValueError, AssertionError, etc.


def assert_close_to_torch_ref(q, k, v, grad_output, block_mask, head_wise):
    high_precision_torch_out_ref = get_sdpa_attn_ref(
        q, k, v, grad_output, block_mask, True
    )
    high_precision_dq_ref, high_precision_dk_ref, high_precision_dv_ref = (
        q.grad,
        k.grad,
        v.grad,
    )
    q.grad, k.grad, v.grad = None, None, None

    low_precision_torch_out_ref = get_sdpa_attn_ref(q, k, v, grad_output, block_mask)
    low_precision_dq_ref, low_precision_dk_ref, low_precision_dv_ref = (
        q.grad,
        k.grad,
        v.grad,
    )
    q.grad, k.grad, v.grad = None, None, None

    ffa_out = get_ffa_result(q, k, v, grad_output, block_mask, head_wise)
    ffa_dq, ffa_dk, ffa_dv = q.grad, k.grad, v.grad

    norm_rtol_ratio = 2.0

    out_norm = calc_inf_norm(ffa_out, high_precision_torch_out_ref)
    out_ref_norm = calc_inf_norm(
        low_precision_torch_out_ref, high_precision_torch_out_ref
    )

    assertLessEqual(
        out_norm,
        norm_rtol_ratio * out_ref_norm,
        msg=f"For {ATTN_IMPL=}: {out_norm=} should be no greater than {norm_rtol_ratio}x of {out_ref_norm=}",
    )

    dq_norm = calc_inf_norm(ffa_dq, high_precision_dq_ref)
    dq_ref_norm = calc_inf_norm(low_precision_dq_ref, high_precision_dq_ref)

    assertLessEqual(
        dq_norm,
        norm_rtol_ratio * dq_ref_norm,
        msg=f"For {ATTN_IMPL=}: {dq_norm=} should be no greater than {norm_rtol_ratio}x of {dq_ref_norm=}",
    )

    dk_norm = calc_inf_norm(ffa_dk, high_precision_dk_ref)
    dk_ref_norm = calc_inf_norm(low_precision_dk_ref, high_precision_dk_ref)

    assertLessEqual(
        dk_norm,
        norm_rtol_ratio * dk_ref_norm,
        msg=f"For {ATTN_IMPL=}: {dk_norm=} should be no greater than {norm_rtol_ratio}x of {dk_ref_norm=}",
    )

    dv_norm = calc_inf_norm(ffa_dv, high_precision_dv_ref)
    dv_ref_norm = calc_inf_norm(low_precision_dv_ref, high_precision_dv_ref)

    assertLessEqual(
        dv_norm,
        norm_rtol_ratio * dv_ref_norm,
        msg=f"For {ATTN_IMPL=}: {dv_norm=} should be no greater than {norm_rtol_ratio}x of {dv_ref_norm=}",
    )

    print("pass all!")


def test_q_headwise_sparse():
    orig_seqlen = SEQLEN
    orig_head = NUM_HEADS_Q

    num_q_blocks_orig = orig_seqlen // BLOCK_M
    num_kv_blocks_orig = orig_seqlen // BLOCK_N

    block_mask, _ = generate_headwise_4D_block_sparse_pattern(
        orig_head, num_q_blocks_orig, num_kv_blocks_orig, SPARSITY_RATIO, device="cuda"
    )

    # --- prepare q, k, v data --- #
    q = torch.randn(
        1,
        orig_seqlen,
        NUM_HEADS_Q,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )
    k = torch.randn(
        1,
        orig_seqlen,
        NUM_HEADS_KV,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )
    v = torch.randn(
        1,
        orig_seqlen,
        NUM_HEADS_KV,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )
    grad_output = torch.rand_like(q)

    assert_close_to_torch_ref(q, k, v, grad_output, block_mask, "q")


def test_kv_headwise_sparse():
    orig_seqlen = SEQLEN

    num_q_blocks_orig = orig_seqlen // BLOCK_M
    num_kv_blocks_orig = orig_seqlen // BLOCK_N

    kv_block_mask, _ = generate_kv_headwise_4D_block_sparse_pattern(
        NUM_HEADS_KV,
        num_q_blocks_orig,
        num_kv_blocks_orig,
        SPARSITY_RATIO,
        device="cuda",
    )

    num_groups = NUM_HEADS_Q // NUM_HEADS_KV
    kv_block_mask_extended = kv_block_mask.repeat_interleave(num_groups, dim=1)

    # --- prepare q, k, v data --- #
    q = torch.randn(
        1,
        orig_seqlen,
        NUM_HEADS_Q,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )
    k = torch.randn(
        1,
        orig_seqlen,
        NUM_HEADS_KV,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )
    v = torch.randn(
        1,
        orig_seqlen,
        NUM_HEADS_KV,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )
    grad_output = torch.rand_like(q)

    assert_close_to_torch_ref(q, k, v, grad_output, kv_block_mask_extended, "kv")


def main():
    test_q_headwise_sparse()
    test_kv_headwise_sparse()


main()
