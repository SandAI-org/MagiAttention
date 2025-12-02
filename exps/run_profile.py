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
from attn.baselines.utils import seed_everything

from magi_attention.common.ranges import AttnRanges
from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import ref_attn_func
from magi_attention.testing.precision import calc_inf_norm
from magi_attention.utils._utils import make_attn_mask_from_ffa_args


def checkdiff(x, y, str):
    print(str)
    # diff = torch.testing.assert_close(x, y)
    # print(f"outputs close:{diff}")
    # print("max diff", (x - y).abs().max())
    norm = calc_inf_norm(x, y)
    print(f"Norm: {norm}")


# def checkdiff(x, y, str):
#     print(str)
#     diff = torch.testing.assert_close(x, y)
#     print(f"outputs close:{diff}")
#     print("max diff", (x - y).abs().max())

num_heads_q = 1  # number of attention (query) heads
num_heads_kv = 1  # number of key/value heads (GQA)
head_dim = 128  # dimension of each attention head
dtype = (
    torch.bfloat16
)  # attention activation / computation dtype (while the reduction dtype is always fp32 for ffa right now)
device = "cuda"

q_ranges = AttnRanges.from_ranges([[0, 8192]])

k_ranges = AttnRanges.from_ranges(
    [[0, 8192]],
)

attn_type_map = [0]
attn_type_map_tensor = torch.tensor([0] * 1, dtype=torch.int32, device=device)

total_seqlen_q = q_ranges.end
total_seqlen_k = k_ranges.end

q = torch.randn(total_seqlen_q, num_heads_q, head_dim, dtype=dtype, device=device)
k = torch.randn(total_seqlen_k, num_heads_kv, head_dim, dtype=dtype, device=device)
v = torch.randn(total_seqlen_k, num_heads_kv, head_dim, dtype=dtype, device=device)

q0 = q.clone().detach().requires_grad_(False)
k0 = k.clone().detach().requires_grad_(False)
v0 = v.clone().detach().requires_grad_(False)

q1 = q.clone().detach().requires_grad_(False)
k1 = k.clone().detach().requires_grad_(False)
v1 = v.clone().detach().requires_grad_(False)

q2 = q.clone().detach().requires_grad_(False)
k2 = k.clone().detach().requires_grad_(False)
v2 = v.clone().detach().requires_grad_(False)

q_ranges_tensor = q_ranges.to_tensor("cuda")
k_ranges_tensor = k_ranges.to_tensor("cuda")


def test_accuracy():
    # --- Attention computation --- #
    # SDPA
    mask = make_attn_mask_from_ffa_args(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        device="cuda",
    )
    seed_everything()
    out0, _ = ref_attn_func(q=q0, k=k0, v=v0, mask=mask, layout="thd")
    seed_everything()
    out0_ref, _ = ref_attn_func(
        q=q0, k=k0, v=v0, mask=mask, layout="thd", high_precision=True
    )

    seed_everything()
    (
        out1,
        _,
    ) = flex_flash_attn_func(  # the second return value is `lse` (log-sum-exp), known as the online-softmax correction factor
        q1,
        k1,
        v1,
        q_ranges=q_ranges_tensor,
        k_ranges=k_ranges_tensor,
        attn_type_map=attn_type_map_tensor,
        auto_range_merge=True,
        sparse_load=False,
        ref_block_size=(128, 128),
    )

    seed_everything()
    (
        out2,
        _,
    ) = flex_flash_attn_func(  # the second return value is `lse` (log-sum-exp), known as the online-softmax correction factor
        q2,
        k2,
        v2,
        q_ranges=q_ranges_tensor,
        k_ranges=k_ranges_tensor,
        attn_type_map=attn_type_map_tensor,
        auto_range_merge=True,
        sparse_load=True,
        ref_block_size=(128, 128),
    )

    # checkdiff(out1, out2, "\n\ncheck output:")
    # --- Compare outputs --- #
    checkdiff(out0_ref, out0, "\nSDPA reference vs. SDPA")
    checkdiff(out0_ref, out1, "\n\n SDPA reference vs. FFA baseline")
    checkdiff(out0_ref, out2, "\n\n SDPA reference vs. FFA Sparse Load")


def test_performance(sparse_load: bool):
    (
        out2,
        _,
    ) = flex_flash_attn_func(  # the second return value is `lse` (log-sum-exp), known as the online-softmax correction factor
        q2,
        k2,
        v2,
        q_ranges=q_ranges_tensor,
        k_ranges=k_ranges_tensor,
        attn_type_map=attn_type_map_tensor,
        auto_range_merge=True,
        sparse_load=sparse_load,
        ref_block_size=(128, 128),
    )


if __name__ == "__main__":
    test_performance(sparse_load=False)
    # test_accuracy()
