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

"""Opt-in native FFA memory-ledger reconciliation on Hopper.

Run from the repository root:
    python -m exps.attn.ffa_mem_budget --seqlen 65536 --num-heads-q 128 --head-dim 128
"""

import argparse

from magi_attention.utils.arch import is_hopper
from magi_attention.utils.mem_budget import FFAMemBudgetConfig, ffa_memory_budget

GIB = 1024**3


def _validate_against_gpu(total_seqlen: int, num_heads_q: int, head_dim: int) -> None:
    """Run one dense FFA fwd+bwd and compare measured vs predicted peaks."""
    import torch

    from magi_attention.functional.flex_flash_attn import flex_flash_attn_func

    device = torch.device("cuda")
    s, nhq, nhk, hd = total_seqlen, num_heads_q, 1, head_dim

    q = torch.randn(s, nhq, hd, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(s, nhk, hd, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(s, nhk, hd, device=device, dtype=torch.bfloat16, requires_grad=True)
    q_ranges = torch.tensor([[0, s]], device=device, dtype=torch.int32)
    k_ranges = torch.tensor([[0, s]], device=device, dtype=torch.int32)

    # warmup (JIT compile + lazy handles) outside measurement
    out, _ = flex_flash_attn_func(q, k, v, q_ranges, k_ranges)
    out.backward(torch.ones_like(out))
    q.grad = k.grad = v.grad = None
    del out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    out, _ = flex_flash_attn_func(q, k, v, q_ranges, k_ranges)
    torch.cuda.synchronize()
    fwd_measured = torch.cuda.max_memory_allocated()
    dout = torch.randn_like(out)
    out.backward(dout)
    torch.cuda.synchronize()
    bwd_measured = torch.cuda.max_memory_allocated()

    fwd_pred = ffa_memory_budget(
        FFAMemBudgetConfig(
            direction="fwd",
            total_seqlen_q=s,
            total_seqlen_k=s,
            num_heads_q=nhq,
            num_heads_k=nhk,
            head_dim=hd,
        )
    )
    bwd_pred = ffa_memory_budget(
        FFAMemBudgetConfig(
            direction="bwd",
            total_seqlen_q=s,
            total_seqlen_k=s,
            num_heads_q=nhq,
            num_heads_k=nhk,
            head_dim=hd,
        )
    )

    print(bwd_pred.format())
    for name, measured, predicted in (
        ("fwd", fwd_measured, fwd_pred.peak_bytes),
        ("bwd", bwd_measured, bwd_pred.peak_bytes),
    ):
        rel_err = abs(measured - predicted) / predicted
        print(
            f"[{name}] measured={measured / GIB:.3f} GiB "
            f"predicted={predicted / GIB:.3f} GiB rel_err={rel_err:.2%}"
        )
        assert rel_err < 0.10, f"{name} ledger off by {rel_err:.2%}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, default=65536)
    parser.add_argument("--num-heads-q", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    assert is_hopper(), "native FFA memory reconciliation requires Hopper"
    _validate_against_gpu(args.seqlen, args.num_heads_q, args.head_dim)
