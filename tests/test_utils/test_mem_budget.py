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

"""Tests for the analytical FFA memory ledger (magi_attention.utils.mem_budget).

The pytest part checks the ledger arithmetic against hand-derived values for
the canonical sparse benchmark config (MQA nhq=128/nhk=1, hd=128, bf16).

Running this file as a script additionally validates the model against a real
GPU run: it executes one dense FFA fwd+bwd and compares
``torch.cuda.max_memory_allocated`` with the predicted bwd peak.
"""

from magi_attention.utils.mem_budget import (
    FFAMemBudgetConfig,
    ffa_max_total_seqlen,
    ffa_memory_budget,
)

GIB = 1024**3

# Observed torch-usable budget on H100-80GB (79.19 GiB device capacity minus
# non-PyTorch overhead observed in the canonical bench OOM reports).
H100_BUDGET_BYTES = 76 * GIB


def _canonical(direction: str, seqlen: int, **kwargs) -> FFAMemBudgetConfig:
    return FFAMemBudgetConfig(
        direction=direction,
        total_seqlen_q=seqlen,
        total_seqlen_k=seqlen,
        **kwargs,
    )


class TestLedgerArithmetic:
    def test_bwd_dq_accum_at_524k_is_exactly_32_gib(self):
        # This is the famous "Tried to allocate 32.00 GiB" of the canonical
        # 524K bwd OOM: dq fp32 = S * nhq * hd * 4 = 2^19 * 2^7 * 2^7 * 2^2.
        budget = ffa_memory_budget(_canonical("bwd", 524288))
        assert budget.items["dq_accum"] == 32 * GIB

    def test_bwd_peak_at_524k_exceeds_h100(self):
        # q/out/dout (16 GiB each) + dq fp32 (32 GiB) + dq cast (16 GiB)
        # dominate: peak lands around 97 GiB, far beyond 80 GiB.
        budget = ffa_memory_budget(_canonical("bwd", 524288))
        assert budget.peak_bytes > 95 * GIB
        assert budget.peak_bytes < 99 * GIB

    def test_fwd_peak_at_524k_fits_h100(self):
        # fwd = q (16) + out fp32 (32) + out cast (16) + kv/lse: ~64.5 GiB,
        # which is why canonical fwd still runs at 524K while bwd OOMs.
        budget = ffa_memory_budget(_canonical("fwd", 524288))
        assert budget.peak_bytes < 66 * GIB

    def test_disable_fwd_atomic_reduction_saves_the_fp32_out(self):
        base = ffa_memory_budget(_canonical("fwd", 524288))
        no_atomic = ffa_memory_budget(
            _canonical("fwd", 524288, disable_fwd_atomic_reduction=True)
        )
        # fp32 out (32 GiB) replaced by io out (16 GiB), no cast copy (16 GiB)
        assert base.peak_bytes - no_atomic.peak_bytes == 32 * GIB


class TestMaxSeqlenSolver:
    def test_bwd_baseline_caps_around_425k(self):
        s_max = ffa_max_total_seqlen(H100_BUDGET_BYTES, _canonical("bwd", 1))
        assert 400_000 < s_max < 450_000, s_max

    def test_free_saved_out_lifts_cap_toward_510k(self):
        s_max = ffa_max_total_seqlen(
            H100_BUDGET_BYTES, _canonical("bwd", 1, bwd_free_saved_out=True)
        )
        assert 480_000 < s_max < 540_000, s_max

    def test_free_out_plus_8_chunks_reaches_600k_plus(self):
        s_max = ffa_max_total_seqlen(
            H100_BUDGET_BYTES,
            _canonical("bwd", 1, bwd_free_saved_out=True, bwd_num_qhead_chunks=8),
        )
        assert s_max > 600_000, s_max

    def test_each_knob_strictly_reduces_peak(self):
        s = 524288
        base = ffa_memory_budget(_canonical("bwd", s)).peak_bytes
        free_out = ffa_memory_budget(
            _canonical("bwd", s, bwd_free_saved_out=True)
        ).peak_bytes
        chunked = ffa_memory_budget(
            _canonical("bwd", s, bwd_num_qhead_chunks=8)
        ).peak_bytes
        both = ffa_memory_budget(
            _canonical("bwd", s, bwd_free_saved_out=True, bwd_num_qhead_chunks=8)
        ).peak_bytes
        assert base > free_out > both
        assert base > chunked > both


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
    # theory part
    import pytest

    rc = pytest.main([__file__, "-q"])
    assert rc == 0, "ledger arithmetic tests failed"

    # GPU reconciliation part (canonical-like heads, small seqlen)
    _validate_against_gpu(total_seqlen=65536, num_heads_q=128, head_dim=128)
    print("GPU reconciliation OK")
