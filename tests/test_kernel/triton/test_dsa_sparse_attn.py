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

"""Correctness tests for the DSA Triton kernels in ``magi_attention.kernel.triton``.

Both kernels are compared against an fp32 gather-softmax reference built from
the same bf16 inputs.  The backward test feeds the reference ``out``/``lse`` to
``dsa_sparse_attn_bwd`` so it does not depend on any FFA forward.

Run:
    pytest tests/test_kernel/triton/test_dsa_sparse_attn.py -v
"""

import math

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from magi_attention.kernel.triton import dsa_sparse_attn_bwd, sparse_gqa_packed_attn
from magi_attention.testing import parameterize

_LN2 = math.log(2.0)


def _assert_rel_close(actual: torch.Tensor, expected: torch.Tensor, name: str, tol):
    actual = actual.float()
    expected = expected.float()
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"
    err = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    assert err.item() < tol, f"{name}: relative error {err.item():.3e} >= {tol:.1e}"


def _unique_indices(rows: int, skv: int, topk: int, device) -> torch.Tensor:
    """[rows, topk] int32, each row a distinct random subset of range(skv)."""
    return torch.stack(
        [torch.randperm(skv, device=device)[:topk] for _ in range(rows)]
    ).to(torch.int32)


def _packed_ref(q, k, v, index_map, scale):
    """fp32 reference for the packed layout; ``-1`` entries are masked out."""
    idx = index_map[0]
    valid = idx >= 0
    safe = idx.long().clamp_min(0)
    kf = k[safe, 0].float()  # [SQ, K, D]
    vf = v[safe, 0].float()
    s = q.float() @ kf.transpose(-2, -1) * scale  # [SQ, H, K]
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    lse = s.logsumexp(dim=-1)
    p = torch.exp(s - lse[..., None])
    return p @ vf, lse


def _grouped_ref(q, k, v, index_map, scale):
    """fp32 reference for [sq, nhq, hd] x [skv, nhkv, hd] with per-head index_map."""
    sq, nhq, hd = q.shape
    nhkv = k.shape[1]
    group = nhq // nhkv
    outs, lses = [], []
    for h in range(nhkv):
        idx = index_map[h].long()  # [sq, topk]
        kf = k[:, h].float()[idx]  # [sq, topk, hd]
        vf = v[:, h].float()[idx]
        qh = q[:, h * group : (h + 1) * group].float()  # [sq, group, hd]
        s = qh @ kf.transpose(-2, -1) * scale
        lse = s.logsumexp(dim=-1)
        outs.append(torch.exp(s - lse[..., None]) @ vf)
        lses.append(lse)
    return torch.cat(outs, dim=1), torch.cat(lses, dim=1)  # [sq, nhq, hd], [sq, nhq]


class TestDSASparseAttn(TestCase):
    @parameterize("num_heads", [16, 32])
    @parameterize("topk", [64, 200])
    @parameterize("bwd_autotune", [False, True])
    def test_sparse_gqa_packed_attn(self, num_heads, topk, bwd_autotune):
        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.bfloat16
        sq, skv, hd = 64, 1024, 128
        scale = hd**-0.5

        q = torch.randn(sq, num_heads, hd, device=device, dtype=dtype)
        k = torch.randn(skv, 1, hd, device=device, dtype=dtype)
        v = torch.randn(skv, 1, hd, device=device, dtype=dtype)
        idx = _unique_indices(sq, skv, topk, device)
        # ragged rows: pad the tail of every other row with -1
        idx[1::2, topk - topk // 4 :] = -1
        index_map = idx.unsqueeze(0).contiguous()

        q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (q, k, v))
        out_ref, lse_ref = _packed_ref(q_ref, k_ref, v_ref, index_map, scale)

        q_tri, k_tri, v_tri = (t.detach().clone().requires_grad_() for t in (q, k, v))
        out, lse_log2 = sparse_gqa_packed_attn(
            q_tri, k_tri, v_tri, index_map, scale, BWD_AUTOTUNE=bwd_autotune
        )

        _assert_rel_close(out, out_ref, "out", 1e-2)
        _assert_rel_close(lse_log2 * _LN2, lse_ref, "lse", 1e-3)

        dout = torch.randn_like(out)
        dlse = torch.randn_like(lse_ref)
        ((out_ref * dout.float()).sum() + (lse_ref * dlse).sum()).backward()
        ((out.float() * dout.float()).sum() + (lse_log2 * _LN2 * dlse).sum()).backward()

        _assert_rel_close(q_tri.grad, q_ref.grad, "dq", 2e-2)
        _assert_rel_close(k_tri.grad, k_ref.grad, "dk", 2e-2)
        _assert_rel_close(v_tri.grad, v_ref.grad, "dv", 2e-2)

    @parameterize("kv_group", [(1, 16), (2, 32), (1, 128)])
    @parameterize("use_bf16_atomic", [False, True])
    def test_dsa_sparse_attn_bwd(self, kv_group, use_bf16_atomic):
        num_heads_kv, group = kv_group
        torch.manual_seed(0)
        device = "cuda"
        dtype = torch.bfloat16
        sq, skv, hd, topk = 32, 512, 128, 192
        nhq = num_heads_kv * group
        scale = hd**-0.5

        q = torch.randn(sq, nhq, hd, device=device, dtype=dtype)
        k = torch.randn(skv, num_heads_kv, hd, device=device, dtype=dtype)
        v = torch.randn(skv, num_heads_kv, hd, device=device, dtype=dtype)
        index_map = torch.stack(
            [_unique_indices(sq, skv, topk, device) for _ in range(num_heads_kv)]
        ).contiguous()

        q_ref, k_ref, v_ref = (t.detach().clone().requires_grad_() for t in (q, k, v))
        out_ref, lse_ref = _grouped_ref(q_ref, k_ref, v_ref, index_map, scale)
        dout = torch.randn(sq, nhq, hd, device=device, dtype=dtype)
        dlse = torch.randn(sq, nhq, device=device, dtype=torch.float32)
        ((out_ref * dout.float()).sum() + (lse_ref * dlse).sum()).backward()

        lse_saved = (
            lse_ref.detach().view(sq, num_heads_kv, group).permute(1, 0, 2).contiguous()
        )
        dq, dk, dv = dsa_sparse_attn_bwd(
            q,
            k,
            v,
            index_map,
            out_ref.detach().to(dtype),
            lse_saved,
            dout,
            dlse,
            scale,
            use_bf16_atomic=use_bf16_atomic,
        )

        _assert_rel_close(dq, q_ref.grad, "dq", 2e-2)
        _assert_rel_close(dk, k_ref.grad, "dk", 2e-2)
        _assert_rel_close(dv, v_ref.grad, "dv", 2e-2)


if __name__ == "__main__":
    run_tests()
