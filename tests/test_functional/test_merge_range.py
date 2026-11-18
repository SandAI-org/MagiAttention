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

import unittest
from typing import Any

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.utils.arch import is_hopper


def _make_mask(q_ranges, k_ranges, attn_type_map, total_q, total_k):
    """Build a full attention mask from ranges + attn_type_map for reference computation."""
    mask = torch.zeros(total_q, total_k, dtype=torch.bool, device="cuda")
    qa = torch.arange(total_q, device="cuda").unsqueeze(1)
    ka = torch.arange(total_k, device="cuda").unsqueeze(0)
    for (q0, q1), (k0, k1), atype in zip(q_ranges, k_ranges, attn_type_map):
        in_q = (qa >= q0) & (qa < q1)
        in_k = (ka >= k0) & (ka < k1)
        rect = in_q & in_k
        m_rel = qa - q0
        n_rel = ka - k0
        sq, sk = q1 - q0, k1 - k0
        if atype == 0:
            mask |= rect
        elif atype == 1:
            mask |= rect & ((m_rel + sk - sq) >= n_rel)
        elif atype == 2:
            mask |= rect & (n_rel >= m_rel)
        elif atype == 3:
            mask |= rect & ((m_rel + sk - sq) >= n_rel) & (n_rel >= m_rel)
    return mask


def _ref_attn(q, k, v, mask):
    """Python fp32 reference attention."""
    Q, Hq, D = q.shape
    _, Hk, _ = k.shape
    q2 = q.permute(1, 0, 2).float()
    k2 = k.repeat_interleave(Hq // Hk, dim=1).permute(1, 0, 2).float()
    v2 = v.repeat_interleave(Hq // Hk, dim=1).permute(1, 0, 2).float()
    s = torch.einsum("hmd,hnd->hmn", q2, k2) / (D**0.5)
    s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
    p = s.softmax(dim=-1)
    p = torch.where(mask.unsqueeze(0), p, torch.zeros_like(p))
    o = torch.einsum("hmn,hnd->hmd", p, v2)
    return o.permute(1, 0, 2)


_RM_BWD_LOOPK_CASES = [
    {
        "name": "rm_dup_full",
        "q_ranges": [[0, 256]] * 2 + [[256, 512]] * 2 + [[512, 768]] * 2,
        "k_ranges": [[0, 128], [128, 256]] * 3,
        "atype_list": [0] * 6,
        "total_q": 768,
        "total_k": 256,
    },
    {
        "name": "rm_dup_causal",
        "q_ranges": [[0, 256]] * 2 + [[256, 512]] * 2 + [[512, 768]] * 2,
        "k_ranges": [[0, 128], [128, 256]] * 3,
        "atype_list": [1] * 6,
        "total_q": 768,
        "total_k": 256,
    },
    {
        "name": "rm_nondup_full",
        "q_ranges": [[0, 256], [256, 512], [512, 768]],
        "k_ranges": [[0, 256], [256, 512], [512, 768]],
        "atype_list": [0] * 3,
        "total_q": 768,
        "total_k": 768,
    },
    {
        "name": "rm_mixed_attn",
        "q_ranges": [[0, 256], [256, 512], [512, 768]],
        "k_ranges": [[0, 256], [256, 512], [512, 768]],
        "atype_list": [0, 1, 2],
        "total_q": 768,
        "total_k": 768,
    },
]


@unittest.skipUnless(is_hopper(), "requires Hopper")
class TestRangeMergeBwdLoopK(DistTestBase):
    """End-to-end correctness test for BWD LoopK path with range_merge=True."""

    @property
    def world_size(self) -> int:
        return 1

    def _run_case(self, cfg: dict[str, Any]):
        torch.manual_seed(42)
        dtype = torch.bfloat16
        Hq, Hk, D = 8, 2, 128
        q_ranges = cfg["q_ranges"]
        k_ranges = cfg["k_ranges"]
        atype_list = cfg["atype_list"]
        total_q = cfg["total_q"]
        total_k = cfg["total_k"]

        q = torch.randn(total_q, Hq, D, dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn(total_k, Hk, D, dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn(total_k, Hk, D, dtype=dtype, device="cuda", requires_grad=True)
        do = torch.randn(total_q, Hq, D, dtype=dtype, device="cuda")

        qr = torch.tensor(q_ranges, dtype=torch.int32, device="cuda")
        kr = torch.tensor(k_ranges, dtype=torch.int32, device="cuda")
        am = torch.tensor(atype_list, dtype=torch.int32, device="cuda")

        o, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=qr,
            k_ranges=kr,
            attn_type_map=am,
            range_merge=True,
            swap_bwd_qk_loop=True,
        )
        torch.cuda.synchronize()
        o.backward(do)
        torch.cuda.synchronize()

        dq_k = q.grad.detach().float()
        dk_k = k.grad.detach().float()
        dv_k = v.grad.detach().float()

        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        mask = _make_mask(q_ranges, k_ranges, atype_list, total_q, total_k)
        o_ref = _ref_attn(q2, k2, v2, mask)
        o_ref.backward(do.float())

        def _hybrid_err(a, b):
            return (a.float() - b.float()).norm().item() / max(
                b.float().norm().item(), 1.0
            )

        e_fwd = _hybrid_err(o.detach(), o_ref)
        e_dq = _hybrid_err(dq_k, q2.grad.float())
        e_dk = _hybrid_err(dk_k, k2.grad.float())
        e_dv = _hybrid_err(dv_k, v2.grad.float())

        threshold = 0.05
        self.assertLess(e_fwd, threshold, f"{cfg['name']} fwd err={e_fwd:.4f}")
        self.assertLess(e_dq, threshold, f"{cfg['name']} dq err={e_dq:.4f}")
        self.assertLess(e_dk, threshold, f"{cfg['name']} dk err={e_dk:.4f}")
        self.assertLess(e_dv, threshold, f"{cfg['name']} dv err={e_dv:.4f}")

    @with_run_in_mp
    @parameterize("config", _RM_BWD_LOOPK_CASES)
    def test_rm_bwd_loopk(self, config: dict[str, Any]):
        self._run_case(config)


if __name__ == "__main__":
    run_tests()
