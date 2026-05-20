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

from typing import Any
from unittest import TestCase

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp

# isort: off
from magi_attention import magi_attn_ext  # type: ignore[attr-defined]

# isort: on


class TestMergeRange(TestCase):
    @property
    def seed(self) -> int:
        return 42

    def merge_ranges_ref(
        self, outer_ranges: torch.Tensor, inner_ranges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sorted_idx = torch.argsort(outer_ranges[:, 0], dim=0, stable=True)
        sorted_outer_ranges = outer_ranges[sorted_idx]
        sorted_inner_ranges = inner_ranges[sorted_idx]

        merge_outer_ranges, _, counts = torch.unique_consecutive(
            sorted_outer_ranges, dim=0, return_inverse=True, return_counts=True
        )
        range_map = torch.cumsum(counts, dim=0, dtype=torch.int32)

        return merge_outer_ranges, sorted_outer_ranges, sorted_inner_ranges, range_map

    def merge_ranges(
        self, outer_ranges: torch.Tensor, inner_ranges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sorted_idx, is_sorted = magi_attn_ext.argsort_ranges(outer_ranges)
        (
            sorted_outer_ranges,
            sorted_inner_ranges,
            sorted_attn_type_map,
        ) = magi_attn_ext.reorder_ranges_and_attn_type_maps(
            outer_ranges, inner_ranges, None, sorted_idx, is_sorted
        )

        (
            merge_outer_ranges,
            range_map,
            unique_count,
        ) = magi_attn_ext.unique_consecutive_pairs(sorted_outer_ranges)

        range_map = range_map[1 : unique_count.item() + 1]

        return (
            merge_outer_ranges[: unique_count.item()],
            sorted_outer_ranges,
            sorted_inner_ranges,
            range_map,
        )

    def test_simple_case(self):
        device = torch.cuda.current_device()

        outer_ranges_small = torch.tensor(
            [
                [0, 256],
                [256, 512],
                [512, 768],
                [768, 1024],
                [0, 256],
                [256, 512],
                [512, 768],
                [768, 1024],
            ],
            dtype=torch.int32,
            device=device,
        )
        inner_ranges_small = torch.tensor(
            [
                [0, 256],
                [256, 512],
                [512, 768],
                [768, 1024],
                [0, 512],
                [0, 768],
                [0, 1024],
                [0, 1024],
            ],
            dtype=torch.int32,
            device=device,
        )

        (
            merge_outer_ranges_ref,
            sorted_outer_ranges_ref,
            sorted_inner_ranges_ref,
            range_map_ref,
        ) = self.merge_ranges_ref(outer_ranges_small, inner_ranges_small)
        (
            merge_outer_ranges,
            sorted_outer_ranges,
            sorted_inner_ranges,
            range_map,
        ) = self.merge_ranges(outer_ranges_small, inner_ranges_small)

        self.assertTrue(torch.equal(merge_outer_ranges, merge_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_outer_ranges, sorted_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_inner_ranges, sorted_inner_ranges_ref))
        self.assertTrue(torch.equal(range_map, range_map_ref))

    def test_simple_sorted(self):
        device = torch.cuda.current_device()

        outer_ranges_sorted = torch.tensor(
            [
                [0, 256],
                [0, 256],
                [256, 512],
                [256, 512],
                [512, 768],
                [512, 768],
                [768, 1024],
                [768, 1024],
            ],
            dtype=torch.int32,
            device=device,
        )
        inner_ranges = torch.tensor(
            [
                [0, 256],
                [0, 512],
                [256, 512],
                [0, 768],
                [512, 768],
                [0, 1024],
                [768, 1024],
                [0, 1024],
            ],
            dtype=torch.int32,
            device=device,
        )

        (
            merge_outer_ranges_ref,
            sorted_outer_ranges_ref,
            sorted_inner_ranges_ref,
            range_map_ref,
        ) = self.merge_ranges_ref(outer_ranges_sorted, inner_ranges)
        (
            merge_outer_ranges,
            sorted_outer_ranges,
            sorted_inner_ranges,
            range_map,
        ) = self.merge_ranges(outer_ranges_sorted, inner_ranges)

        self.assertTrue(torch.equal(merge_outer_ranges, merge_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_outer_ranges, sorted_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_inner_ranges, sorted_inner_ranges_ref))
        self.assertTrue(torch.equal(range_map, range_map_ref))

    def test_compilcate_case(self):
        NUM_PAIRS = 1024 * 1024
        KEY_X_MIN = 0
        KEY_X_MAX = 20
        KEY_Y_MIN = 0
        KEY_Y_MAX = 20
        device = torch.cuda.current_device()

        x_coords_tensor = torch.randint(
            low=KEY_X_MIN,
            high=KEY_X_MAX + 1,
            size=(NUM_PAIRS,),
            dtype=torch.int32,
            device=device,
        )
        y_coords_tensor = torch.randint(
            low=KEY_Y_MIN,
            high=KEY_Y_MAX + 1,
            size=(NUM_PAIRS,),
            dtype=torch.int32,
            device=device,
        )

        outer_ranges_large = torch.stack([x_coords_tensor, y_coords_tensor], dim=1)
        inner_ranges_large = outer_ranges_large.clone()

        (
            merge_outer_ranges_ref,
            sorted_outer_ranges_ref,
            sorted_inner_ranges_ref,
            range_map_ref,
        ) = self.merge_ranges_ref(outer_ranges_large, inner_ranges_large)
        (
            merge_outer_ranges,
            sorted_outer_ranges,
            sorted_inner_ranges,
            range_map,
        ) = self.merge_ranges(outer_ranges_large, inner_ranges_large)

        self.assertTrue(torch.equal(merge_outer_ranges, merge_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_outer_ranges, sorted_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_inner_ranges, sorted_inner_ranges_ref))
        self.assertTrue(torch.equal(range_map, range_map_ref))


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


class TestRangeMergeBwdLoopK(DistTestBase):
    """End-to-end correctness test for BWD LoopK path with auto_range_merge=True."""

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
            auto_range_merge=True,
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
