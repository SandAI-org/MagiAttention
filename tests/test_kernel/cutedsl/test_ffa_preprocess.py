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


"""Bwd preprocess of the SM100 CuTeDSL FFA kernel with and without range merge.

The preprocess grid is bounded by total_q only when the q ranges the kernels
see are pairwise disjoint; the K-merge repeats a q range once per K group.
"""

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl import (
    RangeMergePlan,
    flex_flash_attn_func,
    plan_range_merge,
)
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import calc_inf_norm
from magi_attention.utils import make_attn_mask_from_ffa_args

# The kernel may exceed the low-precision reference error by this factor.
_NORM_RATIO = 2.0


class TestFfaPreprocess(DistTestBase):
    def setUp(self) -> None:
        # Skip in the parent before DistTestBase spawns ranks, including ranks
        # with no assigned parameter cases that would otherwise reach a barrier.
        if not torch.cuda.is_available() or get_device_arch()[1] not in (10, 11):
            self.skipTest("the bwd preprocess under test is the SM100/SM110 path")
        super().setUp()

    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @property
    def timeout(self) -> int:
        return 3600

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    def _log_ran(self, test_case: str) -> None:
        print(f"[RANK {self.rank}] ran {test_case}", flush=True)

    def ref_out_and_grads(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        q_ranges: list[list[int]],
        k_ranges: list[list[int]],
        attn_type_map: list[int],
        high_precision: bool,
    ) -> tuple[torch.Tensor, ...]:
        """(out, dQ, dK, dV) of the torch reference on packed thd inputs; fp64
        under ``high_precision``, else the input dtype."""
        mask = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_ranges),
            k_ranges=AttnRanges.from_ranges(k_ranges),
            attn_type_map=attn_type_map,
            total_seqlen_q=q.shape[0],
            total_seqlen_k=k.shape[0],
            device=q.device,
        )
        qkv = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        out, _ = ref_attn_func(
            q=qkv[0],
            k=qkv[1],
            v=qkv[2],
            mask=mask,
            layout="thd",
            high_precision=high_precision,
        )
        return (out.detach(), *torch.autograd.grad(out, qkv, do))

    def assert_close_to_ref(
        self,
        name: str,
        actual: torch.Tensor,
        hi: torch.Tensor,
        lo: torch.Tensor,
        test_case: str,
    ) -> None:
        """``actual`` within ``_NORM_RATIO`` of the low-precision reference
        error ``lo``, both measured against the fp64 reference ``hi``."""
        err = calc_inf_norm(actual.to(hi.dtype), hi)
        bound = _NORM_RATIO * calc_inf_norm(lo, hi) + 1e-4 * hi.abs().max().item()
        self.assertLessEqual(err, bound, f"{test_case} {name}: {err=} {bound=}")

    @with_run_in_mp
    @parameterize("q_layout", ["overlap", "disjoint"])
    @parameterize("merge", ["none", "merge", "plan"])
    @parameterize("d", [64, 128])
    def test_bwd_preprocess(self, d, merge, q_layout):
        """out/dQ/dK/dV match the reference with and without range merge, and
        the preprocess grid is bounded by total_q only for disjoint q without
        merge. In the overlap layout P's rows appear three times, so a
        total_q-bounded grid would leave U's rows unprocessed."""
        device, dtype, nheads = self.device, torch.bfloat16, 2
        p_rows, u_rows = [0, 512], [512, 640]
        k_segs = [[i * 128, (i + 1) * 128] for i in range(4)]
        if q_layout == "overlap":
            q_ranges, k_ranges = [p_rows, p_rows, p_rows, u_rows], k_segs
        else:
            q_ranges, k_ranges = [p_rows, u_rows], [[0, 384], [384, 512]]
        attn_type_map = [MT_MAP.full] * len(q_ranges)
        q_ranges_t = torch.tensor(q_ranges, device=device, dtype=torch.int32)
        k_ranges_t = torch.tensor(k_ranges, device=device, dtype=torch.int32)
        mask_types_t = torch.tensor(attn_type_map, device=device, dtype=torch.int32)
        range_merge: bool | RangeMergePlan = merge == "merge"
        if merge == "plan":
            range_merge = plan_range_merge(q_ranges_t, k_ranges_t, mask_types_t)
        # Overlapping q without merge needs the atomic dQ path; merge needs the
        # direct one.
        direct_dq = not (q_layout == "overlap" and merge == "none")
        torch.random.manual_seed(self.seed + d)
        q = torch.randn(640, nheads, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(512, nheads, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        test_case = f"[test_bwd_preprocess][{q_layout=}][{merge=}][{d=}]"
        self._log_ran(test_case)
        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_types=mask_types_t,
            max_seqlen_q=max(end - start for start, end in q_ranges),
            max_seqlen_k=max(end - start for start, end in k_ranges),
            disable_fwd_atomic_reduction=direct_dq,
            disable_bwd_dkv_atomic_reduction=True,
            range_merge=range_merge,
        )
        do = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)

        ref_args = dict(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
        )
        refs_hi = self.ref_out_and_grads(**ref_args, high_precision=True)
        refs_lo = self.ref_out_and_grads(**ref_args, high_precision=False)
        u_slice, k3_slice = slice(*u_rows), slice(*k_segs[3])
        for name, actual, ref_idx, rows in (
            ("out", out, 0, slice(None)),
            ("dq", dq, 1, slice(None)),
            ("dk", dk, 2, slice(None)),
            ("dv", dv, 3, slice(None)),
            ("dq[U]", dq, 1, u_slice),
            ("dk[K3]", dk, 2, k3_slice),
        ):
            self.assert_close_to_ref(
                name,
                actual[rows],
                refs_hi[ref_idx][rows],
                refs_lo[ref_idx][rows],
                test_case,
            )


if __name__ == "__main__":
    run_tests()
