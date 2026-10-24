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


"""RangeMerge backward preprocess grid of the SM100 CuTeDSL FFA kernel.

The K-merge repeats a q range once per K group it pairs with, so the bwd
preprocess must not bound its grid by total_q under the merge.
"""

from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl import (
    RangeMergePlan,
    flex_flash_attn_func,
    plan_range_merge,
)
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_bwd_preprocess import (
    FFABwdPreProcess,
    bwd_preprocess,
)
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import calc_inf_norm
from magi_attention.utils import make_attn_mask_from_ffa_args

# The kernel may exceed the low-precision reference error by this factor.
_NORM_RATIO = 2.0


@contextmanager
def _record_bwd_kernels() -> Iterator[dict[str, list]]:
    """Record the SM100 bwd and bwd preprocess kernel objects built within the
    context, under ``"bwd"`` and ``"preprocess"``.

    The specialization is only visible on the kernel object, which a JIT cache
    hit never builds, so the context also swaps in empty compile caches.
    """
    built: dict[str, list] = {"bwd": [], "preprocess": []}
    bwd_init, preprocess_init = FFABwdSm100.__init__, FFABwdPreProcess.__init__

    def record_bwd(obj, *args, **kwargs):
        bwd_init(obj, *args, **kwargs)
        built["bwd"].append(obj)

    def record_preprocess(obj, *args, **kwargs):
        preprocess_init(obj, *args, **kwargs)
        built["preprocess"].append(obj)

    with mock.patch.object(FFABwdSm100, "__init__", record_bwd), mock.patch.object(
        FFABwdPreProcess, "__init__", record_preprocess
    ), mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ), mock.patch.object(
        _flex_flash_attn_bwd, "compile_cache", JITCache()
    ), mock.patch.object(
        bwd_preprocess, "compile_cache", JITCache()
    ):
        yield built


class TestFfaRangeMergePreprocess(DistTestBase):
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

    def _skip_unless_sm100(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("RangeMerge bwd requires SM100/SM110")

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
    @parameterize("path", ["merge", "plan", "disjoint"])
    @parameterize("d", [64, 128])
    def test_range_merge_preprocess_grid(self, d, path):
        """Under the K-merge (range_merge=True or a plan) the bwd preprocess
        grid must cover every merged q tile instead of stopping at total_q;
        disjoint q ranges without the merge keep the total_q-bounded grid."""
        self._skip_unless_sm100()
        device, dtype, nheads = self.device, torch.bfloat16, 2
        p_rows, u_rows = [0, 512], [512, 640]
        k_segs = [[i * 128, (i + 1) * 128] for i in range(4)]
        if path == "disjoint":
            q_ranges, k_ranges = [p_rows, u_rows], [[0, 384], [384, 512]]
        else:
            q_ranges, k_ranges = [p_rows, p_rows, p_rows, u_rows], k_segs
        attn_type_map = [MT_MAP.full] * len(q_ranges)
        q_ranges_t = torch.tensor(q_ranges, device=device, dtype=torch.int32)
        k_ranges_t = torch.tensor(k_ranges, device=device, dtype=torch.int32)
        mask_types_t = torch.tensor(attn_type_map, device=device, dtype=torch.int32)
        range_merge: bool | RangeMergePlan = path == "merge"
        if path == "plan":
            range_merge = plan_range_merge(q_ranges_t, k_ranges_t, mask_types_t)
        torch.random.manual_seed(self.seed + d + len(path))
        q = torch.randn(640, nheads, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(512, nheads, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        test_case = f"[test_range_merge_preprocess_grid][{d=}][{path=}]"
        self._log_ran(test_case)
        with _record_bwd_kernels() as built:
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=q_ranges_t,
                k_ranges=k_ranges_t,
                mask_types=mask_types_t,
                max_seqlen_q=max(end - start for start, end in q_ranges),
                max_seqlen_k=max(end - start for start, end in k_ranges),
                disable_fwd_atomic_reduction=True,
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

        merged = path != "disjoint"
        (bwd_kernel,) = built["bwd"]
        (preprocess,) = built["preprocess"]
        self.assertEqual(bwd_kernel.range_merge, merged, test_case)
        self.assertEqual(bwd_kernel.cta_group_size, 2 if d == 128 else 1, test_case)
        self.assertEqual(bwd_kernel.use_dense_dqacc_for_ranges, not merged, test_case)
        self.assertEqual(preprocess.q_ranges_disjoint, not merged, test_case)


if __name__ == "__main__":
    run_tests()
