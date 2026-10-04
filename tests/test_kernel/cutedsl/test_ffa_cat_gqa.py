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

"""CatGQA backward of the SM100 CuTeDSL FFA kernel.

With ``cat_gqa`` one bwd CTA per (K tile, kv head) walks the q heads of its
group, so dK/dV of the group accumulate in one CTA and are stored once.
"""

import math
from contextlib import contextmanager
from typing import Iterator
from unittest import mock

import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import calc_inf_norm
from magi_attention.testing.utils import switch_envvars
from magi_attention.utils import make_attn_mask_from_ffa_args

# The kernel may exceed the low-precision reference error by this factor.
_NORM_RATIO = 2.0


@contextmanager
def _record_bwd_kernels() -> Iterator[list[FFABwdSm100]]:
    """Record the SM100 bwd kernel objects built within the context.

    The specialization is only visible on the kernel object, which a JIT cache
    hit never builds, so the context also swaps in empty compile caches.
    """
    built: list[FFABwdSm100] = []
    init = FFABwdSm100.__init__

    def record(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    with mock.patch.object(FFABwdSm100, "__init__", record), mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ), mock.patch.object(_flex_flash_attn_bwd, "compile_cache", JITCache()):
        yield built


@contextmanager
def _maybe_disable_2cta(disable: bool) -> Iterator[None]:
    if not disable:
        yield
        return
    switch_back = switch_envvars(
        ["MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA"],
        enable_value_dict={"MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA": "1"},
    )
    try:
        yield
    finally:
        switch_back()


class TestFfaCatGqa(DistTestBase):
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
            self.skipTest("cat_gqa requires SM100/SM110")

    def _log_ran(self, test_case: str) -> None:
        print(f"[RANK {self.rank}] ran {test_case}", flush=True)

    def assert_grads_close_to_ref(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        grads: tuple[torch.Tensor, ...],
        q_ranges: list[list[int]],
        k_ranges: list[list[int]],
        attn_type_map: list[int],
        test_case: str,
    ) -> None:
        """dQ/dK/dV (packed thd) within ``_NORM_RATIO`` of the low-precision
        reference error, both measured against the fp64 reference."""
        mask = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_ranges),
            k_ranges=AttnRanges.from_ranges(k_ranges),
            attn_type_map=attn_type_map,
            total_seqlen_q=q.shape[0],
            total_seqlen_k=k.shape[0],
            device=q.device,
        )

        def ref(high_precision: bool) -> tuple[torch.Tensor, ...]:
            qkv = [t.detach().clone().requires_grad_() for t in (q, k, v)]
            out, _ = ref_attn_func(
                q=qkv[0],
                k=qkv[1],
                v=qkv[2],
                mask=mask,
                layout="thd",
                high_precision=high_precision,
            )
            return torch.autograd.grad(out, qkv, do)

        refs_hi, refs_lo = ref(True), ref(False)
        for name, actual, hi, lo in zip(("dq", "dk", "dv"), grads, refs_hi, refs_lo):
            err = calc_inf_norm(actual.to(hi.dtype), hi)
            bound = _NORM_RATIO * calc_inf_norm(lo, hi) + 1e-4 * hi.abs().max().item()
            self.assertLessEqual(err, bound, f"{test_case} {name}: {err=} {bound=}")

    # ─────────────────────────────────────────────────────────────────────
    # Dense
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("mha_type", ["gqa", "mqa"])
    @parameterize("two_cta", [True, False])
    @parameterize("d", [128, 192])
    def test_dense(self, d, two_cta, mha_type):
        """Dense GQA/MQA with unaligned Q/K lengths on the 1-CTA and 2-CTA
        kernels: matches the reference, and two ``deterministic=True`` calls
        agree bitwise."""
        self._skip_unless_sm100()
        if d == 192 and not two_cta:
            # The hd192 bwd mainloop is 2-CTA only. Per-combination skips
            # return: a SkipTest would end the whole test on this rank only.
            return

        device, dtype = self.device, torch.bfloat16
        batch, seqlen_q, seqlen_k, nheads = 2, 1000, 1200, 8
        nheads_kv = {"gqa": 2, "mqa": 1}[mha_type]
        d_v = 128 if d == 192 else d
        torch.random.manual_seed(self.seed + d + int(two_cta) + len(mha_type))
        q = torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen_k, nheads_kv, d_v, device=device, dtype=dtype)
        q, k, v = (t.requires_grad_() for t in (q, k, v))
        test_case = f"[test_dense][{d=}][{two_cta=}][{mha_type=}]"
        self._log_ran(test_case)

        do = None

        def run(cat_gqa: bool) -> tuple[torch.Tensor, ...]:
            nonlocal do
            out, _ = flex_flash_attn_func(
                q, k, v, mask_types=MT_MAP.causal, deterministic=True, cat_gqa=cat_gqa
            )
            if do is None:
                do = torch.randn_like(out)
            return torch.autograd.grad(out, (q, k, v), do)

        with _maybe_disable_2cta(not two_cta):
            non_cat = run(False)
            with _record_bwd_kernels() as built:
                cat = run(True)
            cat_again = run(True)
        (bwd_kernel,) = built
        self.assertTrue(bwd_kernel.cat_gqa, test_case)
        self.assertFalse(bwd_kernel.dKV_postprocess, test_case)
        self.assertEqual(bwd_kernel.cta_group_size, 2 if two_cta else 1, test_case)
        for name, a, b in zip(("dq", "dk", "dv"), cat, cat_again):
            self.assertTrue(torch.equal(a, b), f"{test_case} {name} not bitwise stable")

        assert do is not None
        q_ranges = [[i * seqlen_q, (i + 1) * seqlen_q] for i in range(batch)]
        k_ranges = [[i * seqlen_k, (i + 1) * seqlen_k] for i in range(batch)]
        for cat_gqa, grads in ((False, non_cat), (True, cat)):
            self.assert_grads_close_to_ref(
                q=rearrange(q.detach(), "b s h d -> (b s) h d"),
                k=rearrange(k.detach(), "b s h d -> (b s) h d"),
                v=rearrange(v.detach(), "b s h d -> (b s) h d"),
                do=rearrange(do, "b s h d -> (b s) h d"),
                grads=tuple(rearrange(g, "b s h d -> (b s) h d") for g in grads),
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=[MT_MAP.causal] * batch,
                test_case=f"{test_case}[{cat_gqa=}]",
            )

    # ─────────────────────────────────────────────────────────────────────
    # Ranges
    # ─────────────────────────────────────────────────────────────────────

    def _ranges_inputs(self, nheads_kv: int, d: int, seed: int):
        """24 tile-unaligned segments of full / causal / inv-causal relations."""
        nseg, seqlen, nheads = 24, 320, 8
        attn_type_map = [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal] * (nseg // 3)
        torch.random.manual_seed(seed)
        dtype, device = torch.bfloat16, self.device
        q = torch.randn(nseg * seqlen, nheads, d, device=device, dtype=dtype)
        k = torch.randn(nseg * seqlen, nheads_kv, d, device=device, dtype=dtype)
        v = torch.randn_like(k)
        ranges = [[i * seqlen, (i + 1) * seqlen] for i in range(nseg)]
        ranges_t = torch.tensor(ranges, device=device, dtype=torch.int32)
        kwargs = dict(
            q_ranges=ranges_t,
            k_ranges=ranges_t,
            mask_types=torch.tensor(attn_type_map, device=device, dtype=torch.int32),
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
        )
        out, meta = flex_flash_attn_func(q, k, v, **kwargs)
        do = torch.randn_like(out)
        return q, k, v, out.to(dtype), meta.lse, do, ranges, attn_type_map, kwargs

    @with_run_in_mp
    @parameterize("mha_type", ["gqa", "mqa"])
    @parameterize("direct", [False, True])
    @parameterize("few_ctas", [False, True])
    @parameterize("d", [64, 128])
    def test_ranges(self, d, few_ctas, direct, mha_type):
        """Ranges on the atomic dK/dV merge and the direct dK/dV store, which
        GQA takes only under ``cat_gqa``. ``few_ctas`` reserves all but two SMs,
        so each resident cluster walks at least three tiles of several q heads
        and the persistent loop carries pipeline phases and ``tile_done``
        across tiles."""
        self._skip_unless_sm100()
        nheads_kv = {"gqa": 2, "mqa": 1}[mha_type]
        q, k, v, out, lse, do, ranges, attn_type_map, kwargs = self._ranges_inputs(
            nheads_kv, d, self.seed + d + int(few_ctas) + int(direct) + nheads_kv
        )
        num_sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        sm_margin = num_sm - 2 if few_ctas else 0
        test_case = f"[test_ranges][{d=}][{few_ctas=}][{direct=}][{mha_type=}]"
        self._log_ran(test_case)

        def bwd(cat_gqa: bool) -> tuple[torch.Tensor, ...]:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q,
                k,
                v,
                out,
                lse,
                do,
                disable_bwd_dkv_atomic_reduction=direct,
                cat_gqa=cat_gqa,
                sm_margin=sm_margin,
                **kwargs,
            )
            return dq, dk, dv

        with _record_bwd_kernels() as built:
            cat = bwd(True)
        (bwd_kernel,) = built
        self.assertTrue(bwd_kernel.cat_gqa, test_case)
        self.assertEqual(bwd_kernel.is_persistent, few_ctas, test_case)
        self.assertEqual(bwd_kernel.dKV_postprocess, not direct, test_case)
        if few_ctas:
            cluster = bwd_kernel.cta_group_size
            seqlen_k = ranges[0][1] - ranges[0][0]
            tiles = len(ranges) * nheads_kv * math.ceil(seqlen_k / (128 * cluster))
            self.assertGreaterEqual(tiles, 3 * ((num_sm - sm_margin) // cluster))

        checked = [(True, cat)]
        if direct:
            with self.assertRaises(AssertionError):
                bwd(False)
        else:
            checked.append((False, bwd(False)))
        for cat_gqa, grads in checked:
            self.assert_grads_close_to_ref(
                q=q,
                k=k,
                v=v,
                do=do,
                grads=grads,
                q_ranges=ranges,
                k_ranges=ranges,
                attn_type_map=attn_type_map,
                test_case=f"{test_case}[{cat_gqa=}]",
            )

    @with_run_in_mp
    @parameterize("few_ctas", [False, True])
    @parameterize("d", [64, 128])
    def test_range_merge(self, d, few_ctas):
        """RangeMerge groups of several relations, some of which leave a K tile
        without any Q block (bi-causal / causal pairs), so every warp walks
        empty pairs inside the q-head loop."""
        self._skip_unless_sm100()
        device, dtype = self.device, torch.bfloat16
        q_groups = [(0, 257), (257, 386), (386, 579)]
        k_groups = [(0, 129), (129, 386), (386, 579)]
        relation_ids = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
        q_ranges = [list(q_groups[qi]) for qi, _ in relation_ids]
        k_ranges = [list(k_groups[ki]) for _, ki in relation_ids]
        attn_type_map = [
            MT_MAP.bi_causal,
            MT_MAP.causal,
            MT_MAP.full,
            MT_MAP.full,
            MT_MAP.inv_causal,
            MT_MAP.full,
        ]
        torch.random.manual_seed(self.seed + d + int(few_ctas))
        q = torch.randn(579, 8, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(579, 2, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        num_sm = torch.cuda.get_device_properties(self.device).multi_processor_count
        test_case = f"[test_range_merge][{d=}][{few_ctas=}]"
        self._log_ran(test_case)
        with _record_bwd_kernels() as built:
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=torch.tensor(q_ranges, device=device, dtype=torch.int32),
                k_ranges=torch.tensor(k_ranges, device=device, dtype=torch.int32),
                mask_types=torch.tensor(
                    attn_type_map, device=device, dtype=torch.int32
                ),
                max_seqlen_q=257,
                max_seqlen_k=257,
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=True,
                range_merge=True,
                sm_margin=num_sm - 2 if few_ctas else 0,
                cat_gqa=True,
            )
            do = torch.randn_like(out)
            grads = torch.autograd.grad(out, (q, k, v), do)
        (bwd_kernel,) = built
        self.assertTrue(bwd_kernel.cat_gqa and bwd_kernel.range_merge, test_case)
        self.assertEqual(bwd_kernel.is_persistent, few_ctas, test_case)
        self.assert_grads_close_to_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            grads=grads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Caller buffers
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("cat_gqa", [False, True])
    @parameterize("path", ["dense", "ranges_atomic", "ranges_direct"])
    def test_caller_buffers(self, path, cat_gqa):
        """dQ always accumulates into a caller buffer. dK/dV accumulate on the
        reducing paths (dense GQA without ``cat_gqa``, ranges with atomic dK/dV)
        and are overwritten by the single-writer paths (dense ``cat_gqa``,
        ranges with the direct dK/dV store)."""
        self._skip_unless_sm100()
        if path == "ranges_direct" and not cat_gqa:
            # GQA takes the direct dK/dV store only with cat_gqa (test_ranges
            # checks the rejection).
            return
        device, d = self.device, 128
        fp32 = dict(dq_type=torch.float32, dk_type=torch.float32, dv_type=torch.float32)
        if path == "dense":
            torch.random.manual_seed(self.seed + int(cat_gqa))
            q = torch.randn(1, 1000, 8, d, device=device, dtype=torch.bfloat16)
            k = torch.randn(1, 1200, 2, d, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            out, lse = _flex_flash_attn_fwd(q, k, v)
            do = torch.randn_like(out)
            kwargs: dict = {}
        else:
            q, k, v, out, lse, do, _, _, kwargs = self._ranges_inputs(
                2, d, self.seed + int(cat_gqa)
            )
            kwargs["disable_bwd_dkv_atomic_reduction"] = path == "ranges_direct"

        def bwd(**bufs) -> tuple[torch.Tensor, ...]:
            dq, dk, dv, _ = _flex_flash_attn_bwd(
                q, k, v, out, lse, do, cat_gqa=cat_gqa, **fp32, **kwargs, **bufs
            )
            return dq, dk, dv

        fresh = bwd()
        init = [torch.randn_like(g) for g in fresh]
        bufs = [t.clone() for t in init]
        got = bwd(dq=bufs[0], dk=bufs[1], dv=bufs[2])
        dkv_accumulates = path == "ranges_atomic" or (path == "dense" and not cat_gqa)
        test_case = f"[test_caller_buffers][{path=}][{cat_gqa=}]"
        self._log_ran(test_case)
        for name, g, buf, base, accumulates in zip(
            ("dq", "dk", "dv"),
            got,
            bufs,
            init,
            (True, dkv_accumulates, dkv_accumulates),
        ):
            self.assertIs(g, buf, f"{test_case} {name} must be the caller buffer")
            expected = fresh[("dq", "dk", "dv").index(name)]
            if accumulates:
                expected = expected + base
            torch.testing.assert_close(
                g,
                expected,
                rtol=0,
                atol=1e-4 * expected.abs().max().item(),
                msg=lambda m: f"{test_case} {name}: {m}",
            )


if __name__ == "__main__":
    run_tests()
