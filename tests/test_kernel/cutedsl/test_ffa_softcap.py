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

import math
import unittest
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.ffa_utils import (
    MT_MAP,
    TorchFlexAttnArgs,
    normalize_softcap,
)
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.cutedsl_utils import (
    SM100_FAMILY_REQUIRED,
    CuteDSLDistTestBase,
    is_sm100_family,
    record_bwd_kernels,
    record_fwd_kernels,
)
from magi_attention.testing.dist_common import with_run_in_mp
from magi_attention.testing.precision import calc_inf_norm
from magi_attention.utils import make_attn_mask_from_ffa_args
from magi_attention.utils.arch import is_sm

_SOFTCAP = 30.0
# The kernel may exceed the low-precision reference error by this factor.
_NORM_RATIO = 2.0


@contextmanager
def _record_kernels() -> Iterator[dict[str, list]]:
    """Record both kernel specializations with empty compile caches."""
    with record_fwd_kernels() as fwd, record_bwd_kernels() as bwd:
        yield {"fwd": fwd, "bwd": bwd}


def _kernel_config(kernel) -> tuple:
    """The specialization fields softcap must leave unchanged."""
    return (
        kernel.cta_group_size,
        kernel.is_persistent,
        getattr(kernel, "pack_gqa", None),
        getattr(kernel, "gqa_fold", None),
        getattr(kernel, "range_merge", None),
    )


@pytest.mark.arch_independent
class TestNormalizeSoftcap(unittest.TestCase):
    def test_disabled_and_positive_values(self):
        self.assertIsNone(normalize_softcap(None))
        self.assertIsNone(normalize_softcap(0.0))
        self.assertEqual(normalize_softcap(30), 30.0)

    @parameterize("value", [-1.0, math.nan, math.inf, -math.inf])
    def test_rejects_negative_and_nonfinite_values(self, value):
        with self.assertRaisesRegex(ValueError, "softcap must be"):
            normalize_softcap(value)


class _SoftcapReference(unittest.TestCase):
    def assert_close_to_ref(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        out: torch.Tensor,
        grads: tuple[torch.Tensor, ...],
        q_ranges: list[list[int]],
        k_ranges: list[list[int]],
        attn_type_map: list[int],
        test_case: str,
        sink: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        softcap: float = _SOFTCAP,
        softmax_scale: float | None = None,
    ) -> None:
        """Use the input-dtype reference error against fp64 as the error budget."""
        mask = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_ranges),
            k_ranges=AttnRanges.from_ranges(k_ranges),
            attn_type_map=attn_type_map,
            total_seqlen_q=q.shape[0],
            total_seqlen_k=k.shape[0],
            device=q.device,
        )

        with_dsink = len(grads) == 4

        def ref(high_precision: bool) -> tuple[torch.Tensor, ...]:
            inputs = [t.detach().clone().requires_grad_() for t in (q, k, v)]
            sink_ref = None
            if sink is not None:
                sink_ref = sink.detach().clone().requires_grad_(with_dsink)
                if with_dsink:
                    inputs.append(sink_ref)
            out_ref, meta_ref = ref_attn_func(
                q=inputs[0],
                k=inputs[1],
                v=inputs[2],
                mask=mask,
                sink=sink_ref,
                softcap=softcap,
                softmax_scale=softmax_scale,
                layout="thd",
                backend="torch",
                high_precision=high_precision,
                return_lse=True,
            )
            grads_ref = torch.autograd.grad(out_ref, inputs, do)
            return (out_ref, *grads_ref, meta_ref.lse)

        refs_hi, refs_lo = ref(True), ref(False)
        if lse is not None:
            lse_ref = refs_hi[-1].to(lse.dtype)
            torch.testing.assert_close(
                lse, lse_ref, atol=1e-3, rtol=1e-3, msg=test_case
            )
        names = ("out", "dq", "dk", "dv", "dsink")[: 1 + len(grads)]
        for name, actual, hi, lo in zip(names, (out, *grads), refs_hi, refs_lo):
            err = calc_inf_norm(actual.to(hi.dtype), hi)
            bound = _NORM_RATIO * calc_inf_norm(lo, hi) + 1e-4 * hi.abs().max().item()
            self.assertLessEqual(err, bound, f"{test_case} {name}: {err=} {bound=}")


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFfaSoftcap(CuteDSLDistTestBase, _SoftcapReference):
    def _log_ran(self, test_case: str) -> None:
        print(f"[RANK {self.rank}] ran {test_case}", flush=True)

    def _fwd_bwd(self, q, k, v, softcap, **kwargs):
        inputs = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        if kwargs.get("sink") is not None:
            kwargs["sink"] = kwargs["sink"].detach().clone().requires_grad_()
            inputs.append(kwargs["sink"])
        with _record_kernels() as built:
            out, meta = flex_flash_attn_func(*inputs[:3], softcap=softcap, **kwargs)
            do = torch.randn_like(out)
            grads = torch.autograd.grad(out, inputs, do)
        return out, meta, do, grads, built

    def _assert_native(self, built: dict[str, list], test_case: str) -> None:
        for direction in ("fwd", "bwd"):
            (kernel,) = built[direction]
            self.assertTrue(kernel.has_softcap, test_case)
            self.assertIsNone(kernel.score_mod, test_case)

    @with_run_in_mp
    @parameterize("mha_type", ["mha", "gqa"])
    @parameterize("mask_type", [MT_MAP.full, MT_MAP.causal])
    @parameterize("d", [64, 128, 192])
    def test_dense(self, d, mask_type, mha_type):
        """Dense fwd/bwd with softcap match the torch reference and keep the
        kernel configuration (incl. 2-CTA at head_dim 128/192) of the
        uncapped call."""
        device, dtype = self.device, torch.bfloat16
        batch, seqlen, nheads = 2, 1000, 8
        nheads_kv = {"mha": nheads, "gqa": 2}[mha_type]
        d_v = 128 if d == 192 else d
        torch.random.manual_seed(self.seed + d + mask_type + nheads_kv)
        # Larger logits make the difference from uncapped attention measurable.
        q = 4.0 * torch.randn(batch, seqlen, nheads, d, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen, nheads_kv, d, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen, nheads_kv, d_v, device=device, dtype=dtype)
        test_case = f"[test_dense][{d=}][{mask_type=}][{mha_type=}]"
        self._log_ran(test_case)

        out, _, do, grads, built = self._fwd_bwd(
            q, k, v, _SOFTCAP, mask_types=mask_type
        )
        _, _, _, _, built_plain = self._fwd_bwd(q, k, v, 0.0, mask_types=mask_type)
        self._assert_native(built, test_case)
        for direction in ("fwd", "bwd"):
            self.assertEqual(
                [_kernel_config(x) for x in built[direction]],
                [_kernel_config(x) for x in built_plain[direction]],
                f"{test_case} {direction}",
            )
            self.assertFalse(any(x.has_softcap for x in built_plain[direction]))

        ranges = [[i * seqlen, (i + 1) * seqlen] for i in range(batch)]

        def thd(t: torch.Tensor) -> torch.Tensor:
            return rearrange(t, "b s h d -> (b s) h d")

        self.assert_close_to_ref(
            q=thd(q),
            k=thd(k),
            v=thd(v),
            do=thd(do),
            out=thd(out),
            grads=tuple(thd(g) for g in grads),
            q_ranges=ranges,
            k_ranges=ranges,
            attn_type_map=[mask_type] * batch,
            test_case=test_case,
        )

    def _ranges_inputs(self, nheads_kv: int, d: int, seed: int):
        """8 tile-unaligned relations cycling over the four mask types."""
        nrel, seqlen_q, seqlen_k, nheads = 8, 300, 520, 8
        attn_type_map = [
            MT_MAP.full,
            MT_MAP.causal,
            MT_MAP.inv_causal,
            MT_MAP.bi_causal,
        ] * (nrel // 4)
        torch.random.manual_seed(seed)
        device, dtype = self.device, torch.bfloat16
        q = 4.0 * torch.randn(nrel * seqlen_q, nheads, d, device=device, dtype=dtype)
        k = torch.randn(nrel * seqlen_k, nheads_kv, d, device=device, dtype=dtype)
        v = torch.randn_like(k)
        q_ranges = [[i * seqlen_q, (i + 1) * seqlen_q] for i in range(nrel)]
        k_ranges = [[i * seqlen_k, (i + 1) * seqlen_k] for i in range(nrel)]
        kwargs = dict(
            q_ranges=torch.tensor(q_ranges, device=device, dtype=torch.int32),
            k_ranges=torch.tensor(k_ranges, device=device, dtype=torch.int32),
            mask_types=torch.tensor(attn_type_map, device=device, dtype=torch.int32),
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
        )
        return q, k, v, q_ranges, k_ranges, attn_type_map, kwargs

    @with_run_in_mp
    @parameterize("few_ctas", [False, True])
    @parameterize("path", ["atomic", "direct", "merge"])
    @parameterize("d", [64, 128])
    def test_ranges(self, d, path, few_ctas):
        """Per-range mask types on the atomic, direct-store and RangeMerge
        paths, optionally on a persistent grid of two CTAs."""
        q, k, v, q_ranges, k_ranges, attn_type_map, kwargs = self._ranges_inputs(
            8, d, self.seed + d + len(path) + int(few_ctas)
        )
        num_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        kwargs.update(
            disable_fwd_atomic_reduction=path != "atomic",
            disable_bwd_dkv_atomic_reduction=path != "atomic",
            range_merge=path == "merge",
            sm_margin=num_sm - 2 if few_ctas else 0,
        )
        test_case = f"[test_ranges][{d=}][{path=}][{few_ctas=}]"
        self._log_ran(test_case)

        out, _, do, grads, built = self._fwd_bwd(q, k, v, _SOFTCAP, **kwargs)
        _, _, _, _, built_plain = self._fwd_bwd(q, k, v, 0.0, **kwargs)
        self._assert_native(built, test_case)
        for direction in ("fwd", "bwd"):
            self.assertEqual(
                [_kernel_config(x) for x in built[direction]],
                [_kernel_config(x) for x in built_plain[direction]],
                f"{test_case} {direction}",
            )
        if few_ctas:
            self.assertTrue(all(x.is_persistent for x in built["fwd"]), test_case)
        self.assert_close_to_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            out=out,
            grads=grads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("d", [64, 128])
    def test_ranges_gqa_packed_fwd_cat_gqa_bwd(self, d):
        """GQA ranges with the packed atomic fwd and the CatGQA bwd."""
        q, k, v, q_ranges, k_ranges, attn_type_map, kwargs = self._ranges_inputs(
            2, d, self.seed + d
        )
        test_case = f"[test_ranges_gqa_packed_fwd_cat_gqa_bwd][{d=}]"
        self._log_ran(test_case)
        out, _, do, grads, built = self._fwd_bwd(
            q, k, v, _SOFTCAP, pack_gqa=True, cat_gqa=True, **kwargs
        )
        self._assert_native(built, test_case)
        self.assertTrue(all(x.pack_gqa for x in built["fwd"]), test_case)
        for x in built["bwd"]:
            self.assertEqual(
                (x.gqa_fold.pack_factor, x.gqa_fold.head_loop), (1, True), test_case
            )
        self.assert_close_to_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            out=out,
            grads=grads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("pack_factor", [4, 2])
    @parameterize("d", [64, 128])
    def test_ranges_bwd_pack_fold(self, d, pack_factor):
        """GQA ranges with the packed bwd: every q head of a kv head shares
        one Q tile (pack factor 4), or two packed groups share it one after
        another in the head loop (pack factor 2)."""
        q, k, v, q_ranges, k_ranges, attn_type_map, kwargs = self._ranges_inputs(
            2, d, self.seed + d + pack_factor
        )
        qhead_per_kvhead = q.shape[1] // k.shape[1]
        test_case = f"[test_ranges_bwd_pack_fold][{d=}][{pack_factor=}]"
        self._log_ran(test_case)
        out, _, do, grads, built = self._fwd_bwd(
            q, k, v, _SOFTCAP, bwd_pack_factor=pack_factor, **kwargs
        )
        self._assert_native(built, test_case)
        (bwd_kernel,) = built["bwd"]
        self.assertEqual(
            (bwd_kernel.gqa_fold.pack_factor, bwd_kernel.gqa_fold.head_loop),
            (pack_factor, pack_factor < qhead_per_kvhead),
            test_case,
        )
        self.assert_close_to_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            out=out,
            grads=grads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )

    @with_run_in_mp
    def test_ranges_sink_and_max_logits(self):
        """The sink joins the capped scores in the softmax, and max logits
        report the capped maximum per q head."""
        q, k, v, q_ranges, k_ranges, attn_type_map, kwargs = self._ranges_inputs(
            2, 128, self.seed
        )
        sink = torch.randn(3, q.shape[1], device=q.device, dtype=torch.float32)
        test_case = "[test_ranges_sink_and_max_logits]"
        self._log_ran(test_case)
        out, meta, do, grads, built = self._fwd_bwd(
            q, k, v, _SOFTCAP, sink=sink, return_max_logits=True, **kwargs
        )
        self._assert_native(built, test_case)
        self.assert_close_to_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            out=out,
            grads=grads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
            sink=sink,
        )
        mask = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_ranges),
            k_ranges=AttnRanges.from_ranges(k_ranges),
            attn_type_map=attn_type_map,
            total_seqlen_q=q.shape[0],
            total_seqlen_k=k.shape[0],
            device=q.device,
        )
        _, meta_ref = ref_attn_func(
            q=q,
            k=k,
            v=v,
            mask=mask,
            softcap=_SOFTCAP,
            layout="thd",
            backend="torch",
            high_precision=True,
            return_max_logits=True,
        )
        self.assertTrue(torch.all(meta.max_logits <= _SOFTCAP), test_case)
        torch.testing.assert_close(
            meta.max_logits, meta_ref.max_logits.float(), atol=2e-2, rtol=1e-3
        )

    def _overlap_inputs(self, nheads_kv: int, d: int, seed: int, holes: bool):
        """Three disjoint q ranges, each covered by several relations against
        distinct K ranges, two K ranges shared by two q ranges, and a relation
        (bi_causal with len_k < len_q) without any attended pair. Q groups are
        identical or disjoint and so are K groups, so the same relations also
        run on RangeMerge. With ``holes``, some rows lie outside every q range
        and a fourth q range is covered only by a relation without any pair."""
        q_a, q_b = [0, 256], [256, 512]
        q_c = [520, 680] if holes else [512, 720]
        k_1, k_2, k_3, k_4, k_5 = (
            [0, 300],
            [300, 600],
            [600, 617],
            [617, 900],
            [900, 1100],
        )
        relations = [
            (q_a, k_1, MT_MAP.full),
            (q_a, k_2, MT_MAP.causal),
            (q_a, k_3, MT_MAP.bi_causal),
            (q_b, k_1, MT_MAP.inv_causal),
            (q_b, k_4, MT_MAP.full),
            (q_c, k_2, MT_MAP.bi_causal),
            (q_c, k_5, MT_MAP.full),
        ]
        if holes:
            relations.append(([680, 712], k_3, MT_MAP.bi_causal))
        total_q, total_k, nheads = 720, 1100, 8
        torch.random.manual_seed(seed)
        device, dtype = self.device, torch.bfloat16
        q = 4.0 * torch.randn(total_q, nheads, d, device=device, dtype=dtype)
        k = torch.randn(total_k, nheads_kv, d, device=device, dtype=dtype)
        v = torch.randn_like(k)
        q_ranges = [r[0] for r in relations]
        k_ranges = [r[1] for r in relations]
        attn_type_map = [r[2] for r in relations]
        kwargs = dict(
            q_ranges=torch.tensor(q_ranges, device=device, dtype=torch.int32),
            k_ranges=torch.tensor(k_ranges, device=device, dtype=torch.int32),
            mask_types=torch.tensor(attn_type_map, device=device, dtype=torch.int32),
            max_seqlen_q=256,
            max_seqlen_k=300,
        )
        return q, k, v, q_ranges, k_ranges, attn_type_map, kwargs

    @with_run_in_mp
    @parameterize("path", ["atomic", "merge_cat_gqa"])
    @parameterize("d", [64, 128])
    def test_overlapping_relations_merge_capped_partials(self, d, path):
        """Several relations per q row (atomic O/LSE merge) or per merged
        group (RangeMerge fwd Q groups and bwd K groups, with CatGQA), incl.
        a relation without any pair: the capped partials merge to the
        reference, i.e. the cap applies per score, not to the merged result."""
        q, k, v, q_ranges, k_ranges, attn_type_map, kwargs = self._overlap_inputs(
            2, d, self.seed + d + len(path), holes=path == "atomic"
        )
        if path == "merge_cat_gqa":
            kwargs.update(
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=True,
                range_merge=True,
                cat_gqa=True,
            )
        test_case = f"[test_overlapping_relations_merge_capped_partials][{d=}][{path=}]"
        self._log_ran(test_case)
        out, meta, do, grads, built = self._fwd_bwd(q, k, v, _SOFTCAP, **kwargs)
        self._assert_native(built, test_case)
        if path == "merge_cat_gqa":
            self.assertTrue(all(x.range_merge for x in built["fwd"]), test_case)
            for x in built["bwd"]:
                self.assertTrue(x.range_merge, test_case)
                self.assertEqual(
                    (x.gqa_fold.pack_factor, x.gqa_fold.head_loop),
                    (1, True),
                    test_case,
                )
        self.assert_close_to_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            out=out,
            grads=grads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
            lse=meta.lse,
        )

    @with_run_in_mp
    def test_runtime_cap_reuses_one_compiled_kernel(self):
        """The cap is a runtime scalar: one fwd and one bwd kernel serve
        several cap values (and a non-default softmax_scale) in turn, each
        matching its reference, incl. returning to the first value."""
        q, k, v, q_ranges, k_ranges, attn_type_map, kwargs = self._overlap_inputs(
            2, 128, self.seed, holes=False
        )
        softmax_scale = 0.07
        test_case = "[test_runtime_cap_reuses_one_compiled_kernel]"
        self._log_ran(test_case)
        with _record_kernels() as built:
            for softcap in (1.0, 30.0, 0.5, 1.0):
                inputs = [t.detach().clone().requires_grad_() for t in (q, k, v)]
                out, meta = flex_flash_attn_func(
                    *inputs, softcap=softcap, softmax_scale=softmax_scale, **kwargs
                )
                do = torch.randn_like(out)
                grads = torch.autograd.grad(out, inputs, do)
                self.assert_close_to_ref(
                    q=q,
                    k=k,
                    v=v,
                    do=do,
                    out=out,
                    grads=grads,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                    test_case=f"{test_case}[{softcap=}]",
                    lse=meta.lse,
                    softcap=softcap,
                    softmax_scale=softmax_scale,
                )
        self.assertEqual((len(built["fwd"]), len(built["bwd"])), (1, 1), test_case)

    @with_run_in_mp
    @parameterize("d", [64, 128])
    def test_saturated_scores(self, d):
        """Scores whose u = softmax_scale * q.k / softcap sits at 0, +-1,
        +-3 and +-6, where 1 - tanh(u)^2 ranges from 1 down to ~2.5e-5."""
        device, dtype = self.device, torch.bfloat16
        nrel, seqlen, nheads = 4, 256, 8
        softmax_scale = 1.0 / d**0.5
        targets = torch.tensor([0.0, 1.0, -1.0, 3.0, -3.0, 6.0, -6.0], device=device)
        torch.random.manual_seed(self.seed + d)
        # q = e_0, so q.k is k[..., 0]; k[..., 0] cycles through the targets.
        q = torch.zeros(nrel * seqlen, nheads, d, device=device, dtype=dtype)
        q[..., 0] = 1.0
        k = 0.01 * torch.randn(nrel * seqlen, nheads, d, device=device, dtype=dtype)
        u = targets[torch.arange(nrel * seqlen, device=device) % len(targets)]
        k[..., 0] = (u * _SOFTCAP / softmax_scale)[:, None].to(dtype)
        v = torch.randn_like(k)
        ranges = [[i * seqlen, (i + 1) * seqlen] for i in range(nrel)]
        attn_type_map = [MT_MAP.full, MT_MAP.causal] * (nrel // 2)
        kwargs = dict(
            q_ranges=torch.tensor(ranges, device=device, dtype=torch.int32),
            k_ranges=torch.tensor(ranges, device=device, dtype=torch.int32),
            mask_types=torch.tensor(attn_type_map, device=device, dtype=torch.int32),
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            softmax_scale=softmax_scale,
        )
        test_case = f"[test_saturated_scores][{d=}]"
        self._log_ran(test_case)
        out, meta, do, grads, built = self._fwd_bwd(q, k, v, _SOFTCAP, **kwargs)
        self._assert_native(built, test_case)
        self.assert_close_to_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            out=out,
            grads=grads,
            q_ranges=ranges,
            k_ranges=ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
            lse=meta.lse,
            softmax_scale=softmax_scale,
        )

    @with_run_in_mp
    def test_sink_dsink_with_holes_and_sink_only_rows(self):
        """Atomic fwd with a sink on the overlap inputs: rows outside every q
        range and rows of a relation without any pair hold only the sink;
        O, LSE, dQ/dK/dV and dsink match the reference."""
        q, k, v, q_ranges, k_ranges, attn_type_map, kwargs = self._overlap_inputs(
            2, 128, self.seed + 7, holes=True
        )
        sink = torch.randn(3, q.shape[1], device=q.device, dtype=torch.float32)
        test_case = "[test_sink_dsink_with_holes_and_sink_only_rows]"
        self._log_ran(test_case)
        out, meta, do, grads, built = self._fwd_bwd(
            q, k, v, _SOFTCAP, sink=sink, **kwargs
        )
        self._assert_native(built, test_case)
        self.assertEqual(len(grads), 4, test_case)
        self.assert_close_to_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            out=out,
            grads=grads,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
            sink=sink,
            lse=meta.lse,
        )

    @with_run_in_mp
    def test_zero_softcap_builds_the_uncapped_kernel(self):
        """``softcap=0.0`` means no cap: the uncapped kernels run and the
        results match the default call bitwise."""
        q, k, v, _, _, _, kwargs = self._ranges_inputs(8, 64, self.seed)
        out, _, _, _, built = self._fwd_bwd(q, k, v, 0.0, **kwargs)
        self.assertFalse(
            any(x.has_softcap for x in (*built["fwd"], *built["bwd"])),
            "softcap=0.0 built a capped kernel",
        )
        out_default, _ = flex_flash_attn_func(q, k, v, **kwargs)
        self.assertTrue(torch.equal(out, out_default))

    @with_run_in_mp
    def test_softcap_rejects_a_user_score_mod(self):
        q, k, v, _, _, _, kwargs = self._ranges_inputs(8, 64, self.seed)

        def score_mod(score, b, h, q_idx, kv_idx, aux_tensors):
            return score

        with self.assertRaisesRegex(AssertionError, "softcap and score_mod"):
            flex_flash_attn_func(
                q,
                k,
                v,
                softcap=_SOFTCAP,
                flex_attn_args=TorchFlexAttnArgs(score_mod=score_mod),
                **kwargs,
            )


class TestDenseSoftcap(_SoftcapReference):
    def _dense_qkv(self):
        torch.manual_seed(0)
        return tuple(
            torch.randn(
                1, 256, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            for _ in range(3)
        )

    def test_invalid_softcap_rejected_by_fwd_and_bwd(self):
        q, k, v = self._dense_qkv()
        lse = torch.empty(1, 4, 256, device=q.device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "softcap must be"):
            _flex_flash_attn_fwd(q, k, v, softcap=-1.0)
        with self.assertRaisesRegex(ValueError, "softcap must be"):
            _flex_flash_attn_bwd(q, k, v, q, lse, q, softcap=-1.0)

    def test_sm120_score_mod_output_and_gradients(self):
        if not is_sm(120, device=None):
            self.skipTest("requires SM120")

        q, k, v = self._dense_qkv()
        # Scores have unit variance; a cap of 1 exercises the tanh derivative.
        softcap = 1.0
        out, _ = flex_flash_attn_func(q, k, v, softcap=softcap)
        do = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), do)
        self.assert_close_to_ref(
            q=q[0],
            k=k[0],
            v=v[0],
            do=do[0],
            out=out[0],
            grads=tuple(g[0] for g in grads),
            q_ranges=[[0, q.shape[1]]],
            k_ranges=[[0, k.shape[1]]],
            attn_type_map=[MT_MAP.full],
            test_case="SM120 dense softcap",
            softcap=softcap,
        )


if __name__ == "__main__":
    run_tests()
