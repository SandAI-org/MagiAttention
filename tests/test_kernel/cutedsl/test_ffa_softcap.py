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

"""Native softcap of the SM100/SM110 CuTeDSL FFA kernels.

The fwd and bwd mainloops compute ``softcap * tanh(s * softmax_scale /
softcap)`` before masking, and the bwd scales dS by ``1 - tanh^2``. No
score_mod is built on SM100/SM110, so softcap keeps the kernel configuration
(2-CTA, head_dim 192 bwd, RangeMerge, persistence, PackGQA, CatGQA) of the
uncapped call.
"""

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
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import (
    MT_MAP,
    TorchFlexAttnArgs,
    get_device_arch,
)
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import calc_inf_norm
from magi_attention.utils import make_attn_mask_from_ffa_args

_SOFTCAP = 30.0
# The kernel may exceed the low-precision reference error by this factor.
_NORM_RATIO = 2.0


@contextmanager
def _record_kernels() -> Iterator[dict[str, list]]:
    """Record the SM100 fwd/bwd kernel objects built within the context.

    The specialization is only visible on the kernel object, which a JIT cache
    hit never builds, so the context also swaps in empty compile caches.
    """
    built: dict[str, list] = {"fwd": [], "bwd": []}
    fwd_init, bwd_init = FFAFwdSm100.__init__, FFABwdSm100.__init__

    def record_fwd(obj, *args, **kwargs):
        fwd_init(obj, *args, **kwargs)
        built["fwd"].append(obj)

    def record_bwd(obj, *args, **kwargs):
        bwd_init(obj, *args, **kwargs)
        built["bwd"].append(obj)

    with mock.patch.object(FFAFwdSm100, "__init__", record_fwd), mock.patch.object(
        FFABwdSm100, "__init__", record_bwd
    ), mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ), mock.patch.object(
        _flex_flash_attn_bwd, "compile_cache", JITCache()
    ):
        yield built


def _kernel_config(kernel) -> tuple:
    """The specialization fields softcap must leave unchanged."""
    return (
        kernel.cta_group_size,
        kernel.is_persistent,
        getattr(kernel, "pack_gqa", None),
        getattr(kernel, "cat_gqa", None),
        getattr(kernel, "range_merge", None),
    )


class TestFfaSoftcap(DistTestBase):
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

    def _fwd_bwd(self, q, k, v, softcap, **kwargs):
        """fwd + bwd through ``flex_flash_attn_func``; returns out, the fwd
        meta, the grads and the kernels built."""
        q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
        with _record_kernels() as built:
            out, meta = flex_flash_attn_func(q, k, v, softcap=softcap, **kwargs)
            do = torch.randn_like(out)
            grads = torch.autograd.grad(out, (q, k, v), do)
        return out, meta, do, grads, built

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
    ) -> None:
        """out/dq/dk/dv (packed thd) within ``_NORM_RATIO`` of the
        low-precision torch reference error, both against fp64."""
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
            out_ref, _ = ref_attn_func(
                q=qkv[0],
                k=qkv[1],
                v=qkv[2],
                mask=mask,
                sink=sink,
                softcap=_SOFTCAP,
                layout="thd",
                backend="torch",
                high_precision=high_precision,
            )
            return (out_ref, *torch.autograd.grad(out_ref, qkv, do))

        refs_hi, refs_lo = ref(True), ref(False)
        for name, actual, hi, lo in zip(
            ("out", "dq", "dk", "dv"), (out, *grads), refs_hi, refs_lo
        ):
            err = calc_inf_norm(actual.to(hi.dtype), hi)
            bound = _NORM_RATIO * calc_inf_norm(lo, hi) + 1e-4 * hi.abs().max().item()
            self.assertLessEqual(err, bound, f"{test_case} {name}: {err=} {bound=}")

    def _assert_native(self, built: dict[str, list], test_case: str) -> None:
        for kernel in (*built["fwd"], *built["bwd"]):
            self.assertTrue(kernel.has_softcap, test_case)
            self.assertIsNone(kernel.score_mod, test_case)

    # ─────────────────────────────────────────────────────────────────────
    # Dense
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    @parameterize("mha_type", ["mha", "gqa"])
    @parameterize("mask_type", [MT_MAP.full, MT_MAP.causal])
    @parameterize("d", [64, 128, 192])
    def test_dense(self, d, mask_type, mha_type):
        """Dense fwd/bwd with softcap match the torch reference and keep the
        kernel configuration (incl. 2-CTA at head_dim 128/192) of the
        uncapped call."""
        if get_device_arch()[1] not in (10, 11):
            return
        device, dtype = self.device, torch.bfloat16
        batch, seqlen, nheads = 2, 1000, 8
        nheads_kv = {"mha": nheads, "gqa": 2}[mha_type]
        d_v = 128 if d == 192 else d
        torch.random.manual_seed(self.seed + d + mask_type + nheads_kv)
        # Scale q so the scores reach the cap.
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

    # ─────────────────────────────────────────────────────────────────────
    # Ranges
    # ─────────────────────────────────────────────────────────────────────

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
        if get_device_arch()[1] not in (10, 11):
            return
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
        if get_device_arch()[1] not in (10, 11):
            return
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
        self.assertTrue(all(x.cat_gqa for x in built["bwd"]), test_case)
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
        if get_device_arch()[1] not in (10, 11):
            return
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
        # Capped logits are bounded by the cap and some of them reach it.
        self.assertTrue(torch.all(meta.max_logits <= _SOFTCAP), test_case)
        torch.testing.assert_close(
            meta.max_logits, meta_ref.max_logits.float(), atol=2e-2, rtol=1e-3
        )

    # ─────────────────────────────────────────────────────────────────────
    # Contract
    # ─────────────────────────────────────────────────────────────────────

    @with_run_in_mp
    def test_zero_softcap_builds_the_uncapped_kernel(self):
        """``softcap=0.0`` means no cap: the uncapped kernels run and the
        results match the default call bitwise."""
        if get_device_arch()[1] not in (10, 11):
            return
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
        if get_device_arch()[1] not in (10, 11):
            return
        q, k, v, _, _, _, kwargs = self._ranges_inputs(8, 64, self.seed)

        def score_mod(score, b, h, q_idx, kv_idx, aux_tensors):
            return score

        with self.assertRaises((AssertionError, NotImplementedError)):
            flex_flash_attn_func(
                q,
                k,
                v,
                softcap=_SOFTCAP,
                flex_attn_args=TorchFlexAttnArgs(score_mod=score_mod),
                **kwargs,
            )


if __name__ == "__main__":
    run_tests()
