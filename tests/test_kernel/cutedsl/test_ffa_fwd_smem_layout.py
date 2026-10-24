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

"""sScale offset in the SharedStorage of the SM100 CuTeDSL FFA fwd kernel.

The struct size cannot detect sScale moving back to the front, so the test
pins its offset. Each case builds one specialization up to its kernel launch;
no kernel compiles.
"""

import itertools
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import (
    MT_MAP,
    get_device_arch,
    is_ffa_2cta_disabled,
)
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd

# Smallest sLSE struct offset measured fast for the hd64 1-CTA bwd (129024 B
# is measured slow). The fwd sScale was measured only at the two ends:
# offsets up to 2428 B are slow and offsets from 214016 B up are fast.
_MIN_FAST_SSCALE_OFFSET = 130048

_DISABLE_2CTA_ENV = "MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA"
# 2-CTA needs more than two 128-row q tiles per relation.
_SEQLEN = 512
_BATCH = 2
_NUM_HEAD = 4


@dataclass(frozen=True)
class _Variant:
    head_dim: int
    # "batched" passes [b, s, h, d] inputs; "scalar" passes q/k ranges with
    # one mask type; "per_range" passes one mask type per range (full, causal).
    form: str
    causal: bool
    direct: bool
    sm_margin: int = 0
    deterministic: bool = False
    num_head_kv: int = _NUM_HEAD
    force_one_cta: bool = False

    @property
    def head_dim_v(self) -> int:
        return 128 if self.head_dim == 192 else self.head_dim

    @property
    def two_cta_candidate(self) -> bool:
        """The host builds 2-CTA for this input unless 2-CTA is disabled.

        Causal and per-range masks, the atomic merge, hd64, and packed GQA on
        q/k ranges build 1-CTA.
        """
        return (
            self.head_dim in (128, 192)
            and not self.causal
            and self.form != "per_range"
            and self.direct
            and (self.form == "batched" or self.num_head_kv == _NUM_HEAD)
        )

    @property
    def overlap_sO_sQ(self) -> bool:
        return self.head_dim == 192

    @property
    def sO_borrow_kv(self) -> bool:
        # The 1-CTA range atomic merge stages fp32 O in finished K/V slots.
        return self.form != "batched" and not self.direct and not self.overlap_sO_sQ

    @property
    def persistent(self) -> bool:
        if self.overlap_sO_sQ:
            return False
        if self.form == "batched":
            return not self.causal
        return False

    @property
    def pack_gqa(self) -> bool:
        return self.num_head_kv != _NUM_HEAD


def _exclusion(v: _Variant) -> str | None:
    """Why a point of the product is not built, or None when it is."""
    if v.form == "batched" and not v.direct:
        return "dense inputs always take the direct store"
    if v.form == "batched" and v.sm_margin > 0:
        return "sm_margin > 0 requires q/k ranges and raises on dense inputs"
    if v.deterministic and v.direct:
        return "deterministic orders only the atomic merge; same kernel as direct"
    if v.deterministic and v.head_dim == 192:
        return "deterministic requires head_dim == head_dim_v in (64, 128) and raises"
    if v.form == "per_range" and v.causal:
        return "per_range mixes full and causal; the causal flag is not a separate case"
    if v.force_one_cta and not v.two_cta_candidate:
        return "the host already builds 1-CTA for this input"
    return None


def _matrix() -> list[tuple[_Variant, str | None]]:
    rows: list[tuple[_Variant, str | None]] = []
    for hd, form, causal, direct, sm_margin, det, one_cta in itertools.product(
        (64, 128, 192),
        ("batched", "scalar", "per_range"),
        (False, True),
        (True, False),
        (0,),
        (False,),
        (False, True),
    ):
        variant = _Variant(
            head_dim=hd,
            form=form,
            causal=causal,
            direct=direct,
            sm_margin=sm_margin,
            deterministic=det,
            force_one_cta=one_cta,
        )
        rows.append((variant, _exclusion(variant)))
    # One packed GQA row per head dim.
    for hd in (64, 128, 192):
        variant = _Variant(
            head_dim=hd,
            form="scalar",
            causal=False,
            direct=True,
            num_head_kv=_NUM_HEAD // 2,
        )
        rows.append((variant, None))
    return rows


class _LaunchReached(Exception):
    pass


@contextmanager
def _record_fwd_kernel_at_launch() -> Iterator[list[FFAFwdSm100]]:
    """Record the SM100 fwd kernel objects and stop each call at its launch.

    The struct is built during the trace, before the launch; stopping there
    skips the kernel compile. A JIT cache hit never traces, so the context
    also swaps in an empty compile cache.
    """
    built: list[FFAFwdSm100] = []
    init = FFAFwdSm100.__init__

    def record(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    with mock.patch.object(FFAFwdSm100, "__init__", record), mock.patch.object(
        FFAFwdSm100, "kernel", side_effect=_LaunchReached
    ), mock.patch.object(_flex_flash_attn_fwd, "compile_cache", JITCache()):
        yield built


class TestFfaFwdSmemLayout(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the SharedStorage under test is the SM100/SM110 fwd")
        torch.manual_seed(42)

    def _build(self, variant: _Variant) -> FFAFwdSm100:
        d, dv, hk = variant.head_dim, variant.head_dim_v, variant.num_head_kv
        dtype, device = torch.bfloat16, "cuda"
        total = _BATCH * _SEQLEN
        mask_type = MT_MAP.causal if variant.causal else MT_MAP.full
        kwargs: dict = {}
        shape_q: tuple[int, ...]
        shape_k: tuple[int, ...]
        shape_v: tuple[int, ...]
        if variant.form == "batched":
            shape_q = (_BATCH, _SEQLEN, _NUM_HEAD, d)
            shape_k = (_BATCH, _SEQLEN, hk, d)
            shape_v = (_BATCH, _SEQLEN, hk, dv)
            kwargs["mask_types"] = mask_type
        else:
            shape_q, shape_k, shape_v = (
                (total, _NUM_HEAD, d),
                (total, hk, d),
                (total, hk, dv),
            )
            ranges = torch.tensor(
                [[b * _SEQLEN, (b + 1) * _SEQLEN] for b in range(_BATCH)],
                dtype=torch.int32,
                device=device,
            )
            if variant.form == "per_range":
                mask_types = torch.tensor(
                    [MT_MAP.full, MT_MAP.causal][:_BATCH],
                    dtype=torch.int32,
                    device=device,
                )
            else:
                mask_types = mask_type
            kwargs.update(
                q_ranges=ranges,
                k_ranges=ranges,
                mask_types=mask_types,
                max_seqlen_q=_SEQLEN,
                max_seqlen_k=_SEQLEN,
                disable_fwd_atomic_reduction=variant.direct,
            )
        q = torch.randn(shape_q, dtype=dtype, device=device)
        k = torch.randn(shape_k, dtype=dtype, device=device)
        v = torch.randn(shape_v, dtype=dtype, device=device)
        # The host reads the variable at call time. Setting it per case makes
        # both the 2-CTA and the forced 1-CTA rows build what they name.
        env = {_DISABLE_2CTA_ENV: "1" if variant.force_one_cta else "0"}
        with mock.patch.dict(os.environ, env):
            # A toolchain without fwd 2-CTA support builds 1-CTA everywhere.
            two_cta = variant.two_cta_candidate and not is_ffa_2cta_disabled(
                is_fwd=True
            )
            with _record_fwd_kernel_at_launch() as built:
                with self.assertRaises(_LaunchReached, msg=str(variant)):
                    _flex_flash_attn_fwd(q, k, v, **kwargs)
        (kernel,) = built
        expected = (2 if two_cta else 1, variant.persistent)
        self.assertEqual(
            (kernel.cta_group_size, kernel.is_persistent), expected, str(variant)
        )
        self.assertEqual(kernel.sO_borrow_kv, variant.sO_borrow_kv, str(variant))
        self.assertEqual(kernel.overlap_sO_sQ, variant.overlap_sO_sQ, str(variant))
        self.assertEqual(kernel.pack_gqa, variant.pack_gqa, str(variant))
        return kernel

    def test_sscale_offset_clears_the_bank_conflict_offset(self):
        """Every fwd struct keeps sScale at or above the smallest offset
        measured fast: hd64/128/192, dense and q/k ranges with scalar and
        per-range masks, direct store and atomic merge, 2-CTA and forced
        1-CTA, and packed GQA."""
        offsets: dict[_Variant, int] = {}
        for variant, reason in _matrix():
            if reason is not None:
                continue
            kernel = self._build(variant)
            offsets[variant] = kernel.shared_storage._offsets["sScale"]
        # The check covers a specialization only when the matrix builds it.
        self.assertEqual({v.persistent for v in offsets}, {False, True})
        self.assertEqual({v.sO_borrow_kv for v in offsets}, {False, True})
        self.assertEqual({v.overlap_sO_sQ for v in offsets}, {False, True})
        self.assertTrue(any(v.force_one_cta for v in offsets))
        low = {v: off for v, off in offsets.items() if off < _MIN_FAST_SSCALE_OFFSET}
        self.assertFalse(
            low,
            f"sScale below {_MIN_FAST_SSCALE_OFFSET} B in {len(low)} of "
            f"{len(offsets)} fwd structs: "
            + "; ".join(f"{v} -> {off}" for v, off in low.items()),
        )


if __name__ == "__main__":
    run_tests()
