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

"""sLSE offset in the SharedStorage of the SM100 CuTeDSL FFA bwd kernel.

sLSE must stay at or above 127 KiB from the struct start in every dK/dV mode
(the sdO size differs between them). Each case builds one specialization up to
its kernel launch; no kernel compiles.
"""

import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_bwd

# Smallest sLSE struct offset measured fast for the hd64 1-CTA bwd; 129024 B
# is measured slow.
_MIN_FAST_SLSE_OFFSET = 130048

_SEQLEN = 256
_BATCH = 2
_NUM_HEAD = 4


@dataclass(frozen=True)
class _Variant:
    head_dim: int
    # "batched" passes [b, s, h, d] inputs; "scalar" and "per_range" pass
    # q/k ranges with one mask type or one mask type per range.
    form: str
    mask_type: int
    num_head_kv: int = _NUM_HEAD
    disable_bwd_dkv_atomic_reduction: bool = False
    disable_fwd_atomic_reduction: bool = False
    deterministic: bool = False
    sm_margin: int = 0

    @property
    def single_writer_dkv(self) -> bool:
        if self.form == "batched":
            return self.num_head_kv == _NUM_HEAD
        return self.disable_bwd_dkv_atomic_reduction

    @property
    def persistent(self) -> bool:
        has_ranges = self.form != "batched"
        return self.sm_margin > 0 or (self.deterministic and has_ranges)


def _one_cta_variants() -> list[_Variant]:
    variants = [
        _Variant(
            head_dim=64,
            form=form,
            mask_type=mask_type,
            disable_bwd_dkv_atomic_reduction=dkv,
            disable_fwd_atomic_reduction=dq,
            deterministic=det,
            sm_margin=sm_margin,
        )
        for form, mask_type, dkv, dq, det, sm_margin in itertools.product(
            ("scalar", "per_range"),
            (MT_MAP.full, MT_MAP.causal),
            (False, True),
            (False, True),
            (False,),
            (0,),
        )
    ]
    variants += [
        _Variant(head_dim=64, form="batched", mask_type=mask_type, num_head_kv=hk)
        for mask_type, hk in itertools.product(
            (MT_MAP.full, MT_MAP.causal), (_NUM_HEAD, _NUM_HEAD // 2)
        )
    ]
    return variants


def _two_cta_variants() -> list[_Variant]:
    return [
        _Variant(
            head_dim=128,
            form="scalar",
            mask_type=MT_MAP.full,
            disable_bwd_dkv_atomic_reduction=single_writer,
            disable_fwd_atomic_reduction=single_writer,
        )
        for single_writer in (False, True)
    ]


class _LaunchReached(Exception):
    pass


@contextmanager
def _record_bwd_kernel_at_launch() -> Iterator[list[FFABwdSm100]]:
    """Record the SM100 bwd kernel objects and stop each call at its launch.

    The struct is built during the trace, before the launch; stopping there
    skips the kernel compile. A JIT cache hit never traces, so the context
    also swaps in an empty compile cache.
    """
    built: list[FFABwdSm100] = []
    init = FFABwdSm100.__init__

    def record(obj, *args, **kwargs):
        init(obj, *args, **kwargs)
        built.append(obj)

    with mock.patch.object(FFABwdSm100, "__init__", record), mock.patch.object(
        FFABwdSm100, "kernel", side_effect=_LaunchReached
    ), mock.patch.object(_flex_flash_attn_bwd, "compile_cache", JITCache()):
        yield built


class TestFfaBwdSmemLayout(TestCase):
    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the SharedStorage under test is the SM100/SM110 bwd")
        torch.manual_seed(42)

    def _build(self, variant: _Variant) -> FFABwdSm100:
        d, hk = variant.head_dim, variant.num_head_kv
        dtype, device = torch.bfloat16, "cuda"
        total = _BATCH * _SEQLEN
        kwargs: dict = {}
        shape_q: tuple[int, ...]
        shape_k: tuple[int, ...]
        if variant.form == "batched":
            shape_q = (_BATCH, _SEQLEN, _NUM_HEAD, d)
            shape_k = (_BATCH, _SEQLEN, hk, d)
            lse = torch.zeros(_BATCH, _NUM_HEAD, _SEQLEN, device=device)
            kwargs["mask_types"] = variant.mask_type
        else:
            shape_q, shape_k = (total, _NUM_HEAD, d), (total, hk, d)
            lse = torch.zeros(total, _NUM_HEAD, device=device)
            ranges = torch.tensor(
                [[b * _SEQLEN, (b + 1) * _SEQLEN] for b in range(_BATCH)],
                dtype=torch.int32,
                device=device,
            )
            if variant.form == "per_range":
                mask_types = torch.tensor(
                    [variant.mask_type, MT_MAP.full][:_BATCH],
                    dtype=torch.int32,
                    device=device,
                )
            else:
                mask_types = variant.mask_type
            kwargs.update(
                q_ranges=ranges,
                k_ranges=ranges,
                mask_types=mask_types,
                max_seqlen_q=_SEQLEN,
                max_seqlen_k=_SEQLEN,
                disable_bwd_dkv_atomic_reduction=(
                    variant.disable_bwd_dkv_atomic_reduction
                ),
                disable_fwd_atomic_reduction=variant.disable_fwd_atomic_reduction,
            )
        q = torch.randn(shape_q, dtype=dtype, device=device)
        k = torch.randn(shape_k, dtype=dtype, device=device)
        v = torch.randn(shape_k, dtype=dtype, device=device)
        with _record_bwd_kernel_at_launch() as built:
            with self.assertRaises(_LaunchReached, msg=str(variant)):
                _flex_flash_attn_bwd(q, k, v, q, lse, q, **kwargs)
        (kernel,) = built
        self.assertEqual(
            kernel.dKV_postprocess, not variant.single_writer_dkv, str(variant)
        )
        self.assertEqual(kernel.is_persistent, variant.persistent, str(variant))
        return kernel

    def test_one_cta_slse_offset_at_least_127kib(self):
        """hd64 1-CTA bwd: sLSE >= 127 KiB in both dK/dV modes, both dQ modes,
        scalar and per-range masks."""
        variants = _one_cta_variants()
        offsets: dict[_Variant, int] = {}
        for variant in variants:
            kernel = self._build(variant)
            self.assertEqual(kernel.cta_group_size, 1, str(variant))
            offsets[variant] = kernel.shared_storage._offsets["sLSE"]
        # Both sdO sizes must be in the matrix for the check to cover them.
        self.assertEqual(
            {v.single_writer_dkv for v in variants}, {False, True}, "dK/dV modes"
        )
        low = {v: off for v, off in offsets.items() if off < _MIN_FAST_SLSE_OFFSET}
        self.assertFalse(
            low,
            f"sLSE below {_MIN_FAST_SLSE_OFFSET} B in {len(low)} of "
            f"{len(variants)} 1-CTA bwd structs: "
            + "; ".join(f"{v} -> {off}" for v, off in low.items()),
        )

    def test_two_cta_slse_offset_at_least_127kib(self):
        """hd128 2-CTA bwd: sLSE >= 127 KiB in both dK/dV modes."""
        for variant in _two_cta_variants():
            kernel = self._build(variant)
            self.assertEqual(kernel.cta_group_size, 2, str(variant))
            offset = kernel.shared_storage._offsets["sLSE"]
            self.assertGreaterEqual(offset, _MIN_FAST_SLSE_OFFSET, str(variant))


if __name__ == "__main__":
    run_tests()
