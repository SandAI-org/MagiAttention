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

"""Smoke-test suite for the forked cutedsl kernel.

Covers the simplest training-relevant cases so that after each round of
changes we can quickly verify correctness has not regressed:

  * Non-varlen fwd+bwd: full / causal  x  MHA / GQA / MQA
  * Varlen (packed cu_seqlens) fwd+bwd: full / causal  x  MHA / GQA / MQA

Run:
    pytest tests/test_kernel/cutedsl/test_ffa_simple.py -v
"""

import random
from contextlib import contextmanager

import pytest
import torch
from einops import rearrange

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, get_device_arch
from magi_attention.testing import assert_close, ref_attn_func
from magi_attention.testing.utils import switch_envvars
from magi_attention.utils import make_attn_mask_from_ffa_args
from magi_attention.utils.arch import is_ampere

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


# Tolerance formula copied from test_flash_attn_fast.py: account for rounding
# errors in the reference itself.
def _fwd_atol(out_ref, out_pt):
    return (
        2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        + 2 * (out_pt - out_ref).abs().max().item()
    )


def _bwd_atol(grad_ref, grad_pt):
    return (
        2 * (grad_ref + 0.3 - 0.3 - grad_ref).abs().max().item()
        + 2 * (grad_pt - grad_ref).abs().max().item()
    )


def _ref_attn_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask_types: int,
    high_precision: bool,
) -> torch.Tensor:
    """Block-diagonal reference attention over a batched ``(b, s, h, d)`` tensor.

    Each batch element attends only within itself (matching the per-batch
    semantics of the kernel), with ``mask_types`` selecting full (0) / causal
    (1) within each block. Computed via ``ref_attn_func`` on the flattened
    ``(total, h, d)`` layout, then reshaped back to ``(b, s, h, d)``.

    The autograd graph is preserved end-to-end, so gradients flow back to the
    input batched tensors.
    """
    b, sq = q.shape[0], q.shape[1]
    sk = k.shape[1]

    q_thd = rearrange(q, "b s h d -> (b s) h d")
    k_thd = rearrange(k, "b s h d -> (b s) h d")
    v_thd = rearrange(v, "b s h d -> (b s) h d")

    q_ranges = AttnRanges.from_ranges([[i * sq, (i + 1) * sq] for i in range(b)])
    k_ranges = AttnRanges.from_ranges([[i * sk, (i + 1) * sk] for i in range(b)])
    mask = make_attn_mask_from_ffa_args(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=[mask_types] * b,
        total_seqlen_q=b * sq,
        total_seqlen_k=b * sk,
        device=q.device,
    )

    out_thd, _ = ref_attn_func(
        q=q_thd,
        k=k_thd,
        v=v_thd,
        mask=mask,
        layout="thd",
        backend="sdpa",
        high_precision=high_precision,
    )

    return rearrange(out_thd, "(b s) h d -> b s h d", b=b)


# ─────────────────────────────────────────────────────────────────────────────
# SM80 kernel selection
# ─────────────────────────────────────────────────────────────────────────────
#
# The SM80 kernel path is selected via the MAGI_ATTENTION_FFA_CUTEDSL_ARCH override
# rather than the real device capability, so it can be exercised on newer GPUs
# (the compiled SM80 SASS runs fine on sm90/sm100). get_device_arch() is
# lru_cached, so we must clear the cache whenever we toggle the override.


@contextmanager
def _maybe_force_sm80(enabled: bool):
    """Force the FFA kernel path to SM80 within the context when ``enabled``."""
    if not enabled:
        yield
        return

    switch_back = switch_envvars(
        ["MAGI_ATTENTION_FFA_CUTEDSL_ARCH"],
        enable_value_dict={"MAGI_ATTENTION_FFA_CUTEDSL_ARCH": "sm_80"},
    )
    get_device_arch.cache_clear()
    try:
        yield
    finally:
        switch_back()
        get_device_arch.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# Non-varlen: fwd + bwd
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("mask_types", [MT_MAP.full, MT_MAP.causal])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("force_sm80", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (256, 256),
        (1024, 1024),
        (203, 123),
    ],
)
def test_non_varlen_fwd_bwd(
    seqlen_q, seqlen_k, force_sm80, d, mask_types, mha_type, dtype
):
    """Non-varlen flex_flash_attn_func: fwd + bwd for full/causal x MHA/GQA/MQA."""
    if force_sm80 and is_ampere():
        pytest.skip(
            "No need to force SM80 on Ampere+ hardware, the kernel path is selected automatically"
        )

    device = "cuda"
    seed = seqlen_q + seqlen_k + d + mask_types * 3
    torch.random.manual_seed(seed)
    random.seed(seed)

    batch_size = 4
    nheads = 6
    nheads_kv = {"mha": nheads, "gqa": 3, "mqa": 1}[mha_type]

    q_ref = torch.randn(
        batch_size, seqlen_q, nheads, d, device=device, dtype=dtype
    ).requires_grad_()
    k_ref = torch.randn(
        batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype
    ).requires_grad_()
    v_ref = torch.randn(
        batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype
    ).requires_grad_()

    q = q_ref.detach().requires_grad_()
    k = k_ref.detach().requires_grad_()
    v = v_ref.detach().requires_grad_()

    out_ref = _ref_attn_batched(
        q_ref, k_ref, v_ref, mask_types=mask_types, high_precision=True
    )
    out_pt = _ref_attn_batched(
        q_ref, k_ref, v_ref, mask_types=mask_types, high_precision=False
    )

    with _maybe_force_sm80(force_sm80):
        out, _ = flex_flash_attn_func(q, k, v, mask_types=mask_types)

        atol = _fwd_atol(out_ref, out_pt)
        assert_close(
            out,
            out_ref,
            atol=atol,
            rtol=0,
            mismatch_threshold=1e-5,
            test_case=f"{force_sm80=},{seqlen_q=},{seqlen_k=},{d=},{mask_types=},{mha_type=},{dtype=} => fwd",
        )

        # ── backward ──
        g = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)

    errors = []
    for tensor, ref, pt, name in [
        (dq, dq_ref, dq_pt, "dQ"),
        (dk, dk_ref, dk_pt, "dK"),
        (dv, dv_ref, dv_pt, "dV"),
    ]:
        try:
            assert_close(
                tensor,
                ref,
                atol=_bwd_atol(ref, pt),
                rtol=0,
                mismatch_threshold=1e-5,
                test_case=f"{force_sm80=},{seqlen_q=},{seqlen_k=},{d=},{mask_types=},{mha_type=},{dtype=} => {name}",
            )
        except AssertionError as e:
            errors.append(str(e))
    if errors:
        raise AssertionError("\n\n".join(errors))


# ─────────────────────────────────────────────────────────────────────────────
# Varlen (packed, q/k ranges): fwd + bwd
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("mask_types", [MT_MAP.full, MT_MAP.causal])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("force_sm80", [False, True])
@pytest.mark.parametrize("seqlen", [128, 512, 1024])
def test_varlen_fwd_bwd(seqlen, force_sm80, d, mask_types, mha_type, dtype):
    """Varlen flex_flash_attn_func (packed q/k ranges): fwd + bwd."""
    if force_sm80:
        if is_ampere():
            pytest.skip(
                "No need to force SM80 on Ampere+ hardware, the kernel path is selected automatically"
            )
        else:
            # FIXME(sm80-varlen): forcing the SM80 path for varlen crashes
            # with an illegal memory access when the arch override is toggled mid-process
            # on hardware with higher capability.
            pytest.skip("SM80-forced varlen crashes under mid-process arch")

    device = "cuda"
    seed = seqlen + d + mask_types * 5
    torch.random.manual_seed(seed)
    random.seed(seed)

    batch_size = 8
    nheads = 6
    nheads_kv = {"mha": nheads, "gqa": 3, "mqa": 1}[mha_type]

    q_ref = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype
    ).requires_grad_()
    k_ref = torch.randn(
        batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype
    ).requires_grad_()
    v_ref = torch.randn(
        batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype
    ).requires_grad_()

    out_ref = _ref_attn_batched(
        q_ref, k_ref, v_ref, mask_types=mask_types, high_precision=True
    )
    out_pt = _ref_attn_batched(
        q_ref, k_ref, v_ref, mask_types=mask_types, high_precision=False
    )

    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32
    )
    # q/k ranges equivalent to the cu_seqlens partition: [[0, s], [s, 2s], ...]
    q_ranges = torch.stack([cu_seqlens[:-1], cu_seqlens[1:]], dim=1)
    k_ranges = q_ranges.clone()
    q_v = rearrange(q_ref.detach(), "b s h d -> (b s) h d").requires_grad_()
    k_v = rearrange(k_ref.detach(), "b s h d -> (b s) h d").requires_grad_()
    v_v = rearrange(v_ref.detach(), "b s h d -> (b s) h d").requires_grad_()

    with _maybe_force_sm80(force_sm80):
        out_v, _ = flex_flash_attn_func(
            q_v,
            k_v,
            v_v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=mask_types,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
        )

        out_reshaped = rearrange(out_v, "(b s) h d -> b s h d", b=batch_size)
        atol = _fwd_atol(out_ref, out_pt)
        assert_close(
            out_reshaped,
            out_ref,
            atol=atol,
            rtol=0,
            mismatch_threshold=1e-5,
            test_case=f"{force_sm80=},{seqlen=},{d=},{mask_types=},{mha_type=},{dtype=} => varlen fwd",
        )

        # ── backward ──
        g = torch.randn_like(out_v)
        dq_v, dk_v, dv_v = torch.autograd.grad(out_v, (q_v, k_v, v_v), g)

    g_b = rearrange(g, "(b s) h d -> b s h d", b=batch_size)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g_b)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g_b)

    errors = []
    for tensor, ref, pt, name in [
        (dq_v, dq_ref, dq_pt, "dQ"),
        (dk_v, dk_ref, dk_pt, "dK"),
        (dv_v, dv_ref, dv_pt, "dV"),
    ]:
        ref_thd = rearrange(ref, "b s h d -> (b s) h d")
        pt_thd = rearrange(pt, "b s h d -> (b s) h d")
        try:
            assert_close(
                tensor,
                ref_thd,
                atol=_bwd_atol(ref_thd, pt_thd),
                rtol=0,
                mismatch_threshold=1e-5,
                test_case=f"{force_sm80=},{seqlen=},{d=},{mask_types=},{mha_type=},{dtype=} => varlen {name}",
            )
        except AssertionError as e:
            errors.append(str(e))
    if errors:
        raise AssertionError("\n\n".join(errors))
