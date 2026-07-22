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

"""IndexSparse (token-level scatter) sweep tests for the cutedsl SM100 kernel.

Mirrors the structure of ``tests/test_attn/test_index_sparse.py`` on main, but
drives the kernel-level entry points directly (``_flex_flash_attn_fwd`` /
``_flex_flash_attn_bwd`` + ``prepare_index_sparse_tiles``) since this fork does
not expose the ``index_sparse_indices`` autograd wrapper yet.

Sweeps (all with *random* per-Q-tile token indices):
  * Classic:      seqlen_q x seqlen_kv x topk         (MQA8 + PackGQA, D=128)
  * Head config:  MHA / GQA / MQA  x  pack_gqa on/off
  * Partial topk: topk not a multiple of the 128 N-tile (64 / 192)
  * Batch:        B=2
  * Head dim:     D=64
  * Unaligned:    seqlen_q not a multiple of the M-tile

Scatter store modes (bulk cp.async vs per-element atomicAdd) are selected via
MAGI_ATTENTION_FFA_CUTEDSL_IS_SCATTER_ATOMIC at kernel-compile time, which is
NOT part of the JIT cache key — so each mode must run in its own process:

    pytest tests/test_kernel/cutedsl/test_ffa_index_sparse.py -v
    MAGI_ATTENTION_FFA_CUTEDSL_IS_SCATTER_ATOMIC=1 \
        pytest tests/test_kernel/cutedsl/test_ffa_index_sparse.py -v

Known constraints (asserted in the kernel):
  * BWD IndexSparse requires swap_bwd_qk_loop=True (LoopK) and head_dim <= 128
  * All queries inside one FWD sparse Q-tile must share the same token set
    (prepare_index_sparse_tiles takes the tile's first query as representative)
"""

import os

import pytest
import torch

from magi_attention.kernel.cutedsl.ffa_utils import TorchFlexAttnArgs
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.sparse_utils import prepare_index_sparse_tiles

SEED = 42

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="IndexSparse cutedsl path requires SM100+",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fwd_sparse_m_block(seqlen_q: int, qhpk: int, pack_gqa: bool) -> int:
    """FWD sparse Q-tile rows: q_stage=2 doubles the 128-row tile when the
    packed Q extent exceeds one tile (mirrors _flex_flash_attn_fwd)."""
    seqlen_q_packgqa = seqlen_q * qhpk if pack_gqa else seqlen_q
    return 256 if seqlen_q_packgqa > 128 else 128


def _build_block_random_indices(
    B: int,
    NHQ: int,
    NHK: int,
    SQ: int,
    SK: int,
    topk: int,
    tokens_per_block: int,
    device: str,
) -> torch.Tensor:
    """Random sorted token indices, shared by all queries within one index
    block and by all Q heads within one KV group.

    Returns (B, NHQ, SQ, topk) int32.
    """
    qhpk = NHQ // NHK
    num_blocks = (SQ + tokens_per_block - 1) // tokens_per_block
    picks = torch.stack(
        [torch.randperm(SK, device=device)[:topk] for _ in range(B * NHK * num_blocks)]
    ).view(B, NHK, num_blocks, topk)
    picks = picks.sort(dim=-1).values.int()
    # expand KV-head blocks to Q heads and per-token rows
    q_to_block = torch.arange(SQ, device=device) // tokens_per_block
    per_token = picks[:, :, q_to_block, :]  # (B, NHK, SQ, topk)
    indices = (
        per_token.unsqueeze(2)
        .expand(B, NHK, qhpk, SQ, topk)
        .reshape(B, NHQ, SQ, topk)
        .contiguous()
    )
    return indices


def _sdpa_ref_fwd_bwd(q, k, v, dO, indices, softmax_scale):
    """Vectorized fp32 SDPA reference with token-level index mask.

    indices: (B, NHQ, SQ, topk) local KV token ids (all valid, >= 0).
    Returns (O, dQ, dK, dV) in bf16.
    """
    B, SQ, NHQ, HD = q.shape
    _, SK, NHK, _ = k.shape
    HDV = v.shape[-1]
    qhpk = NHQ // NHK

    q_f = q.float().detach().requires_grad_(True)
    k_f = k.float().detach().requires_grad_(True)
    v_f = v.float().detach().requires_grad_(True)

    k_exp = k_f.unsqueeze(3).expand(B, SK, NHK, qhpk, HD).reshape(B, SK, NHQ, HD)
    v_exp = v_f.unsqueeze(3).expand(B, SK, NHK, qhpk, HDV).reshape(B, SK, NHQ, HDV)

    scores = torch.einsum("bsnh,btnh->bnst", q_f, k_exp) * softmax_scale

    mask = torch.zeros(B, NHQ, SQ, SK, dtype=torch.bool, device=q.device)
    mask.scatter_(3, indices.long(), True)

    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bnst,btnh->bsnh", probs, v_exp)
    out.backward(dO.float())

    return (
        out.bfloat16(),
        q_f.grad.bfloat16(),
        k_f.grad.bfloat16(),
        v_f.grad.bfloat16(),
    )


def _check(name, got, ref, errors):
    diff = (got.float() - ref.float()).abs()
    mean_rel = (diff / ref.float().abs().clamp(min=1e-6)).mean().item()
    max_abs = diff.max().item()
    cosine = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    rel_limit = 0.05 if name in ("dK", "dV") else 0.02
    abs_limit = 1.0 if name in ("dK", "dV") else 0.5
    ok = mean_rel < rel_limit and max_abs < abs_limit and cosine >= 0.999
    line = (
        f"{name}: mean_rel={mean_rel:.6f} max_abs={max_abs:.4f} "
        f"cosine={cosine:.6f} [{'PASS' if ok else 'FAIL'}]"
    )
    print(f"  {line}", flush=True)
    if not ok:
        errors.append(line)


def _run_index_sparse_case(
    *,
    B: int = 1,
    NHQ: int,
    NHK: int,
    D: int = 128,
    SQ: int,
    SK: int,
    topk: int,
    pack_gqa: bool,
    device: str = "cuda",
):
    torch.manual_seed(SEED)
    qhpk = NHQ // NHK
    softmax_scale = D**-0.5

    q = torch.randn(B, SQ, NHQ, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, SK, NHK, D, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dO = torch.randn_like(q)

    fwd_m_block = _fwd_sparse_m_block(SQ, qhpk, pack_gqa)
    # Indices must be uniform at the coarsest tile granularity (FWD tile);
    # the 128-row BWD LoopK tiles are then automatically uniform too.
    tokens_per_block = max(fwd_m_block // (qhpk if pack_gqa else 1), 1)
    indices = _build_block_random_indices(
        B, NHQ, NHK, SQ, SK, topk, tokens_per_block, device
    )

    o_ref, dq_ref, dk_ref, dv_ref = _sdpa_ref_fwd_bwd(
        q, k, v, dO, indices, softmax_scale
    )

    fwd_tiles = prepare_index_sparse_tiles(
        indices,
        batch_size=B,
        seqlen_q=SQ,
        seqlen_k=SK,
        num_kv_heads=NHK,
        num_q_heads=NHQ,
        m_block_size=fwd_m_block,
        n_block_size=128,
        pack_gqa=pack_gqa,
    )
    out, lse = _flex_flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        flex_attn_args=TorchFlexAttnArgs(index_sparse_tiles=fwd_tiles),
        pack_gqa=pack_gqa,
    )

    bwd_tiles = prepare_index_sparse_tiles(
        indices,
        batch_size=B,
        seqlen_q=SQ,
        seqlen_k=SK,
        num_kv_heads=NHK,
        num_q_heads=NHQ,
        m_block_size=128,  # LoopK BWD native packed-M tile
        n_block_size=128,
        pack_gqa=pack_gqa,
    )
    dq, dk, dv = _flex_flash_attn_bwd(
        q,
        k,
        v,
        out,
        lse,
        dO,
        softmax_scale=softmax_scale,
        flex_attn_args=TorchFlexAttnArgs(index_sparse_tiles=bwd_tiles),
        swap_bwd_qk_loop=True,
        pack_gqa=pack_gqa,
    )

    errors: list[str] = []
    _check("O", out, o_ref, errors)
    _check("dQ", dq, dq_ref, errors)
    _check("dK", dk, dk_ref, errors)
    _check("dV", dv, dv_ref, errors)
    assert not errors, (
        f"[B={B},NHQ={NHQ},NHK={NHK},D={D},SQ={SQ},SK={SK},topk={topk},"
        f"pack_gqa={pack_gqa},atomic="
        f"{os.environ.get('MAGI_ATTENTION_FFA_CUTEDSL_IS_SCATTER_ATOMIC', '0')}]\n"
        + "\n".join(errors)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Classic sweep: seqlen_q x seqlen_kv x topk (MQA8 + PackGQA, D=128)
# ─────────────────────────────────────────────────────────────────────────────


@requires_sm100
@pytest.mark.parametrize("SQ", [128, 512])
@pytest.mark.parametrize("SK", [512, 2048])
@pytest.mark.parametrize("topk", [128, 256])
def test_classic_sweep(SQ, SK, topk):
    _run_index_sparse_case(NHQ=8, NHK=1, SQ=SQ, SK=SK, topk=topk, pack_gqa=True)


@requires_sm100
def test_long_seq():
    _run_index_sparse_case(NHQ=8, NHK=1, SQ=1024, SK=4096, topk=384, pack_gqa=True)


# ─────────────────────────────────────────────────────────────────────────────
# Head-config sweep: MHA / GQA / MQA x pack_gqa (SQ=256, SK=1024, topk=256)
# ─────────────────────────────────────────────────────────────────────────────


@requires_sm100
@pytest.mark.parametrize(
    "NHQ,NHK,pack_gqa",
    [
        (1, 1, False),  # single head
        (8, 1, False),  # MQA, no packing
        (8, 1, True),  # MQA + PackGQA
        (16, 2, True),  # GQA + PackGQA
        (4, 4, False),  # MHA
    ],
)
def test_head_configs(NHQ, NHK, pack_gqa):
    _run_index_sparse_case(
        NHQ=NHQ, NHK=NHK, SQ=256, SK=1024, topk=256, pack_gqa=pack_gqa
    )


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases: partial topk / batch / head dim / unaligned seqlen_q
# ─────────────────────────────────────────────────────────────────────────────


@requires_sm100
@pytest.mark.parametrize("topk", [64])
def test_partial_topk(topk):
    """topk not a multiple of the 128 N-tile — exercises is_valid_total tails."""
    _run_index_sparse_case(NHQ=8, NHK=1, SQ=256, SK=512, topk=topk, pack_gqa=True)


@requires_sm100
@pytest.mark.parametrize("topk", [192])
@pytest.mark.xfail(
    reason="Pre-existing bug: softmax_block_sparse_sm100 seqlen_k masking fails "
    "when mask_block_cnt > 1 and last block is partial (topk % 128 != 0). "
    "Dense multi-N-iter and single-block partial both work; issue is specific "
    "to multi-iter correction pipeline in block-sparse softmax.",
    strict=True,
)
def test_partial_topk_multi_niter_xfail(topk):
    """Known issue: partial last N-iter when topk > n_block_size."""
    _run_index_sparse_case(NHQ=8, NHK=1, SQ=256, SK=512, topk=topk, pack_gqa=True)


@requires_sm100
def test_batch2():
    _run_index_sparse_case(B=2, NHQ=8, NHK=1, SQ=256, SK=512, topk=128, pack_gqa=True)


@requires_sm100
def test_head_dim_64():
    _run_index_sparse_case(NHQ=8, NHK=1, D=64, SQ=256, SK=512, topk=128, pack_gqa=True)


@requires_sm100
def test_unaligned_seqlen_q():
    _run_index_sparse_case(NHQ=1, NHK=1, SQ=192, SK=512, topk=128, pack_gqa=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
