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

"""
Can ``ffa_bwd_sm100_index`` replace the sm90 IndexSparse backward?

Replays ``tests/test_attn/test_index_sparse.py`` on SM100: same configs, same
data (seed, q/k/v, ``build_index_sparse_indices``), same SDPA reference and
tolerances. The backward runs ``ffa_bwd_sm100_index`` through
``ffa_bwd_sm100_index_sm90_layout``, which takes the sm90 tensors unchanged.
SM100 has no IndexSparse forward, so out / lse come from the fp32 gather
``index_sparse_ref_fwd``, itself checked against SDPA with the sm90 forward
tolerance.

Every case ends as one of:
  PASS — dq / dk / dv match SDPA within the sm90 tolerances
  GAP  — the case needs an ``Sm100IndexGap`` (nothing is launched)
  FAIL — the comparison failed or the kernel raised

Each test method runs all of its cases, prints the table, then fails on any
FAIL, and also on any GAP unless MAGI_ATTENTION_TEST_SM100_INDEX_ALLOW_GAPS=1
(regression gate for the supported subset).

Structure:
  TestIndexSparseBwdSm100Sweep              — sm90 Classic sweep: q_seqlen × kv_seqlen × topk
  TestIndexSparseBwdSm100ComprehensiveSweep — sm90 Comprehensive sweep: head_config × head_dim × kbs
  TestIndexSparseBwdSm100Harness            — the adapter's multi-kv-head refold, and that the
                                              comparison rejects wrong gradients

Not ported from the sm90 test:
  - inner_dir / inner_load_mode / inner_store_mode / swap_bwd_qk_loop: sm90
    kernel knobs that do not change the function computed; SM100 has one kernel.
  - the deterministic check: it never runs in the sm90 test (its condition
    ``swap_bwd_qk_loop is not True`` fails for every case that does not also
    set ``check_deterministic=False``).
  - TestIndexSparseViewTrick: forward-only checks of a user-side layout trick;
    every view-trick config there has 4 q heads per kv head after folding,
    which the comprehensive MQA4 / GQA cases already cover.

Run (from the repo root):
    pytest tests/test_kernel/cutedsl/test_index_sparse_bwd_sm100.py -s
"""

import itertools
import os
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.utils import set_random_seed
from magi_attention.utils.sparse_utils import (
    build_index_sparse_indices,
    get_sdpa_mask_from_index_sparse_indices,
)
from tests.test_attn.sparse_test_utils import (
    DEFAULT_BWD_DQ_ATOL,
    DEFAULT_FWD_ATOL,
    SEED,
    compare_sdpa_bwd_all,
    compare_sdpa_fwd,
    sdpa_ref_bwd_grads,
    sdpa_ref_output,
)
from tests.test_kernel.cutedsl.index_sparse_sm100_utils import (
    Sm100IndexUnsupportedError,
    ffa_bwd_sm100_index_sm90_layout,
    index_sparse_ref_fwd,
)

ALLOW_GAPS_ENVVAR = "MAGI_ATTENTION_TEST_SM100_INDEX_ALLOW_GAPS"

BwdGrads = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _is_sm100_family() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


# ═══════════════════════════════════════════════════════════
# Case runner
# ═══════════════════════════════════════════════════════════


def _run_index_sparse_config(
    device,
    cfg: dict[str, Any],
    bwd_fn: Callable[..., BwdGrads] = ffa_bwd_sm100_index_sm90_layout,
) -> None:
    """Run one sm90 index_sparse_indices config on SM100 and assert against SDPA.

    Mirrors ``_run_index_sparse_config`` of the sm90 test, with the forward
    replaced by ``index_sparse_ref_fwd`` and the backward by ``bwd_fn``.
    """
    set_random_seed(SEED)
    B = cfg["B"]
    S = cfg.get("S", None)
    S_kv = cfg.get("S_kv", S)
    S_q = cfg.get("S_q", min(S_kv, 256))
    NHQ = cfg["NHQ"]
    NHK = cfg["NHK"]
    D = cfg.get("D", 128)
    topk = cfg["topk"]
    default_max = max(topk) if isinstance(topk, list) else topk
    max_topk = cfg.get("max_topk", default_max)
    sparse_k_block_size = cfg.get("sparse_k_block_size", 1)
    dtype = cfg.get("dtype", torch.bfloat16)
    atol = cfg.get("atol", DEFAULT_FWD_ATOL)
    bwd_atol = cfg.get("bwd_atol", DEFAULT_BWD_DQ_ATOL)
    # The sm90 test flattens kv heads into the sequence; keep_kv_heads feeds
    # the kernel num_heads_kv > 1 instead (sm90 accepts both).
    flatten_kv_heads = not cfg.get("keep_kv_heads", False)

    q = torch.randn(B, S_q, NHQ, D, dtype=dtype, device=device)
    k = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)
    v = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)

    if NHK > 1 and flatten_kv_heads:
        q = rearrange(q, "b s (h1 h2) d -> b (s h1) h2 d", h1=NHK)
        k = rearrange(k, "b s h d -> b (s h) 1 d")
        v = rearrange(v, "b s h d -> b (s h) 1 d")
        S_q = S_q * NHK
        S_kv = S_kv * NHK
        NHQ = NHQ // NHK
        NHK = 1

    index_sparse_indices = build_index_sparse_indices(
        B,
        NHK,
        S_q,
        S_kv,
        topk,
        max_topk,
        device,
        sparse_k_block_size=sparse_k_block_size,
    )

    # Token-major with native heads. With NHK == 1 (every flattened case) this
    # is exactly the sm90 SEQ_MAJOR packing (pack_q_for_ffa / pack_kv_for_ffa),
    # which cannot express NHK > 1.
    q_ffa = rearrange(q, "b s h d -> (b s) h d").contiguous()
    k_ffa = rearrange(k, "b s h d -> (b s) h d").contiguous()
    v_ffa = rearrange(v, "b s h d -> (b s) h d").contiguous()
    out, lse = index_sparse_ref_fwd(
        q_ffa,
        k_ffa,
        v_ffa,
        index_sparse_indices,
        sparse_k_block_size=sparse_k_block_size,
    )
    do = torch.randn_like(out)

    test_case = (
        f"[NHQ={cfg['NHQ']},NHK={cfg['NHK']},S_q={cfg.get('S_q', cfg.get('S'))},"
        f"S_kv={cfg.get('S_kv', cfg.get('S'))},"
        f"B={B},D={D},topk={topk},max_topk={max_topk},"
        f"sparse_k_block_size={sparse_k_block_size},dtype={dtype},"
        f"flat:NHQ_eff={NHQ},NHK_eff={NHK},S_q_eff={S_q},S_kv_eff={S_kv}]"
    )

    # Raises Sm100IndexUnsupportedError before any launch for a gap case.
    dq_ffa, dk_ffa, dv_ffa = bwd_fn(
        q_ffa,
        k_ffa,
        v_ffa,
        out,
        do,
        lse,
        index_sparse_indices,
        sparse_k_block_size=sparse_k_block_size,
    )

    sdpa_mask = get_sdpa_mask_from_index_sparse_indices(
        index_sparse_indices,
        B,
        NHQ,
        NHK,
        S_q,
        S_kv,
        device,
        sparse_k_block_size=sparse_k_block_size,
    )

    # The forward reference feeds the backward: hold it to the sm90 forward bar.
    o_ffa = rearrange(out, "(b s) h d -> b s h d", b=B)
    o_ref = sdpa_ref_output(q, k, v, sdpa_mask, B=B, NHQ=NHQ, NHK=NHK)
    compare_sdpa_fwd(o_ffa, o_ref, atol=atol, test_case=f"{test_case} ref_fwd")

    do_unpacked = rearrange(do, "(b s) h d -> b s h d", b=B)
    sdpa_dq, sdpa_dk, sdpa_dv = sdpa_ref_bwd_grads(
        do_unpacked, q, k, v, sdpa_mask, NHQ=NHQ, NHK=NHK
    )
    dq_ffa = rearrange(dq_ffa, "(b s) h d -> b h s d", b=B)
    dk_ffa = rearrange(dk_ffa, "(b s) h d -> b h s d", b=B)
    dv_ffa = rearrange(dv_ffa, "(b s) h d -> b h s d", b=B)
    compare_sdpa_bwd_all(
        ffa_dq=dq_ffa,
        ffa_dk=dk_ffa,
        ffa_dv=dv_ffa,
        sdpa_dq=sdpa_dq,
        sdpa_dk=sdpa_dk,
        sdpa_dv=sdpa_dv,
        test_case=test_case,
        dq_atol=bwd_atol,
    )


# ═══════════════════════════════════════════════════════════
# Sweep collection and report
# ═══════════════════════════════════════════════════════════


class CaseStatus(Enum):
    PASS = "PASS"
    GAP = "GAP"
    FAIL = "FAIL"


@dataclass
class CaseResult:
    name: str
    status: CaseStatus
    detail: str


def _run_case(device, name: str, cfg: dict[str, Any]) -> CaseResult:
    try:
        _run_index_sparse_config(device, cfg)
    except Sm100IndexUnsupportedError as e:
        return CaseResult(name, CaseStatus.GAP, ",".join(g.name for g in e.gaps))
    except Exception as e:
        msg = str(e).strip().splitlines()
        return CaseResult(
            name, CaseStatus.FAIL, f"{type(e).__name__}: {msg[0] if msg else ''}"
        )
    finally:
        torch.cuda.empty_cache()
    return CaseResult(name, CaseStatus.PASS, "")


def _run_sweep(device, title: str, cases: list[tuple[str, dict[str, Any]]]) -> None:
    """Run every case (no fail-fast), print the table, then assert.

    Not ``@parameterize``: it stops at the first failure, which would hide the
    status of the remaining cases.
    """
    results = [_run_case(device, name, cfg) for name, cfg in cases]

    width = max(len(r.name) for r in results)
    print(f"\n===== {title}: {len(results)} cases =====")
    for r in results:
        print(f"{r.status.value:<5} {r.name:<{width}}  {r.detail}")
    counts = {s: sum(r.status is s for r in results) for s in CaseStatus}
    print(" | ".join(f"{s.value} {n}" for s, n in counts.items()))

    fails = [r for r in results if r.status is CaseStatus.FAIL]
    gaps = [r for r in results if r.status is CaseStatus.GAP]
    if fails:
        raise AssertionError(
            f"{title}: {len(fails)} supported case(s) failed:\n"
            + "\n".join(f"  {r.name}: {r.detail}" for r in fails)
        )
    if gaps and os.environ.get(ALLOW_GAPS_ENVVAR, "0") != "1":
        raise AssertionError(
            f"{title}: {len(gaps)} case(s) need features SM100 lacks "
            f"(set {ALLOW_GAPS_ENVVAR}=1 to gate only the supported cases):\n"
            + "\n".join(f"  {r.name}: {r.detail}" for r in gaps)
        )


# ═══════════════════════════════════════════════════════════
# TestIndexSparseBwdSm100Sweep — sm90 Classic sweep
# ═══════════════════════════════════════════════════════════


@unittest.skipUnless(_is_sm100_family(), "ffa_bwd_sm100_index needs an SM100 GPU")
class TestIndexSparseBwdSm100Sweep(DistTestBase):
    """sm90 Classic sweep: q_seqlen × kv_seqlen × topk, MQA128, D=128, kbs=1."""

    Q_SEQLENS = [512, 1000, 8192]
    KV_SEQLENS = [512, 1000, 8192]
    TOPKS = [128, 256]

    @property
    def seed(self):
        return SEED

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    @with_run_in_mp
    def test_index_sparse_bwd_sm100_classic(self):
        cases = []
        for q_seqlen, kv_seqlen, topk in itertools.product(
            self.Q_SEQLENS, self.KV_SEQLENS, self.TOPKS
        ):
            if topk > kv_seqlen:
                continue
            config = {
                "B": 1,
                "S_q": q_seqlen,
                "S_kv": kv_seqlen,
                "NHQ": 128,
                "NHK": 1,
                "D": 128,
                "topk": topk,
            }
            cases.append((f"q{q_seqlen}_kv{kv_seqlen}_topk{topk}", config))
        _run_sweep(self.device, "classic", cases)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseBwdSm100ComprehensiveSweep — sm90 Comprehensive sweep
# ═══════════════════════════════════════════════════════════


@unittest.skipUnless(_is_sm100_family(), "ffa_bwd_sm100_index needs an SM100 GPU")
class TestIndexSparseBwdSm100ComprehensiveSweep(DistTestBase):
    """sm90 Comprehensive sweep: head_config × head_dim × kbs.

    The sm90 inner_* / LoopQ-LoopK axes are dropped (see module docstring),
    so each remaining combination runs once.
    """

    _PARAM_SPACE: dict[str, list] = dict(
        head_config=[
            (128, 1, True),  # MQA128
            (4, 1, True),  # MQA4
            (128, 2, True),  # GQA 128:2
            (32, 4, True),  # GQA 32:4
            (4, 4, True),  # MHA4
            (32, 32, True),  # MHA32
        ],
        head_dim=[64, 128],
        kbs=[1, 8, 128],
    )

    @classmethod
    def _iter_cases(cls):
        for head_config, hd, kbs in itertools.product(
            cls._PARAM_SPACE["head_config"],
            cls._PARAM_SPACE["head_dim"],
            cls._PARAM_SPACE["kbs"],
        ):
            nhq, nhk, pack_gqa = head_config
            # Same validity rule as the sm90 sweep: kbs > 1 only for NHK=1, PackGQA, D=128.
            if kbs > 1 and (nhk > 1 or not pack_gqa or hd != 128):
                continue
            if kbs <= 1:
                S, topk = 256, 128
            else:
                S = 1024
                topk = max(2, 128 // kbs)
            config: dict[str, Any] = {
                "B": 1,
                "S": S,
                "NHQ": nhq,
                "NHK": nhk,
                "D": hd,
                "topk": topk,
                "sparse_k_block_size": kbs,
            }
            if kbs > 1:
                config["max_topk"] = topk
            yield f"nhq{nhq}_nhk{nhk}_hd{hd}_kbs{kbs}", config

    @property
    def seed(self):
        return SEED

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    @with_run_in_mp
    def test_index_sparse_bwd_sm100_comprehensive(self):
        _run_sweep(self.device, "comprehensive", list(self._iter_cases()))


# ═══════════════════════════════════════════════════════════
# TestIndexSparseBwdSm100Harness — the harness itself
# ═══════════════════════════════════════════════════════════


@unittest.skipUnless(_is_sm100_family(), "ffa_bwd_sm100_index needs an SM100 GPU")
class TestIndexSparseBwdSm100Harness(unittest.TestCase):
    @property
    def device(self):
        return torch.cuda.current_device()

    def test_refold_keeps_multiple_kv_heads(self):
        """num_heads_kv = 2 with 128 q heads each: the refold goes through a copy."""
        config = {
            "B": 1,
            "S": 256,
            "NHQ": 256,
            "NHK": 2,
            "D": 128,
            "topk": 128,
            "keep_kv_heads": True,
        }
        _run_index_sparse_config(self.device, config)

    def test_comparison_rejects_dk_on_wrong_rows(self):
        """A PASS must mean something: dK scattered one K row off has to fail."""

        def shifted_dk_bwd(*args, **kwargs) -> BwdGrads:
            dq, dk, dv = ffa_bwd_sm100_index_sm90_layout(*args, **kwargs)
            return dq, dk.roll(1, dims=0), dv

        config = {"B": 1, "S": 256, "NHQ": 128, "NHK": 1, "D": 128, "topk": 128}
        with self.assertRaises(AssertionError) as ctx:
            _run_index_sparse_config(self.device, config, bwd_fn=shifted_dk_bwd)
        self.assertIn("=> dk", str(ctx.exception))


if __name__ == "__main__":
    run_tests()
