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
Benchmark: SM100 IndexSparse forward (``ffa_fwd_sm100_index``) and backward
(``ffa_bwd_sm100_index``) on sm90 bench data.

Data is generated exactly as in
``exps/attn/sparse/run_index_sparse_comparison_benchmark.py``: per q token,
``topk`` sorted ``randint`` K ids (repeats allowed), q_block_size = 1, and the
same 128x128 block mask for FlexAttention. The sm90 tensors reach the SM100
kernels through ``ffa_fwd_sm100_index_sm90_layout`` and
``ffa_bwd_sm100_index_sm90_layout``, the adapters of
``tests/test_kernel/cutedsl/index_sparse_sm100_utils.py``.

Scenarios (``SCENARIOS``) extend the sm90 bench (MQA128, D=128, topk=2048)
with the axes of the sm90 IndexSparse test: head_dim 64, GQA / MHA (kv heads
flattened into the sequence as that test does) and sparse_k_block_size > 1,
plus q_block_size > 1 (``qbs`` consecutive q tokens share one index row; then
the ``qbs`` per-block ids are drawn once per block instead of once per token).
Combinations an SM100 kernel cannot serve yet report N/A (the ``Sm100IndexGap``
names are printed) and start reporting numbers once the gap is closed.

Directions (``--directions``, both by default):
  - fwd : the timed call is the whole forward, out / lse included
  - bwd : out / lse are computed once, outside the timed region, by the Triton
          forward or the fp32 reference

Methods:
  - ffa_sm100_index      : fwd ffa_fwd_sm100_index; bwd ffa_bwd_sm100_index
                           (preprocess + main kernel + postprocess)
  - triton_token_sparse  : Triton token-sparse forward / autograd backward, MQA
                           tl.dot; with qbs > 1 each q block's tokens are folded
                           into its heads (the sm90 view trick), which computes
                           the same function
  - flexattention        : FlexAttention forward / backward on the 128x128 block mask

Reporting effective sparse TFLOPs/s: fwd 4 * S * topk_tokens * nhq * hd, bwd
2.5x the fwd FLOPs.

Run:
    cd exps/attn/cutedsl
    PYTHONPATH=../../.. python run_index_sparse_sm100_benchmark.py
    PYTHONPATH=../../.. python run_index_sparse_sm100_benchmark.py \\
        --directions fwd --scenarios gqa8x1_qbs16_hd128_topk4096 \\
        --methods ffa_sm100_index --seqlens 32768
"""

import argparse
import math
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from magi_attention.benchmarking import (
    BENCH_CASE_NOT_SUPPORTED,
    BENCH_CASE_OOM,
    Benchmark,
    do_bench_flops,
    gen_save_path,
    perf_report,
)
from magi_attention.utils import set_random_seed

# isort: split
from exps.attn.baselines.token_sparse_attn_triton import (
    token_sparse_attn,
    token_sparse_fwd,
)
from tests.test_kernel.cutedsl.index_sparse_sm100_utils import (
    Sm100IndexGap,
    ffa_bwd_sm100_index_sm90_layout,
    ffa_fwd_sm100_index_sm90_layout,
    index_sparse_ref_fwd,
    sm100_index_bwd_gaps,
    sm100_index_fwd_gaps,
)

# ─── Config ───────────────────────────────────────────────────────────────────

b = 1
dtype = torch.bfloat16
quantiles = [0.5, 0.2, 0.8]
# Same sweep as the sm90 bench.
seqlen_vals = [32768, 65536, 131072, 262144, 524288]
SEED = 42
FLEX_BLOCK = 128
# Triton token-sparse: BLOCK_M = next_pow2(nhq) must reach the tl.dot minimum.
TRITON_MIN_HEADS_Q = 16
TRITON_HEAD_DIMS = (64, 128, 256)
# Sanity check bar: max|err| / max|ref| per output.
SANITY_TOL = 0.05


class Direction(Enum):
    FWD = "fwd"
    BWD = "bwd"


# The backward runs 5 matmuls (S recompute, dP, dV, dQ, dK) against the forward's 2.
FLOPS_RATIO_TO_FWD: dict[Direction, float] = {
    Direction.FWD: 1.0,
    Direction.BWD: 2.5,
}


@dataclass(frozen=True)
class IndexSparseScenario:
    nhq: int
    nhk: int
    hd: int
    # index entries per q block: K tokens when kbs == 1, K blocks of kbs tokens otherwise
    topk: int
    kbs: int = 1
    # q tokens per index row (the sm90 q_block_size)
    qbs: int = 1

    def __post_init__(self):
        # Flattening interleaves kv heads into the token axis, which would split
        # a q block of real tokens across index rows.
        assert self.qbs == 1 or self.nhk == 1, "qbs > 1 needs nhk == 1"

    @property
    def topk_tokens(self) -> int:
        return self.topk * self.kbs

    @property
    def nhq_flat(self) -> int:
        """q heads per token once kv heads are flattened into the sequence."""
        return self.nhq // self.nhk

    @property
    def nhq_triton(self) -> int:
        """q heads per Triton row once a q block's tokens are folded into heads."""
        return self.qbs * self.nhq_flat


SCENARIOS: dict[str, IndexSparseScenario] = {
    # the sm90 bench scenario
    "mqa128_hd128_topk2048": IndexSparseScenario(nhq=128, nhk=1, hd=128, topk=2048),
    # "mqa128_hd128_topk1024": IndexSparseScenario(nhq=128, nhk=1, hd=128, topk=1024),
    "mqa128_hd128_topk4096": IndexSparseScenario(nhq=128, nhk=1, hd=128, topk=4096),
    # "mqa128_hd64_topk2048": IndexSparseScenario(nhq=128, nhk=1, hd=64, topk=2048), # Not Support hd = 64
    # "gqa128x2_hd128_topk2048": IndexSparseScenario(nhq=128, nhk=2, hd=128, topk=2048),
    # "gqa32x4_hd128_topk2048": IndexSparseScenario(nhq=32, nhk=4, hd=128, topk=2048),
    # "mha32_hd128_topk2048": IndexSparseScenario(nhq=32, nhk=32, hd=128, topk=2048),
    # "mqa128_hd128_kbs8_topk256": IndexSparseScenario(
    #     nhq=128, nhk=1, hd=128, topk=256, kbs=8
    # ), # Not Support kbs = 8
    # "mqa128_hd128_kbs128_topk16": IndexSparseScenario(
    #     nhq=128, nhk=1, hd=128, topk=16, kbs=128
    # ),
    # tokens x q heads = 128 rows share one index row: one full SM100 Q tile
    "gqa8x1_qbs16_hd128_topk4096": IndexSparseScenario(
        nhq=8, nhk=1, hd=128, topk=4096, qbs=16
    ),
    "gqa8x1_qbs16_hd128_topk2048": IndexSparseScenario(
        nhq=8, nhk=1, hd=128, topk=2048, qbs=16
    ),
    "gqa4x1_qbs32_hd128_topk4096": IndexSparseScenario(
        nhq=4, nhk=1, hd=128, topk=4096, qbs=32
    ),
    "gqa16x1_qbs8_hd128_topk4096": IndexSparseScenario(
        nhq=16, nhk=1, hd=128, topk=4096, qbs=8
    ),
}

# Line labels; make_configs appends the direction.
METHODS: dict[str, tuple[str, tuple[str, str]]] = {
    "ffa_sm100_index": ("FFA SM100 IndexSparse", ("red", "-")),
    "triton_token_sparse": ("Triton Token-Sparse (MQA tl.dot)", ("blue", "--")),
    "flexattention": ("FlexAttention (GQA, sparse mask)", ("green", "-.")),
}


# ─── Data generation (as in run_index_sparse_comparison_benchmark.py) ─────────


def build_index_sparse_indices(b, S, nhk, topk, device, num_k_ids=None):
    """Build index_sparse_indices: (b*S, nhk, topk) int32 with logical K positions.

    The sm90 bench generator; ``num_k_ids`` (default ``S``) is the per-batch id
    range, ``S // kbs`` when the ids name K blocks.
    """
    num_k_ids = S if num_k_ids is None else num_k_ids
    total_q = b * S
    local_pos = (
        torch.randint(0, num_k_ids, (total_q, topk), device=device).sort(dim=1).values
    )
    batch_idx = torch.arange(total_q, device=device) // S
    global_pos = batch_idx.unsqueeze(1) * num_k_ids + local_pos
    # Expand to (total_q, nhk, topk) — same logical K positions per head
    return global_pos.unsqueeze(1).expand(-1, nhk, -1).int().contiguous()


def build_flex_sparse_block_mask(b, S, topk, nhq, device="cuda"):
    """Build a flex_attention block_mask for token-sparse pattern (sm90 bench)."""
    num_q_blocks = S // FLEX_BLOCK
    num_kv_blocks = S // FLEX_BLOCK
    kv_blocks_needed = min((topk + FLEX_BLOCK - 1) // FLEX_BLOCK, num_kv_blocks)

    selected_kv_blocks = torch.rand(num_q_blocks, num_kv_blocks, device=device).argsort(
        dim=1
    )[:, :kv_blocks_needed]
    mask_dense = torch.zeros(
        num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )
    mask_dense.scatter_(1, selected_kv_blocks, True)

    def sparse_mask_mod(b_idx, h_idx, q_idx, kv_idx):
        return mask_dense[q_idx // FLEX_BLOCK, kv_idx // FLEX_BLOCK]

    return create_block_mask(
        sparse_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
    )


@dataclass
class FlatInputs:
    """sm90 IndexSparse tensors with kv heads flattened into the sequence (nhk = 1)."""

    q: torch.Tensor  # (S * nhk, nhq_flat, hd)
    k: torch.Tensor  # (S * nhk, 1, hd)
    v: torch.Tensor  # (S * nhk, 1, hd)
    indices: torch.Tensor  # (S * nhk / qbs, 1, topk): K token or K block ids


def make_flat_inputs(sc: IndexSparseScenario, S: int, device) -> FlatInputs:
    total = b * S * sc.nhk
    q = torch.randn(total, sc.nhq_flat, sc.hd, device=device, dtype=dtype)
    k = torch.randn(total, 1, sc.hd, device=device, dtype=dtype)
    v = torch.randn(total, 1, sc.hd, device=device, dtype=dtype)
    indices = build_index_sparse_indices(
        b, S * sc.nhk // sc.qbs, 1, sc.topk, device, num_k_ids=S * sc.nhk // sc.kbs
    )
    return FlatInputs(q=q, k=k, v=v, indices=indices)


def token_ids_2d(inp: FlatInputs, kbs: int) -> torch.Tensor:
    """(num_q_blocks, topk_tokens) K token ids, K blocks expanded to their tokens."""
    ids = inp.indices[:, 0, :]
    if kbs == 1:
        return ids
    offsets = torch.arange(kbs, device=ids.device, dtype=ids.dtype)
    return (ids[..., None] * kbs + offsets).flatten(-2).contiguous()


def fold_q_block(x: torch.Tensor, qbs: int) -> torch.Tensor:
    """(total_q, nh, ...) -> (total_q / qbs, qbs * nh, ...): a q block's tokens become heads."""
    return x.reshape(x.shape[0] // qbs, qbs * x.shape[1], *x.shape[2:])


def triton_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    if sc.nhq_triton < TRITON_MIN_HEADS_Q:
        return f"q heads per Triton row {sc.nhq_triton} < {TRITON_MIN_HEADS_Q}"
    if sc.hd not in TRITON_HEAD_DIMS:
        return f"head_dim {sc.hd} not in {TRITON_HEAD_DIMS}"
    return None


def triton_fwd_out_lse(
    inp: FlatInputs, sc: IndexSparseScenario
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton token-sparse forward, returned in the flat sm90 layout."""
    out, lse = token_sparse_fwd(
        fold_q_block(inp.q, sc.qbs),
        inp.k,
        inp.v,
        token_ids_2d(inp, sc.kbs),
        return_lse=True,
    )
    return out.reshape(inp.q.shape), lse.reshape(inp.q.shape[:2])


def ref_fwd_out_lse(
    inp: FlatInputs, sc: IndexSparseScenario
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 gather forward: a repeated K id is attended twice."""
    return index_sparse_ref_fwd(
        inp.q,
        inp.k,
        inp.v,
        inp.indices,
        q_block_size=sc.qbs,
        sparse_k_block_size=sc.kbs,
    )


def forward_out_lse(
    inp: FlatInputs, sc: IndexSparseScenario
) -> tuple[torch.Tensor, torch.Tensor]:
    """out / lse for the backward: Triton forward when it applies, fp32 reference otherwise."""
    if triton_unsupported_reason(sc) is None:
        return triton_fwd_out_lse(inp, sc)
    return ref_fwd_out_lse(inp, sc)


# ─── Method support ───────────────────────────────────────────────────────────


def gap_reason(gaps: list[Sm100IndexGap]) -> str | None:
    return ",".join(g.name for g in gaps) if gaps else None


def ffa_sm100_fwd_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    return gap_reason(
        sm100_index_fwd_gaps(
            head_dim=sc.hd,
            sparse_k_block_size=sc.kbs,
            q_block_size=sc.qbs,
            group=sc.nhq_flat,
        )
    )


def ffa_sm100_bwd_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    return gap_reason(sm100_index_bwd_gaps(head_dim=sc.hd, sparse_k_block_size=sc.kbs))


def flex_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    return None


UNSUPPORTED_REASON: dict[
    tuple[Direction, str], Callable[[IndexSparseScenario], str | None]
] = {
    (Direction.FWD, "ffa_sm100_index"): ffa_sm100_fwd_unsupported_reason,
    (Direction.FWD, "triton_token_sparse"): triton_unsupported_reason,
    (Direction.FWD, "flexattention"): flex_unsupported_reason,
    (Direction.BWD, "ffa_sm100_index"): ffa_sm100_bwd_unsupported_reason,
    (Direction.BWD, "triton_token_sparse"): triton_unsupported_reason,
    (Direction.BWD, "flexattention"): flex_unsupported_reason,
}


# ─── Sanity check ─────────────────────────────────────────────────────────────


def ref_bwd_grads(
    inp: FlatInputs, sc: IndexSparseScenario, do: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 autograd gather reference: a repeated K id is attended twice."""
    q, k, v = (t.detach().float().requires_grad_(True) for t in (inp.q, inp.k, inp.v))
    ids = token_ids_2d(inp, sc.kbs).long().repeat_interleave(sc.qbs, dim=0)
    k_sel, v_sel = k[ids, 0], v[ids, 0]
    s = torch.einsum("thd,tkd->thk", q, k_sel) / math.sqrt(sc.hd)
    o = torch.einsum("thk,tkd->thd", torch.softmax(s, dim=-1), v_sel)
    o.backward(do.float())
    assert q.grad is not None and k.grad is not None and v.grad is not None
    return q.grad, k.grad, v.grad


def check_fwd(
    method: str,
    *,
    inp: FlatInputs,
    sc: IndexSparseScenario,
    ref_out: torch.Tensor,
    ref_lse: torch.Tensor,
    assume_valid_ids: bool,
) -> tuple[dict[str, float], bool]:
    """out / lse errors of one forward method, and whether they pass."""
    if method == "ffa_sm100_index":
        out, lse = ffa_fwd_sm100_index_sm90_layout(
            inp.q,
            inp.k,
            inp.v,
            inp.indices,
            q_block_size=sc.qbs,
            sparse_k_block_size=sc.kbs,
            assume_valid_ids=assume_valid_ids,
        )
    elif method == "triton_token_sparse":
        out, lse = triton_fwd_out_lse(inp, sc)
    else:
        raise ValueError(f"Unknown FWD sanity method: {method}")
    finite = torch.isfinite(ref_lse)
    errs = {
        "out": (
            (out.float() - ref_out.float()).abs()[finite].max()
            / ref_out.float()[finite].abs().max()
        ).item(),
        "lse": (
            (lse - ref_lse).abs()[finite].max() / ref_lse[finite].abs().max()
        ).item(),
    }
    # A row that attends nothing must report -inf and a zero output.
    empty_ok = bool(torch.equal(torch.isneginf(lse), ~finite)) and bool(
        (out.float()[~finite] == 0).all()
    )
    return errs, empty_ok and all(e < SANITY_TOL for e in errs.values())


def check_bwd(
    method: str,
    *,
    inp: FlatInputs,
    sc: IndexSparseScenario,
    do: torch.Tensor,
    ref_grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[dict[str, float], bool]:
    """dq / dk / dv errors of one backward method, and whether they pass."""
    if method == "ffa_sm100_index":
        out, lse = forward_out_lse(inp, sc)
        grads = ffa_bwd_sm100_index_sm90_layout(
            inp.q,
            inp.k,
            inp.v,
            out,
            do,
            lse,
            inp.indices,
            q_block_size=sc.qbs,
            sparse_k_block_size=sc.kbs,
        )
    elif method == "triton_token_sparse":
        q, k, v = (
            t.detach().clone().requires_grad_(True)
            for t in (fold_q_block(inp.q, sc.qbs), inp.k, inp.v)
        )
        token_sparse_attn(q, k, v, token_ids_2d(inp, sc.kbs)).backward(
            fold_q_block(do, sc.qbs)
        )
        assert q.grad is not None and k.grad is not None and v.grad is not None
        grads = (q.grad.reshape(inp.q.shape), k.grad, v.grad)
    else:
        raise ValueError(f"Unknown BWD sanity method: {method}")
    errs = {
        name: ((g.float() - r).abs().max() / r.abs().max()).item()
        for name, g, r in zip(("dq", "dk", "dv"), grads, ref_grads)
    }
    return errs, all(e < SANITY_TOL for e in errs.values())


def sanity_check(
    scenario_names: list[str],
    method_names: list[str],
    directions: list[Direction],
    assume_valid_ids: bool,
    S_check: int = 1024,
) -> dict[tuple[Direction, str], set[str]]:
    """Check each runnable (direction, scenario, method) against the fp32 reference.

    Returns, per (direction, scenario), the methods whose outputs failed (or
    raised); they report OOM in the benchmark, as in the sm90 bench.
    """
    device = torch.cuda.current_device()
    failed: dict[tuple[Direction, str], set[str]] = {
        (direction, name): set() for direction in directions for name in scenario_names
    }
    print(
        f"\n{'=' * 78}\nSanity check: S={S_check}, max|err| / max|ref| < {SANITY_TOL}"
    )
    print(f"{'scenario':<28} {'dir':<4} {'method':<20} {'errors':<40} status")
    for name in scenario_names:
        sc = SCENARIOS[name]
        # the sm90 bench sanity check uses topk 128; keep 128 K tokens per row
        sc_check = IndexSparseScenario(
            nhq=sc.nhq,
            nhk=sc.nhk,
            hd=sc.hd,
            topk=max(1, 128 // sc.kbs),
            kbs=sc.kbs,
            qbs=sc.qbs,
        )
        set_random_seed(SEED)
        inp = make_flat_inputs(sc_check, S_check, device)
        do = torch.randn_like(inp.q)
        for direction in directions:
            check: Callable[[str], tuple[dict[str, float], bool]]
            if direction is Direction.FWD:
                ref_out, ref_lse = ref_fwd_out_lse(inp, sc_check)
                check = partial(
                    check_fwd,
                    inp=inp,
                    sc=sc_check,
                    ref_out=ref_out,
                    ref_lse=ref_lse,
                    assume_valid_ids=assume_valid_ids,
                )
            else:
                ref_grads = ref_bwd_grads(inp, sc_check, do)
                check = partial(
                    check_bwd, inp=inp, sc=sc_check, do=do, ref_grads=ref_grads
                )
            row = f"{name:<28} {direction.value:<4}"
            for method in method_names:
                if method == "flexattention":
                    continue  # different (block) sparsity pattern
                reason = UNSUPPORTED_REASON[direction, method](sc_check)
                if reason is not None:
                    print(f"{row} {method:<20} {'':<40} N/A ({reason})")
                    continue
                try:
                    errs, ok = check(method)
                    err_text = " ".join(f"{k} {e:.2e}" for k, e in errs.items())
                    status = "PASS" if ok else "FAIL"
                    print(f"{row} {method:<20} {err_text:<40} {status}")
                except Exception as e:
                    ok = False
                    print(f"{row} {method:<20} {'':<40} ERROR: {e}")
                if not ok:
                    failed[direction, name].add(method)
            torch.cuda.empty_cache()
        del inp, do
        torch.cuda.empty_cache()
    print()
    return failed


# ─── Benchmark ────────────────────────────────────────────────────────────────


def make_fwd_fn(
    method: str,
    sc: IndexSparseScenario,
    S: int,
    device,
    assume_valid_ids: bool,
) -> Callable[[], object]:
    if method == "ffa_sm100_index":
        inp = make_flat_inputs(sc, S, device)
        return partial(
            ffa_fwd_sm100_index_sm90_layout,
            inp.q,
            inp.k,
            inp.v,
            inp.indices,
            q_block_size=sc.qbs,
            sparse_k_block_size=sc.kbs,
            assume_valid_ids=assume_valid_ids,
        )

    if method == "triton_token_sparse":
        inp = make_flat_inputs(sc, S, device)
        return partial(
            token_sparse_fwd,
            fold_q_block(inp.q, sc.qbs),
            inp.k,
            inp.v,
            token_ids_2d(inp, sc.kbs),
        )

    if method == "flexattention":
        assert S % FLEX_BLOCK == 0, "FlexAttention requires S divisible by 128"
        q = torch.randn(b, sc.nhq, S, sc.hd, device=device, dtype=dtype)
        k = torch.randn(b, sc.nhk, S, sc.hd, device=device, dtype=dtype)
        v = torch.randn(b, sc.nhk, S, sc.hd, device=device, dtype=dtype)
        block_mask = build_flex_sparse_block_mask(b, S, sc.topk_tokens, sc.nhq, device)
        flex_fn = torch.compile(flex_attention)
        return partial(flex_fn, q, k, v, block_mask=block_mask, enable_gqa=True)

    raise ValueError(f"Unknown FWD method: {method}")


def make_bwd_fn(
    method: str,
    sc: IndexSparseScenario,
    S: int,
    device,
) -> Callable[[], object]:
    if method == "ffa_sm100_index":
        inp = make_flat_inputs(sc, S, device)
        out, lse = forward_out_lse(inp, sc)
        do = torch.randn_like(out)
        return partial(
            ffa_bwd_sm100_index_sm90_layout,
            inp.q,
            inp.k,
            inp.v,
            out,
            do,
            lse,
            inp.indices,
            q_block_size=sc.qbs,
            sparse_k_block_size=sc.kbs,
        )

    if method == "triton_token_sparse":
        inp = make_flat_inputs(sc, S, device)
        q, k, v = (
            t.detach().requires_grad_(True)
            for t in (fold_q_block(inp.q, sc.qbs), inp.k, inp.v)
        )
        out = token_sparse_attn(q, k, v, token_ids_2d(inp, sc.kbs))
        do = torch.randn_like(out)
        return partial(out.backward, do, retain_graph=True)

    if method == "flexattention":
        assert S % FLEX_BLOCK == 0, "FlexAttention requires S divisible by 128"
        q = torch.randn(
            b, sc.nhq, S, sc.hd, device=device, dtype=dtype, requires_grad=True
        )
        k = torch.randn(
            b, sc.nhk, S, sc.hd, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            b, sc.nhk, S, sc.hd, device=device, dtype=dtype, requires_grad=True
        )
        block_mask = build_flex_sparse_block_mask(b, S, sc.topk_tokens, sc.nhq, device)
        flex_fn = torch.compile(flex_attention)
        out = flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)
        do = torch.randn_like(out)
        return partial(out.backward, do, retain_graph=True)

    raise ValueError(f"Unknown BWD method: {method}")


# Filled in place by __main__ before running: the Mark below keeps a reference
# and spawned workers re-import this module, so both must live at module level.
flops_configs: list[Benchmark] = []


@perf_report(flops_configs)
def bench(S, method, scenario, direction, sanity_failed, assume_valid_ids):
    sc = SCENARIOS[scenario]
    tag = f"[{direction.name} {method}] {scenario} S={S}"
    reason = UNSUPPORTED_REASON[direction, method](sc)
    if reason is not None:
        print(f"{tag}: N/A ({reason})")
        return {"flops": [BENCH_CASE_NOT_SUPPORTED] * 3}
    if method in sanity_failed:
        return {"flops": [BENCH_CASE_OOM] * 3}

    device = torch.cuda.current_device()
    # Forward FLOPs: 2 (mul + add) x 2 (QK, PV) per (row, key, head-dim) triple.
    sparse_flops = (
        4 * S * sc.topk_tokens * sc.nhq * sc.hd * FLOPS_RATIO_TO_FWD[direction]
    )
    set_random_seed(SEED + S)

    try:
        if direction is Direction.FWD:
            fn = make_fwd_fn(method, sc, S, device, assume_valid_ids)
        else:
            fn = make_bwd_fn(method, sc, S, device)
        perf_dict = do_bench_flops(fn, quantiles=quantiles)
        perf_dict["flops"] = [sparse_flops / ms * 1e-9 for ms in perf_dict["flops"]]

    except torch.cuda.OutOfMemoryError as e:
        print(f"{tag}: OOM — {e}")
        perf_dict = {"flops": [BENCH_CASE_OOM] * 3}
    except Exception as e:
        print(f"{tag}: {type(e).__name__}: {e}")
        perf_dict = {"flops": [BENCH_CASE_OOM] * 3}

    torch.cuda.empty_cache()
    return perf_dict


def make_configs(
    scenario_names: list[str],
    method_names: list[str],
    directions: list[Direction],
    seqlens: list[int],
    sanity_failed: dict[tuple[Direction, str], set[str]],
    assume_valid_ids: bool,
) -> list[Benchmark]:
    configs = []
    for direction in directions:
        for name in scenario_names:
            sc = SCENARIOS[name]
            configs.append(
                Benchmark(
                    x_names=["S"],
                    x_vals=seqlens,
                    x_log=True,
                    line_arg="method",
                    line_vals=method_names,
                    line_names=[
                        f"{METHODS[m][0]} {direction.name}" for m in method_names
                    ],
                    styles=[METHODS[m][1] for m in method_names],
                    ylabel={"flops": "Effective Sparse TFLOPs/s"},
                    # also the output dir name: no newline / spaces
                    plot_name=(
                        f"index-sparse-{direction.value}-sm100-nhq{sc.nhq}-nhk{sc.nhk}"
                        f"-hd{sc.hd}-topk{sc.topk}-kbs{sc.kbs}-qbs{sc.qbs}"
                    ),
                    args={
                        "scenario": name,
                        "direction": direction,
                        "sanity_failed": tuple(
                            sorted(sanity_failed.get((direction, name), ()))
                        ),
                        "assume_valid_ids": assume_valid_ids,
                    },
                )
            )
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SM100 IndexSparse FWD / BWD benchmark on sm90 bench data"
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=[d.value for d in Direction],
        default=[d.value for d in Direction],
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS)
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
    )
    parser.add_argument("--seqlens", nargs="+", type=int, default=seqlen_vals)
    parser.add_argument(
        "--skip-sanity", action="store_true", help="Skip correctness check"
    )
    parser.add_argument(
        "--assume-valid-ids",
        action="store_true",
        help="fwd only: the generated index has no -1 and no out-of-range id, so "
        "ffa_fwd_sm100_index may take its all-valid fast path (skips the index "
        "scan and the softmax mask)",
    )
    parser.add_argument("--num-workers", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    directions = [Direction(d) for d in args.directions]

    sanity_failed: dict[tuple[Direction, str], set[str]] = {}
    if not args.skip_sanity:
        sanity_failed = sanity_check(
            args.scenarios, args.methods, directions, args.assume_valid_ids
        )

    flops_configs.extend(
        make_configs(
            args.scenarios,
            args.methods,
            directions,
            args.seqlens,
            sanity_failed,
            args.assume_valid_ids,
        )
    )
    out_root = gen_save_path("bench_index_sparse_sm100")
    bench.run(
        print_data=True,
        print_value_on_bar=False,
        save_path=out_root,
        num_workers=min(args.num_workers, len(flops_configs)),
    )
