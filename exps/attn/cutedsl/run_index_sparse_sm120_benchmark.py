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
Benchmark: SM120 IndexSparse forward (``ffa_fwd_sm120_index``) on sm90 bench data.

The forward half of ``run_index_sparse_sm100_benchmark.py`` on SM120 (RTX 50xx /
RTX PRO 6000); there is no SM120 index-sparse backward. Data is generated
exactly as in ``exps/attn/sparse/run_index_sparse_comparison_benchmark.py``:
per q block, ``topk`` sorted ``randint`` K ids (repeats allowed), and the same
128x128 block mask for FlexAttention. The sm90 tensors reach the SM120 kernel
through ``ffa_fwd_sm120_index_sm90_layout``, the adapter of
``tests/test_kernel/cutedsl/index_sparse_sm120_utils.py``.

Scenarios (``SCENARIOS``) are those of the SM100 bench: q_block_size > 1
(``qbs`` consecutive q tokens share one index row) with ``qbs * nhq == 128``,
plus MQA128 with qbs = 1. Combinations the SM120 kernel cannot serve report N/A
(the ``Sm120IndexGap`` names are printed).

Methods (the timed call is the whole forward, out / lse included):
  - ffa_sm120_index      : ffa_fwd_sm120_index
  - triton_token_sparse  : Triton token-sparse forward, MQA tl.dot; with qbs > 1
                           each q block's tokens are folded into its heads (the
                           sm90 view trick), which computes the same function
  - flexattention        : FlexAttention forward on the 128x128 block mask

Sequence lengths: the swept ``S`` is the q and the kv length. ``--seqlen-q``
fixes the q length instead and leaves ``S`` as the kv length, the shape of a
context-parallel rank whose local q is a slice of the full sequence.

Reporting effective sparse TFLOPs/s: 4 * S_q * topk_tokens * nhq * hd.

Run:
    cd exps/attn/cutedsl
    PYTHONPATH=../../.. python run_index_sparse_sm120_benchmark.py
    PYTHONPATH=../../.. python run_index_sparse_sm120_benchmark.py \\
        --scenarios gqa8x1_qbs16_hd128_topk4096 \\
        --methods ffa_sm120_index --seqlens 32768
    PYTHONPATH=../../.. python run_index_sparse_sm120_benchmark.py \\
        --scenarios gqa8x1_qbs16_hd128_topk4096 --seqlen-q 16384 --seqlens 131072
"""

import argparse
from dataclasses import dataclass
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
from exps.attn.baselines.token_sparse_attn_triton import token_sparse_fwd
from tests.test_kernel.cutedsl.index_sparse_sm100_utils import index_sparse_ref_fwd
from tests.test_kernel.cutedsl.index_sparse_sm120_utils import (
    Sm120IndexGap,
    ffa_fwd_sm120_index_sm90_layout,
    sm120_index_fwd_gaps,
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
    "mqa128_hd128_topk4096": IndexSparseScenario(nhq=128, nhk=1, hd=128, topk=4096),
    # tokens x q heads = 128 rows share one index row: one full SM120 Q tile
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
    "mqa128_hd128_topk8192": IndexSparseScenario(nhq=128, nhk=1, hd=128, topk=8192),
    "gqa8x1_qbs16_hd128_topk8192": IndexSparseScenario(
        nhq=8, nhk=1, hd=128, topk=8192, qbs=16
    ),
    "gqa4x1_qbs32_hd128_topk8192": IndexSparseScenario(
        nhq=4, nhk=1, hd=128, topk=8192, qbs=32
    ),
    "gqa16x1_qbs8_hd128_topk8192": IndexSparseScenario(
        nhq=16, nhk=1, hd=128, topk=8192, qbs=8
    ),
}

METHODS: dict[str, tuple[str, tuple[str, str]]] = {
    "ffa_sm120_index": ("FFA SM120 IndexSparse", ("red", "-")),
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


def build_flex_sparse_block_mask(b, S_q, S_kv, topk, nhq, device="cuda"):
    """Build a flex_attention block_mask for token-sparse pattern (sm90 bench)."""
    num_q_blocks = S_q // FLEX_BLOCK
    num_kv_blocks = S_kv // FLEX_BLOCK
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
        sparse_mask_mod, B=None, H=None, Q_LEN=S_q, KV_LEN=S_kv, device=device
    )


@dataclass
class FlatInputs:
    """sm90 IndexSparse tensors with kv heads flattened into the sequence (nhk = 1)."""

    q: torch.Tensor  # (S_q * nhk, nhq_flat, hd)
    k: torch.Tensor  # (S_kv * nhk, 1, hd)
    v: torch.Tensor  # (S_kv * nhk, 1, hd)
    indices: torch.Tensor  # (S_q * nhk / qbs, 1, topk): K token or K block ids


def make_flat_inputs(
    sc: IndexSparseScenario, S_q: int, S_kv: int, device
) -> FlatInputs:
    total_q = b * S_q * sc.nhk
    total_kv = b * S_kv * sc.nhk
    q = torch.randn(total_q, sc.nhq_flat, sc.hd, device=device, dtype=dtype)
    k = torch.randn(total_kv, 1, sc.hd, device=device, dtype=dtype)
    v = torch.randn(total_kv, 1, sc.hd, device=device, dtype=dtype)
    indices = build_index_sparse_indices(
        b,
        S_q * sc.nhk // sc.qbs,
        1,
        sc.topk,
        device,
        num_k_ids=S_kv * sc.nhk // sc.kbs,
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


# ─── Method support ───────────────────────────────────────────────────────────


def gap_reason(gaps: list[Sm120IndexGap]) -> str | None:
    return ",".join(g.name for g in gaps) if gaps else None


def ffa_sm120_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    return gap_reason(
        sm120_index_fwd_gaps(
            head_dim=sc.hd,
            sparse_k_block_size=sc.kbs,
            q_block_size=sc.qbs,
            group=sc.nhq_flat,
        )
    )


def triton_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    if sc.nhq_triton < TRITON_MIN_HEADS_Q:
        return f"q heads per Triton row {sc.nhq_triton} < {TRITON_MIN_HEADS_Q}"
    if sc.hd not in TRITON_HEAD_DIMS:
        return f"head_dim {sc.hd} not in {TRITON_HEAD_DIMS}"
    return None


def flex_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    return None


UNSUPPORTED_REASON: dict[str, Callable[[IndexSparseScenario], str | None]] = {
    "ffa_sm120_index": ffa_sm120_unsupported_reason,
    "triton_token_sparse": triton_unsupported_reason,
    "flexattention": flex_unsupported_reason,
}


# ─── Sanity check ─────────────────────────────────────────────────────────────


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
    if method == "ffa_sm120_index":
        out, lse = ffa_fwd_sm120_index_sm90_layout(
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
        raise ValueError(f"Unknown sanity method: {method}")
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


def sanity_check(
    scenario_names: list[str],
    method_names: list[str],
    assume_valid_ids: bool,
    S_check: int = 1024,
) -> dict[str, set[str]]:
    """Check each runnable (scenario, method) against the fp32 reference.

    Returns, per scenario, the methods whose outputs failed (or raised); they
    report OOM in the benchmark, as in the sm90 bench.
    """
    device = torch.cuda.current_device()
    failed: dict[str, set[str]] = {name: set() for name in scenario_names}
    print(
        f"\n{'=' * 78}\nSanity check: S={S_check}, max|err| / max|ref| < {SANITY_TOL}"
    )
    print(f"{'scenario':<28} {'method':<20} {'errors':<40} status")
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
        inp = make_flat_inputs(sc_check, S_check, S_check, device)
        ref_out, ref_lse = ref_fwd_out_lse(inp, sc_check)
        check = partial(
            check_fwd,
            inp=inp,
            sc=sc_check,
            ref_out=ref_out,
            ref_lse=ref_lse,
            assume_valid_ids=assume_valid_ids,
        )
        for method in method_names:
            if method == "flexattention":
                continue  # different (block) sparsity pattern
            reason = UNSUPPORTED_REASON[method](sc_check)
            if reason is not None:
                print(f"{name:<28} {method:<20} {'':<40} N/A ({reason})")
                continue
            try:
                errs, ok = check(method)
                err_text = " ".join(f"{k} {e:.2e}" for k, e in errs.items())
                status = "PASS" if ok else "FAIL"
                print(f"{name:<28} {method:<20} {err_text:<40} {status}")
            except Exception as e:
                ok = False
                print(f"{name:<28} {method:<20} {'':<40} ERROR: {e}")
            if not ok:
                failed[name].add(method)
        del inp, ref_out, ref_lse
        torch.cuda.empty_cache()
    print()
    return failed


# ─── Benchmark ────────────────────────────────────────────────────────────────


def make_fwd_fn(
    method: str,
    sc: IndexSparseScenario,
    S_q: int,
    S_kv: int,
    device,
    assume_valid_ids: bool,
) -> Callable[[], object]:
    if method == "ffa_sm120_index":
        inp = make_flat_inputs(sc, S_q, S_kv, device)
        return partial(
            ffa_fwd_sm120_index_sm90_layout,
            inp.q,
            inp.k,
            inp.v,
            inp.indices,
            q_block_size=sc.qbs,
            sparse_k_block_size=sc.kbs,
            assume_valid_ids=assume_valid_ids,
        )

    if method == "triton_token_sparse":
        inp = make_flat_inputs(sc, S_q, S_kv, device)
        return partial(
            token_sparse_fwd,
            fold_q_block(inp.q, sc.qbs),
            inp.k,
            inp.v,
            token_ids_2d(inp, sc.kbs),
        )

    if method == "flexattention":
        assert (
            S_q % FLEX_BLOCK == 0 and S_kv % FLEX_BLOCK == 0
        ), "FlexAttention requires S_q and S_kv divisible by 128"
        q = torch.randn(b, sc.nhq, S_q, sc.hd, device=device, dtype=dtype)
        k = torch.randn(b, sc.nhk, S_kv, sc.hd, device=device, dtype=dtype)
        v = torch.randn(b, sc.nhk, S_kv, sc.hd, device=device, dtype=dtype)
        block_mask = build_flex_sparse_block_mask(
            b, S_q, S_kv, sc.topk_tokens, sc.nhq, device
        )
        flex_fn = torch.compile(flex_attention)
        return partial(flex_fn, q, k, v, block_mask=block_mask, enable_gqa=True)

    raise ValueError(f"Unknown method: {method}")


# Filled in place by __main__ before running: the Mark below keeps a reference
# and spawned workers re-import this module, so both must live at module level.
flops_configs: list[Benchmark] = []


@perf_report(flops_configs)
def bench(S, method, scenario, sanity_failed, assume_valid_ids, seqlen_q):
    """One timed call; ``S`` is the kv length, and the q length unless ``seqlen_q`` is set."""
    sc = SCENARIOS[scenario]
    S_kv = S
    S_q = S if seqlen_q is None else seqlen_q
    tag = f"[FWD {method}] {scenario} S_q={S_q} S_kv={S_kv}"
    reason = UNSUPPORTED_REASON[method](sc)
    if reason is not None:
        print(f"{tag}: N/A ({reason})")
        return {"flops": [BENCH_CASE_NOT_SUPPORTED] * 3}
    if method in sanity_failed:
        return {"flops": [BENCH_CASE_OOM] * 3}

    device = torch.cuda.current_device()
    # 2 (mul + add) x 2 (QK, PV) per (q row, key, head-dim) triple.
    sparse_flops = 4 * S_q * sc.topk_tokens * sc.nhq * sc.hd
    set_random_seed(SEED + S)

    try:
        fn = make_fwd_fn(method, sc, S_q, S_kv, device, assume_valid_ids)
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
    seqlens: list[int],
    sanity_failed: dict[str, set[str]],
    assume_valid_ids: bool,
    seqlen_q: int | None,
) -> list[Benchmark]:
    configs = []
    for name in scenario_names:
        sc = SCENARIOS[name]
        configs.append(
            Benchmark(
                x_names=["S"],
                x_vals=seqlens,
                x_log=True,
                line_arg="method",
                line_vals=method_names,
                line_names=[f"{METHODS[m][0]} FWD" for m in method_names],
                styles=[METHODS[m][1] for m in method_names],
                ylabel={"flops": "Effective Sparse TFLOPs/s"},
                # also the output dir name: no newline / spaces
                plot_name=(
                    f"index-sparse-fwd-sm120-nhq{sc.nhq}-nhk{sc.nhk}"
                    f"-hd{sc.hd}-topk{sc.topk}-kbs{sc.kbs}-qbs{sc.qbs}"
                    + ("" if seqlen_q is None else f"-sq{seqlen_q}")
                ),
                args={
                    "scenario": name,
                    "sanity_failed": tuple(sorted(sanity_failed.get(name, ()))),
                    "assume_valid_ids": assume_valid_ids,
                    "seqlen_q": seqlen_q,
                },
            )
        )
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SM120 IndexSparse FWD benchmark on sm90 bench data"
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS)
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
    )
    parser.add_argument("--seqlens", nargs="+", type=int, default=seqlen_vals)
    parser.add_argument(
        "--seqlen-q",
        type=int,
        default=None,
        help="fixed q length; --seqlens then sweeps the kv length only",
    )
    parser.add_argument(
        "--skip-sanity", action="store_true", help="Skip correctness check"
    )
    parser.add_argument(
        "--assume-valid-ids",
        action="store_true",
        help="the generated index has no -1 and no out-of-range id, so "
        "ffa_fwd_sm120_index may take its all-valid fast path (skips the "
        "validity ballots and the softmax mask)",
    )
    parser.add_argument("--num-workers", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    sanity_failed: dict[str, set[str]] = {}
    if not args.skip_sanity:
        sanity_failed = sanity_check(
            args.scenarios, args.methods, args.assume_valid_ids
        )

    flops_configs.extend(
        make_configs(
            args.scenarios,
            args.methods,
            args.seqlens,
            sanity_failed,
            args.assume_valid_ids,
            args.seqlen_q,
        )
    )
    out_root = gen_save_path("bench_index_sparse_sm120")
    bench.run(
        print_data=True,
        print_value_on_bar=False,
        save_path=out_root,
        num_workers=min(args.num_workers, len(flops_configs)),
    )
