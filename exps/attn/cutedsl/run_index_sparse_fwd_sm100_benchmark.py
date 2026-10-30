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
Benchmark: SM100 IndexSparse forward (``ffa_fwd_sm100_index``).

Deliberately mirrors ``run_index_sparse_sm100_benchmark.py`` (the backward bench)
so the two numbers are directly comparable: the same scenarios, the same
seqlens, the same index data, and the same timing harness. Only the FLOPs
denominator differs -- forward is ``4 * S * topk * nhq * hd`` where backward
multiplies that by the 2.5x backward-to-forward FLOPs ratio.

Data is generated as in ``exps/attn/sparse/run_index_sparse_comparison_benchmark.py``:
per index row, ``topk`` sorted ``randint`` K ids (repeats allowed, no ``-1``
padding), so every row carries its full sparse budget. The scenarios keep
``nhq * qbs == 128``: one index row covers exactly one 128-row Q tile, the case
with no wasted MMA rows. All scenarios use one KV head, as the backward bench
does.

Methods (FWD -- the timed call is the whole forward, ``out``/``lse`` included):
  - ffa_sm100_index      : ffa_fwd_sm100_index
  - triton_token_sparse  : Triton token-sparse forward (MQA tl.dot); with
                           qbs > 1 each q block's tokens are folded into its
                           heads (the sm90 view trick), which computes the same
                           function
  - flexattention        : FlexAttention forward on the 128x128 block mask

Reporting effective sparse TFLOPs/s: 4 * S * topk_tokens * nhq * hd.

Run:
    cd exps/attn/cutedsl
    PYTHONPATH=../../.. python run_index_sparse_fwd_sm100_benchmark.py
    PYTHONPATH=../../.. python run_index_sparse_fwd_sm100_benchmark.py \\
        --scenarios gqa8x1_qbs16_hd128_topk4096 --methods ffa_sm100_index \\
        --seqlens 32768
"""

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

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
from magi_attention.kernel.cutedsl.ffa_fwd_sm100_index import ffa_fwd_sm100_index

# ─── Config ───────────────────────────────────────────────────────────────────

b = 1
dtype = torch.bfloat16
quantiles = [0.5, 0.2, 0.8]
# Same sweep as the backward bench.
seqlen_vals = [32768, 65536, 131072, 262144, 524288]
SEED = 42
FLEX_BLOCK = 128
# Triton token-sparse: BLOCK_M = next_pow2(nhq) must reach the tl.dot minimum.
TRITON_MIN_HEADS_Q = 16
TRITON_HEAD_DIMS = (64, 128, 256)
# ffa_fwd_sm100_index fixes the Q tile at 128 rows of q_block_size * group.
SM100_FWD_TILE_M = 128


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
    # The four rows of the backward bench's table, all at topk = 4096.
    "mqa128_hd128_topk4096": IndexSparseScenario(nhq=128, nhk=1, hd=128, topk=4096),
    "gqa16x1_qbs8_hd128_topk4096": IndexSparseScenario(
        nhq=16, nhk=1, hd=128, topk=4096, qbs=8
    ),
    "gqa8x1_qbs16_hd128_topk4096": IndexSparseScenario(
        nhq=8, nhk=1, hd=128, topk=4096, qbs=16
    ),
    "gqa4x1_qbs32_hd128_topk4096": IndexSparseScenario(
        nhq=4, nhk=1, hd=128, topk=4096, qbs=32
    ),
    # The backward table's first row is quoted at topk = 2048; keep it runnable so
    # the forward row can be compared at that setting too.
    "mqa128_hd128_topk2048": IndexSparseScenario(nhq=128, nhk=1, hd=128, topk=2048),
    # Not benchmarked here: the forward serves these natively (unlike the
    # backward it takes a real num_heads_kv), but the backward bench does not
    # quote them, so there is no row to compare against.
    # "gqa128x2_hd128_topk4096": IndexSparseScenario(nhq=128, nhk=2, hd=128, topk=4096),
    # "gqa32x4_hd128_topk4096": IndexSparseScenario(nhq=32, nhk=4, hd=128, topk=4096),
    # "mha32_hd128_topk4096": IndexSparseScenario(nhq=32, nhk=32, hd=128, topk=4096),
}

METHODS: dict[str, tuple[str, tuple[str, str]]] = {
    "ffa_sm100_index": ("FFA SM100 IndexSparse FWD", ("red", "-")),
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


# ─── Method support ───────────────────────────────────────────────────────────


def ffa_sm100_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    if sc.hd != 128:
        return "head_dim != 128: FFABwdSm100Index-style fwd fixes head_dim at 128"
    if sc.kbs != 1:
        return (
            "sparse_k_block_size > 1: the kernel gathers single K tokens, not K blocks"
        )
    if sc.qbs * (sc.nhq // sc.nhk) != SM100_FWD_TILE_M:
        return (
            f"q_block_size * group = {sc.qbs * (sc.nhq // sc.nhk)} != {SM100_FWD_TILE_M}: "
            "the forward requires one index row per 128-row Q tile"
        )
    return None


def triton_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    if sc.nhq_triton < TRITON_MIN_HEADS_Q:
        return f"q heads per Triton row {sc.nhq_triton} < {TRITON_MIN_HEADS_Q}"
    if sc.hd not in TRITON_HEAD_DIMS:
        return f"head_dim {sc.hd} not in {TRITON_HEAD_DIMS}"
    return None


def flex_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    return None


UNSUPPORTED_REASON: dict[str, Callable[[IndexSparseScenario], str | None]] = {
    "ffa_sm100_index": ffa_sm100_unsupported_reason,
    "triton_token_sparse": triton_unsupported_reason,
    "flexattention": flex_unsupported_reason,
}


# ─── Reference forward ────────────────────────────────────────────────────────


def index_sparse_ref_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    q_block_size: int = 1,
    sparse_k_block_size: int = 1,
    softmax_scale: float | None = None,
    chunk_tokens: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 gather forward for the sm90 IndexSparse contract; returns (out, lse).

    Same semantics as the backward branch's helper of the same name, inlined
    because that module lives on the backward branch. Token ``t`` uses index row
    ``t // q_block_size``. A K id listed twice is attended twice; a row without a
    valid id gets out = 0 and lse = -inf.
    """
    total_q, num_heads_q, head_dim = q.shape
    num_heads_kv = k.shape[1]
    group = num_heads_q // num_heads_kv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    lse = torch.empty(total_q, num_heads_q, dtype=torch.float32, device=q.device)
    head_kv = torch.arange(num_heads_kv, device=q.device)[None, :, None]
    block_offsets = torch.arange(sparse_k_block_size, device=q.device)
    for t0 in range(0, total_q, chunk_tokens):
        t1 = min(t0 + chunk_tokens, total_q)
        token_blocks = torch.arange(t0, t1, device=q.device) // q_block_size
        ids = index_sparse_indices[token_blocks].long()
        valid = ids >= 0
        if sparse_k_block_size > 1:
            ids = (ids[..., None] * sparse_k_block_size + block_offsets).flatten(-2)
            valid = valid.repeat_interleave(sparse_k_block_size, dim=-1)
        rows = ids.clamp(min=0)
        # k_sel / v_sel: (c, nhk, topk, d)
        k_sel = k[rows, head_kv].float()
        v_sel = v[rows, head_kv].float()
        q_c = q[t0:t1].float().view(t1 - t0, num_heads_kv, group, head_dim)
        s = torch.einsum("chgd,chtd->chgt", q_c, k_sel) * softmax_scale
        s = s.masked_fill(~valid[:, :, None, :], float("-inf"))
        lse_c = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse_c.masked_fill(lse_c.isneginf(), 0.0)[..., None])
        o = torch.einsum("chgt,chtd->chgd", p, v_sel)
        out[t0:t1] = o.reshape(t1 - t0, num_heads_q, head_dim).to(q.dtype)
        lse[t0:t1] = lse_c.reshape(t1 - t0, num_heads_q)
    return out, lse


# ─── Sanity check ─────────────────────────────────────────────────────────────


def sanity_check(
    scenario_names: list[str],
    method_names: list[str],
    S_check: int = 1024,
    assume_valid_ids: bool = False,
) -> dict[str, set[str]]:
    """Check each runnable (scenario, method) forward against the fp32 reference.

    Returns, per scenario, the methods whose out/lse failed (or raised); they
    report OOM in the benchmark, as in the backward bench.
    """
    device = torch.cuda.current_device()
    tol = 0.05
    failed: dict[str, set[str]] = {name: set() for name in scenario_names}
    print(f"\n{'=' * 78}\nFWD sanity check: S={S_check}, max|err| / max|ref| < {tol}")
    print(f"{'scenario':<28} {'method':<20} {'out':>9} {'lse':>9}  status")
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
        ref_out, ref_lse = index_sparse_ref_fwd(
            inp.q,
            inp.k,
            inp.v,
            inp.indices,
            q_block_size=sc_check.qbs,
            sparse_k_block_size=sc_check.kbs,
        )
        finite = torch.isfinite(ref_lse)
        for method in method_names:
            if method == "flexattention":
                continue  # different (block) sparsity pattern
            reason = UNSUPPORTED_REASON[method](sc_check)
            if reason is not None:
                print(f"{name:<28} {method:<20} {'':>19}  N/A ({reason})")
                continue
            try:
                if method == "ffa_sm100_index":
                    out, lse = ffa_fwd_sm100_index(
                        inp.q,
                        inp.k,
                        inp.v,
                        inp.indices,
                        q_block_size=sc_check.qbs,
                        assume_valid_ids=assume_valid_ids,
                    )
                else:
                    out, lse = token_sparse_fwd(
                        fold_q_block(inp.q, sc_check.qbs),
                        inp.k,
                        inp.v,
                        token_ids_2d(inp, sc_check.kbs),
                        return_lse=True,
                    )
                    out = out.reshape(inp.q.shape)
                    lse = lse.reshape(inp.q.shape[:2])
                torch.cuda.synchronize()
                err_out = (
                    (out.float() - ref_out.float()).abs()[finite].max()
                    / ref_out.float().abs().max()
                ).item()
                err_lse = (
                    (lse - ref_lse).abs()[finite].max() / ref_lse.abs().max()
                ).item()
                # A row that attends nothing must report -inf and a zero output.
                empty_ok = bool(torch.equal(torch.isneginf(lse), ~finite)) and bool(
                    (out.float()[~finite] == 0).all()
                )
                ok = err_out < tol and err_lse < tol and empty_ok
                status = "PASS" if ok else "FAIL"
                print(
                    f"{name:<28} {method:<20} {err_out:9.2e} {err_lse:9.2e}  {status}"
                )
            except Exception as e:  # noqa: BLE001 - a method failure becomes a bench row
                ok = False
                print(f"{name:<28} {method:<20} {'':>19}  ERROR: {e}")
            if not ok:
                failed[name].add(method)
        del inp, ref_out, ref_lse
        torch.cuda.empty_cache()
    print()
    return failed


# ─── FWD benchmark ────────────────────────────────────────────────────────────

# Filled in place by __main__ before running: the Mark below keeps a reference
# and spawned workers re-import this module, so both must live at module level.
fwd_flops_configs: list[Benchmark] = []


@perf_report(fwd_flops_configs)
def fwd_benchmark(S, method, scenario, sanity_failed, assume_valid_ids=False):
    sc = SCENARIOS[scenario]
    reason = UNSUPPORTED_REASON[method](sc)
    if reason is not None:
        print(f"[FWD {method}] {scenario} S={S}: N/A ({reason})")
        return {"flops": [BENCH_CASE_NOT_SUPPORTED] * 3}
    if method in sanity_failed:
        return {"flops": [BENCH_CASE_OOM] * 3}

    device = torch.cuda.current_device()
    # Forward FLOPs: 2 (mul + add) x 2 (QK, PV) per (row, key, head-dim) triple.
    sparse_flops = 4 * S * sc.topk_tokens * sc.nhq * sc.hd
    set_random_seed(SEED + S)

    try:
        if method == "ffa_sm100_index":
            inp = make_flat_inputs(sc, S, device)
            fn = partial(
                ffa_fwd_sm100_index,
                inp.q,
                inp.k,
                inp.v,
                inp.indices,
                q_block_size=sc.qbs,
                assume_valid_ids=assume_valid_ids,
            )

        elif method == "triton_token_sparse":
            inp = make_flat_inputs(sc, S, device)
            q = fold_q_block(inp.q, sc.qbs)
            ids = token_ids_2d(inp, sc.kbs)

            def fn():
                token_sparse_fwd(q, inp.k, inp.v, ids)

        elif method == "flexattention":
            assert S % FLEX_BLOCK == 0, "FlexAttention requires S divisible by 128"
            q = torch.randn(b, sc.nhq, S, sc.hd, device=device, dtype=dtype)
            k = torch.randn(b, sc.nhk, S, sc.hd, device=device, dtype=dtype)
            v = torch.randn(b, sc.nhk, S, sc.hd, device=device, dtype=dtype)
            block_mask = build_flex_sparse_block_mask(
                b, S, sc.topk_tokens, sc.nhq, device
            )
            flex_fn = torch.compile(flex_attention)

            def fn():
                flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)

        else:
            raise ValueError(f"Unknown FWD method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles)
        perf_dict["flops"] = [sparse_flops / ms * 1e-9 for ms in perf_dict["flops"]]

    except torch.cuda.OutOfMemoryError as e:
        print(f"[FWD {method}] {scenario} S={S}: OOM — {e}")
        perf_dict = {"flops": [BENCH_CASE_OOM] * 3}
    except Exception as e:  # noqa: BLE001 - any method failure becomes a bench row
        print(f"[FWD {method}] {scenario} S={S}: {type(e).__name__}: {e}")
        perf_dict = {"flops": [BENCH_CASE_OOM] * 3}

    torch.cuda.empty_cache()
    return perf_dict


def make_fwd_configs(
    scenario_names: list[str],
    method_names: list[str],
    seqlens: list[int],
    sanity_failed: dict[str, set[str]],
    assume_valid_ids: bool = False,
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
                line_names=[METHODS[m][0] for m in method_names],
                styles=[METHODS[m][1] for m in method_names],
                ylabel={"flops": "Effective Sparse TFLOPs/s"},
                # also the output dir name: no newline / spaces
                plot_name=(
                    f"index-sparse-fwd-sm100-nhq{sc.nhq}-nhk{sc.nhk}-hd{sc.hd}"
                    f"-topk{sc.topk}-kbs{sc.kbs}-qbs{sc.qbs}"
                ),
                args={
                    "scenario": name,
                    "sanity_failed": tuple(sorted(sanity_failed.get(name, ()))),
                    "assume_valid_ids": assume_valid_ids,
                },
            )
        )
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SM100 IndexSparse FWD benchmark on sm90 bench data"
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
        help="the generated index has no -1 and no out-of-range id, so the kernel may take its "
        "all-valid fast path (skips the index scan and the softmax mask)",
    )
    parser.add_argument("--num-workers", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    sanity_failed: dict[str, set[str]] = {}
    if not args.skip_sanity:
        sanity_failed = sanity_check(
            args.scenarios, args.methods, assume_valid_ids=args.assume_valid_ids
        )

    fwd_flops_configs.extend(
        make_fwd_configs(
            args.scenarios,
            args.methods,
            args.seqlens,
            sanity_failed,
            args.assume_valid_ids,
        )
    )
    out_root = gen_save_path("bench_index_sparse_fwd_sm100")
    fwd_benchmark.run(
        print_data=True,
        print_value_on_bar=False,
        save_path=out_root,
        num_workers=min(args.num_workers, len(fwd_flops_configs)),
    )
