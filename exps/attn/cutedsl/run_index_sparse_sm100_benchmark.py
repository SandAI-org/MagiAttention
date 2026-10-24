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
Benchmark: SM100 IndexSparse backward (``ffa_bwd_sm100_index``) on sm90 bench data.

Data is generated exactly as in
``exps/attn/sparse/run_index_sparse_comparison_benchmark.py``: per q token,
``topk`` sorted ``randint`` K ids (repeats allowed), q_block_size = 1, and the
same 128x128 block mask for FlexAttention. The sm90 tensors reach the SM100
kernel through ``ffa_bwd_sm100_index_sm90_layout``, the adapter that
``tests/test_kernel/cutedsl/test_index_sparse_bwd_sm100.py`` validates.

Scenarios (``SCENARIOS``) extend the sm90 bench (MQA128, D=128, topk=2048)
with the axes of the sm90 IndexSparse test: head_dim 64, GQA / MHA (kv heads
flattened into the sequence as that test does) and sparse_k_block_size > 1,
plus q_block_size > 1 (``qbs`` consecutive q tokens share one index row; then
the ``qbs`` per-block ids are drawn once per block instead of once per token).
Combinations SM100 cannot serve yet report N/A (the ``Sm100IndexGap`` names are
printed) and start reporting numbers once the gap is closed.

Methods (BWD only — SM100 has no IndexSparse forward; out / lse are computed
once, outside the timed region, by the Triton forward or the fp32 reference):
  - ffa_sm100_index      : ffa_bwd_sm100_index (preprocess + main kernel + postprocess)
  - triton_token_sparse  : Triton token-sparse backward (autograd), MQA tl.dot; with
                           qbs > 1 each q block's tokens are folded into its heads
                           (the sm90 view trick), which computes the same function
  - flexattention        : FlexAttention backward on the 128x128 block mask

Reporting effective sparse TFLOPs/s: 2.5 * 4 * S * topk_tokens * nhq * hd.

Run:
    cd exps/attn/cutedsl
    PYTHONPATH=../../.. python run_index_sparse_sm100_benchmark.py
    PYTHONPATH=../../.. python run_index_sparse_sm100_benchmark.py \\
        --scenarios mqa128_hd128_topk2048 --methods ffa_sm100_index --seqlens 32768
"""

import argparse
import math
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
from exps.attn.baselines.token_sparse_attn_triton import (
    token_sparse_attn,
    token_sparse_fwd,
)
from tests.test_kernel.cutedsl.index_sparse_sm100_utils import (
    ffa_bwd_sm100_index_sm90_layout,
    index_sparse_ref_fwd,
    sm100_index_bwd_gaps,
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
    # "mqa128_hd128_topk2048": IndexSparseScenario(nhq=128, nhk=1, hd=128, topk=2048),
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
    # tokens x q heads = 128 rows share one index row: one full SM100 Q tile
    "gqa8x1_qbs16_hd128_topk4096": IndexSparseScenario(
        nhq=8, nhk=1, hd=128, topk=4096, qbs=16
    ),
    "gqa4x1_qbs32_hd128_topk4096": IndexSparseScenario(
        nhq=4, nhk=1, hd=128, topk=4096, qbs=32
    ),
    "gqa16x1_qbs8_hd128_topk4096": IndexSparseScenario(
        nhq=16, nhk=1, hd=128, topk=4096, qbs=8
    ),
}

METHODS: dict[str, tuple[str, tuple[str, str]]] = {
    "ffa_sm100_index": ("FFA SM100 IndexSparse BWD", ("red", "-")),
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


def forward_out_lse(
    inp: FlatInputs, sc: IndexSparseScenario
) -> tuple[torch.Tensor, torch.Tensor]:
    """out / lse for the backward: Triton forward when it applies, fp32 reference otherwise."""
    if triton_unsupported_reason(sc) is None:
        out, lse = token_sparse_fwd(
            fold_q_block(inp.q, sc.qbs),
            inp.k,
            inp.v,
            token_ids_2d(inp, sc.kbs),
            return_lse=True,
        )
        return out.reshape(inp.q.shape), lse.reshape(inp.q.shape[:2])
    return index_sparse_ref_fwd(
        inp.q,
        inp.k,
        inp.v,
        inp.indices,
        q_block_size=sc.qbs,
        sparse_k_block_size=sc.kbs,
    )


# ─── Method support ───────────────────────────────────────────────────────────


def ffa_sm100_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    gaps = sm100_index_bwd_gaps(head_dim=sc.hd, sparse_k_block_size=sc.kbs)
    return ",".join(g.name for g in gaps) if gaps else None


def flex_unsupported_reason(sc: IndexSparseScenario) -> str | None:
    return None


UNSUPPORTED_REASON: dict[str, Callable[[IndexSparseScenario], str | None]] = {
    "ffa_sm100_index": ffa_sm100_unsupported_reason,
    "triton_token_sparse": triton_unsupported_reason,
    "flexattention": flex_unsupported_reason,
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


def sanity_check(
    scenario_names: list[str], method_names: list[str], S_check: int = 1024
) -> dict[str, set[str]]:
    """Check each runnable (scenario, method) backward against the fp32 reference.

    Returns, per scenario, the methods whose grads failed (or raised); they
    report OOM in the benchmark, as in the sm90 bench.
    """
    device = torch.cuda.current_device()
    tol = 0.05
    failed: dict[str, set[str]] = {name: set() for name in scenario_names}
    print(f"\n{'=' * 78}\nBWD sanity check: S={S_check}, max|err| / max|ref| < {tol}")
    print(f"{'scenario':<28} {'method':<20} {'dq':>9} {'dk':>9} {'dv':>9}  status")
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
        ref = ref_bwd_grads(inp, sc_check, do)
        for method in method_names:
            if method == "flexattention":
                continue  # different (block) sparsity pattern
            reason = UNSUPPORTED_REASON[method](sc_check)
            if reason is not None:
                print(f"{name:<28} {method:<20} {'':>29}  N/A ({reason})")
                continue
            try:
                if method == "ffa_sm100_index":
                    out, lse = forward_out_lse(inp, sc_check)
                    grads = ffa_bwd_sm100_index_sm90_layout(
                        inp.q,
                        inp.k,
                        inp.v,
                        out,
                        do,
                        lse,
                        inp.indices,
                        q_block_size=sc_check.qbs,
                        sparse_k_block_size=sc_check.kbs,
                    )
                else:
                    q, k, v = (
                        t.detach().clone().requires_grad_(True)
                        for t in (fold_q_block(inp.q, sc_check.qbs), inp.k, inp.v)
                    )
                    token_sparse_attn(
                        q, k, v, token_ids_2d(inp, sc_check.kbs)
                    ).backward(fold_q_block(do, sc_check.qbs))
                    assert q.grad is not None
                    grads = (q.grad.reshape(inp.q.shape), k.grad, v.grad)
                errs = [
                    ((g.float() - r).abs().max() / r.abs().max()).item()
                    for g, r in zip(grads, ref)
                ]
                ok = all(e < tol for e in errs)
                status = "PASS" if ok else "FAIL"
                print(
                    f"{name:<28} {method:<20} "
                    + " ".join(f"{e:9.2e}" for e in errs)
                    + f"  {status}"
                )
            except Exception as e:
                ok = False
                print(f"{name:<28} {method:<20} {'':>29}  ERROR: {e}")
            if not ok:
                failed[name].add(method)
        del inp, do, ref
        torch.cuda.empty_cache()
    print()
    return failed


# ─── BWD benchmark ────────────────────────────────────────────────────────────

# Filled in place by __main__ before running: the Mark below keeps a reference
# and spawned workers re-import this module, so both must live at module level.
bwd_flops_configs: list[Benchmark] = []


@perf_report(bwd_flops_configs)
def bwd_benchmark(S, method, scenario, sanity_failed):
    sc = SCENARIOS[scenario]
    reason = UNSUPPORTED_REASON[method](sc)
    if reason is not None:
        print(f"[BWD {method}] {scenario} S={S}: N/A ({reason})")
        return {"flops": [BENCH_CASE_NOT_SUPPORTED] * 3}
    if method in sanity_failed:
        return {"flops": [BENCH_CASE_OOM] * 3}

    device = torch.cuda.current_device()
    sparse_flops = 2.5 * 4 * S * sc.topk_tokens * sc.nhq * sc.hd
    set_random_seed(SEED + S)

    try:
        if method == "ffa_sm100_index":
            inp = make_flat_inputs(sc, S, device)
            out, lse = forward_out_lse(inp, sc)
            do = torch.randn_like(out)
            fn = partial(
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

        elif method == "triton_token_sparse":
            inp = make_flat_inputs(sc, S, device)
            q, k, v = (
                t.detach().requires_grad_(True)
                for t in (fold_q_block(inp.q, sc.qbs), inp.k, inp.v)
            )
            out = token_sparse_attn(q, k, v, token_ids_2d(inp, sc.kbs))
            do = torch.randn_like(out)

            def fn():
                out.backward(do, retain_graph=True)

        elif method == "flexattention":
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
            block_mask = build_flex_sparse_block_mask(
                b, S, sc.topk_tokens, sc.nhq, device
            )
            flex_fn = torch.compile(flex_attention)
            out = flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)
            do = torch.randn_like(out)

            def fn():
                out.backward(do, retain_graph=True)

        else:
            raise ValueError(f"Unknown BWD method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles)
        perf_dict["flops"] = [sparse_flops / ms * 1e-9 for ms in perf_dict["flops"]]

    except torch.cuda.OutOfMemoryError as e:
        print(f"[BWD {method}] {scenario} S={S}: OOM — {e}")
        perf_dict = {"flops": [BENCH_CASE_OOM] * 3}
    except Exception as e:
        print(f"[BWD {method}] {scenario} S={S}: {type(e).__name__}: {e}")
        perf_dict = {"flops": [BENCH_CASE_OOM] * 3}

    torch.cuda.empty_cache()
    return perf_dict


def make_bwd_configs(
    scenario_names: list[str],
    method_names: list[str],
    seqlens: list[int],
    sanity_failed: dict[str, set[str]],
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
                    f"index-sparse-bwd-sm100-nhq{sc.nhq}-nhk{sc.nhk}-hd{sc.hd}"
                    f"-topk{sc.topk}-kbs{sc.kbs}-qbs{sc.qbs}"
                ),
                args={
                    "scenario": name,
                    "sanity_failed": tuple(sorted(sanity_failed.get(name, ()))),
                },
            )
        )
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SM100 IndexSparse BWD benchmark on sm90 bench data"
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
    parser.add_argument("--num-workers", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    sanity_failed: dict[str, set[str]] = {}
    if not args.skip_sanity:
        sanity_failed = sanity_check(args.scenarios, args.methods)

    bwd_flops_configs.extend(
        make_bwd_configs(args.scenarios, args.methods, args.seqlens, sanity_failed)
    )
    out_root = gen_save_path("bench_index_sparse_sm100")
    bwd_benchmark.run(
        print_data=True,
        print_value_on_bar=False,
        save_path=out_root,
        num_workers=min(args.num_workers, len(bwd_flops_configs)),
    )
