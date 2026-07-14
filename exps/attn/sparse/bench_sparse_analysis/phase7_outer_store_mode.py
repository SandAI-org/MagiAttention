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

"""Phase 7: outer-store-mode — FWD/BWD OuterStoreMode (TMA vs Stg) comparison.

Dense-only benchmark: d1b (single batch), NHQ=128, NHK=1, PackGQA, bf16.
FWD default is Stg (SwapAB=false), BWD LoopQ default is TMA (TMA_REDUCE_ADD).
Tests whether forcing a different outer store mode improves TFLOPS.

FWD compares: default(Stg) vs TMA (env override).
BWD LoopQ compares: default(TMA_REDUCE_ADD, fp32 dKV) vs no_atomic(Stg, bf16 dKV).
BWD LoopK compares: default(Stg) vs TMA (env override).
"""

import gc
import os
import time

from bench_sparse_analysis._common import (
    HD,
    NHK,
    NHQ,
    _bench_kernel,
    _load_results,
    _results_path,
    _save_results,
    _set_entry,
    _set_gpu,
    _ts,
)

PHASE = "7-outer-store-mode"

SEQLENS = [4096, 8192, 16384, 32768, 65536]

# pass_type → list of (label, extra_kw, env_osm)
#   env_osm: None = no override, "tma"/"stg" = set MAGI_ATTENTION_FFA_OUTER_STORE_MODE
CONFIGS = {
    "fwd": [
        ("default(Stg)", {}, None),
        ("OSM=tma", {}, "tma"),
    ],
    "bwd_loopq": [
        ("default(TMA+fp32)", {}, None),
        ("no_atomic(Stg+bf16)", {"disable_bwd_dkv_atomic_reduction": True}, None),
        ("no_atomic+OSM=tma", {"disable_bwd_dkv_atomic_reduction": True}, "tma"),
    ],
    "bwd_loopk": [
        ("default(Stg)", {"swap_bwd_qk_loop": True}, None),
        ("OSM=tma", {"swap_bwd_qk_loop": True}, "tma"),
    ],
}


def _calc_flops_dense(S, is_bwd):
    fwd = 4 * S * S * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _phase7_bench(force=False):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    results = _load_results(PHASE)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"

    print(f"[{_ts()}] Phase 7: OuterStoreMode (gpu{gpu})", flush=True)
    print(
        f"  nhq={NHQ}, nhk={NHK}, hd={HD}, d1b, PackGQA, bf16\n",
        flush=True,
    )

    for pass_type, configs in CONFIGS.items():
        is_bwd = pass_type != "fwd"
        print(f"  ── {pass_type} ──", flush=True)

        for S in SEQLENS:
            for label, extra_kw, env_osm in configs:
                key = f"{pass_type}/{label}"

                if not force:
                    d = results.get(key, {})
                    if S in d.get("topk", []):
                        idx = d["topk"].index(S)
                        tf = d["tflops"][idx]
                        if tf is not None:
                            print(
                                f"    S={S:>6d} {label:>25s}: "
                                f"{tf:>7.1f} T (cached)",
                                flush=True,
                            )
                            continue

                gc.collect()
                torch.cuda.empty_cache()

                if env_osm is not None:
                    os.environ["MAGI_ATTENTION_FFA_OUTER_STORE_MODE"] = env_osm
                elif "MAGI_ATTENTION_FFA_OUTER_STORE_MODE" in os.environ:
                    del os.environ["MAGI_ATTENTION_FFA_OUTER_STORE_MODE"]

                try:
                    torch.manual_seed(42)
                    q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device=device)
                    k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
                    v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
                    if is_bwd:
                        q.requires_grad_(True)
                        k.requires_grad_(True)
                        v.requires_grad_(True)

                    q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
                    k_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
                    atm = torch.zeros(1, dtype=torch.int32, device=device)

                    kw = dict(
                        q_ranges=q_ranges,
                        k_ranges=k_ranges,
                        attn_type_map=atm,
                        pack_gqa=True,
                        disable_fwd_atomic_reduction=True,
                        **extra_kw,
                    )

                    flops = _calc_flops_dense(S, is_bwd)

                    t0 = time.time()
                    o, *_ = flex_flash_attn_func(q, k, v, **kw)

                    if is_bwd:
                        do = torch.randn_like(o)

                        def run_fn():
                            o.backward(do, retain_graph=True)

                    else:

                        def run_fn():
                            flex_flash_attn_func(q, k, v, **kw)

                    tf, ms = _bench_kernel(run_fn, flops, device)
                    elapsed = time.time() - t0
                    _set_entry(results, key, S, round(tf, 1), round(ms, 3))
                    print(
                        f"    S={S:>6d} {label:>25s}: "
                        f"{tf:>7.1f} T ({ms:.3f}ms, {elapsed:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    _set_entry(results, key, S, None, None)
                    print(
                        f"    S={S:>6d} {label:>25s}: FAIL - {e}",
                        flush=True,
                    )
                finally:
                    if "MAGI_ATTENTION_FFA_OUTER_STORE_MODE" in os.environ:
                        del os.environ["MAGI_ATTENTION_FFA_OUTER_STORE_MODE"]
                    q = k = v = None
                    gc.collect()
                    torch.cuda.empty_cache()

                _save_results(PHASE, results)

        print(flush=True)

    print(f"\n[{_ts()}] Phase 7 DONE -> {_results_path(PHASE)}", flush=True)
    _print_summary(results)


def _print_summary(results):
    """Print comparison table per pass_type."""
    for pass_type, configs in CONFIGS.items():
        labels = [c[0] for c in configs]
        hdr = " | ".join(f"{lb:>15s}" for lb in labels)
        print(f"\n  {pass_type}:")
        print(f"  {'SeqLen':>8s} | {hdr}")
        print(f"  {'-' * (10 + 18 * len(labels))}")
        for S in SEQLENS:
            vals = []
            for label, _, _ in configs:
                key = f"{pass_type}/{label}"
                d = results.get(key, {})
                if S in d.get("topk", []):
                    idx = d["topk"].index(S)
                    tf = d["tflops"][idx]
                    vals.append(f"{tf:>15.1f}" if tf else f"{'FAIL':>15s}")
                else:
                    vals.append(f"{'-':>15s}")
            print(f"  {S:>8d} | {' | '.join(vals)}")
    print()
