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

"""CLI entry point for bench_sparse_analysis package.

Usage:
  python -m bench_sparse_analysis --exp  0-method-parity
  python -m bench_sparse_analysis --plot 4-loopk-debug
  python -m bench_sparse_analysis --ncu  2-kbs-compare
"""

import argparse

from bench_sparse_analysis._common import PHASES, _parse_rerun, _ts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp", choices=PHASES, help="Run benchmark experiment")
    group.add_argument("--plot", choices=PHASES, help="Generate plot from results")
    group.add_argument("--ncu", choices=PHASES, help="Run NCU profiling")
    parser.add_argument(
        "--force", action="store_true", help="Re-run all (ignore cache)"
    )
    parser.add_argument(
        "--rerun",
        type=str,
        default=None,
        help="Re-run subset: 'pass/method' or 'pass/method/topk', comma-separated",
    )
    parser.add_argument(
        "--max-kvseqlen",
        type=str,
        default=None,
        help="Max kvseqlen to test (e.g. '256k', '512k', or raw int). "
        "Applies to phase5/phase6.",
    )
    parser.add_argument(
        "--iss",
        action="store_true",
        help="Run ISS (InnerStoreStages) sub-benchmark for phase4",
    )

    args = parser.parse_args()
    rerun_filter = _parse_rerun(args.rerun) if args.rerun else None

    max_kvseqlen = None
    if args.max_kvseqlen:
        s = args.max_kvseqlen.lower().strip()
        if s.endswith("k"):
            max_kvseqlen = int(s[:-1]) * 1024
        else:
            max_kvseqlen = int(s)

    if args.exp:
        phase = args.exp
        print(f"[{_ts()}] === --exp {phase} ===", flush=True)
        if phase == "0-method-parity":
            from bench_sparse_analysis.phase0_method_parity import _phase0_bench

            _phase0_bench(force=args.force, rerun_filter=rerun_filter)
        elif phase == "1-topk-sweep":
            from bench_sparse_analysis.phase1_topk_sweep import _phase1_bench

            _phase1_bench(force=args.force, rerun_filter=rerun_filter)
        elif phase == "2-kbs-compare":
            from bench_sparse_analysis.phase2_kbs_compare import _phase2_bench

            _phase2_bench(force=args.force)
        elif phase == "3-l2-inflection":
            parser.error("Phase 3 has no --exp. Use --ncu 3-l2-inflection")
        elif phase == "4-loopk-debug":
            if args.iss:
                from bench_sparse_analysis.phase4_loopk_debug import _phase4_iss_bench

                _phase4_iss_bench(force=args.force)
            else:
                from bench_sparse_analysis.phase4_loopk_debug import _phase4_bench

                _phase4_bench(force=args.force)
        elif phase == "4_1-skip-ablation":
            from bench_sparse_analysis.phase4_1_skip_ablation import _phase4_1_bench

            _phase4_1_bench(force=args.force)
        elif phase == "4_2-iss-double-buffer":
            from bench_sparse_analysis.phase4_2_iss_double_buffer import _phase4_2_bench

            _phase4_2_bench(force=args.force)
        elif phase == "5-scaling":
            from bench_sparse_analysis.phase5_scaling import _phase5_bench

            _phase5_bench(force=args.force, max_kvseqlen=max_kvseqlen)
        elif phase == "6-video-production":
            from bench_sparse_analysis.phase6_video_production import _phase6_bench

            _phase6_bench(
                force=args.force,
                max_kvseqlen=max_kvseqlen,
                rerun_filter=rerun_filter,
            )
        elif phase == "7-outer-store-mode":
            from bench_sparse_analysis.phase7_outer_store_mode import _phase7_bench

            _phase7_bench(force=args.force)
    elif args.plot:
        phase = args.plot
        print(f"[{_ts()}] === --plot {phase} ===", flush=True)
        if phase == "0-method-parity":
            from bench_sparse_analysis.phase0_method_parity import _phase0_plot

            _phase0_plot()
        elif phase == "1-topk-sweep":
            from bench_sparse_analysis.phase1_topk_sweep import _phase1_plot

            _phase1_plot()
        elif phase == "2-kbs-compare":
            from bench_sparse_analysis.phase2_kbs_compare import _phase2_plot

            _phase2_plot()
        elif phase == "3-l2-inflection":
            parser.error("Phase 3 has no --plot. Use --ncu 3-l2-inflection")
        elif phase == "4-loopk-debug":
            if args.iss:
                from bench_sparse_analysis.phase4_loopk_debug import _phase4_iss_plot

                _phase4_iss_plot()
            else:
                from bench_sparse_analysis.phase4_loopk_debug import (
                    _phase4_summary_plot,
                )

                _phase4_summary_plot()
        elif phase == "4_1-skip-ablation":
            from bench_sparse_analysis.phase4_1_skip_ablation import _phase4_1_plot

            _phase4_1_plot()
        elif phase == "4_2-iss-double-buffer":
            from bench_sparse_analysis.phase4_2_iss_double_buffer import _phase4_2_plot

            _phase4_2_plot()
        elif phase == "5-scaling":
            from bench_sparse_analysis.phase5_scaling import _phase5_plot

            _phase5_plot()
        elif phase == "6-video-production":
            from bench_sparse_analysis.phase6_video_production import _phase6_plot

            _phase6_plot()
    elif args.ncu:
        phase = args.ncu
        print(f"[{_ts()}] === --ncu {phase} ===", flush=True)
        if phase == "0-method-parity":
            from bench_sparse_analysis.phase0_method_parity import _phase0_ncu

            _phase0_ncu()
        elif phase == "1-topk-sweep":
            parser.error("Phase 1 has no --ncu")
        elif phase == "2-kbs-compare":
            from bench_sparse_analysis.phase2_kbs_compare import _phase2_ncu

            _phase2_ncu()
        elif phase == "3-l2-inflection":
            from bench_sparse_analysis.phase3_l2_inflection import _phase3_ncu

            _phase3_ncu()
        elif phase == "4-loopk-debug":
            from bench_sparse_analysis.phase4_loopk_debug import _phase4_ncu

            _phase4_ncu()
        elif phase == "4_1-skip-ablation":
            from bench_sparse_analysis.phase4_1_skip_ablation import _phase4_1_ncu

            _phase4_1_ncu()
        elif phase == "4_2-iss-double-buffer":
            parser.error("Phase 4_2 has no --ncu")
        elif phase == "5-scaling":
            from bench_sparse_analysis.phase5_scaling import _phase5_ncu

            _phase5_ncu()
        elif phase == "6-video-production":
            from bench_sparse_analysis.phase6_video_production import _phase6_ncu

            _phase6_ncu()

    print(f"\n[{_ts()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
