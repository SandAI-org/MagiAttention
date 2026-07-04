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

"""Sparse attention benchmark & analysis — modular package.

Phases:
  0-method-parity  : 5 methods at S=topk (sparse framework overhead baseline)
  1-topk-sweep     : Fixed S=32K varying topk (L2 cache / CTA starvation)
  2-kbs-compare    : kbs=1 (CpAsync) vs kbs=128 (TMA2D) TFLOPS (FWD+BWD)
  3-l2-inflection  : NCU at specific TFLOPS inflection points
  4-loopk-debug    : LoopK vs LoopQ gap analysis with perf-debug skip flags

Usage:
  python -m bench_sparse_analysis --exp  0-method-parity
  python -m bench_sparse_analysis --plot 4-loopk-debug
"""
