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

"""Phase 4: loopk-debug — backward-compat shim.

Split into:
  phase4_1_skip_ablation.py    — LoopK vs LoopQ gap analysis (skip flags, structural, symmetry)
  phase4_2_iss_double_buffer.py — InnerStoreStages / union / tile ablation
"""

from bench_sparse_analysis.phase4_1_skip_ablation import (  # noqa: F401
    ALL_CONFIGS as _DEBUG_CONFIGS,
)
from bench_sparse_analysis.phase4_1_skip_ablation import (
    _phase4_1_plot as _phase4_summary_plot,
)
from bench_sparse_analysis.phase4_2_iss_double_buffer import (  # noqa: F401
    _phase4_2_bench as _phase4_iss_bench,
)


def _phase4_plot():
    """Deprecated: use _phase4_summary_plot() (phase4_1) or _phase4_iss_plot() (phase4_2)."""
    print("[SKIP] _phase4_plot() deprecated — use phase4_1/phase4_2 directly.")


def _phase4_opt_plot():
    """Deprecated: merged into _phase4_summary_plot()."""
    _phase4_summary_plot()
