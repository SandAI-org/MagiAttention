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

import cutlass
import pytest
import quack
from cutlass.base_dsl.common import DSLCudaVersion

from magi_attention.kernel.cutedsl import cache_utils


@pytest.fixture
def fingerprint(monkeypatch):
    """Recompute the cached fingerprint after each patch and restore it afterwards."""

    def compute() -> str:
        cache_utils._compute_source_fingerprint.cache_clear()
        return cache_utils._compute_source_fingerprint()

    yield compute
    monkeypatch.undo()
    cache_utils._compute_source_fingerprint.cache_clear()


def test_fingerprint_is_stable_for_unchanged_inputs(fingerprint):
    assert fingerprint() == fingerprint()


def test_fingerprint_changes_with_quack_version(fingerprint, monkeypatch):
    baseline = fingerprint()
    monkeypatch.setattr(quack, "__version__", "0.0.0+other")
    assert fingerprint() != baseline


def test_fingerprint_changes_with_dsl_cuda_backend(fingerprint, monkeypatch):
    baseline = fingerprint()
    # Same DSL release, other wheel: cu12 (12.9 compiler) versus cu13.
    other = DSLCudaVersion("12.9" if cutlass.CUDA_VERSION.major >= 13 else "13.0")
    monkeypatch.setattr(cutlass, "CUDA_VERSION", other)
    assert fingerprint() != baseline
