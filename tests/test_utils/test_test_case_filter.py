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

import pytest

from magi_attention.common.enum import MagiAttentionKernelBackend
from magi_attention.testing.dist_common import should_run_test_case


class TestShouldRunTestCaseBackendFilter:
    def test_canonical_lowercase_value_matches_ffa(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MAGI_ATTENTION_TEST_BACKEND", "ffa")
        assert should_run_test_case(backend=MagiAttentionKernelBackend.FFA)
        assert not should_run_test_case(backend=MagiAttentionKernelBackend.SDPA)

    def test_sdpa_does_not_match_sdpa_ol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAGI_ATTENTION_TEST_BACKEND", "sdpa")
        assert should_run_test_case(backend=MagiAttentionKernelBackend.SDPA)
        assert not should_run_test_case(backend=MagiAttentionKernelBackend.SDPA_OL)
