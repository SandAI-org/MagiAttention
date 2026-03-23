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

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="skip slow tests"
    )
    parser.addoption(
        "--test-attn-config",
        default=None,
        help="comma-separated attn_config names to run (supports fnmatch wildcards)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks a test as slow to run")

    attn_config_filter = config.getoption("--test-attn-config", default=None)
    if attn_config_filter is not None:
        os.environ["MAGI_ATTENTION_TEST_ATTN_CONFIG"] = attn_config_filter


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="skipped because --skip-slow was provided")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
