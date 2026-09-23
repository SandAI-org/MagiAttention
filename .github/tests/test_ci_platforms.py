#!/usr/bin/env python3

# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci_platforms.py"
SPEC = importlib.util.spec_from_file_location("ci_platforms", SCRIPT)
assert SPEC and SPEC.loader
platforms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(platforms)


def main() -> None:
    config = platforms.load_config()
    assert set(config["platforms"]) == {"h100", "b300"}
    assert "NCCL_NVLS_ENABLE" in config["platforms"]["h100"]["test_environment"]
    assert "NCCL_NVLS_ENABLE" not in config["platforms"]["b300"]["test_environment"]
    assert (
        config["platforms"]["h100"]["test_environment"]["MAGI_ATTENTION_TEST_BACKEND"]
        == "ffa,sdpa"
    )
    assert (
        config["platforms"]["b300"]["test_environment"]["MAGI_ATTENTION_TEST_BACKEND"]
        == "fa4,cutedsl,sdpa"
    )
    assert (
        config["platforms"]["b300"]["test_environment"][
            "MAGI_ATTENTION_TEST_FLASH_ATTN_CUTE_ARCHS"
        ]
        == "sm100"
    )
    assert (
        "MAGI_ATTENTION_TEST_FLASH_ATTN_CUTE_ARCHS"
        not in config["platforms"]["h100"]["test_environment"]
    )
    assert (
        config["platforms"]["h100"]["test_environment"]["PIP_INDEX_URL"]
        == "https://pypi.tuna.tsinghua.edu.cn/simple/"
    )
    assert (
        config["platforms"]["b300"]["test_environment"]["PIP_INDEX_URL"]
        == "https://mirrors.aliyun.com/pypi/simple/"
    )

    original = os.environ.get("B300_CI_ENABLED")
    try:
        os.environ["B300_CI_ENABLED"] = "false"
        assert platforms.matrix() == {"include": [{"platform": "h100"}]}
        os.environ["B300_CI_ENABLED"] = "true"
        assert platforms.matrix() == {
            "include": [{"platform": "h100"}, {"platform": "b300"}]
        }
    finally:
        if original is None:
            os.environ.pop("B300_CI_ENABLED", None)
        else:
            os.environ["B300_CI_ENABLED"] = original

    print("CI platform tests passed")


if __name__ == "__main__":
    main()
