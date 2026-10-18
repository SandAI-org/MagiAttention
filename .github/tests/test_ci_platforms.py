#!/usr/bin/env python3

# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci_platforms.py"
SPEC = importlib.util.spec_from_file_location("ci_platforms", SCRIPT)
assert SPEC and SPEC.loader
platforms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(platforms)


def main() -> None:
    config = platforms.load_config()
    assert set(config["platforms"]) == {"h100", "b300", "b300_fa4"}
    environments = {
        name: profile["test_environment"]
        for name, profile in config["platforms"].items()
    }
    h100 = environments["h100"]
    b300 = environments["b300"]
    b300_fa4 = environments["b300_fa4"]

    assert config["platforms"]["h100"]["runner_label"] == "h100"
    assert config["platforms"]["b300"]["runner_label"] == "b300"
    assert config["platforms"]["b300_fa4"]["runner_label"] == "b300"

    assert "NCCL_NVLS_ENABLE" in h100
    assert "NCCL_NVLS_ENABLE" not in b300
    assert "NCCL_NVLS_ENABLE" not in b300_fa4
    assert h100["MAGI_ATTENTION_TEST_BACKEND"] == "ffa,sdpa"
    assert b300["MAGI_ATTENTION_TEST_BACKEND"] == "cutedsl,sdpa"
    assert b300_fa4["MAGI_ATTENTION_TEST_BACKEND"] == "fa4,cutedsl"
    assert h100["PIP_INDEX_URL"] == "https://pypi.tuna.tsinghua.edu.cn/simple/"
    assert b300["PIP_INDEX_URL"] == "https://mirrors.aliyun.com/pypi/simple/"
    assert b300_fa4["PIP_INDEX_URL"] == "https://mirrors.aliyun.com/pypi/simple/"

    # Only the fa4 profile installs flash_attn_cute and precompiles FFA_FA4
    # kernels; the precompile script fails to import with the newest DSL.
    assert b300_fa4["MAGI_ATTENTION_TEST_FLASH_ATTN_CUTE_ARCHS"] == "sm100"
    assert "MAGI_ATTENTION_TEST_FLASH_ATTN_CUTE_ARCHS" not in h100
    assert "MAGI_ATTENTION_TEST_FLASH_ATTN_CUTE_ARCHS" not in b300

    # The DSL pair is the pinned constraint for fa4 and the upper bound of
    # requirements.txt otherwise.
    assert [
        name
        for name, environment in environments.items()
        if "PIP_CONSTRAINT" in environment
    ] == ["b300_fa4"]
    assert b300_fa4["PIP_CONSTRAINT"] == ".github/configs/ci_constraints.txt"

    # The profile constraint file is appended to the PIP_CONSTRAINT of the
    # base image, once, and other variables keep their profile values.
    image = "/etc/pip/constraint.txt"
    both = f"{image} {b300_fa4['PIP_CONSTRAINT']}"
    for inherited, constraint in (
        (None, b300_fa4["PIP_CONSTRAINT"]),
        ("", b300_fa4["PIP_CONSTRAINT"]),
        (image, both),
        (both, both),
    ):
        with mock.patch.dict(os.environ):
            os.environ.pop("PIP_CONSTRAINT", None)
            if inherited is not None:
                os.environ["PIP_CONSTRAINT"] = inherited
            assert platforms.test_environment("b300_fa4") == {
                **b300_fa4,
                "PIP_CONSTRAINT": constraint,
            }
            assert platforms.test_environment("h100") == h100
            assert platforms.test_environment("b300") == b300

    assert "MAGI_ATTENTION_FFA_CUTEDSL_CACHE_ENABLED" not in h100
    assert b300["MAGI_ATTENTION_FFA_CUTEDSL_CACHE_ENABLED"] == "1"
    assert b300_fa4["MAGI_ATTENTION_FFA_CUTEDSL_CACHE_ENABLED"] == "1"

    # b300 runs the newest DSL, under which the FA4 extension tests cannot run.
    assert h100["CI_TEST_EXTENSIONS"] == "true"
    assert b300["CI_TEST_EXTENSIONS"] == "false"
    assert b300_fa4["CI_TEST_EXTENSIONS"] == "true"

    original = os.environ.get("B300_CI_ENABLED")
    try:
        os.environ["B300_CI_ENABLED"] = "false"
        assert platforms.matrix() == {
            "include": [{"platform": "h100", "runner_label": "h100"}]
        }
        os.environ["B300_CI_ENABLED"] = "true"
        assert platforms.matrix() == {
            "include": [
                {"platform": "h100", "runner_label": "h100"},
                {"platform": "b300", "runner_label": "b300"},
                {"platform": "b300_fa4", "runner_label": "b300"},
            ]
        }
    finally:
        if original is None:
            os.environ.pop("B300_CI_ENABLED", None)
        else:
            os.environ["B300_CI_ENABLED"] = original

    # runner_label must name a runner label, not the profile.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ci_platforms.json"
        invalid = json.loads(platforms.config_path().read_text(encoding="utf-8"))
        invalid["platforms"]["b300_fa4"]["runner_label"] = "b300_fa4"
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with mock.patch.object(platforms, "config_path", return_value=path):
            try:
                platforms.load_config()
            except platforms.ConfigurationError as exc:
                assert "runner_label" in str(exc)
            else:
                raise AssertionError("unknown runner label was accepted")

    print("CI platform tests passed")


if __name__ == "__main__":
    main()
