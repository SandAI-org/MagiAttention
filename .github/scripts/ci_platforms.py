#!/usr/bin/env python3

# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Validate package-owned CI platform profiles and plan the test matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PLATFORM_RE = re.compile(r"[a-z][a-z0-9_-]*")
ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*")


class ConfigurationError(ValueError):
    """Raised when the platform configuration is malformed."""


def config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs/ci_platforms.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "platforms"}:
        raise ConfigurationError("config must contain schema_version and platforms")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ConfigurationError(f"schema_version must be {SCHEMA_VERSION}")
    platforms = data["platforms"]
    if not isinstance(platforms, dict) or set(platforms) != {"h100", "b300"}:
        raise ConfigurationError("platforms must define exactly h100 and b300")
    for name, profile in platforms.items():
        if not PLATFORM_RE.fullmatch(name):
            raise ConfigurationError(f"invalid platform name: {name!r}")
        if not isinstance(profile, dict) or set(profile) != {
            "runner_label",
            "test_environment",
        }:
            raise ConfigurationError(
                f"platform {name} must define runner_label and test_environment"
            )
        if profile["runner_label"] != name:
            raise ConfigurationError(f"platform {name} runner_label must be {name!r}")
        environment = profile["test_environment"]
        if not isinstance(environment, dict):
            raise ConfigurationError(
                f"platform {name} test_environment must be an object"
            )
        for variable, value in environment.items():
            if not isinstance(variable, str) or not ENV_NAME_RE.fullmatch(variable):
                raise ConfigurationError(f"invalid environment variable: {variable!r}")
            if not isinstance(value, str) or "\n" in value:
                raise ConfigurationError(
                    f"platform {name} environment value {variable} must be a single-line string"
                )
    return data


def platform_profile(platform: str) -> dict[str, Any]:
    try:
        return load_config()["platforms"][platform]
    except KeyError as exc:
        raise ConfigurationError(f"unsupported CI platform: {platform}") from exc


def enabled_platforms() -> list[str]:
    enabled = os.environ.get("B300_CI_ENABLED", "false").strip().lower()
    if enabled not in {"true", "false"}:
        raise ConfigurationError("B300_CI_ENABLED must be 'true' or 'false'")
    return ["h100", "b300"] if enabled == "true" else ["h100"]


def matrix() -> dict[str, list[dict[str, str]]]:
    load_config()
    return {"include": [{"platform": platform} for platform in enabled_platforms()]}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("matrix")
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("platform")
    environment_parser = subparsers.add_parser("environment")
    environment_parser.add_argument("platform")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            load_config()
        elif args.command == "matrix":
            print(json.dumps(matrix(), sort_keys=True, separators=(",", ":")))
        elif args.command == "profile":
            print(
                json.dumps(
                    platform_profile(args.platform),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        elif args.command == "environment":
            for name, value in sorted(
                platform_profile(args.platform)["test_environment"].items()
            ):
                print(f"{name}={value}")
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
