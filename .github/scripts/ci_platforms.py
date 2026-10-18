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
# Platforms in matrix order. A platform is a test profile, and several
# profiles may run on one runner label.
PLATFORMS = ("h100", "b300", "b300_fa4")
RUNNER_LABELS = frozenset({"h100", "b300"})


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
    if not isinstance(platforms, dict) or set(platforms) != set(PLATFORMS):
        raise ConfigurationError(
            f"platforms must define exactly {', '.join(PLATFORMS)}"
        )
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
        if profile["runner_label"] not in RUNNER_LABELS:
            raise ConfigurationError(
                f"platform {name} runner_label must be one of {sorted(RUNNER_LABELS)}"
            )
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


def test_environment(platform: str) -> dict[str, str]:
    """Variables to export for ``platform``.
    A profile PIP_CONSTRAINT is appended to a non-empty inherited
    PIP_CONSTRAINT, so a constraint file of the base image stays in force; pip
    reads the value as whitespace-separated constraint files. A file that the
    inherited value already names is not added again, because a test job
    applies the profile through GITHUB_ENV and again in portable_validation.sh."""
    environment = dict(platform_profile(platform)["test_environment"])
    profile_constraint = environment.get("PIP_CONSTRAINT")
    inherited = os.environ.get("PIP_CONSTRAINT", "").split()
    if profile_constraint is not None and inherited:
        if profile_constraint not in inherited:
            inherited.append(profile_constraint)
        environment["PIP_CONSTRAINT"] = " ".join(inherited)
    return environment


def enabled_platforms(platforms: dict[str, Any]) -> list[str]:
    """Platforms of the validated ``platforms`` config to schedule.
    B300_CI_ENABLED gates every platform that runs on the b300 runner label,
    because GitHub keeps a job without a matching runner queued instead of
    skipping it."""
    enabled = os.environ.get("B300_CI_ENABLED", "false").strip().lower()
    if enabled not in {"true", "false"}:
        raise ConfigurationError("B300_CI_ENABLED must be 'true' or 'false'")
    return [
        platform
        for platform in PLATFORMS
        if enabled == "true" or platforms[platform]["runner_label"] != "b300"
    ]


def matrix() -> dict[str, list[dict[str, str]]]:
    platforms = load_config()["platforms"]
    return {
        "include": [
            {"platform": platform, "runner_label": platforms[platform]["runner_label"]}
            for platform in enabled_platforms(platforms)
        ]
    }


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
            for name, value in sorted(test_environment(args.platform).items()):
                print(f"{name}={value}")
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
