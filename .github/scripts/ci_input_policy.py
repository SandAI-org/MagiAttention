#!/usr/bin/env python3

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

"""Select tracked CI inputs from a package-local declarative policy.

Every exclusion is relative to its node's ``root``.  The policy itself is
represented by a canonical projection in every source digest, even when
``.github/**`` is excluded from the selected source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

SCHEMA_VERSION = 1
LAYERS = ("trigger", "wheel", "portable")
PROTOCOL_ID = "ci-input-policy-v1"
POLICY_NAME = "ci_input_policy.json"
POLICY_RELATIVE_PATH = Path(".github/configs") / POLICY_NAME


class PolicyError(ValueError):
    """Raised when a policy is invalid or unsafe."""


@dataclass(frozen=True)
class TrackedEntry:
    mode: str
    oid: str
    repo_path: PurePosixPath
    node_path: PurePosixPath

    @property
    def object_type(self) -> str:
        return "commit" if self.mode == "160000" else "blob"

    @property
    def is_gitlink(self) -> bool:
        return self.mode == "160000"


def _git(repo_root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def find_repo_root(start: Path | None = None) -> Path:
    cwd = Path.cwd() if start is None else start
    return Path(_git(cwd, "rev-parse", "--show-toplevel").decode().strip()).resolve()


def _safe_relative(value: str, field: str, *, allow_dot: bool) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{field} must be a non-empty string")
    if "\\" in value or value.startswith("/") or value.endswith("/"):
        raise PolicyError(
            f"{field} must be a normalized relative POSIX path: {value!r}"
        )
    path = PurePosixPath(value)
    if value == ".":
        if allow_dot:
            return path
        raise PolicyError(f"{field} may not be '.'")
    if (
        value.startswith("./")
        or "//" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PolicyError(
            f"{field} must be a normalized relative POSIX path: {value!r}"
        )
    return path


def _validate_pattern(value: Any, field: str) -> str:
    path = _safe_relative(value, field, allow_dot=False)
    if any(char in value for char in "[]{}!"):
        raise PolicyError(f"{field} uses an unsupported glob construct: {value!r}")
    if "***" in value:
        raise PolicyError(f"{field} contains an invalid '**' sequence: {value!r}")
    if any("**" in part and part != "**" for part in path.parts):
        raise PolicyError(
            f"{field} requires '**' to occupy a complete path segment: {value!r}"
        )
    return value


def load_policy(policy_path: Path) -> dict[str, Any]:
    policy_path = policy_path.resolve()
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"Cannot read policy {policy_path}: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"version", "nodes"}:
        raise PolicyError("policy must contain exactly 'version' and 'nodes'")
    if data["version"] != SCHEMA_VERSION:
        raise PolicyError(f"policy version must be {SCHEMA_VERSION}")
    nodes = data["nodes"]
    if not isinstance(nodes, dict) or not nodes:
        raise PolicyError("policy nodes must be a non-empty object")
    normalized: dict[str, Any] = {"version": SCHEMA_VERSION, "nodes": {}}
    for name, value in nodes.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise PolicyError(f"invalid node name: {name!r}")
        if not isinstance(value, dict) or set(value) != {"root", "exclusions"}:
            raise PolicyError(
                f"node {name} must contain exactly 'root' and 'exclusions'"
            )
        root = _safe_relative(value["root"], f"nodes.{name}.root", allow_dot=True)
        exclusions = value["exclusions"]
        if not isinstance(exclusions, dict) or set(exclusions) != set(LAYERS):
            raise PolicyError(
                f"nodes.{name}.exclusions must contain exactly {', '.join(LAYERS)}"
            )
        normalized_layers: dict[str, list[str]] = {}
        for layer in LAYERS:
            patterns = exclusions[layer]
            if not isinstance(patterns, list):
                raise PolicyError(f"nodes.{name}.exclusions.{layer} must be an array")
            checked = [
                _validate_pattern(pattern, f"nodes.{name}.exclusions.{layer}")
                for pattern in patterns
            ]
            if len(set(checked)) != len(checked):
                raise PolicyError(
                    f"nodes.{name}.exclusions.{layer} contains duplicates"
                )
            normalized_layers[layer] = sorted(checked, key=os.fsencode)
        normalized["nodes"][name] = {
            "root": root.as_posix(),
            "exclusions": normalized_layers,
        }
    return normalized


def package_root_for_policy(policy_path: Path) -> Path:
    policy_path = policy_path.resolve()
    if policy_path.name != POLICY_NAME:
        raise PolicyError(f"policy must be named {POLICY_NAME}: {policy_path}")
    if (
        policy_path.parent.name == "configs"
        and policy_path.parent.parent.name == ".github"
    ):
        return policy_path.parent.parent.parent
    raise PolicyError(
        f"policy must be named .github/configs/{POLICY_NAME}: {policy_path}"
    )


def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts = pattern.split("/")
    expression = ""
    for index, part in enumerate(parts):
        if index:
            expression += "/"
        if part == "**":
            expression += ".*"
            continue
        for char in part:
            if char == "*":
                expression += "[^/]*"
            elif char == "?":
                expression += "[^/]"
            else:
                expression += re.escape(char)
    return re.compile(f"^{expression}$")


def is_excluded(path: PurePosixPath, patterns: Iterable[str]) -> bool:
    value = path.as_posix()
    return any(_glob_regex(pattern).fullmatch(value) for pattern in patterns)


def _relative_to_node(
    repo_path: PurePosixPath,
    package_prefix: PurePosixPath,
    node_root: PurePosixPath,
) -> PurePosixPath | None:
    absolute_root = (
        package_prefix
        if node_root == PurePosixPath(".")
        else package_prefix / node_root
    )
    if repo_path == absolute_root:
        return PurePosixPath(repo_path.name)
    if absolute_root not in repo_path.parents:
        return None
    return repo_path.relative_to(absolute_root)


def _parse_entries(
    output: bytes, package_prefix: PurePosixPath, node_root: PurePosixPath
) -> list[TrackedEntry]:
    entries: list[TrackedEntry] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise PolicyError(f"unresolved merge conflict: {os.fsdecode(raw_path)}")
        repo_path = PurePosixPath(os.fsdecode(raw_path))
        node_path = _relative_to_node(repo_path, package_prefix, node_root)
        if node_path is not None:
            entries.append(TrackedEntry(mode, oid, repo_path, node_path))
    entries.sort(key=lambda entry: os.fsencode(entry.node_path.as_posix()))
    return entries


def tracked_entries(
    repo_root: Path,
    policy_path: Path,
    policy: dict[str, Any],
    node: str,
    *,
    ref: str | None = None,
) -> list[TrackedEntry]:
    if node not in policy["nodes"]:
        raise PolicyError(f"unknown policy node: {node}")
    package_root = package_root_for_policy(policy_path)
    try:
        package_prefix_value = package_root.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise PolicyError(
            f"policy package root is outside Git repository: {package_root}"
        ) from exc
    package_prefix = (
        PurePosixPath(".")
        if package_prefix_value == "."
        else PurePosixPath(package_prefix_value)
    )
    node_root = PurePosixPath(policy["nodes"][node]["root"])
    source = (
        package_prefix
        if node_root == PurePosixPath(".")
        else package_prefix / node_root
    )
    if ref is None:
        output = _git(repo_root, "ls-files", "--stage", "-z", "--", source.as_posix())
    else:
        output = _git(
            repo_root,
            "ls-tree",
            "-r",
            "-z",
            "--format=%(objectmode) %(objectname) 0%x09%(path)",
            ref,
            "--",
            source.as_posix(),
        )
    return _parse_entries(output, package_prefix, node_root)


def selected_entries(
    repo_root: Path,
    policy_path: Path,
    policy: dict[str, Any],
    node: str,
    layer: str,
    *,
    ref: str | None = None,
) -> list[TrackedEntry]:
    if layer not in LAYERS:
        raise PolicyError(f"unknown layer: {layer}")
    patterns = policy["nodes"][node]["exclusions"][layer]
    return [
        entry
        for entry in tracked_entries(repo_root, policy_path, policy, node, ref=ref)
        if not is_excluded(entry.node_path, patterns)
    ]


def _worktree_identity(repo_root: Path, entry: TrackedEntry) -> tuple[str, str] | None:
    source = repo_root / entry.repo_path
    if not source.exists() and not source.is_symlink():
        return None
    if source.is_symlink():
        content = os.fsencode(os.readlink(source))
        mode = "120000"
    elif source.is_file():
        content = source.read_bytes()
        mode = "100755" if source.stat().st_mode & 0o111 else "100644"
    else:
        raise PolicyError(f"tracked path is not a file, symlink, or gitlink: {source}")
    oid = (
        _git(
            repo_root,
            "hash-object",
            f"--path={entry.repo_path.as_posix()}",
            "--stdin",
            input_bytes=content,
        )
        .decode()
        .strip()
    )
    return mode, oid


def canonical_projection(policy: dict[str, Any], node: str, layer: str) -> bytes:
    value = {
        "exclusions": policy["nodes"][node]["exclusions"][layer],
        "layer": layer,
        "node": node,
        "protocol": PROTOCOL_ID,
        "root": policy["nodes"][node]["root"],
        "version": policy["version"],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def source_digest(
    repo_root: Path,
    policy_path: Path,
    policy: dict[str, Any],
    node: str,
    layer: str,
    *,
    ref: str | None = None,
) -> str:
    entries = selected_entries(repo_root, policy_path, policy, node, layer, ref=ref)
    if not entries:
        raise PolicyError(f"no tracked {layer} inputs found for node {node}")
    modified: set[PurePosixPath] = set()
    if ref is None:
        modified = {
            PurePosixPath(os.fsdecode(path))
            for path in _git(
                repo_root, "diff-files", "--name-only", "-z", "--ignore-submodules=all"
            ).split(b"\0")
            if path
        }
    records: list[bytes] = []
    for entry in entries:
        mode, oid = entry.mode, entry.oid
        if ref is None and not entry.is_gitlink and entry.repo_path in modified:
            identity = _worktree_identity(repo_root, entry)
            if identity is None:
                continue
            mode, oid = identity
        kind = "commit" if mode == "160000" else "blob"
        records.append(f"{mode} {kind} {oid}\t{entry.node_path.as_posix()}".encode())
    payload = (
        b"policy\0"
        + canonical_projection(policy, node, layer)
        + b"\0entries\0"
        + b"\n".join(records)
        + b"\n"
    )
    return hashlib.sha256(payload).hexdigest()


def validate_policy(repo_root: Path, policy_path: Path, policy: dict[str, Any]) -> None:
    package_root = package_root_for_policy(policy_path)
    roots = {
        name: PurePosixPath(value["root"]) for name, value in policy["nodes"].items()
    }
    for name, root in roots.items():
        if not (
            package_root if root == PurePosixPath(".") else package_root / root
        ).is_dir():
            raise PolicyError(f"node root does not exist for {name}: {root}")
    for parent_name, parent_root in roots.items():
        for child_name, child_root in roots.items():
            if parent_name == child_name or parent_root == child_root:
                continue
            parent_abs = (
                PurePosixPath() if parent_root == PurePosixPath(".") else parent_root
            )
            if parent_abs not in child_root.parents and parent_root != PurePosixPath(
                "."
            ):
                continue
            relative = (
                child_root
                if parent_root == PurePosixPath(".")
                else child_root.relative_to(parent_root)
            )
            required = relative.as_posix() + "/**"
            for layer in LAYERS:
                patterns = policy["nodes"][parent_name]["exclusions"][layer]
                if required not in patterns:
                    raise PolicyError(
                        f"overlapping node {parent_name} must exclude {required!r} from {layer}"
                    )
    for name in policy["nodes"]:
        selected = {
            layer: {
                entry.repo_path
                for entry in selected_entries(
                    repo_root, policy_path, policy, name, layer
                )
            }
            for layer in LAYERS
        }
        if not selected["wheel"] <= selected["trigger"]:
            paths = sorted(
                path.as_posix() for path in selected["wheel"] - selected["trigger"]
            )
            raise PolicyError(f"wheel inputs outside trigger for {name}: {paths[:5]}")
        if not selected["portable"] <= selected["trigger"]:
            paths = sorted(
                path.as_posix() for path in selected["portable"] - selected["trigger"]
            )
            raise PolicyError(
                f"portable inputs outside trigger for {name}: {paths[:5]}"
            )
        for layer in LAYERS:
            if not selected[layer]:
                raise PolicyError(f"no tracked {layer} inputs found for node {name}")


def changed(
    repo_root: Path,
    policy_path: Path,
    policy: dict[str, Any],
    node: str,
    layer: str,
    base: str,
    head: str,
) -> bool:
    if node not in policy["nodes"] or layer not in LAYERS:
        raise PolicyError(f"unknown node/layer: {node}/{layer}")
    output = _git(
        repo_root, "diff", "--no-renames", "--name-only", "-z", base, head, "--"
    )
    changed_paths = [
        PurePosixPath(os.fsdecode(value)) for value in output.split(b"\0") if value
    ]
    policy_repo_path = PurePosixPath(
        policy_path.resolve().relative_to(repo_root).as_posix()
    )
    package_root = package_root_for_policy(policy_path)
    package_prefix_value = package_root.relative_to(repo_root).as_posix()
    package_prefix = (
        PurePosixPath(".")
        if package_prefix_value == "."
        else PurePosixPath(package_prefix_value)
    )
    node_root = PurePosixPath(policy["nodes"][node]["root"])
    patterns = policy["nodes"][node]["exclusions"][layer]
    for path in changed_paths:
        if (
            path == policy_repo_path
            or path.as_posix().endswith("/.github/scripts/ci_input_policy.py")
            or path == PurePosixPath("tools/ci_input_policy.py")
        ):
            return True
        relative = _relative_to_node(path, package_prefix, node_root)
        if relative is not None and not is_excluded(relative, patterns):
            return True
    return False


def _default_policy(repo_root: Path) -> Path:
    return repo_root / POLICY_RELATIVE_PATH


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--policy")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    digest_parser = subparsers.add_parser("digest")
    digest_parser.add_argument("--layer", choices=("wheel", "portable"), required=True)
    digest_parser.add_argument("--node", required=True)
    changed_parser = subparsers.add_parser("changed")
    changed_parser.add_argument("--layer", choices=("trigger",), required=True)
    changed_parser.add_argument("--node", required=True)
    changed_parser.add_argument("--base", required=True)
    changed_parser.add_argument("--head", required=True)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    policy_path = (
        Path(args.policy).resolve() if args.policy else _default_policy(repo_root)
    )
    policy = load_policy(policy_path)
    if args.command == "validate":
        validate_policy(repo_root, policy_path, policy)
        return 0
    validate_policy(repo_root, policy_path, policy)
    if args.command == "digest":
        print(source_digest(repo_root, policy_path, policy, args.node, args.layer))
    elif args.command == "changed":
        print(
            "true"
            if changed(
                repo_root,
                policy_path,
                policy,
                args.node,
                args.layer,
                args.base,
                args.head,
            )
            else "false"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PolicyError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ci input policy error: {exc}") from exc
