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

"""Resolve source dependencies and install verified content-addressed wheels."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

CACHE_SCHEMA = 1


def git(args: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    command = [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        "http.https://github.com/.extraheader=",
    ]
    token = os.environ.get("DEPENDENCY_REPO_TOKEN", "")
    if token and os.environ.get("CI_DEPENDENCY_USE_TOKEN") == "true":
        credential = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        command += [
            "-c",
            f"http.https://github.com/.extraheader=AUTHORIZATION: basic {credential}",
        ]
    result = subprocess.run(
        command + args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def resolve(url: str, ref_type: str, ref: str) -> str:
    if ref_type == "commit":
        if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
            raise SystemExit(f"Commit refs must be full 40-character SHAs: {ref!r}")
        return ref.lower()
    queries = (
        [f"refs/heads/{ref}"]
        if ref_type == "branch"
        else [f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}"]
        if ref_type == "tag"
        else []
    )
    for query in queries:
        lines = git(["ls-remote", url, query], capture=True).splitlines()
        if len(lines) == 1:
            return lines[0].split()[0]
    raise SystemExit(f"Cannot resolve {ref_type}:{ref} in {url}")


def checked_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise SystemExit(f"Invalid {label}: {value!r}")
    return value


def checked_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"Invalid dependency install path: {value!r}")
    return value


def source_digest(checkout: Path, relative: str) -> str:
    output = subprocess.check_output(
        ["git", "-C", str(checkout), "ls-files", "--stage", "-z", "--", relative]
    )
    prefix = PurePosixPath(relative)
    records: list[bytes] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        if stage != "0":
            raise SystemExit("Cannot cache dependency with unresolved merge conflicts")
        path = PurePosixPath(os.fsdecode(raw_path))
        normalized = path if relative == "." else path.relative_to(prefix)
        records.append(f"{mode} {oid}\t{normalized.as_posix()}".encode())
    if not records:
        raise SystemExit(f"No tracked source inputs found at {checkout / relative}")
    return hashlib.sha256(b"\n".join(sorted(records)) + b"\n").hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_inputs(
    *,
    consumer: str,
    dependency_id: str,
    repository: str,
    relative: str,
    source: str,
    base_tag: str,
    platform: str,
    recipe: str,
) -> dict[str, object]:
    return {
        "base_image_tag": base_tag,
        "consumer_namespace": consumer,
        "dependency_id": dependency_id,
        "install_path": relative,
        "platform": platform,
        "recipe_digest": recipe,
        "repository": repository,
        "schema_version": CACHE_SCHEMA,
        "source_digest": source,
    }


def fingerprint(inputs: dict[str, object]) -> str:
    return f"v{CACHE_SCHEMA}-{hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}"


def verify_artifact(
    directory: Path, inputs: dict[str, object], identity: str
) -> list[Path]:
    manifest = directory / "manifest.json"
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
    ):
        raise ValueError(f"Invalid dependency artifact layout: {directory}")
    data = json.loads(manifest.read_text())
    if (
        data.get("fingerprint") != identity
        or data.get("inputs") != inputs
        or data.get("status") != "built"
    ):
        raise ValueError(f"Incompatible dependency artifact manifest: {manifest}")
    declared = data.get("wheels")
    if not isinstance(declared, list) or not declared:
        raise ValueError(f"Dependency artifact has no declared wheels: {manifest}")
    wheels: list[Path] = []
    for item in declared:
        if not isinstance(item, dict) or set(item) != {"filename", "sha256"}:
            raise ValueError(f"Invalid wheel entry in {manifest}")
        name = item["filename"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".whl")
        ):
            raise ValueError(f"Invalid wheel filename in {manifest}: {name!r}")
        wheel = directory / name
        if (
            wheel.is_symlink()
            or not wheel.is_file()
            or file_sha256(wheel) != item["sha256"]
        ):
            raise ValueError(f"Dependency wheel failed verification: {wheel}")
        wheels.append(wheel)
    if {path.name for path in directory.glob("*.whl")} != {
        path.name for path in wheels
    }:
        raise ValueError(f"Undeclared dependency wheels in {directory}")
    return wheels


def install_wheels(wheels: list[Path]) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            *map(str, wheels),
        ],
        check=True,
    )


def build_or_reuse(
    *,
    checkout: Path,
    relative: str,
    dependency_id: str,
    repository: str,
    consumer: str,
    cache_root: Path,
    base_tag: str,
    platform: str,
    recipe: str,
) -> tuple[str, Path]:
    source = source_digest(checkout, relative)
    inputs = expected_inputs(
        consumer=consumer,
        dependency_id=dependency_id,
        repository=repository,
        relative=relative,
        source=source,
        base_tag=base_tag,
        platform=platform,
        recipe=recipe,
    )
    identity = fingerprint(inputs)
    unit = (
        "root"
        if relative == "."
        else checked_name(relative.replace("/", "-"), "cache unit")
    )
    target = cache_root / dependency_id / unit / identity
    if target.exists():
        install_wheels(verify_artifact(target, inputs, identity))
        print(f"Reused dependency wheel artifact: {target}")
        return source, target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".build.", dir=target.parent))
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(staging),
                str(checkout / relative),
            ],
            check=True,
        )
        wheels = sorted(staging.glob("*.whl"))
        if not wheels:
            raise SystemExit(
                f"Dependency build produced no wheel: {checkout / relative}"
            )
        manifest = {
            "fingerprint": identity,
            "inputs": inputs,
            "resolved_commit": git(["rev-parse", "HEAD"], cwd=checkout, capture=True),
            "status": "built",
            "wheels": [
                {"filename": wheel.name, "sha256": file_sha256(wheel)}
                for wheel in wheels
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_artifact(staging, inputs, identity)
        try:
            staging.rename(target)
        except FileExistsError:
            shutil.rmtree(staging)
        install_wheels(verify_artifact(target, inputs, identity))
        print(f"Published dependency wheel artifact: {target}")
        return source, target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(
        os.environ.get(
            "CI_DEPENDENCY_CONFIG", repo_root / ".github/ci_dependencies.json"
        )
    )
    checkout_root = Path(
        os.environ.get(
            "CI_DEPENDENCY_ROOT",
            Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "ci-source-dependencies",
        )
    )
    lock_path = Path(
        os.environ.get("CI_DEPENDENCY_RUNTIME_LOCK", checkout_root / "resolved.json")
    )
    consumer = checked_name(
        os.environ.get("STANDALONE_ARTIFACT_NAMESPACE", repo_root.name),
        "consumer namespace",
    )
    cache_root = (
        Path(os.environ.get("CI_WORKSPACE_ROOT", "/workspace"))
        / "v2/dependency-artifacts"
        / consumer
    )
    base_tag = (repo_root / ".github/workflows/base_image_tag.txt").read_text().strip()
    platform = checked_name(os.environ.get("TASK_CI_PLATFORM", "h100"), "platform")
    recipe = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1 or not isinstance(
        config.get("repositories"), list
    ):
        raise SystemExit("Unsupported CI dependency schema")
    checkout_root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, object] = {"schema_version": 1, "repositories": {}}
    for dependency in config["repositories"]:
        dependency_id = checked_name(dependency["id"], "dependency id")
        repository = dependency["repository"]
        commit = resolve(
            f"https://github.com/{repository}.git",
            dependency["ref_type"],
            dependency["ref"],
        )
        checkout = checkout_root / dependency_id
        shutil.rmtree(checkout, ignore_errors=True)
        git(["init", str(checkout)])
        git(
            ["remote", "add", "origin", f"https://github.com/{repository}.git"],
            cwd=checkout,
        )
        git(["fetch", "--depth=1", "origin", commit], cwd=checkout)
        git(["checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
        if git(["rev-parse", "HEAD"], cwd=checkout, capture=True) != commit:
            raise SystemExit(f"Checkout identity mismatch for {repository}")
        git(["submodule", "update", "--init", "--recursive", "--depth=1"], cwd=checkout)
        artifacts: dict[str, object] = {}
        for configured_path in dependency.get("install_paths", ["."]):
            relative = checked_relative_path(configured_path)
            requirements = checkout / relative / "requirements.txt"
            if requirements.is_file():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                    check=True,
                )
            source, artifact = build_or_reuse(
                checkout=checkout,
                relative=relative,
                dependency_id=dependency_id,
                repository=repository,
                consumer=consumer,
                cache_root=cache_root,
                base_tag=base_tag,
                platform=platform,
                recipe=recipe,
            )
            artifacts[relative] = {"artifact": str(artifact), "source_digest": source}
        resolved["repositories"][dependency_id] = {  # type: ignore[index]
            "artifacts": artifacts,
            "path": str(checkout),
            "repository": repository,
            "requested_ref": dependency["ref"],
            "requested_ref_type": dependency["ref_type"],
            "resolved_commit": commit,
        }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    print(f"Resolved dependency lock: {lock_path}")


if __name__ == "__main__":
    main()
