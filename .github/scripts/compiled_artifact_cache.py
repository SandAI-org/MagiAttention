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

"""Cache MagiAttention native extension and AOT kernel artifacts.

The cache is intentionally below the wheel cache: Python-only changes can reuse
ABI-compatible native artifacts and then assemble a fresh wheel.

TODO: Split the native artifacts into a separately versioned
``magi-attention-kernels`` distribution and keep ``magi-attention`` as a light
Python frontend once the frontend/kernel ABI and release policy are mature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sysconfig
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA = 1
PACKAGE = "magi_attention"
BINARY_INPUTS = (
    "setup.py",
    "pyproject.toml",
    "magi_attention/csrc",
    "magi_attention/common/jit",
    "magi_attention/functional/_flex_flash_attn_jit.py",
)
CI_BINARY_INPUTS = (
    "magi_attention/testing/precompile.py",
    "tests",
)
BUILD_ENV = (
    "MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12",
    "MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY",
    "MAGI_ATTENTION_DISABLE_SM90_FEATURES",
    "MAGI_ATTENTION_FORCE_CXX11_ABI",
    "MAGI_ATTENTION_PREBUILD_FFA",
    "MAGI_ATTENTION_PREBUILD_LEVEL",
    "MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD",
    "MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD",
    "DISABLE_AGGRESSIVE_PTX_INSTRS",
)


def run(
    *command: str, cwd: Path | None = None, input_bytes: bytes | None = None
) -> bytes:
    return subprocess.run(
        command,
        cwd=cwd,
        input=input_bytes,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def git_root(source_root: Path) -> Path:
    return Path(
        run("git", "-C", str(source_root), "rev-parse", "--show-toplevel")
        .decode()
        .strip()
    )


def input_paths(source_root: Path, root: Path) -> tuple[str, ...]:
    level = os.environ.get("MAGI_ATTENTION_PREBUILD_LEVEL", "lite").lower()
    paths = BINARY_INPUTS + (CI_BINARY_INPUTS if level == "ci" else ())
    prefix = source_root.relative_to(root)
    return tuple((prefix / path).as_posix() for path in paths)


def gitlink_identity(path: Path, recorded_oid: str) -> str:
    if not path.is_dir():
        return recorded_oid
    try:
        head = run("git", "-C", str(path), "rev-parse", "HEAD")
        diff = run("git", "-C", str(path), "diff", "--binary", "HEAD", "--")
        untracked = run(
            "git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"
        )
    except subprocess.CalledProcessError:
        return recorded_oid
    digest = hashlib.sha256(recorded_oid.encode() + b"\0" + head + b"\0" + diff)
    for raw_name in sorted(name for name in untracked.split(b"\0") if name):
        candidate = path / os.fsdecode(raw_name)
        if candidate.is_file() and not candidate.is_symlink():
            digest.update(raw_name + b"\0" + candidate.read_bytes() + b"\0")
        elif candidate.is_symlink():
            digest.update(
                raw_name + b"\0" + os.fsencode(os.readlink(candidate)) + b"\0"
            )
    return digest.hexdigest()


def source_digest(source_root: Path) -> str:
    root = git_root(source_root)
    paths = input_paths(source_root, root)
    output = run("git", "-C", str(root), "ls-files", "--stage", "-z", "--", *paths)
    records: list[bytes] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        if stage != "0":
            raise SystemExit(f"Unresolved binary input: {os.fsdecode(raw_path)}")
        path = root / os.fsdecode(raw_path)
        if mode == "160000":
            oid = gitlink_identity(path, oid)
        elif path.exists() or path.is_symlink():
            content = (
                os.fsencode(os.readlink(path))
                if path.is_symlink()
                else path.read_bytes()
            )
            oid = (
                run(
                    "git",
                    "-C",
                    str(root),
                    "hash-object",
                    "--stdin",
                    input_bytes=content,
                )
                .decode()
                .strip()
            )
            mode = (
                "120000"
                if path.is_symlink()
                else ("100755" if path.stat().st_mode & 0o111 else "100644")
            )
        relative = PurePosixPath(os.fsdecode(raw_path)).relative_to(
            PurePosixPath(source_root.relative_to(root).as_posix())
        )
        records.append(f"{mode} {oid}\t{relative.as_posix()}".encode())
    if not records:
        raise SystemExit("No tracked MagiAttention binary inputs found")
    return hashlib.sha256(b"\n".join(sorted(records)) + b"\n").hexdigest()


def command_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_inputs(source_root: Path) -> dict[str, object]:
    import torch

    capability = os.environ.get("MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY")
    if not capability:
        capability = (
            f"{torch.cuda.get_device_capability()[0]}0"
            if torch.cuda.is_available()
            else "unavailable"
        )
    return {
        "build_environment": {name: os.environ.get(name, "") for name in BUILD_ENV},
        "compute_capability": capability,
        "cuda": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "nvcc": command_text(
            [os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/bin/nvcc", "--version"]
        ),
        "platform": sysconfig.get_platform(),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "recipe_digest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schema_version": SCHEMA,
        "source_digest": source_digest(source_root),
        "torch": torch.__version__,
    }


def fingerprint(inputs: dict[str, object]) -> str:
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return f"v{SCHEMA}-{hashlib.sha256(payload).hexdigest()}"


def default_cache_root() -> Path:
    configured = os.environ.get("MAGI_ATTENTION_COMPILED_CACHE_ROOT")
    if configured:
        return Path(configured).expanduser()
    workspace = os.environ.get("CI_WORKSPACE_ROOT")
    if workspace:
        return Path(workspace) / "v2/compiled-artifacts/magi-attention"
    return Path.home() / ".cache/magi_attention/compiled-artifacts"


def artifact_dir(
    source_root: Path, cache_root: Path
) -> tuple[Path, dict[str, object], str]:
    inputs = build_inputs(source_root)
    identity = fingerprint(inputs)
    return cache_root / f"v{SCHEMA}" / identity, inputs, identity


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(
    directory: Path, inputs: dict[str, object], identity: str
) -> list[dict[str, str]]:
    manifest = directory / "manifest.json"
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
    ):
        raise ValueError(f"Invalid compiled artifact directory: {directory}")
    data = json.loads(manifest.read_text())
    if (
        data.get("inputs") != inputs
        or data.get("fingerprint") != identity
        or data.get("status") != "built"
    ):
        raise ValueError(f"Incompatible compiled artifact manifest: {manifest}")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Compiled artifact manifest has no files: {manifest}")
    for item in files:
        relative = item.get("path") if isinstance(item, dict) else None
        path = directory / relative if isinstance(relative, str) else None
        if path is None:
            assert relative is not None  # mypy
            if PurePosixPath(relative).is_absolute():
                raise ValueError(f"Unsafe compiled artifact path in {manifest}")
            if ".." in PurePosixPath(relative).parts:
                raise ValueError(f"Unsafe compiled artifact path in {manifest}")

        assert path is not None  # mypy
        parent = path.parent
        while parent != directory:
            if parent.is_symlink():
                raise ValueError(f"Symlinked compiled artifact parent: {path}")
            parent = parent.parent
        if (
            path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"Compiled artifact verification failed: {path}")
    return files


def restore(source_root: Path, destination: Path, cache_root: Path) -> bool:
    directory, inputs, identity = artifact_dir(source_root, cache_root)
    if not directory.exists():
        print(f"Compiled artifact cache miss: {identity}")
        return False
    files = verify(directory, inputs, identity)
    for item in files:
        relative = PurePosixPath(item["path"])
        target = destination / Path(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / item["path"], target)
    print(f"Restored MagiAttention compiled artifacts: {directory}")
    return True


def binary_member(name: str) -> bool:
    path = PurePosixPath(name)
    parts = path.parts
    return (len(parts) == 2 and parts[0] == PACKAGE and parts[1].endswith(".so")) or (
        len(parts) == 4
        and parts[0] == PACKAGE
        and parts[1] == "lib"
        and parts[3].endswith(".so")
    )


def publish(source_root: Path, wheel: Path, cache_root: Path) -> None:
    directory, inputs, identity = artifact_dir(source_root, cache_root)
    if directory.exists():
        verify(directory, inputs, identity)
        print(f"Reused MagiAttention compiled artifact cache: {directory}")
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".publish.", dir=directory.parent))
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = [name for name in archive.namelist() if binary_member(name)]
            if not members:
                raise SystemExit(
                    f"Wheel has no MagiAttention compiled artifacts: {wheel}"
                )
            for name in members:
                target = staging / Path(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        files = [
            {"path": path.relative_to(staging).as_posix(), "sha256": file_sha256(path)}
            for path in sorted(staging.rglob("*.so"))
        ]
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "files": files,
                    "fingerprint": identity,
                    "inputs": inputs,
                    "status": "built",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        verify(staging, inputs, identity)
        try:
            staging.rename(directory)
        except FileExistsError:
            verify(directory, inputs, identity)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"Published MagiAttention compiled artifacts: {directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fingerprint", "restore", "publish-wheel"))
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--cache-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    cache_root = (args.cache_root or default_cache_root()).resolve()
    if args.command == "fingerprint":
        print(artifact_dir(source_root, cache_root)[2])
        return 0
    if args.command == "restore":
        destination = (args.destination or source_root).resolve()
        return 0 if restore(source_root, destination, cache_root) else 1
    if args.wheel is None:
        raise SystemExit("publish-wheel requires --wheel")
    publish(source_root, args.wheel.resolve(), cache_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
