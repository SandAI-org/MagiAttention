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

"""Build a bilingual Sphinx preview and hand off its latest successful version."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

_REPOSITORY = "SandAI-org/MagiAttention"
_ROOT = Path(__file__).resolve().parents[1]
_WATCH_ROOTS = ("docs", "assets", "magi_attention")
_POLL_INTERVAL = 1.0


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(_ROOT), *args])


def _check_repository() -> None:
    for remote in _git("remote").decode().splitlines():
        urls = _git("remote", "get-url", "--all", remote).decode().splitlines()
        for url in urls:
            path = url.removesuffix("/").removesuffix(".git")
            if re.fullmatch(
                r"(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)"
                + _REPOSITORY,
                path,
                re.IGNORECASE,
            ):
                return

    raise RuntimeError(f"Add a Git remote for github.com/{_REPOSITORY}.")


def _checkout_state() -> tuple[str, bytes, dict[bytes, tuple[int, int, int]]]:
    sha = _git("rev-parse", "HEAD").decode().strip()
    status = _git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    paths = _git(
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        *_WATCH_ROOTS,
    )
    inputs = {}
    for encoded in paths.split(b"\0"):
        if not encoded:
            continue
        try:
            stat = (_ROOT / os.fsdecode(encoded)).stat()
        except FileNotFoundError:
            continue
        inputs[encoded] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    return sha, status, inputs


class _ResourceLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in ("script", "img"):
            link = attributes.get("src")
        elif tag == "a" or (tag == "link" and attributes.get("rel") == "stylesheet"):
            link = attributes.get("href")
        else:
            link = None
        if link:
            self.links.append(link)


def _check_entries(html: Path) -> None:
    for language in (html, html / "zh_CN"):
        for page in (language / "index.html", language / "blog" / "toc.html"):
            resources = _ResourceLinks()
            resources.feed(page.read_text(encoding="utf-8"))
            for link in resources.links:
                url = urlsplit(link)
                if url.scheme or url.netloc or not url.path:
                    continue
                if url.path.startswith("/"):
                    raise RuntimeError(
                        f"{page}: root-relative link cannot be relocated: {link}"
                    )
                target = (page.parent / unquote(url.path)).resolve()
                if not target.is_relative_to(html) or not target.exists():
                    raise RuntimeError(f"{page}: missing preview resource {link}")


def _build_html(html: Path, doctrees: Path) -> None:
    for lang, destination in (("en", html), ("zh_CN", html / "zh_CN")):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",
                "-d",
                str(doctrees / lang),
                "source",
                str(destination),
            ],
            cwd=_ROOT / "docs",
            env={**os.environ, "DOCS_LANGUAGE": lang},
            check=True,
        )

    _check_entries(html)


def _manifest(scope: str, sha: str, dirty: bool, html: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "repository": _REPOSITORY,
        "scope": scope,
        "source_sha": sha,
        "dirty": dirty,
        "html": html,
        "entries": [
            {"label": "Open docs", "path": "index.html"},
            {"label": "Chinese docs", "path": "zh_CN/index.html"},
        ],
    }


def _build_artifact(destination: Path, scope: str, expected_sha: str) -> None:
    state = _checkout_state()
    sha, status, _ = state
    if sha != expected_sha or status:
        raise RuntimeError(
            "CI artifacts require a clean checkout at the expected source SHA."
        )
    if destination.exists():
        raise RuntimeError(f"Artifact destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        stage = Path(staging)
        artifact = stage / "artifact"
        _build_html(artifact / "html", stage / "doctrees")
        for file in artifact.rglob("*"):
            if any(char in file.name for char in '"<>:|*?\\\r\n'):
                raise RuntimeError(
                    f"Invalid Actions artifact path: {file.relative_to(artifact)}"
                )
        if _checkout_state() != state:
            raise RuntimeError("Checkout changed during the artifact build.")

        manifest = _manifest(scope, sha, False, "html")
        (artifact / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        artifact.rename(destination)

    print(f"Built CI artifact at {destination} (source SHA {sha})", flush=True)


def _build(scope_dir: Path, scope: str) -> None:
    with (scope_dir / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _checkout_state()
        sha, status, _ = state
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        # mkdtemp also distinguishes concurrent or repeated builds in one second.
        build = Path(tempfile.mkdtemp(prefix=f"{sha[:8]}-{timestamp}-", dir=scope_dir))
        # A registered publisher may read the HTML as a different local user.
        build.chmod(0o755)
        html = build / "html"
        _build_html(html, build / "doctrees")
        if _checkout_state() != state:
            raise RuntimeError(
                "Checkout changed during the build. Wait for the next build."
            )

        manifest = _manifest(
            scope, sha, bool(status), html.relative_to(scope_dir).as_posix()
        )
        temp = scope_dir / f".{build.name}.json.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temp.replace(scope_dir / "latest.json")

    print(
        f"Built {build.name} (dirty={bool(status)}): {html / 'index.html'}", flush=True
    )
    print(f"Local handoff: {scope_dir / 'latest.json'}", flush=True)


def _build_once(scope_dir: Path, scope: str) -> bool:
    try:
        _build(scope_dir, scope)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(
            f"Preview build failed: {exc}. Previous latest.json preserved.",
            file=sys.stderr,
        )
        return False

    return True


def _watch(scope_dir: Path, scope: str) -> None:
    previous = _checkout_state()
    _build_once(scope_dir, scope)
    while True:
        time.sleep(_POLL_INTERVAL)
        current = _checkout_state()
        if current == previous:
            continue
        # Wait until edits have stopped for a full interval.
        time.sleep(_POLL_INTERVAL)
        if _checkout_state() != current:
            continue

        previous = current
        _build_once(scope_dir, scope)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog="docs/preview_pr.sh")
    parser.add_argument(
        "--pr", required=True, help="PR number for the current checkout"
    )
    parser.add_argument(
        "--watch", action="store_true", help="Rebuild after local edits"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--local-only", action="store_true", help="Build a personal preview"
    )
    mode.add_argument(
        "--artifact-dir", type=Path, help="Write a clean repository CI artifact"
    )
    parser.add_argument("--source-sha", help="Expected full HEAD for --artifact-dir")
    args = parser.parse_args()

    if not re.fullmatch(r"[1-9][0-9]*", args.pr):
        parser.error("--pr must be a positive PR number without leading zeros")
    scope = f"pr-{args.pr}"
    if args.artifact_dir is not None:
        if (
            args.watch
            or not args.source_sha
            or not re.fullmatch(r"[0-9a-f]{40}", args.source_sha)
        ):
            parser.error(
                "--artifact-dir requires a full --source-sha and cannot use --watch"
            )
        _check_repository()
        _build_artifact(args.artifact_dir.resolve(), scope, args.source_sha)
        return 0

    if args.source_sha:
        parser.error("--source-sha requires --artifact-dir")
    _check_repository()
    scope_dir = _ROOT / ".tmp" / ".tmp_tests" / "infra-docs-preview" / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    if args.watch:
        _watch(scope_dir, scope)
        return 0

    return 0 if _build_once(scope_dir, scope) else 1


if __name__ == "__main__":
    try:
        sys.exit(_main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
