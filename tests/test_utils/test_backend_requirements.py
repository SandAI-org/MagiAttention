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

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest import TestCase, mock

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.meta.collection import calc_meta
from magi_attention.testing import parameterize
from magi_attention.utils import backend_requirements
from magi_attention.utils.backend_requirements import (
    CUTEDSL_DSL_REQUIREMENT,
    CUTEDSL_LIBS_BASE_DISTRIBUTION,
    CUTEDSL_PAIRS,
    CUTEDSL_QUACK_REQUIREMENT,
    CUTEDSL_UPGRADE_COMMAND,
    FA4_DSL_REQUIREMENT,
    FA4_TESTED_PAIR,
    CuteDslRequirementError,
    check_cutedsl_installation,
)
from magi_attention.utils.version import (
    is_cutedsl_version_eq,
    is_cutedsl_version_ge,
    is_cutedsl_version_gt,
    is_cutedsl_version_le,
    is_cutedsl_version_lt,
    is_cutedsl_version_ne,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"
CI_CONSTRAINTS_TXT = REPO_ROOT / ".github" / "configs" / "ci_constraints.txt"
# Keep child imports tied to the package under test.
PACKAGE_PARENT = Path(magi_attention.__file__).resolve().parents[1]
EXPECTED_UPGRADE_COMMAND = (
    'pip install --upgrade "nvidia-cutlass-dsl[cu13]<=4.8.0" "quack-kernels<=0.6.5"'
)
_MATCH_DSL = object()


def _requirements_txt() -> dict[str, Requirement]:
    path = REQUIREMENTS_TXT
    assert path.is_file(), path
    requirements = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            requirements[requirement.name] = requirement
    return requirements


@contextmanager
def _fake_installed_versions(
    dsl: str | None,
    quack: str | None,
    libs_base: str | None | object = _MATCH_DSL,
) -> Iterator[None]:
    """Report a DSL/quack pair and matching libs-base unless overridden."""
    if libs_base is _MATCH_DSL:
        libs_base = dsl
    assert libs_base is None or isinstance(libs_base, str)
    # Isolate cached successful checks between cases.
    check_cutedsl_installation.cache_clear()
    try:
        with (
            mock.patch.object(
                backend_requirements, "installed_versions", lambda: (dsl, quack)
            ),
            mock.patch.object(
                backend_requirements,
                "_installed_cutedsl_libs_base_version",
                lambda: libs_base,
            ),
        ):
            yield
    finally:
        check_cutedsl_installation.cache_clear()


def _run_in_fresh_interpreter(code: str) -> dict:
    """Run ``code`` in a fresh interpreter and read JSON from its last stdout line."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE_PARENT), *filter(None, [env.get("PYTHONPATH")])]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _versions_at_the_bounds(*specifier_sets: SpecifierSet) -> set[Version]:
    """Return named versions and their adjacent patch releases."""
    versions = set()
    for specifier_set in specifier_sets:
        for spec in specifier_set:
            version = Version(spec.version)
            major, minor, patch = (*version.release, 0, 0)[:3]
            versions.add(version)
            versions.add(Version(f"{major}.{minor}.{patch + 1}"))
            if patch > 0:
                versions.add(Version(f"{major}.{minor}.{patch - 1}"))
    return versions


CUTEDSL_VERSION_PREDICATES = {
    "==": is_cutedsl_version_eq,
    "!=": is_cutedsl_version_ne,
    ">=": is_cutedsl_version_ge,
    ">": is_cutedsl_version_gt,
    "<=": is_cutedsl_version_le,
    "<": is_cutedsl_version_lt,
}


class TestBackendRequirements(TestCase):
    def test_version_constants_are_consistent(self):
        requirements = _requirements_txt()
        for constant in (CUTEDSL_DSL_REQUIREMENT, CUTEDSL_QUACK_REQUIREMENT):
            listed = requirements[constant.name]
            assert listed.specifier == constant.specifier
            assert listed.extras == constant.extras

        dsl_range = requirements[CUTEDSL_DSL_REQUIREMENT.name].specifier
        quack_range = requirements[CUTEDSL_QUACK_REQUIREMENT.name].specifier
        for pair_spec, range_spec in [
            *((dsl_spec, dsl_range) for dsl_spec, _ in CUTEDSL_PAIRS),
            *((quack_spec, quack_range) for _, quack_spec in CUTEDSL_PAIRS),
        ]:
            # Check bounds, exclusions, and adjacent patch releases.
            operators = {spec.operator for spec in pair_spec}
            assert operators & {">=", ">", "=="}, f"{pair_spec} has no lower bound"
            assert operators & {"<=", "<", "=="}, f"{pair_spec} has no upper bound"
            for version in _versions_at_the_bounds(pair_spec, range_spec):
                if pair_spec.contains(version):
                    assert range_spec.contains(version), (
                        pair_spec,
                        version,
                        range_spec,
                    )

        assert CUTEDSL_UPGRADE_COMMAND == EXPECTED_UPGRADE_COMMAND
        newest_dsl_spec, newest_quack_spec = CUTEDSL_PAIRS[-1]
        assert newest_dsl_spec.contains("4.8.0")
        assert newest_quack_spec.contains("0.6.5")

        tested_dsl, tested_quack = FA4_TESTED_PAIR
        assert FA4_DSL_REQUIREMENT.specifier.contains(tested_dsl)
        assert any(
            dsl_spec.contains(tested_dsl) and quack_spec.contains(tested_quack)
            for dsl_spec, quack_spec in CUTEDSL_PAIRS
        )

        # The fa4 install advice must match the pair tested in CI.
        pins = {}
        for line in CI_CONSTRAINTS_TXT.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                requirement = Requirement(line)
                (spec,) = requirement.specifier
                assert spec.operator == "==", line
                pins[canonicalize_name(requirement.name)] = Version(spec.version)
        assert pins == {
            canonicalize_name(CUTEDSL_DSL_REQUIREMENT.name): tested_dsl,
            canonicalize_name(CUTEDSL_QUACK_REQUIREMENT.name): tested_quack,
        }

    @parameterize(
        "installed",
        [
            ("4.8.0", "0.4.1"),
            ("4.5.0", "0.6.5"),
            ("4.6.0", "0.6.4"),
            ("4.8.1", "0.6.5"),
            # Individually supported releases can form an unsupported pair.
            ("4.6.2", "0.6.5"),
            ("4.7.0", "0.6.4"),
            ("4.4.2", "0.6.4"),
            ("4.6.2", "0.4.1"),
            (None, None),
            ("4.8.0", None),
            (None, "0.6.5"),
        ],
    )
    def test_check_rejects_a_pair_outside_the_table(self, installed):
        dsl, quack = installed
        with _fake_installed_versions(dsl, quack):
            with self.assertRaises(CuteDslRequirementError) as raised:
                check_cutedsl_installation()
        message = str(raised.exception)
        assert EXPECTED_UPGRADE_COMMAND in message
        assert f"nvidia-cutlass-dsl {dsl or 'not installed'}" in message
        assert f"quack-kernels {quack or 'not installed'}" in message
        assert str(CUTEDSL_DSL_REQUIREMENT) in message
        assert str(CUTEDSL_QUACK_REQUIREMENT) in message

    @parameterize(
        "installed",
        [
            ("4.8.0", "0.6.5"),
            ("4.7.0", "0.6.5"),
            ("4.4.2", "0.4.1"),
            ("4.6.2", "0.6.4"),
        ],
    )
    def test_check_accepts_a_pair_in_the_table(self, installed):
        dsl, quack = installed
        with _fake_installed_versions(dsl, quack):
            assert check_cutedsl_installation() == (dsl, quack)

    @parameterize("libs_base", [None, "4.4.1", "4.4.3", "4.8.0"])
    def test_check_rejects_missing_or_mismatched_libs_base(self, libs_base):
        with _fake_installed_versions("4.4.2", "0.4.1", libs_base):
            with self.assertRaises(CuteDslRequirementError) as raised:
                check_cutedsl_installation()
        message = str(raised.exception)
        assert (
            f"{CUTEDSL_LIBS_BASE_DISTRIBUTION} {libs_base or 'not installed'}"
            in message
        )
        assert "must have the same version" in message

    # (dsl, quack, version, release_only, operators that hold or the raised error)
    @parameterize(
        "case",
        [
            ("4.6.2", "0.6.4", "4.6.2", True, {"==", ">=", "<="}),
            ("4.6.2", "0.6.4", "4.6.1", True, {"!=", ">=", ">"}),
            ("4.6.2", "0.6.4", "4.7.0", True, {"!=", "<=", "<"}),
            # release_only compares numeric tuples, including their length.
            ("4.6.2", "0.6.4", "4.6", True, {"!=", ">=", ">"}),
            ("4.7.0.post1", "0.6.5", "4.7.0", True, {"==", ">=", "<="}),
            ("4.7.0.post1", "0.6.5", "4.7.0", False, {"!=", ">=", ">"}),
            # Version predicates also reject unsupported pairs.
            ("4.6.0", "0.6.4", "4.5.0", True, CuteDslRequirementError),
        ],
    )
    def test_cutedsl_version_predicates(self, case):
        dsl, quack, version, release_only, holds = case
        with _fake_installed_versions(dsl, quack):
            if holds is CuteDslRequirementError:
                for predicate in CUTEDSL_VERSION_PREDICATES.values():
                    with self.assertRaises(CuteDslRequirementError):
                        predicate(version, release_only=release_only)
                return
            assert {
                operator
                for operator, predicate in CUTEDSL_VERSION_PREDICATES.items()
                if predicate(version, release_only=release_only)
            } == holds

    def test_import_checks_the_pair_before_importing_quack(self):
        # Block the fa4 probe so imported module versions cannot override test metadata.
        child = "\n".join(
            [
                "import importlib.metadata, importlib.util, json, sys",
                "sys.modules['flash_attn_cute'] = None",
                "fake = {'nvidia-cutlass-dsl': '4.8.0', 'quack-kernels': '0.4.1'}",
                "real = importlib.metadata.version",
                "importlib.metadata.version = lambda name: fake.get(name) or real(name)",
                "result = {'origin': importlib.util.find_spec('magi_attention').origin}",
                "try:",
                "    import magi_attention",
                "except Exception as exc:",
                "    result['error'] = type(exc).__module__ + '.' + type(exc).__qualname__",
                "    result['message'] = str(exc)",
                "result['quack_imported'] = 'quack' in sys.modules",
                "result['kernel_imported'] = 'magi_attention.kernel.cutedsl.ffa_utils' in sys.modules",
                "print(json.dumps(result))",
            ]
        )
        result = _run_in_fresh_interpreter(child)
        assert result["origin"] == magi_attention.__file__, result
        assert result.get("error") == (
            "magi_attention.utils.backend_requirements.CuteDslRequirementError"
        ), result
        assert EXPECTED_UPGRADE_COMMAND in result["message"]
        assert result["quack_imported"] is False
        assert result["kernel_imported"] is False

    @parameterize("quack_error", ["ImportError", "AttributeError"])
    def test_import_reports_the_pair_when_the_fa4_probe_fails_in_quack(
        self, quack_error
    ):
        # The fa4 probe can fail in quack before the pair check runs.
        # quack 0.5.0 ensures rejection regardless of the installed DSL version.
        child = "\n".join(
            [
                "import importlib.metadata, importlib.util, json, sys",
                "result = {'origin': importlib.util.find_spec('magi_attention').origin}",
                "class FlashAttnCuteWithBrokenQuack:",
                "    def find_spec(self, name, path=None, target=None):",
                "        if name == 'quack':",
                f"            raise {quack_error}('quack does not match cutlass')",
                "        if name == 'flash_attn_cute':",
                "            result['flash_attn_cute_imported'] = True",
                "            import cutlass",
                "            import quack",
                "sys.meta_path.insert(0, FlashAttnCuteWithBrokenQuack())",
                "fake = {'quack-kernels': '0.5.0'}",
                "real = importlib.metadata.version",
                "importlib.metadata.version = lambda name: fake.get(name) or real(name)",
                "try:",
                "    import magi_attention",
                "except Exception as exc:",
                "    result['error'] = type(exc).__module__ + '.' + type(exc).__qualname__",
                "    result['message'] = str(exc)",
                "result['cutlass_imported'] = 'cutlass' in sys.modules",
                "result['quack_imported'] = 'quack' in sys.modules",
                "result['kernel_imported'] = 'magi_attention.kernel.cutedsl.ffa_utils' in sys.modules",
                "print(json.dumps(result))",
            ]
        )
        result = _run_in_fresh_interpreter(child)
        assert result["origin"] == magi_attention.__file__, result
        assert result.get("error") == (
            "magi_attention.utils.backend_requirements.CuteDslRequirementError"
        ), result
        assert "quack-kernels 0.5.0" in result["message"]
        assert EXPECTED_UPGRADE_COMMAND in result["message"]
        assert result["flash_attn_cute_imported"] is True
        assert result["cutlass_imported"] is True
        assert result["quack_imported"] is False
        assert result["kernel_imported"] is False

    def test_a_failing_flash_attn_cute_import_disables_the_fa4_backend(self):
        # Preserve package imports and report the original error at fa4 entry points.
        error = "module 'cutlass.cute.core' has no attribute 'ThrMma'"
        child = "\n".join(
            [
                "import json, sys",
                "class BrokenFlashAttnCute:",
                "    def find_spec(self, name, path=None, target=None):",
                "        if name == 'flash_attn_cute':",
                f"            raise AttributeError({error!r})",
                "sys.meta_path.insert(0, BrokenFlashAttnCute())",
                "result = {}",
                "def record(name, call):",
                "    try:",
                "        call()",
                "    except RuntimeError as exc:",
                "        result[name] = [str(exc), repr(exc.__cause__)]",
                "try:",
                "    import magi_attention",
                "    from magi_attention.common import AttnRanges",
                "    from magi_attention.functional import fa4, fa4_utils",
                "    from magi_attention.meta.collection import calc_meta",
                "    result['origin'] = magi_attention.__file__",
                "    result['is_fa4_installed'] = [",
                "        calc_meta.is_fa4_installed, fa4.is_fa4_installed]",
                "    result['has_compiled_meta_dict'] = hasattr(",
                "        fa4_utils, 'COMPILED_META_DICT')",
                "    ranges = AttnRanges.from_ranges([[0, 128]])",
                "    record('FA4AttnArg', lambda: calc_meta.FA4AttnArg(",
                "        q_ranges=ranges, k_ranges=ranges, attn_type_map=[0]))",
                "    record('load_precompiled_ffa_fa4', fa4_utils.load_precompiled_ffa_fa4)",
                "    record('fa4_fwd', lambda: fa4.fa4_fwd(None, None, None, None, None))",
                "    record('fa4_bwd', lambda: fa4.fa4_bwd(",
                "        None, None, None, None, None, None, None, None))",
                # The SM90 tile-size query must work when flash_attn_cute cannot import.
                "    calc_meta.COMPUTE_CAPABILITY = 9",
                "    result['tile_sizes'] = calc_meta._resolve_tile_sizes('forward', 128)",
                "except Exception as exc:",
                "    result['error'] = repr(exc)",
                "print(json.dumps(result))",
            ]
        )
        result = _run_in_fresh_interpreter(child)
        assert "error" not in result, result
        assert result["origin"] == magi_attention.__file__, result
        assert result["is_fa4_installed"] == [False, False]
        assert result["has_compiled_meta_dict"] is False
        assert result["tile_sizes"] == list(calc_meta._DEFAULT_FA4_TILE_SIZE)
        for name in ("FA4AttnArg", "load_precompiled_ffa_fa4", "fa4_fwd", "fa4_bwd"):
            message, cause = result[name]
            assert str(FA4_DSL_REQUIREMENT) in message
            assert 'pip install "nvidia-cutlass-dsl[cu13]==4.4.2"' in message
            assert cause == repr(AttributeError(error))


if __name__ == "__main__":
    run_tests()
