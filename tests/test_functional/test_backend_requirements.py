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
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import magi_attention
from magi_attention.meta.collection import calc_meta
from magi_attention.utils import backend_requirements
from magi_attention.utils.backend_requirements import (
    CUTEDSL_DSL_REQUIREMENT,
    CUTEDSL_PAIRS,
    CUTEDSL_QUACK_REQUIREMENT,
    CUTEDSL_UPGRADE_COMMAND,
    FA4_DSL_REQUIREMENT,
    FA4_TESTED_PAIR,
    CuteDslRequirementError,
    check_cutedsl_installation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"
# The directory that holds the imported magi_attention package; a child
# interpreter with it first on PYTHONPATH imports the same package.
PACKAGE_PARENT = Path(magi_attention.__file__).resolve().parents[1]
EXPECTED_UPGRADE_COMMAND = (
    'pip install --upgrade "nvidia-cutlass-dsl[cu13]<=4.8.0" "quack-kernels<=0.6.5"'
)


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


@pytest.fixture
def fake_installed_versions(monkeypatch):
    def install(dsl: str | None, quack: str | None) -> None:
        monkeypatch.setattr(
            backend_requirements, "installed_versions", lambda: (dsl, quack)
        )

    # lru_cache keeps a previous pass; it never keeps a raised exception.
    check_cutedsl_installation.cache_clear()
    yield install
    check_cutedsl_installation.cache_clear()


def _run_in_fresh_interpreter(code: str) -> dict:
    """Run ``code`` in a new interpreter that imports the same magi_attention
    and return the JSON object on its last stdout line."""
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
    """Return every version a specifier names, and the patch releases on both
    sides of it."""
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


def test_version_constants_are_consistent():
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
        # A pair specifier with both bounds is an interval, and a range is an
        # interval without its `!=` versions. The interval lies inside the range
        # when the range contains every pair version among the bounds of both,
        # the releases next to them, and the excluded versions.
        operators = {spec.operator for spec in pair_spec}
        assert operators & {">=", ">", "=="}, f"{pair_spec} has no lower bound"
        assert operators & {"<=", "<", "=="}, f"{pair_spec} has no upper bound"
        for version in _versions_at_the_bounds(pair_spec, range_spec):
            if pair_spec.contains(version):
                assert range_spec.contains(version), (pair_spec, version, range_spec)

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


@pytest.mark.parametrize(
    "dsl, quack",
    [
        ("4.8.0", "0.4.1"),
        ("4.5.0", "0.6.5"),
        ("4.6.0", "0.6.4"),
        ("4.8.1", "0.6.5"),
        # The DSL version of one pair with the quack version of another.
        ("4.6.2", "0.6.5"),
        ("4.7.0", "0.6.4"),
        ("4.4.2", "0.6.4"),
        ("4.6.2", "0.4.1"),
        (None, None),
        ("4.8.0", None),
        (None, "0.6.5"),
    ],
)
def test_check_rejects_a_pair_outside_the_table(fake_installed_versions, dsl, quack):
    fake_installed_versions(dsl, quack)
    with pytest.raises(CuteDslRequirementError) as excinfo:
        check_cutedsl_installation()
    message = str(excinfo.value)
    assert EXPECTED_UPGRADE_COMMAND in message
    assert f"nvidia-cutlass-dsl {dsl or 'not installed'}" in message
    assert f"quack-kernels {quack or 'not installed'}" in message
    assert str(CUTEDSL_DSL_REQUIREMENT) in message
    assert str(CUTEDSL_QUACK_REQUIREMENT) in message


@pytest.mark.parametrize(
    "dsl, quack",
    [
        ("4.8.0", "0.6.5"),
        ("4.7.0", "0.6.5"),
        ("4.4.2", "0.4.1"),
        ("4.6.2", "0.6.4"),
    ],
)
def test_check_accepts_a_pair_in_the_table(fake_installed_versions, dsl, quack):
    fake_installed_versions(dsl, quack)
    assert check_cutedsl_installation() == (dsl, quack)


def test_import_checks_the_pair_before_importing_quack():
    # A fresh interpreter reports an unsupported pair through the distribution
    # metadata; the import must fail with CuteDslRequirementError before quack
    # or a kernel module is imported. flash_attn_cute is blocked because the fa4
    # import probe of magi_attention.meta imports cutlass and quack before
    # magi_attention.kernel.cutedsl when flash_attn_cute is installed, and an
    # imported module's version takes precedence over the faked metadata.
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


@pytest.mark.parametrize("quack_error", ["ImportError", "AttributeError"])
def test_import_reports_the_pair_when_the_fa4_probe_fails_in_quack(quack_error):
    # When flash_attn_cute is installed, the fa4 import probe of
    # magi_attention.meta imports cutlass and quack before
    # magi_attention.kernel.cutedsl checks the pair. Here flash_attn_cute
    # imports the real cutlass and then a quack that fails the way an
    # unsupported pair does. The probe must catch that error so that the import
    # still fails with CuteDslRequirementError. quack-kernels 0.5.0 is in no
    # supported pair, so the check rejects every nvidia-cutlass-dsl version.
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


def test_a_failing_flash_attn_cute_import_disables_the_fa4_backend():
    # flash_attn_cute built against another CUTLASS DSL release raises
    # AttributeError inside its import. magi_attention and fa4_utils must still
    # import, report fa4 as not installed, and raise the not-installed
    # RuntimeError, chained to the import error, when the fa4 backend is used.
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
            # Before SM100 the tile sizes come from flash_attn_cute, whose
            # import raises the AttributeError again.
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
