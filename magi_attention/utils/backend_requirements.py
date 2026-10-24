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

# Keep requirement checks importable without loading cutlass, quack or FA4.

import sys
from functools import lru_cache
from importlib import metadata

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

# Must equal the nvidia-cutlass-dsl and quack-kernels lines of requirements.txt.
CUTEDSL_DSL_REQUIREMENT = Requirement(
    "nvidia-cutlass-dsl[cu13]>=4.4.2,<=4.8.0,!=4.6.0,!=4.6.1"
)
CUTEDSL_QUACK_REQUIREMENT = Requirement("quack-kernels>=0.4.1,<=0.6.5")
CUTEDSL_LIBS_BASE_DISTRIBUTION = "nvidia-cutlass-dsl-libs-base"

# Supported pairs must satisfy requirements.txt and quack's DSL API dependencies.
CUTEDSL_PAIRS: tuple[tuple[SpecifierSet, SpecifierSet], ...] = (
    (SpecifierSet("==4.4.2"), SpecifierSet("==0.4.1")),
    (SpecifierSet("==4.6.2"), SpecifierSet("==0.6.4")),
    (SpecifierSet(">=4.7.0,<=4.8.0"), SpecifierSet("==0.6.5")),
)

# flash_attn_cute uses cute.core.ThrMma and cute.make_fragment, removed in
# DSL 4.6.0; its metadata omits this upper bound. The bound can be lifted when
# the FlashAttention submodule moves to a flash_attn_cute release that uses
# neither API.
FA4_DSL_REQUIREMENT = Requirement("nvidia-cutlass-dsl>=4.4.2,<4.6.0")

# Must satisfy CUTEDSL_PAIRS and FA4_DSL_REQUIREMENT and match the FA4 CI pins.
FA4_TESTED_PAIR: tuple[Version, Version] = (Version("4.4.2"), Version("0.4.1"))

_FA4_REINSTALL_COMMAND = "bash scripts/install_flash_attn_cute.sh <archs, e.g. sm100>"


def _name_with_extras(requirement: Requirement) -> str:
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    return requirement.name + extras


def _upper_bound(requirement: Requirement) -> str:
    return next(spec.version for spec in requirement.specifier if spec.operator == "<=")


# The requirement upper bounds must form a supported pair.
CUTEDSL_UPGRADE_COMMAND = "pip install --upgrade " + " ".join(
    f'"{_name_with_extras(requirement)}<={_upper_bound(requirement)}"'
    for requirement in (CUTEDSL_DSL_REQUIREMENT, CUTEDSL_QUACK_REQUIREMENT)
)

_FA4_INSTALL_COMMAND = "pip install " + " ".join(
    f'"{_name_with_extras(requirement)}=={version}"'
    for requirement, version in zip(
        (CUTEDSL_DSL_REQUIREMENT, CUTEDSL_QUACK_REQUIREMENT), FA4_TESTED_PAIR
    )
)


class CuteDslRequirementError(RuntimeError):
    """Missing or inconsistent CuTeDSL dependencies, or an unsupported pair."""


def _module_version(module_name: str, distribution: str) -> str | None:
    # Kernels use imported modules; metadata may describe another installation.
    module = sys.modules.get(module_name)
    if module is not None:
        return module.__version__
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def installed_versions() -> tuple[str | None, str | None]:
    """Return (DSL, quack) versions, preferring loaded modules over metadata.

    A missing distribution returns ``None`` for that package.
    """
    return (
        _module_version("cutlass", CUTEDSL_DSL_REQUIREMENT.name),
        _module_version("quack", CUTEDSL_QUACK_REQUIREMENT.name),
    )


def _installed_cutedsl_libs_base_version() -> str | None:
    try:
        return metadata.version(CUTEDSL_LIBS_BASE_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def _format_version(version: str | None) -> str:
    return "not installed" if version is None else version


@lru_cache(maxsize=None)
def check_cutedsl_installation() -> tuple[str, str]:
    """Return installed (DSL, quack) versions after validating the DSL stack.

    Raises:
        CuteDslRequirementError: A package is missing, DSL and libs-base differ,
            or the DSL/quack pair is unsupported.
    """
    dsl_version, quack_version = installed_versions()
    libs_base_version = _installed_cutedsl_libs_base_version()
    if (
        dsl_version is not None
        and quack_version is not None
        and libs_base_version is not None
        and Version(dsl_version) == Version(libs_base_version)
        and any(
            dsl_spec.contains(dsl_version) and quack_spec.contains(quack_version)
            for dsl_spec, quack_spec in CUTEDSL_PAIRS
        )
    ):
        return dsl_version, quack_version

    supported_pairs = "\n".join(
        f"  {CUTEDSL_DSL_REQUIREMENT.name}{dsl_spec} with "
        f"{CUTEDSL_QUACK_REQUIREMENT.name}{quack_spec}"
        for dsl_spec, quack_spec in CUTEDSL_PAIRS
    )
    raise CuteDslRequirementError(
        "The CuTeDSL attention kernels need a supported "
        f"{CUTEDSL_DSL_REQUIREMENT.name} / {CUTEDSL_QUACK_REQUIREMENT.name} pair.\n"
        f"Installed: {CUTEDSL_DSL_REQUIREMENT.name} {_format_version(dsl_version)}, "
        f"{CUTEDSL_LIBS_BASE_DISTRIBUTION} "
        f"{_format_version(libs_base_version)}, "
        f"{CUTEDSL_QUACK_REQUIREMENT.name} {_format_version(quack_version)}.\n"
        f"{CUTEDSL_LIBS_BASE_DISTRIBUTION} must have the same version as "
        f"{CUTEDSL_DSL_REQUIREMENT.name}.\n"
        f"Required: {CUTEDSL_DSL_REQUIREMENT}, {CUTEDSL_QUACK_REQUIREMENT}, "
        f"as one of these pairs:\n{supported_pairs}\n"
        f"Install the newest supported pair with:\n  {CUTEDSL_UPGRADE_COMMAND}"
    )


def fa4_not_installed_message(exc: BaseException) -> str:
    """Describe a flash_attn_cute import failure and its installation requirements."""
    dsl_version, _ = installed_versions()
    return (
        "The fa4 backend needs flash_attn_cute, which failed to import.\n"
        f"It requires {FA4_DSL_REQUIREMENT}; installed: "
        f"{CUTEDSL_DSL_REQUIREMENT.name} {_format_version(dsl_version)}.\n"
        f"To use the fa4 backend, install the tested pair:\n  {_FA4_INSTALL_COMMAND}\n"
        "then reinstall flash_attn_cute from the MagiAttention source root:\n"
        f"  {_FA4_REINSTALL_COMMAND}\n"
        f"Import error: {exc!r}"
    )
