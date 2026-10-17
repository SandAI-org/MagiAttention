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

# Package requirements of the CuTeDSL-based attention backends.
#
# This module imports only the standard library and `packaging`, so the checks
# run before cutlass, quack or flash_attn_cute is imported.

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

# The (nvidia-cutlass-dsl, quack-kernels) combinations the CuTeDSL kernels
# support. Each quack-kernels release calls DSL APIs that other DSL releases
# removed, so the project rejects a combination outside this table before
# cutlass and quack are imported. pip accepts any combination inside the two
# requirements.txt ranges, so this check is stricter than pip. Every pair must
# lie inside both ranges.
CUTEDSL_PAIRS: tuple[tuple[SpecifierSet, SpecifierSet], ...] = (
    (SpecifierSet("==4.4.2"), SpecifierSet("==0.4.1")),
    (SpecifierSet("==4.6.2"), SpecifierSet("==0.6.4")),
    (SpecifierSet(">=4.7.0,<=4.8.0"), SpecifierSet("==0.6.5")),
)

# flash_attn_cute (the fa4 backend) uses `cute.core.ThrMma` and
# `cute.make_fragment`, which nvidia-cutlass-dsl 4.6.0 removed. Its own package
# metadata declares only the lower bound.
FA4_DSL_REQUIREMENT = Requirement("nvidia-cutlass-dsl>=4.4.2,<4.6.0")

# The (nvidia-cutlass-dsl, quack-kernels) pair the fa4 backend was tested with.
# It must be in CUTEDSL_PAIRS, satisfy FA4_DSL_REQUIREMENT, and equal the two
# pins of .github/configs/ci_constraints.txt, which the CI fa4 job installs.
FA4_TESTED_PAIR: tuple[Version, Version] = (Version("4.4.2"), Version("0.4.1"))

_FA4_REINSTALL_COMMAND = "bash scripts/install_flash_attn_cute.sh <archs, e.g. sm100>"


def _name_with_extras(requirement: Requirement) -> str:
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    return requirement.name + extras


def _upper_bound(requirement: Requirement) -> str:
    return next(spec.version for spec in requirement.specifier if spec.operator == "<=")


# Installs the newest pair: pip picks the highest release under each upper
# bound, and the two upper bounds form the last entry of CUTEDSL_PAIRS.
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
    """The installed nvidia-cutlass-dsl / quack-kernels pair is not in
    ``CUTEDSL_PAIRS``, or one of the two packages is not installed."""


def _module_version(module_name: str, distribution: str) -> str | None:
    # A module that is already imported is the one the kernels bind to, and
    # sys.path can point the distribution metadata at another installation.
    module = sys.modules.get(module_name)
    if module is not None:
        return module.__version__
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def installed_versions() -> tuple[str | None, str | None]:
    """Return the ``(nvidia-cutlass-dsl, quack-kernels)`` versions in use.

    For each package, the ``__version__`` of the imported module (``cutlass``,
    ``quack``) is used when the module is already imported; otherwise the
    version comes from the installed distribution metadata. ``None`` means the
    package is not installed.
    """
    return (
        _module_version("cutlass", CUTEDSL_DSL_REQUIREMENT.name),
        _module_version("quack", CUTEDSL_QUACK_REQUIREMENT.name),
    )


def _format_version(version: str | None) -> str:
    return "not installed" if version is None else version


@lru_cache(maxsize=None)
def check_cutedsl_installation() -> tuple[str, str]:
    """Check that the installed nvidia-cutlass-dsl / quack-kernels versions are
    one of ``CUTEDSL_PAIRS``.

    Returns:
        The ``(nvidia-cutlass-dsl, quack-kernels)`` versions that passed.

    Raises:
        CuteDslRequirementError: if either package is not installed or the pair
            is not in ``CUTEDSL_PAIRS``. The message names the installed
            versions, the supported pairs, and the pip command that installs
            the newest supported pair.
    """
    dsl_version, quack_version = installed_versions()
    if (
        dsl_version is not None
        and quack_version is not None
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
        f"{CUTEDSL_QUACK_REQUIREMENT.name} {_format_version(quack_version)}.\n"
        f"Required: {CUTEDSL_DSL_REQUIREMENT}, {CUTEDSL_QUACK_REQUIREMENT}, "
        f"as one of these pairs:\n{supported_pairs}\n"
        f"Install the newest supported pair with:\n  {CUTEDSL_UPGRADE_COMMAND}"
    )


def fa4_not_installed_message(exc: BaseException) -> str:
    """Return the error message for a failed import of flash_attn_cute.

    Args:
        exc: The exception raised while importing flash_attn_cute.
    """
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
