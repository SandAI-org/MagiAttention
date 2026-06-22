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

import torch
import triton
from packaging.version import Version

__version__ = "1.0.0"


def _parse(version: str, release_only: bool) -> tuple[int, ...] | Version:
    v = Version(version)
    return v.release if release_only else v


# --------  torch version predicates --------

_TORCH_VERSION = Version(torch.__version__)
_TORCH_RELEASE = _TORCH_VERSION.release


def is_torch_version_eq(version: str, release_only: bool = True) -> bool:
    """Return True if the current torch version is == *version*.

    Args:
        version: Version string to compare against, e.g. ``"2.12"`` or ``"2.12.1"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``2.12.0a0 < 2.12.0``.
    """
    lhs = _TORCH_RELEASE if release_only else _TORCH_VERSION
    return lhs == _parse(version, release_only)


def is_torch_version_ne(version: str, release_only: bool = True) -> bool:
    """Return True if the current torch version is != *version*.

    Args:
        version: Version string to compare against, e.g. ``"2.12"`` or ``"2.12.1"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``2.12.0a0 < 2.12.0``.
    """
    lhs = _TORCH_RELEASE if release_only else _TORCH_VERSION
    return lhs != _parse(version, release_only)


def is_torch_version_ge(version: str, release_only: bool = True) -> bool:
    """Return True if the current torch version is >= *version*.

    Args:
        version: Version string to compare against, e.g. ``"2.12"`` or ``"2.12.1"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``2.12.0a0 < 2.12.0``.
    """
    lhs = _TORCH_RELEASE if release_only else _TORCH_VERSION
    return lhs >= _parse(version, release_only)


def is_torch_version_gt(version: str, release_only: bool = True) -> bool:
    """Return True if the current torch version is > *version*.

    Args:
        version: Version string to compare against, e.g. ``"2.12"`` or ``"2.12.1"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``2.12.0a0 < 2.12.0``.
    """
    lhs = _TORCH_RELEASE if release_only else _TORCH_VERSION
    return lhs > _parse(version, release_only)


def is_torch_version_le(version: str, release_only: bool = True) -> bool:
    """Return True if the current torch version is <= *version*.

    Args:
        version: Version string to compare against, e.g. ``"2.12"`` or ``"2.12.1"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``2.12.0a0 < 2.12.0``.
    """
    lhs = _TORCH_RELEASE if release_only else _TORCH_VERSION
    return lhs <= _parse(version, release_only)


def is_torch_version_lt(version: str, release_only: bool = True) -> bool:
    """Return True if the current torch version is < *version*.

    Args:
        version: Version string to compare against, e.g. ``"2.12"`` or ``"2.12.1"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``2.12.0a0 < 2.12.0``.
    """
    lhs = _TORCH_RELEASE if release_only else _TORCH_VERSION
    return lhs < _parse(version, release_only)


# --------  triton version predicates --------

_TRITON_VERSION = Version(triton.__version__)
_TRITON_RELEASE = _TRITON_VERSION.release


def is_triton_version_eq(version: str, release_only: bool = True) -> bool:
    """Return True if the current triton version is == *version*.

    Args:
        version: Version string to compare against, e.g. ``"3.6"`` or ``"3.6.0"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``3.6.0a0 < 3.6.0``.
    """
    lhs = _TRITON_RELEASE if release_only else _TRITON_VERSION
    return lhs == _parse(version, release_only)


def is_triton_version_ge(version: str, release_only: bool = True) -> bool:
    """Return True if the current triton version is >= *version*.

    Args:
        version: Version string to compare against, e.g. ``"3.6"`` or ``"3.6.0"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``3.6.0a0 < 3.6.0``.
    """
    lhs = _TRITON_RELEASE if release_only else _TRITON_VERSION
    return lhs >= _parse(version, release_only)


def is_triton_version_gt(version: str, release_only: bool = True) -> bool:
    """Return True if the current triton version is > *version*.

    Args:
        version: Version string to compare against, e.g. ``"3.6"`` or ``"3.6.0"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``3.6.0a0 < 3.6.0``.
    """
    lhs = _TRITON_RELEASE if release_only else _TRITON_VERSION
    return lhs > _parse(version, release_only)


def is_triton_version_le(version: str, release_only: bool = True) -> bool:
    """Return True if the current triton version is <= *version*.

    Args:
        version: Version string to compare against, e.g. ``"3.6"`` or ``"3.6.0"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``3.6.0a0 < 3.6.0``.
    """
    lhs = _TRITON_RELEASE if release_only else _TRITON_VERSION
    return lhs <= _parse(version, release_only)


def is_triton_version_lt(version: str, release_only: bool = True) -> bool:
    """Return True if the current triton version is < *version*.

    Args:
        version: Version string to compare against, e.g. ``"3.6"`` or ``"3.6.0"``.
        release_only: If True (default), compare only the numeric release segment
            ``(major, minor, patch)``, ignoring pre/dev/post/local suffixes such as
            ``a0`` or ``+nv26.04``. If False, use full PEP 440 semantics where
            e.g. ``3.6.0a0 < 3.6.0``.
    """
    lhs = _TRITON_RELEASE if release_only else _TRITON_VERSION
    return lhs < _parse(version, release_only)


# --------  cuda version predicates --------

_CUDA_VERSION_STR = torch.version.cuda
_CUDA_VERSION = Version(_CUDA_VERSION_STR) if _CUDA_VERSION_STR is not None else None
_CUDA_RELEASE = _CUDA_VERSION.release if _CUDA_VERSION is not None else None


def has_cuda_version() -> bool:
    """Return True if torch is built with CUDA and exposes a CUDA version string."""
    return _CUDA_VERSION is not None


def is_cuda_version_eq(version: str, release_only: bool = True) -> bool:
    """Return True if the current CUDA version is == *version*.

    Returns False when torch is not built with CUDA.
    """
    if _CUDA_VERSION is None:
        return False
    assert _CUDA_RELEASE is not None  # mypy

    lhs = _CUDA_RELEASE if release_only else _CUDA_VERSION
    return lhs == _parse(version, release_only)


def is_cuda_version_ne(version: str, release_only: bool = True) -> bool:
    """Return True if the current CUDA version is != *version*.

    Returns False when torch is not built with CUDA.
    """
    if _CUDA_VERSION is None:
        return False
    assert _CUDA_RELEASE is not None  # mypy

    lhs = _CUDA_RELEASE if release_only else _CUDA_VERSION
    return lhs != _parse(version, release_only)


def is_cuda_version_ge(version: str, release_only: bool = True) -> bool:
    """Return True if the current CUDA version is >= *version*.

    Returns False when torch is not built with CUDA.
    """
    if _CUDA_VERSION is None:
        return False
    assert _CUDA_RELEASE is not None  # mypy

    lhs = _CUDA_RELEASE if release_only else _CUDA_VERSION
    return lhs >= _parse(version, release_only)


def is_cuda_version_gt(version: str, release_only: bool = True) -> bool:
    """Return True if the current CUDA version is > *version*.

    Returns False when torch is not built with CUDA.
    """
    if _CUDA_VERSION is None:
        return False
    assert _CUDA_RELEASE is not None  # mypy

    lhs = _CUDA_RELEASE if release_only else _CUDA_VERSION
    return lhs > _parse(version, release_only)


def is_cuda_version_le(version: str, release_only: bool = True) -> bool:
    """Return True if the current CUDA version is <= *version*.

    Returns False when torch is not built with CUDA.
    """
    if _CUDA_VERSION is None:
        return False
    assert _CUDA_RELEASE is not None  # mypy

    lhs = _CUDA_RELEASE if release_only else _CUDA_VERSION
    return lhs <= _parse(version, release_only)


def is_cuda_version_lt(version: str, release_only: bool = True) -> bool:
    """Return True if the current CUDA version is < *version*.

    Returns False when torch is not built with CUDA.
    """
    if _CUDA_VERSION is None:
        return False
    assert _CUDA_RELEASE is not None  # mypy

    lhs = _CUDA_RELEASE if release_only else _CUDA_VERSION
    return lhs < _parse(version, release_only)
