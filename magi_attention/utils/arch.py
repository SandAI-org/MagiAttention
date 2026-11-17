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

"""CUDA capability queries and project GPU aliases (shared helper v1.1.0).

Three layers: exact ``is_sm`` queries, named aliases in ``ARCH_SMS``, then
backend-specific support decisions at call sites. An alias does not establish
support for every kernel, dtype, compiler version, or architecture suffix.

For backwards compatibility, omitted ``device`` means CUDA device 0. Pass
``None`` or ``"cuda"`` for the current device; resolve it before cache lookup so
switching current devices never reuses another GPU's properties. Expensive
queries share per-device LRU caches; derived predicates are not separately
cached. Every query accepts ``flush=True`` to clear the caches and repopulate
from the current runtime (for example, after changing a mock). Refresh again
when leaving a mock, or call ``clear_arch_cache``. Global predicate values in
callers are snapshots and cannot be refreshed by this module.
"""

from functools import lru_cache

import torch

__version__ = "1.1.0"

# Project aliases, deliberately explicit rather than open-ended SM ranges.
# Blackwell means the B200/GB200/B300/GB300 training family here, not every GPU
# marketed as Blackwell (SM110/120/121), nor Rubin (SM107).
ARCH_SMS: dict[str, frozenset[int]] = {
    "ampere": frozenset({80, 86, 87, 88}),
    "ada": frozenset({89}),
    "hopper": frozenset({90}),
    "blackwell": frozenset({100, 103}),
    "rubin": frozenset({107}),
}


def _device_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


@lru_cache(maxsize=None)
def _get_dev_cap_num(device: int) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


@lru_cache(maxsize=None)
def _get_num_sms(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def clear_arch_cache() -> None:
    """Clear all device-query caches, including values populated under mocks."""
    _get_dev_cap_num.cache_clear()
    _get_num_sms.cache_clear()


def get_num_sms(
    device: torch.device | str | int | None = 0, *, flush: bool = False
) -> int:
    """Return the GPU's SM count; ``flush=True`` refreshes cached queries."""
    if flush:
        clear_arch_cache()
    return _get_num_sms(_device_index(device))


def get_dev_cap_num(
    device: torch.device | str | int | None = 0, *, flush: bool = False
) -> int:
    """Return the exact capability, e.g. 90, 100, 103, or 120 (no a/f suffix)."""
    if flush:
        clear_arch_cache()
    return _get_dev_cap_num(_device_index(device))


def get_dev_cap_str(
    upper: bool = False,
    device: torch.device | str | int | None = 0,
    *,
    flush: bool = False,
) -> str:
    """Return e.g. ``sm103`` or ``SM103``, preserving the legacy argument order."""
    capability = get_dev_cap_num(device, flush=flush)
    name = f"sm{capability}"
    return name.upper() if upper else name


def is_sm(
    *sms: int | str,
    device: torch.device | str | int | None = 0,
    flush: bool = False,
) -> bool:
    """Match any exact capability: ``is_sm("100", "103", device=..., flush=True)``.

    Accept integers and numeric strings, optionally prefixed by ``sm``/``SM``.
    Unknown numeric capabilities remain queryable without updating ARCH_SMS.
    Compiler targets such as ``sm100a``/``sm100f`` are not runtime capabilities.
    """
    if not sms:
        raise ValueError("is_sm requires at least one exact capability")
    capabilities = tuple(
        int(sm.lower().removeprefix("sm")) if isinstance(sm, str) else sm for sm in sms
    )
    return get_dev_cap_num(device, flush=flush) in capabilities


def is_ampere(
    device: torch.device | str | int | None = 0, *, flush: bool = False
) -> bool:
    """Match Ampere SM80/86/87/88; excludes Ada SM89 and later GPUs."""
    return get_dev_cap_num(device, flush=flush) in ARCH_SMS["ampere"]


def is_ada(device: torch.device | str | int | None = 0, *, flush: bool = False) -> bool:
    """Match Ada SM89, including RTX 4090 and L4/L40/L40S."""
    return get_dev_cap_num(device, flush=flush) in ARCH_SMS["ada"]


def is_hopper(
    device: torch.device | str | int | None = 0, *, flush: bool = False
) -> bool:
    """Match Hopper SM90 (H100/H200/H800/GH200), excluding Ada/RTX 4090."""
    return get_dev_cap_num(device, flush=flush) in ARCH_SMS["hopper"]


def is_blackwell(
    device: torch.device | str | int | None = 0, *, flush: bool = False
) -> bool:
    """Match the project's Blackwell training alias: SM100/103, B200/B300/GB200/GB300.

    This intentionally excludes SM110/120/121 (even GB10/DGX Spark), despite
    their Blackwell branding, and excludes Rubin SM107. Use ``is_sm`` for exact
    capabilities outside this alias; kernel support still belongs to its backend.
    """
    return get_dev_cap_num(device, flush=flush) in ARCH_SMS["blackwell"]


def is_rubin(
    device: torch.device | str | int | None = 0, *, flush: bool = False
) -> bool:
    """Match the explicitly listed Rubin SM107 alias, not arbitrary future GPUs."""
    return get_dev_cap_num(device, flush=flush) in ARCH_SMS["rubin"]
