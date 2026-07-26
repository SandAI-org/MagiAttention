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

"""Utility helpers for flex_flash_attn: arch detection, tile configs, tensor helpers,
and fake-tensor builders for bwd kernels."""

import hashlib
import inspect
import os
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, ClassVar, Tuple

import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

# isort: split
from quack.compile_utils import make_fake_tensor as fake_tensor

from magi_attention.utils.arch import get_dev_cap_num
from magi_attention.utils.version import is_cuda_version_ge, is_cuda_version_lt

if TYPE_CHECKING:
    from .sparse_utils import BlockSparseTensorsTorch

# ---------------------------------------------------------------------------
# Mask Type Map helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MaskTypeMap:
    """Immutable int keys identifying the attention mask types.

    Uses an empty ``__slots__`` so instances carry no
    per-instance state; the keys live as class-level constants and the frozen
    dataclass guarantees they cannot be reassigned on an instance.
    """

    __slots__ = ()

    full: ClassVar[int] = 0
    causal: ClassVar[int] = 1

    inv_causal: ClassVar[int] = 2
    bi_causal: ClassVar[int] = 3

    def is_valid(self, mask_type: int) -> bool:
        """Check if the given mask type is valid."""
        return mask_type in range(4)  # Update if more mask types are added


MT_MAP = _MaskTypeMap()


class MaskMode(Enum):
    """Host-side specialization mode for attention masks.

    Static modes preserve the existing compile-time-specialized kernels. The
    per-range mode identifies a runtime ``int32[R]`` mask tensor holding one
    ``MT_MAP`` entry per q/k range row.
    """

    STATIC_FULL = "static_full"
    STATIC_CAUSAL = "static_causal"
    PER_RANGE = "per_range"


@dataclass(frozen=True, eq=False)
class NormalizedMaskTypes:
    """Normalized host representation of the public ``mask_types`` argument."""

    mode: MaskMode
    static_mask_type: int | None = None
    per_range_mask_types: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.mode == MaskMode.PER_RANGE:
            if self.static_mask_type is not None or self.per_range_mask_types is None:
                raise ValueError(
                    "Per-range mask mode requires only per_range_mask_types"
                )
        elif self.per_range_mask_types is not None or self.static_mask_type is None:
            raise ValueError("Static mask mode requires only static_mask_type")

    @property
    def is_per_range(self) -> bool:
        """True when using a per-range mask tensor."""
        return self.mode == MaskMode.PER_RANGE


def normalize_mask_type_spec(
    mask_types: torch.Tensor | int | None,
    *,
    num_ranges: int | None = None,
    batch_size: int | None = None,
    is_varlen: bool | None = None,
) -> NormalizedMaskTypes:
    """Normalize ``mask_types`` into an explicit host-side mask mode.

    Args:
        mask_types: ``None`` for static full attention, a supported integer for
            a static mask, or a CUDA ``int32[R]`` tensor for per-range
            mask selection (one entry per q/k range row).
        num_ranges: Optional number of attention ranges ``R``. When provided, a
            per-range tensor must have exactly this many entries.
        batch_size: Deprecated alias for ``num_ranges`` (varlen sequence count
            when ranges form a cu_seqlens partition).
        is_varlen: Optional indication of whether the attention problem uses
            q/k ranges. Passing ``False`` with a per-range tensor is rejected.

    Returns:
        A :class:`NormalizedMaskTypes` carrying either a static mask type or the
        original per-range CUDA tensor.

    Note:
        Tensor values are contractually restricted to ``MT_MAP`` entries. They
        are not read back on the host here because doing so would introduce a
        device synchronization in the attention hot path.
    """
    if num_ranges is not None and batch_size is not None and num_ranges != batch_size:
        raise ValueError(
            f"num_ranges and batch_size disagree: {num_ranges} vs {batch_size}"
        )
    range_count = num_ranges if num_ranges is not None else batch_size
    if range_count is not None and range_count < 0:
        raise ValueError(f"num_ranges must be non-negative, got {range_count}")

    if mask_types is None:
        return NormalizedMaskTypes(
            mode=MaskMode.STATIC_FULL,
            static_mask_type=MT_MAP.full,
        )

    if isinstance(mask_types, bool):
        raise TypeError("mask_types must not be a bool")

    if isinstance(mask_types, int):
        if not MT_MAP.is_valid(mask_types):
            raise ValueError(f"Invalid mask type: {mask_types}")
        if mask_types in (MT_MAP.inv_causal, MT_MAP.bi_causal):
            # Only the per-range runtime path implements these; a scalar would
            # otherwise fall through to the Full specialization and silently
            # compute the wrong mask.
            raise NotImplementedError(
                f"Scalar mask_types={mask_types} (inv_causal / bi_causal) is not "
                "supported; pass a per-range int32[R] tensor instead"
            )
        mode = (
            MaskMode.STATIC_CAUSAL
            if mask_types == MT_MAP.causal
            else MaskMode.STATIC_FULL
        )
        return NormalizedMaskTypes(mode=mode, static_mask_type=mask_types)

    if not isinstance(mask_types, torch.Tensor):
        raise TypeError(
            "mask_types must be None, an int, or a torch.Tensor, "
            f"got {type(mask_types)!r}"
        )
    if is_varlen is False:
        raise ValueError("Per-range mask_types is supported only with q/k ranges")
    if not mask_types.is_cuda:
        raise ValueError("Per-range mask_types must be a CUDA tensor")
    if mask_types.dtype != torch.int32:
        raise TypeError(
            "Per-range mask_types must have dtype torch.int32, "
            f"got {mask_types.dtype}"
        )
    if mask_types.dim() != 1:
        raise ValueError(
            "Per-range mask_types must have shape [num_ranges], "
            f"got {tuple(mask_types.shape)}"
        )
    if not mask_types.is_contiguous():
        raise ValueError("Per-range mask_types must be contiguous")
    if range_count is not None and mask_types.numel() != range_count:
        raise ValueError(
            "Per-range mask_types length must match num_ranges: "
            f"got {mask_types.numel()} and {range_count}"
        )

    return NormalizedMaskTypes(
        mode=MaskMode.PER_RANGE,
        per_range_mask_types=mask_types,
    )


def validate_per_range_mask_feature_support(
    spec: NormalizedMaskTypes,
    *,
    major_arch: int,
    is_local: bool = False,
    has_mask_mod: bool = False,
    has_block_sparse: bool = False,
    has_score_mod: bool = False,
    has_softcap: bool = False,
) -> None:
    """Reject per-range mask combinations that V1 does not support.

    Static mask modes are always accepted here. Value-domain checks for the
    per-range CUDA tensor remain a caller contract (no host D2H sync).
    """
    if not spec.is_per_range:
        return

    if major_arch not in (10, 11):
        raise NotImplementedError(
            "Per-range mask_types is only supported on SM100/SM110"
        )
    if is_local:
        raise NotImplementedError(
            "Per-range mask_types cannot be combined with local / "
            "sliding-window masking"
        )
    if has_mask_mod:
        raise NotImplementedError(
            "Per-range mask_types cannot be combined with mask_mod"
        )
    if has_block_sparse:
        raise NotImplementedError(
            "Per-range mask_types cannot be combined with block sparsity"
        )
    if has_score_mod:
        raise NotImplementedError(
            "Per-range mask_types cannot be combined with score_mod"
        )
    if has_softcap:
        raise NotImplementedError(
            "Per-range mask_types cannot be combined with softcap"
        )


def normalize_mask_types(mask_types: torch.Tensor | int | None) -> int:
    """Translate supported legacy cases into a single mask-type integer.

    Prefer :func:`normalize_mask_type_spec` in new code. This wrapper keeps the
    historical scalar-only API:

    - ``None``  -> ``MT_MAP.full``
    - ``int``   -> validated static mask type
    - ``Tensor``-> rejected (use the FlexFlashAttn host path instead)
    """
    spec = normalize_mask_type_spec(mask_types)
    if spec.is_per_range:
        raise NotImplementedError(
            "Per-range mask_types must go through flex_flash_attn_func / "
            "normalize_mask_type_spec; the scalar normalize_mask_types API "
            "only supports a single static mask type."
        )
    assert spec.static_mask_type is not None
    return spec.static_mask_type


def merge_ranges(
    outer_ranges: torch.Tensor,
    inner_ranges: torch.Tensor,
    mask_types: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Sort relations by outer range and deduplicate for merged-range scheduling.

    Torch-native mirror of the magi_attn_ext helper of the same name: returns
    (merge_outer_ranges, sorted_outer_ranges, sorted_inner_ranges,
    sorted_mask_types, qk_map, unique_count). Rows past unique_count are
    zero-padded; qk_map[j] is where merged range j starts in the sorted arrays.
    """
    num_ranges = outer_ranges.shape[0]
    device = outer_ranges.device
    key = (outer_ranges[:, 0].to(torch.int64) << 32) | outer_ranges[:, 1].to(
        torch.int64
    )
    order = torch.argsort(key, stable=True)
    sorted_outer = outer_ranges.index_select(0, order)
    sorted_inner = inner_ranges.index_select(0, order)
    sorted_types = mask_types.index_select(0, order)
    sorted_key = key.index_select(0, order)

    is_first = torch.ones(num_ranges, dtype=torch.bool, device=device)
    is_first[1:] = sorted_key[1:] != sorted_key[:-1]
    unique_count = is_first.sum(dtype=torch.int32)
    merged_idx = torch.cumsum(is_first, dim=0) - 1

    qk_map = torch.zeros(num_ranges, dtype=torch.int32, device=device)
    qk_map.scatter_reduce_(
        0,
        merged_idx,
        torch.arange(num_ranges, dtype=torch.int32, device=device),
        reduce="amin",
        include_self=False,
    )
    merge_outer = torch.zeros_like(sorted_outer)
    merge_outer.scatter_reduce_(
        0,
        merged_idx.unsqueeze(1).expand(-1, 2),
        sorted_outer,
        reduce="amin",
        include_self=False,
    )
    return merge_outer, sorted_outer, sorted_inner, sorted_types, qk_map, unique_count


def ranges_to_cu_seqlens(ranges: torch.Tensor | None) -> torch.Tensor | None:
    """Collapse q/k ranges down to a cu_seqlens tensor.

    # DEVIATION: still collapse validated ranges to cu_seqlens for SM100
    # Reason: SeqlenInfo / kernels still read cu_seqlens until Phase 2 true-range
    # Recovery: call validate_true_ranges first; Phase 2 removes this helper

    Args:
        ranges: an ``[R, 2]`` int32 cuda tensor of [start, end) intervals, or
            ``None`` for the dense path. Must already satisfy
            :func:`validate_true_ranges` (including the temporary cu-partition
            requirement while this collapse remains).

    Returns:
        An ``[R + 1]`` int32 cu_seqlens tensor, or ``None`` if ``ranges`` is None.
    """
    if ranges is None:
        return None
    assert (
        ranges.dim() == 2 and ranges.shape[1] == 2
    ), f"ranges must be an [R, 2] tensor, got shape {tuple(ranges.shape)}"
    cu_seqlens = torch.cat([ranges[:1, 0], ranges[:, 1]]).to(torch.int32)
    return cu_seqlens.contiguous()


def _validate_ranges_tensor(ranges: torch.Tensor, *, name: str) -> int:
    """Check one ``[R, 2]`` ranges tensor structurally and return ``R``."""
    if not isinstance(ranges, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(ranges)!r}")
    if not ranges.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if ranges.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {ranges.dtype}")
    if ranges.dim() != 2 or ranges.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape [num_ranges, 2], got {tuple(ranges.shape)}"
        )
    if not ranges.is_contiguous():
        raise ValueError(f"{name} must be contiguous")

    num_ranges = int(ranges.shape[0])
    if num_ranges == 0:
        raise ValueError(f"{name} must contain at least one range")
    return num_ranges


def validate_true_ranges(
    q_ranges: torch.Tensor | None,
    k_ranges: torch.Tensor | None,
    *,
    mask_types: torch.Tensor | int | None = None,
) -> int | None:
    """Check q/k ranges structurally and return ``R``; both ``None`` is dense.

    Range geometry (bounds, overlap, cross-relation disjointness) is a caller
    contract: reading the values back would sync the device on the hot path.
    """
    if q_ranges is None and k_ranges is None:
        return None
    if q_ranges is None or k_ranges is None:
        raise ValueError(
            "q_ranges and k_ranges must both be provided, or both be None"
        )

    num_q = _validate_ranges_tensor(q_ranges, name="q_ranges")
    num_k = _validate_ranges_tensor(k_ranges, name="k_ranges")
    if num_q != num_k:
        raise ValueError(
            f"q_ranges and k_ranges must have the same number of rows: "
            f"{num_q} vs {num_k}"
        )

    if isinstance(mask_types, torch.Tensor) and mask_types.numel() != num_q:
        raise ValueError(
            "Per-range mask_types length must match num_ranges: "
            f"got {mask_types.numel()} and {num_q}"
        )

    return num_q


# ---------------------------------------------------------------------------
# Torch FlexAttention-style / block-sparse args bundle
# ---------------------------------------------------------------------------


@dataclass
class TorchFlexAttnArgs:
    """Bundle of the optional torch FlexAttention-style / block-sparse args.

    These mirror torch's ``flex_attention`` programmable interface
    (``score_mod`` / ``mask_mod``) plus block sparsity, and are threaded as a
    single object through the FFA fwd/bwd entry points so the common dense /
    varlen signature stays clean.

    fwd reads: ``score_mod``, ``mask_mod``, ``aux_tensors``,
    ``block_sparse_tensors``.
    bwd reads: ``score_mod``, ``score_mod_bwd``, ``mask_mod``, ``aux_tensors``,
    ``block_sparse_tensors_bwd``.
    """

    score_mod: Callable | None = None
    score_mod_bwd: Callable | None = None
    mask_mod: Callable | None = None
    aux_tensors: list[torch.Tensor] | None = None
    block_sparse_tensors: "BlockSparseTensorsTorch | None" = None
    block_sparse_tensors_bwd: "BlockSparseTensorsTorch | None" = None

    def drop_aux_tensors(self) -> "TorchFlexAttnArgs":
        """Return a copy with ``aux_tensors`` cleared.

        Used in autograd ``forward`` before stashing this bundle on ``ctx``:
        the real aux tensors are tracked via ``save_for_backward`` instead, so
        keeping a direct reference here would bypass autograd's bookkeeping.
        """
        return replace(self, aux_tensors=None)

    def with_aux_tensors(
        self, aux_tensors: "list[torch.Tensor] | tuple[torch.Tensor, ...] | None"
    ) -> "TorchFlexAttnArgs":
        """Return a copy with ``aux_tensors`` restored from the given tensors.

        Used in autograd ``backward`` to refill the aux tensors recovered from
        ``ctx.saved_tensors`` (which were dropped in ``forward``).
        """
        return replace(self, aux_tensors=list(aux_tensors) if aux_tensors else None)


# ---------------------------------------------------------------------------
# Arch helpers
# ---------------------------------------------------------------------------


def parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re

    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def get_device_arch() -> tuple[int, int]:
    """Cached device arch check.

    Override with MAGI_ATTENTION_FFA_CUTEDSL_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      MAGI_ATTENTION_FFA_CUTEDSL_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)

    Returns:
        A tuple of (arch, major_arch) where:
        - arch: int (e.g. 80, 90, 100, 120)
        - major_arch: int (e.g. 8 for 80, 9 for 90, 10 for 100/103/120)
    """
    arch_override = os.environ.get("MAGI_ATTENTION_FFA_CUTEDSL_ARCH", None)

    arch = (
        parse_arch_str(arch_override)
        if arch_override is not None
        else get_dev_cap_num()
    )

    major_arch = arch // 10

    return arch, major_arch


def validate_arch(arch: int, major_arch: int) -> None:
    """Validate supported architectures."""
    assert major_arch in range(8, 13), f"Unsupported compute capability: {arch}"


# ---------------------------------------------------------------------------
# Head-dim validation
# ---------------------------------------------------------------------------


def validate_head_dims(
    head_dim: int, head_dim_v: int, compute_capability: int, alignment: int
) -> None:
    """Validate head dimension constraints based on compute capability."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_dedicate_kernel_shape = head_dim == 256 and head_dim_v == 256
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128

    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    if compute_capability == 9:
        assert (
            is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0
        ), (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM90. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )
    elif compute_capability in [10, 11]:
        assert (
            (is_standard_range or is_deepseek_shape or is_dedicate_kernel_shape)
            and head_dim % alignment == 0
            and head_dim_v % alignment == 0
        ), (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
            f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, "
            f"or (192, 128) for DeepSeek, or (256, 256) for hd256."
        )


# ---------------------------------------------------------------------------
# Tile size configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def tile_size_fwd_sm90(
    head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None
):
    """Return FwdConfig for SM90 forward.

    Tile sizes and flags based on tile_size_fwd_sm90 in hopper/tile_size.h, adjusted
    for the Python kernel's different register/smem tradeoffs (benchmarked on H100 SXM).

    When sparse_block_size_q is set, tile_m must divide it. For head_dim <= 96 the
    optimal tile_m=192 is used when compatible, otherwise we fall back to 128.
    """
    if head_dim <= 64:
        # C++: 192×192 non-causal, 192×128 causal/local.
        # Python: 192×128 RS+OL is consistently best across seqlens.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        # C++: 192×144 noRS+OL for all cases.
        # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
        # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        else:
            return FwdConfig(192, 144, False, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return FwdConfig(128, tile_n, True, True)


@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warp groups (total threads = (num_wg + 1) * 128)
    dQ_single_wg: bool = False


def tile_size_bwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=None):
    """Return BwdConfig for SM90.

    Configs based on C++ FA3 hopper/flash_bwd_launch_template.h,
    benchmarked on H100 SXM.
    """
    if head_dim <= 64:
        # C++ FA3: 128, 128, 64, ..., 2, 2, true, false, false, 2, 1, 2, 2
        return BwdConfig(
            m_block_size=128,
            n_block_size=128,
            num_stages_Q=2,
            num_stages_dO=2,
            num_stages_PdS=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        # C++ FA3: 64, 128, 96, dQ_swapAB=False
        return BwdConfig(
            m_block_size=64,
            n_block_size=128,
            num_stages_Q=2,
            num_stages_dO=2,
            num_stages_PdS=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2,
            num_stages_dO=2,
            num_stages_PdS=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        if hdimv128:
            return BwdConfig(
                m_block_size=64,
                n_block_size=96,
                num_stages_Q=2,
                num_stages_dO=2,
                num_stages_PdS=1,
                SdP_swapAB=False,
                dKV_swapAB=True,
                dQ_swapAB=False,
                AtomLayoutMSdP=1,
                AtomLayoutNdKV=2,
                AtomLayoutMdQ=1,
                num_wg=2,
            )
        else:
            return BwdConfig(
                m_block_size=64,
                n_block_size=96,
                num_stages_Q=2,
                num_stages_dO=1,
                num_stages_PdS=1,
                SdP_swapAB=False,
                dKV_swapAB=True,
                dQ_swapAB=False,
                AtomLayoutMSdP=1,
                AtomLayoutNdKV=2,
                AtomLayoutMdQ=1,
                num_wg=2,
            )
    else:
        # hdim 256
        return BwdConfig(
            m_block_size=64,
            n_block_size=64,
            num_stages_Q=1,
            num_stages_dO=1,
            num_stages_PdS=1,
            SdP_swapAB=False,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=1,
            AtomLayoutMdQ=1,
        )


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert (
        t.shape == expected_shape
    ), f"{name} shape {t.shape} != expected {expected_shape}"
    assert (
        t.dtype == expected_dtype
    ), f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert (
        t.device == expected_device
    ), f"{name} device {t.device} != expected {expected_device}"
    assert t.is_cuda, f"{name} must be on CUDA"


# ---------------------------------------------------------------------------
# Backward fake-tensor builder
# ---------------------------------------------------------------------------


def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k):
    sym = cute.sym_int
    # divisibility in elements: assumed_align_bytes = divisibility * dtype.width // 8
    # For 16-byte align: fp16/bf16 → divisibility=8, float32 → divisibility=4
    div = 128 // dtype.width  # 8 for fp16/bf16
    # Shared sym_ints for dimensions that must match across tensors
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_dv_rounded = sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_dv_rounded = sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    if not varlen_q:
        mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    else:
        mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (h_q, total_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(
                Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4
            )
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return (
        mQ,
        mK,
        mV,
        mO,
        mdO,
        mdQ,
        mdK,
        mdV,
        mLSE,
        mLSElog2,
        mPdPsum,
        dQaccum,
        mdKaccum,
        mdVaccum,
    )


# ---------------------------------------------------------------------------
# Host-side orchestration helpers (config flags, callable hashing, score mods)
# ---------------------------------------------------------------------------

_MIXER_ATTRS = ("__vec_size__",)


def _is_cuda_12() -> bool:
    """Check if the CUDA toolkit version is 12.x."""
    return is_cuda_version_ge("12") and is_cuda_version_lt("13")


def is_ffa_clc_enabled() -> bool:
    return os.environ.get("MAGI_ATTENTION_FFA_CUTEDSL_CLC", "0") == "1"


def is_ffa_2cta_disabled(is_fwd: bool = False) -> bool:
    _ffa_disable_2cta_enabled: bool = (
        os.environ.get("MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA", "0") == "1"
    )

    if is_fwd:
        # NOTE: 2CTA forward non-causal has a codegen regression on CUDA 12.x
        # that causes ~18% slowdown compared to 1CTA. This is fixed in CUDA 13.x.
        return _ffa_disable_2cta_enabled or _is_cuda_12()
    else:
        return _ffa_disable_2cta_enabled


def _compute_base_hash(func: Callable) -> str:
    """Compute hash from source code or bytecode and closure values."""
    try:
        data = inspect.getsource(func).encode()
    except (OSError, TypeError):
        if hasattr(func, "__code__") and func.__code__ is not None:
            data = func.__code__.co_code
        else:
            data = repr(func).encode()

    hasher = hashlib.sha256(data)

    if hasattr(func, "__closure__") and func.__closure__ is not None:
        for cell in func.__closure__:
            hasher.update(repr(cell.cell_contents).encode())

    return hasher.hexdigest()


def hash_callable(
    func: Callable, mixer_attrs: Tuple[str] = _MIXER_ATTRS, set_cute_hash: bool = True
) -> str:
    """Hash a callable based on the source code or bytecode and closure values.
    Fast-path: if the callable (or its __wrapped__ base) has a ``__cute_hash__``
    attribute, that value is returned immediately as the base hash, then
    metadata dunders are mixed in to produce the final dict-key hash.
    set_cute_hash: whether or not to set func.__cute_hash__
    """
    # Resolve base hash
    if hasattr(func, "__cute_hash__"):
        base_hash = func.__cute_hash__
    else:
        # Unwrap decorated functions (e.g., cute.jit wrappers).
        base_func = getattr(func, "__wrapped__", func)

        if hasattr(base_func, "__cute_hash__"):
            base_hash = base_func.__cute_hash__
        else:
            base_hash = _compute_base_hash(base_func)

            if set_cute_hash:
                base_func.__cute_hash__ = base_hash  # type: ignore[union-attr]

    # Mix in mutable metadata dunders
    mixer_values = tuple(getattr(func, attr, None) for attr in mixer_attrs)

    if all(v is None for v in mixer_values):
        return base_hash

    hasher = hashlib.sha256(base_hash.encode())

    for attr, val in zip(_MIXER_ATTRS, mixer_values):
        hasher.update(f"{attr}={val!r}".encode())

    return hasher.hexdigest()


def create_softcap_scoremod(softcap_val):
    @cute.jit
    def scoremod_premask_fn(
        acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        scores = acc_S_SSA / softcap_val
        return softcap_val * cute.math.tanh(scores, fastmath=True)

    return scoremod_premask_fn


def create_softcap_scoremod_bwd(softcap_val):
    @cute.jit
    def scoremod_bwd_fn(
        grad_out_SSA,
        score_SSA,
        batch_idx,
        head_idx,
        q_idx,
        kv_idx,
        seqlen_info,
        aux_tensors,
    ):
        scores = score_SSA / softcap_val
        tanh_scores = cute.math.tanh(scores, fastmath=True)
        return grad_out_SSA * (1.0 - tanh_scores * tanh_scores)

    return scoremod_bwd_fn


def convert_from_dlpack_leading_static(
    x, leading_dim, alignment=16, static_modes=None, stride_order=None
) -> cute.Tensor:
    if stride_order is None:
        stride_order = x.dim_order()
    x_ = from_dlpack(x, assumed_align=alignment)
    for i in range(x.ndim):
        if i != leading_dim and (static_modes is None or i not in static_modes):
            x_ = x_.mark_compact_shape_dynamic(mode=i, stride_order=stride_order)
    return x_
