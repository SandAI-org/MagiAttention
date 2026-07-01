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

    # TODO: support inv_causal and bi_causal
    # inv_causal: ClassVar[int] = 2
    # bi_causal: ClassVar[int] = 3

    def is_valid(self, mask_type: int) -> bool:
        """Check if the given mask type is valid."""
        return mask_type in range(2)  # Update if more mask types are added


MT_MAP = _MaskTypeMap()


def normalize_mask_types(mask_types: torch.Tensor | int | None) -> int:
    """Translate the public ``mask_types`` argument into a single mask-type int.

    The full q/k ranges semantics allow a distinct mask type per range, but the
    current kernels only support a single mask type shared by all ranges. So this
    helper collapses the supported cases down to one ``MT_MAP`` int:

    - ``None``  -> all ranges use full attention (``MT_MAP.full``).
    - ``int``   -> all ranges share the same mask type (validated against ``MT_MAP``).
    - ``Tensor``-> per-range mask types (not yet supported by the kernel).
    """
    if mask_types is None:
        return MT_MAP.full
    if isinstance(mask_types, int):
        if not MT_MAP.is_valid(mask_types):
            raise ValueError(f"Invalid mask type: {mask_types}")
        return mask_types

    # TODO: support per-range mask_types (a cuda int32 tensor) once the kernel
    # can read a distinct mask type for each q/k range.
    raise NotImplementedError(
        "Per-range mask_types (torch.Tensor) is not supported yet."
    )


def ranges_to_cu_seqlens(ranges: torch.Tensor | None) -> torch.Tensor | None:
    """Collapse q/k ranges down to a cu_seqlens tensor (step-1 hack).

    The full q/k ranges semantics allow arbitrary (possibly overlapping /
    non-contiguous) per-range [start, end) intervals, but the current kernels
    only understand the varlen cu_seqlens layout. So as a first step we only
    accept ranges that are *equivalent* to a cu_seqlens partition, i.e.
    contiguous, non-overlapping intervals starting at 0:
    ``[[0, e0], [e0, e1], ...]`` -> ``[0, e0, e1, ...]``.

    The caller is responsible for guaranteeing this equivalence (no validating
    device sync is done here); a non-conforming input silently produces a
    wrong cu_seqlens.

    Args:
        ranges: an ``[N, 2]`` int32 cuda tensor of [start, end) intervals, or
            ``None`` for the dense (non-varlen) path.

    Returns:
        An ``[N + 1]`` int32 cu_seqlens tensor, or ``None`` if ``ranges`` is None.
    """
    if ranges is None:
        return None
    assert (
        ranges.dim() == 2 and ranges.shape[1] == 2
    ), f"ranges must be an [N, 2] tensor, got shape {tuple(ranges.shape)}"
    cu_seqlens = torch.cat([ranges[:1, 0], ranges[:, 1]]).to(torch.int32)
    return cu_seqlens.contiguous()


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
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
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
