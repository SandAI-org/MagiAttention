# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import jinja2
import torch

from magi_attention.common.jit import env as jit_env
from magi_attention.common.jit.core import JitSpec, gen_jit_spec
from magi_attention.common.jit.utils import write_if_different

# isort: off
# We need to import the CUDA kernels after importing torch
is_ffa_utils_installed = False
try:
    from magi_attention import flexible_flash_attention_utils_cuda as ffa_utils  # type: ignore[attr-defined]

    is_ffa_utils_installed = True
except ImportError:
    pass

# isort: on

_DTYPE_TO_CUTLASS = {
    torch.float16: "cutlass::half_t",
    torch.bfloat16: "cutlass::bfloat16_t",
    torch.float32: "float",
}


def tile_size_fwd_sm90(head_dim: int, softcap: bool) -> tuple[int, int]:
    if head_dim <= 64:
        # return (192 if same_hdim else 64, 128 if same_hdim else 64, same_hdim, same_hdim)
        # With this workaround in Cutlass 3.8, tile size 192 x 128 got slower for non-causal, idk why
        # https://github.com/NVIDIA/cutlass/blob/v3.8.2/include/cute/container/tuple.hpp#L131
        return (192, 128)
        # Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
        # return (192, 192 if is_causal or is_local else 176, True, False)
    elif head_dim <= 128:
        return (128, 128)
        # (128, 192, False, False) and (192, 128, False, True) are quite good too
        # 128 x 192 hits the limit of smem if MmaPV_is_RS, 128 x 144 hits the limit if not MmaPV_is_RS
    elif head_dim <= 192:
        return (128, 96)  # 128 x 112 hits the limit of smem
    else:
        return (128, 64)


def tile_size_bwd_sm90(head_dim: int, softcap: bool) -> tuple[int, int]:
    if head_dim <= 64:
        return (128, 128)
    elif head_dim <= 128:
        return (64, 128)
    elif head_dim <= 192:
        return (64, 64)
    else:
        return (64, 64)


def get_fwd_tile_shape(
    head_dim: int, softcap: bool, ref_block_size: tuple[int, int] | None = None
) -> tuple[int, int]:
    if ref_block_size is not None:
        # TODO: we need a more reasonable method to get bwd tileshape from ref_block_size
        kblock_m, kblock_n = ref_block_size
    else:
        kblock_m, kblock_n = tile_size_fwd_sm90(head_dim, softcap)

    return kblock_m, kblock_n


def get_bwd_tile_shape(
    head_dim: int, softcap: bool, ref_block_size: tuple[int, int] | None = None
) -> tuple[int, int]:
    if ref_block_size is not None:
        # TODO: we need a more reasonable method to get bwd tileshape from ref_block_size
        kblock_m, kblock_n = ref_block_size
    else:
        kblock_m, kblock_n = tile_size_bwd_sm90(head_dim, softcap)

    return kblock_m, kblock_n


def round_up_headdim(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    elif head_dim <= 128:
        return 128
    elif head_dim <= 192:
        return 192
    else:
        return 256


def get_ffa_uri(
    arch_sm_num: str,
    direction: str,
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
    softcap: bool,
    disable_atomic_reduction: bool,
    deterministic: bool,
    profile_mode: bool,
    kblock_m: int | None,
    kblock_n: int | None,
) -> str:
    def _dtype_name(dt: torch.dtype) -> str:
        return str(dt).split(".")[-1]

    return (
        f"flex_flash_attn_sm_{arch_sm_num}_"
        f"{direction}_"
        f"{head_dim}hd_"
        f"compute_{_dtype_name(compute_dtype)}_"
        f"out_{_dtype_name(output_dtype)}"
        f"{'_softcap' if softcap else ''}"
        f"{'' if disable_atomic_reduction else '_atomic'}"
        f"{'_deterministic' if deterministic else ''}"
        f"{'_profile_mode' if profile_mode else ''}"
        + (
            f"_m{kblock_m}n{kblock_n}"
            if kblock_m is not None and kblock_n is not None
            else ""
        )
    )


def check_cuda_compute_capability(arch: tuple[int, int]):
    assert arch == (9, 0), "flex_flash_attn only supports sm90"


def sanity_check(
    arch: tuple[int, int],
    direction: Literal["fwd", "bwd"],
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    check_cuda_compute_capability(arch)
    assert direction in ("fwd", "bwd"), "direction must be either fwd or bwd"
    assert head_dim <= 128, "head_dim must be <= 128 for now"
    assert round_up_headdim(head_dim) in (
        64,
        128,
    ), "round_up_headdim(head_dim) must be 64 or 128 for now"
    assert compute_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "compute_dtype must be float16 or bfloat16"
    assert output_dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "output_dtype must be float16, bfloat16 or float32"


def get_ffa_jit_spec(
    arch: tuple[int, int],
    direction: Literal["fwd", "bwd"],
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
    softcap: bool,
    disable_atomic_reduction: bool,
    deterministic: bool,
    profile_mode: bool,
    ref_block_size: tuple[int, int] | None = None,
) -> tuple[JitSpec, str]:
    sanity_check(arch, direction, head_dim, compute_dtype, output_dtype)

    # Convert arch to SM number
    arch_sm_num = f"{arch[0]}{arch[1]}"

    if direction == "fwd":
        kblock_m, kblock_n = get_fwd_tile_shape(head_dim, softcap, ref_block_size)
    else:
        kblock_m, kblock_n = get_bwd_tile_shape(head_dim, softcap, ref_block_size)

    uri = get_ffa_uri(
        arch_sm_num,
        direction,
        head_dim,
        compute_dtype,
        output_dtype,
        softcap,
        disable_atomic_reduction,
        deterministic,
        profile_mode,
        kblock_m,
        kblock_n,
    )

    gen_directory = jit_env.MAGI_ATTENTION_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)

    # Read and render the Jinja template
    template_path = (
        Path(__file__).resolve().parents[1]
        / "csrc"
        / "flexible_flash_attention"
        / f"{direction}_inst_template.jinja"
    )
    template = jinja2.Template(template_path.read_text(encoding="utf-8"))

    compute_t = _DTYPE_TO_CUTLASS[compute_dtype]
    out_t = _DTYPE_TO_CUTLASS[output_dtype]
    has_softcap = bool(softcap)
    disable_atomic = bool(disable_atomic_reduction)
    profile_mode = bool(profile_mode)

    rendered = template.render(
        arch_sm_num=arch_sm_num,
        compute_t=compute_t,
        out_t=out_t,
        head_dim=head_dim,
        has_softcap=str(has_softcap).lower(),
        disable_atomic=str(disable_atomic).lower(),
        deterministic=str(deterministic).lower(),
        profile_mode=str(profile_mode).lower(),
        kblock_m=(kblock_m if kblock_m is not None else ""),
        kblock_n=(kblock_n if kblock_n is not None else ""),
    )

    inst_cu = gen_directory / f"{direction}_inst.cu"
    write_if_different(inst_cu, rendered)
    inst_sources = [
        inst_cu,
    ]

    common_sources = [
        jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR / "flex_flash_common.cpp",
        jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR / "fast_zero_fill.cu",
    ]

    include_dirs = [
        jit_env.MAGI_ATTENTION_INCLUDE_DIR.resolve(),
        jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR.resolve(),
        jit_env.CUTLASS_INCLUDE_DIRS[0].resolve(),
        jit_env.CUTLASS_INCLUDE_DIRS[1].resolve(),
    ]

    # Disable other head dimensions to reduce compile time
    disable_dims = {64, 128, 192, 256} - {head_dim}
    extra_cflags = []
    for d in sorted(disable_dims):
        extra_cflags.append(f"-DFLASHATTENTION_DISABLE_HDIM{d}")
    extra_cuda_cflags = []
    arch_sm_num_with_suffix = f"{arch_sm_num}a" if arch == (9, 0) else arch_sm_num
    extra_cuda_cflags.append(
        f"-gencode=arch=compute_{arch_sm_num_with_suffix},code=sm_{arch_sm_num_with_suffix}"
    )

    def extra_objects_cb():
        common_uri = f"{head_dim}hd_common"
        common_spec = gen_jit_spec(
            name=common_uri,
            sources=[str(x) for x in common_sources],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=None,
            extra_include_paths=[str(x) for x in include_dirs],
            needs_device_linking=False,
        )

        common_objects = common_spec.build_and_get_objects()

        assert is_ffa_utils_installed, (
            "The `flexible_flash_attention_utils_cuda` "
            "extension module is not installed. "
            "This is a required dependency for JIT compilation."
        )
        # add utils.so(Dynamic linking)
        utils_so_path = Path(ffa_utils.__file__)

        return common_objects + [str(utils_so_path)]

    spec = gen_jit_spec(
        name=uri,
        sources=[str(x) for x in inst_sources],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=None,
        extra_include_paths=[str(x) for x in include_dirs],
        extra_objects_cb=extra_objects_cb,
        needs_device_linking=False,
    )

    return spec, uri


def get_ffa_jit_mod(
    direction: Literal["fwd", "bwd"],
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
    softcap: bool,
    disable_atomic_reduction: bool,
    deterministic: bool,
    profile_mode: bool,
    ref_block_size: tuple[int, int] | None = None,
) -> Any:
    assert torch.cuda.is_available(), "CUDA is not available"
    arch = torch.cuda.get_device_capability()
    check_cuda_compute_capability(arch)

    spec, _ = get_ffa_jit_spec(
        arch,
        direction,
        head_dim,
        compute_dtype,
        output_dtype,
        softcap,
        disable_atomic_reduction,
        deterministic,
        profile_mode,
        ref_block_size,
    )

    return spec.build_and_load()


# Disable caching when MAGI_ATTENTION_NO_CACHE=1 (caching is enabled by default)
if os.getenv("MAGI_ATTENTION_NO_CACHE", "0") != "1":
    get_ffa_jit_mod = lru_cache(maxsize=None)(get_ffa_jit_mod)
