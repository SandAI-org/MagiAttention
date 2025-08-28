import torch
import jinja2

from magi_attention.common.jit.core import JitSpec, gen_jit_spec
from magi_attention.common.jit import env as jit_env
from pathlib import Path
import os
from functools import lru_cache
from typing import Literal

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
    arch: str,
    direction: str,
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
    softcap: bool,
    disable_atomic_reduction: bool,
    kblock_m: int | None,
    kblock_n: int | None,
) -> str:
    def _dtype_name(dt: torch.dtype) -> str:
        return str(dt).split(".")[-1]

    return (
        f"flex_flash_attn_sm_{arch}_"
        f"{direction}_"
        f"{head_dim}hd_"
        f"compute_{_dtype_name(compute_dtype)}_"
        f"out_{_dtype_name(output_dtype)}_"
        f"{'softcap' if softcap else 'nosoftcap'}_"
        f"{'noatomic' if disable_atomic_reduction else 'atomic'}"
        + (f"_m{kblock_m}n{kblock_n}" if kblock_m is not None and kblock_n is not None else "")
    )

@lru_cache(maxsize=None)
def get_ffa_jit_mod(
    arch: str,
    direction: Literal["fwd", "bwd"],
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
    softcap: bool,
    disable_atomic_reduction: bool,
    ref_block_size: tuple[int, int] | None = None,
) -> JitSpec:
    # sanity check
    assert direction in ("fwd", "bwd"), "direction must be either fwd or bwd"
    assert arch == "90a", "arch must be 90a for now"
    assert head_dim <= 128, "head_dim must be <= 128 for now"
    assert round_up_headdim(head_dim) in (64, 128), "round_up_headdim(head_dim) must be 64 or 128 for now"

    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    if ref_block_size is not None:
        kblock_m, kblock_n = ref_block_size
    else:
        if direction == "fwd":
            kblock_m, kblock_n = tile_size_fwd_sm90(head_dim, softcap)
        else:
            kblock_m, kblock_n = None, None

    uri = get_ffa_uri(
        arch,
        direction,
        head_dim,
        compute_dtype,
        output_dtype,
        softcap,
        disable_atomic_reduction,
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

    # Use function arguments as the single source of truth
    # Normalize arch into integer SM (supports inputs like "90", "90a")
    try:
        arch_sm = int(arch)
    except Exception:
        import re as _re
        m = _re.search(r"(\\d+)", arch)
        arch_sm = int(m.group(1)) if m else 90

    compute_t = _DTYPE_TO_CUTLASS[compute_dtype]
    out_t = _DTYPE_TO_CUTLASS[output_dtype]
    has_softcap = bool(softcap)
    disable_atomic = bool(disable_atomic_reduction)

    rendered = template.render(
        arch_sm=arch_sm,
        compute_t=compute_t,
        out_t=out_t,
        head_dim=head_dim,
        has_softcap=str(has_softcap).lower(),
        disable_atomic=str(disable_atomic).lower(),
        kblock_m=(kblock_m if kblock_m is not None else ""),
        kblock_n=(kblock_n if kblock_n is not None else ""),
    )

    inst_cu = gen_directory / "fwd_inst.cu"
    inst_cu.write_text(rendered, encoding="utf-8")

    # Minimal source set to build
    base_dir = Path(__file__).resolve().parents[1]
    csrc_dir = base_dir / "csrc" / "flexible_flash_attention"
    sources = [
        inst_cu,
        csrc_dir / "flex_flash_common.cpp",
        csrc_dir / "fast_zero_fill.cu",
    ]

    # Disable other head dimensions to reduce compile time
    disable_dims = {64, 128, 192, 256} - {head_dim}
    extra_cflags = []
    for d in sorted(disable_dims):
        extra_cflags.append(f"-DFLASHATTENTION_DISABLE_HDIM{d}")

    include_dirs = [
        csrc_dir,
        jit_env.CUTLASS_INCLUDE_DIRS[0],
        jit_env.CUTLASS_INCLUDE_DIRS[1],
    ]

    spec = gen_jit_spec(
        name=uri,
        sources=[str(x) for x in sources],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=[str(x) for x in include_dirs],
        needs_device_linking=False,
    )

    return spec.build_and_load(), uri