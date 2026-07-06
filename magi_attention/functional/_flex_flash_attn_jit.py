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

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import jinja2
import torch

from magi_attention.common.jit import env as jit_env
from magi_attention.common.jit.core import JitSpec, gen_jit_spec
from magi_attention.common.jit.utils import write_if_different

logger = logging.getLogger(__name__)


_DTYPE_TO_CUTLASS = {
    torch.float16: "cutlass::half_t",
    torch.bfloat16: "cutlass::bfloat16_t",
    torch.float32: "float",
}

from magi_attention.env.build import is_no_build_cache  # noqa: E402

no_build_cache = is_no_build_cache()


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


def _assert_register_quota(producer_regs: int, consumer_regs: int):
    """Mirror of the kernel-side setmaxnreg static_asserts (flash_{fwd,bwd}_kernel_sm90.h)."""
    for regs in (producer_regs, consumer_regs):
        assert (
            regs % 8 == 0 and 24 <= regs <= 256
        ), f"setmaxnreg quota must be a multiple of 8 in [24, 256], got {regs}"


def _ffa_register_quota(
    direction: str,
    head_dim: int,
    kblock_m: int | None,
    swap_ab: bool,
    swap_bwd_qk_loop: bool,
    block_sparse: bool,
    index_sparse: bool,
    sparse_dx_tma_reduce: bool,
    k_block_size: int = 1,
) -> tuple[int, int]:
    """Select the setmaxnreg quotas (producer/load WG, consumer/mma WG) for one variant.

    This is the single source of truth for register allocation; the kernels
    (flash_{fwd,bwd}_kernel_sm90.h) only static_assert the constraints.

    fwd (producer, consumer) by mode:
      - scatter load ((index_sparse or block_sparse) with kbs<kBlockN): (64, 216)
        cp.async producer warpgroup needs more registers than the TMA producer warp.
      - TMA KV (kbs>=kBlockN, i.e. _is_contiguous=true): use Dense quota
        since only thread 0 issues TMA.
      - dense, by MMA warpgroup count (1/2/3 from kBlockM/64, or 1 when SwapAB): (56, 256),
        (40, 232), (32, 160).

    bwd (producer, consumer) by mode:
      - dense: (40, 232) at 2 MMA WGs, (40, 152) at 3 (kHeadDim=192).
      - scatter + TMA inner (sparse_dx_tma_reduce=True): (40, 232). Inner Q/dO are
        loaded via TMA (1 warp, minimal regs), so producer matches Dense quota.
        cuobjdump verified: pr=56→STACK=32 (consumer spills), pr=40→STACK=0.
      - scatter + cp.async inner (sparse_dx_tma_reduce=False, scalar-atomicAdd dX
        store fallback): (104, ...). The store-warp code spills at 88 (STACK 3272B).
      Historical note: the sweep that found pr=56 as sweet spot was done with cp.async
      (BlockSparse LoopQ, S=64K/256K), not TMA. With TMA inner loads, the producer
      needs far fewer live variables, and pr=40 gives consumer enough regs to avoid
      MMA accumulator spills.

    Env override MAGI_ATTENTION_FFA_BWD_PRODUCER_REGS (bwd only): producer is forced to
    the given value and the consumer is rederived from the weighted budget.
    """
    kblock_n_fwd = 128  # Default FWD tile N
    if direction == "fwd":
        assert kblock_m is not None
        # CpAsync scatter path is used when _is_contiguous is false:
        #   _is_contiguous = (!IndexSparse && !BlockSparse) || (KBlockSize >= kBlockN)
        # So scatter is needed when (IndexSparse || BlockSparse) && KBlockSize < kBlockN.
        uses_scatter = (index_sparse or block_sparse) and k_block_size < kblock_n_fwd
        if uses_scatter:
            producer_regs, consumer_regs = 64, 216
        else:
            # TMA KV paths (Dense, BlockSparse, IndexSparse kbs>=kBlockN): same quota
            # mirrors NumMmaWarpGroups = size(TiledMmaPV)/128 (AtomLayoutQK in mainloop_fwd)
            num_mma_wgs = 1 if swap_ab else kblock_m // 64
            producer_regs, consumer_regs = {1: (56, 256), 2: (40, 232), 3: (32, 160)}[
                num_mma_wgs
            ]
    else:
        # mirrors NumMmaWarpGroups in run_mha_bwd_ (flash_bwd_launch_template.h)
        num_mma_wgs = 2 if swap_bwd_qk_loop else (3 if head_dim == 192 else 2)
        inner_use_scatter = block_sparse or index_sparse
        budget = 168 * (1 + num_mma_wgs)
        if inner_use_scatter:
            if sparse_dx_tma_reduce:
                # TMA path: inner Q/dO loaded via TMA (1 warp), producer needs
                # minimal regs — match Dense quota so consumer gets 232 for MMA
                # accumulators. cuobjdump: pr=56→STACK=32 (spills), pr=40→STACK=0.
                producer_regs = 40
            else:
                producer_regs = 104
            consumer_regs = ((budget - producer_regs) // num_mma_wgs) // 8 * 8
        else:
            producer_regs, consumer_regs = 40, (232 if num_mma_wgs == 2 else 152)
        _bpr = os.environ.get("MAGI_ATTENTION_FFA_BWD_PRODUCER_REGS")
        if _bpr is not None and int(_bpr) != 0:
            producer_regs = int(_bpr)
            consumer_regs = ((budget - producer_regs) // num_mma_wgs) // 8 * 8
        assert (
            producer_regs + num_mma_wgs * consumer_regs <= budget
        ), f"bwd register quota {producer_regs}+{num_mma_wgs}x{consumer_regs} exceeds budget {budget}"
    _assert_register_quota(producer_regs, consumer_regs)
    return producer_regs, consumer_regs


def get_ffa_uri(
    arch_sm_num: str,
    direction: str,
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
    softcap: bool,
    disable_atomic_reduction: bool,
    disable_dq_atomic_reduction: bool,
    deterministic: bool,
    kblock_m: int | None,
    kblock_n: int | None,
    auto_range_merge: bool,
    swap_ab: bool,
    pack_gqa: bool,
    cat_gqa: bool,
    pack_gqa_factor: int,
    block_sparse: bool,
    index_sparse: bool,
    swap_bwd_qk_loop: bool,
    profile_mode: bool,
    return_max_logits: bool,
    dq_dtype: torch.dtype | None = None,
    dkv_dtype: torch.dtype | None = None,
) -> str:
    def _dtype_name(dt: torch.dtype) -> str:
        return str(dt).split(".")[-1]

    return (
        f"flex_flash_attn_sm_{arch_sm_num}_"
        f"{direction}_"
        f"{head_dim}hd_"
        f"compute_{_dtype_name(compute_dtype)}"
        f"{f'_out_{_dtype_name(output_dtype)}' if output_dtype is not None else ''}"
        f"{f'_dq_{_dtype_name(dq_dtype)}' if dq_dtype is not None else ''}"
        f"{f'_dkv_{_dtype_name(dkv_dtype)}' if dkv_dtype is not None else ''}"
        f"{'_softcap' if softcap else ''}"
        f"{'' if disable_atomic_reduction else '_atomic'}"
        f"{'' if not disable_dq_atomic_reduction else '_nodqatomic'}"
        f"{'_deterministic' if deterministic else ''}"
        f"{'_autorangemerge' if auto_range_merge else ''}"
        f"{'_swapab' if swap_ab else ''}"
        f"{f'_packgqa{pack_gqa_factor}' if pack_gqa else ''}"
        f"{f'_catgqa{pack_gqa_factor}' if cat_gqa else ''}"
        f"{'_block_sparse' if block_sparse else ''}"
        f"{'_index_sparse' if index_sparse else ''}"
        f"{'_swapbwdqkloop' if swap_bwd_qk_loop else ''}"
        f"{'_profile_mode' if profile_mode else ''}"
        f"{'_return_max_logits' if return_max_logits else ''}"
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
    output_dtype: torch.dtype | None,
    ref_block_size: tuple[int, int] | None = None,
    swap_ab: bool = False,
    block_sparse: bool = False,
    index_sparse: bool = False,
    swap_bwd_qk_loop: bool = False,
    return_max_logits: bool = False,
    dq_dtype: torch.dtype | None = None,
    dkv_dtype: torch.dtype | None = None,
    pack_gqa: bool = False,
    cat_gqa: bool = False,
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
    if direction == "fwd":
        assert output_dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ), "output_dtype must be float16, bfloat16 or float32"
        assert dq_dtype is None, "dq_dtype must be None when direction == 'fwd'"
        assert dkv_dtype is None, "dkv_dtype must be None when direction == 'fwd'"
    if direction == "bwd":
        assert output_dtype is None, "output_dtype must be None when direction == 'bwd'"
        assert dq_dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ), "dq_dtype must be float16, bfloat16 or float32"
        assert dkv_dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ), "dkv_dtype must be float16, bfloat16 or float32"
    if swap_ab:
        assert direction == "fwd", "swap_ab only take effect when direction == 'fwd'"
        assert ref_block_size in (
            (8, 64),
            (16, 64),
            (32, 64),
            (64, 64),
        ), "ref_block_size must be (8, 64), (16, 64), (32, 64) or (64, 64) when swap_ab == True"
    else:
        if ref_block_size is not None:
            kblock_m, kblock_n = ref_block_size
            assert kblock_m in (
                64,
                128,
                192,
            ), "ref_block_size: (kblock_m, kblock_n), kblock_m must be 64, 128 or 192 when swapab == False"
            assert (
                kblock_n % 16 == 0 and kblock_n <= 256
            ), "ref_block_size: (kblock_m, kblock_n), kblock_n <= 256 and kblock_n % 16 == 0 must be True"
    if swap_bwd_qk_loop:
        assert (
            direction == "bwd"
        ), "swap_bwd_qk_loop only take effect when direction == 'bwd'"
    if return_max_logits:
        assert (
            direction == "fwd"
        ), "return_max_logits only take effect when direction == 'fwd'"
    assert not (pack_gqa and cat_gqa), "pack_gqa and cat_gqa cannot be both True"
    if cat_gqa:
        assert direction == "bwd", "cat_gqa only take effect when direction == 'bwd'"


def get_ffa_jit_spec(
    arch: tuple[int, int],
    direction: Literal["fwd", "bwd"],
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype | None,
    softcap: bool,
    disable_atomic_reduction: bool,
    disable_dq_atomic_reduction: bool,
    deterministic: bool,
    ref_block_size: tuple[int, int] | None = None,
    auto_range_merge: bool = False,
    swap_ab: bool = False,
    pack_gqa: bool = False,
    cat_gqa: bool = False,
    pack_gqa_factor: int = 1,
    block_sparse: bool = False,
    index_sparse: bool = False,
    swap_bwd_qk_loop: bool = False,
    profile_mode: bool = False,
    return_max_logits: bool = False,
    dq_dtype: torch.dtype | None = None,
    dkv_dtype: torch.dtype | None = None,
    k_block_size: int = 1,
) -> tuple[JitSpec, str]:
    # TODO: add more sanity checks for the combinations of options
    sanity_check(
        arch=arch,
        direction=direction,
        head_dim=head_dim,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        ref_block_size=ref_block_size,
        swap_ab=swap_ab,
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        swap_bwd_qk_loop=swap_bwd_qk_loop,
        return_max_logits=return_max_logits,
        dq_dtype=dq_dtype,
        dkv_dtype=dkv_dtype,
        pack_gqa=pack_gqa,
        cat_gqa=cat_gqa,
    )

    # Convert arch to SM number
    arch_sm_num = f"{arch[0]}{arch[1]}"

    if ref_block_size is not None:
        kblock_m, kblock_n = ref_block_size
    else:
        if direction == "fwd":
            kblock_m, kblock_n = tile_size_fwd_sm90(head_dim, softcap)
        else:
            kblock_m, kblock_n = None, None

    uri = get_ffa_uri(
        arch_sm_num=arch_sm_num,
        direction=direction,
        head_dim=head_dim,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        softcap=softcap,
        disable_atomic_reduction=disable_atomic_reduction,
        disable_dq_atomic_reduction=disable_dq_atomic_reduction,
        deterministic=deterministic,
        kblock_m=kblock_m,
        kblock_n=kblock_n,
        auto_range_merge=auto_range_merge,
        swap_ab=swap_ab,
        pack_gqa=pack_gqa,
        cat_gqa=cat_gqa,
        pack_gqa_factor=pack_gqa_factor,
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        swap_bwd_qk_loop=swap_bwd_qk_loop,
        profile_mode=profile_mode,
        return_max_logits=return_max_logits,
        dq_dtype=dq_dtype,
        dkv_dtype=dkv_dtype,
    )

    logger.info("Generating FFA JIT spec for URI: %s", uri)
    logger.info(
        "FFA JIT params: arch=sm%s, direction=%s, head_dim=%d, compute_dtype=%s, "
        "output_dtype=%s, softcap=%s, deterministic=%s, block_size=%s, "
        "swap_ab=%s, pack_gqa=%s, cat_gqa=%s, pack_gqa_factor=%d, "
        "block_sparse=%s, index_sparse=%s, profile_mode=%s, return_max_logits=%s",
        arch_sm_num,
        direction,
        head_dim,
        compute_dtype,
        output_dtype,
        softcap,
        deterministic,
        ref_block_size,
        swap_ab,
        pack_gqa,
        cat_gqa,
        pack_gqa_factor,
        block_sparse,
        index_sparse,
        profile_mode,
        return_max_logits,
    )

    # k_block_size is a compile-time constant baked into the kernel.
    # Different values produce separate JIT cache entries.
    if k_block_size > 1:
        uri += f"_kbs{k_block_size}"

    # Optional compile-time overrides for internal kernel tuning knobs (test/bench only)
    extra_template_args: dict[str, str] = {"k_block_size": str(k_block_size)}
    _iwg = os.environ.get("MAGI_ATTENTION_FFA_INTRA_WG_OVERLAP")
    if _iwg is not None and direction == "fwd":
        extra_template_args["intra_wg_overlap"] = _iwg.lower()
        uri += f"_iwg{_iwg}"
    _umd = os.environ.get("MAGI_ATTENTION_FFA_USE_MASK_DISPATCH")
    if _umd is not None and direction == "bwd":
        extra_template_args["use_mask_dispatch"] = _umd.lower()
        uri += f"_umd{_umd}"
    _idm = os.environ.get("MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN")
    if _idm is not None:
        extra_template_args["inner_dir_max_to_min"] = _idm.lower()
        uri += f"_idm{_idm}"
    # mask_mode: "regular"=0 (direct apply), "dispatch"=1 (3-lambda), "unified"=2
    # BWD default: unified (avoids 3-lambda code bloat that causes register spill).
    # FWD default: dispatch (template default '1' in jinja).
    _mask_mode_map = {"regular": "0", "dispatch": "1", "unified": "2"}
    _mm = os.environ.get("MAGI_ATTENTION_FFA_MASK_MODE")
    if _mm is not None:
        _mm_lower = _mm.lower()
        assert (
            _mm_lower in _mask_mode_map
        ), f"MAGI_ATTENTION_FFA_MASK_MODE must be regular/dispatch/unified, got {_mm}"
        extra_template_args["mask_mode_int"] = _mask_mode_map[_mm_lower]
        uri += f"_mm{_mm_lower}"
    elif direction == "bwd":
        extra_template_args["mask_mode_int"] = _mask_mode_map["unified"]
        uri += "_mmunified"
    # inner_use_scatter mirrors the mainloop predicate (mainloop_bwd_sm90_tma_gmma_ws.hpp):
    # InnerLoopK scatters K/V when block_sparse or index_sparse, InnerLoopQ scatters Q/dO when block_sparse
    # or index_sparse (inner_indices).
    _inner_use_scatter = block_sparse or index_sparse
    _dxp = os.environ.get("MAGI_ATTENTION_FFA_INNER_DX_STORE_IN_PRODUCER")
    # Applies to ALL bwd configs (Dense, IndexSparse, BlockSparse).
    # Default is true (producer store warp handles dX reduce-add to GMEM).
    # false → consumer WGs handle dX store directly; frees one producer warp but
    # shifts code pressure to the consumer (Dense: cr needs ~232 to avoid spill).
    if _dxp is not None and direction == "bwd":
        _dxp_lower = _dxp.lower()
        assert _dxp_lower in (
            "true",
            "false",
        ), f"MAGI_ATTENTION_FFA_INNER_DX_STORE_IN_PRODUCER must be true/false, got {_dxp}"
        extra_template_args["inner_dx_store_in_producer"] = _dxp_lower
        uri += f"_dxp{_dxp_lower}"
    # ─── InnerLoadMode: tma=0 (2D auto), tma1d=1 (bulk per-row, no-swizzle K), cpasync=2 ───
    if _inner_use_scatter:
        _load_env = os.environ.get("MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD")
        if _load_env is not None:
            _load_lower = _load_env.lower()
            _load_mode_map = {"tma": "0", "tma1d": "1", "cpasync": "2"}
            assert (
                _load_lower in _load_mode_map
            ), f"MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD must be tma/tma1d/cpasync, got {_load_env}"
            extra_template_args["inner_load_mode"] = _load_mode_map[_load_lower]
            uri += f"_sload{_load_mode_map[_load_lower]}"
    # ─── InnerStoreMode (BWD only): tma1d=1 (bulk reduce-add), cpasync=2 (atomicAdd) ───
    if direction == "bwd" and _inner_use_scatter:
        _store_env = os.environ.get(
            "MAGI_ATTENTION_FFA_SPARSE_INNER_STORE",
            os.environ.get("MAGI_ATTENTION_FFA_SPARSE_DX_TMA_REDUCE"),
        )
        if _store_env is not None:
            _store_lower = _store_env.lower()
            _store_mode_map = {"tma1d": "1", "cpasync": "2", "true": "1", "false": "2"}
            assert (
                _store_lower in _store_mode_map
            ), f"MAGI_ATTENTION_FFA_SPARSE_INNER_STORE must be tma1d/cpasync, got {_store_env}"
            extra_template_args["inner_store_mode"] = _store_mode_map[_store_lower]
            uri += f"_sstore{_store_mode_map[_store_lower]}"
        else:
            extra_template_args["inner_store_mode"] = "1"
    # Tile/stage overrides for A/B benchmarking (BWD only).
    # Each distinct combo produces a separate JIT URI → separate .so cache.
    if direction == "bwd":
        for _env_name, _tpl_key, _uri_key in [
            ("MAGI_ATTENTION_FFA_BWD_TILE_M", "bwd_tile_m", "tm"),
            ("MAGI_ATTENTION_FFA_BWD_TILE_N", "bwd_tile_n", "tn"),
            ("MAGI_ATTENTION_FFA_BWD_STAGES", "bwd_stages", "stg"),
            ("MAGI_ATTENTION_FFA_BWD_STAGES_DS", "bwd_stages_ds", "stds"),
            ("MAGI_ATTENTION_FFA_BWD_STAGES_V", "bwd_stages_v", "stv"),
            ("MAGI_ATTENTION_FFA_BWD_SCATTER_PAD", "bwd_scatter_pad", "sp"),
            ("MAGI_ATTENTION_FFA_BWD_LSE_UNION", "bwd_lse_union", "lu"),
            ("MAGI_ATTENTION_FFA_BWD_DKVACC_BYPASS", "bwd_dkvacc_bypass", "db"),
            ("MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC", "bwd_ununion_dkvacc", "uu"),
        ]:
            _val = os.environ.get(_env_name)
            if _val is not None:
                extra_template_args[_tpl_key] = str(int(_val))
                _uri_val = _val.replace("-", "n")
                uri += f"_{_uri_key}{_uri_val}"

        # Default LseDpsumUnion=1 for InnerLoopQ (saves ~5 producer spills, zero perf cost).
        # Env MAGI_ATTENTION_FFA_BWD_LSE_UNION overrides (handled above); only set default if not overridden.
        # CatGQA has different SMEM layout that is incompatible with LseDpsumUnion.
        if (
            "bwd_lse_union" not in extra_template_args
            and not swap_bwd_qk_loop
            and not cat_gqa
        ):
            extra_template_args["bwd_lse_union"] = "1"
            uri += "_lu1"

        # Default Ununion+stgV1 for LoopK: separates dK/dV SMEM buffers (+38T, +11%).
        # SMEM: 198KB → 214KB (stgV=1 frees 16KB for ununion's +32KB). No register change.
        # Convenience override: MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2=1 restores legacy behavior.
        # Fine-grained overrides: BWD_UNUNION_DKVACC / BWD_STAGES_V still work individually.
        _perf_union = os.environ.get("MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2")
        if _perf_union is not None and _perf_union != "0":
            extra_template_args.setdefault("bwd_ununion_dkvacc", "0")
            extra_template_args.setdefault("bwd_stages_v", "2")
            uri += "_pus1"
        elif swap_bwd_qk_loop:
            if "bwd_ununion_dkvacc" not in extra_template_args:
                extra_template_args["bwd_ununion_dkvacc"] = "1"
                uri += "_uu1"
            if "bwd_stages_v" not in extra_template_args:
                extra_template_args["bwd_stages_v"] = "1"
                uri += "_stv1"

        # Force Mma_dKV to SS mode (SMEM-SMEM) for benchmarking register pressure
        _force_ss = os.environ.get("MAGI_ATTENTION_FFA_BWD_FORCE_MMA_DKV_SS")
        if _force_ss is not None and _force_ss == "1":
            extra_template_args["force_mma_dkv_ss"] = "true"
            uri += "_fss1"

        # Per-switch debug overrides (LoopK perf isolation, correctness NOT guaranteed).
        # Require DKVACC_BYPASS=1 for dV/dK store skips.
        for _env_name, _tpl_key, _uri_key in [
            ("MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD", "bwd_skip_v_load", "svl"),
            ("MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE", "bwd_skip_dv_store", "svs"),
            ("MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE", "bwd_skip_dk_store", "sks"),
            ("MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA", "bwd_skip_dv_mma", "svm"),
            (
                "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK",
                "bwd_skip_dv_writeback",
                "svw",
            ),
            (
                "MAGI_ATTENTION_FFA_BWD_SKIP_DK_WRITEBACK",
                "bwd_skip_dk_writeback",
                "skw",
            ),
            ("MAGI_ATTENTION_FFA_BWD_DEFER_DV_R2S", "bwd_defer_dv_r2s", "ddv"),
        ]:
            _val = os.environ.get(_env_name)
            if _val is not None and _val != "0":
                extra_template_args[_tpl_key] = "true"
                uri += f"_{_uri_key}1"

        # IndexSparse InnerLoopQ with block-level K: override tile_n to match k_block_size
        # so that kBlockN = k_block_size (full-tile outer K, no waste).
        if (
            index_sparse
            and not swap_bwd_qk_loop
            and "bwd_tile_n" not in extra_template_args
        ):
            if k_block_size >= 128:
                extra_template_args["bwd_tile_n"] = str(k_block_size)
                uri += f"_tn{k_block_size}"

    # Register quota selection (single source of truth, kernels only assert)
    _producer_regs, _consumer_regs = _ffa_register_quota(
        direction=direction,
        head_dim=head_dim,
        kblock_m=kblock_m,
        swap_ab=swap_ab,
        swap_bwd_qk_loop=swap_bwd_qk_loop,
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        sparse_dx_tma_reduce=extra_template_args.get("inner_store_mode", "2") == "1",
        k_block_size=k_block_size,
    )
    extra_template_args[f"{direction}_producer_regs"] = str(_producer_regs)
    extra_template_args[f"{direction}_consumer_regs"] = str(_consumer_regs)
    uri += f"_pr{_producer_regs}_cr{_consumer_regs}"
    gen_directory = jit_env.MAGI_ATTENTION_GEN_SRC_DIR / uri
    gen_directory.mkdir(parents=True, exist_ok=True)
    logger.info("Generated source directory: %s", gen_directory)

    template_path = (
        Path(__file__).resolve().parents[1]
        / "csrc"
        / "flexible_flash_attention"
        / f"{direction}_inst_template.jinja"
    )
    logger.info("Loading Jinja template: %s", template_path)
    template = jinja2.Template(template_path.read_text(encoding="utf-8"))

    compute_t = _DTYPE_TO_CUTLASS[compute_dtype]
    out_t = (
        _DTYPE_TO_CUTLASS[output_dtype]
        if output_dtype is not None
        else _DTYPE_TO_CUTLASS[dq_dtype]
    )
    # set dq_t and dkv_t to out_t by default
    dq_t = _DTYPE_TO_CUTLASS[dq_dtype] if dq_dtype is not None else out_t
    dkv_t = _DTYPE_TO_CUTLASS[dkv_dtype] if dkv_dtype is not None else out_t
    has_softcap = bool(softcap)
    disable_atomic = bool(disable_atomic_reduction)
    disable_dq_atomic = bool(disable_dq_atomic_reduction)
    deterministic = bool(deterministic)
    profile_mode = bool(profile_mode)
    auto_range_merge = bool(auto_range_merge)
    swap_ab = bool(swap_ab)
    pack_gqa = bool(pack_gqa)
    cat_gqa = bool(cat_gqa)
    block_sparse = bool(block_sparse)
    swap_bwd_qk_loop = bool(swap_bwd_qk_loop)

    rendered = template.render(
        arch_sm_num=arch_sm_num,
        compute_t=compute_t,
        out_t=out_t,
        dq_t=dq_t,
        dkv_t=dkv_t,
        head_dim=head_dim,
        has_softcap=str(has_softcap).lower(),
        disable_atomic=str(disable_atomic).lower(),
        disable_dq_atomic=str(disable_dq_atomic).lower(),
        deterministic=str(deterministic).lower(),
        profile_mode=str(profile_mode).lower(),
        kblock_m=(kblock_m if kblock_m is not None else ""),
        kblock_n=(kblock_n if kblock_n is not None else ""),
        auto_range_merge=str(auto_range_merge).lower(),
        swap_ab=str(swap_ab).lower(),
        pack_gqa=str(pack_gqa).lower(),
        cat_gqa=str(cat_gqa).lower(),
        pack_gqa_factor=pack_gqa_factor,
        block_sparse=str(block_sparse).lower(),
        index_sparse=str(index_sparse).lower(),
        swap_bwd_qk_loop=str(swap_bwd_qk_loop).lower(),
        return_max_logits=str(bool(return_max_logits)).lower(),
        **extra_template_args,
    )

    inst_cu = gen_directory / f"{direction}_inst.cu"
    changed = write_if_different(inst_cu, rendered)
    logger.info(
        "Rendered template -> %s (%s)",
        inst_cu,
        "updated" if changed else "unchanged",
    )
    inst_sources = [
        inst_cu,
    ]

    common_sources = [
        jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR / "flex_flash_common.cpp",
        jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR / "flash_fwd_postprocess.cu",
        jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR / "flash_bwd_postprocess.cu",
    ]

    # For CUDA13+: the cccl header path needs to be explicitly included
    CUDA13_CCCL_PATH = os.path.join(
        os.getenv("CUDA_HOME", "/usr/local/cuda"), "include", "cccl"
    )

    include_dirs = [
        jit_env.MAGI_ATTENTION_INCLUDE_DIR.resolve(),
        jit_env.FLEXIBLE_FLASH_ATTENTION_CSRC_DIR.resolve(),
        jit_env.CUTLASS_INCLUDE_DIRS[0].resolve(),
        jit_env.CUTLASS_INCLUDE_DIRS[1].resolve(),
        CUDA13_CCCL_PATH,
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

        if profile_mode:
            from magi_attention import magi_attn_ext  # type: ignore[attr-defined]

            utils_so_path = Path(magi_attn_ext.__file__)

            common_objects += [str(utils_so_path)]

        return common_objects

    logger.info("Creating JIT spec for FFA URI: %s", uri)
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
    logger.info("FFA JIT spec ready for URI: %s", uri)

    return spec, uri


_ENV_KEYS_AFFECTING_COMPILATION: tuple[str, ...] = (
    "MAGI_ATTENTION_FFA_INTRA_WG_OVERLAP",
    "MAGI_ATTENTION_FFA_USE_MASK_DISPATCH",
    "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN",
    "MAGI_ATTENTION_FFA_MASK_MODE",
    "MAGI_ATTENTION_FFA_INNER_DX_STORE_IN_PRODUCER",
    "MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD",
    "MAGI_ATTENTION_FFA_SPARSE_INNER_STORE",
    "MAGI_ATTENTION_FFA_SPARSE_DX_TMA_REDUCE",
    "MAGI_ATTENTION_FFA_BWD_PRODUCER_REGS",
    "MAGI_ATTENTION_FFA_BWD_FORCE_MMA_DKV_SS",
    "MAGI_ATTENTION_FFA_BWD_TILE_M",
    "MAGI_ATTENTION_FFA_BWD_TILE_N",
    "MAGI_ATTENTION_FFA_BWD_STAGES",
    "MAGI_ATTENTION_FFA_BWD_STAGES_DS",
    "MAGI_ATTENTION_FFA_BWD_STAGES_V",
    "MAGI_ATTENTION_FFA_BWD_SCATTER_PAD",
    "MAGI_ATTENTION_FFA_BWD_LSE_UNION",
    "MAGI_ATTENTION_FFA_BWD_DKVACC_BYPASS",
    "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC",
    "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DK_WRITEBACK",
    "MAGI_ATTENTION_FFA_BWD_DEFER_DV_R2S",
    "MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2",
)


def _snapshot_env() -> tuple[tuple[str, str | None], ...]:
    """Capture all env vars that affect kernel compilation into a hashable tuple.

    Passed as ``_env_snapshot`` to ``get_ffa_jit_mod`` so that ``lru_cache``
    sees different keys when env vars change between calls.
    """
    return tuple((k, os.environ.get(k)) for k in _ENV_KEYS_AFFECTING_COMPILATION)


def get_ffa_jit_mod(
    direction: Literal["fwd", "bwd"],
    head_dim: int,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
    softcap: bool,
    disable_atomic_reduction: bool,
    disable_dq_atomic_reduction: bool,
    deterministic: bool,
    ref_block_size: tuple[int, int] | None = None,
    auto_range_merge: bool = False,
    swap_ab: bool = False,
    pack_gqa: bool = False,
    cat_gqa: bool = False,
    pack_gqa_factor: int = 1,
    block_sparse: bool = False,
    index_sparse: bool = False,
    swap_bwd_qk_loop: bool = False,
    profile_mode: bool = False,
    return_max_logits: bool = False,
    dq_dtype: torch.dtype | None = None,
    dkv_dtype: torch.dtype | None = None,
    k_block_size: int = 1,
    _env_snapshot: tuple[tuple[str, str | None], ...] = (),
) -> Any:
    assert torch.cuda.is_available(), "CUDA is not available"
    arch = torch.cuda.get_device_capability()
    check_cuda_compute_capability(arch)

    logger.info(
        "get_ffa_jit_mod called: direction=%s, head_dim=%d, compute_dtype=%s, "
        "output_dtype=%s, arch=%s",
        direction,
        head_dim,
        compute_dtype,
        output_dtype,
        arch,
    )

    pack_gqa_factor = 1 if not pack_gqa and not cat_gqa else pack_gqa_factor

    spec, _ = get_ffa_jit_spec(
        arch=arch,
        direction=direction,
        head_dim=head_dim,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        softcap=softcap,
        disable_atomic_reduction=disable_atomic_reduction,
        disable_dq_atomic_reduction=disable_dq_atomic_reduction,
        deterministic=deterministic,
        ref_block_size=ref_block_size,
        auto_range_merge=auto_range_merge,
        swap_ab=swap_ab,
        pack_gqa=pack_gqa,
        cat_gqa=cat_gqa,
        pack_gqa_factor=pack_gqa_factor,
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        swap_bwd_qk_loop=swap_bwd_qk_loop,
        profile_mode=profile_mode,
        return_max_logits=return_max_logits,
        dq_dtype=dq_dtype,
        dkv_dtype=dkv_dtype,
        k_block_size=k_block_size,
    )

    logger.info(
        "Building and loading FFA JIT module for direction=%s, head_dim=%d",
        direction,
        head_dim,
    )
    mod = spec.build_and_load()
    logger.info(
        "FFA JIT module loaded successfully for direction=%s, head_dim=%d",
        direction,
        head_dim,
    )
    return mod


if not no_build_cache:
    get_ffa_jit_mod = lru_cache(maxsize=None)(get_ffa_jit_mod)
