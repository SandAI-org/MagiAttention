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

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

# mypy: disable-error-code="attr-defined,assignment"

import math

import cutlass.cute as cute
import torch
from cutlass import Int32

import magi_attention.kernel.cutedsl as magiattn_cutedsl
from magi_attention.common import AttnForwardMeta
from magi_attention.common.enum import AttnSinkLayout
from magi_attention.utils.dtype import to_cute_dtype

from .cache_utils import get_jit_cache
from .cutedsl_utils import (
    get_aux_tensor_metadata,
    get_broadcast_dims,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from .ffa_bwd_postprocess import (
    bwd_grad_zero_holes,
    bwd_postprocess,
    bwd_postprocess_rowmajor,
)
from .ffa_bwd_preprocess import bwd_preprocess
from .ffa_bwd_sm80 import FFABwdSm80
from .ffa_bwd_sm90 import FFABwdSm90
from .ffa_bwd_sm100 import FFABwdSm100
from .ffa_bwd_sm120 import FFABwdSm120
from .ffa_fwd_postprocess import fwd_postprocess
from .ffa_fwd_sm80 import FFAFwdSm80
from .ffa_fwd_sm90 import FFAFwdSm90
from .ffa_fwd_sm100 import FFAFwdSm100, fwd_atomic_can_borrow_kv_smem
from .ffa_fwd_sm120 import FFAFwdSm120
from .ffa_utils import (
    MT_MAP,
    MaskMode,
    TorchFlexAttnArgs,
    convert_from_dlpack_leading_static,
    create_softcap_scoremod,
    create_softcap_scoremod_bwd,
    get_device_arch,
    hash_callable,
    is_ffa_2cta_disabled,
    maybe_contiguous,
    normalize_mask_type_spec,
    ranges_to_cu_seqlens,
    tile_size_bwd_sm90,
    tile_size_fwd_sm90,
    validate_arch,
    validate_head_dims,
    validate_per_range_mask_feature_support,
    validate_tensor,
    validate_true_ranges,
)
from .range_merge import RangeMergePlan, bwd_range_merge_arg, merge_qk_ranges
from .sparse_utils import (
    block_sparse_call_tuple,
    get_sparse_q_block_size,
    prepare_block_sparse_bwd,
    prepare_block_sparse_fwd,
    to_cute_block_sparse_tensors,
)

# Pool for the async k_ranges snapshot: a fresh pin_memory is a
# cudaHostAlloc on the hot path.
_pinned_ranges_pool: dict[int, list[tuple[torch.Tensor, torch.cuda.Event]]] = {}


def _acquire_pinned_ranges_snapshot(
    num_ranges: int,
) -> tuple[torch.Tensor, torch.cuda.Event]:
    cap = max(64, 1 << (num_ranges - 1).bit_length())
    bucket = _pinned_ranges_pool.get(cap)
    if bucket:
        return bucket.pop()
    return (
        torch.empty((cap, 2), dtype=torch.int32, pin_memory=True),
        torch.cuda.Event(),
    )


def _release_pinned_ranges_snapshot(
    buf: torch.Tensor, event: torch.cuda.Event
) -> None:
    _pinned_ranges_pool.setdefault(buf.shape[0], []).append((buf, event))


def _validate_dense_dqacc_flag(
    flag: bool,
    q: torch.Tensor,
    q_ranges: torch.Tensor | None,
    k_ranges: torch.Tensor | None,
    major_arch: int,
    deterministic: bool,
    range_merge_active: bool,
    disable_fwd_atomic_reduction: bool,
) -> None:
    if not flag:
        return
    name = "use_dense_dqacc_for_ranges"
    if q_ranges is None or k_ranges is None:
        raise ValueError(f"{name} requires q/k ranges")
    if major_arch not in (10, 11):
        raise NotImplementedError(f"{name} is supported only on SM100/SM110")
    if deterministic:
        raise ValueError(f"{name} requires deterministic=False")
    if range_merge_active:
        raise ValueError(f"{name} is incompatible with range_merge")
    if not disable_fwd_atomic_reduction:
        raise ValueError(f"{name} requires disable_fwd_atomic_reduction=True")
    if q.shape[-1] % 32 != 0:
        raise NotImplementedError(
            f"{name} currently requires head_dim divisible by 32, "
            f"got {q.shape[-1]}"
        )
    if magiattn_cutedsl.is_ffa_debug_mode_enabled():
        sorted_disjoint = bool((q_ranges[1:, 0] >= q_ranges[:-1, 1]).all().item())
        if not sorted_disjoint:
            raise ValueError(
                f"{name} requires sorted, pairwise-disjoint q_ranges "
                "(r[i+1].start >= r[i].end)"
            )


def _flex_flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    q_ranges: torch.Tensor | None = None,
    k_ranges: torch.Tensor | None = None,
    mask_type: int = MT_MAP.full,
    mask_mode: MaskMode = MaskMode.STATIC_FULL,
    mask_types_tensor: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    softcap: float | None = None,
    sink: torch.Tensor | None = None,
    sink_layout: AttnSinkLayout = "sh",
    pack_gqa: bool | None = None,
    flex_attn_args: TorchFlexAttnArgs | None = None,
    disable_fwd_atomic_reduction: bool = False,
    range_merge: "bool | RangeMergePlan" = False,
    clc_scheduler: bool = False,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlexFlashAttention.

    Args:
        ...
        q_ranges/k_ranges: ``[R, 2]`` int32 cuda tensors of [start, end) q/k
            ranges (relation IR). Validated by :func:`validate_true_ranges`.
        mask_type: static ``MT_MAP`` int used by kernel constructors that still
            specialize on a single mask type.
        mask_mode: host-side :class:`MaskMode` used in the compile key. Static
            Full/Causal keep today's specialized kernels; ``PER_RANGE``
            selects the runtime per-range kernel.
        mask_types_tensor: optional CUDA ``int32[R]`` tensor for per-range
            mask selection. Must be ``None`` for static modes.
        flex_attn_args: optional torch FlexAttention-style / block-sparse args
            (``score_mod`` / ``mask_mod`` / ``aux_tensors`` /
            ``block_sparse_tensors``). See :class:`TorchFlexAttnArgs`.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.

    Returns:
        A tuple of (output, lse) where:
        - output is the result of the attention operation, with shape (batch_size, seqlen_q, num_head, head_dim_v) or
          (total_q, num_head, head_dim_v) if q_ranges is provided.
        - lse is the log-sum-exp of the attention scores, with shape (batch_size, num_head, seqlen_q) or
          (num_head, total_q) if q_ranges is provided.
    """
    arch, major_arch = get_device_arch()
    validate_arch(arch, major_arch)

    assert (
        sink_layout == "sh"
    ), f"only sink_layout='sh' is supported, got {sink_layout!r}"
    if mask_mode == MaskMode.PER_RANGE:
        assert mask_types_tensor is not None, "PER_RANGE requires mask_types_tensor"
    else:
        assert (
            mask_types_tensor is None
        ), "static mask modes must not pass mask_types_tensor"

    # Unpack the torch FlexAttention-style / block-sparse args (fwd uses these).
    flex_attn_args = flex_attn_args or TorchFlexAttnArgs()
    score_mod = flex_attn_args.score_mod
    mask_mod = flex_attn_args.mask_mod
    aux_tensors = flex_attn_args.aux_tensors
    block_sparse_tensors = flex_attn_args.block_sparse_tensors

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    has_ranges = q_ranges is not None or k_ranges is not None
    if has_ranges:
        validate_true_ranges(q_ranges, k_ranges, mask_types=mask_types_tensor)
    range_merge_active = bool(range_merge) and has_ranges
    cu_batches = None
    if range_merge_active:
        assert major_arch in (10, 11), "RangeMerge is SM100/SM110 only"
        assert disable_fwd_atomic_reduction, (
            "RangeMerge requires non-atomic forward "
            "(merged Q intervals must be pairwise disjoint)"
        )
        assert (
            score_mod is None and mask_mod is None
        ), "RangeMerge does not compose with score/mask mods"
        assert flex_attn_args.block_sparse_tensors is None
        if isinstance(range_merge, RangeMergePlan):
            plan = range_merge
            assert (
                plan.merged_outer_ranges.shape[0] == q_ranges.shape[0]
            ), "RangeMergePlan row count disagrees with q_ranges"
            q_ranges = plan.merged_outer_ranges
            k_ranges = plan.sorted_inner_ranges
            mask_types_tensor = plan.sorted_mask_types
            cu_batches = plan.cu_batches
            mask_mode = MaskMode.PER_RANGE
        else:
            if mask_mode != MaskMode.PER_RANGE:
                mask_types_tensor = torch.full(
                    (q_ranges.shape[0],),
                    mask_type,
                    dtype=torch.int32,
                    device=q_ranges.device,
                )
                mask_mode = MaskMode.PER_RANGE
            (
                q_ranges,
                _sorted_q_ranges,
                k_ranges,
                mask_types_tensor,
                cu_batches,
            ) = merge_qk_ranges(q_ranges, k_ranges, mask_types_tensor)
        pack_gqa = False
    # SM100/SM110 kernels consume mQRanges/mKRanges directly. Other arches
    # still read cu_seqlens.
    if has_ranges and (
        major_arch not in (10, 11) or flex_attn_args.block_sparse_tensors is not None
    ):
        cu_seqlens_q = ranges_to_cu_seqlens(q_ranges)
        cu_seqlens_k = ranges_to_cu_seqlens(k_ranges)
    else:
        cu_seqlens_q = cu_seqlens_k = None
    if not has_ranges:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        num_ranges = q_ranges.shape[0]
        batch_size = num_ranges
        seqlen_q = None
        total_q = q.shape[0]
    seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if not has_ranges:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
            assert t.stride(0) == 1, "cu_seqlens_q, cu_seqlens_k must be contiguous"
    if sink is not None:
        assert sink.shape == (num_head,)
        assert sink.dtype == torch.bfloat16, "sink must be bfloat16"

    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if major_arch not in [8, 12]:
        validate_head_dims(head_dim, head_dim_v, major_arch, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    # The SM80 fwd kernel indexes mQ by query head directly; the packed-GQA
    # epilogue path is unsupported, so force unpacked store.
    if major_arch == 8:
        pack_gqa = False

    # The atomic merge is a ranges-overlap path; a no-op for dense.
    if not has_ranges:
        disable_fwd_atomic_reduction = True
    if out_dtype is not None:
        assert (
            not disable_fwd_atomic_reduction
        ), "out_dtype only applies to the atomic fwd path"
        assert out_dtype in (torch.float16, torch.bfloat16, torch.float32)
    if not disable_fwd_atomic_reduction:
        assert has_ranges and major_arch in (
            10,
        ), "fwd atomic reduction is the SM100/SM103 true-range overlap path"
        # The atomic epilogue reads/writes prev-O by unpacked row.
        pack_gqa = False
    # Overlapping relations would re-add the sink per merge; fold it once in
    # the fwd postprocess instead.
    kernel_sink = sink if disable_fwd_atomic_reduction else None

    out_torch_dtype = (
        q.dtype
        if disable_fwd_atomic_reduction
        else (out_dtype if out_dtype is not None else q.dtype)
    )
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if not has_ranges else (total_q,)
    lse_shape = (
        (batch_size, num_head, seqlen_q) if not has_ranges else (num_head, total_q)
    )

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=out_torch_dtype,
            device=device,
        )
    else:
        validate_tensor(
            out,
            "out",
            (*q_batch_seqlen_shape, num_head, head_dim_v),
            out_torch_dtype,
            device,
        )

    if lse is None:
        # Atomic merge detects never-written rows via LSE=-inf.
        if disable_fwd_atomic_reduction:
            lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
        else:
            lse = torch.full(
                lse_shape, float("-inf"), dtype=torch.float32, device=device
            )
    else:
        validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if seqlen_k == 0 or total_q == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        return out, lse

    dtype = to_cute_dtype(q.dtype)
    use_block_sparsity = block_sparse_tensors is not None

    local = False
    # Static modes collapse to a causal boolean for host-side heuristics
    # and for kernels that still take is_causal. Per-range mode is treated as
    # may-be-causal for tiling / 2CTA / CLC decisions.
    #
    # DEVIATION: per-range forces causal-compatible host heuristics
    # Reason: one mixed compile entry must reserve causal trip/register budget
    # Recovery: static Full/Causal paths keep their original specialization
    use_per_range_mask = mask_mode == MaskMode.PER_RANGE
    causal = mask_type == MT_MAP.causal or use_per_range_mask
    if mask_mod is not None:
        assert not use_per_range_mask, "per-range mask cannot combine with mask_mod"
        causal = False
        mask_type = MT_MAP.full
        mask_mode = MaskMode.STATIC_FULL

    requested_use_clc_scheduler = clc_scheduler
    requested_disable_2cta = is_ffa_2cta_disabled(is_fwd=True)
    if use_per_range_mask:
        # DEVIATION: force-disable 2CTA/CLC for per-range masks
        # Reason: the per-range kernel supports only 1CTA + static scheduler
        # Recovery: none; static Full/Causal paths keep 2CTA/CLC
        requested_disable_2cta = True
        requested_use_clc_scheduler = False

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # default
    tile_m, tile_n = 128, 128
    mma_pv_is_rs = True
    intra_wg_overlap = True
    match major_arch:
        case 12:
            # SM120 tile sizes tuned for 99 KB SMEM capacity:
            # D<=64:  128x128 → 48 KB (good occupancy)
            # D>64:   128x64  → 64 KB (128x128 would use 96 KB, hurting occupancy)
            if head_dim <= 64:
                tile_m, tile_n = 128, 128
            else:
                tile_m, tile_n = 128, 64
        case 8:
            tile_m, tile_n = 128, 64  # SM80, should tune
        case 9:
            sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
            fwd_cfg = tile_size_fwd_sm90(
                head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q
            )
            tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
            mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
            intra_wg_overlap = fwd_cfg.intra_wg_overlap

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if not has_ranges else total_q
    if has_ranges and magiattn_cutedsl.is_ffa_debug_mode_enabled():
        # The fwd per-range launch bound is sized from max_seqlen_q; an
        # underestimate silently drops relations (out partial, lse=-inf).
        _q_max = int((q_ranges[:, 1] - q_ranges[:, 0]).max().item())
        assert (
            max_seqlen_q >= _q_max
        ), f"max_seqlen_q={max_seqlen_q} < longest q range {_q_max}"
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if major_arch == 10:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1
    fwd_atomic_borrow_kv = False
    if not disable_fwd_atomic_reduction:
        # fp32 sO doubles smem at q_stage=2: borrow KV ring slots, else fall
        # back to single-stage Q. dtype-O has no smem premium.
        fwd_atomic_borrow_kv = (
            major_arch in (10, 11)
            and out_torch_dtype is torch.float32
            and fwd_atomic_can_borrow_kv_smem(
                int(math.ceil(head_dim / 16) * 16),
                int(math.ceil(head_dim_v / 16) * 16),
                tile_m,
                tile_n,
                1,
                q_stage,
            )
        )
        if not fwd_atomic_borrow_kv and out_torch_dtype is torch.float32:
            q_stage = 1

    use_2cta_instrs = (
        major_arch in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not use_block_sparsity
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
        and (
            not has_ranges
            or (
                disable_fwd_atomic_reduction
                and not use_per_range_mask
                and not pack_gqa
            )
        )
    )

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = create_softcap_scoremod(softcap)
    elif score_mod is not None:
        if major_arch == 8:
            raise NotImplementedError(
                "Custom user-provided score_mod is not supported on SM8x architectures."
            )

    # hash score and mask mods for compile cache
    score_mod_hash = hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = has_ranges

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen shapes
    # keep more K/V blocks in flight and hurt L2; dense noncausal mostly just
    # pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = (
        requested_use_clc_scheduler
        and not is_varlen_mha
        and not is_dense_noncausal
        and not fwd_atomic_borrow_kv
    )
    persistent_launch = (
        not causal
        and not local
        and not use_clc_scheduler
        and not fwd_atomic_borrow_kv
    )

    # Prepare block sparse for forward
    (
        normalized_block_sparse_tensors,
        block_sparse_broadcast_pattern,
        q_subtile_factor,
        pack_gqa,
    ) = prepare_block_sparse_fwd(
        block_sparse_tensors,
        pack_gqa=pack_gqa,
        cu_seqlens_q=cu_seqlens_q,
        batch_size=batch_size,
        num_head=num_head,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
        q_stage=q_stage,
    )

    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None

    range_locks = None
    if not disable_fwd_atomic_reduction:
        # One int32 lock per (physical Q block, head); +1 guard tile.
        num_lock_blocks = (total_q + tile_m - 1) // tile_m + 1
        range_locks = torch.zeros(
            num_lock_blocks, num_head, dtype=torch.int32, device=device
        )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        mask_mode,
        disable_fwd_atomic_reduction,
        out_torch_dtype,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        q_ranges is None,
        k_ranges is None,
        kernel_sink is not None,
        block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
        block_sparse_tensors is None
        or block_sparse_tensors.cu_block_idx_offsets is None,
        tile_m,
        tile_n,
        q_stage,
        pack_gqa,
        arch,
        use_2cta_instrs,
        q_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_clc_scheduler,
        persistent_launch,
        range_merge_active,
        magiattn_cutedsl.is_ffa_debug_mode_enabled(),
    )

    if major_arch in (10, 11):
        kernel_varlen_q_meta, kernel_varlen_k_meta = q_ranges, k_ranges
    else:
        kernel_varlen_q_meta, kernel_varlen_k_meta = cu_seqlens_q, cu_seqlens_k

    if compile_key not in _flex_flash_attn_fwd.compile_cache:
        (
            varlen_q_meta_tensor,
            varlen_k_meta_tensor,
            sink_tensor,
        ) = [
            (
                to_cute_tensor(t, assumed_align=4, leading_dim=t.ndim - 1)
                if t is not None
                else None
            )
            for t in (kernel_varlen_q_meta, kernel_varlen_k_meta, kernel_sink)
        ]
        mask_types_cute_tensor = (
            to_cute_tensor(mask_types_tensor, assumed_align=4, leading_dim=0)
            if mask_types_tensor is not None
            else None
        )
        seqused_q_tensor = seqused_k_tensor = None
        page_table_tensor = None
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        if lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = (
            to_cute_block_sparse_tensors(normalized_block_sparse_tensors)
            if normalized_block_sparse_tensors is not None
            else None
        )

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        match major_arch:
            case 8:
                ffa_fwd_obj = FFAFwdSm80(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    mask_type=mask_type,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    num_stages=1,
                    num_threads=128,
                    Q_in_regs=False,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
                )
            case 9:
                ffa_fwd_obj = FFAFwdSm90(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    mask_type=mask_type,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    num_stages=2,
                    Q_in_regs=False,
                    intra_wg_overlap=intra_wg_overlap,
                    mma_pv_is_rs=mma_pv_is_rs,
                    mask_mod=mask_mod,
                    score_mod=score_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    paged_kv_non_tma=False,
                    debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
                )
            case 10 | 11:
                ffa_fwd_obj = FFAFwdSm100(
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    mask_type=mask_type,
                    is_local=local,
                    is_split_kv=False,
                    pack_gqa=pack_gqa,
                    m_block_size=tile_m,
                    n_block_size=tile_n,
                    q_stage=q_stage,
                    # Ranges keep persistence too: static grid-stride walks
                    # tiles in the same order as a full launch.
                    is_persistent=persistent_launch,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=False,
                    is_varlen_q=has_ranges,
                    q_subtile_factor=q_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=use_clc_scheduler,
                    use_per_range_mask=use_per_range_mask,
                    range_merge=range_merge_active,
                    disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
                    o_dtype=to_cute_dtype(out_torch_dtype),
                    debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
                )
            case 12:
                # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity
                assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
                ffa_fwd_obj = FFAFwdSm120(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    mask_type=mask_type,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    num_stages=1,
                    num_threads=128,
                    Q_in_regs=False,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
                )
            case _:
                raise ValueError(
                    f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
                )
        compile_args = [
            ffa_fwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            varlen_q_meta_tensor,
            varlen_k_meta_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            None,  # window_size_left
            None,  # window_size_right
            sink_tensor,
        ]
        if major_arch in [10, 11]:
            compile_args.append(None)
            compile_args.append(mask_types_cute_tensor)
            compile_args.append(
                to_cute_tensor(range_locks, assumed_align=4)
                if range_locks is not None
                else None
            )
            compile_args.append(Int32(max_seqlen_q))
            compile_args.append(
                to_cute_tensor(cu_batches, assumed_align=4, leading_dim=0)
                if cu_batches is not None
                else None
            )
        compile_args.extend(
            [
                sparse_tensors,
                cute_aux_tensors,
            ]
        )
        compile_args.append(current_stream)

        _flex_flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            *compile_args, options="--enable-tvm-ffi"
        )

    q_call, k_call, v_call = q.detach(), k.detach(), v.detach()
    call_args = [
        q_call,
        k_call,
        v_call,
        out.detach(),
        lse,
        softmax_scale,
        kernel_varlen_q_meta,
        kernel_varlen_k_meta,
        None,  # seqlen_used_q
        None,  # seqlen_used_k
        None,  # page_table
        None,  # window_size_left
        None,  # window_size_right
        kernel_sink,
    ]
    if major_arch in [10, 11]:
        # FP8 descale tensors removed; SM100 kernel descale slot is always None.
        call_args.append(None)
        call_args.append(mask_types_tensor)
        call_args.append(range_locks)
        call_args.append(max_seqlen_q)
        call_args.append(cu_batches)
    call_args.extend(
        [
            block_sparse_call_tuple(normalized_block_sparse_tensors),
            aux_tensors,
        ]
    )

    _flex_flash_attn_fwd.compile_cache[compile_key](*call_args)

    if not disable_fwd_atomic_reduction:
        fwd_postprocess(out, lse, sink)

    return out, lse


_flex_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


def _flex_flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    dq: torch.Tensor | None = None,
    dk: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    q_ranges: torch.Tensor | None = None,
    k_ranges: torch.Tensor | None = None,
    mask_type: int = MT_MAP.full,
    mask_mode: MaskMode = MaskMode.STATIC_FULL,
    mask_types_tensor: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    # Exact k-tile totals (cluster-1, cluster-2) for the undeclared ranges
    # bwd grid; None keeps the quota grid.
    k_tile_hints: tuple[int, int] | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sink: torch.Tensor | None = None,
    sink_layout: AttnSinkLayout = "sh",
    pack_gqa: bool = False,
    deterministic: bool = False,
    disable_fwd_atomic_reduction: bool = False,
    disable_bwd_dkv_atomic_reduction: bool = False,
    flex_attn_args: TorchFlexAttnArgs | None = None,
    range_merge: "bool | RangeMergePlan" = False,
    declared_q_full_coverage: bool = False,
    declared_k_full_coverage: bool = False,
    use_dense_dqacc_for_ranges: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for FlexFlashAttention.

    Args:
        declared_q_full_coverage: caller declaration that the union of
            q_ranges covers the whole Q token space, so no dQ coverage holes
            exist and the dQ hole-zeroing sweep is skipped. Trust-based: never
            validated against the geometry (that would sync).
        declared_k_full_coverage: same as above but for k_ranges / the dK/dV
            hole-zeroing sweep. Declared independently of the Q flag because
            the dQ and dK/dV sweeps read different ranges (q_ranges vs the
            merged k_ranges), so one may be full-coverage while the other has
            holes.
        use_dense_dqacc_for_ranges: see :func:`flex_flash_attn_func`.

    Returns:
        A tuple of (dQ, dK, dV) gradients with the same shapes and dtypes as the input q, k, v tensors.
    """
    arch, major_arch = get_device_arch()
    validate_arch(arch, major_arch)

    assert (
        sink_layout == "sh"
    ), f"only sink_layout='sh' is supported, got {sink_layout!r}"
    if mask_mode == MaskMode.PER_RANGE:
        assert mask_types_tensor is not None, "PER_RANGE requires mask_types_tensor"
    else:
        assert (
            mask_types_tensor is None
        ), "static mask modes must not pass mask_types_tensor"

    has_ranges = q_ranges is not None or k_ranges is not None
    if has_ranges:
        validate_true_ranges(q_ranges, k_ranges, mask_types=mask_types_tensor)
        if deterministic:
            raise NotImplementedError(
                "deterministic backward with q/k ranges is unsupported"
            )
    range_merge_active = bool(range_merge) and has_ranges
    _validate_dense_dqacc_flag(
        use_dense_dqacc_for_ranges,
        q,
        q_ranges,
        k_ranges,
        major_arch,
        deterministic,
        range_merge_active,
        disable_fwd_atomic_reduction,
    )
    cu_batches = None
    if range_merge_active:
        assert major_arch in (10, 11), "bwd RangeMerge is SM100/SM110 only"
        assert disable_bwd_dkv_atomic_reduction, (
            "bwd RangeMerge requires disable_bwd_dkv_atomic_reduction "
            "(merged K intervals pairwise disjoint)"
        )
        assert not deterministic
        assert flex_attn_args is None or (
            flex_attn_args.score_mod is None
            and flex_attn_args.score_mod_bwd is None
            and flex_attn_args.mask_mod is None
            and flex_attn_args.block_sparse_tensors_bwd is None
        )
        if isinstance(range_merge, RangeMergePlan):
            plan = range_merge
            assert plan.merged_outer_ranges.shape[0] == k_ranges.shape[0]
            k_ranges = plan.merged_outer_ranges
            q_ranges = plan.sorted_inner_ranges
            mask_types_tensor = plan.sorted_mask_types
            cu_batches = plan.cu_batches
            mask_mode = MaskMode.PER_RANGE
        else:
            if mask_mode != MaskMode.PER_RANGE:
                mask_types_tensor = torch.full(
                    (q_ranges.shape[0],),
                    mask_type,
                    dtype=torch.int32,
                    device=q_ranges.device,
                )
                mask_mode = MaskMode.PER_RANGE
            (
                k_ranges,
                _sorted_k_ranges,
                q_ranges,
                mask_types_tensor,
                cu_batches,
            ) = merge_qk_ranges(k_ranges, q_ranges, mask_types_tensor)
    if has_ranges and major_arch not in (10, 11):
        cu_seqlens_q = ranges_to_cu_seqlens(q_ranges)
        cu_seqlens_k = ranges_to_cu_seqlens(k_ranges)
    else:
        cu_seqlens_q = cu_seqlens_k = None

    # Unpack the torch FlexAttention-style / block-sparse args (bwd uses these;
    # note block sparsity reads the bwd-specific tensors).
    flex_attn_args = flex_attn_args or TorchFlexAttnArgs()
    score_mod = flex_attn_args.score_mod
    score_mod_bwd = flex_attn_args.score_mod_bwd
    mask_mod = flex_attn_args.mask_mod
    aux_tensors = flex_attn_args.aux_tensors
    block_sparse_tensors = flex_attn_args.block_sparse_tensors_bwd

    local = False
    use_per_range_mask = mask_mode == MaskMode.PER_RANGE
    # DEVIATION: per-range forces causal-compatible host heuristics
    # Reason: one mixed compile entry must reserve causal trip/register budget
    # Recovery: static Full/Causal paths keep their original specialization
    causal = mask_type == MT_MAP.causal or use_per_range_mask
    sparse_q = None
    if block_sparse_tensors is not None and major_arch == 9:
        sparse_q = (
            block_sparse_tensors.block_size[0]
            if block_sparse_tensors.block_size is not None
            else 128
        )

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    match major_arch:
        case 8:
            # SM80 (Ampere): uses the dedicated FFABwdSm80 kernel (SM80 MMA, 256
            # threads / 8 warps). Its tiled-MMA expects AtomLayout 1/8/1 with
            # n_block_size == permutation_M (n_block_size = AtomLayoutNdKV * 16 = 128).
            m_block_size = 64
            n_block_size = 128
            if head_dim <= 64:
                num_stages_Q = 2
                num_stages_dO = 2
            else:
                num_stages_Q = 1
                num_stages_dO = 1
            SdP_swapAB = False
            dKV_swapAB = False
            dQ_swapAB = False
            AtomLayoutMSdP = 1
            AtomLayoutNdKV = 8
            # The dQ MMA tiles the head_dim (N) across (num_warps // AtomLayoutMdQ)
            # warp-columns, each contributing 16 elements. With 8 warps, AtomLayoutMdQ=1
            # needs head_dim >= 128; for head_dim <= 64 that overshoots the output tile
            # and the dQ gemm fails to verify. Use AtomLayoutMdQ=2 (4 warp-columns ->
            # 64-wide N) for head_dim <= 64, matching the SM90 config.
            AtomLayoutMdQ = 2 if head_dim <= 64 else 1
            V_in_regs = False
            cluster_size = 1
            use_2cta_instrs = False
            num_threads = 256
            dQ_single_wg = False
            assert not (
                block_sparse_tensors is not None
            ), "Block sparsity backward not supported on SM 8.0"
            assert (
                score_mod is None and score_mod_bwd is None
            ), "score_mod backward not supported on SM 8.0"
            assert mask_mod is None, "mask_mod backward not supported on SM 8.0"
            assert (
                deterministic is False
            ), "deterministic backward not supported on SM 8.0"
        case 12:
            # SM120: uses SM80 MMA with 99 KB SMEM, 128 threads (4 warps).
            m_block_size = 64
            n_block_size = 64
            if head_dim <= 64:
                num_stages_Q = 2
                num_stages_dO = 2
            else:
                num_stages_Q = 1
                num_stages_dO = 1
            SdP_swapAB = False
            dKV_swapAB = False
            dQ_swapAB = False
            AtomLayoutMSdP = 4
            AtomLayoutNdKV = 4
            AtomLayoutMdQ = 4
            V_in_regs = False
            cluster_size = 1
            use_2cta_instrs = False
            num_threads = 128
            dQ_single_wg = False
            assert not (
                block_sparse_tensors is not None
            ), "Block sparsity backward not supported on SM 12.0"
            assert (
                score_mod is None and score_mod_bwd is None
            ), "score_mod backward not supported on SM 12.0"
            assert mask_mod is None, "mask_mod backward not supported on SM 12.0"
            assert (
                deterministic is False
            ), "deterministic backward not supported on SM 12.0"
        case 9:
            cfg = tile_size_bwd_sm90(
                head_dim,
                head_dim_v,
                causal,
                local,
                sparse_block_size_q=sparse_q,
            )
            m_block_size = cfg.m_block_size
            n_block_size = cfg.n_block_size
            num_stages_Q = cfg.num_stages_Q
            num_stages_dO = cfg.num_stages_dO
            num_stages_PdS = cfg.num_stages_PdS
            SdP_swapAB = cfg.SdP_swapAB
            dKV_swapAB = cfg.dKV_swapAB
            dQ_swapAB = cfg.dQ_swapAB
            AtomLayoutMSdP = cfg.AtomLayoutMSdP
            AtomLayoutNdKV = cfg.AtomLayoutNdKV
            AtomLayoutMdQ = cfg.AtomLayoutMdQ
            V_in_regs = False
            num_threads = (cfg.num_wg + 1) * 128
            dQ_single_wg = cfg.dQ_single_wg
            cluster_size = 1
            use_2cta_instrs = False
        case _:
            m_block_size = 128
            n_block_size = 128
            dQ_swapAB = False
            dKV_swapAB = False
            AtomLayoutMdQ = 1
            AtomLayoutNdKV = 1
            requested_disable_2cta = is_ffa_2cta_disabled()
            # RangeMerge consumes per-pair mask constants (2-CTA-compatible);
            # genuine runtime per-range masks still need 1-CTA.
            per_range_requires_1cta = use_per_range_mask and not range_merge_active
            # 2-CTA hdim-192 fuses Q/Qt on one pipeline; the RangeMerge
            # per-pair Q walk needs them separate.
            merge_192_requires_1cta = range_merge_active and head_dim == 192
            disable_2cta = (
                requested_disable_2cta
                or score_mod is not None
                or score_mod_bwd is not None
                or mask_mod is not None
                or block_sparse_tensors is not None
                or per_range_requires_1cta
                or merge_192_requires_1cta
            )
            cluster_size = (
                1
                if per_range_requires_1cta or merge_192_requires_1cta
                else (2 if head_dim >= 128 and not disable_2cta else 1)
            )
            use_2cta_instrs = cluster_size == 2

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k = [
        maybe_contiguous(t)
        for t in (
            q,
            k,
            v,
            out,
            dout,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
        )
    ]
    if not has_ranges:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        num_ranges = q_ranges.shape[0]
        batch_size = num_ranges
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q
        total_k = k.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]

    use_block_sparsity = block_sparse_tensors is not None
    subtile_factor = sparse_q // m_block_size if sparse_q is not None else 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    if not has_ranges:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)

    if has_ranges:
        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (
            num_head,
            total_q,
        ), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (
            batch_size,
            num_head,
            seqlen_q,
        ), "lse must have shape (batch_size, num_head, seqlen_q)"

    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert (
        q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    ), "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if major_arch not in [8, 12]:
        validate_head_dims(head_dim, head_dim_v, major_arch, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1  # type: ignore[unreachable]
    # pack_gqa backward not yet supported in bwd
    pack_gqa = False
    if disable_bwd_dkv_atomic_reduction and has_ranges and major_arch in (10, 11):
        assert qhead_per_kvhead == 1, (
            "disable_bwd_dkv_atomic_reduction requires MHA "
            "(unique dK/dV writer per KV head)"
        )

    if softcap != 0.0:
        assert (
            score_mod is None and score_mod_bwd is None
        ), "softcap and score_mod/score_mod_bwd cannot be used together"
        score_mod = create_softcap_scoremod(softcap)
        score_mod_bwd = create_softcap_scoremod_bwd(softcap)
    if score_mod is not None:
        assert (
            score_mod_bwd is not None
        ), "score_mod_bwd is required when score_mod is provided"
        assert not has_ranges, "varlen + score_mod not supported in bwd yet"
        if major_arch == 8:
            raise NotImplementedError(
                "Custom user-provided score_mod is not supported on SM8x architectures."
            )

    device = q.device
    out_torch_dtype = q.dtype

    # Each flag certifies sorted, pairwise-disjoint k projections.
    k_ranges_sorted_disjoint = has_ranges and (
        declared_k_full_coverage or disable_bwd_dkv_atomic_reduction
    )
    # Undeclared ranges: exact tile count (async snapshot from fwd) sizes the
    # prefix-sum grid; safe for overlapping k ranges.
    k_total_blocks_hint: int | None = None
    if (
        k_tile_hints is not None
        and has_ranges
        and not k_ranges_sorted_disjoint
        and cu_batches is None
        and major_arch in (10, 11)
        and n_block_size == 128
        and cluster_size in (1, 2)
    ):
        exact_ctas = k_tile_hints[0] if cluster_size == 1 else 2 * k_tile_hints[1]
        quota_blocks = (seqlen_k + n_block_size - 1) // n_block_size
        quota_units = (quota_blocks + cluster_size - 1) // cluster_size
        quota_ctas = quota_units * cluster_size * k_ranges.shape[0]
        # The per-CTA prefix-sum decode only pays off above ~2x quota waste.
        if quota_ctas >= 2 * exact_ctas:
            k_total_blocks_hint = exact_ctas

    # Buffer policy: empty_like where every row is provably written — dense,
    # the unique-writer contracts (coverage holes blanked on device below),
    # or a full-coverage declaration (no holes) — else zeros_like.
    direct_dkv = (
        disable_bwd_dkv_atomic_reduction and has_ranges and major_arch in (10, 11)
    )
    direct_dq_init = (
        disable_fwd_atomic_reduction and has_ranges and major_arch in (10, 11)
    )
    dkv_empty = not has_ranges or direct_dkv or declared_k_full_coverage
    dq_empty = not has_ranges or direct_dq_init or declared_q_full_coverage
    dkv_alloc = torch.empty_like if dkv_empty else torch.zeros_like
    dq_alloc = torch.empty_like if dq_empty else torch.zeros_like

    # DEVIATION: caller-provided dk/dv keep their own content in rows no
    # range covers
    # Reason: hole-zeroing only runs for self-allocated gradient buffers
    # Recovery: callers passing their own buffers must pre-clear them or accept stale holes
    dk_self_alloc = dk is None
    dv_self_alloc = dv is None
    dq_self_alloc = dq is None

    # Q ranges covered by the direct contract may still arrive unsorted; the
    # dQ cleanup therefore uses its ordering-agnostic metadata scan.
    if dq is None:
        dq = dq_alloc(q)
    else:
        validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = dkv_alloc(k)
    else:
        validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = dkv_alloc(v)
    else:
        validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    if direct_dkv and not declared_k_full_coverage:
        if dk_self_alloc:
            bwd_grad_zero_holes(dk, k_ranges)
        if dv_self_alloc:
            bwd_grad_zero_holes(dv, k_ranges)

    rowmajor_accum = has_ranges and major_arch in (10, 11)
    dq_accum_hdim_multiple = (
        16
        if rowmajor_accum and not use_dense_dqacc_for_ranges
        else 32
    )
    dq_head_dim_rounded = (
        (head_dim + dq_accum_hdim_multiple - 1)
        // dq_accum_hdim_multiple
        * dq_accum_hdim_multiple
    )
    dkv_accum_hdim_multiple = 16 if rowmajor_accum else 32
    dkv_head_dim_rounded = (
        (head_dim + dkv_accum_hdim_multiple - 1)
        // dkv_accum_hdim_multiple
        * dkv_accum_hdim_multiple
    )
    if not has_ranges:
        dq_accum = torch.empty(
            batch_size,
            num_head,
            seqlen_q_rounded * dq_head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        if major_arch in (10, 11):
            if use_dense_dqacc_for_ranges:
                # Dense slot per range: aligned capacity = total + num_ranges * tile
                total_q_rounded_padded = (
                    (total_q + m_block_size - 1) // m_block_size + num_ranges
                ) * m_block_size
            else:
                # Row-major accum: total_q aligned + 1 tile TMA descriptor guard
                total_q_rounded_padded = (
                    (total_q + m_block_size - 1) // m_block_size + 1
                ) * m_block_size
        else:
            total_q_rounded_padded = (
                (total_q + (num_ranges + 1) * m_block_size - 1)
                // m_block_size
                * m_block_size
            )
        dq_accum_alloc = torch.empty if direct_dq_init else torch.zeros
        dq_accum = dq_accum_alloc(
            num_head,
            total_q_rounded_padded * dq_head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        # Zeros, not empty: a range's tail tile may read coverage-gap rows,
        # where garbage (e.g. -inf) can turn exp() into NaN.
        dpsum = torch.zeros(
            num_head, total_q_rounded_padded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.zeros(
            num_head, total_q_rounded_padded, dtype=torch.float32, device=device
        )

    # Ranges force accum+postprocess even for MHA: overlapping K ranges
    # would clobber direct stores from multiple CTAs.
    dKV_postprocess = qhead_per_kvhead > 1 or (
        has_ranges and major_arch in (10, 11) and not disable_bwd_dkv_atomic_reduction
    )
    if dKV_postprocess:
        head_dim_v_rounded = (
            (head_dim_v + dkv_accum_hdim_multiple - 1)
            // dkv_accum_hdim_multiple
            * dkv_accum_hdim_multiple
        )
        if not has_ranges:
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * dkv_head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            cluster_tile_n = cluster_size * n_block_size
            if major_arch in (10, 11):
                # Row-major accum: total_k aligned + 1 tile TMA descriptor guard
                total_k_rounded_padded = (
                    (total_k + cluster_tile_n - 1) // cluster_tile_n + 1
                ) * cluster_tile_n
            else:
                # Dense slot per range: aligned capacity = total + (num_ranges + 1) * tile
                total_k_rounded_padded = (
                    (total_k + (num_ranges + 1) * cluster_tile_n - 1)
                    // cluster_tile_n
                    * cluster_tile_n
                )
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * dkv_head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = to_cute_dtype(q.dtype)
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if deterministic:
        dQ_semaphore = torch.zeros(
            batch_size,
            num_head,
            seqlen_q_rounded // m_block_size,
            cluster_size,
            dtype=torch.int32,
            device=device,
        )
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(
            batch_size,
            num_head_kv,
            seqlen_k_rounded // n_block_size,
            2,
            dtype=torch.int32,
            device=device,
        )
        dV_semaphore = torch.zeros(
            batch_size,
            num_head_kv,
            seqlen_k_rounded // n_block_size,
            2,
            dtype=torch.int32,
            device=device,
        )
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1) - dLSE and lse * log2_e.
    # SM100/SM110: pre/main/post all consume the same [R, 2] range metadata;
    # other arches keep cu_seqlens (cu_seqlens_q is None on the SM100 path).
    kernel_varlen_q_meta = q_ranges if major_arch in (10, 11) else cu_seqlens_q
    kernel_varlen_k_meta = k_ranges if major_arch in (10, 11) else cu_seqlens_k
    pre_post_q_ranges = q_ranges if major_arch in (10, 11) else None
    pre_post_k_ranges = k_ranges if major_arch in (10, 11) else None

    bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        # Overlapping Q: skip per-range zero — PDL can race another
        # relation's main-kernel dQ reductions on the same physical row.
        (
            dq_accum
            if direct_dq_init or pre_post_q_ranges is None or deterministic
            else None
        ),
        cu_seqlens_q,
        None,  # seqused_q
        None,  # dlse
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        use_padded_offsets=False,
        q_ranges=pre_post_q_ranges,
        max_seqlen_q=seqlen_q if pre_post_q_ranges is not None else 0,
        range_workspace_padded=use_dense_dqacc_for_ranges,
    )

    # num_threads: SM80 (256) and SM120 (128) are set above, SM90 derives from
    # BwdConfig.num_wg, SM100/SM110 uses default from function signature (384).
    if major_arch not in [8, 9, 12]:
        num_threads = 384

    # Prepare block sparse for backward.
    score_mod_hash = hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [
            to_cute_tensor(buf, assumed_align=None, fully_dynamic=True)
            for buf in aux_tensors
        ]
    (
        normalized_block_sparse_tensors,
        block_sparse_broadcast_pattern,
        spt,
    ) = prepare_block_sparse_bwd(
        block_sparse_tensors,
        deterministic=deterministic,
        causal=causal,
        local=local,
        batch_size=batch_size,
        num_head=num_head,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        subtile_factor=subtile_factor,
    )

    # Backward kernel: compute dk, dv, dq_accum.
    if major_arch in [8, 9, 12]:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            mask_mode,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            dQ_single_wg,
            deterministic,
            q_ranges is None,
            k_ranges is None,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            (seqlen_q_rounded // m_block_size == 1),
            (seqlen_k_rounded // n_block_size == 1),
            magiattn_cutedsl.is_ffa_debug_mode_enabled(),
        )
    else:  # SM100
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            mask_mode,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            use_2cta_instrs,
            deterministic,
            spt,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            q_ranges is None,
            k_ranges is None,
            disable_bwd_dkv_atomic_reduction,
            range_merge_active,
            use_dense_dqacc_for_ranges,
            k_ranges_sorted_disjoint,
            k_total_blocks_hint is not None,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            (seqlen_q_rounded // m_block_size == 1),
            (seqlen_k_rounded // n_block_size == 1),
            magiattn_cutedsl.is_ffa_debug_mode_enabled(),
        )

    if compile_key not in _flex_flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        lse_log2_tensor, dpsum_tensor = [to_cute_tensor(t) for t in (lse_log2, dpsum)]
        dq_accum_tensor = to_cute_tensor(dq_accum) if dq_accum is not None else None
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        varlen_q_meta_tensor, varlen_k_meta_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (kernel_varlen_q_meta, kernel_varlen_k_meta)
        ]
        seqused_q_tensor = seqused_k_tensor = None
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            (
                convert_from_dlpack_leading_static(
                    t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order()
                )
                if t is not None
                else None
            )
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        match major_arch:
            case 8 | 12:
                ffa_bwd_cls = FFABwdSm120 if major_arch == 12 else FFABwdSm80
                ffa_bwd_kwargs = dict(
                    V_in_regs=V_in_regs,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
                )

                ffa_bwd_obj = ffa_bwd_cls(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    m_block_size,
                    n_block_size,
                    num_stages_Q,
                    num_stages_dO,
                    num_threads,
                    pack_gqa,
                    mask_type,
                    SdP_swapAB,
                    dKV_swapAB,
                    dQ_swapAB,
                    AtomLayoutMSdP,
                    AtomLayoutNdKV,
                    AtomLayoutMdQ,
                    **ffa_bwd_kwargs,  # type: ignore[arg-type]
                )
            case 9:
                ffa_bwd_obj = FFABwdSm90(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    mask_type,
                    is_local=local,
                    deterministic=deterministic,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    Q_stage=num_stages_Q,
                    dO_stage=num_stages_dO,
                    PdS_stage=num_stages_PdS,
                    SdP_swapAB=SdP_swapAB,
                    dKV_swapAB=dKV_swapAB,
                    dQ_swapAB=dQ_swapAB,
                    AtomLayoutMSdP=AtomLayoutMSdP,
                    AtomLayoutNdKV=AtomLayoutNdKV,
                    AtomLayoutMdQ=AtomLayoutMdQ,
                    num_threads=num_threads,
                    V_in_regs=V_in_regs,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    subtile_factor=subtile_factor,
                    dQ_single_wg=dQ_single_wg,
                    debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
                )
            case _:
                ffa_bwd_obj = FFABwdSm100(
                    head_dim,
                    head_dim_v,
                    mask_type=mask_type,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    deterministic=deterministic,
                    spt=spt,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    subtile_factor=subtile_factor,
                    use_per_range_mask=use_per_range_mask,
                    disable_bwd_dkv_atomic_reduction=disable_bwd_dkv_atomic_reduction,
                    range_merge=range_merge_active,
                    use_dense_dqacc_for_ranges=use_dense_dqacc_for_ranges,
                    k_ranges_sorted_disjoint=k_ranges_sorted_disjoint,
                    debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
                )

        # Block sparse tensors for backward use Q-direction indexing (transposed from forward).
        sparse_tensors_compile = (
            to_cute_block_sparse_tensors(normalized_block_sparse_tensors)
            if normalized_block_sparse_tensors is not None
            else None
        )
        mask_types_cute_tensor = (
            to_cute_tensor(mask_types_tensor, assumed_align=4, leading_dim=0)
            if mask_types_tensor is not None
            else None
        )

        bwd_compile_args = [
            ffa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            varlen_q_meta_tensor,
            varlen_k_meta_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
        ]
        bwd_compile_args += [
            None,  # window_size_left
            None,  # window_size_right
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
        ]
        if major_arch in [10, 11]:
            bwd_compile_args.append(mask_types_cute_tensor)
            bwd_compile_args.append(
                to_cute_tensor(cu_batches, assumed_align=4, leading_dim=0)
                if cu_batches is not None
                else None
            )
            # Runtime scalar: the compiled variant stays max_seqlen_k-agnostic.
            bwd_compile_args.append(Int32(seqlen_k))
            # Runtime scalar: presence is a compile-key bit, the value is not.
            bwd_compile_args.append(
                Int32(k_total_blocks_hint)
                if k_total_blocks_hint is not None
                else None
            )
        bwd_compile_args.extend(
            [
                cute_aux_tensors,
                sparse_tensors_compile,
                current_stream,
            ]
        )
        _flex_flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            *bwd_compile_args,
            options="--enable-tvm-ffi",
        )
    bwd_call_args = [
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk if not dKV_postprocess else dk_accum,
        dv if not dKV_postprocess else dv_accum,
        softmax_scale,
        kernel_varlen_q_meta,
        kernel_varlen_k_meta,
        None,  # seqlen_used_q
        None,  # seqlen_used_k
    ]
    bwd_call_args += [
        None,  # window_size_left
        None,  # window_size_right
        dQ_semaphore,
        dK_semaphore,
        dV_semaphore,
    ]
    if major_arch in [10, 11]:
        bwd_call_args.append(mask_types_tensor)
        bwd_call_args.append(cu_batches)
        bwd_call_args.append(seqlen_k)
        bwd_call_args.append(k_total_blocks_hint)
    bwd_call_args.extend(
        [
            aux_tensors,
            block_sparse_call_tuple(normalized_block_sparse_tensors),
        ]
    )
    _flex_flash_attn_bwd.compile_cache[compile_key](*bwd_call_args)

    # Postprocess: convert dq_accum from float32 to dq in bf16/fp16
    match major_arch:
        case 9:
            # dQ postprocess: match main kernel's MMA WG count, unless dQ_single_wg
            num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
            num_threads_post_dKV = cfg.num_wg * 128
        case 8:
            # SM80: the dQ/dKV accumulator buffers are written by the main kernel's
            # tiled-MMA, whose accumulator->linear layout depends on the warp (thread)
            # count. The postprocess re-derives that layout from its own tiled-MMA, so
            # it must use the *same* number of threads as the main kernel (256, i.e.
            # 8 warps). Using fewer threads (e.g. 128) reshapes the linear accumulator
            # with a different MMA layout and scrambles the result (was the SM80 dQ bug).
            num_threads_post_dQ = 256
            num_threads_post_dKV = 256
        case _:
            num_threads_post_dQ = 128
            num_threads_post_dKV = 128

    # True-range bwd stores row-major fp32 TMA reductions; the row-major
    # postprocess sweeps only in-range rows.
    rowmajor_post_dq = (
        pre_post_q_ranges is not None
        and not deterministic
        and not use_dense_dqacc_for_ranges
    )
    rowmajor_post_dkv = pre_post_k_ranges is not None and not deterministic
    if rowmajor_post_dq:
        bwd_postprocess_rowmajor(dq_accum, dq, q_ranges, seqlen_q, softmax_scale)
    else:
        bwd_postprocess(
            dq_accum,
            dq,
            softmax_scale,
            cu_seqlens_q,
            None,
            arch,
            dtype,
            head_dim,
            m_block_size,
            num_threads_post_dQ,
            AtomLayoutMdQ,
            dQ_swapAB,
            use_2cta_instrs=use_2cta_instrs,
            cluster_size=1,
            ranges=pre_post_q_ranges,
            range_workspace_padded=use_dense_dqacc_for_ranges,
        )

    if direct_dq_init and dq_self_alloc and not declared_q_full_coverage:
        assert q_ranges is not None
        bwd_grad_zero_holes(dq, q_ranges, ranges_sorted=False)

    if dKV_postprocess and rowmajor_post_dkv:
        bwd_postprocess_rowmajor(dk_accum, dk, k_ranges, seqlen_k, softmax_scale)
        bwd_postprocess_rowmajor(dv_accum, dv, k_ranges, seqlen_k, 1.0)
    elif dKV_postprocess:
        bwd_postprocess(
            dk_accum,
            dk,
            softmax_scale,
            cu_seqlens_k,
            None,
            arch,
            dtype,
            head_dim,
            n_block_size,
            num_threads_post_dKV,
            AtomLayoutNdKV,
            dKV_swapAB,
            cluster_size=cluster_size,
            ranges=pre_post_k_ranges,
        )
        bwd_postprocess(
            dv_accum,
            dv,
            1.0,
            cu_seqlens_k,
            None,
            arch,
            dtype,
            head_dim_v,
            n_block_size,
            num_threads_post_dKV,
            AtomLayoutNdKV,
            dKV_swapAB,
            cluster_size=cluster_size,
            ranges=pre_post_k_ranges,
        )

    return dq, dk, dv


_flex_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


# ---------------------------------------------------------------------------
# FFA autograd function and interface
# ---------------------------------------------------------------------------


class FlexFlashAttnFunc(torch.autograd.Function):
    """Autograd function for FFA (dense / varlen).

    The optional torch FlexAttention-style / block-sparse capabilities
    (``score_mod`` / ``score_mod_bwd`` / ``mask_mod`` / ``aux_tensors`` /
    ``block_sparse_tensors[_bwd]``) are bundled into a single
    :class:`TorchFlexAttnArgs` (``flex_attn_args``) to keep the common
    signature clean.

    NOTE: ``softcap`` is implemented internally via the score_mod machinery
    (see ``_flex_flash_attn_fwd``), and is exposed here as a plain scalar.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_ranges: torch.Tensor | None = None,
        k_ranges: torch.Tensor | None = None,
        mask_types: torch.Tensor | int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
        sink: torch.Tensor | None = None,
        sink_layout: AttnSinkLayout = "sh",
        pack_gqa: bool | None = None,
        deterministic: bool = False,
        flex_attn_args: TorchFlexAttnArgs | None = None,
        disable_fwd_atomic_reduction: bool = False,
        disable_bwd_dkv_atomic_reduction: bool = False,
        range_merge: "bool | RangeMergePlan" = False,
        bwd_q_full_coverage: bool = False,
        bwd_k_full_coverage: bool = False,
        use_dense_dqacc_for_ranges: bool = False,
        out_dtype: torch.dtype | None = None,
    ):
        arch, major_arch = get_device_arch()
        is_varlen = q_ranges is not None or k_ranges is not None
        _validate_dense_dqacc_flag(
            use_dense_dqacc_for_ranges,
            q,
            q_ranges,
            k_ranges,
            major_arch,
            deterministic,
            bool(range_merge),
            disable_fwd_atomic_reduction,
        )
        if (bwd_q_full_coverage or bwd_k_full_coverage) and not is_varlen:
            raise ValueError(
                "bwd_q_full_coverage/bwd_k_full_coverage require "
                "q_ranges/k_ranges"
            )
        if is_varlen:
            num_ranges = validate_true_ranges(q_ranges, k_ranges, mask_types=mask_types)
        else:
            num_ranges = int(q.shape[0])

        mask_spec = normalize_mask_type_spec(
            mask_types,
            num_ranges=num_ranges if is_varlen else None,
            batch_size=None if is_varlen else num_ranges,
            is_varlen=is_varlen,
        )
        flex_attn_args = flex_attn_args or TorchFlexAttnArgs()
        validate_per_range_mask_feature_support(
            mask_spec,
            major_arch=major_arch,
            has_mask_mod=flex_attn_args.mask_mod is not None,
            has_block_sparse=flex_attn_args.block_sparse_tensors is not None,
            has_score_mod=flex_attn_args.score_mod is not None,
            has_softcap=bool(softcap),
        )
        mask_mode = mask_spec.mode
        mask_types_tensor = mask_spec.per_range_mask_types
        if mask_spec.is_per_range:
            # DEVIATION: report causal mask_type for may-be-causal mixed compile
            # Reason: the kernel's is_causal specialization and host heuristics key off mask_type
            # Recovery: runtime mMaskTypes[batch] selects Full vs Causal
            mask_type = MT_MAP.causal
        else:
            assert mask_spec.static_mask_type is not None
            mask_type = int(mask_spec.static_mask_type)

        out, lse = _flex_flash_attn_fwd(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            mask_type=mask_type,
            mask_mode=mask_mode,
            mask_types_tensor=mask_types_tensor,
            sink=sink,
            sink_layout=sink_layout,
            softcap=softcap,
            pack_gqa=pack_gqa,
            flex_attn_args=flex_attn_args,
            disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
            range_merge=range_merge,
            out_dtype=out_dtype,
        )

        aux_tensors = flex_attn_args.aux_tensors if flex_attn_args else None
        # mask_types needs no grad tracking; keep it on ctx directly.
        ctx.mask_mode = mask_mode
        ctx.mask_type = mask_type
        ctx.mask_types_tensor = mask_types_tensor
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            q_ranges,
            k_ranges,
            sink,
            *(aux_tensors or ()),
        )
        ctx.softmax_scale = softmax_scale
        ctx.sink_layout = sink_layout
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.disable_fwd_atomic_reduction = disable_fwd_atomic_reduction
        ctx.disable_bwd_dkv_atomic_reduction = disable_bwd_dkv_atomic_reduction
        # Forward merges Q ranges; backward merges the dual K ranges.
        # Derived from the fwd flag, not a second user flag.
        ctx.bwd_range_merge = bwd_range_merge_arg(range_merge)
        ctx.bwd_q_full_coverage = bwd_q_full_coverage
        ctx.bwd_k_full_coverage = bwd_k_full_coverage
        ctx.use_dense_dqacc_for_ranges = use_dense_dqacc_for_ranges
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        # Async k_ranges snapshot so backward derives the exact tile count
        # without a device sync (stream capture can't replay the event).
        ctx.k_tile_hint_state = None
        # torch.is_grad_enabled() is always off inside autograd.Function.forward.
        if (
            k_ranges is not None
            and any(t.requires_grad for t in (q, k, v))
            and not (ctx.bwd_k_full_coverage or disable_bwd_dkv_atomic_reduction)
            and not torch.cuda.is_current_stream_capturing()
        ):
            pinned, done = _acquire_pinned_ranges_snapshot(k_ranges.shape[0])
            view = pinned[: k_ranges.shape[0]]
            view.copy_(k_ranges, non_blocking=True)
            done.record()
            ctx.k_tile_hint_state = (pinned, view, done)
        # Drop the direct aux_tensors reference on ctx; the real tensors are
        # tracked via save_for_backward and restored in backward. Keeping them
        # here too would bypass autograd's save_for_backward bookkeeping.
        ctx.flex_attn_args = (
            flex_attn_args.drop_aux_tensors() if flex_attn_args is not None else None
        )
        ctx.set_materialize_grads(False)

        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):  # pragma: no cover
        (
            q,
            k,
            v,
            out,
            lse,
            q_ranges,
            k_ranges,
            sink,
            *aux,
        ) = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)
        if out.dtype != q.dtype:
            # Atomic-reduction fwd returns fp32 O; the bwd kernels consume the
            # target dtype.
            out = out.to(q.dtype)
            dout = dout.to(q.dtype)

        # Restore aux_tensors from the saved tail (kept tracked by autograd).
        flex_attn_args: TorchFlexAttnArgs | None = ctx.flex_attn_args
        if flex_attn_args is not None:
            flex_attn_args = flex_attn_args.with_aux_tensors(aux)

        # Resolve the async k_ranges snapshot from forward (host wait, not a
        # device sync); cached on ctx for retain_graph repeated backwards.
        k_tile_hints = getattr(ctx, "k_tile_hints_resolved", None)
        if (
            k_tile_hints is None
            and ctx.k_tile_hint_state is not None
            and not torch.cuda.is_current_stream_capturing()
        ):
            pinned, view, done = ctx.k_tile_hint_state
            ctx.k_tile_hint_state = None
            done.synchronize()
            kblocks = (view[:, 1] - view[:, 0] + 127) // 128
            k_tile_hints = (
                int(kblocks.sum()),
                int(((kblocks + 1) // 2).sum()),
            )
            ctx.k_tile_hints_resolved = k_tile_hints
            _release_pinned_ranges_snapshot(pinned, done)

        dq, dk, dv = _flex_flash_attn_bwd(
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            dout=dout,
            softmax_scale=ctx.softmax_scale,
            mask_type=ctx.mask_type,
            mask_mode=ctx.mask_mode,
            mask_types_tensor=ctx.mask_types_tensor,
            sink=sink,
            sink_layout=ctx.sink_layout,
            softcap=ctx.softcap,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            k_tile_hints=k_tile_hints,
            deterministic=ctx.deterministic,
            disable_fwd_atomic_reduction=ctx.disable_fwd_atomic_reduction,
            disable_bwd_dkv_atomic_reduction=ctx.disable_bwd_dkv_atomic_reduction,
            flex_attn_args=flex_attn_args,
            range_merge=ctx.bwd_range_merge,
            declared_q_full_coverage=ctx.bwd_q_full_coverage,
            declared_k_full_coverage=ctx.bwd_k_full_coverage,
            use_dense_dqacc_for_ranges=ctx.use_dense_dqacc_for_ranges,
        )

        return dq, dk, dv, *((None,) * 33)  # Extra Nones is fine


def flex_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_ranges: torch.Tensor | None = None,
    k_ranges: torch.Tensor | None = None,
    mask_types: torch.Tensor | int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sink: torch.Tensor | None = None,
    sink_layout: AttnSinkLayout = "sh",
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    flex_attn_args: TorchFlexAttnArgs | None = None,
    disable_fwd_atomic_reduction: bool = False,
    disable_bwd_dkv_atomic_reduction: bool = False,
    range_merge: "bool | RangeMergePlan" = False,
    bwd_q_full_coverage: bool = False,
    bwd_k_full_coverage: bool = False,
    use_dense_dqacc_for_ranges: bool = False,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, AttnForwardMeta]:
    """
    Flex-flash-attention interface (dense / range / varlen).

    Explanation of some optional arguments:

    q_ranges/k_ranges: ``[R, 2]`` int32 cuda tensors describing per-range
        [start, end) intervals over the packed (total_seqlen, nheads, headdim)
        q/k layout — the CuteDSL relation IR; under non-overlapping Q/K
        (``disable_*_atomic_reduction=True``) each row has a unique writer.
        When provided, both must be set, q/k/v must be packed, and the device
        must be SM100/SM110. Leave as ``None`` for the dense
        (batch, seqlen, nheads, headdim) path. On SM100/SM110 the pre/main/post
        kernels consume ranges natively; other arches collapse them via
        ``ranges_to_cu_seqlens``.

    bwd_q_full_coverage / bwd_k_full_coverage: caller declarations (default
        ``False``) that the union of q_ranges / k_ranges covers the whole
        Q / K token space, letting backward skip the dQ / dK / dV hole-zeroing
        sweeps. They apply to every ranges backward path, including
        range-merge. Trust-based like the other contract flags: a wrong
        ``True`` leaves uncovered gradient rows uninitialized.

    use_dense_dqacc_for_ranges: SM100/SM110 backward-only optimization for
        ranges-native calls. When ``True``, q/k ranges still drive the
        scheduler, loads and masks, but dQ uses range-local tile-padded
        accumulator slots and the dense 1D bulk-reduce protocol.
        Requires ``deterministic=False``,
        ``disable_fwd_atomic_reduction=True``, sorted pairwise-disjoint
        q_ranges, head_dim divisible by 32, and no ``range_merge``. This
        is a trust-based contract; debug mode validates the Q projection and
        raises on overlap/order violations. There is no runtime fallback.
        Prefer it for a small number of long ranges; many short ranges can
        spend more memory on private padding than they recover in kernel time.

    max_seqlen_q/max_seqlen_k: max sequence length over the ranges (varlen).
        Hard contract: each must be >= its longest range; per-range launch
        bounds are sized from these, so an underestimate silently truncates
        work. Omitting falls back to total_q/total_k — always safe, but the
        loosest bound. Debug mode validates against the ranges (a sync).

    mask_types: the attention mask type applied to the q/k ranges, using the
        int keys from ``MT_MAP`` (0=full, 1=causal, 2=inv_causal, 3=bi_causal).
        It may be:
        - ``None``: all ranges use full attention (the default).
        - ``int``: all ranges share the same mask type. Only 0/1 are accepted
          here; 2/3 exist solely on the per-range path below.
        - ``torch.Tensor`` (cuda int32 ``[R]``): a distinct mask type per range
          (SM100/SM110). Forward and backward both use a runtime per-range path
          (1CTA, no CLC).

    softcap: tanh logit soft-capping value. Implemented internally via the
        score_mod machinery, but exposed here as a plain scalar.

    disable_fwd_atomic_reduction: caller contract that ``q_ranges`` are
        pairwise disjoint (not necessarily sorted), so forward can skip the
        fp32 atomic out reduction. On SM100/SM110 ranges backward it also
        selects the unique-writer dQ path: dq/dq_accum start uninitialized,
        preprocess clears the covered rows, and a final cleanup zeroes the
        holes of self-allocated dq. Defaults to False — pass True only when
        q ranges are pairwise disjoint.

    out_dtype: GMEM O dtype on the atomic path (``None`` = input dtype).
        Pass ``torch.float32`` for the
        lossless merge — K-1 cascading truncation applies otherwise when K
        relations overlap a row.

    disable_bwd_dkv_atomic_reduction: caller contract that
        ``k_ranges`` are sorted and pairwise disjoint, so every
        dK/dV row has a unique CTA writer. SM100/SM110 ranges backward then
        stores dK/dV directly from the main kernel (native dtype, no fp32
        accumulator, no dK/dV postprocess) and only zeroes coverage holes.
        MHA only.

    range_merge: ``True`` or a precomputed :class:`RangeMergePlan` to fold
        relations into (group, pair) CSR form so the forward skips the
        atomic fp32 out merge; backward derives the dual merge. SM100/SM110
        only; requires ``disable_fwd_atomic_reduction=True`` and
        ``disable_bwd_dkv_atomic_reduction=True``.

    flex_attn_args: optional :class:`TorchFlexAttnArgs` bundling the
        FlexAttention-style programmable (``score_mod`` / ``score_mod_bwd`` /
        ``mask_mod`` / ``aux_tensors``) and block-sparse
        (``block_sparse_tensors`` / ``block_sparse_tensors_bwd``) capabilities.
        Leave as ``None`` for the plain dense / range path.
    """
    out, lse = FlexFlashAttnFunc.apply(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        mask_types,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        softcap,
        sink,
        sink_layout,
        pack_gqa,
        deterministic,
        flex_attn_args,
        disable_fwd_atomic_reduction,
        disable_bwd_dkv_atomic_reduction,
        range_merge,
        bwd_q_full_coverage,
        bwd_k_full_coverage,
        use_dense_dqacc_for_ranges,
        out_dtype,
    )

    return out, AttnForwardMeta(lse=lse, max_logits=None)
