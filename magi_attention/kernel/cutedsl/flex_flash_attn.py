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
from .ffa_bwd_postprocess import bwd_postprocess, bwd_postprocess_rowmajor
from .ffa_bwd_preprocess import bwd_preprocess
from .ffa_fwd_postprocess import fwd_postprocess
from .ffa_bwd_sm80 import FFABwdSm80
from .ffa_bwd_sm90 import FFABwdSm90
from .ffa_bwd_sm100 import FFABwdSm100
from .ffa_bwd_sm120 import FFABwdSm120
from .ffa_fwd_sm80 import FFAFwdSm80
from .ffa_fwd_sm90 import FFAFwdSm90
from .ffa_fwd_sm100 import FFAFwdSm100
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
    is_ffa_clc_enabled,
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
from .sparse_utils import (
    block_sparse_call_tuple,
    get_sparse_q_block_size,
    prepare_block_sparse_bwd,
    prepare_block_sparse_fwd,
    to_cute_block_sparse_tensors,
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
    disable_fwd_atomic_reduction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlexFlashAttention.

    Args:
        ...
        q_ranges/k_ranges: ``[R, 2]`` int32 cuda tensors of [start, end) q/k
            ranges (relation IR). Validated by :func:`validate_true_ranges`.
            While Phase 2 true-range wiring is incomplete, ranges must still
            form a cu_seqlens partition and are collapsed via
            :func:`ranges_to_cu_seqlens` (SM100/SM110 only).
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
        assert (
            mask_types_tensor is not None
        ), "PER_RANGE requires mask_types_tensor"
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
    # SM100/SM110 kernels consume mQRanges/mKRanges directly. Other arches
    # still read cu_seqlens, and the host block-sparse prep does too, so the
    # collapse remains only for them (ranges are cu-partition-equivalent
    # until 2B).
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
        batch_size = num_ranges  # each relation occupies one scheduler batch slot
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

    # The forked SM80 fwd kernel handles GQA in unpacked mode (one work-tile per
    # query head, indexing mQ by query head directly), so the packed-GQA epilogue
    # path (pack_gqa.store_O) is unsupported. Force it off so the unpacked store
    # path is used consistently with the unpacked mainloop.
    if major_arch == 8:
        pack_gqa = False

    if not disable_fwd_atomic_reduction:
        assert has_ranges and major_arch in (10, 11), (
            "fwd atomic reduction is the SM100/SM110 true-range overlap path"
        )
        # The atomic epilogue reads/writes prev-O by unpacked row.
        pack_gqa = False
    # Under atomic reduction each overlapping relation would re-add the sink,
    # so it leaves the main kernel and folds in once in the fwd postprocess.
    kernel_sink = sink if disable_fwd_atomic_reduction else None

    out_torch_dtype = q.dtype if disable_fwd_atomic_reduction else torch.float32
    device = q.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if not has_ranges else (total_q,)
    )
    lse_shape = (  # (b, nh, sq) or (nh, tq)
        (batch_size, num_head, seqlen_q)
        if not has_ranges
        else (num_head, total_q)
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
        # Atomic reduction merges through LSE: -inf marks a never-written row,
        # so the buffer must be prefilled (callers passing lse own this).
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
    # Static modes collapse to the legacy causal bool for host-side heuristics
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

    requested_use_clc_scheduler = is_ffa_clc_enabled()
    requested_disable_2cta = is_ffa_2cta_disabled(is_fwd=True)
    if use_per_range_mask:
        # DEVIATION: force-disable 2CTA/CLC for per-range masks
        # Reason: V1 mixed kernel keeps 1CTA + static scheduler for correctness
        # Recovery: none in V1; revisit after runtime mixed path is stable
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
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if major_arch == 10:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1
    if not disable_fwd_atomic_reduction:
        # fp32 sO at q_stage=2 eats the smem budget down to kv_stage=1, which
        # deadlocks the KV pipeline. Correctness path runs single-stage Q.
        q_stage = 1

    use_2cta_instrs = (
        major_arch in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not has_ranges
        and not use_block_sparsity
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
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
        requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal
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
        # One int32 per (physical Q block, head); +1 block so the second lock
        # of a straddling tile always has a slot.
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
        magiattn_cutedsl.is_ffa_debug_mode_enabled(),
    )

    # The SM100/SM110 kernel takes [R, 2] ranges in the slots where the other
    # arches take cu_seqlens; both sides of the positional ABI switch together.
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
            to_cute_tensor(t, assumed_align=4, leading_dim=t.ndim - 1)
            if t is not None
            else None
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
                    is_persistent=not causal and not local and not has_ranges,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=False,
                    is_varlen_q=has_ranges,
                    q_subtile_factor=q_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=use_clc_scheduler,
                    use_per_range_mask=use_per_range_mask,
                    disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
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
            # FP8 descale tensors removed; SM100 kernel descale slot is always None.
            compile_args.append(None)
            compile_args.append(mask_types_cute_tensor)
            compile_args.append(
                to_cute_tensor(range_locks, assumed_align=4)
                if range_locks is not None
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
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sink: torch.Tensor | None = None,
    sink_layout: AttnSinkLayout = "sh",
    pack_gqa: bool = False,
    deterministic: bool = False,
    flex_attn_args: TorchFlexAttnArgs | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for FlexFlashAttention.

    Returns:
        A tuple of (dQ, dK, dV) gradients with the same shapes and dtypes as the input q, k, v tensors.
    """
    arch, major_arch = get_device_arch()
    validate_arch(arch, major_arch)

    assert (
        sink_layout == "sh"
    ), f"only sink_layout='sh' is supported, got {sink_layout!r}"
    if mask_mode == MaskMode.PER_RANGE:
        assert (
            mask_types_tensor is not None
        ), "PER_RANGE requires mask_types_tensor"
    else:
        assert (
            mask_types_tensor is None
        ), "static mask modes must not pass mask_types_tensor"

    has_ranges = q_ranges is not None or k_ranges is not None
    if has_ranges:
        validate_true_ranges(q_ranges, k_ranges, mask_types=mask_types_tensor)
        if deterministic:
            # The deterministic bwd still runs the interleaved bulk
            # accumulator layout, which aliases between unaligned
            # neighbouring ranges once offsets are physical: gradients
            # come out bit-stable but wrong. 2E4 rebuilds it on the
            # row-major layout; until then reject loudly.
            raise NotImplementedError(
                "deterministic backward with q/k ranges is unsupported "
                "until the row-major deterministic path lands"
            )
    # SM100/SM110 pre/main/post kernels consume ranges directly; other arches
    # still read cu_seqlens (ranges are cu-partition-equivalent until 2B).
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
            disable_2cta = (
                requested_disable_2cta
                or score_mod is not None
                or score_mod_bwd is not None
                or mask_mod is not None
                or block_sparse_tensors is not None
                or use_per_range_mask
            )
            # DEVIATION: force 1CTA whenever ranges are present — the
            # atomic accum path asserts 1CTA, and det+ranges is rejected
            # upstream until 2E4.
            # Reason: V1 mixed bwd keeps 1CTA for correctness
            # Recovery: none in V1
            cluster_size = (
                1
                if use_per_range_mask or has_ranges
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
        batch_size = num_ranges  # each relation occupies one scheduler batch slot
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

    # Uncovered rows keep zero gradients on the ranges path; kernels never
    # visit them, so the allocation must provide the zeros.
    grad_alloc = torch.zeros_like if has_ranges else torch.empty_like

    if dq is None:
        dq = grad_alloc(q)
    else:
        validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = grad_alloc(k)
    else:
        validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = grad_alloc(v)
    else:
        validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if not has_ranges:
        dq_accum = torch.empty(
            batch_size,
            num_head,
            seqlen_q_rounded * head_dim_rounded,
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
        # Original-token-space accumulators: overlap is absorbed by the same
        # atomics, and a tail tile running past a range end only adds rows the
        # mask already zeroed. One extra tile keeps the last tail in bounds.
        total_q_rounded_padded = (
            (total_q + m_block_size - 1) // m_block_size + 1
        ) * m_block_size
        # Zeros: the row-major postprocess sweeps every row, so uncovered
        # rows must read back the accumulator's zeros.
        dq_accum = torch.zeros(
            num_head,
            total_q_rounded_padded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        # Zeros, not empty: rows in coverage gaps are never written by the
        # preprocess, yet a neighbour range's tail tile may read them —
        # exp(-inf - garbage) can go NaN if the garbage happens to be -inf.
        dpsum = torch.zeros(
            num_head, total_q_rounded_padded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.zeros(
            num_head, total_q_rounded_padded, dtype=torch.float32, device=device
        )

    # GQA (qhead_per_kvhead > 1) needs dK/dV accum+postprocess since multiple Q heads
    # accumulate into the same dK/dV. SM90 varlen_k with qhead_per_kvhead==1 now uses
    # ragged TMA tensors for direct store, so no longer needs accum+postprocess.
    # Ranges force the accum+postprocess path even for MHA: overlapping K
    # ranges make several CTAs write the same dK/dV rows, so direct stores
    # would clobber each other. (GQA needs it regardless.)
    dKV_postprocess = qhead_per_kvhead > 1 or (has_ranges and major_arch in (10, 11))
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if not has_ranges:
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
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
            total_k_rounded_padded = (
                (total_k + cluster_tile_n - 1) // cluster_tile_n + 1
            ) * cluster_tile_n
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
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

    # Preprocess kernel: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum.
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
        # zeros-allocated accum needs no per-range clearing; skipping it also
        # removes the PDL window where one range's clear could race another
        # range's main-kernel atomics on shared physical rows.
        None if pre_post_q_ranges is not None and not deterministic else dq_accum,
        cu_seqlens_q,
        None,  # seqused_q
        None,  # dlse
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        use_padded_offsets=False,
        q_ranges=pre_post_q_ranges,
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
            convert_from_dlpack_leading_static(
                t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order()
            )
            if t is not None
            else None
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

    # Non-deterministic true-range bwd stores row-major fp32 atomics; its
    # postprocess is a per-(range, tile) scale+cast sweep that never touches
    # rows outside every range.
    rowmajor_post = pre_post_q_ranges is not None and not deterministic
    if rowmajor_post:
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
        )

    if dKV_postprocess and rowmajor_post:
        bwd_postprocess_rowmajor(dk_accum, dk, k_ranges, seqlen_k, softmax_scale)
        bwd_postprocess_rowmajor(dv_accum, dv, k_ranges, seqlen_k, 1.0)
    elif dKV_postprocess:
        # Postprocess: convert dk_accum from float32 to dk in bf16/fp16
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
        # Postprocess: convert dv_accum from float32 to dv in bf16/fp16
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
        disable_fwd_atomic_reduction: bool = True,
    ):
        arch, major_arch = get_device_arch()
        is_varlen = q_ranges is not None or k_ranges is not None
        if is_varlen:
            num_ranges = validate_true_ranges(
                q_ranges, k_ranges, mask_types=mask_types
            )
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
            # Reason: FFAFwdSm100.is_causal / host heuristics key off mask_type
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
        )

        aux_tensors = flex_attn_args.aux_tensors if flex_attn_args else None
        # mask_types does not require grad tracking; keep it on ctx directly so
        # save_for_backward stays focused on tensors that participate in autograd.
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
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
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
            # target dtype (same contract as the distributed C++ path).
            out = out.to(q.dtype)
            dout = dout.to(q.dtype)

        # Restore aux_tensors from the saved tail (kept tracked by autograd).
        flex_attn_args: TorchFlexAttnArgs | None = ctx.flex_attn_args
        if flex_attn_args is not None:
            flex_attn_args = flex_attn_args.with_aux_tensors(aux)

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
            deterministic=ctx.deterministic,
            flex_attn_args=flex_attn_args,
        )

        return dq, dk, dv, *((None,) * 30)  # Extra Nones is fine


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
    disable_fwd_atomic_reduction: bool = True,
) -> tuple[torch.Tensor, AttnForwardMeta]:
    """
    Flex-flash-attention interface (dense / range / varlen).

    Explanation of some optional arguments:

    q_ranges/k_ranges: ``[R, 2]`` int32 cuda tensors describing per-range
        [start, end) intervals over the packed (total_seqlen, nheads, headdim)
        q/k layout — the CuteDSL relation IR (aligned with C++ FFA ranges under
        non-overlapping Q/K, i.e. ``disable_*_atomic_reduction=True``).
        When provided, both must be set, q/k/v must be packed, and the device
        must be SM100/SM110. Leave as ``None`` for the dense
        (batch, seqlen, nheads, headdim) path.

        # DEVIATION: ranges must still be equivalent to a cu_seqlens partition
        # Reason: kernels still collapse via ranges_to_cu_seqlens until Phase 2
        # Recovery: Phase 2 passes mQRanges/mKRanges and drops the collapse

    max_seqlen_q/max_seqlen_k: max sequence length over the ranges (varlen).

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
    )

    return out, AttnForwardMeta(lse=lse, max_logits=None)
