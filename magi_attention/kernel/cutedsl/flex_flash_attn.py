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

# mypy: disable-error-code="arg-type,union-attr,attr-defined,unreachable,assignment"

import math
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from quack.compile_utils import make_fake_tensor as fake_tensor

import magi_attention.kernel.cutedsl as magiattn_cutedsl
from magi_attention.common import AttnForwardMeta
from magi_attention.utils.dtype import to_cute_dtype

from .cache_utils import get_jit_cache
from .cutedsl_utils import (
    get_aux_tensor_metadata,
    get_broadcast_dims,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from .ffa_bwd_postprocess import FFABwdPostProcess
from .ffa_bwd_preprocess import FFABwdPreProcess
from .ffa_bwd_sm80 import FFABwdSm80
from .ffa_bwd_sm90 import FFABwdSm90
from .ffa_bwd_sm100 import FFABwdSm100
from .ffa_fwd_sm80 import FFAFwdSm80
from .ffa_fwd_sm90 import FFAFwdSm90
from .ffa_fwd_sm100 import FFAFwdSm100
from .ffa_utils import (
    _get_disable_2cta_default,
    _get_use_clc_scheduler_default,
    convert_from_dlpack_leading_static,
    create_softcap_scoremod,
    create_softcap_scoremod_bwd,
    get_device_arch,
    hash_callable,
    make_fake_bwd_tensors,
    maybe_contiguous,
    tile_size_bwd_sm90,
    tile_size_fwd_sm90,
    validate_arch,
    validate_head_dims,
    validate_tensor,
)
from .sparse_utils import (
    BlockSparseTensorsTorch,
    get_sparse_q_block_size,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
    to_cute_block_sparse_tensors,
)

# isort: split
from .legacy.flash_bwd_sm120 import FlashAttentionBackwardSm120
from .legacy.flash_fwd_sm120 import FlashAttentionForwardSm120


def _flex_flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    softcap: float | None = None,
    learnable_sink: torch.Tensor | None = None,
    pack_gqa: bool | None = None,
    score_mod: Callable | None = None,
    mask_mod: Callable | None = None,
    block_sparse_tensors: BlockSparseTensorsTorch | None = None,
    aux_tensors: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlexFlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors.
            This is how we thread them through to the inner kernel.

    Returns:
        A tuple of (output, lse) where:
        - output is the result of the attention operation, with shape (batch_size, seqlen_q, num_head, head_dim_v) or
          (total_q, num_head, head_dim_v) if cu_seqlens_q is provided.
        - lse is the log-sum-exp of the attention scores, with shape (batch_size, num_head, seqlen_q) or
          (num_head, total_q) if cu_seqlens_q is provided.
    """
    arch, major_arch = get_device_arch()
    validate_arch(arch, major_arch)

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (
            batch_size + 1,
        ), "cu_seqlens_k must have shape (batch_size + 1,)"

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
            assert t.stride(0) == 1, "cu_seqlens_q, cu_seqlens_k must be contiguous"
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

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

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
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
        lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
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
    if mask_mod is not None:
        causal = False

    requested_use_clc_scheduler = _get_use_clc_scheduler_default()
    requested_disable_2cta = _get_disable_2cta_default(is_fwd=True)

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # default
    tile_m, tile_n = 128, 128
    mma_pv_is_rs = True
    intra_wg_overlap = True
    if major_arch == 12:
        # SM120 tile sizes tuned for 99 KB SMEM capacity:
        # D<=64:  128x128 → 48 KB (good occupancy)
        # D>64:   128x64  → 64 KB (128x128 would use 96 KB, hurting occupancy)
        if head_dim <= 64:
            tile_m, tile_n = 128, 128
        else:
            tile_m, tile_n = 128, 64
    elif major_arch == 8:
        tile_m, tile_n = 128, 64  # SM80, should tune
    elif major_arch == 9:
        sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
        fwd_cfg = tile_size_fwd_sm90(
            head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q
        )
        tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if major_arch == 10:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1

    use_2cta_instrs = (
        major_arch in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and cu_seqlens_q is None
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

    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen shapes
    # keep more K/V blocks in flight and hurt L2; dense noncausal mostly just
    # pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = (
        requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal
    )

    if use_block_sparsity:
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[head_dim_idx] != 1:
            pack_gqa = False
        if cu_seqlens_q is not None:
            assert (
                block_sparse_tensors.cu_total_m_blocks is not None
            ), "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
            q_subtile_factor,
        ) = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        learnable_sink is not None,
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

    if compile_key not in _flex_flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, learnable_sink)
        ]
        seqused_q_tensor = seqused_k_tensor = None
        page_table_tensor = None
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        if lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(
                normalized_block_sparse_tensors
            )

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        if major_arch == 8:
            fa_fwd = FFAFwdSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
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
        elif major_arch == 9:
            fa_fwd = FFAFwdSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=2,
                num_threads=384,
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
        elif major_arch in [10, 11]:
            fa_fwd = FFAFwdSm100(
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=False,
                pack_gqa=pack_gqa,
                m_block_size=tile_m,
                n_block_size=tile_n,
                q_stage=q_stage,
                is_persistent=not causal and not local and cu_seqlens_q is None,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                paged_kv_non_tma=False,
                is_varlen_q=cu_seqlens_q is not None,
                q_subtile_factor=q_subtile_factor,
                use_2cta_instrs=use_2cta_instrs,
                use_clc_scheduler=use_clc_scheduler,
                debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
            )
        elif major_arch == 12:
            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity
            assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
            fa_fwd = FlashAttentionForwardSm120(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
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
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
            )
        # TODO: check @can_implement
        compile_args = [
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            None,  # window_size_left
            None,  # window_size_right
            learnable_sink_tensor,
        ]
        if major_arch in [10, 11]:
            # FP8 descale tensors removed; SM100 kernel descale slot is always None.
            compile_args.append(None)
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
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        None,
        None,  # window_size_left
        None,  # window_size_right
        learnable_sink,
    ]
    if major_arch in [10, 11]:
        # FP8 descale tensors removed; SM100 kernel descale slot is always None.
        call_args.append(None)
    call_args.extend(
        [
            (
                normalized_block_sparse_tensors.mask_block_cnt,
                normalized_block_sparse_tensors.mask_block_idx,
                normalized_block_sparse_tensors.full_block_cnt,
                normalized_block_sparse_tensors.full_block_idx,
                normalized_block_sparse_tensors.cu_total_m_blocks,
                normalized_block_sparse_tensors.cu_block_idx_offsets,
                normalized_block_sparse_tensors.dq_write_order,
                normalized_block_sparse_tensors.dq_write_order_full,
            )
            if normalized_block_sparse_tensors is not None
            else None,
            aux_tensors,
        ]
    )

    _flex_flash_attn_fwd.compile_cache[compile_key](*call_args)

    return out, lse


_flex_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


def _compile_bwd_preprocess(
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    has_cuseqlens_q,
    has_seqused_q,
    has_dlse,
    has_dq_accum,
    use_padded_offsets,
):
    """Compile bwd preprocess kernel using cute fake tensors (no real GPU tensors needed)."""
    (
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
        mdQaccum,
        mdKaccum,
        mdVaccum,
    ) = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = (
        fake_tensor(cutlass.Int32, (batchp1,), divisibility=1)
        if has_cuseqlens_q
        else None
    )
    mSequsedQ = (
        fake_tensor(cutlass.Int32, (batch,), divisibility=1) if has_seqused_q else None
    )
    mdLSE = (
        fake_tensor(cutlass.Float32, mLSE.shape, divisibility=1) if has_dlse else None
    )
    mdQaccum = mdQaccum if has_dq_accum else None
    fa_bwd_pre = FFABwdPreProcess(
        dtype, head_dim, head_dim_v, m_block_size, use_padded_offsets=use_padded_offsets
    )
    return cute.compile(
        fa_bwd_pre,
        mO,
        mdO,
        mPdPsum,
        mLSE,
        mLSElog2,
        mdQaccum,
        mCuSeqlensQ,
        mSequsedQ,
        mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out: torch.Tensor,
    dout: torch.Tensor,
    dpsum: torch.Tensor,
    lse: torch.Tensor,
    lse_log2: torch.Tensor,
    dq_accum: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    dlse: torch.Tensor | None,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    m_block_size: int,
    use_padded_offsets: bool = True,
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum."""
    is_varlen = cu_seqlens_q is not None
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        is_varlen,
        seqused_q is not None,
        dlse is not None,
        dq_accum is not None,
        use_padded_offsets,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(
            *compile_key
        )
    _bwd_preprocess.compile_cache[compile_key](
        out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse
    )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(
    dtype,
    hdim,
    block_size,
    num_threads,
    atom_layout,
    swap_ab,
    has_cuseqlens_q,
    has_seqused_q,
    use_2cta_instrs,
    cluster_size,
    arch,
):
    """Compile bwd postprocess kernel using cute fake tensors."""
    (
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
        mdQaccum,
        mdKaccum,
        mdVaccum,
    ) = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = (
        fake_tensor(cutlass.Int32, (batchp1,), divisibility=1)
        if has_cuseqlens_q
        else None
    )
    mSeqUsedQ = (
        fake_tensor(cutlass.Int32, (batch,), divisibility=1) if has_seqused_q else None
    )
    fa_bwd_post = FFABwdPostProcess(
        dtype,
        hdim,
        arch,
        block_size,
        num_threads,
        atom_layout,
        swap_ab,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
    )
    return cute.compile(
        fa_bwd_post,
        mdQaccum,
        mdQ,
        cutlass.Float32(0.0),
        mCuSeqlensQ,
        mSeqUsedQ,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum,
    output,
    scale,
    cu_seqlens,
    seqused,
    arch,
    dtype,
    hdim,
    block_size,
    num_threads,
    atom_layout,
    swap_ab,
    use_2cta_instrs=False,
    cluster_size=1,
):
    """Backward postprocess: convert float32 accumulator to bf16/fp16 output."""
    compile_key = (
        dtype,
        hdim,
        block_size,
        num_threads,
        atom_layout,
        swap_ab,
        cu_seqlens is not None,
        seqused is not None,
        use_2cta_instrs,
        cluster_size,
        arch,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(
            *compile_key
        )
    _bwd_postprocess_convert.compile_cache[compile_key](
        accum,
        output,
        scale,
        cu_seqlens,
        seqused,
    )


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


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
    softmax_scale: float | None = None,
    causal: bool = False,
    softcap: float = 0.0,
    pack_gqa: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    deterministic: bool = False,
    score_mod: Callable | None = None,
    score_mod_bwd: Callable | None = None,
    mask_mod: Callable | None = None,
    aux_tensors: list[torch.Tensor] | None = None,
    block_sparse_tensors: BlockSparseTensorsTorch | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for FlexFlashAttention.

    Returns:
        A tuple of (dQ, dK, dV) gradients with the same shapes and dtypes as the input q, k, v tensors.
    """
    arch, major_arch = get_device_arch()
    validate_arch(arch, major_arch)

    local = False
    sparse_q = None
    if block_sparse_tensors is not None and major_arch == 9:
        sparse_q = (
            block_sparse_tensors.block_size[0]
            if block_sparse_tensors.block_size is not None
            else 128
        )

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    if major_arch == 8:
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
        AtomLayoutMdQ = 1
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
        assert deterministic is False, "deterministic backward not supported on SM 8.0"
    elif major_arch == 12:
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
        assert deterministic is False, "deterministic backward not supported on SM 12.0"
    elif major_arch == 9:
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
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        requested_disable_2cta = _get_disable_2cta_default()
        disable_2cta = (
            requested_disable_2cta
            or score_mod is not None
            or score_mod_bwd is not None
            or mask_mod is not None
            or block_sparse_tensors is not None
        )
        cluster_size = 2 if head_dim >= 128 and not disable_2cta else 1
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
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
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

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (
            batch_size + 1,
        ), "cu_seqlens_k must have shape (batch_size + 1,)"

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"

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
        pack_gqa = qhead_per_kvhead > 1
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
        assert (
            cu_seqlens_q is None and cu_seqlens_k is None
        ), "varlen + score_mod not supported in bwd yet"
        if major_arch == 8:
            raise NotImplementedError(
                "Custom user-provided score_mod is not supported on SM8x architectures."
            )

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
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
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1)
            // m_block_size
            * m_block_size
        )
        dq_accum = torch.empty(
            num_head,
            total_q_rounded_padded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            num_head, total_q_rounded_padded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            num_head, total_q_rounded_padded, dtype=torch.float32, device=device
        )

    # GQA (qhead_per_kvhead > 1) needs dK/dV accum+postprocess since multiple Q heads
    # accumulate into the same dK/dV. SM90 varlen_k with qhead_per_kvhead==1 now uses
    # ragged TMA tensors for direct store, so no longer needs accum+postprocess.
    dKV_postprocess = qhead_per_kvhead > 1
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
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
                (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1)
                // cluster_tile_n
                * cluster_tile_n
            )
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
    _bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        cu_seqlens_q,
        None,  # seqused_q
        None,  # dlse
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        use_padded_offsets=False,
    )

    # num_threads: SM80 (256) and SM120 (128) are set above, SM90 derives from
    # BwdConfig.num_wg, SM100/SM110 uses default from function signature (384).
    if major_arch not in [8, 9, 12]:
        num_threads = 384

    # Backward kernel: compute dk, dv, dq_accum.
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

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            subtile_factor=subtile_factor,
        )
        if deterministic:
            if normalized_block_sparse_tensors.dq_write_order is None:
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order in block_sparse_tensors"
                )
            if (
                normalized_block_sparse_tensors.full_block_cnt is not None
                and normalized_block_sparse_tensors.dq_write_order_full is None
            ):
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order_full when full blocks are present"
                )
            if normalized_block_sparse_tensors.spt is None:
                raise ValueError(
                    "deterministic block-sparse backward requires block_sparse_tensors.spt "
                    "to match dq_write_order direction"
                )
    if (
        normalized_block_sparse_tensors is not None
        and normalized_block_sparse_tensors.spt is not None
    ):
        spt = normalized_block_sparse_tensors.spt and deterministic
    else:
        spt = (causal or local) and deterministic

    if major_arch in [8, 9, 12]:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
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
            cu_seqlens_q is None,
            cu_seqlens_k is None,
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
            causal,
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
            cu_seqlens_q is None,
            cu_seqlens_k is None,
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
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k)
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
        if major_arch in [8, 12]:
            flash_bwd_obj_cls = (
                FlashAttentionBackwardSm120 if major_arch == 12 else FFABwdSm80
            )
            bwd_obj_kwargs = dict(
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
            )
            if major_arch == 8:
                bwd_obj_kwargs[
                    "debug_print"
                ] = magiattn_cutedsl.is_ffa_debug_mode_enabled()
            fa_bwd_obj = flash_bwd_obj_cls(
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
                causal,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                **bwd_obj_kwargs,
            )
        elif major_arch == 9:
            fa_bwd_obj = FFABwdSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
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
        else:
            fa_bwd_obj = FFABwdSm100(
                head_dim,
                head_dim_v,
                is_causal=causal,
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
                debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
            )

        # Block sparse tensors for backward use Q-direction indexing (transposed from forward).
        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(
                normalized_block_sparse_tensors
            )

        # TODO: check @can_implement
        _flex_flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
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
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            None,  # window_size_left
            None,  # window_size_right
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            cute_aux_tensors,
            sparse_tensors_compile,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flex_flash_attn_bwd.compile_cache[compile_key](
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
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        None,  # window_size_left
        None,  # window_size_right
        dQ_semaphore,
        dK_semaphore,
        dV_semaphore,
        aux_tensors,
        (
            normalized_block_sparse_tensors.mask_block_cnt,
            normalized_block_sparse_tensors.mask_block_idx,
            normalized_block_sparse_tensors.full_block_cnt,
            normalized_block_sparse_tensors.full_block_idx,
            normalized_block_sparse_tensors.cu_total_m_blocks,
            normalized_block_sparse_tensors.cu_block_idx_offsets,
            normalized_block_sparse_tensors.dq_write_order,
            normalized_block_sparse_tensors.dq_write_order_full,
        )
        if normalized_block_sparse_tensors is not None
        else None,
    )

    # Postprocess: convert dq_accum from float32 to dq in bf16/fp16
    if major_arch == 9:
        # dQ postprocess: match main kernel's MMA WG count, unless dQ_single_wg
        num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
        num_threads_post_dKV = cfg.num_wg * 128
    elif major_arch == 8:
        # SM80: the postprocess MMA layout is (AtomLayout, num_warps // AtomLayout, 1),
        # which requires num_warps to be a multiple of AtomLayout. The dKV
        # postprocess uses AtomLayoutNdKV=8, so it needs >= 8 warps (256 threads,
        # matching the main kernel). dQ uses AtomLayoutMdQ=1 and is fine with 4 warps.
        num_threads_post_dQ = 128
        num_threads_post_dKV = 256
    else:
        num_threads_post_dQ = 128
        num_threads_post_dKV = 128

    _bwd_postprocess_convert(
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
    )

    if dKV_postprocess:
        # Postprocess: convert dk_accum from float32 to dk in bf16/fp16
        _bwd_postprocess_convert(
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
        )
        # Postprocess: convert dv_accum from float32 to dv in bf16/fp16
        _bwd_postprocess_convert(
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
        )

    return dq, dk, dv


_flex_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


class FlexFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        learnable_sink: torch.Tensor | None = None,
        softcap: float = 0.0,
        pack_gqa: bool | None = None,
        deterministic: bool = False,
        score_mod: Callable | None = None,
        score_mod_bwd: Callable | None = None,
        mask_mod: Callable | None = None,
        aux_tensors: list | None = None,
        block_sparse_tensors: BlockSparseTensorsTorch | None = None,
        block_sparse_tensors_bwd: BlockSparseTensorsTorch | None = None,
    ):
        out, lse = _flex_flash_attn_fwd(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            learnable_sink=learnable_sink,
            softcap=softcap,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            block_sparse_tensors=block_sparse_tensors,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            *(aux_tensors or ()),
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.block_sparse_tensors_bwd = block_sparse_tensors_bwd
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
            cu_seqlens_q,
            cu_seqlens_k,
            *aux,
        ) = ctx.saved_tensors
        aux_tensors = aux if aux else None
        if dout is None:
            dout = torch.zeros_like(out)

        dq, dk, dv = _flex_flash_attn_bwd(
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            dout=dout,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            softcap=ctx.softcap,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            deterministic=ctx.deterministic,
            score_mod=ctx.score_mod,
            score_mod_bwd=ctx.score_mod_bwd,
            mask_mod=ctx.mask_mod,
            aux_tensors=aux_tensors,
            block_sparse_tensors=ctx.block_sparse_tensors_bwd,
        )

        return dq, dk, dv, *((None,) * 30)  # Extra Nones is fine


def flex_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    learnable_sink: torch.Tensor | None = None,
    softcap: float = 0.0,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    score_mod: Callable | None = None,
    score_mod_bwd: Callable | None = None,
    mask_mod: Callable | None = None,
    aux_tensors: list | None = None,
    block_sparse_tensors: BlockSparseTensorsTorch | None = None,
    block_sparse_tensors_bwd: BlockSparseTensorsTorch | None = None,
) -> tuple[torch.Tensor, AttnForwardMeta]:
    """
    Explanation of some optional arguments:

    cu_seqlens_q/cu_seqlens_k: cumulative sequence lengths for variable-length
        (varlen) batches. When provided, q/k/v are expected in the packed
        (total_seqlen, nheads, headdim) layout. varlen is a special case of the
        more general flex q/k ranges that ffa targets.

    max_seqlen_q/max_seqlen_k: max sequence length over the batch (varlen).
    """
    out, lse = FlexFlashAttnFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        learnable_sink,
        softcap,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
    )
    return out, AttnForwardMeta(lse=lse, max_logits=None)
