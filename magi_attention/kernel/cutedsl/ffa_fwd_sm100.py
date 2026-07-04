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

# mypy: disable-error-code="union-attr,index,assignment,misc"
# pyright: reportInvalidTypeForm=false

# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import math
from functools import partial
from typing import Callable, Literal, NamedTuple, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Boolean, Float32, Int32, Int64, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler

# isort: split
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase

from . import cutedsl_utils
from . import pipeline as ffa_pipeline
from . import sm100_utils
from .block_info import BlockInfo
from .cutedsl_utils import ThreadCooperativeGroup
from .ffa_utils import MT_MAP
from .mask import AttentionMask
from .named_barrier import NamedBarrierFwdSm100
from .pack_gqa import PackGQA, pack_gqa_layout
from .paged_kv import PagedKVManager
from .seqlen_info import SeqlenInfoQK
from .softmax import SoftmaxSm100, apply_score_mod_inner
from .sparse_utils import (
    BlockSparseTensors,
    get_total_block_count,
    handle_block_sparse_empty_tile_correction_sm100,
    produce_block_sparse_inner_iters_sm100,
    softmax_block_sparse_sm100,
)
from .tile_scheduler import (
    ClcState,
    SchedulingMode,
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)

# === TUNING KNOBS (agent-editable) ===
# Keys: (use_2cta_instrs: bool, is_causal: bool, head_dim_padded: int, is_sm103: bool)
# Values:
#   ex2_emu_freq: int — how often to use emulated exp2 (0=all hardware exp2, higher=more emulation).
#                        SM103 has fast native exp2, so set freq=0 there.
#   ex2_emu_res: int — (hd256 only) number of fragment-pairs per freq period to emulate.
#   ex2_emu_start_frg: int — fragment index to start emulation from
#   num_regs_softmax: int — register count for softmax warps (multiple of 8)
#   num_regs_correction: int — register count for correction warps (multiple of 8)
#   num_regs_other is derived: 512 - num_regs_softmax * 2 - num_regs_correction
#                  (hd256 exception: num_regs_other is fixed at 32, not derived)
_TUNING_CONFIG = {
    (True, False, 128, False): {
        "ex2_emu_freq": 10,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 176,
        "num_regs_correction": 88,
    },
    (False, True, 128, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 192,
        "num_regs_correction": 72,
    },
    (True, False, 192, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 184,
        "num_regs_correction": 80,
    },
    (False, True, 192, False): {
        "ex2_emu_freq": 32,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 192,
        "num_regs_correction": 72,
    },
    (True, False, 128, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 80,
    },
    (False, True, 128, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 64,
    },
    (True, False, 192, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 64,
    },
    (False, True, 192, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 72,
    },
    (True, False, 256, False): {
        "ex2_emu_freq": 14,
        "ex2_emu_res": 6,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 256,
        "num_regs_correction": 160,
    },
    (True, True, 256, False): {
        "ex2_emu_freq": 14,
        "ex2_emu_res": 6,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 256,
        "num_regs_correction": 160,
    },
}
_FP8_TUNING_CONFIG = {
    (True, False, 128, False): {
        "ex2_emu_freq": 10,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 160,
        "num_regs_correction": 72,
    },
}
_FP8_SMALL_HDIM_REGS = {
    False: {"num_regs_softmax": 168, "num_regs_correction": 96, "num_regs_other": 80},
    True: {"num_regs_softmax": 152, "num_regs_correction": 96, "num_regs_other": 112},
}
# === END TUNING KNOBS ===


class DescaleTensors(NamedTuple):
    q_descale: Optional[cute.Tensor] = None
    k_descale: Optional[cute.Tensor] = None
    v_descale: Optional[cute.Tensor] = None

    def __new_from_mlir_values__(self, values):
        return DescaleTensors(*((*values, None, None, None)[:3]))


class FFAFwdSm100:
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        mask_type: int = MT_MAP.full,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
        debug_print: bool = False,
    ):
        self.use_tma_KV = not paged_kv_non_tma
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size  # tileQ128
        self.n_block_size = n_block_size  # tileK128
        self.q_stage = q_stage  # shard Q/S/P/O to Qi/Si/Pi/Oi, i =[0, q_stage)
        assert self.q_stage in [1, 2]
        self.use_2cta_instrs = use_2cta_instrs

        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size

        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert (
            self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f
        ), "Only SM 10.x and 11.x are supported"

        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        # NOTE: cta_tiler M includes only 1 CTA, the scheduler will take into account the cluster shape
        # (tileQ128*stageQ,tileK128,tileHD128) per CTA
        # which shards Q/S/P/O along sq dim to Qi/Si/Pi/Oi, i={0,1}
        self.cta_tiler = (
            self.q_stage * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )

        # NOTE: With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
        # Each CTA owns m_block_size rows and n_block_size//2 cols of sA/sB across 2 CTAs,
        # and then produces [m_block_size x n_block_size] partial tC each
        self.mma_tiler_qk = (  # (tileQ128*CTA2,tileK128,tileHD128) per MMA_QK
            self.cta_group_size * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (  # (tileQ128*CTA2,tileHD128,tileK128) per MMA_PV
            self.cta_group_size * m_block_size,
            self.head_dim_v_padded,
            n_block_size,
        )

        # epi_tile is per-CTA (not full 2CTA)
        # since each CTA writes its own O portion
        self.epi_tile = (
            self.m_block_size,
            self.head_dim_v_padded,
        )  # (tileQ128, tileHD128)

        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        self.q_subtile_factor = q_subtile_factor

        assert not (
            self.is_split_kv and self.head_dim_v_padded >= 192
        ), "SplitKV is not supported for hdim >= 192"

        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if const_expr(has_aux_tensors) else 2
        )

        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        self.is_sm103 = is_sm103

        # enable_ex2_emu is derived: True if tuning config has freq > 0, else fallback to default logic
        _default_enable_ex2_emu = (
            self.head_dim_padded <= 128
            or (
                self.head_dim_padded == 192
                and self.use_2cta_instrs
                and not self.is_causal
                and not self.is_local
            )
            # NOTE: B300 (sm103) has fast SFU, thus this approximation is only for B200 (sm100)
        ) and not is_sm103

        self.enable_ex2_emu = _default_enable_ex2_emu

        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        self.s0_s1_barrier = False

        self.overlap_sO_sQ = (
            self.head_dim_padded == 192 and self.head_dim_v_padded >= 64
        ) or (self.head_dim_v_padded >= 128 and self.is_split_kv)
        if self.overlap_sO_sQ:
            self.is_persistent = False

        assert self.use_tma_KV or not (
            self.check_hdim_oob or self.check_hdim_v_oob
        ), "Paged KV does not support irregular head dim"

        # ClC does not compose with these other features, so disable even if requested
        self.use_clc_scheduler = (
            use_clc_scheduler and self.use_tma_KV and not self.overlap_sO_sQ
        )
        if self.use_clc_scheduler:
            assert (
                self.cluster_shape_mn[1] == 1
            ), f"CLC requires cluster N == 1: {self.cluster_shape_mn}"
            assert self.cluster_shape_mn[0] in (
                1,
                2,
            ), f"bad CLC cluster M: {self.cluster_shape_mn}"
            assert (
                self.cluster_shape_mn[0] == self.cta_group_size
            ), f"CLC cluster M != cta_group_size: {self.cluster_shape_mn}, {self.cta_group_size}"

        self.sched_stages = 1
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )

        if is_varlen_q:
            self.TileScheduler = SingleTileVarlenScheduler
        elif self.is_causal or self.is_local or self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        self.use_tma_Q = not (
            self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0
        )

        if self.q_stage == 1:
            if not self.use_tma_KV or not self.use_tma_Q:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            else:
                self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()

        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q:  # fallback
            self.epilogue_warp_ids = (13, 14)

        self.clc_scheduler_warp_id = (
            self.empty_warp_ids[0] if self.use_clc_scheduler else None
        )

        self.tmem_s_offset = [0, self.n_block_size]  # [0, tileK) = [0, 128)
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # [2*tileK, 2*tileK + tileHD) = [256, 384)
        self.tmem_total = (
            self.tmem_o_offset[-1] + self.head_dim_v_padded
        )  # 2 * (tileK + tileHD) = 512
        assert self.tmem_total <= self.tmem_alloc_cols

        # bf16 tP only needs half of fp32 tS space
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [  # [tileK//2, tileK//2 + tileK] = [64, 192)
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]

        # vec buffer for row_max & row_sum with tmem shape [128, 2)
        self.tmem_vec_offset = (
            self.tmem_s_offset
        )  # reuse tS space since we don't need tS after softmax

        # Look up tuning config for register counts and ex2_emu params
        _tune_key = (
            self.use_2cta_instrs,
            self.is_causal,
            self.head_dim_padded,
            self.is_sm103,
        )
        self._tune = _TUNING_CONFIG.get(_tune_key, {})
        if "ex2_emu_freq" in self._tune:
            self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0
        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
            self.num_regs_correction = 64
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        else:
            if not paged_kv_non_tma and "num_regs_softmax" in self._tune:
                self.num_regs_softmax = self._tune["num_regs_softmax"]
                self.num_regs_correction = self._tune["num_regs_correction"]
            elif not paged_kv_non_tma:
                self.num_regs_softmax = 192
                self.num_regs_correction = 80
            else:
                self.num_regs_softmax = 184
                self.num_regs_correction = 64

            self.num_regs_total = 512
            self.num_regs_other = (
                self.num_regs_total
                - self.num_regs_softmax * 2
                - self.num_regs_correction
            )

        self.buffer_align_bytes = 1024

        self.debug_print = debug_print

        # --- Debug print ---

        if self.debug_print:
            prefix = "[fwd_sm100_init] "
            print()
            print(f"{prefix}Initialized FFAFwdSm100 with: ")
            print(f"{prefix}{head_dim=} | {head_dim_v=} | {qhead_per_kvhead=}")
            print(
                f"{prefix}{mask_type=} | {self.is_causal=} | {is_local=} | "
                f"{is_split_kv=} | {pack_gqa=}"
            )
            print(
                f"{prefix}{q_subtile_factor=} | {m_block_size=} | {n_block_size=} | {q_stage=}"
            )
            print(
                f"{prefix}{is_persistent=} | {score_mod=} | {mask_mod=} | {has_aux_tensors=} | {paged_kv_non_tma=}"
            )
            print(
                f"{prefix}{use_2cta_instrs=} | {use_clc_scheduler=} | {self.enable_ex2_emu=} | {self.threads_per_cta=}"
            )
            print(
                f"{prefix}{self.sched_stages=} | {self.TileScheduler.__name__=} | {self.scheduling_mode=}"
            )
            print(f"{prefix}{self.epi_tile=} | {self.cta_tiler=} | {self.cta_group=}")
            print(
                f"{prefix}{self.head_dim_padded=} | {self.head_dim_v_padded=} | {self.use_tma_KV=} | {self.use_tma_Q=}"
            )
            print(f"{prefix}{self.cluster_shape_mn=} | {self.cluster_shape_mnk=}")
            print(
                f"{prefix}{self.cta_group_size=} | {self.mma_tiler_qk=} | {self.mma_tiler_pv=}"
            )
            print(
                f"{prefix}{self.num_regs_softmax=} | {self.num_regs_correction=} | {self.num_regs_other=}"
            )
            print(
                f"{prefix}{self.tmem_alloc_cols=} | {self.tmem_total=} | {self.use_correction_warps_for_epi=}"
            )
            print(
                f"{prefix}{self.tmem_s_offset=} | {self.tmem_o_offset=} | {self.tmem_p_offset=} | {self.tmem_vec_offset=}"
            )
            print(
                f"{prefix}{self.mma_warp_id=} | {self.epilogue_warp_ids=} | {self.load_warp_ids=} | {self.empty_warp_ids=}"
            )
            print(
                f"{prefix}{self.softmax0_warp_ids=} | {self.softmax1_warp_ids=} | {self.correction_warp_ids=}"
            )
            print(
                f"{prefix}{self.split_P_arrive=} | {self.s0_s1_barrier=} | {self.overlap_sO_sQ=}"
            )
            print()

    @property
    def is_causal(self) -> bool:
        return self.mask_type == MT_MAP.causal

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        smem_size_q = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_padded
            * self.q_dtype.width
            // 8
        )
        smem_size_o = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_v_padded
            * self.o_dtype.width
            // 8
        )
        smem_size_q_o = (
            smem_size_q + smem_size_o
            if not self.overlap_sO_sQ
            else max(smem_size_q, smem_size_o)
        )
        smem_size_k_per_stage = (
            self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        )
        smem_size_v_per_stage = (
            self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        )
        smem_size_kv_per_stage = (
            max(smem_size_k_per_stage, smem_size_v_per_stage) // self.cta_group_size
        )
        kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
        if (
            self.head_dim_padded == 192
            and self.head_dim_v_padded == 128
            and kv_stage == 2
        ):
            # For hdim 192,128, we can fit 3 stages if we use uneven_kv_smem
            kv_stage = 3
        self.kv_stage = kv_stage
        # print("kv_stage", self.kv_stage)
        self.s_stage = 2
        assert self.s_stage >= self.q_stage
        # For hdim 192,128 1CTA, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            self.head_dim_padded == 192
            and self.head_dim_v_padded == 128
            and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.n_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm100_setup_attributes] "
            print()
            print(f"{prefix}{self.kv_stage=} | {self.s_stage=} | {self.q_stage=}")
            print(f"{prefix}{self.uneven_kv_smem=} | {self.uneven_kv_smem_offset=}")
            print()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """

        # ///////////////////////////////////////////////////////////////////////////////
        # Make mQ/mK/mV/mO/mLSE tensors
        # with layout transformations for specific memory access patterns
        # ///////////////////////////////////////////////////////////////////////////////

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [
            cutedsl_utils.assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)
        ]

        # --- Make mQ ---

        # (b, sq, nhq, hd) -> (sq, hd, nhq, b)
        # or (sq, nhq, hd) -> (sq, hd, nhq) if there's cu_seqlens_q
        Q_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )

        # --- Make mK/mV ---

        # (b, sk, nhk, hd) -> (sk, hd, nhk, b)
        # or (sk, nhk, hd) -> (sk, hd, nhk) if there's cu_seqlens_k
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]

        # (sk, hd, nhk, b) -> (hd, sk, nhk, b)
        # or (sk, nhk, hd) -> (hd, sk, nhk) if there's cu_seqlens_k
        V_layout_transpose = (
            [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        )
        mV = cute.make_tensor(  # actually => actually V.T
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )

        # --- Make mO/mLSE ---

        if const_expr(self.is_split_kv):
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = (
                [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            )
            num_splits = mO.shape[0]
        else:
            # (b, sq, nhq, hd) -> (sq, hd, nhq, b)
            # or (sq, nhq, hd) -> (sq, hd, nhq) if there's cu_seqlens_q
            O_layout_transpose = (
                [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            )

            # (b, nhq, sq) -> (sq, nhq, b)
            # or (nhq, sq) -> (sq, nhq) if there's cu_seqlens_q
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
            num_splits = Int32(1)

        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if const_expr(mLSE is not None)
            else None
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Set up attributes
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Check type consistency ---

        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        if const_expr(self.q_dtype.width == 8):
            paged_kv_non_tma = not self.use_tma_KV
            if const_expr(self.head_dim_padded < 96):
                fp8_regs = _FP8_SMALL_HDIM_REGS[paged_kv_non_tma]
                self.num_regs_softmax = fp8_regs["num_regs_softmax"]
                self.num_regs_correction = fp8_regs["num_regs_correction"]
                self.num_regs_other = fp8_regs["num_regs_other"]
            else:
                fp8_tune = _FP8_TUNING_CONFIG.get(
                    (
                        self.use_2cta_instrs,
                        self.is_causal,
                        self.head_dim_padded,
                        self.is_sm103,
                    ),
                    {},
                )
                if const_expr("ex2_emu_freq" in fp8_tune):
                    self._tune = {**self._tune, **fp8_tune}
                    self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0
                if const_expr(not paged_kv_non_tma and "num_regs_softmax" in fp8_tune):
                    self.num_regs_softmax = fp8_tune["num_regs_softmax"]
                    self.num_regs_correction = fp8_tune["num_regs_correction"]
                    self.num_regs_other = (
                        512 - self.num_regs_softmax * 2 - self.num_regs_correction
                    )

        # --- Set up attributes ---

        self._setup_attributes()

        self.use_tma_O = (
            self.arch >= Arch.sm_90
            and mCuSeqlensQ is None
            and mSeqUsedQ is None
            and not (self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0)
            and not (self.pack_gqa and self.is_split_kv)
        )
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 1)
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = self._tune.get("ex2_emu_freq", 16)
            if const_expr(
                self.pack_gqa
                and self.head_dim_padded > 64
                and not self.is_causal
                and not self.is_local
            ):
                self.ex2_emu_freq = (
                    32
                    if mCuSeqlensQ is not None or mSeqUsedQ is not None
                    else self._tune.get("ex2_emu_freq", 10)
                )

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tiled MMA, tiled TMA copy, and SMEM layouts
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tiled MMA for S=QK.T ---

        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN  # V.T
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           2:1
        # Shape MNK:       (256,128,16)
        # TV Layout A:     (2,(128,16)):(128,(1,256))
        # TV Layout B:     (2,(64,16)):(64,(1,128))
        # TV Layout C:     (2,(128,128)):(128,(1,256))
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            self.cta_group,
            self.mma_tiler_qk[:2],
        )

        self.cta_group_shape = tiled_mma_qk.thr_id.shape  # (2,1)
        cta_layout_vmnk = (  # (CTA_V(2),CTA_M1,CTA_N1,CTA_K1):((1),0,0,0)
            cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (self.cta_group_shape,)
            )
        )
        if const_expr(cute.size(self.cta_group_shape) != self.cta_group_size):
            raise ValueError(
                f"CTA group shape {self.cta_group_shape} "
                f"does not match expected size {self.cta_group_size}"
            )

        # --- Make tiled MMA for O=PV ---

        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K

        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           2:1
        # Shape MNK:       (256,128,16)
        # TV Layout A:     (2,(128,16)):(128,(1,256))
        # TV Layout B:     (2,(64,16)):(64,(1,128))
        # TV Layout C:     (2,(128,128)):(128,(1,256))
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            self.cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        # --- Make smem layout for sQ/sK/sV/sO ---

        # sQ: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_Q1,MMA_HD=(4,2),stageQ):((64,1),0,(16,8192),16384)
        # sK: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_K1,MMA_HD=(4,2),stageK):((64,1),0,(16,4096),8192)
        # tP: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_Q1,MMA_K=(4,2),stageS):((64,1),0,(16,8192),16384)
        # sV: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_K1,MMA_HD=(4,2),stageK):((1,64),0,1024,8192)
        # sO: S<3,4,3> o 0 o (EPI_Q=(8,16),EPI_HD=(64,2),EPI_STAGE=(1,2)):((64,512),(1,8192),(0,16384))
        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )

        # TODO: review the logics here
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem
            # so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        # TODO: review the logics here
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type,
                cute.select(layout, mode=[0, 1, 2]),  # slice out stage dim
            )
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        for name in ("Q", "K", "V"):
            # NOTE: for both MMA sA/sB, we need to times cta_group_size
            # since the smem layouts are only for single CTA
            self.tma_copy_bytes[name] *= self.cta_group_size

        # --- Make tiled TMA G2S-copy for Q/K/V ---

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        tma_atom_Q = None
        if const_expr(self.use_tma_Q):
            # atom: layout_src_tv=(2,8192):(8192,1) | layout_dst_tv=(2,8192):(8192,1)
            tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mQ,
                cute.select(sQ_layout, mode=[0, 1, 2]),  # slice out stage dim
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            gmem_tiled_copy_Q = None
        else:  # no TMA for Q, use cp.async instead
            async_copy_elems = 128 // self.q_dtype.width
            num_load_threads = cute.arch.WARP_SIZE * len(self.load_warp_ids)
            threads_per_row = math.gcd(
                self.head_dim_padded // async_copy_elems, num_load_threads
            )
            gmem_tiled_copy_Q = copy_utils.tiled_copy_2d(
                self.q_dtype,
                threads_per_row,
                num_load_threads,
                async_copy_elems,
                is_async=True,  # using cp.async, otherwise ld.shared
            )

        tma_atom_K = None
        tma_atom_V = None
        if const_expr(self.use_tma_KV):
            # TMA load for K
            # atom: layout_src_tv=(2,4096):(4096,1) | layout_dst_tv=(2,4096):(4096,1)
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),  # slice out stage dim
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            # TMA load for V
            # atom: layout_src_tv=(2,8192):(8192,1), layout_dst_tv=(2,8192):(8192,1)
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),  # slice out stage dim
                self.mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk.shape,
            )

        # --- Make tiled TMA S2G-copy of O ---

        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)

        tma_atom_O = None
        if const_expr(self.use_tma_O):
            # TMA store atom for O
            # layout_src_tv=(1,8192):(0,1)
            # layout_dst_tv=(1,8192):(0,1)
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )
            gmem_tiled_copy_O = None
        else:
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(  # st.shared
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(
                atom_universal_copy, tO_layout, vO_layout
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tile scheduler class/args, SMEM storage, and others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tile scheduler class/args ---

        TileScheduler = self.TileScheduler
        _num_block_divisor = self.cta_tiler[0] * (
            self.cta_group_size
            if not self.is_persistent and self.cta_group_size > 1
            else 1
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), _num_block_divisor),
            num_head=cute.size(mQ.shape[2]),
            num_batch=cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits=num_splits,
            seqlen_k=cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            headdim=mQ.shape[1],
            headdim_v=mV.shape[
                0
            ],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=not self.is_persistent and self.cta_group_size > 1,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler

        # (max_ctas, 1, 1), where max_ctas = sm_counts // cluster_size
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # --- Make smem storage ---

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout)
            if const_expr(not self.overlap_sO_sQ)
            else cutlass.max(
                cute.cosize(sQ_layout),
                cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
            )
        )

        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            # ---  mbarriers for pipelines ---

            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
            # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]

            # --- CLC buffers ---

            # CLC buffers placed here to utilize padding before sO's 1024-byte alignment.
            # This avoids adding bytes at the end when we're at the smem limit.
            # PipelineClcFetchAsync expects 2 * sched_stages mbarriers (full + empty).
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            # CLC response storage (16 bytes per stage, stored as 4 Int32s).
            clc_response: cute.struct.MemRange[Int32, clc_response_size]

            # --- tmem ptr ---

            # Tmem dealloc cluster mbarrier
            tmem_dealloc_mbar_ptr: Int64
            # Tmem holding buffer ptr
            tmem_holding_buf_ptr: Int32

            # --- smem tensors ---
            # row max and row sum for backward
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]

            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]

        self.shared_storage = SharedStorage

        # --- Make others ---

        softmax_scale_log2, softmax_scale = cutedsl_utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = (
            Int32(window_size_left) if window_size_left is not None else None
        )
        window_size_right = (
            Int32(window_size_right) if window_size_right is not None else None
        )
        fastdiv_mods = cutedsl_utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, mPageTable
        )

        head_divmod = None
        if const_expr(self.pack_gqa):
            head_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = const_expr(blocksparse_tensors is not None)
        if const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError(
                "Block sparsity + paged KV not supported on SM100"
            )
        if const_expr(self.use_block_sparsity and self.is_varlen_q):
            assert const_expr(
                blocksparse_tensors.cu_total_m_blocks is not None
            ), "blocksparse_tensors.cu_total_m_blocks must be provided for varlen blocksparsity"

        # ///////////////////////////////////////////////////////////////////////////////
        # Launch the kernel
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm100_call] "

            print()
            print(f"{prefix}tiled_mma_qk: {tiled_mma_qk}")
            print()
            print(f"{prefix}tiled_mma_pv: {tiled_mma_pv}")
            print()
            print(
                f"{prefix}cta_group_shape: {self.cta_group_shape} | cta_layout_vmnk: {cta_layout_vmnk}"
            )
            print(
                f"{prefix}q_major_mode: {q_major_mode} | k_major_mode: {k_major_mode} | "
                f"v_major_mode: {v_major_mode} | o_layout: {self.o_layout}"
            )
            print(f"{prefix}sQ_layout: {sQ_layout}")
            print(f"{prefix}sK_layout: {sK_layout}")
            print(f"{prefix}tP_layout: {tP_layout}")
            print(f"{prefix}sV_layout: {sV_layout}")
            print(f"{prefix}sO_layout: {sO_layout}")
            print(f"{prefix}epi_tile: {self.epi_tile}")
            print(
                f"{prefix}q_stage: {self.q_stage} | kv_stage: {self.kv_stage} | s_stage: {self.s_stage}"
            )
            print(
                f"{prefix}use_tma_Q: {self.use_tma_Q} | use_tma_KV: {self.use_tma_KV} | use_tma_O: {self.use_tma_O}"
            )
            print(f"{prefix}threads_per_cta: {self.threads_per_cta}")
            print()

            cute.printf("")
            cute.printf(prefix + "mQ.layout: {}", mQ.layout)
            cute.printf(prefix + "mK.layout: {}", mK.layout)
            cute.printf(prefix + "mV.layout: {}", mV.layout)
            cute.printf(prefix + "mO.layout: {}", mO.layout)
            cute.printf("")
            cute.printf(prefix + "grid_dim: {}", grid_dim)
            cute.printf(
                prefix + "tma_copy_bytes: Q={} K={} V={}",
                self.tma_copy_bytes["Q"],
                self.tma_copy_bytes["K"],
                self.tma_copy_bytes["V"],
            )
            cute.printf(
                prefix + "softmax_scale_log2={} softmax_scale={}",
                softmax_scale_log2,
                softmax_scale,
            )
            if const_expr(self.use_tma_Q):
                cute.printf(
                    prefix + "tma_atom_Q: layout_src_tv={}, layout_dst_tv={}",
                    tma_atom_Q.layout_src_tv,
                    tma_atom_Q.layout_dst_tv,
                )
            if const_expr(self.use_tma_KV):
                cute.printf(
                    prefix + "tma_atom_K: layout_src_tv={}, layout_dst_tv={}",
                    tma_atom_K.layout_src_tv,
                    tma_atom_K.layout_dst_tv,
                )
                cute.printf(
                    prefix + "tma_atom_V: layout_src_tv={}, layout_dst_tv={}",
                    tma_atom_V.layout_src_tv,
                    tma_atom_V.layout_dst_tv,
                )
            if const_expr(self.use_tma_O):
                cute.printf(
                    prefix + "tma_atom_O: layout_src_tv={}, layout_dst_tv={}",
                    tma_atom_O.layout_src_tv,
                    tma_atom_O.layout_dst_tv,
                )
            cute.printf("")

        # --- Launch the kernel ---

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            descale_tensors,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            cta_layout_vmnk,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            head_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk
            if cute.size(self.cluster_shape_mnk) > 1
            else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        cta_layout_vmnk: cute.Layout,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K.T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        # /////////////////////////////////////////////////////////////////////////////
        #  Set up before warp specialization
        # /////////////////////////////////////////////////////////////////////////////

        # --- Set up thread info ---

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = (
            bidx % cute.size(self.cta_group_shape)
            if const_expr(cute.size(self.cta_group_shape) > 1)
            else 0
        )
        is_leader_cta = mma_tile_coord_v == 0  # only CTA0 is the leader CTA

        # Used only for debug print
        # guarded by const_expr so zero overhead when debug_print=False
        is_print_block = const_expr(self.debug_print) and (
            (bidx == 0) and (bidy == 0) and (bidz == 0)
        )
        is_print_thread = const_expr(self.debug_print) and (
            (tidx == 127) and is_print_block
        )

        # --- Prefetch TMA descriptor ---

        if warp_idx == self.load_warp_ids[0]:  # only one warp is enough
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # --- Alloc smem storage and fetch ptrs ---

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline mbarrier ptrs
        mbar_load_Q = storage.mbar_load_Q.data_ptr()
        mbar_load_KV = storage.mbar_load_KV.data_ptr()
        mbar_S_full_P_full_O_rescaled = storage.mbar_S_full_P_full_O_rescaled.data_ptr()
        mbar_P_full_lastsplit = storage.mbar_P_full_lastsplit.data_ptr()
        mbar_O_full = storage.mbar_O_full.data_ptr()
        mbar_s0_s1_sequence = storage.mbar_s0_s1_sequence.data_ptr()
        mbar_softmax_stats = storage.mbar_softmax_stats.data_ptr()
        mbar_O_epi = storage.mbar_O_epi.data_ptr()

        # tmem buf/dealloc ptrs
        tmem_holding_buf_ptr = storage.tmem_holding_buf_ptr
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr

        # --- Alloc tmem alloc/dealloc barrier ---

        # NOTE: Only the mma warp drives tmem alloc/dealloc, and TmemAllocator internally
        # initializes the dealloc mbar only for the mma warp (covering both CTAs in a
        # 2-CTA cluster). This means the dealloc mbar alone cannot block until softmax
        # and correction warps finish using tmem. Therefore, all three warp groups
        # (mma + softmax + correction) must arrive on this shared barrier, giving the
        # mma warp a safe signal that tmem is no longer in use. And only by that point,
        # the mma warp (in 2-CTA cluster) can wait on dealloc mbar before it deallocates.

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
        tmem = cutlass.utils.TmemAllocator(
            alloc_result_dst_smem_ptr=tmem_holding_buf_ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )

        # --- Make pipeline cooperative groups ---

        load_warp = ThreadCooperativeGroup(len(self.load_warp_ids))
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        load_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.load_warp_ids)
        )
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )

        # NOTE: For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE
            * len(self.softmax0_warp_ids + self.correction_warp_ids)
            * self.cta_group_size
        )

        # --- Make pipelines ---

        # Load Q pipeline:
        #   producer: load warp
        #     acquire: pipeline_q.producer_acquire_w_index_phase(stage, phase)  [before TMA issue]
        #     commit:  TMA hardware arrive on mbar_load_Q                       [implicit]
        #     tail:    pipeline_q.producer_acquire_w_index_phase(stage, phase)  [drain at end]
        #   consumer: mma warp
        #     wait:    pipeline_q.consumer_wait_w_index_phase(stage, phase)     [prologue, per Q-stage]
        #     release: pipeline_q.consumer_release_w_index(stage)               [mma epilogue, per Q-stage]
        #   full  = sQ[stage] written by TMA
        #   empty = mma warp released after all KV-blocks for this Q-tile are done
        if const_expr(self.use_tma_Q):
            pipeline_q = ffa_pipeline.PipelineTmaUmma.create(
                barrier_storage=mbar_load_Q,
                num_stages=self.q_stage,
                producer_group=load_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["Q"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,  # sync later by our own
            )
        else:
            pipeline_q = ffa_pipeline.PipelineAsyncUmma.create(
                barrier_storage=mbar_load_Q,
                num_stages=self.q_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )

        # Load KV pipeline:
        #   producer: load warp
        #     acquire: pipeline_kv.producer_acquire(producer_state)             [before TMA issue]
        #     commit:  TMA hardware arrive on mbar_load_KV                      [implicit]
        #     tail:    pipeline_kv.producer_tail(kv_producer_state)             [drain at end]
        #   consumer: mma warp
        #     wait:    pipeline_kv.consumer_wait(mma_kv_consumer_state)         [before each QK/PV GEMM]
        #     release: pipeline_kv.consumer_release(mma_kv_consumer_state)      [after each GEMM done]
        #   full  = sK[stage]/sV[stage] written by TMA
        #   empty = mma warp finished QK GEMM (K) or PV GEMM (V)
        if const_expr(self.use_tma_KV):
            pipeline_kv = ffa_pipeline.PipelineTmaUmma.create(
                barrier_storage=mbar_load_KV,
                num_stages=self.kv_stage,
                producer_group=load_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_kv = pipeline.PipelineAsyncUmma.create(
                barrier_storage=mbar_load_KV,
                num_stages=self.kv_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )

        # S/P/O triple-state pipeline (MMA ↔ softmax+correction):
        #   This pipeline has dual semantics — it tracks two transitions per slot:
        #     (1) S-full:        MMA commits after QK GEMM;      softmax waits before T2R load S
        #     (2) P+O-empty:     softmax + correction both release; MMA waits before PV GEMM
        #   The consumer group is softmax_correction_threads_cluster (both warps), so the
        #   "empty" barrier requires arrivals from BOTH softmax (P written) AND correction (O rescaled).
        #   This is why softmax+correction are bundled together as consumers.
        #
        #   producer: mma warp
        #     commit:  pipeline_s_p_o.producer_commit_w_index(stage)              [after QK GEMM → S full]
        #     acquire: pipeline_s_p_o.producer_acquire_w_index_phase(stage, phase)[before PV GEMM → wait P+O]
        #     no tail: last acquire has no dangling mbar (acquire only blocks, no mbar to drain)
        #   consumer (softmax warp):
        #     wait:    pipeline_s_p_o.consumer_wait_w_index_phase(stage, phase)   [wait S full → T2R S]
        #     release: pipeline_s_p_o.consumer_release_w_index(stage)             [after R2T P done → P ready]
        #              (also early-release the 1st half of P when split_P_arrive > 0)
        #   consumer (correction warp):
        #     release: pipeline_s_p_o.consumer_release_w_index(stage)             [after rescale O → O ready]
        #              (in prologue: skip rescale, still release to unblock MMA)
        #              (in epilogue: after correction_epilogue)
        pipeline_s_p_o = ffa_pipeline.PipelineUmmaAsync.create(  # MMA(P) -> Async(C)
            barrier_storage=mbar_S_full_P_full_O_rescaled,
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # P-lastsplit pipeline (softmax → MMA hardware, split_P_arrive mode only):
        #   Used only when self.split_P_arrive > 0. In this mode the PV GEMM starts as soon as
        #   the 1st half of P is ready in tmem (instead of waiting for full P). The 2nd half
        #   arrival is handled by pipeline_p_lastsplit so the hardware GEMM knows when the full P
        #   is ready before reading the 2nd half.
        #
        #   producer: softmax warp
        #     commit:  pipeline_p_lastsplit.producer_commit_w_index(stage)        [after full R2T P done]
        #     (no acquire/tail: softmax never waits on this pipeline's empty side)
        #   consumer: mma hardware (PV GEMM)
        #     waits:   hardware uses mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage)
        #              passed into gemm_Pi(); UMMA waits on this mbar before issuing 2nd half read
        #     release: UMMA hardware releases implicitly on GEMM completion
        #   full  = P[stage] fully written to tmem (both halves)
        #   empty = PV GEMM consumed P (UMMA hardware done)

        pipeline_p_lastsplit = (
            ffa_pipeline.PipelineAsyncUmma.create(  # Async(P) -> MMA(C)
                barrier_storage=mbar_P_full_lastsplit,
                num_stages=self.q_stage,
                producer_group=softmax_warps_cluster,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        )

        # O-accumulation pipeline (MMA → correction, FINAL tile only):
        #   Unlike a typical ring-buffer pipeline, this is used ONLY for the last KV-block of each
        #   Q-tile. During the mainloop, correction does NOT need to wait for O because:
        #     by the time softmax signals (sm_stats_barrier) that S(i) is done,
        #     O(i-1) from PV GEMM must have also completed (GEMM ordering guarantee).
        #   Only in the epilogue (last KV-block) does the ordering break — softmax signals row_sum
        #   before the next tile's GEMM starts, so correction MUST explicitly wait for final O.
        #
        #   producer: mma warp
        #     commit:  pipeline_o_acc.producer_commit_w_index(stage)              [epilogue GEMM only]
        #     (no acquire: MMA never re-acquires O; no ring-buffer semantics)
        #     (no tail: no dangling acquire to drain)
        #   consumer: correction warp (epilogue only)
        #     wait:    pipeline_o_acc.consumer_wait_w_index_phase(stage, phase)   [before correction_epilogue]
        #     (no release: correction owns O until end of tile; MMA is already done)
        #   full  = final O[stage] written to tmem by last PV GEMM
        #   empty = (never explicitly released; O data consumed by correction_epilogue into smem)
        pipeline_o_acc = ffa_pipeline.PipelineUmmaAsync.create(
            barrier_storage=mbar_O_full,
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # Softmax-sequence pipeline (s0_s1_sequence):
        #   producer: softmax0 warpgroup signals after finishing exp2+type-convert for one KV-block
        #   consumer: softmax1 warpgroup waits before starting exp2 for the same KV-block
        #   full  = softmax0 finished exp2, softmax1 can start exp2
        #   empty = softmax1 finished exp2+R2T-store, softmax0 can proceed to next block
        #   NOTE: serializes exp2+R2T-store between softmax0 and softmax1 to avoid
        #         contention on the tmem write port (both write to different P0/P1 regions
        #         but share the same physical tmem write bandwidth)
        pipeline_s0_s1_sequence = None
        if const_expr(self.s0_s1_barrier and self.q_stage > 1):
            # This is not a typical producer-consumer pipeline. We will directly use
            # pipeline_s0_s1_sequence.sync_object_full and will not use
            # pipeline_s0_s1_sequence.sync_object_empty.
            pipeline_s0_s1_sequence = ffa_pipeline.PipelineAsync.create(
                barrier_storage=mbar_s0_s1_sequence,
                num_stages=2,
                producer_group=softmax_threads,
                consumer_group=softmax_threads,
                defer_sync=True,
            )

        # Softmax-stats pipeline (softmax → correction): WAR guard on the sScale slot.
        #   sScale[stage] is written by softmax (corr_scale in mainloop, final row_sum/row_max
        #   in epilogue) and read by correction. Two hazards exist on this shared slot:
        #     - RAW: correction must not read before softmax writes  → handled by sm_stats_barrier
        #     - WAR: softmax must not overwrite before correction reads the prev value → handled HERE
        #   (A single pipeline mbar could in principle cover both RAW+WAR via commit/wait +
        #    acquire/release; this design instead splits them — RAW on the NamedBarrier below,
        #    WAR on this pipeline.)
        #
        #   producer = softmax warp, consumer = correction warp, num_stages = q_stage
        #     producer_acquire (softmax):   block until correction released this slot
        #                                   → safe to overwrite sScale[stage] with the next value
        #     consumer_release (correction): slot has been read → free for softmax to reuse
        #
        #   The correction mainloop releases with a CROSS index (q_stage-1-stage) instead of stage,
        #   turning this WAR backpressure into round-robin traffic control: a single correction
        #   warp group serves two softmax warp groups (one per q_stage), staggering them so only
        #   one softmax wg overlaps with correction at a time while the other parks on its acquire.
        pipeline_sm_stats = ffa_pipeline.PipelineAsync.create(
            barrier_storage=mbar_softmax_stats,
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )

        # sm_stats NamedBarrier (softmax → correction): RAW guard on the sScale slot.
        #   A 2-warp rendezvous signalling "sScale[stage] has been written, safe to read".
        #   Used for BOTH corr_scale (mainloop) and final row_sum/row_max (epilogue).
        #   Pairs with pipeline_sm_stats above: this barrier = RAW (data ready),
        #   pipeline_sm_stats = WAR (slot free to overwrite). Note the softmax-side arrive is
        #   non-blocking, so it gives no backpressure — all backpressure comes from the pipeline.
        #
        #   softmax warp (after writing sScale): arrive_w_index(stage * num_softmax_warps + warp_idx)
        #   correction warp (before reading):    arrive_and_wait_w_index(stage * num_corr_warps + warp_idx)
        sm_stats_barrier = ffa_pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )

        # O-epilogue pipeline (correction → epilogue warp, non-corr-epi mode only):
        #   Used only when use_correction_warps_for_epi=False (TMA store mode).
        #   In use_correction_warps_for_epi=True mode, correction warps do the S2G copy directly
        #   without going through a pipeline.
        #
        #   producer: correction warp
        #     acquire: pipeline_o_epi.producer_acquire_w_index_phase(stage, phase)
        #              [before correction_epilogue → waits for epilogue to drain prev sO]
        #     commit:  pipeline_o_epi.producer_commit_w_index(stage)
        #              [after correction_epilogue writes sO → sO ready for TMA store]
        #     tail:    pipeline_o_epi.producer_acquire_w_index_phase(q_stage-1, phase)
        #              [at end of correction_loop, drains the last epilogue consumer]
        #   consumer: epilogue warp
        #     wait:    pipeline_o_epi.consumer_wait_w_index_phase(stage, phase)   [before TMA store]
        #     release: pipeline_o_epi.consumer_release_w_index(stage)             [after TMA store done]
        #   full  = sO[stage] written by correction_epilogue, ready for TMA store
        #   empty = epilogue warp finished TMA store, sO slot can be reused by next correction_epilogue
        pipeline_o_epi = None
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = ffa_pipeline.PipelineAsync.create(
                barrier_storage=mbar_O_epi,
                num_stages=self.q_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        # --- Cluster arrive after mbarrier init ---

        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # --- Make smem tensors of sQ/sK/sV/sO/sScale ---

        # sQ: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_Q1,MMA_HD=(4,2),stageQ):((64,1),0,(16,8192),16384)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # sK: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_K1,MMA_HD=(4,2),stageK):((64,1),0,(16,4096),8192)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # sV: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_K1,MMA_HD=(4,2),stageK):((1,64),0,1024,8192)
        sV = cute.make_tensor(
            # Strip swizzle info to reuse smem
            cute.recast_ptr(sK.iterator, sV_layout.inner),
            sV_layout.outer,
        )
        # sO: S<3,4,3> o 0 o (EPI_Q=(8,16),EPI_HD=(64,2),EPI_STAGE=(1,2)):((64,512),(1,8192),(0,16384))
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype),
                sO_layout.outer,
            )

        # sScale: (stageQ*tileQ*2=512):(1)
        # each q stage x each q row has 2 scale slot:
        #   1. corr_scale for mainloop correction
        #   2. final row_sum/row_max for epilogue correction
        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.q_stage * self.m_block_size * 2)
        )

        # --- Make tmem fragments of tS/tP/tO ---

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        # NOTE: `make_fragment_C` returns a fake tensor with tmem col offset always at 0,
        # by right we need to explicitly retrieve tmem_ptr with `cute.arch.retrieve_tmem_ptr`.
        # But we know that we always request 512 columns of tmem, so we know that it must start at 0.

        # tStS: (MMA_tC=(128,128),1,1, S_STAGE2):((65536,1),0,0,128)
        qk_acc_shape = thr_mma_qk.partition_shape_C(  # (tileQ128*CTA2,tileK128)
            self.mma_tiler_qk[:2]
        )
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))

        # tOtO: (MMA_tC=(128,128),1,1,2):((65536,1),0,0,128)
        pv_acc_shape = thr_mma_pv.partition_shape_C(  # (tileQ128*CTA2,tileHD128)
            self.mma_tiler_pv[:2]
        )
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)

        # tOrP: (MMA_tA=(128,16),MMA_Q1,MMA_HD=(4,2),S_STAGE=(2)):((65536,1),0,(16,64),(256))
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)  # reuse tS for tP
        tOrP = thr_mma_pv.make_fragment_A(tP)[
            None, None, None, 0
        ]  # slice for stage dim, will expand later
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width  # 2 for bf16
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (  # 256 for bf16
            self.tmem_p_offset[1] - self.tmem_p_offset[0]  # 192-64=128
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        # --- Make other info dataclass ---

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0]
            if const_expr(not self.pack_gqa)
            else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks
                if blocksparse_tensors is not None
                else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )

        # --- Cluster wait before tensor memory alloc ---

        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # --- Make tile scheduler ---

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = ThreadCooperativeGroup(1)
            num_clc_consumer_warps_per_cta = self.threads_per_cta // cute.arch.WARP_SIZE
            # NB on CTA0 warp15 == scheduler on CTA1 == empty but still both consume
            num_clc_consumer_warps = (
                num_clc_consumer_warps_per_cta * self.cta_group_size
            )
            clc_pipeline_consumer_group = ThreadCooperativeGroup(
                cute.arch.WARP_SIZE * num_clc_consumer_warps
            )

            block_idx = cute.arch.block_idx()
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    block_idx,
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(
            tile_scheduler, TileSchedulerProtocol
        ), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm100_kernel_setup] "
            if is_print_thread:
                cute.printf("")
                cute.printf(prefix + "warp_id: {}, thread_id: {}", warp_idx, tidx)
                cute.printf(
                    prefix + "cta_id: {}, is_leader_cta: {}",
                    mma_tile_coord_v,
                    is_leader_cta,
                )
                cute.printf("")
                cute.printf("")
                cute.printf(prefix + "sQ: {}", sQ)
                cute.printf(prefix + "sK: {}", sK)
                cute.printf(prefix + "sV: {}", sV)
                cute.printf(prefix + "sO: {}", sO)
                cute.printf(prefix + "sScale: {}", sScale)
                cute.printf("")
                cute.printf(prefix + "tStS.layout (QK acc, tmem): {}", tStS.layout)
                cute.printf(prefix + "tOtO.layout (PV acc, tmem): {}", tOtO.layout)
                cute.printf("")
                cute.printf(prefix + "tP_width_ratio: {}", tP_width_ratio)
                cute.printf(prefix + "tP_stage_stride: {}", tP_stage_stride)
                cute.printf(prefix + "tP.layout: {}", tP.layout)
                cute.printf(prefix + "tOrP.layout: {}", tOrP.layout)
                cute.printf("")

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty / CLC Scheduler Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if (
                    warp_idx == self.empty_warp_ids[i]
                    and warp_idx != self.clc_scheduler_warp_id
                ):
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            # --- Decrease rmem usage ---

            cute.arch.setmaxregister_decrease(self.num_regs_other)

            # --- Enter load loop ---

            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                gmem_tiled_copy_Q,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
                is_print_block=is_print_block,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # --- Decrease rmem usage ---

            cute.arch.setmaxregister_decrease(self.num_regs_other)

            # --- Alloc and retrieve tmem buffer ---

            tmem.allocate(self.tmem_alloc_cols)  # alias for `cute.arch.alloc_tmem`
            tmem.wait_for_alloc()  # alias for `tmem_alloc_barrier.arrive_and_wait`
            tmem_ptr = tmem.retrieve_ptr(  # alias for `cute.arch.retrieve_tmem_ptr`
                self.qk_acc_dtype
            )

            # --- Enter mma loop ---

            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
                is_print_block=is_print_block,
            )

            # --- Dealloc tmem buffer ---

            tmem.relinquish_alloc_permit()  # alias for `cute.arch.relinquish_tmem_alloc_permit`
            tmem.wait_for_alloc()  # alias for `tmem_alloc_barrier.arrive_and_wait`
            tmem.free(
                tmem_ptr
            )  # alias for `deallc_mbar.arrive_wait + cute.arch.dealloc_tmem`

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if (
                warp_idx >= self.epilogue_warp_ids[0]
                and warp_idx <= self.epilogue_warp_ids[-1]
            ):
                # --- Decrease rmem usage ---

                cute.arch.setmaxregister_decrease(self.num_regs_other)

                # --- Enter epilogue loop ---

                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                    mma_tile_coord_v=mma_tile_coord_v,
                    is_print_block=is_print_block,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax WarpGroup 0/1
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]
        ) or (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1]):
            # --- Increase rmem usage ---

            cute.arch.setmaxregister_increase(self.num_regs_softmax)

            # --- Wait and retrieve tmem buffer ---

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

            # --- Enter softmax loop ---

            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                descale_tensors=descale_tensors,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                learnable_sink=learnable_sink,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                tile_scheduler=tile_scheduler,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                blocksparse_tensors=blocksparse_tensors,
                is_print_block=is_print_block,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(
                    0
                    if const_expr(self.q_stage == 1)
                    or warp_idx < self.softmax1_warp_ids[0]
                    else 1
                )
                softmax_loop(stage=stage, tStS=tStS)
            else:
                # NOTE: If there's s0_s1_barrier,
                # it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    softmax_loop(stage=0, tStS=tStS)
                if (
                    warp_idx < self.correction_warp_ids[0]
                    and warp_idx >= self.softmax1_warp_ids[0]
                ):
                    softmax_loop(stage=1, tStS=tStS)

            # --- Arrive mma warp's tmem dealloc ---

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction WarpGroup
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            # --- Decrease rmem usage ---

            cute.arch.setmaxregister_decrease(self.num_regs_correction)

            # --- Wait and retrieve tmem buffer ---

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

            # --- Enter correction loop ---

            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                learnable_sink,
                descale_tensors,
                gmem_tiled_copy_O,
                tma_atom_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                blocksparse_tensors=blocksparse_tensors,
                is_print_block=is_print_block,
            )

            # --- Arrive mma warp's tmem dealloc ---

            tmem_alloc_barrier.arrive()

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler: TileSchedulerProtocol,
        is_print_block: bool = False,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Make dummy CTA coord/layout since we do not use TMA multicast
        tma_cta_coord = 0
        tma_cta_layout = cute.make_layout(1)

        issue_kv_for_this_warp = (
            const_expr(not self.use_tma_KV or len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        )
        issue_q_for_this_warp = (
            const_expr(not self.use_tma_Q or len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        )

        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead
                if const_expr(not self.pack_gqa)
                else head_idx
            )
            seqlen_info = SeqlenInfoCls(batch_idx)

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 0)
                and is_print_block
                and (m_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            # //////////////////////////////////////////////
            #  Make gQ/gK/gV
            # //////////////////////////////////////////////

            # mQ_cur: (seqQ,HD):(1@1,1@0)
            # mK_cur: (seqK,HD):(1@1,1@0)
            # mV_cur: (HD,seqK):(1@0,1@1) => actually V.T
            mQ_cur = seqlen_info.offset_batch_Q(mQ, batch_idx, dim=3)[
                None, None, head_idx
            ]
            mK_cur, mV_cur = None, None
            if const_expr(mPageTable is None):
                if const_expr(not seqlen_info.has_cu_seqlens_k):
                    mK_cur, mV_cur = [
                        t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)
                    ]
                else:
                    mK_cur = cute.domain_offset(
                        (seqlen_info.offset_k, 0), mK[None, None, head_idx_kv]
                    )
                    mV_cur = cute.domain_offset(
                        (0, seqlen_info.offset_k), mV[None, None, head_idx_kv]
                    )
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
                )
            else:  # TODO: review the logics
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )

            # //////////////////////////////////////////////
            #  TMA Partition gQ/gK/gV and
            #  define G2S-load fn for sQ/sK/sV
            # //////////////////////////////////////////////

            # gQ: (tileQ128*CTA2,tileHD128,stageQ):(1@1,1@0,256@1)
            # gK: (tileK128,tileHD128,restK):(1@1,1@0,128@1)
            # gV: (tileHD128,tileK128,restK):(1@0,1@1,128@1) => actually V.T
            # where: restK = seqK // tileK
            if const_expr(self.use_tma_Q):
                # gQ: (tileQ128*CTA2*stageQ, tileHD) -> (tileQ128*CTA2,tileHD,stageQ)
                tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
                gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))
                gQ = layout_utils.select(
                    cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
                )

                # tSgQ: (MMA_sA=(128,16),MMA_Q1,MMA_HD8,stageQ):((1@1,1@0),0,16@0,256@1)
                tSgQ = thr_mma_qk.partition_A(gQ)

                # tQgQ: (TMA_ATOM_GMEM=((64,128),2),stageQ):(((1@0,1@1),64@0),256@1)
                # tQsQ: (TMA_ATOM_SMEM=(8192,2),stageQ):((1,8192),16384)
                load_Q_fn, tQsQ, tQgQ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, tma_cta_coord, tma_cta_layout, tSgQ, sQ
                )
                load_Q = partial(
                    self.load_Q,
                    load_Q_fn,
                    pipeline_q=pipeline_q,
                    phase=q_producer_phase,
                )
            else:
                assert gmem_tiled_copy_Q is not None
                load_Q = partial(
                    self.load_Q_non_tma,
                    mQ_cur,
                    sQ,
                    gmem_tiled_copy_Q,
                    pipeline_q,
                    tidx,
                    seqlen_info.seqlen_q,
                    m_block,
                    phase=q_producer_phase,
                )

            # tSgK: (MMA_sB=(64,16),MMA_K1,MMA_HD8,restK):((1@1,1@0),0,16@0,128@1)
            # tOgV: (MMA_sB=(64,16),MMA_K1,MMA_HD8,restK):((1@0,1@1),0,16@1,128@1)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)

            if const_expr(self.use_tma_KV):
                # tKgK: (TMA_ATOM_GMEM=((64,64),2),restK):(((1@0,1@1),64@0),128@1)
                # tKsK: (TMA_ATOM_SMEM=(4096,2),stageK):((1,4096),8192)
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    tma_cta_coord,
                    tma_cta_layout,
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                # tVgV: (TMA_ATOM_GMEM=((64,128),1),restK):(((1@0,1@1),0),128@1)
                # tVsV: (TMA_ATOM_SMEM=(8192,1),stageK):((1,0),8192)
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    tma_cta_coord,
                    tma_cta_layout,
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                paged_kv_manager = None
            else:  # TODO: review the logics
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmodDivisor(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen_info.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
            )

            # --- Debug print ---

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[fwd_sm100_load] "
                    cute.printf("")
                    cute.printf(
                        prefix + "m_block={} head_idx={} batch_idx={} split_idx={}",
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                    )
                    cute.printf(prefix + "mQ_cur.layout: {}", mQ_cur.layout)
                    cute.printf(prefix + "mK_cur.layout: {}", mK_cur.layout)
                    cute.printf(prefix + "mV_cur.layout: {}", mV_cur.layout)
                    cute.printf("")
                    if const_expr(self.use_tma_Q):
                        # tiler_gQ
                        cute.printf(prefix + "tiler_gQ: {}", tiler_gQ)
                        cute.printf(prefix + "gQ.layout: {}", gQ.layout)
                    cute.printf(prefix + "gK.layout: {}", gK.layout)
                    cute.printf(prefix + "gV.layout: {}", gV.layout)
                    cute.printf("")
                    cute.printf(prefix + "tSgK.layout: {}", tSgK.layout)
                    cute.printf(prefix + "tOgV.layout: {}", tOgV.layout)
                    if const_expr(self.use_tma_Q):
                        cute.printf(prefix + "tSgQ.layout: {}", tSgQ.layout)
                        cute.printf(prefix + "tQgQ.layout: {}", tQgQ.layout)
                        cute.printf(prefix + "tQsQ.layout: {}", tQsQ.layout)
                    if const_expr(self.use_tma_KV):
                        cute.printf(prefix + "tKgK.layout: {}", tKgK.layout)
                        cute.printf(prefix + "tKsK.layout: {}", tKsK.layout)
                        cute.printf(prefix + "tVgV.layout: {}", tVgV.layout)
                        cute.printf(prefix + "tVsV.layout: {}", tVsV.layout)
                    cute.printf("")

            # //////////////////////////////////////////////
            #  G2S-load sQ/sK/sV
            # //////////////////////////////////////////////

            if const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen_info, m_block, split_idx, num_splits
                )
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0

                    page_idx = (
                        mPageTable[batch_idx, n_block_first]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)

                    # --- Prologue: load Q0,Q1,K0,V0 ---

                    # Load K0
                    if issue_kv_for_this_warp:
                        load_K(
                            block=n_block_max - 1,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                            is_print_thread_and_tile=is_print_thread_and_tile,
                        )

                    # Load Q0
                    if issue_q_for_this_warp:
                        load_Q(block=0, stage=0)
                    if issue_kv_for_this_warp:
                        kv_producer_state.advance()

                    # Load Q1
                    if const_expr(self.q_stage == 2) and issue_q_for_this_warp:
                        load_Q(block=1, stage=1)

                    q_producer_phase ^= 1

                    # Load V0
                    if issue_kv_for_this_warp:
                        load_V(
                            block=n_block_max - 1,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                            is_print_thread_and_tile=is_print_thread_and_tile,
                        )
                        kv_producer_state.advance()

                    # --- Mainloop: load Ki,Vi ---

                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)

                        # Load Ki/Vi
                        if issue_kv_for_this_warp:
                            load_K(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
                            load_V(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
            else:  # block sparse load (TODO: review the logics)
                (
                    kv_producer_state,
                    q_producer_phase,
                ) = produce_block_sparse_inner_iters_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen_info,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

        if issue_kv_for_this_warp:
            pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail for the TMA-Q producer warp.
        if issue_q_for_this_warp:
            pipeline_q.producer_acquire_w_index_phase(
                self.q_stage - 1, q_producer_phase
            )

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: ffa_pipeline.PipelineUmmaAsync,
        pipeline_p_lastsplit: ffa_pipeline.PipelineAsyncUmma,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler: TileSchedulerProtocol,
        is_print_block: bool = False,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Set up GEMM fragments, desc and handler
        # /////////////////////////////////////////////////////////////////////////////

        # tSrQ: (MMA_ATOM1,MMA_Q1,MMA_HD=(4,2),stageQ):(0,0,(2,1024),2048)
        # tSrK: (MMA_ATOM1,MMA_K1,MMA_HD=(4,2),stageK):(0,0,(2,512),1024)
        # tOrV: (MMA_ATOM1,MMA_K1,MMA_HD8,stageK):(0,0,128,1024)
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        # --- Precompute PTX smem descriptors and issue descriptors for UMMA ---
        # NOTE: Why not just call cute.gemm() directly?
        #
        # SM100 tcgen05.mma requires four operands:
        #   [tmem_acc]  : tmem column address for accumulator (dynamic, changes per tile)
        #   smem_desc_a : 64-bit = base (static: swizzle/stride) | start_addr (dynamic: 16B granule)
        #   smem_desc_b : same structure for B
        #   idesc       : 32-bit instruction descriptor (MMA shape/dtype, fully static)
        #
        # cute.gemm() recomputes the full descriptor (base | start_addr) on every call.
        #
        # But for GEMM_QK, Q's base never changes across KV-blocks, and K's base never changes either —
        # only their start_addr portions differ by a fixed per-stage stride.
        #
        # By pre-injecting the static parts into named PTX register variables once,
        # each gemm_Si call reduces to a single add.s32 on the start_addr bits
        # instead of rebuilding the full 64-bit descriptor from scratch.

        # Extract the raw MmaOp objects (encode shape/dtype/source for the PTX instruction).
        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        # kind string used in "tcgen05.mma.kind::f16 / bf16 / ..." PTX syntax.
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)

        # Compute the static base portion of the smem descriptor for Q and K.
        # base encodes: layout_type (swizzle pattern), leading_byte_offset, stride_byte_offset.
        # This is purely compile-time and does NOT change across stages or KV-blocks.
        q_smem_base = sm100_utils.smem_desc_base_from_tensor(sQ, sm100_utils.Major.K)
        k_smem_base = sm100_utils.smem_desc_base_from_tensor(sK, sm100_utils.Major.K)

        # Compute the dynamic start_addr portion for each Q stage.
        # start_addr = (smem_byte_offset & 0x3FFFF) >> 4  (14-bit, 16-byte granule).
        q_smem_start = [
            sm100_utils.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator)
            for stage in range(self.q_stage)
        ]

        # Inject Q smem descriptor into a named PTX register variable.
        # Initialized to the LAST stage's address; gemm_Si will slide it forward/backward
        # via add.s32 to reach each stage without recomputing from scratch.
        # Also injects idesc constants for QK and PV MMA ops as named PTX register variables.
        sm100_utils.declare_ptx_smem_desc(
            q_smem_start[self.q_stage - 1],
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="fa_fwd_q_smem_desc",
        )
        # Inject "fa_fwd_qk_mma_idesc" and "fa_fwd_pv_mma_idesc" as compile-time constants
        # in PTX registers — referenced by name in every tcgen05.mma PTX call below.
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        # Stage stride for Q in descriptor units (16-byte granules).
        # Used by gemm_Si to slide fa_fwd_q_smem_desc between stage 0 and stage 1:
        #   gemm_Si[0]: desc += -sQ_stage_stride  (stage1_addr -> stage0_addr)
        #   gemm_Si[1]: desc +=  sQ_stage_stride  (stage0_addr -> stage1_addr)
        # With q_stage==1 there is only one stage so no sliding is needed.
        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_stage == 1):
            sQ_stage_stride = 0

        # gemm_Si[stage](smem_desc_start_b=...) issues:
        #   tcgen05.mma [tmem_s_offset[stage]], fa_fwd_q_smem_desc_*, smem_desc_b, fa_fwd_qk_mma_idesc
        # The Q descriptor register is updated in-place (sliding by smem_offset) before
        # the instruction fires, so the register always reflects the correct stage address
        # for the next call.
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fa_fwd_q_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                kind=qk_mma_kind,
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]

        # gemm_Pi[stage](tCrB=tOrVi, sB=sV_cur, zero_init=...) issues:
        #   tcgen05.mma [tmem_o_offset[stage]], [tmem_p_offset[stage]], smem_desc_v, fa_fwd_pv_mma_idesc
        # A operand (P) comes from tmem (OperandSource.TMEM), so no smem_desc_a is needed;
        # B operand (V) comes from smem, its descriptor is computed on the fly from sV_cur.
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=None,
                split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):  # TODO: review the logics
                block_iter_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen_info,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen_info, m_block, split_idx, num_splits
                )
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            # --- Debug print ---

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE == 0)
                and is_print_block
                and (m_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[fwd_sm100_mma] "
                    cute.printf("")
                    cute.printf(
                        prefix + "m_block={} head_idx={} batch_idx={} split_idx={}",
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                    )
                    cute.printf(
                        prefix + "block_iter_count={} process_tile={}",
                        block_iter_count,
                        process_tile,
                    )
                    if const_expr(not self.use_block_sparsity):
                        cute.printf(
                            prefix + "n_block_min={} n_block_max={}",
                            n_block_min,
                            n_block_max,
                        )
                    cute.printf("")
                    cute.printf(prefix + "tSrQ.layout: {}", tSrQ.layout)
                    cute.printf(prefix + "tSrK.layout: {}", tSrK.layout)
                    cute.printf(prefix + "tOrV.layout: {}", tOrV.layout)
                    cute.printf(prefix + "tOrP.layout: {}", tOrP.layout)
                    cute.printf("")
                    cute.printf(
                        prefix + "q_smem_base={} k_smem_base={}",
                        q_smem_base,
                        k_smem_base,
                    )
                    cute.printf(
                        prefix + "q_smem_start[0]={} q_smem_start[1]={}",
                        q_smem_start[0],
                        q_smem_start[1],
                    )
                    cute.printf(prefix + "sQ_stage_stride={}", sQ_stage_stride)
                    cute.printf("")

            # NOTE: Only the mma warp of the leader CTA actually issues UMMA
            if process_tile and is_leader_cta:
                # //////////////////////////////////////////////
                #  Prologue: GEMM Q0/Q1K0
                # //////////////////////////////////////////////

                # Double Q/S stages
                for stage in cutlass.range_constexpr(self.q_stage):
                    # --- GEMM: S0/S1(0) = Q0/Q1 * K0 ---

                    # Wait for Q0/Q1 to be full
                    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)

                    # Wait for K0 to be full before Q0K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index, Ki_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )

                    # NOTE: For the first iteration,
                    # we we're guaranteed S0/S1 are empty thus no need to acquire.
                    # For subsequent iterations, the wait happened at the end of the while loop.

                    # Issue UMMA for S0/S1(0)
                    sK_cur = sK[None, None, None, Ki_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                    gemm_Si[stage](
                        smem_desc_start_b=sm100_utils.make_smem_desc_start_addr(
                            sK_cur.iterator
                        )
                    )

                    # Commit S0/S1(0) to be full
                    pipeline_s_p_o.producer_commit_w_index(stage)

                # Flip the Q double buffer phase for mainloop
                mma_q_consumer_phase ^= 1

                # Release K0 to be empty
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # NOTE: Q0/Q1 are still needed in the mainloop
                # so we don't release them until epilogue.

                # //////////////////////////////////////////////
                #  Mainloop: GEMM P0/P1(i-1)V(i-1), GEMM Q0/Q1Ki
                # //////////////////////////////////////////////

                # NOTE: O hasn't been accumulated yet,
                # so its first MMA calculation doesn't need to accumulate
                O_should_accumulate = False
                for i in cutlass.range(1, block_iter_count, unroll=1):
                    # Wait for V(i-1) to be full
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, Vi_index]

                    # Double Q/S/P/O stages
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # --- GEMM: O0/O1(i-1) = P0/P1(i-1) * V(i-1) ---

                        # Acquire O0/O1(i-1) to be empty
                        # and wait for P0/P1(i-1) to be full
                        #
                        # NOTE: For the first iteration (i==1) in this work tile,
                        # acquiring O0/O1(i-1) means that the correction warps has finished
                        # reading tO during the last iteration of the previous work tile.
                        pipeline_s_p_o.producer_acquire_w_index_phase(
                            stage, P_full_O_rescaled_phase
                        )

                        # Issue UMMA for O0/O1(i-1)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(
                                stage
                            )
                            if self.split_P_arrive > 0
                            else None,
                            mbar_phase=P_full_O_rescaled_phase,
                        )

                        # NOTE: Don't need to commit O0/O1(i-1) to be full
                        # since the correction warps wait for the softmax warps anyway.
                        # By the time the softmax warps finished,
                        # S0/S1(i) for the next iteration must have been done,
                        # so O0/O1(i-1) must have been done as well.
                        #
                        # pipeline_o_acc.producer_commit_w_index(stage)

                        # Release V(i-1) to be empty after P1V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()

                        # --- GEMM: S0/S1(i) = Q0/Q1 * Ki ---

                        # Wait for Ki to be full before Q0Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )

                        # Issue UMMA for S0/S1(i)
                        #
                        # NOTE: Don't need to acquire S0/S1(i) to be empty
                        # since this UMMA is scheduled after the PV one,
                        # so P0/P1(i-1) is full => PV UMMA => S0/S1(i) is empty
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_utils.make_smem_desc_start_addr(
                                sK_cur.iterator
                            )
                        )

                        # Commit S0/S1(i) to be full
                        pipeline_s_p_o.producer_commit_w_index(stage)

                    # Release Ki to be empty
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

                    # Flip the P/O double buffer phase for the K/V tile
                    P_full_O_rescaled_phase ^= 1

                    # After O0/O1(0), we need to accumulate O0/O1
                    # for the subsequent iterations in this work tile
                    O_should_accumulate = True

                # //////////////////////////////////////////////
                # Epilogue: GEMM P0/P1(-1)V(-1)
                # //////////////////////////////////////////////

                # Release Q0/Q1 to be empty
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)

                # --- GEMM: O0/O1(-1) = P0/P1(-1) * V(-1) ---

                # Wait for V(-1) to be full
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = (
                    mma_kv_consumer_state.index,
                    mma_kv_consumer_state.phase,
                )
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # Acquire O0/O1(-1) to be empty
                    # and wait for P0/P1(-1) to be full
                    pipeline_s_p_o.producer_acquire_w_index_phase(
                        stage, P_full_O_rescaled_phase
                    )

                    # Issue UMMA for O0/O1(-1)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(
                            stage
                        )
                        if self.split_P_arrive > 0
                        else None,
                        mbar_phase=P_full_O_rescaled_phase,
                    )

                    # Commit O0/O1(-1) to be full
                    #
                    # NOTE: We do need commit here since for the last tile,
                    # by the time the softmax warp has signaled to the correction warps,
                    # the softmax warp has just finished computing the row sum of the current tile.
                    # It does not guarantee that the first tile of the next work tile has been computed yet.
                    pipeline_o_acc.producer_commit_w_index(stage)

                # Flip the P/O double buffer phase for next Q tile
                P_full_O_rescaled_phase ^= 1

                # Release V(-1) to be empty
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

        # NOTE:
        # 1. We don't need to call `pipeline_s_p_o.producer_tail()`
        # since there's no dangling mbarrier at the end of
        # `pipeline_s_p_o.producer_acquire_w_index_phase()`
        #
        # 2. We don't need `pipeline_o_acc.producer_tail()`
        # since we don't call `pipeline_o_acc.producer_acquire()` inside the loop.

    @cute.jit
    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        """Map query-head tile index -> KV-head index (FA3 descale semantics)."""
        if const_expr(self.pack_gqa):
            return head_idx
        return head_idx // self.qhead_per_kvhead

    @cute.jit
    def _load_effective_descales(
        self,
        descale_tensors: Optional[DescaleTensors],
        batch_idx: Int32,
        kv_head_idx: Int32,
    ) -> Tuple[Float32, Float32]:
        """Load effective QK and V descales, defaulting unspecified tensors to identity."""
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(descale_tensors is not None):
            if const_expr(descale_tensors.q_descale is not None):
                qk_descale = qk_descale * Float32(
                    descale_tensors.q_descale[batch_idx, kv_head_idx]
                )
            if const_expr(descale_tensors.k_descale is not None):
                qk_descale = qk_descale * Float32(
                    descale_tensors.k_descale[batch_idx, kv_head_idx]
                )
            if const_expr(descale_tensors.v_descale is not None):
                v_descale = Float32(descale_tensors.v_descale[batch_idx, kv_head_idx])
        return qk_descale, v_descale

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        descale_tensors: Optional[DescaleTensors],
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: ffa_pipeline.PipelineUmmaAsync,
        pipeline_p_lastsplit: ffa_pipeline.PipelineAsyncUmma,
        pipeline_sm_stats: ffa_pipeline.PipelineAsync,
        sm_stats_barrier: ffa_pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[ffa_pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        AttentionMaskCls: Callable[..., AttentionMask],
        tile_scheduler: TileSchedulerProtocol,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        is_print_block: bool = False,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        num_softmax_warps = (
            len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
        )
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * num_softmax_warps)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % num_softmax_warps

        # /////////////////////////////////////////////////////////////////////////////
        #  Set up tmem tensors and T2R/R2T tiled copy for S/P/Scale
        # /////////////////////////////////////////////////////////////////////////////

        # --- Make tmem (coord) tensor of tS/tP ---

        # tSAcc: (tileQ128,tileK128):(65536,1)
        # tStScale: (tileQ128,scale1):(65536,0)
        # tScS: (tileQ128,tileK128):(1@0,1@1)
        tSAcc = tStS[(None, None), 0, 0, stage]
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)

        # tStP: (tileQ128,tileKFP32View64):(65536,1)
        tileKFP32View = (  # tileK // 2
            self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        )
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tileKFP32View))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        # --- Make T2R tiled copy for S ---

        # T2R copy atom of `tcgen05.ld.sync.aligned.32x32b.x32`
        # layout_src_tv=(32,1024):(0,1) => (row32,col32) cells in tmem per warp
        # layout_dst_tv=(32,32):(32,1) => 32 fp32 elems in rmem per thread
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )

        # T2R tiled copy atom:
        # layout_src_tv_tiled=((32,4),((32,32),1)):((0,1),((128,4),0))
        #   => 4 x (row32,col32) cells in tmem per warp group
        # layout_dst_tv_tiled=((32,4),(32,1)):((4,1),(128,0))
        #   => still 32 fp32 elems in rmem per thread, but tiled in a warp group
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)

        # tStS_t2r: (T2R_CPY_ATOM=((col32,row32),1),CPY_Q1,CPY_K4):(((1,65536),0),0,32)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)

        # --- Make R2T tiled copy for Scale ---

        # R2T copy atom of `tcgen05.st.sync.aligned.32x32b.x1`
        # layout_src_tv=(32,1):(1,1) => 1 fp32 elem in rmem per thread
        # layout_dst_tv=(32,32):(0,1) => (row32,col1) cells in tmem per warp
        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )

        # R2T tiled copy atom:
        # layout_src_tv_tiled=((32,4),(1,1)):((4,1),(0,0))
        #   => still 1 fp32 elem in rmem per thread, but tiled in a warp group
        # layout_dst_tv_tiled=((32,4),(32,1)):((0,1),(4,0))
        #   => 4 x (row32,col1) cells in tmem per warp group
        thr_tmem_store_scale = tcgen05.make_tmem_copy(
            tmem_store_scale_atom, tStScale
        ).get_slice(tidx)

        # tStScale_r2t: (T2R_CPY_ATOM=(row32,col1),CPY_Q1,CPY_scale1):((65536,0),0,0)
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)

        # --- Make R2T tiled copy for P ---

        # R2T copy atom of `tcgen05.st.sync.aligned.32x32b.x16`
        # layout_src_tv=(32,16):(16,1) => 16 fp32 elems in rmem per thread
        # layout_dst_tv=(32,512):(0,1) => (row32,col16) cells in tmem per warp
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(
                    8 if const_expr(self.q_dtype.width == 8) else 16
                )
            ),
            Float32,
        )

        # R2T tiled copy atom:
        # layout_src_tv_tiled=((32,4),(16,1)):((4,1),(128,0))
        #   => still 16 fp32 elems in rmem per thread, but tiled in a warp group
        # layout_dst_tv_tiled=((32,4),((16,32),1)):((0,1),((128,4),0))
        #   => 4 x (row32,col16) cells in tmem per warp group
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)

        # tStP_r2t: (T2R_CPY_ATOM=((col16,row32),1),CPY_Q1,CPY_KFP32View4):(((1,65536),0),0,16)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen_info = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen_info, m_block, split_idx, num_splits
            )

            mask = AttentionMaskCls(seqlen_info)
            shared_mask_kwargs = dict(
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=aux_tensors,
            )

            # --- Recompute fastdiv_mods if necessary ---

            recompute_fastdiv_mods_q = const_expr(
                aux_tensors is not None
                and (seqlen_info.has_cu_seqlens_q or seqlen_info.has_seqused_q)
            )
            recompute_fastdiv_mods_k = const_expr(
                aux_tensors is not None
                and (seqlen_info.has_cu_seqlens_k or seqlen_info.has_seqused_k)
            )

            if const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen_info.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen_info.seqlen_k),
                )

            # --- Define attn mask apply fn ---

            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            # --- Compute sm_scale factor ---

            qk_descale, _ = self._load_effective_descales(
                descale_tensors, batch_idx, kv_head_idx
            )

            max_offset = 8 if const_expr(self.q_dtype.width == 8) else 0
            if const_expr(self.score_mod is None):
                softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
                softmax_scale_eff = None
            else:
                softmax_scale_log2_eff = softmax_scale_log2
                softmax_scale_eff = softmax_scale * qk_descale

            rescale_threshold = (
                8.0
                if const_expr(self.q_dtype.width == 16)
                else 4.0
                if const_expr(self.q_dtype.width == 8)
                else 0.0
            )

            # --- Define softmax handler ---

            softmax = SoftmaxSm100.create(
                softmax_scale_log2_eff,
                rescale_threshold=rescale_threshold,
                softmax_scale=softmax_scale_eff,
                max_offset=max_offset,
            )
            softmax.reset()

            # --- Determine tile counts ---

            if const_expr(self.use_block_sparsity):  # TODO: review the logics
                tile_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen_info,
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(
                    0
                )

            # --- Define softmax step fn ---

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                seqlen=seqlen_info,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
            )

            # --- Debug print ---

            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 0)
                and (stage == 0)
                and is_print_block
                and (m_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )
            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[fwd_sm100_softmax] "
                    cute.printf("")
                    cute.printf(
                        prefix
                        + "stage={} m_block={} head_idx={} batch_idx={} split_idx={}",
                        stage,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                    )
                    cute.printf(
                        prefix
                        + "n_block_min={} n_block_max={} tile_block_count={} has_work={}",
                        n_block_min,
                        n_block_max,
                        tile_block_count,
                        has_work,
                    )
                    cute.printf(
                        prefix + "softmax_scale_log2_eff={} tileKFP32View={}",
                        softmax_scale_log2_eff,
                        tileKFP32View,
                    )
                    cute.printf("")
                    cute.printf(
                        prefix + "tmem_load_atom: layout_src_tv={} layout_dst_tv={}",
                        tmem_load_atom.layout_src_tv,
                        tmem_load_atom.layout_dst_tv,
                    )
                    cute.printf(
                        prefix
                        + "thr_tmem_load: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                        thr_tmem_load.layout_src_tv_tiled,
                        thr_tmem_load.layout_dst_tv_tiled,
                    )
                    cute.printf("")
                    cute.printf(
                        prefix
                        + "tmem_store_scale_atom: layout_src_tv={} layout_dst_tv={}",
                        tmem_store_scale_atom.layout_src_tv,
                        tmem_store_scale_atom.layout_dst_tv,
                    )
                    cute.printf(
                        prefix
                        + "thr_tmem_store_scale: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                        thr_tmem_store_scale.layout_src_tv_tiled,
                        thr_tmem_store_scale.layout_dst_tv_tiled,
                    )
                    cute.printf("")
                    cute.printf(
                        prefix + "tmem_store_atom: layout_src_tv={} layout_dst_tv={}",
                        tmem_store_atom.layout_src_tv,
                        tmem_store_atom.layout_dst_tv,
                    )
                    cute.printf(
                        prefix
                        + "thr_tmem_store: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                        thr_tmem_store.layout_src_tv_tiled,
                        thr_tmem_store.layout_dst_tv_tiled,
                    )
                    cute.printf("")
                    cute.printf(prefix + "tSAcc.layout: {}", tSAcc.layout)
                    cute.printf(prefix + "tStScale.layout: {}", tStScale.layout)
                    cute.printf(prefix + "tScS.layout: {}", tScS.layout)
                    cute.printf(prefix + "tStP.layout: {}", tStP.layout)
                    cute.printf(prefix + "tStS_t2r.layout: {}", tStS_t2r.layout)
                    cute.printf(prefix + "tStP_r2t.layout: {}", tStP_r2t.layout)
                    cute.printf(prefix + "tStScale_r2t.layout: {}", tStScale_r2t.layout)
                    cute.printf("")

            if const_expr(self.use_block_sparsity) or has_work:
                # WAR acquire (per-tile): gate this whole Q tile on the slot being free,
                # i.e. correction has finished reading the previous tile's row_sum from sScale[stage].
                pipeline_sm_stats.producer_acquire_w_index_phase(
                    stage, sm_stats_producer_phase
                )
                sm_stats_producer_phase ^= 1

            # --- Block sparse or dense softmax loop ---

            if const_expr(self.use_block_sparsity):  # TODO: review the logics
                # When aux_tensors exist, Q indices beyond seqlen_q must be wrapped to avoid
                # OOB aux_tensor access. Only edge tiles (where m_tile_end > seqlen_q) need this.
                if const_expr(aux_tensors is not None):
                    m_tile_end = (
                        (self.q_stage * m_block + stage + 1) * self.cta_group_size
                    ) * self.m_block_size
                    check_m_boundary = m_tile_end > seqlen_info.seqlen_q
                else:
                    check_m_boundary = False
                (
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen_info,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    self.q_stage,
                    Int32(stage),
                    check_m_boundary,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                if not empty_tile:
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
            else:
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    # --- Prologue: S0/S1(0) ---

                    (
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                    ) = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block=n_block_max - 1,
                        is_first=True,  # we don't need to correct for first KV tile
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                        is_print_thread_and_tile=is_print_thread_and_tile,
                    )
                    n_block_max -= 1

                    # --- Mainloop-1: S0/S1 with causal masking ---
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = (
                            block_info.get_n_block_min_causal_local_mask(
                                seqlen_info, m_block, n_block_min
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )

                    # --- Mainloop-2: S0/S1 w/o masking ---
                    # NOTE: The remaining iterations have no masking, but may still need mask_mod
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen_info, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        n_block = n_block_max - n_tile - 1
                        if const_expr(
                            self.mask_mod is not None
                        ):  # TODO: review the logics
                            (
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block=n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        else:
                            (
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block=n_block,
                            )

                    # --- Mainloop-3: S0/S1 with local masking on the left ---
                    if const_expr(  # TODO: review the logics
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            0, n_block_max - n_block_min, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block=n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )

                    # --- Epilogue: Copy final row_sum/row_max to sScale ---

                    # R2S copy final row_sum/row_max to sScale
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]

                    # Arrive final row_sum/row_max to be full
                    # to notify the correction warp group
                    # for final O row-sum normalization
                    sm_stats_barrier.arrive_w_index(
                        index=stage * num_softmax_warps + warp_idx
                    )

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

        # WAR acquire (tail): equivalent to pipeline_sm_stats.producer_tail
        # to drain the final outstanding slot so softmax wg does not exit while corr wg still holds it.
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

        if const_expr(self.s0_s1_barrier):
            if stage == 0:
                # NOTE: This is equivalent to pipeline_s0_s1.producer_tail
                pipeline_s0_s1_sequence.sync_object_full.wait(
                    stage, s0_s1_sequence_phase
                )

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: ffa_pipeline.PipelineUmmaAsync,
        pipeline_p_lastsplit: ffa_pipeline.PipelineAsyncUmma,
        pipeline_sm_stats: ffa_pipeline.PipelineAsync,
        sm_stats_barrier: ffa_pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[ffa_pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        is_print_thread_and_tile: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        num_softmax_warps = (
            len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % num_softmax_warps

        # /////////////////////////////////////////////////////////////////////////////
        #  Set up rmem tensors for softmax
        # /////////////////////////////////////////////////////////////////////////////

        # tStS_t2r: (T2R_CPY_ATOM=((col32,row32),1),CPY_Q1,CPY_K4):(((1,65536),0),0,32)
        # tScS: (tileQ128,tileK128):(1@0,1@1)
        # tScS_t2r: (T2R_CPY_ATOM=(32,1),CPY_Q1,CPY_K4):((1@1,0),0,32@1)
        # tSrS_t2r: (T2R_CPY_ATOM=(32,1),CPY_Q1,CPY_K4):((1,0),0,32)
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        tScS_shape = (  # (tileQ128,tileK128)
            self.mma_tiler_qk[0] // self.cta_group_shape,
            self.mma_tiler_qk[1],
        )
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        tSrS_t2r = cute.make_rmem_tensor_like(tScS_t2r, self.qk_acc_dtype)

        # tStP_r2t: (R2T_CPY_ATOM=((col16,row32),1),CPY_Q1,CPY_KFP32View4):(((1,65536),0),0,16)
        # tScP: (tileQ128,tileKFP32View64):(1@0,1@1)
        # tScP_r2t: (R2T_CPY_ATOM=(16,1),CPY_Q1,CPY_KFP32View4):((1@1,0),0,16@1)
        # tSrP_r2t_f32: (R2T_CPY_ATOM=(16,1),CPY_Q1,CPY_KFP32View4):((1,0),0,16)
        # tSrP_r2t: (R2T_CPY_ATOM=(32,1),CPY_Q1,CPY_KFP32View4):((1,0),0,32)
        tileKFP32View = (  # tileK // 2
            self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        )
        tScP_shape = (tScS_shape[0], tileKFP32View)
        tScP = cute.make_identity_tensor(tScP_shape)
        tScP_r2t = thr_tmem_store.partition_S(tScP)
        tSrP_r2t_f32 = cute.make_rmem_tensor_like(tScP_r2t, Float32)
        tSrP_r2t = cute.make_tensor(  # bf16 output buffer for fp32 softmax
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm100_softmax_step] "
                cute.printf("")
                cute.printf(
                    prefix + "stage={} n_block={} is_first={}",
                    stage,
                    n_block,
                    is_first,
                )
                cute.printf("")
                cute.printf(prefix + "tStS_t2r.layout: {}", tStS_t2r.layout)
                cute.printf(prefix + "tScS.layout: {}", tScS.layout)
                cute.printf(prefix + "tScS_t2r.layout: {}", tScS_t2r.layout)
                cute.printf(prefix + "tSrS_t2r.layout: {}", tSrS_t2r.layout)
                cute.printf("")
                cute.printf(prefix + "tStP_r2t.layout: {}", tStP_r2t.layout)
                cute.printf(prefix + "tScP_r2t.layout: {}", tScP_r2t.layout)
                cute.printf(prefix + "tSrP_r2t_f32.layout: {}", tSrP_r2t_f32.layout)
                cute.printf(prefix + "tSrP_r2t.layout: {}", tSrP_r2t.layout)
                cute.printf("")

        # /////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # /////////////////////////////////////////////////////////////////////////////

        # --- T2R copy S ---

        # Wait for tSi to be full
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)

        # T2R copy from tSi to rSi
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        # --- Update row_max/corr_scale ---

        # Apply score_mod on rSi if needed
        if const_expr(self.score_mod is not None):  # TODO: review the logics
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
            )

        # Apply mask fn on rSi if needed
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)

        # Update row_max and corr_scale
        row_max, corr_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        # R2S copy corr_scale if not the first KV tile
        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = corr_scale

        # Arrive corr_scale to be full
        # to notify the correction warp group to correct O
        sm_stats_barrier.arrive_w_index(index=stage * num_softmax_warps + warp_idx)

        # --- Apply unnormalized softmax ---

        # Apply (rSi - row_max)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)

        # Inter-softmax sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)

        # Apply exp2((rSi - row_max)) and copy to rPi
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,  # bf16 view of rPi
            ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )

        # Inter-softmax sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)

        # --- R2T copy P ---

        # R2T copy rPi to tPi
        r2t_cpy_iter_count = cute.size(tStP_r2t.shape[2])  # CPY_KFP32View4
        for i in cutlass.range_constexpr(r2t_cpy_iter_count):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )

            # Release 1st half tSi to be empty => 1st half tPi to be full
            # to notify mma warp that the 1st half of tPi is ready
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    r2t_cpy_iter_count * self.split_P_arrive // self.n_block_size
                )
                if const_expr(split_P_arrive_idx == i + 1):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)

        # Release tSi to be empty / Commit (2nd half) tPi to be full
        # to notify mma warp that (2nd half) tPi is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)

        # --- Backpressure ---

        # WAR acquire: before the next write to sScale[stage] (next step's corr_scale, or the
        # tile-end row_sum), wait until correction wg has read the value just published above.
        # NOTE: With the correction wg mainloop's cross-release between two stages,
        # this also staggers the two softmax wgs, and allows current stage of softmax computation
        # to overlap with its corresponding correction of O.
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

        # Update row_sum with corr_scale in rmem
        # REVIEW: why not update row_sum before acquiring the barrier above ?
        softmax.update_row_sum(tSrS_t2r.load(), corr_scale, is_first)

        # Flip phases for the next KV tile
        return (
            mma_si_consumer_phase ^ 1,
            sm_stats_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: ffa_pipeline.PipelineAsync,
        sm_stats_barrier: ffa_pipeline.NamedBarrier,
        pipeline_o_epi: Optional[ffa_pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        is_print_block: bool = False,
    ):
        num_corr_warps = len(self.correction_warp_ids)
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * num_corr_warps)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % num_corr_warps
        mma_tile_coord_v = thr_mma_qk.thr_idx

        # tStScales: (tileQ128,scale1):(65536,0)
        tStScale_layout = cute.composition(
            tStS.layout, cute.make_layout((self.m_block_size, 1))
        )
        tStScales = tuple(
            cute.make_tensor(
                tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout
            )
            for stage in range(self.q_stage)
        )

        # NOTE: since no correction is required for the first iter
        # we just release to notify mma warp that O has been rescaled
        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen_info = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen_info, m_block, split_idx, num_splits
            )

            # --- Compute sm_scale factor ---

            qk_descale, v_descale = self._load_effective_descales(
                descale_tensors, batch_idx, kv_head_idx
            )
            if const_expr(self.score_mod is None):
                softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
            else:
                softmax_scale_log2_eff = softmax_scale_log2

            max_offset = (
                Float32(8.0) if const_expr(self.q_dtype.width == 8) else Float32(0.0)
            )
            max_offset_scale = (
                Float32(256.0) if const_expr(self.q_dtype.width == 8) else Float32(1.0)
            )

            # --- Make gO of current tile ---

            if const_expr(self.is_split_kv):
                mO_cur = seqlen_info.offset_batch_Q(mO, batch_idx, dim=3)[
                    None, None, head_idx, split_idx
                ]
            else:
                mO_cur = seqlen_info.offset_batch_Q(mO, batch_idx, dim=3)[
                    None, None, head_idx
                ]
            gO = None
            if const_expr(self.use_tma_O or not self.pack_gqa):
                # gO_2CTA: (tileQ128*CTA2,tileHD128,stageQ):(1@1,1@0,256@1)
                tiler_gO = (  # (tileQ128*CTA2*stageQ,tileHD128)
                    (self.mma_tiler_pv[0] * self.q_stage),
                    self.head_dim_v_padded,
                )
                gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
                gO = layout_utils.select(
                    cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                )

                # Slice current CTA of gO: (tileQ128,tileHD128,stageQ):(1@1,1@0,256@1)
                gO = cute.flat_divide(
                    gO, (self.mma_tiler_pv[0] // self.cta_group_size,)
                )[None, mma_tile_coord_v, None, None]

            # --- Init softmax stats ---

            # (row_sum, row_max, acc_O_mn_row_is_zero_or_nan) for each Q
            stats = [
                (
                    0.0,
                    -Float32.inf
                    if const_expr(mLSE is not None or learnable_sink is not None)
                    else None,
                    True,
                )
            ] * self.q_stage

            # --- Determine tile counts ---

            if const_expr(self.use_block_sparsity):  # TODO: review the logics
                total_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen_info,
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(
                    not self.is_split_kv
                ) or total_block_count > Int32(0)

            # --- Debug print ---

            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 0)
                and is_print_block
                and (m_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )
            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[fwd_sm100_corr] "
                    cute.printf("")
                    cute.printf(
                        prefix + "m_block={} head_idx={} batch_idx={} split_idx={}",
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                    )
                    cute.printf(
                        prefix
                        + "n_block_min={} n_block_max={} total_block_count={} has_work={}",
                        n_block_min,
                        n_block_max,
                        total_block_count,
                        has_work,
                    )
                    cute.printf(
                        prefix + "softmax_scale_log2_eff={} qk_descale={} v_descale={}",
                        softmax_scale_log2_eff,
                        qk_descale,
                        v_descale,
                    )
                    cute.printf("")
                    cute.printf(prefix + "tStScales[0].layout: {}", tStScales[0].layout)
                    cute.printf(prefix + "tOtO.layout: {}", tOtO.layout)
                    cute.printf(prefix + "sO.layout: {}", sO.layout)
                    if const_expr(gO is not None):
                        cute.printf(prefix + "gO.layout: {}", gO.layout)
                    cute.printf("")

            # --- Correct tO ---

            if has_work:
                # --- Prologue: no correction and skip ---

                # Wait for sScale0(0) to be full
                sm_stats_barrier.arrive_and_wait_w_index(
                    index=0 * num_corr_warps + warp_idx
                )

                # WAR release (bootstrap): the first KV tile needs no correction, so just
                # release slot 0 to unblock softmax's acquire and prime the cross-release cycle.
                pipeline_sm_stats.consumer_release_w_index(0)

                # Wait for sScale1(0) to be full
                if const_expr(self.q_stage == 2):
                    sm_stats_barrier.arrive_and_wait_w_index(
                        index=1 * num_corr_warps + warp_idx
                    )

                # Flip phase for next KV tile
                sm_stats_consumer_phase ^= 1

                # --- Mainloop: correct tO(i-1) with corr_scale ---

                for i in cutlass.range(1, total_block_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Wait for sScale(i) with corr_scale(i) to be full
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * num_corr_warps + warp_idx
                        )

                        # Load corr_scale(i) from sScale(i)
                        corr_scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = (
                            cute.arch.vote_ballot_sync(corr_scale < 1.0) != 0
                        )

                        # NOTE: we don't need wait tO(i-1) to be full,
                        # since by the time the sScale(i) is ready,
                        # tS(i) must have been done, so tO(i-1) must have been done as well.

                        # Rescale tO(i-1) with corr_scale(i) if needed
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv,
                                tOtO[None, None, None, stage],
                                tidx,
                                corr_scale,
                                is_print_thread_and_tile=(
                                    is_print_thread_and_tile and i == 1 and stage == 0
                                ),
                            )

                        # Release tO(i) to be empty
                        # to notify mma warp that tO(i-1) has been rescaled
                        # and the tO buffer can be accumulated for tO(i) now
                        pipeline_s_p_o.consumer_release_w_index(stage)

                        # WAR release with CROSS index (q_stage-1-stage) to backpressure:
                        # free the *other* stage's slot instead of this one,
                        # so a single correction wg serves both softmax wgs round-robin
                        # to let softmax wg(1-stage) run while wg(stage) staggers on its next acquire.
                        pipeline_sm_stats.consumer_release_w_index(
                            self.q_stage - 1 - stage
                        )

                    # Flip phase for next KV tile
                    sm_stats_consumer_phase ^= 1

                # --- Epilogue: correct tO(-1) with row_sum ---

                # WAR release (handoff): drain the slot left dangling by the mainloop's
                # cross-release before the epilogue switches to direct release(stage) below.
                if const_expr(self.q_stage == 2):
                    pipeline_sm_stats.consumer_release_w_index(1)

                # Load learnable sink value(s) if needed
                #
                # NOTE: Even in the case of self.overlap_sO_sQ,
                # we can write to stage 0 of sO without additional sync
                # because the MMA in the top half must have been done.
                # Similarly, we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                (
                                    (
                                        (m_block * self.q_stage + stage)
                                        * self.cta_group_size
                                        + mma_tile_coord_v
                                    )
                                    * self.m_block_size
                                    + tidx
                                )
                                % self.qhead_per_kvhead
                                + head_idx * self.qhead_per_kvhead
                            )
                            learnable_sink_val[stage] = Float32(
                                learnable_sink[q_head_idx]
                            )

                # Correct tO(-1) and write to smem/gmem
                for stage in cutlass.range_constexpr(self.q_stage):
                    # Wait for sScale(end) with final row_sum/row_max to be full
                    sm_stats_barrier.arrive_and_wait_w_index(
                        index=stage * num_corr_warps + warp_idx
                    )

                    # Load final row_sum/row_max from sScale(end)
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ]
                    else:
                        row_max = None

                    # WAR release (direct, epilogue): final row_sum/row_max have been read out
                    # of sScale[stage]; release the same stage's slot (no staggering at tile end).
                    pipeline_sm_stats.consumer_release_w_index(stage)

                    # Correct final row_sum/row_max with learnable sink if needed
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2_eff)
                                row_sum = max_offset_scale
                            else:
                                row_sum += cute.math.exp2(
                                    sink_val * LOG2_E
                                    - row_max * softmax_scale_log2_eff
                                    + max_offset,
                                    fastmath=True,
                                )

                    # Compute scale for tO(-1) row-sum normalization
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    rowsum_norm_scale = cute.arch.rcp_approx(
                        row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0
                    )
                    rowsum_norm_scale = rowsum_norm_scale * v_descale

                    # Wait for tO(-1) to be full
                    # NOTE: we need to explicitly wait for tO(-1) to be full
                    # since we don't have tS(end) to guarantee that,
                    # tO(-1) is ready by the time sScale(end) is ready.
                    pipeline_o_acc.consumer_wait_w_index_phase(
                        stage, o_corr_consumer_phase
                    )

                    # Acquire sO to be empty by the specified epilogue warp
                    # in non-corr-epi mode
                    if const_expr(not self.use_correction_warps_for_epi):
                        assert pipeline_o_epi is not None  # mypy
                        pipeline_o_epi.producer_acquire_w_index_phase(
                            stage, corr_epi_producer_phase
                        )

                    # Correct tO(-1) with row-sum normalization,
                    # and write to smem buffer, and then gmem buffer in corr-epi mode
                    gO_stage = (
                        gO[None, None, stage] if const_expr(gO is not None) else None
                    )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen_info.seqlen_q,
                        rowsum_norm_scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO_stage,
                        gmem_tiled_copy_O,
                        is_print_thread_and_tile=is_print_thread_and_tile,
                    )

                    # Signal for the next work tile that tO are already read,
                    # so mma warp can write to them
                    pipeline_s_p_o.consumer_release_w_index(stage)

                    # Commit sO to be full
                    # to notify the epilogue warp to write to gmem
                    # in non-corr-epi mode
                    if const_expr(not self.use_correction_warps_for_epi):
                        assert pipeline_o_epi is not None  # mypy
                        pipeline_o_epi.producer_commit_w_index(stage)

                # Flip phases for next Q tile
                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:  # TODO: review the logics
                gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                if const_expr(self.use_block_sparsity):
                    (
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen_info,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtO,
                        sO,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        pipeline_o_epi,
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2_eff,
                        max_offset,
                        max_offset_scale,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            # --- Compute LSE and write to gmem ---

            if const_expr(mLSE is not None):
                if const_expr(not seqlen_info.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen_info.offset_q
                        if const_expr(not self.pack_gqa)
                        else (0, seqlen_info.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset(
                            (offset,), mLSE[None, head_idx, split_idx]
                        )
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])

                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = (
                        m_block * self.q_stage + stage
                    ) * self.cta_group_size + mma_tile_coord_v
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (
                            row_max * softmax_scale_log2_eff
                            + (cute.math.log2(row_sum, fastmath=True) - max_offset)
                        )
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen_info.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen_info.seqlen_q * self.qhead_per_kvhead
                    )
                    if const_expr(
                        not self.pack_gqa
                        or self.m_block_size % self.qhead_per_kvhead == 0
                    ):
                        gLSE = cute.local_tile(
                            mLSE_cur, (self.m_block_size,), (m_tile_idx,)
                        )
                        if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                            # This actually just works with PackGQA too
                            gLSE[tidx] = lse
                    else:
                        idx = m_tile_idx * self.m_block_size + tidx
                        if idx < seqlen_q:
                            m_idx = idx // self.qhead_per_kvhead
                            h_idx = idx - m_idx * self.qhead_per_kvhead
                            lse_ptr_i64 = cutedsl_utils.elem_pointer(
                                mLSE_cur, ((h_idx, m_idx),)
                            ).toint()
                            lse_gmem_ptr = cute.make_ptr(
                                mLSE_cur.element_type,
                                lse_ptr_i64,
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            cute.make_tensor(lse_gmem_ptr, (1,))[0] = lse

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

        # This is equivalent to pipeline_o_epi.consumer_tail
        if const_expr(not self.use_correction_warps_for_epi):
            assert pipeline_o_epi is not None  # mypy
            pipeline_o_epi.producer_acquire_w_index_phase(
                self.q_stage - 1, corr_epi_producer_phase
            )

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
        is_print_thread_and_tile: bool = False,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        corr_tile_hd = (
            16  # corrHD: a tunable parameter of the correction tile size along head dim
        )
        num_corr_tiles_hd = (
            self.head_dim_v_padded // corr_tile_hd
        )  # restHD = tileHD // corrHD
        corr_tile_layout = cute.make_layout((self.m_block_size, corr_tile_hd))

        # tOtO: (MMA_tC=(128,128),MMA_Q1,MMA_HD1):((65536,1),0,0)
        # tOcO: (MMA_tC=(128,128),MMA_Q1,MMA_HD1):((1@0,1@1),0,0)
        # tOtO_i: (tileQ128, corrHD16):(65536,1)
        # tOcO_i: (tileQ128, corrHD16):(1@0,1@1)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        tOtO_i = cute.composition(tOtO, corr_tile_layout)
        tOcO_i = cute.composition(tOcO, corr_tile_layout)

        # --- Make T2R copy for O before rescale ---

        # T2R copy atom of `tcgen05.ld.sync.aligned.32x32b.x16`
        # layout_src_tv=(32,512):(0,1) => (row32,col16) cells in tmem per warp
        # layout_dst_tv=(32,16):(16,1) => 16 fp32 elems in rmem per thread
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_hd)),
            self.pv_acc_dtype,
        )

        # T2R tiled copy atom:
        # layout_src_tv_tiled=((32,4),((16,32),1)):((0,1),((128,4),0))
        #   => 4 x (row32,col16) cells in tmem per warp group
        # layout_dst_tv_tiled=((32,4),(16,1)):((4,1),(128,0))
        #   => still 16 fp32 elems in rmem per thread, but tiled in a warp group
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)

        # tOtO_t2r: (T2R_CPY_ATOM=((16,32),1),CPY_Q1,CPY_corrHD1):(((1,65536),0),0,0)
        # tOrO_t2r_shape_i: (T2R_CPY_ATOM=(16,1),CPY_Q1,CPY_corrHD1)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape_i = thr_tmem_load.partition_D(tOcO_i).shape

        # --- Make R2T copy for O after rescale ---

        # R2T copy atom of `tcgen05.st.async.aligned.32x32b.x16`
        # layout_src_tv=(32,16):(16,1) => 16 fp32 elems in rmem per thread
        # layout_dst_tv=(32,512):(0,1) => (row32,col16) cells in tmem per warp
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_hd)),
            self.pv_acc_dtype,
        )

        # R2T tiled copy atom:
        # layout_src_tv_tiled=((32,4),(16,1)):((4,1),(128,0))
        #   => still 16 fp32 elems in rmem per thread, but tiled in a warp group
        # layout_dst_tv_tiled=((32,4),((16,32),1)):((0,1),((128,4),0))
        #   => 4 x (row32,col16) cells in tmem per warp group
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)

        # tOtO_r2t: (R2T_CPY_ATOM=((16,32),1),CPY_Q1,CPY_HD1):(((1,65536),0),0,0)
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm100_corr_rescale] "
                cute.printf("")
                cute.printf(prefix + "scale={}", scale)
                cute.printf(
                    prefix + "corr_tile_size={} frg_count={}",
                    corr_tile_hd,
                    self.head_dim_v_padded // corr_tile_hd,
                )
                cute.printf(prefix + "corr_tile_layout: {}", corr_tile_layout)
                cute.printf("")
                cute.printf(prefix + "tOtO.layout: {}", tOtO.layout)
                cute.printf(prefix + "tOcO.layout: {}", tOcO.layout)
                cute.printf(prefix + "tOtO_i.layout: {}", tOtO_i.layout)
                cute.printf(prefix + "tOcO_i.layout: {}", tOcO_i.layout)
                cute.printf("")
                cute.printf(
                    prefix + "tmem_load_atom: layout_src_tv={} layout_dst_tv={}",
                    tmem_load_atom.layout_src_tv,
                    tmem_load_atom.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "thr_tmem_load: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    thr_tmem_load.layout_src_tv_tiled,
                    thr_tmem_load.layout_dst_tv_tiled,
                )
                cute.printf(prefix + "tOtO_t2r.layout: {}", tOtO_t2r.layout)
                cute.printf(prefix + "tOrO_t2r_shape_i: {}", tOrO_t2r_shape_i)
                cute.printf("")
                cute.printf(
                    prefix + "tmem_store_atom: layout_src_tv={} layout_dst_tv={}",
                    tmem_store_atom.layout_src_tv,
                    tmem_store_atom.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "thr_tmem_store: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    thr_tmem_store.layout_src_tv_tiled,
                    thr_tmem_store.layout_dst_tv_tiled,
                )
                cute.printf("")
                cute.printf(prefix + "tOtO_r2t.layout: {}", tOtO_r2t.layout)
                cute.printf("")

        # --- Rescale tO(i) ---

        for i in cutlass.range_constexpr(num_corr_tiles_hd):  # restHD loops
            # T2R copy tO(i) -> rO(i)
            tOrO_i = cute.make_rmem_tensor(tOrO_t2r_shape_i, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + i * corr_tile_hd, tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_i)

            # Rescale rO(i) with corr_scale
            for j in cutlass.range(0, cute.size(tOrO_i), 2, unroll_full=True):
                tOrO_i[j], tOrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_i[j], tOrO_i[j + 1]), (scale, scale)
                )

            # R2T copy rO(i) -> tO(i)
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_hd, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_i, tOtO_r2t_i)

        # Ensure all stores to tO are visible to mma warps
        # with `tcgen05.wait::st`
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
        is_print_thread_and_tile: bool = False,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_hd = 8 * 32 // self.o_dtype.width  # 32B per tile => corrHD=16
        num_corr_tiles_hd = (
            self.head_dim_v_padded // corr_tile_hd
        )  # restHD = tileHD // corrHD
        corr_tile_layout = cute.make_layout((self.m_block_size, corr_tile_hd))
        epi_subtile = (self.epi_tile[0], corr_tile_hd)

        # tOtO: (MMA_tC=(128,128),MMA_Q1,MMA_HD1):((65536,1),0,0)
        # tOcO: (MMA_tC=(128,128),MMA_Q1,MMA_HD1):((1@0,1@1),0,0)
        # sO: (EPI_Q=(8,16),EPI_HD=(64,2)):((64,512),(1,8192))
        # tOsO: (EPI_ATOM=(EPI_Q128,EPI_HD=(64,2)),restQ1,restHD1):((64,(1,8192)),0,0)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        tOsO = thr_mma.get_slice(0).partition_C(
            sO
        )  # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized

        # tOtO_i: (tileCorr=(128,16),restHD8):((65536,1),16)
        # tOcO_i: (tileCorr=(128,16),restHD8):((1@0,1@1),16@1)
        # tOsO_i: (tileCorr=(128,16),restHD=(4,2)):((64,1),(16,8192))
        tOtO_i = cute.logical_divide(tOtO, corr_tile_layout)
        tOcO_i = cute.logical_divide(tOcO, corr_tile_layout)
        tOsO_i = cute.logical_divide(tOsO, corr_tile_layout)

        # --- Make T2R copy for O ---

        # T2R copy atom of `tcgen05.ld.sync.aligned.32x32b.x16`
        # layout_src_tv=(32,512):(0,1) => (row32,col16) cells in tmem per warp
        # layout_dst_tv=(32,16):(16,1) => 16 fp32 elems in rmem per thread
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_tile=epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )

        # T2R tiled copy atom:
        # layout_src_tv_tiled=((32,4),((16,32),1)):((0,1),((128,4),0))
        #   => 4 x (row32,col16) cells in tmem per warp group
        # layout_dst_tv_tiled=((32,4),(16,1)):((4,1),(128,0))
        #   => still 16 fp32 elems in rmem per thread, but tiled in a warp group
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)

        # tOtO_t2r: (T2R_CPY_ATOM=((16,32),1),CPY_Q1,CPY_corrHD1,restHD8):(((1,65536),0),0,0,16)
        # tOcO_t2r: (T2R_CPY_ATOM=(16,1),CPY_Q1,CPY_corrHD1,restHD8):((1@1,0),0,0,16@1)
        # tOsO_r2s: (R2S_CPY_ATOM=((8,2),1),CPY_Q1,CPY_corrHD1,restHD=((2,2),2)):(((1,8),0),0,0,((16,32),8192))
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        # partition_D_position_independent: fixes incorrect address computation when indexing a
        # swizzle smem tensor across multiple tiles.
        #
        # Background:
        #   smem tensors use a swizzle pointer (swizzle transform encoded in the pointer itself).
        #   A plain `thr_copy.partition_D(tensor)` produces a position-dependent layout: it assumes
        #   the swizzle offset at the base pointer applies uniformly to all tiles. However, the
        #   restHD dimension of tOsO_i spans multiple smem tiles at different memory offsets, each
        #   requiring a different swizzle transform.
        #
        #   Incorrect example (plain partition_D):
        #     tOsO_r2s = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        #     tOsO_r2s[None, 0, 0, 0]  # i=0: correct, base pointer swizzle happens to be valid
        #     tOsO_r2s[None, 0, 0, 1]  # i=1: WRONG, swizzle unchanged but actual address should differ
        #
        # Fix (partition_D_position_independent):
        #   1. Layout part: calls as_position_independent_swizzle_tensor to move the swizzle out of
        #      the pointer and into an explicit ComposedLayout (swizzle_fn ∘ linear_layout), so that
        #      the layout itself correctly computes the swizzled address for any offset i.
        #   2. Pointer part: applies swizzle_ptr to the base pointer to obtain its actual physical
        #      address, which pairs correctly with the position-independent layout above.
        #
        #   Correct example (after fix):
        #     tOsO_r2s[None, 0, 0, 0]  # i=0: layout computes swizzle explicitly, correct
        #     tOsO_r2s[None, 0, 0, 1]  # i=1: layout independently computes swizzle for offset 1*corr_tile_hd, correct
        tOsO_r2s = copy_utils.partition_D_position_independent(
            thr_tmem_load, tOsO_i[(None, None), None]
        )

        # --- Make R2S copy for O ---

        # R2S copy atom of `universal copy`
        # layout_src_tv=(1,1):(0,0)
        # layout_dst_tv=(1,1):(0,0)
        #
        # NOTE: stmatrix is NOT selected here because all stmatrix conditions require num_dp=16,
        # but for N-major output (typical case), get_tmem_load_op selects Ld32x32b (num_dp=32).
        # With num_dp=32, get_smem_store_op falls through all stmatrix branches so returns CopyUniversalOp.
        # Only M-major output would pick Ld16x256b with num_dp=16, enabling `stmatrix.m8n8.x4`.
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )

        # R2S tiled copy atom:
        # layout_src_tv_tiled=((32,4),(1,16)):((4,1),(0,128))
        #   => still 16 fp32 elems in rmem per thread, but tiled in a warp group
        # layout_dst_tv_tiled=((32,4),(1,16)):((4,1),(0,128))
        #   => each thread writes its 16 fp32 elems from rmem to smem
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        # --- Debug print ---

        if const_expr(self.debug_print):
            if stage == 0 and is_print_thread_and_tile:
                prefix = "[fwd_sm100_corr_epilogue] "
                cute.printf("")
                cute.printf(
                    prefix + "stage={} m_block={} scale={}", stage, m_block, scale
                )
                cute.printf(
                    prefix + "corr_tile_size={} num_tiles={}",
                    corr_tile_hd,
                    self.head_dim_v_padded // corr_tile_hd,
                )
                cute.printf("")
                cute.printf(prefix + "sO.layout: {}", sO.layout)
                cute.printf(prefix + "tOtO.layout: {}", tOtO.layout)
                cute.printf(prefix + "tOcO.layout: {}", tOcO.layout)
                cute.printf(prefix + "tOsO.layout: {}", tOsO.layout)
                cute.printf(prefix + "tOtO_i.layout: {}", tOtO_i.layout)
                cute.printf(prefix + "tOcO_i.layout: {}", tOcO_i.layout)
                cute.printf(prefix + "tOsO_i.layout: {}", tOsO_i.layout)
                cute.printf("")
                cute.printf(
                    prefix + "tmem_copy_atom: layout_src_tv={} layout_dst_tv={}",
                    tmem_copy_atom.layout_src_tv,
                    tmem_copy_atom.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "thr_tmem_load: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    thr_tmem_load.layout_src_tv_tiled,
                    thr_tmem_load.layout_dst_tv_tiled,
                )
                cute.printf(prefix + "tOtO_t2r.layout: {}", tOtO_t2r.layout)
                cute.printf(prefix + "tOcO_t2r.layout: {}", tOcO_t2r.layout)
                cute.printf("")
                cute.printf(
                    prefix + "smem_copy_atom: layout_src_tv={} layout_dst_tv={}",
                    smem_copy_atom.layout_src_tv,
                    smem_copy_atom.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "tiled_smem_store: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    tiled_smem_store.layout_src_tv_tiled,
                    tiled_smem_store.layout_dst_tv_tiled,
                )
                cute.printf(prefix + "tOsO_r2s.layout: {}", tOsO_r2s.layout)
                cute.printf("")

        # --- Correct O and write to smem ---

        for i in cutlass.range(num_corr_tiles_hd, unroll_full=True):  # restHD loops
            # T2R copy tO(i) -> rO(i)
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOrO_i = cute.make_rmem_tensor_like(
                tOcO_t2r[None, 0, 0, i], self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_i)

            # Rescale rO(i) for row-sum normalization
            tOsO_r2s_i = tOsO_r2s[None, 0, 0, i]
            for j in cutlass.range(0, cute.size(tOrO_i), 2, unroll_full=True):
                tOrO_i[j], tOrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_i[j], tOrO_i[j + 1]), (scale, scale)
                )

            # R2S copy rO(i) -> sO(i) with dtype downcast
            copy_utils.cvt_copy(tiled_smem_store, tOrO_i, tOsO_r2s_i)

        # Make R2S stores visible to the subsequent TMA S2G copy
        # by `fence.proxy.async.shared::cta`
        cute.arch.fence_view_async_shared()

        # --- S2G copy O to gemm (if needed) ---

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O
            assert gmem_tiled_copy_O is not None

            # Sync this correction warp group to ensure all R2S stores done
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE,
            )

            # S2G copy sO -> gO using non-TMA by:
            #   1. S2R copy sO -> rO
            #   2. R2G copy rO -> gO with predicate for OOB guard
            mma_tile_coord_v = thr_mma.thr_idx
            m_tile_idx = (
                m_block * self.q_stage + stage
            ) * self.cta_group_size + mma_tile_coord_v
            self._store_O_to_gmem(
                sO,
                gO,
                mO_cur,
                gmem_tiled_copy_O,
                tidx,
                seqlen_q,
                m_tile_idx,
                is_print_thread_and_tile=(stage == 0 and is_print_thread_and_tile),
            )

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: Optional[cute.Tensor],
        mO_cur: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        seqlen_q: Int32,
        m_tile_idx: Int32,
        is_print_thread_and_tile: bool = False,
    ):
        """Copy a single stage of O from smem to gmem via registers."""
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO_stage)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = copy_utils.predicate_k(tOcO, limit=mO_cur.shape[1])
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_v_padded,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
        )

        # load acc O from smem to rmem for wider vectorization
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm100_store_O_to_gmem] "
                cute.printf("")
                cute.printf(prefix + "m_tile_idx={} seqlen_q={}", m_tile_idx, seqlen_q)
                cute.printf(
                    prefix + "gmem_tiled_copy_O: layout_src_tv={} layout_dst_tv={}",
                    gmem_tiled_copy_O.layout_src_tv,
                    gmem_tiled_copy_O.layout_dst_tv,
                )
                cute.printf(prefix + "tOsO.layout: {}", tOsO.layout)
                cute.printf(prefix + "tOcO.layout: {}", tOcO.layout)
                cute.printf(prefix + "t0OcO.layout: {}", t0OcO.layout)
                cute.printf(prefix + "tOrO.layout: {}", tOrO.layout)
                cute.printf("")

        # copy acc O from rmem to gmem
        if const_expr(not self.pack_gqa):
            assert gO is not None
            tOgO = gmem_thr_copy_O.partition_D(gO)
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0]
                    < seqlen_q - m_tile_idx * self.m_block_size - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_tile_idx, seqlen_q
            )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: Optional[ffa_pipeline.PipelineAsync],
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
        mma_tile_coord_v: Int32 = 0,
        is_print_block: bool = False,
    ):
        assert pipeline_o_epi is not None  # mypy

        num_epilogue_threads = len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_epilogue_threads

        # Make dummy CTA coord/layout since we do not use TMA multicast
        tma_cta_coord = 0
        tma_cta_layout = cute.make_layout(1)

        epi_consumer_phase = Int32(0)

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen_info, m_block, split_idx, num_splits
            )

            # --- Debug print ---

            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 0)
                and is_print_block
                and (m_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )
            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[fwd_sm100_epi_s2g] "
                    cute.printf("")
                    cute.printf(
                        prefix + "m_block={} head_idx={} batch_idx={} split_idx={}",
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                    )
                    cute.printf(
                        prefix + "n_block_min={} n_block_max={} mma_tile_coord_v={}",
                        n_block_min,
                        n_block_max,
                        mma_tile_coord_v,
                    )
                    cute.printf(prefix + "sO.layout: {}", sO.layout)
                    cute.printf("")

            # --- Make gO of current tile ---

            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen_info.offset_batch_Q(mO, batch_idx, dim=3)[
                        None, None, head_idx, split_idx
                    ]
                else:
                    mO_cur = seqlen_info.offset_batch_Q(mO, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                gO = None
                if const_expr(self.use_tma_O or not self.pack_gqa):
                    # gO_2CTA: (tileQ*CTA2,tileHD128,stageQ):(1@1,1@0,256@1)
                    tiler_gO = (  # (tileQ128*CTA2*stageQ,tileHD128)
                        (self.mma_tiler_pv[0] * self.q_stage),
                        self.head_dim_v_padded,
                    )
                    gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
                    gO = layout_utils.select(
                        cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                    )

                    # Slice current CTA of gO: (tileQ128,tileHD128,stageQ):(1@1,1@0,256@1)
                    gO = cute.flat_divide(
                        gO, (self.mma_tiler_pv[0] // self.cta_group_size,)
                    )[None, mma_tile_coord_v, None, None]

                # --- S2G copy O to gmem with or w/o TMA ---

                if const_expr(self.use_tma_O):
                    # Define TMA store fn for O
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, tma_cta_coord, tma_cta_layout, sO, gO
                    )

                    # Issue TMA store
                    for stage in cutlass.range(self.q_stage, unroll_full=True):
                        # Wait for final corrected sOi to be full
                        pipeline_o_epi.consumer_wait_w_index_phase(
                            stage, epi_consumer_phase
                        )
                        # S2G copy sO -> gO using TMA
                        store_O(src_idx=stage, dst_idx=stage)
                        cute.arch.cp_async_bulk_commit_group()

                    # Wait for TMA store
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Wait sOi buffer is read and ready to be released
                        cute.arch.cp_async_bulk_wait_group(
                            # NOTE: with `.read` quanlifier, the `cp.async.bulk.wait`
                            # only waits for the completion of "smem read", instead of "gmem write"
                            self.q_stage - 1 - stage,
                            read=True,
                        )

                        # Release sOi buffer to be empty for next tile
                        pipeline_o_epi.consumer_release_w_index(stage)
                else:
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Wait for final corrected sOi to be full
                        pipeline_o_epi.consumer_wait_w_index_phase(
                            stage, epi_consumer_phase
                        )

                        # S2G copy sO -> gO using non-TMA by:
                        #   1. S2R copy sO -> rO
                        #   2. R2G copy rO -> gO with predicate for OOB guard
                        m_tile_idx = (
                            m_block * self.q_stage + stage
                        ) * self.cta_group_size + mma_tile_coord_v
                        gO_stage = (
                            gO[None, None, stage]
                            if const_expr(gO is not None)
                            else None
                        )
                        self._store_O_to_gmem(
                            sO[None, None, stage],
                            gO_stage,
                            mO_cur,
                            gmem_tiled_copy_O,
                            tidx,
                            seqlen_info.seqlen_q,
                            m_tile_idx,
                            is_print_thread_and_tile=(
                                stage == 0 and is_print_thread_and_tile
                            ),
                        )

                        # Release sOi buffer to be empty for next tile
                        pipeline_o_epi.consumer_release_w_index(stage)

                # Flip consumer phase after consuming all stages for this tile
                epi_consumer_phase ^= 1

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(
            src_idx=block,
            dst_idx=stage,
            tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage),
        )

    def load_Q_non_tma(
        self,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        gmem_tiled_copy_Q: cute.TiledCopy,
        pipeline_q: pipeline.PipelineAsync,
        tidx: Int32,
        seqlen_q: Int32,
        m_block: Int32,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        assert self.cta_group_size == 1, "cta_group_size must be 1 for non-tma Q load"
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_padded,
            self.check_hdim_oob,
            self.qhead_per_kvhead,
        )
        sQ_stage = sQ[None, None, None, stage]
        sQ_pi = cute.make_tensor(
            sQ_stage.iterator,
            cute.make_layout(
                (sQ_stage.shape[0][0], (sQ_stage.shape[0][1], sQ_stage.shape[2])),
                stride=(
                    sQ_stage.stride[0][0],
                    (sQ_stage.stride[0][1], sQ_stage.stride[2]),
                ),
            ),
        )
        pack_gqa.load_Q(
            mQ, sQ_pi, gmem_tiled_copy_Q, tidx, m_block * self.q_stage + block, seqlen_q
        )
        cute.arch.cp_async_commit_group()
        pipeline_q.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        extra_tx_count: Optional[Int32] = None,
        is_print_thread_and_tile: bool = False,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        extra_tx_count_kv = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        extra_tx_count = (
            extra_tx_count_kv + (extra_tx_count if extra_tx_count is not None else 0)
            if const_expr(self.use_tma_KV)
            else None
        )
        extra_kwargs = (
            {"extra_tx_count": extra_tx_count} if const_expr(self.use_tma_KV) else {}
        )
        pipeline_kv.producer_acquire(producer_state, **extra_kwargs)

        # TODO: review the logics
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                pipeline_kv.sync_object_empty.wait(1, phase)

        if const_expr(self.use_tma_KV):
            assert tXgX is not None and tXsX is not None and tma_atom is not None
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)

            # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
            tXgX_cur = (
                tXgX[None, block]
                if const_expr(page_idx is None)
                else tXgX[None, 0, page_idx]
            )

            # Issue TMA G2S copy for sKi/sVi
            cute.copy(
                tma_atom,
                src=tXgX_cur,
                dst=tXsX_cur,
                tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
            )
        else:  # TODO: review the logics
            assert paged_kv_manager is not None
            assert extra_tx_count is None
            sX_cur = sX[None, None, None, stage]
            if const_expr(self.uneven_kv_smem):
                sX_cur = self.offset_kv_smem(sX_cur, stage, phase ^ 1)
            paged_kv_manager.load_KV(block, sX_cur, K_or_V)
            cute.arch.cp_async_commit_group()
            pipeline_kv.sync_object_full.arrive_cp_async_mbarrier(stage)

        # --- Debug prints ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = f"[fwd_sm100_load_{K_or_V}] "
                cute.printf(prefix + "block={} stage={} phase={}", block, stage, phase)
                if const_expr(self.use_tma_KV):
                    cute.printf(prefix + "tXgX_cur.layout: {}", tXgX_cur.layout)
                    cute.printf(prefix + "tXsX_cur.layout: {}", tXsX_cur.layout)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            # Hint that the offset is 128-bit aligned so that
            # ptr + offset preserves the alignment needed by cp.async.
            offset = cute.assume(offset, divby=128 // self.k_dtype.width)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset(
            (m_block * self.m_block_size, n_block * self.n_block_size), cS
        )
        tScS = thr_mma_qk.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if const_expr(self.pack_gqa):
            assert head_divmod is not None
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical, head_offset = divmod(q_physical, head_divmod)
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = divmod(q_idx_logical, seqlen_q_divmod)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen_info,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
