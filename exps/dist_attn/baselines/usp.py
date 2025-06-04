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

from typing import Dict

import torch
import torch.distributed as dist

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges

from .interface import AttnBaselineInterface
from .ring_attn import FA3RingAttnFunc, TERingAttnFunc
from .shard import (
    ParallelMode,
    ShardMeta,
    get_cu_seqlens_padded,
    get_max_seqlen,
    get_pad_factor,
    zigzag_dispatch,
    zigzag_undispatch,
)
from .utils_cp import (
    AttnBackend,
    _pre_process,
    _SeqAllToAll,
    generate_runtime_meta_per_step,
    get_cu_seqlens_on_cp_rank,
    unflatten_data_from_varlen,
)


class USP(AttnBaselineInterface):
    def __init__(
        self,
        cp_process_group: Dict,
        qkv_format: str,
        backend: AttnBackend,
    ):
        self.pg_p2p = cp_process_group[ParallelMode.RING]
        self.pg_a2a = cp_process_group[ParallelMode.ULYSESS]
        # pad factor for ulysess & ring
        self.pad_factor_p2p, self.pad_factor_a2a = get_pad_factor(
            cp_group_p2p=self.pg_p2p, cp_group_a2a=self.pg_a2a
        )
        self.backend = backend
        self.qkv_format = qkv_format
        self.shard_meta = {}  # type: ignore
        self.runtime_meta_per_step = []  # type: ignore

    # to call after q,k,v dispatch
    def pre_compute_attn_runtime_meta(self, attn_mask_type: AttnMaskType, device):
        if self.backend == AttnBackend.FA3:
            causal = attn_mask_type == AttnMaskType.CAUSAL
            shard_q_meta = self.shard_meta["q"]
            shard_kv_meta = self.shard_meta["k"]

            cp_rank = dist.get_rank(group=self.pg_p2p)
            cp_size = dist.get_world_size(group=self.pg_p2p)
            self.runtime_meta_per_step = [None for i in range(cp_size)]

            for i in range(cp_size):
                first_idx_q, second_idx_q, first_idx_kv, second_idx_kv = (
                    True,
                    True,
                    True,
                    True,
                )
                factor_q, factor_kv = cp_size, cp_size
                if causal:
                    if i == 0:  # q, k, v
                        pass
                    elif i <= cp_rank:  # q, k0, v0
                        second_idx_kv = False
                        factor_kv *= 2
                    else:  # q1, k, v
                        first_idx_q = False
                        factor_q *= 2
                else:  # full
                    pass

                cu_seqlens_q_per_step = get_cu_seqlens_on_cp_rank(
                    shard_q_meta.cu_seqlens,
                    shard_q_meta.cu_seqlens_padded // cp_size,
                    cp_size,
                    cp_rank,
                    first_idx_q,
                    second_idx_q,
                )
                cu_seqlens_kv_per_step = get_cu_seqlens_on_cp_rank(
                    shard_kv_meta.cu_seqlens,
                    shard_kv_meta.cu_seqlens_padded // cp_size,
                    cp_size,
                    (cp_rank - i) % cp_size,
                    first_idx_kv,
                    second_idx_kv,
                )
                host_cu_seqlens_q_per_step = cu_seqlens_q_per_step.tolist()
                host_cu_seqlens_kv_per_step = cu_seqlens_kv_per_step.tolist()
                rumtime_meta = generate_runtime_meta_per_step(
                    cu_seqlens_q_per_step,
                    cu_seqlens_kv_per_step,
                    shard_q_meta.cu_seqlens_padded // factor_q,
                    shard_kv_meta.cu_seqlens_padded // factor_kv,
                    host_cu_seqlens_q_per_step,
                    host_cu_seqlens_kv_per_step,
                    shard_q_meta.host_cu_seqlens_padded[-1] // factor_q,
                    shard_kv_meta.host_cu_seqlens_padded[-1] // factor_kv,
                    device,
                )
                self.runtime_meta_per_step[i] = rumtime_meta

    def dispatch(
        self,
        x_global: torch.Tensor,
        ranges: AttnRanges,
        valid_total_seqlen: int,  # required by AttnRanges.to_cu_seqlens
        name: str,  # key name for shard_meta
        **kwargs,
    ):
        # pre-process data
        x_global_varlen, origin_shape, cu_seqlens, host_cu_seqlens = _pre_process(
            x_global, ranges, valid_total_seqlen, self.qkv_format, x_global.device
        )
        # compute cu_seqlens_padded and host_cu_seqlens_padded
        cu_seqlens_padded, host_cu_seqlens_padded = get_cu_seqlens_padded(
            cu_seqlens,
            host_cu_seqlens,
            "thd",
            pad_factor_p2p=self.pad_factor_p2p,
            pad_factor_a2a=self.pad_factor_a2a,
        )

        x_local, restore_shape = zigzag_dispatch(
            x_global_varlen,
            host_cu_seqlens,
            host_cu_seqlens_padded,
            "thd",
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=self.pg_a2a,
        )

        max_seqlen_padded = get_max_seqlen(host_cu_seqlens_padded)
        self.shard_meta[name] = ShardMeta(
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            host_cu_seqlens=host_cu_seqlens,
            host_cu_seqlens_padded=host_cu_seqlens_padded,
            origin_shape=origin_shape,
            max_seqlen_padded=max_seqlen_padded,
        )
        return x_local

    def undispatch(
        self,
        x_local: torch.Tensor,
        name: str,  # key name for shard_meta
        **kwargs,
    ) -> torch.Tensor:
        smeta = self.shard_meta[name]
        x_global_varlen = zigzag_undispatch(
            x_local,
            smeta.host_cu_seqlens,
            smeta.host_cu_seqlens_padded,
            "thd",
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=self.pg_a2a,
        )
        x_global = unflatten_data_from_varlen(
            x_global_varlen, smeta.cu_seqlens, smeta.origin_shape, self.qkv_format
        )

        return x_global

    def apply_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask_type: AttnMaskType,
        dropout_p: float,
        softmax_scale: float,
        deterministic: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_size_p2p = dist.get_world_size(group=self.pg_p2p)
        # ulysess all2all
        batch_dim, seq_dim = 0, 1

        # thd -> 1,thd
        q, k, v = [x.unsqueeze(0) for x in [q, k, v]]
        q_layer, k_layer, v_layer = [
            _SeqAllToAll.apply(self.pg_a2a, x, 2, seq_dim, batch_dim) for x in [q, k, v]
        ]
        q_layer, k_layer, v_layer = [x.squeeze(0) for x in [q_layer, k_layer, v_layer]]

        batch_p2p_comm = kwargs.get("batch_p2p_comm", True)
        with torch.cuda.device(q.device):
            cp_stream = torch.cuda.Stream()

        # ring attention p2p
        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]
        if self.backend == AttnBackend.TE:
            if attn_mask_type == AttnMaskType.CAUSAL:
                attn_mask = "padding_causal"
            elif attn_mask_type == AttnMaskType.FULL:
                attn_mask = "padding"

            out, lse = TERingAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.max_seqlen_padded // cp_size_p2p,
                shard_kv_meta.max_seqlen_padded // cp_size_p2p,
                shard_q_meta.cu_seqlens_padded // cp_size_p2p,
                shard_kv_meta.cu_seqlens_padded // cp_size_p2p,
                dropout_p,
                softmax_scale,
                "thd",
                self.pg_p2p,
                attn_mask,
                cp_stream,
                deterministic,
                batch_p2p_comm,
            )
        elif self.backend == AttnBackend.FA3:
            if attn_mask_type == AttnMaskType.CAUSAL:
                is_causal = True
            elif attn_mask_type == AttnMaskType.FULL:
                is_causal = False

            out, lse = FA3RingAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                shard_q_meta.cu_seqlens_padded // cp_size_p2p,
                shard_kv_meta.cu_seqlens_padded // cp_size_p2p,
                self.runtime_meta_per_step,
                is_causal,
                dropout_p,
                softmax_scale,
                self.pg_p2p,
                cp_stream,
                deterministic,
                batch_p2p_comm,
            )

        # ulysess all2all
        # thd -> 1,thd
        out = out.unsqueeze(0)
        out_layer = _SeqAllToAll.apply(self.pg_a2a, out, seq_dim, 2, batch_dim)
        out_layer = out_layer.squeeze(0)

        return out_layer, lse
