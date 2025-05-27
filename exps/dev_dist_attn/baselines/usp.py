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

from typing import Dict, List

import torch
import torch.distributed as dist

from magi_attention.common.enum import AttnMaskType

from .attn import AttnBackend, FA3RingAttnFunc, TERingAttnFunc
from .interface import AttnBaselineInterface
from .shard import (
    ParallelMode,
    ShardMeta,
    get_cu_seqlens_padded,
    get_pad_factor,
    zigzag_dispatch,
    zigzag_undispatch,
)
from .utils_cp import _SeqAllToAll, divide_lst


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

    def dispatch(
        self,
        x_global: torch.Tensor,
        cu_seqlens: torch.Tensor,
        host_cu_seqlens: List[int],
        name: str,  # key name for shard_meta
        **kwargs,
    ):
        # compute cu_seqlens_padded and host_cu_seqlens_padded
        cu_seqlens_padded, host_cu_seqlens_padded = get_cu_seqlens_padded(
            cu_seqlens,
            host_cu_seqlens,
            self.qkv_format,
            pad_factor_p2p=self.pad_factor_p2p,
            pad_factor_a2a=self.pad_factor_a2a,
        )

        x_local, restore_shape = zigzag_dispatch(
            x_global,
            cu_seqlens,
            cu_seqlens_padded,
            host_cu_seqlens,
            host_cu_seqlens_padded,
            self.qkv_format,
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=self.pg_a2a,
        )
        max_seqlen = max(
            [
                (host_cu_seqlens[i + 1] - host_cu_seqlens[i])
                for i in range(len(host_cu_seqlens) - 1)
            ]
        )
        max_seqlen_padded = max(
            [
                (host_cu_seqlens_padded[i + 1] - host_cu_seqlens_padded[i])
                for i in range(len(host_cu_seqlens_padded) - 1)
            ]
        )
        self.shard_meta[name] = ShardMeta(
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            host_cu_seqlens=host_cu_seqlens,
            host_cu_seqlens_padded=host_cu_seqlens_padded,
            restore_shape=restore_shape,
            max_seqlen=max_seqlen,
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
        x_global = zigzag_undispatch(
            x_local,
            smeta.cu_seqlens,
            smeta.cu_seqlens_padded,
            smeta.host_cu_seqlens,
            smeta.host_cu_seqlens_padded,
            self.qkv_format,
            smeta.restore_shape,
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=self.pg_a2a,
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
        if self.qkv_format != "sbhd":
            batch_dim, seq_dim = 0, 1
        else:
            batch_dim, seq_dim = 1, 0

        if self.qkv_format == "thd":  # thd -> 1,thd
            q, k, v = [x.unsqueeze(0) for x in [q, k, v]]
        q_layer, k_layer, v_layer = [
            _SeqAllToAll.apply(self.pg_a2a, x, 2, seq_dim, batch_dim) for x in [q, k, v]
        ]
        if self.qkv_format == "thd":
            q_layer, k_layer, v_layer = [
                x.squeeze(0) for x in [q_layer, k_layer, v_layer]
            ]

        batch_p2p_comm = kwargs.get("batch_p2p_comm", True)
        with torch.cuda.device(q.device):
            cp_stream = torch.cuda.Stream()

        # ring attention p2p
        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]
        pad_between_seqs = not (
            shard_q_meta.host_cu_seqlens[-1] == shard_q_meta.host_cu_seqlens_padded[-1]
        )
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
                self.qkv_format,
                self.pg_p2p,
                attn_mask,
                cp_stream,
                deterministic,
                pad_between_seqs,
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
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.cu_seqlens_padded // cp_size_p2p,
                shard_kv_meta.cu_seqlens_padded // cp_size_p2p,
                is_causal,
                dropout_p,
                softmax_scale,
                self.qkv_format,
                self.pg_p2p,
                cp_stream,
                deterministic,
                pad_between_seqs,
                batch_p2p_comm,
                [
                    shard_q_meta.host_cu_seqlens,
                    divide_lst(shard_q_meta.host_cu_seqlens_padded, cp_size_p2p),
                    shard_kv_meta.host_cu_seqlens,
                    divide_lst(shard_kv_meta.host_cu_seqlens_padded, cp_size_p2p),
                ],
            )

        # ulysess all2all
        if self.qkv_format == "thd":  # thd -> 1,thd
            out = out.unsqueeze(0)
        out_layer = _SeqAllToAll.apply(self.pg_a2a, out, seq_dim, 2, batch_dim)
        if self.qkv_format == "thd":
            out_layer = out_layer.squeeze(0)

        return out_layer, lse
