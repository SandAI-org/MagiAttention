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


# This script reproduces the Hybrid Context Parallel algorithm introduced in
# Megatron-LM PR #2282, and is used for performance benchmarking and comparison.


import copy
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, log2
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from flash_attn_interface import flash_attn_func
from megatron.core.pipeline_parallel.hybrid_cp_schedule import BalancedCPScheduler

from exps.dist_attn.baselines.interface import AttnBaselineInterface
from exps.dist_attn.baselines.ring_attn import RingAttnP2P
from exps.dist_attn.baselines.shard import (
    ParallelMode,
    _pad_narrow_seq_dim,
    get_group_meta,
)
from exps.dist_attn.baselines.utils_cp import AttnBackend
from magi_attention.comm.functional import all_gather_fwd_scatter_bwd
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges


@dataclass
class HybridShardMeta:
    # total global sample num
    total_samples_num: int
    # seqlens of all global samples
    global_unsharded_seqlens: List[int]
    # seqlens of local dispatch samples
    local_sharded_seqlens: List[int]
    # BalancedCPScheduler results, per micro-batch, per-rank global sample ids
    micro_batch_global_ids: List[List[List[int]]]


class RobustBalancedCPScheduler(BalancedCPScheduler):
    # HACK: override gpus_needed with an upper bound of the CP world size to avoid assertion errors in `fill_empty_gpus`
    @lru_cache(maxsize=128)
    def gpus_needed(self, seq_len: int) -> int:
        """
        Calculates the number of GPUs needed for a given sequence length
        and max sequence length per CP rank.
        This is used to determine the CP size of a sub-sample.

        The number is rounded up to the next power of 2 to match the available
        hybrid context parallel process group sizes.
        """
        return min(
            max(1, 2 ** ceil(log2((seq_len / self.max_seq_len_per_rank)))),
            self.total_hdp_gpus,
        )


class HybridMegatronDCP(AttnBaselineInterface):
    def __init__(
        self,
        cp_process_group: Dict,
        qkv_format: str,
        backend: AttnBackend,
    ):
        # NOTE: We set DP=1 to focus on evaluating CP performance. Imbalance across DP ranks should be handled by DP itself.
        self.world_pg = cp_process_group[ParallelMode.RING]
        self.hybrid_pg = cp_process_group[ParallelMode.HYBRID_SET]
        self.backend = backend
        self.qkv_format = qkv_format
        # global sample id: Attn
        self.micro_batch_attns: Dict[int, RingAttnP2P] | None = None
        self.shard_meta = {}  # type: ignore

    # to call after q,k,v dispatch
    def pre_compute_attn_runtime_meta(self, attn_mask_type: AttnMaskType, device):
        assert (
            self.shard_meta["q"] == self.shard_meta["k"]
        ), "Hybrid DCP acquires same allocation."
        assert (
            self.shard_meta["k"] == self.shard_meta["v"]
        ), "Hybrid DCP acquires same allocation."
        for attn in self.micro_batch_attns.values():  # type: ignore[union-attr]
            if attn is not None:
                attn.pre_compute_attn_runtime_meta(attn_mask_type, device)

    # per group-stage (micro batch) sample dispatch
    def dispatch(
        self,
        x_global: torch.Tensor,
        ranges: AttnRanges,
        valid_total_seqlen: int,  # required by AttnRanges.to_cu_seqlens
        name: str | List[str],  # key names for shard_meta
        **kwargs,
    ):
        # we set max_seqlen_per_rank=8k for our bench.
        cp_rank, _ = get_group_meta(self.world_pg)
        max_seqlen_per_rank = kwargs.get("max_seqlen_per_rank", 8 * 1024)
        # step0: re-allocate all samples to different ranks
        self.cp_balancing_scheduler = RobustBalancedCPScheduler(
            max_seq_len_per_rank=max_seqlen_per_rank, dp_cp_group=self.world_pg
        )
        global_unsharded_seqlens: List[int] = []
        sample_id_seqlens: List[Tuple[int, int]] = []
        total_samples_num = len(ranges)
        cu_seqlens = ranges.to_cu_seqlens(valid_total_seqlen)
        for idx in range(total_samples_num):
            global_unsharded_seqlens.append(cu_seqlens[idx + 1] - cu_seqlens[idx])
            sample_id_seqlens.append((idx, global_unsharded_seqlens[-1]))
        # [stage0-[rk0-ids[], rk1-ids[],...], srage1-[rk0-ids[], rk1-ids[],...]
        (
            _,
            micro_batch_global_ids,
        ) = self.cp_balancing_scheduler.get_groups_and_subsamples(
            sample_id_seqlens, None
        )
        if self.micro_batch_attns is None:
            self.micro_batch_attns = {}
        # split varlen-packed data to all sub-samples
        global_split_samples = torch.split(x_global, global_unsharded_seqlens, dim=0)
        local_samples: List[torch.Tensor] = []
        local_sharded_seqlens: List[int] = []
        # step1: dispatch for each micro-batch group for this rank
        for micro_batch_group in micro_batch_global_ids:
            single_micro_batch_ids = micro_batch_group[cp_rank]
            for sample_id in single_micro_batch_ids:
                # get cp_size for this sample
                sub_cp_size = len(
                    [
                        True
                        for sample_ids_per_rank in micro_batch_group
                        if sample_id in sample_ids_per_rank
                    ]
                )
                sub_cp_pg = self.hybrid_pg.get(sub_cp_size, None)
                assert (
                    sub_cp_pg is not None or sub_cp_size == 1
                ), f"sub_cp_group not found for cp_size={sub_cp_size}"
                if sub_cp_size == 1:
                    # NOTE: for cp_size=1, use fa3 without cp dist attn or dispatch/undispatch
                    local_sharded_sample = global_split_samples[sample_id]
                    self.micro_batch_attns[sample_id] = None  # type: ignore[assignment]
                else:
                    # TODO: maybe pack samples with same group
                    attn = self.micro_batch_attns.get(sample_id, None)
                    if attn is None:
                        attn = RingAttnP2P(
                            {ParallelMode.RING: sub_cp_pg}, "thd", self.backend
                        )
                        self.micro_batch_attns[sample_id] = attn
                    local_sharded_sample = attn.dispatch(
                        global_split_samples[sample_id],
                        AttnRanges.from_ranges(
                            [[0, global_unsharded_seqlens[sample_id]]]
                        ),
                        global_unsharded_seqlens[sample_id],
                        name,
                    )
                local_samples.append(local_sharded_sample)
                local_sharded_seqlens.append(local_sharded_sample.shape[0])
        # step3: save meta for undispatch
        dispatch_keys = name
        if isinstance(name, str):
            dispatch_keys = [name]
        shard_meta = HybridShardMeta(
            total_samples_num=total_samples_num,
            global_unsharded_seqlens=global_unsharded_seqlens,
            local_sharded_seqlens=local_sharded_seqlens,
            micro_batch_global_ids=micro_batch_global_ids,
        )
        for key in dispatch_keys:
            self.shard_meta[key] = copy.deepcopy(shard_meta)
        # we directly discard and do not apply all2all for remote samples since we only have cp groups.
        return local_samples

    def undispatch(
        self,
        local_split_samples: List[torch.Tensor],
        name: str,  # key name for shard_meta
    ) -> torch.Tensor:
        # get meta
        cp_rank, cp_size = get_group_meta(self.world_pg)
        shape = local_split_samples[0].shape
        total_samples_num = self.shard_meta[name].total_samples_num
        global_unsharded_seqlens = self.shard_meta[name].global_unsharded_seqlens
        micro_batch_global_ids = self.shard_meta[name].micro_batch_global_ids
        # flatten global micro-batches for each rank
        flatten_micro_batch_ids: List[List[int]] = [[] for _ in range(cp_size)]
        for micro_batch_group in micro_batch_global_ids:
            for rank in range(cp_size):
                flatten_micro_batch_ids[rank].extend(micro_batch_group[rank])
        local_samples_ids = flatten_micro_batch_ids[cp_rank]
        # step0: undispatch local samples
        local_samples: List[torch.Tensor] = []
        for local_sample_id, local_split_sample in zip(
            local_samples_ids, local_split_samples
        ):
            local_attn = self.micro_batch_attns[local_sample_id]  # type: ignore[index]
            if local_attn is not None:
                local_sample = local_attn.undispatch(local_split_sample, name)
            else:
                local_sample = local_split_sample
            local_samples.append(local_sample)
        # step1: prepare for gather all samples
        global_samples = [None] * total_samples_num
        seqlens_per_rank: List[List[int]] = []
        for rank in range(cp_size):
            seqlens_per_rank.append(
                [global_unsharded_seqlens[idx] for idx in flatten_micro_batch_ids[rank]]
            )
        total_seqlen_per_rank = [
            sum(local_seqlens) for local_seqlens in seqlens_per_rank
        ]
        total_seqlen_padded_per_rank = max(total_seqlen_per_rank)
        all_gather_input = _pad_narrow_seq_dim(
            torch.cat(local_samples, dim=0), "thd", total_seqlen_padded_per_rank
        )
        all_gather_output = all_gather_fwd_scatter_bwd(
            all_gather_input, self.world_pg, dim=0
        ).contiguous()
        # step3: get each sample
        all_gather_output = all_gather_output.view(
            cp_size, total_seqlen_padded_per_rank, *shape[1:]
        )
        for rank in range(cp_size):
            samples_buffer = all_gather_output[rank].narrow(
                dim=0, start=0, length=total_seqlen_per_rank[rank]
            )
            split_samples_buffer = torch.split(
                samples_buffer, seqlens_per_rank[rank], dim=0
            )
            for sample_id, sample in zip(
                flatten_micro_batch_ids[rank], split_samples_buffer
            ):
                if global_samples[sample_id] is None:
                    global_samples[sample_id] = sample

        x_global = torch.cat(global_samples, dim=0)
        return x_global

    def apply_fwd_attn(
        self,
        local_q_samples: List[torch.Tensor],
        local_k_samples: List[torch.Tensor],
        local_v_samples: List[torch.Tensor],
        attn_mask_type: AttnMaskType,
        dropout_p: float,
        softmax_scale: float,
        deterministic: bool,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        cp_rank, _ = get_group_meta(self.world_pg)
        is_causal = attn_mask_type == AttnMaskType.CAUSAL
        if is_causal:
            window_size = (-1, 0)
        else:
            window_size = (-1, -1)
        local_sharded_seqlens = self.shard_meta["q"].local_sharded_seqlens
        micro_batch_global_ids = self.shard_meta["q"].micro_batch_global_ids
        local_samples_num = len(local_sharded_seqlens)
        out_per_step: List[Optional[torch.Tensor]] = [None] * local_samples_num
        lse_per_step: List[Optional[torch.Tensor]] = [None] * local_samples_num
        fa_func_kwargs = {
            "softmax_scale": softmax_scale,
            "causal": is_causal,
            "window_size": window_size,
            "deterministic": deterministic,
            "return_attn_probs": True,
        }
        sample_step = 0
        micro_batch_num = len(micro_batch_global_ids)
        for batch_idx, micro_batch_group in enumerate(micro_batch_global_ids):
            for sample_id in micro_batch_group[cp_rank]:
                attn = self.micro_batch_attns[sample_id]  # type: ignore[index]
                if attn is None:
                    out, lse = flash_attn_func(
                        local_q_samples[sample_step][None, ...],
                        local_k_samples[sample_step][None, ...],
                        local_v_samples[sample_step][None, ...],
                        **fa_func_kwargs,
                    )
                    out, lse = out[0], lse[0]
                else:
                    out, lse = attn.apply_attn(
                        local_q_samples[sample_step],
                        local_k_samples[sample_step],
                        local_v_samples[sample_step],
                        attn_mask_type=attn_mask_type,
                        dropout_p=dropout_p,
                        softmax_scale=softmax_scale,
                        deterministic=deterministic,
                    )
                out_per_step[sample_step] = out
                lse_per_step[sample_step] = lse
                sample_step += 1
            # barrier between two cp group settings, skip barrier for last micro batch
            if batch_idx < micro_batch_num - 1:
                dist.barrier()

        return out_per_step, lse_per_step

    def apply_bwd_attn(
        self,
        out_per_step: List[torch.Tensor],
        dout_per_step: List[torch.Tensor],
        retain_graph: bool = False,
    ):
        cp_rank, _ = get_group_meta(self.world_pg)
        micro_batch_global_ids = self.shard_meta["q"].micro_batch_global_ids
        micro_batch_num = len(micro_batch_global_ids)
        sample_step = 0
        for batch_idx, micro_batch_group in enumerate(micro_batch_global_ids):
            for sample_id in micro_batch_group[cp_rank]:
                out_per_step[sample_step].backward(
                    dout_per_step[sample_step], retain_graph=retain_graph
                )
                sample_step += 1
            # barrier between two cp group settings, skip barrier for last micro batch
            if batch_idx < micro_batch_num - 1:
                dist.barrier()

    def reset(self):
        if self.micro_batch_attns is not None:
            self.micro_batch_attns.clear()
        self.shard_meta.clear()

    def apply_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask_type: AttnMaskType,
        dropout_p: float,
        softmax_scale: float,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        print("Unimplemented.")
        return None, None
