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
from enum import Enum

import torch
import torch.distributed as dist

from exps.dist_attn.baselines.loongtrain import LoongTrain
from exps.dist_attn.baselines.ring_attn import RingAttnAllGather, RingAttnP2P
from exps.dist_attn.baselines.shard import (
    ParallelMode,
    get_loongtrain_pg,
    get_ring_pg,
    get_ulysess_pg,
    get_usp_pg,
    init_distributed,
    set_seed,
)
from exps.dist_attn.baselines.ulysess import Ulysess
from exps.dist_attn.baselines.usp import USP
from exps.dist_attn.baselines.utils_cp import AttnBackend
from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.mask import MaskGenerator
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges


class AttnImpl(Enum):
    ULYSSESS = 1
    RING_P2P = 2
    RING_ALLGATHER = 3
    USP = 4
    LOONGTRAIN = 5


SEED = 42
TOTAL_SEQLEN = 64 * 1024
Q_HEADS = 48
KV_HEADS = 8
HIDDEN_SIZE = 128
DTYPE = torch.bfloat16
DROPOUT = 0.0
SOFTMAX_SCALE = None
DETERMINISTIC = False
WORLD_SIZE = 8
CP_PG_META = {
    ParallelMode.RING: 8,
    # ParallelMode.ULYSESS: 4,
    # ParallelMode.RING: 2,
}
ATTN_IMPL = AttnImpl.RING_P2P
ATTN_BACKEND = AttnBackend.FA3
MASK_NUMS = 1
MASK_TYPE = FlashMaskType.FULL_DOCUMENT
ITERATION = 10


def init_dist_environment(
    attn_impl: AttnImpl,
    world_size: int,
    cp_pg_meta,
):
    rank = int(os.environ.get("RANK", 0))

    # -----    test ring or all-gather   ---- #
    if attn_impl == AttnImpl.RING_ALLGATHER or attn_impl == AttnImpl.RING_P2P:
        # cp_pg_meta = {
        #     ParallelMode.RING: 4,
        # }
        # world_size = 4
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_ring_pg(device_shard)

    # -----    test ulysess   ---- #
    elif attn_impl == AttnImpl.ULYSSESS:
        # cp_pg_meta = {
        #     ParallelMode.ULYSESS: 4,
        # }
        # world_size = 4
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_ulysess_pg(device_shard)

    # -----    test usp   ---- #
    elif attn_impl == AttnImpl.USP:
        # cp_pg_meta = {
        #     ParallelMode.ULYSESS: 2,
        #     ParallelMode.RING: 2,
        # }
        # ulysess [0,1] or ring [0,1]
        # cp_pg_meta = {
        #     ParallelMode.RING: 2,
        #     ParallelMode.ULYSESS: 2,
        # }
        # world_size = 4
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_usp_pg(device_shard)
    elif attn_impl == AttnImpl.LOONGTRAIN:
        # cp_pg_meta = {
        #     ParallelMode.ULYSESS: 1,
        #     ParallelMode.RING: 4,
        # }
        # cp_pg_meta = {
        #     ParallelMode.RING: 4,
        #     ParallelMode.ULYSESS: 1,
        # }
        # world_size = 4
        # NOTE: param for loongtrain double ring-attention
        window_num = 2
        # assert world_size % window_num == 0
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_loongtrain_pg(device_shard, window_num, rank)

    return cp_group


def run_dist_attn(
    seed: int,
    total_seqlen: int,
    q_heads: int,
    kv_heads: int,
    hidden_size: int,
    dtype,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    dropout: float,
    softmax_scale: float,
    deterministic: bool,
    world_size: int,
    attn_mask_type: AttnMaskType,
    cp_pg_meta,
    attn_impl: AttnImpl,
    attn_backend: AttnBackend,
    cp_group,
    iteration: int,
):
    rank = int(os.environ.get("RANK", 0))
    device = torch.cuda.current_device()

    # -----    init attn module   ---- #

    if attn_impl == AttnImpl.RING_ALLGATHER:
        attn = RingAttnAllGather(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.RING_P2P:
        attn = RingAttnP2P(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.ULYSSESS:
        attn = Ulysess(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [device]
    elif attn_impl == AttnImpl.USP:
        attn = USP(cp_process_group=cp_group, qkv_format="thd", backend=attn_backend)  # type: ignore[assignment]
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.LOONGTRAIN:
        attn = LoongTrain(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]

    # -----    init test data   ---- #

    q = torch.randn(total_seqlen, q_heads, hidden_size, dtype=dtype, device=device)
    k = torch.randn(total_seqlen, kv_heads, hidden_size, dtype=dtype, device=device)
    v = torch.randn(total_seqlen, kv_heads, hidden_size, dtype=dtype, device=device)
    dout = torch.randn(total_seqlen, q_heads, hidden_size, dtype=dtype, device=device)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    # -----    dispatch   ---- #

    q_local = attn.dispatch(q, q_ranges, total_seqlen, "q")
    k_local = attn.dispatch(k, k_ranges, total_seqlen, "k")
    v_local = attn.dispatch(v, k_ranges, total_seqlen, "v")
    dout_local = attn.dispatch(dout, q_ranges, total_seqlen, "dout")

    # -----   pre_compute ---- #

    attn.pre_compute_attn_runtime_meta(*cal_runtime_args)

    # -----    forward   ---- #

    for i in range(iteration):
        if rank == 0 and i == 6:
            torch.cuda.profiler.start()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
        if rank == 0 and i == 9:
            torch.cuda.profiler.stop()

        # -----    barrier at the beginning of each iteration   ---- #

        dist.barrier()
        torch.cuda.synchronize()

        out, lse = attn.apply_attn(
            q_local,
            k_local,
            v_local,
            attn_mask_type,
            dropout,
            softmax_scale,
            deterministic,
        )
        out.backward(dout_local, retain_graph=True)

    # -----    undispatch   ---- #

    _ = attn.undispatch(out, "q")


def run_benchmark(
    mask_nums: int,
    mask_type: FlashMaskType,
    seed: int = 42,
):
    set_seed(seed)
    mask_generator = MaskGenerator(
        generate_times=mask_nums,
        generate_mask=mask_type,
        total_seqlen=TOTAL_SEQLEN,
        to_attn_ranges=True,
    )
    cp_group = init_dist_environment(
        attn_impl=ATTN_IMPL,
        world_size=WORLD_SIZE,
        cp_pg_meta=CP_PG_META,
    )
    for q_ranges, k_ranges, attn_mask_type in mask_generator:
        run_dist_attn(
            seed=SEED,
            total_seqlen=TOTAL_SEQLEN,
            q_heads=Q_HEADS,
            kv_heads=KV_HEADS,
            hidden_size=HIDDEN_SIZE,
            dtype=DTYPE,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            dropout=DROPOUT,
            softmax_scale=SOFTMAX_SCALE,  # type: ignore
            deterministic=DETERMINISTIC,
            world_size=WORLD_SIZE,
            attn_mask_type=attn_mask_type[0],
            cp_pg_meta=CP_PG_META,
            attn_impl=ATTN_IMPL,
            attn_backend=ATTN_BACKEND,
            cp_group=cp_group,
            iteration=ITERATION,
        )


if __name__ == "__main__":
    run_benchmark(
        mask_nums=MASK_NUMS,
        mask_type=MASK_TYPE,
        seed=SEED,
    )
