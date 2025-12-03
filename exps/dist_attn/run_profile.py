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
from datetime import timedelta
from enum import Enum

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import magi_attention
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
from exps.dist_attn.benchmark.mask import MaskIterator
from magi_attention.api import (
    calc_attn,
    compute_pad_size,
    magi_attn_flex_dispatch,
    undispatch,
)
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import DistAttnConfig
from magi_attention.meta.solver.dispatch_solver import (
    DispatchConfig,
    MinHeapDispatchAlg,
)
from magi_attention.meta.solver.overlap_solver import OverlapConfig, UniformOverlapAlg


class AttnImpl(Enum):
    ULYSSESS = 1
    RING_P2P = 2
    RING_ALLGATHER = 3
    USP = 4
    LOONGTRAIN = 5
    MAGI_ATTENTION = 6


# attention params
SEED = 42
TOTAL_SEQLEN = 512 * 1024
Q_HEADS = 48
KV_HEADS = 16
EMBED_DIM = 1024
HIDDEN_SIZE = 128
DTYPE = torch.bfloat16
WORLD_SIZE = 64
ATTN_IMPL = AttnImpl.MAGI_ATTENTION
ATTN_BACKEND = AttnBackend.FA3

# mask params
MASK_NUMS = 1
MASK_TYPE = FlashMaskType.FULL_DOCUMENT
ITERATION = 10

# Optional baseline params (except magi)
DROPOUT = 0.0
SOFTMAX_SCALE = None
DETERMINISTIC = False
CP_PG_META = {
    ParallelMode.RING: 8,
    ParallelMode.ULYSESS: 8,
}

# Optional Magi params
DISPATCH_ALG = MinHeapDispatchAlg()
CHUNK_SIZE = 512


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
        device_shard = init_distributed(world_size=world_size, pg_meta=None)
        cp_group = get_loongtrain_pg(cp_pg_meta, window_num, rank)
    elif attn_impl == AttnImpl.MAGI_ATTENTION:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30),
        )
        local_rank = rank % 8
        torch.cuda.set_device(local_rank)
        if magi_attention.comm.is_hierarchical_comm_enable():
            cp_group = None
        else:
            cp_group = dist.new_group(list(range(world_size)), backend="nccl")

    return cp_group


def init_hierarchical_mesh(world_size: int):
    if magi_attention.comm.is_hierarchical_comm_enable() and world_size in (
        8,
        16,
        32,
        64,
    ):
        world_size_inter_node, world_size_intra_node = {
            8: (1, 8),
            16: (2, 8),
            32: (4, 8),
            64: (8, 8),
        }[world_size]
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size_inter_node, world_size_intra_node),
            mesh_dim_names=("inter", "intra"),
        )
    else:
        device_mesh = None

    return device_mesh


def run_dist_attn(
    seed: int,
    total_seqlen: int,
    embed_dim: int,
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

    x = torch.randn(total_seqlen, embed_dim, dtype=dtype, device=device)

    q_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )
    k_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    v_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    dout_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )

    x.requires_grad_(True)

    # -----    dispatch   ---- #

    # HACK dispatch only support (t, h, d)
    # assert embed_dim % 2 == 0, "only support (t, h, d) in dispatch, so embed_dim must divided 2 in zero"
    x = x.view(total_seqlen, 1, embed_dim)

    x_local = attn.dispatch(x, q_ranges, total_seqlen, "q")
    _ = attn.dispatch(x, k_ranges, total_seqlen, "k")
    _ = attn.dispatch(x, k_ranges, total_seqlen, "v")
    _ = attn.dispatch(x, q_ranges, total_seqlen, "dout")

    x_local = x_local.view(-1, embed_dim)

    # -----   projection ----- #

    q_local = q_proj(x_local).view(-1, q_heads, hidden_size)
    k_local = k_proj(x_local).view(-1, kv_heads, hidden_size)
    v_local = v_proj(x_local).view(-1, kv_heads, hidden_size)
    dout_local = dout_proj(x_local).view(-1, q_heads, hidden_size)

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


def run_magi_attn(
    total_seqlen: int,
    embed_dim: int,
    q_heads: int,
    kv_heads: int,
    hidden_size: int,
    dtype,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    world_size: int,
    chunk_size: int,
    attn_mask_type: list[AttnMaskType],
    attn_impl: AttnImpl,
    cp_group,
    iteration: int,
):
    assert attn_impl == AttnImpl.MAGI_ATTENTION

    rank = int(os.environ.get("RANK", 0))
    device = torch.cuda.current_device()

    # -----    init test data   ---- #

    x = torch.randn(total_seqlen, embed_dim, dtype=dtype, device=device)

    q_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )
    k_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    v_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    dout_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )

    x.requires_grad_(True)

    # -----   init dispatch mata ----- #

    pad_size = compute_pad_size(
        total_seqlen_q=total_seqlen,
        cp_size=world_size,
        chunk_size=chunk_size,
    )

    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(alg=DISPATCH_ALG),
        overlap_config=OverlapConfig(
            enable=True,
            mode=AttnOverlapMode.STATIC,
            degree=2,
            min_chunk_size=512,
            max_num_chunks=64,
            alg=UniformOverlapAlg(
                random_costs=True,
                random_seed=42,
            ),
        ),
    )

    cp_mesh = init_hierarchical_mesh(world_size)

    # -----    dispatch   ---- #

    (
        x_local,
        magi_attn_runtime_key,
    ) = magi_attn_flex_dispatch(  # local_x with shape (total_seqlen_q + pad_size) / cp_size, h)
        x,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen,
        total_seqlen_k=total_seqlen,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_mesh
        if magi_attention.comm.is_hierarchical_comm_enable()
        else cp_group,
        dist_attn_config=dist_attn_config,
    )

    # -----   projection  ----- #

    q_local = q_proj(x_local).view(-1, q_heads, hidden_size)
    k_local = k_proj(x_local).view(-1, kv_heads, hidden_size)
    v_local = v_proj(x_local).view(-1, kv_heads, hidden_size)
    dout_local = dout_proj(x_local).view(-1, q_heads, hidden_size)

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

        out_local, _ = calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)
        out_local.backward(dout_local, retain_graph=True)

    # ----- undispatch ----- #

    _ = undispatch(out_local, magi_attn_runtime_key)


def run_benchmark(
    mask_nums: int,
    mask_type: FlashMaskType,
    seed: int = 42,
):
    set_seed(seed)
    mask_iterator = MaskIterator(
        num_iterations=mask_nums,
        mask_type=mask_type,
        total_seqlen=TOTAL_SEQLEN,
        data_path="./benchmark/datasets/default/doc_length_distribution.csv",
        chunk_ratio=0.25,
        is_binned=True,
        to_attn_ranges=True,
        seed=seed,
    )
    cp_group = init_dist_environment(
        attn_impl=ATTN_IMPL,
        world_size=WORLD_SIZE,
        cp_pg_meta=CP_PG_META,
    )
    for q_ranges, k_ranges, attn_mask_type, _ in mask_iterator:
        if ATTN_IMPL != AttnImpl.MAGI_ATTENTION:
            run_dist_attn(
                seed=SEED,
                total_seqlen=TOTAL_SEQLEN,
                embed_dim=EMBED_DIM,
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
        else:
            run_magi_attn(
                total_seqlen=TOTAL_SEQLEN,
                embed_dim=EMBED_DIM,
                q_heads=Q_HEADS,
                kv_heads=KV_HEADS,
                hidden_size=HIDDEN_SIZE,
                dtype=DTYPE,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                world_size=WORLD_SIZE,
                chunk_size=CHUNK_SIZE,
                attn_mask_type=attn_mask_type,
                attn_impl=ATTN_IMPL,
                cp_group=cp_group,
                iteration=ITERATION,
            )


if __name__ == "__main__":
    run_benchmark(
        mask_nums=MASK_NUMS,
        mask_type=MASK_TYPE,
        seed=SEED,
    )
