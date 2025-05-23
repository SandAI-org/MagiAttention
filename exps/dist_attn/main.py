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
import random
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from magi_attention import init_dist_attn_runtime_mgr
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    OverlapConfig,
    UniformOverlapAlg,
)
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeMgr
from magi_attention.meta.solver.dispatch_solver import ToppHeapDispatchAlg
from magi_attention.utils import nvtx

TP_SIZE = 8
CP_SIZE = None
NUM_SAMPLES = 100
FULL_ATTN = False

if __name__ == "__main__":
    # -------------------       setup env   ------------------- #
    # set random seed to eliminate randomness in training
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # env config
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    prof_iters, prof_start_iter, prof_end_iter = 10, 4, 6

    assert world_size % TP_SIZE == 0, "world size must be divisible by TP_SIZE"
    CP_SIZE = world_size // TP_SIZE

    print(f"world_size: {world_size}, CP_SIZE: {CP_SIZE}, TP_SIZE: {TP_SIZE}")

    # init dist env
    dist.init_process_group(
        backend="nccl",
        init_method=None,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
        store=None,
    )

    # init device
    device_count = torch.cuda.device_count()
    device = dist.get_rank() % device_count
    assert local_rank == device, "local rank does not match device"
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()

    mesh = torch.arange(0, world_size).reshape(CP_SIZE, TP_SIZE)
    deivce_mesh = DeviceMesh("cuda", mesh=mesh, mesh_dim_names=("cp", "tp"))

    # init cp group(s)
    nccl_groups = [deivce_mesh.get_group(mesh_dim="cp") for _ in range(2)]

    # -------------------       prepare model   ------------------- #

    # model config
    hidden_size = 1024
    num_heads_q = 6
    num_heads_k = 1
    head_dim = 128
    dtype = torch.bfloat16

    w = torch.randn(hidden_size, num_heads_q * head_dim, device=device, dtype=dtype)

    # -------------------       prepare data   ------------------- #

    # data config
    total_seqlen = 1024 * 1000

    # block size config
    head_dim_to_q_block_size = {
        64: 128,
        128: 80,
        256: 64,
    }
    q_block_size = head_dim_to_q_block_size[head_dim]

    # mask config
    if FULL_ATTN:
        q_ranges = AttnRanges.from_ranges([[0, total_seqlen]])
        k_ranges = AttnRanges.from_ranges([[0, total_seqlen]])
    else:
        random_indices = (
            torch.randperm(total_seqlen - 1)[: NUM_SAMPLES - 1] + 1
        ).tolist()
        random_indices = sorted(random_indices)
        random_indices = [0] + random_indices + [total_seqlen]
        cu_seqlens = torch.tensor(random_indices, device="cuda", dtype=torch.int32)
        ranges = torch.stack([cu_seqlens[:-1], cu_seqlens[1:]], dim=1).tolist()
        q_ranges = AttnRanges.from_ranges(ranges)
        k_ranges = AttnRanges.from_ranges(ranges)
    is_causal_mapping = False

    # init global data
    total_q = torch.randn(
        total_seqlen,
        num_heads_q,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    total_k = torch.randn(
        total_seqlen,
        num_heads_k,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    total_v = torch.randn(
        total_seqlen,
        num_heads_k,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    grad_total_out = torch.randn_like(total_q).detach()
    # dist.all_reduce(total_q.data, group=nccl_groups[0])
    # dist.all_reduce(total_k.data, group=nccl_groups[0])
    # dist.all_reduce(total_v.data, group=nccl_groups[0])
    # dist.all_reduce(grad_total_out.data, group=nccl_groups[0])

    # -------------------       init magi_attention   ------------------- #

    # magi_attention config
    chunk_size = q_block_size * 10
    # TODO: test top-p minhp dispatch alg
    dispatch_config = DispatchConfig(alg=ToppHeapDispatchAlg(top_p=1))
    overlap_config = OverlapConfig(
        # enable=False,
        enable=True,
        mode=AttnOverlapMode.STATIC,
        degree=4,
        min_chunk_size=256,
        max_num_chunks=64,
        alg=UniformOverlapAlg(
            random_costs=True,
            random_seed=42,
        ),
    )
    hb_domain_size = 1

    dist_attn_config = DistAttnConfig(
        dispatch_config=dispatch_config,
        overlap_config=overlap_config,
        high_bandwith_domain_size=hb_domain_size,
        deterministic=False,
    )

    # magi_attention mgr
    dist_attn_runtime_mgr: DistAttnRuntimeMgr = init_dist_attn_runtime_mgr(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=[AttnMaskType.FULL] * len(q_ranges),
        total_seqlen_q=total_seqlen,
        total_seqlen_k=total_seqlen,
        chunk_size=chunk_size,
        cp_group=nccl_groups[0],
        is_same_source=True,
        is_q_permutable=True,
        is_k_permutable=True,
        dist_attn_config=dist_attn_config,
    )
    # HACK: seperate cp group for dkv group-reduce
    dist_attn_runtime_mgr.dist_attn_runtime.cp_group_dkv = nccl_groups[1]

    # -------------------       run   ------------------- #

    for iter in range(prof_iters):
        # init for nvtx
        nvtx.switch_profile(
            iter_id=iter,
            start=prof_start_iter,
            end=prof_end_iter,
            profile_ranks=[0],
        )

        dist.barrier()
        torch.cuda.synchronize()

        # -----    dispatch   ---- #

        local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
        local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
        local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)

        # -----    forward   ---- #

        local_out, _ = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)

        # -----    undispatch   ---- #

        total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)

        # -----    backward   ---- #

        total_out.backward(grad_total_out)

    # -------------------       clearup env   ------------------- #

    dist.barrier()
    dist.destroy_process_group()
