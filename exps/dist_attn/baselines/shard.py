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

import operator
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from exps.dist_attn.baselines.utils_cp import divide_lst
from magi_attention.comm.functional import all_gather_fwd_scatter_bwd


@dataclass
class ParallelMode:
    ULYSESS = "ulysess"
    RING = "ring"
    INTER_WINDOW = "inter_window"
    INTRA_WINDOW = "intra_window"
    DKV_INTER_WINDOW = "dkv_inter_window"
    DKV_INTRA_WINDOW = "dkv_intra_window"


@dataclass
class ShardMeta:
    cu_seqlens: torch.Tensor
    cu_seqlens_padded: torch.Tensor
    host_cu_seqlens: List[int]
    host_cu_seqlens_padded: List[int]
    # restore_shape: torch.Size
    origin_shape: torch.Size
    # max_seqlen: int
    max_seqlen_padded: int


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# init distribute environment
# create DeviceMesh for all pg
# def init_distributed(world_size, pg_meta={}):
#     print(f"world_size: {world_size}, meta info: {pg_meta}")
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#     pg_sizes = tuple(pg_meta.values())
#     pg_names = tuple(pg_meta.keys())
#     assert world_size == reduce(
#         operator.mul, pg_sizes
#     ), "world size does not match pg sizes"
#     rank = int(os.environ.get("RANK", 0))
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))

#     # init dist env
#     dist.init_process_group(
#         backend="nccl",
#         init_method=None,
#         world_size=world_size,
#         rank=rank,
#         timeout=timedelta(minutes=30),
#         store=None,
#     )

#     # init device
#     # device_count = torch.cuda.device_count()
#     # device = dist.get_rank() % device_count
#     # assert local_rank == device, "local rank does not match device"
#     # torch.cuda.set_device(device)
#     # device = torch.cuda.current_device()

#     # init process group
#     mesh = torch.arange(0, world_size).reshape(pg_sizes)
#     deivce_mesh = DeviceMesh("cuda", mesh=mesh, mesh_dim_names=pg_names)

#     return deivce_mesh


def init_distributed(world_size, pg_meta={}):
    print(f"world_size: {world_size}, meta info: {pg_meta}")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if pg_meta is not None:
        pg_sizes = tuple(pg_meta.values())
        pg_names = tuple(pg_meta.keys())
        assert world_size == reduce(
            operator.mul, pg_sizes
        ), "world size does not match pg sizes"
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
    )

    if pg_meta is not None:
        mesh = torch.arange(0, world_size).reshape(pg_sizes)
        deivce_mesh = DeviceMesh("cuda", mesh=mesh, mesh_dim_names=pg_names)
    else:
        deivce_mesh = None

    return deivce_mesh


# basic ring cp group
def get_ring_pg(device_mesh):
    pg = device_mesh.get_group(mesh_dim=ParallelMode.RING)
    return {ParallelMode.RING: pg}


# basic ulysess cp group
def get_ulysess_pg(device_mesh):
    pg = device_mesh.get_group(mesh_dim=ParallelMode.ULYSESS)
    return {ParallelMode.ULYSESS: pg}


# usp cp group
def get_usp_pg(device_mesh):
    p2p_pg = device_mesh.get_group(mesh_dim=ParallelMode.RING)
    a2a_pg = device_mesh.get_group(mesh_dim=ParallelMode.ULYSESS)
    return {ParallelMode.ULYSESS: a2a_pg, ParallelMode.RING: p2p_pg}


# 非正交 group
# def get_loongtrain_pg(device_mesh, window_num, rank):
#     p2p_pg = device_mesh.get_group(mesh_dim=ParallelMode.RING)
#     a2a_pg = device_mesh.get_group(mesh_dim=ParallelMode.ULYSESS)
#     cp_pg = {ParallelMode.ULYSESS: a2a_pg, ParallelMode.RING: p2p_pg}

#     cp_size = dist.get_world_size(p2p_pg)
#     context_ranks = []
#     for i in range(cp_size):
#         context_ranks.append(dist.get_global_rank(p2p_pg, i))
#     assert cp_size % window_num == 0, "cp_size must be divisible by window_num"
#     window_size = cp_size // window_num

#     print(f"{context_ranks=}")

#     # create the intra_window process group when using sliding window
#     for j in range(window_num):
#         intra_window_ranks = context_ranks[j * window_size : (j + 1) * window_size]
#         # intra_window
#         intra_window_group = dist.new_group(intra_window_ranks)
#         if rank in intra_window_ranks:
#             cp_pg[ParallelMode.INTRA_WINDOW] = intra_window_group
#         # dkv_intra_window
#         dkv_intra_window_group = dist.new_group(intra_window_ranks)
#         if rank in intra_window_ranks:
#             cp_pg[ParallelMode.DKV_INTRA_WINDOW] = dkv_intra_window_group

#     # inter_window
#     for j in range(window_size):
#         inter_window_ranks = []
#         for t in range(window_num):
#             inter_window_ranks.append(context_ranks[t * window_size + j])
#         # inter_window
#         inter_window_group = dist.new_group(inter_window_ranks)
#         if rank in inter_window_ranks:
#             print(f"{rank=},{inter_window_ranks=}")
#             cp_pg[ParallelMode.INTER_WINDOW] = inter_window_group
#         # dkv_inter_window
#         dkv_inter_window_group = dist.new_group(inter_window_ranks)
#         if rank in inter_window_ranks:
#             cp_pg[ParallelMode.DKV_INTER_WINDOW] = dkv_inter_window_group

#     return cp_pg


def get_loongtrain_pg(cp_pg_meta, window_num, rank):
    keys = list(cp_pg_meta.keys())
    cp_size_a2a = cp_pg_meta[ParallelMode.ULYSESS]
    cp_size_p2p = cp_pg_meta[ParallelMode.RING]
    num_pgs_a2a = cp_size_p2p
    num_pgs_p2p = cp_size_a2a
    window_size = cp_size_p2p // window_num
    cp_world_size = cp_size_a2a * cp_size_p2p
    ulysess_first = not (keys[0] == ParallelMode.ULYSESS)
    groups = {}

    def get_sliding_window_pg(window_num, context_ranks):
        # context_ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # window_size = 4
        # window_num = 4
        # intra_window = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        # inter_window = [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]

        # create the intra_window process group when using sliding window
        for j in range(window_num):
            intra_window_ranks = context_ranks[j * window_size : (j + 1) * window_size]
            # print(f"{rank=},{intra_window_ranks=}")

            # intra_window
            intra_window_group = dist.new_group(intra_window_ranks)
            if rank in intra_window_ranks:
                groups[ParallelMode.INTRA_WINDOW] = intra_window_group

            # dkv_intra_window
            dkv_intra_window_group = dist.new_group(intra_window_ranks)
            if rank in intra_window_ranks:
                groups[ParallelMode.DKV_INTRA_WINDOW] = dkv_intra_window_group

        # create the inter_window process group when using sliding window
        for j in range(window_size):
            inter_window_ranks = []
            for t in range(window_num):
                inter_window_ranks.append(context_ranks[t * window_size + j])

            # print(f"{rank=},{inter_window_ranks=}")

            # inter_window
            inter_window_group = dist.new_group(inter_window_ranks)
            if rank in inter_window_ranks:
                groups[ParallelMode.INTER_WINDOW] = inter_window_group

            # dkv_inter_window
            dkv_inter_window_group = dist.new_group(inter_window_ranks)
            if rank in inter_window_ranks:
                groups[ParallelMode.DKV_INTER_WINDOW] = dkv_inter_window_group

    if ulysess_first:
        # ULYSESS
        for i in range(num_pgs_a2a):
            ulysess_ranks = list(range(i * cp_size_a2a, (i + 1) * cp_size_a2a))
            ulysess_pg = dist.new_group(ulysess_ranks)
            if rank in ulysess_ranks:
                groups[ParallelMode.ULYSESS] = ulysess_pg
            # print(f"{rank=},{ulysess_ranks=}")

        # RING
        for i in range(num_pgs_p2p):
            ring_ranks = list(range(i, cp_world_size, cp_size_a2a))
            ring_pg = dist.new_group(ring_ranks)
            if rank in ring_ranks:
                groups[ParallelMode.RING] = ring_pg
            get_sliding_window_pg(window_num, ring_ranks)
            # print(f"{rank=},{ring_ranks=}")
    else:
        # RING
        for i in range(num_pgs_p2p):
            ring_ranks = list(range(i * cp_size_p2p, (i + 1) * cp_size_p2p))
            ring_pg = dist.new_group(ring_ranks)
            if rank in ring_ranks:
                groups[ParallelMode.RING] = ring_pg
            get_sliding_window_pg(window_num, ring_ranks)
            # print(f"{rank=},{ring_ranks=}")

        # ULYSESS
        for i in range(num_pgs_a2a):
            ulysess_ranks = list(range(i, cp_world_size, cp_size_p2p))
            ulysess_pg = dist.new_group(ulysess_ranks)
            if rank in ulysess_ranks:
                groups[ParallelMode.ULYSESS] = ulysess_pg

            # print(f"{rank=},{ulysess_ranks=}")

    return groups


############################################################
# dispatch undispatch
############################################################


# thd without pad
# bshd, sbhd
def zigzag_dispatch(
    x_global: torch.Tensor,
    # cu_seqlens: torch.Tensor,
    # cu_seqlens_padded: torch.Tensor,
    host_cu_seqlens: List[int],  # python list
    host_cu_seqlens_padded: List[int],  # python list
    qkv_format,
    cp_group_p2p=None,  # ring pg
    cp_group_a2a=None,  # ulysess pg
):
    assert qkv_format == "thd"
    restore_shape = x_global.shape
    batch_size = len(host_cu_seqlens) - 1

    cp_size_p2p = dist.get_world_size(cp_group_p2p) if cp_group_p2p is not None else -1
    cp_size_a2a = dist.get_world_size(cp_group_a2a) if cp_group_a2a is not None else -1
    cp_rank_p2p = dist.get_rank(cp_group_p2p) if cp_group_p2p is not None else -1
    cp_rank_a2a = dist.get_rank(cp_group_a2a) if cp_group_a2a is not None else -1

    # ring load balance dispatch
    if cp_rank_p2p != -1:
        cu_seqlens_padded_shard = divide_lst(host_cu_seqlens_padded, cp_size_p2p)
        x_shard = _zigzag_dispatch_varlen(
            x_global,
            host_cu_seqlens[:batch_size],
            cu_seqlens_padded_shard[:batch_size],
            host_cu_seqlens,
            host_cu_seqlens_padded,
            cp_size_p2p,
            cp_rank_p2p,
        )
    else:  # ulysess pad
        x_shard = _pad_narrow_seq_dim(x_global, qkv_format, host_cu_seqlens_padded[-1])
    # ulysess dispatch
    seq_dim = 0
    if cp_rank_a2a != -1:
        x_local = torch.chunk(x_shard, cp_size_a2a, dim=seq_dim)[
            cp_rank_a2a
        ].contiguous()
    else:
        x_local = x_shard

    return x_local, restore_shape


def zigzag_undispatch(
    x_local: torch.Tensor,
    # cu_seqlens: torch.Tensor,
    # cu_seqlens_padded: torch.Tensor,
    host_cu_seqlens: List[int],  # python list
    host_cu_seqlens_padded: List[int],  # python list
    qkv_format,
    # restore_shape,
    cp_group_p2p=None,  # ring pg
    cp_group_a2a=None,  # ulysess pg
):
    cp_size_p2p = dist.get_world_size(cp_group_p2p) if cp_group_p2p is not None else -1
    cp_rank_p2p = dist.get_rank(cp_group_p2p) if cp_group_p2p is not None else -1
    cp_rank_a2a = dist.get_rank(cp_group_a2a) if cp_group_a2a is not None else -1

    seq_dim = 0
    if cp_rank_a2a != -1:
        # ulysess all gather
        x_shard = all_gather_fwd_scatter_bwd(
            x_local, cp_group_a2a, dim=seq_dim
        ).contiguous()
    else:
        x_shard = x_local

    if cp_rank_p2p != -1:
        # cu_seqlens_padded_shard = cu_seqlens_padded // cp_size_p2p
        x_global = _zigzag_undispatch_varlen(
            x_shard,
            host_cu_seqlens,
            host_cu_seqlens_padded,
            cp_size_p2p,
            cp_group_p2p,
        )
    else:
        x_global = _pad_narrow_seq_dim(x_shard, qkv_format, host_cu_seqlens[-1])
    return x_global


# pad or narrow data at seq dim
def _pad_narrow_seq_dim(
    input: torch.Tensor,
    qkv_format,
    target_len,
):
    seq_dim = 0 if qkv_format != "bshd" else 1
    seq_len = input.shape[seq_dim]
    if target_len <= seq_len:
        output = input.narrow(seq_dim, 0, target_len)
    else:
        pad = get_pad_dim(qkv_format, target_len - seq_len)
        output = F.pad(input, pad, mode="constant", value=0)
    return output


def _zigzag_dispatch_varlen(
    input: torch.Tensor,
    zigzag_indices_base: List[int],  # indices offset of each seq in original data
    shard_indices_base: List[int],  # indices offset of each seq in shard tensor
    host_cu_seqlens: List[int],  # python list
    host_cu_seqlens_padded: List[int],  # python list
    cp_size_p2p,
    cp_rank_p2p,
):
    device = input.device
    other_shape = input.shape[1:]
    zigzag_indices_np, shard_indices_np = generate_zigzag_dispatch_indices(
        host_cu_seqlens,
        host_cu_seqlens_padded,
        zigzag_indices_base,
        shard_indices_base,
        cp_size_p2p,
        cp_rank_p2p,
        # device,
    )
    zigzag_indices = torch.from_numpy(zigzag_indices_np).to(
        device=device, dtype=torch.int64
    )
    shard_indices = torch.from_numpy(shard_indices_np).to(
        device=device, dtype=torch.int64
    )
    # load balance ring shard
    x_shard = torch.zeros(
        (host_cu_seqlens_padded[-1] // cp_size_p2p, *other_shape),
        device=device,
        dtype=input.dtype,
    )
    # index of x_global
    x_selected = torch.gather(
        input, dim=0, index=zigzag_indices[:, None, None].expand(-1, *other_shape)
    )
    # index of shard tensor
    x_shard.scatter_(
        0, shard_indices[:, None, None].expand(-1, *other_shape), x_selected
    )
    return x_shard  # t,h,d


def _zigzag_undispatch_varlen(
    input: torch.Tensor,
    host_cu_seqlens: List[int],  # python list
    host_cu_seqlens_padded: List[int],  # python list
    cp_size_p2p,
    cp_group_p2p,
):
    device = input.device
    other_shape = input.shape[1:]
    # ring all-gather
    input_shard = all_gather_fwd_scatter_bwd(input, cp_group_p2p, dim=0)

    undispatch_indices_np = generate_zigzag_undispatch_indices(
        host_cu_seqlens_padded, cp_size_p2p, host_cu_seqlens
    )
    undispatch_indices = torch.from_numpy(undispatch_indices_np).to(
        device=device, dtype=torch.int64
    )
    output = torch.gather(
        input_shard,
        dim=0,
        index=undispatch_indices[:, None, None].expand(-1, *other_shape),
    )

    return output


def get_pad_dim(qkv_format, pad_len):
    pad = [0] * (2 * len(qkv_format))
    seq_dim = 0 if qkv_format != "bshd" else 1
    pad[-2 * seq_dim - 1] = pad_len  # seq dim right
    return pad


# zigzag pad 2cp
def get_pad_factor(
    cp_group_p2p=None,  # ring pg
    cp_group_a2a=None,  # ulysess pg
):
    assert (
        cp_group_p2p is not None or cp_group_a2a is not None
    ), "at least one cp group should be provided"
    if cp_group_p2p is not None:
        pad_factor_p2p = 2 * dist.get_world_size(cp_group_p2p)
    else:
        pad_factor_p2p = 1
    if cp_group_a2a is not None:
        pad_factor_a2a = dist.get_world_size(cp_group_a2a)
    else:
        pad_factor_a2a = 1
    pad_factor_a2a *= pad_factor_p2p
    return pad_factor_p2p, pad_factor_a2a


# cu_seqlens_padded
def get_cu_seqlens_padded(
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: List[int],
    qkv_format,
    pad_factor_p2p=1,  # padding factor per seq
    pad_factor_a2a=1,  # padding factor total seq
):
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqlens_padded = _get_seqlens_padded(seqlens, pad_factor_p2p)
    if qkv_format in ["bshd", "sbhd"]:
        max_len = seqlens_padded.max()
        max_len = ((max_len + pad_factor_a2a - 1) // pad_factor_a2a) * pad_factor_a2a
        seqlens_padded.fill_(max_len)
        cu_seqlens_padded = F.pad(
            torch.cumsum(seqlens_padded, dim=0, dtype=torch.int32), (1, 0)
        )
    elif qkv_format == "thd":  # thd
        cu_seqlens_padded = F.pad(
            torch.cumsum(seqlens_padded, dim=0, dtype=torch.int32), (1, 0)
        )
        cu_seqlens_padded[-1] = (
            (cu_seqlens_padded[-1] + pad_factor_a2a - 1) // pad_factor_a2a
        ) * pad_factor_a2a

    cu_seqlens_padded_host = get_cu_seqlens_padded_host(
        cu_seqlens_host, qkv_format, pad_factor_p2p, pad_factor_a2a
    )

    return cu_seqlens_padded, cu_seqlens_padded_host


# cu_seqlens_padded host
def get_cu_seqlens_padded_host(
    cu_seqlens_host: List[int],
    qkv_format: str,
    pad_factor_p2p: int = 1,
    pad_factor_a2a: int = 1,
):
    cu_seqlens_np = np.array(cu_seqlens_host, dtype=np.int32)
    seqlens = cu_seqlens_np[1:] - cu_seqlens_np[:-1]
    seqlens_padded = np.ceil(seqlens / pad_factor_p2p).astype(np.int32) * pad_factor_p2p

    if qkv_format in ["bshd", "sbhd"]:
        max_len = seqlens_padded.max()
        max_len = int(np.ceil(max_len / pad_factor_a2a) * pad_factor_a2a)
        seqlens_padded[:] = max_len
        cu_seqlens_padded = np.concatenate(
            [np.array([0], dtype=np.int32), np.cumsum(seqlens_padded, dtype=np.int32)]
        )
    else:  # thd
        cu_seqlens_padded = np.concatenate(
            [np.array([0], dtype=np.int32), np.cumsum(seqlens_padded, dtype=np.int32)]
        )
        total = cu_seqlens_padded[-1]
        padded_total = int(np.ceil(total / pad_factor_a2a) * pad_factor_a2a)
        cu_seqlens_padded[-1] = padded_total

    return cu_seqlens_padded.tolist()


# pad to padding_factor‘s integer multiple
def _get_seqlens_padded(
    seqlens_in_batch: torch.Tensor, padding_factor: int
) -> torch.Tensor:
    seqlens_padded = (
        (seqlens_in_batch + padding_factor - 1) // (padding_factor) * (padding_factor)
    )
    return seqlens_padded


def get_max_seqlen(host_cu_seqlens: List[int]):
    max_seqlen = max(b - a for a, b in zip(host_cu_seqlens, host_cu_seqlens[1:]))
    return max_seqlen


# ring load balance dispatch indices
def generate_zigzag_dispatch_indices(
    host_cu_seqlens: List[int],
    host_cu_seqlens_padded: List[int],
    zigzag_indices_base: List[int],  # indices offset of each seq in original data
    shard_indices_base: List[int],  # indices offset of each seq in shard tensor
    cp_size: int,
    rank: int,
    # device,
):
    batch_size = len(host_cu_seqlens_padded) - 1
    host_cu_seqlens_np = np.array(host_cu_seqlens, dtype=np.int32)
    host_cu_seqlens_padded_np = np.array(host_cu_seqlens_padded, dtype=np.int32)
    host_seqlens_np = host_cu_seqlens_np[1:] - host_cu_seqlens_np[:-1]
    host_padded_np = host_cu_seqlens_padded_np[1:] - host_cu_seqlens_padded_np[:-1]
    chunk_lens = host_padded_np // (2 * cp_size)
    # cpu implement
    front_start = np.minimum(rank * chunk_lens, host_seqlens_np)
    front_end = np.minimum(front_start + chunk_lens, host_seqlens_np)
    back_start = np.minimum((2 * cp_size - 1 - rank) * chunk_lens, host_seqlens_np)
    back_end = np.minimum(back_start + chunk_lens, host_seqlens_np)

    zigzag_indices_list = []
    shard_indices_list = []
    for i in range(batch_size):
        zigzag_base = zigzag_indices_base[i]
        first_indices = np.arange(front_start[i], front_end[i], dtype=np.int64)
        second_indices = np.arange(back_start[i], back_end[i], dtype=np.int64)
        zigzag_full = np.concatenate([first_indices, second_indices]) + zigzag_base
        zigzag_indices_list.append(zigzag_full)

        if shard_indices_base is not None:
            shard_base = shard_indices_base[i]
            valid_len = len(first_indices) + len(second_indices)
            valid_indices = np.arange(0, valid_len, dtype=np.int64)
            shard_full = valid_indices + shard_base
            shard_indices_list.append(shard_full)

    zigzag_indices_np = np.concatenate(zigzag_indices_list)
    shard_indices_np = None
    if shard_indices_base is not None:
        shard_indices_np = np.concatenate(shard_indices_list)

    # zigzag_indices = torch.from_numpy(zigzag_indices_np).to(device=device)
    # shard_indices = torch.from_numpy(shard_indices_np).to(device=device)

    return zigzag_indices_np, shard_indices_np


# ring load balance zigzag to contiguous indices
def generate_zigzag_undispatch_indices(
    host_cu_seqlens_padded: List[int],
    cp_size: int,
    # device,
    host_cu_seqlens=None,
):
    batch_size = len(host_cu_seqlens_padded) - 1
    host_cu_seqlens_padded = np.array(host_cu_seqlens_padded, dtype=np.int32)
    host_cu_padded_np = host_cu_seqlens_padded[1:] - host_cu_seqlens_padded[:-1]
    chunk_lens = host_cu_padded_np // (2 * cp_size)
    cp_chunk_len = host_cu_seqlens_padded[-1] // cp_size

    indices_lst = []
    for i in range(batch_size):
        if i == 0:
            seq_off = 0
        else:
            seq_off += chunk_lens[i - 1] * 2
        batch_indices = np.empty((2 * cp_size, chunk_lens[i]), dtype=np.int64)
        for rk in range(cp_size):
            offset = rk * cp_chunk_len + seq_off
            head = np.arange(offset, offset + chunk_lens[i], dtype=np.int64)
            tail = np.arange(
                offset + chunk_lens[i], offset + 2 * chunk_lens[i], dtype=np.int64
            )
            batch_indices[2 * rk] = head
            batch_indices[2 * rk + 1] = tail

        reorder_chunk_ids = generate_reorder_chunk_ids_contiguous_np(cp_size)
        reordered_indices = batch_indices[reorder_chunk_ids].reshape(-1)
        if host_cu_seqlens is not None:
            valid_len = host_cu_seqlens[i + 1] - host_cu_seqlens[i]
            reordered_indices = reordered_indices[:valid_len]

        indices_lst.append(reordered_indices)

    total_indices_np = np.concatenate(indices_lst, axis=0)
    return total_indices_np
    # return torch.from_numpy(total_indices).to(device=device, dtype=torch.int64)


# contiguous load balance dispatch indices
# e.g. cp = 4 : [0,7,1,6,2,5,3,4]
def generate_reorder_chunk_ids_zigzag(
    cp_size,
    device,
):
    head = torch.arange(cp_size, device=device)
    tail = torch.arange(2 * cp_size - 1, cp_size - 1, -1, device=device)
    chunk_reorder_ids = torch.stack([head, tail], dim=1).flatten()
    return chunk_reorder_ids


# e.g. cp = 4 : [0,2,4,6,7,5,3,1]
def generate_reorder_chunk_ids_contiguous(
    cp_size,
    device,
):
    first_ids = torch.arange(start=0, end=2 * cp_size, step=2, device=device)
    second_ids = torch.arange(start=2 * cp_size - 1, end=0, step=-2, device=device)
    chunk_reorder_ids = torch.cat([first_ids, second_ids], dim=0)
    return chunk_reorder_ids


def generate_reorder_chunk_ids_contiguous_np(cp_size: int) -> np.ndarray:
    first_ids = np.arange(0, 2 * cp_size, 2, dtype=np.int64)
    second_ids = np.arange(2 * cp_size - 1, 0, -2, dtype=np.int64)
    return np.concatenate([first_ids, second_ids], axis=0)
