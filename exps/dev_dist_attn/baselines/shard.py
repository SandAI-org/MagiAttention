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
    restore_shape: torch.Size
    max_seqlen: int
    max_seqlen_padded: int


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# init distribute environment
# create DeviceMesh for all pg
def init_distributed(world_size, pg_meta={}):
    print(f"world_size: {world_size}, meta info: {pg_meta}")
    pg_sizes = tuple(pg_meta.values())
    pg_names = tuple(pg_meta.keys())
    assert world_size == reduce(
        operator.mul, pg_sizes
    ), "world size does not match pg sizes"
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

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

    # init process group
    mesh = torch.arange(0, world_size).reshape(pg_sizes)
    deivce_mesh = DeviceMesh("cuda", mesh=mesh, mesh_dim_names=pg_names)

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
def get_loongtrain_pg(device_mesh, window_num, rank):
    p2p_pg = device_mesh.get_group(mesh_dim=ParallelMode.RING)
    a2a_pg = device_mesh.get_group(mesh_dim=ParallelMode.ULYSESS)
    cp_pg = {ParallelMode.ULYSESS: a2a_pg, ParallelMode.RING: p2p_pg}

    cp_size = dist.get_world_size(p2p_pg)
    context_ranks = dist.get_global_rank(p2p_pg)
    assert cp_size % window_num == 0, "cp_size must be divisible by window_num"
    window_size = cp_size // window_num

    # create the intra_window process group when using sliding window
    for j in range(window_num):
        intra_window_ranks = context_ranks[j * window_size : (j + 1) * window_size]
        # intra_window
        intra_window_group = dist.new_group(intra_window_ranks)
        if rank in intra_window_ranks:
            cp_pg[ParallelMode.INTRA_WINDOW] = intra_window_group
        # dkv_intra_window
        dkv_intra_window_group = dist.new_group(intra_window_ranks)
        if rank in intra_window_ranks:
            cp_pg[ParallelMode.DKV_INTRA_WINDOW] = dkv_intra_window_group

    # inter_window
    for j in range(window_size):
        inter_window_ranks = []
        for t in range(window_num):
            inter_window_ranks.append(context_ranks[t * window_size + j])
        # inter_window
        inter_window_group = dist.new_group(inter_window_ranks)
        if rank in inter_window_ranks:
            cp_pg[ParallelMode.INTER_WINDOW] = inter_window_group
        # dkv_inter_window
        dkv_inter_window_group = dist.new_group(inter_window_ranks)
        if rank in inter_window_ranks:
            cp_pg[ParallelMode.DKV_INTER_WINDOW] = dkv_inter_window_group

    return cp_pg


############################################################
# dispatch undispatch
############################################################


# thd without pad
# bshd, sbhd
def zigzag_dispatch(
    x_global: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    host_cu_seqlens: List[int],  # python list
    host_cu_seqlens_padded: List[int],  # python list
    qkv_format,
    cp_group_p2p=None,  # ring pg
    cp_group_a2a=None,  # ulysess pg
):
    restore_shape = x_global.shape
    batch_size = cu_seqlens.shape[0] - 1

    cp_size_p2p = dist.get_world_size(cp_group_p2p) if cp_group_p2p is not None else -1
    cp_size_a2a = dist.get_world_size(cp_group_a2a) if cp_group_a2a is not None else -1
    cp_rank_p2p = dist.get_rank(cp_group_p2p) if cp_group_p2p is not None else -1
    cp_rank_a2a = dist.get_rank(cp_group_a2a) if cp_group_a2a is not None else -1

    # ring load balance dispatch
    if cp_rank_p2p != -1:
        cu_seqlens_padded_shard = cu_seqlens_padded // cp_size_p2p
        if qkv_format == "thd":  # thd
            x_shard = _zigzag_dispatch_varlen(
                x_global,
                cu_seqlens[:batch_size],
                cu_seqlens_padded_shard[:batch_size],
                host_cu_seqlens,
                host_cu_seqlens_padded,
                cp_size_p2p,
                cp_rank_p2p,
            )
        else:  # bshd, sbhd
            x_shard = _zigzag_dispatch_non_varlen(
                x_global,
                host_cu_seqlens_padded[-1] - host_cu_seqlens_padded[-2],
                qkv_format,
                cp_size_p2p,
                cp_rank_p2p,
            )
    else:  # ulysess pad
        if qkv_format == "thd":
            x_shard = _pad_narrow_seq_dim(
                x_global, qkv_format, host_cu_seqlens_padded[-1]
            )
        else:
            x_shard = _pad_narrow_seq_dim(
                x_global,
                qkv_format,
                host_cu_seqlens_padded[-1] - host_cu_seqlens_padded[-2],
            )
    # ulysess dispatch
    seq_dim = 0 if qkv_format != "bshd" else 1
    if cp_rank_a2a != -1:
        x_local = torch.chunk(x_shard, cp_size_a2a, dim=seq_dim)[
            cp_rank_a2a
        ].contiguous()
    else:
        x_local = x_shard

    return x_local, restore_shape


def zigzag_undispatch(
    x_local: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    host_cu_seqlens: List[int],  # python list
    host_cu_seqlens_padded: List[int],  # python list
    qkv_format,
    restore_shape,
    cp_group_p2p=None,  # ring pg
    cp_group_a2a=None,  # ulysess pg
):
    cp_size_p2p = dist.get_world_size(cp_group_p2p) if cp_group_p2p is not None else -1
    cp_rank_p2p = dist.get_rank(cp_group_p2p) if cp_group_p2p is not None else -1
    cp_rank_a2a = dist.get_rank(cp_group_a2a) if cp_group_a2a is not None else -1

    seq_dim = 0 if qkv_format != "bshd" else 1
    if cp_rank_a2a != -1:
        # ulysess all gather
        x_shard = all_gather_fwd_scatter_bwd(
            x_local, cp_group_a2a, dim=seq_dim
        ).contiguous()
    else:
        x_shard = x_local

    if cp_rank_p2p != -1:
        if qkv_format == "thd":
            # cu_seqlens_padded_shard = cu_seqlens_padded // cp_size_p2p
            x_global = _zigzag_undispatch_varlen(
                x_shard,
                host_cu_seqlens,
                host_cu_seqlens_padded,
                cp_size_p2p,
                cp_group_p2p,
            )
        else:
            x_global = _zigzag_undispatch_non_varlen(
                x_shard, qkv_format, cp_size_p2p, cp_group_p2p
            )
            x_global = _pad_narrow_seq_dim(x_global, qkv_format, restore_shape[seq_dim])
    else:
        if qkv_format == "thd":
            x_global = _pad_narrow_seq_dim(x_shard, qkv_format, host_cu_seqlens[-1])
        else:
            x_global = _pad_narrow_seq_dim(x_shard, qkv_format, restore_shape[seq_dim])
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
    zigzag_indices_base: torch.Tensor,  # indices offset of each seq in original data
    shard_indices_base: torch.Tensor,  # indices offset of each seq in shard tensor
    host_cu_seqlens: List[int],  # python list
    host_cu_seqlens_padded: List[int],  # python list
    cp_size_p2p,
    cp_rank_p2p,
):
    device = input.device
    other_shape = input.shape[1:]
    zigzag_indices, shard_indices = generate_zigzag_dispatch_indices(
        host_cu_seqlens,
        host_cu_seqlens_padded,
        zigzag_indices_base,
        shard_indices_base,
        cp_size_p2p,
        cp_rank_p2p,
        device,
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


def _zigzag_dispatch_non_varlen(
    input: torch.Tensor, target_len, qkv_format, cp_size_p2p, cp_rank_p2p
):
    seq_dim = qkv_format.index("s")
    other_shape = input.shape[2:]
    first_idx, second_idx = cp_rank_p2p, 2 * cp_size_p2p - cp_rank_p2p - 1
    pad_input = _pad_narrow_seq_dim(input, qkv_format, target_len)
    if qkv_format == "bshd":  # b,s,h,d -> b,2cp,s',h,d
        batch_size = pad_input.shape[0]
        pad_input = pad_input.view(batch_size, 2 * cp_size_p2p, -1, *other_shape)
        chunk0 = pad_input[:, first_idx, ...].contiguous()
        chunk1 = pad_input[:, second_idx, ...].contiguous()
    elif qkv_format == "sbhd":  # s,b,h,d -> 2cp,s',b,h,d
        batch_size = pad_input.shape[1]
        pad_input = pad_input.view(2 * cp_size_p2p, -1, batch_size, *other_shape)
        chunk0 = pad_input[first_idx, ...].contiguous()
        chunk1 = pad_input[second_idx, ...].contiguous()
    x_shard = torch.cat([chunk0, chunk1], dim=seq_dim)
    return x_shard  # b,s',h,d or s',b,h,d


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

    undispatch_indices = generate_zigzag_undispatch_indices(
        host_cu_seqlens_padded, cp_size_p2p, device, host_cu_seqlens
    )
    output = torch.gather(
        input_shard,
        dim=0,
        index=undispatch_indices[:, None, None].expand(-1, *other_shape),
    )

    return output


def _zigzag_undispatch_non_varlen(
    input: torch.Tensor, qkv_format, cp_size_p2p, cp_group_p2p
):
    device = input.device
    seq_dim = qkv_format.index("s")
    batch_size = input.shape[0] if qkv_format == "bshd" else input.shape[1]
    other_shape = input.shape[2:]
    # ring all-gather
    input_chaos = all_gather_fwd_scatter_bwd(input, cp_group_p2p, dim=seq_dim)
    # construct reorder ids contiguous
    chunk_reorder_ids = generate_reorder_chunk_ids_contiguous(cp_size_p2p, device)
    # b,cp,s,h,d or cp,s,b,h,d
    if qkv_format == "bshd":
        input_chaos = input_chaos.view(batch_size, 2 * cp_size_p2p, -1, *other_shape)
    else:
        input_chaos = input_chaos.view(2 * cp_size_p2p, -1, batch_size, *other_shape)

    output = torch.index_select(input_chaos, seq_dim, chunk_reorder_ids)
    if qkv_format == "bshd":
        output = output.view(batch_size, -1, *other_shape)
    else:
        output = output.view(-1, batch_size, *other_shape)
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
    else:  # thd
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


# ring load balance dispatch indices
def generate_zigzag_dispatch_indices(
    host_cu_seqlens: List[int],
    host_cu_seqlens_padded: List[int],
    zigzag_indices_base: torch.Tensor,  # indices offset of each seq in original data
    shard_indices_base: torch.Tensor,  # indices offset of each seq in shard tensor
    cp_size: int,
    rank: int,
    device,
):
    batch_size = len(host_cu_seqlens_padded) - 1
    host_cu_seqlens_np = np.array(host_cu_seqlens, dtype=np.int32)
    host_cu_seqlens_padded = np.array(host_cu_seqlens_padded, dtype=np.int32)
    host_cu_padded_np = host_cu_seqlens_padded[1:] - host_cu_seqlens_padded[:-1]
    host_seqlens_np = host_cu_seqlens_np[1:] - host_cu_seqlens_np[:-1]
    chunk_lens = host_cu_padded_np // (2 * cp_size)

    front_start = np.minimum(rank * chunk_lens, host_seqlens_np)
    front_end = np.minimum(front_start + chunk_lens, host_seqlens_np)
    back_start = np.minimum((2 * cp_size - 1 - rank) * chunk_lens, host_seqlens_np)
    back_end = np.minimum(back_start + chunk_lens, host_seqlens_np)

    zigzag_indices_varlen = []
    shard_indices_varlen = []
    for i in range(batch_size):
        zigzag_base = zigzag_indices_base[i]
        shard_base = shard_indices_base[i]
        first_indices = torch.arange(
            start=front_start[i], end=front_end[i], device=device
        )
        second_indices = torch.arange(
            start=back_start[i], end=back_end[i], device=device
        )
        valid_len = front_end[i] - front_start[i] + back_end[i] - back_start[i]
        valid_indices = torch.arange(start=0, end=valid_len, device=device)
        zigzag_indices_varlen.append(
            torch.cat([first_indices + zigzag_base, second_indices + zigzag_base])
        )
        shard_indices_varlen.append(valid_indices + shard_base)

    zigzag_indices = torch.cat(zigzag_indices_varlen, dim=0).to(torch.int64)
    shard_indices = torch.cat(shard_indices_varlen, dim=0).to(torch.int64)
    return zigzag_indices, shard_indices


# ring load balance zigzag to contiguous indices
def generate_zigzag_undispatch_indices(
    host_cu_seqlens_padded: List[int],
    cp_size: int,
    device,
    host_cu_seqlens=None,
):
    batch_size = len(host_cu_seqlens_padded) - 1
    host_cu_seqlens_padded = np.array(host_cu_seqlens_padded, dtype=np.int32)
    host_cu_padded_np = host_cu_seqlens_padded[1:] - host_cu_seqlens_padded[:-1]
    chunk_lens = host_cu_padded_np // (2 * cp_size)
    cp_chunk_len = host_cu_seqlens_padded[-1] // cp_size

    indices_lst = []
    for i in range(batch_size):
        indices_batch_lst = []
        if i == 0:
            seq_off = 0
        else:
            seq_off += chunk_lens[i - 1] * 2
        for rk in range(cp_size):
            offset = rk * cp_chunk_len + seq_off
            indices_head = torch.arange(
                start=offset, end=offset + chunk_lens[i], device=device
            )
            indices_tail = torch.arange(
                start=offset + chunk_lens[i],
                end=offset + 2 * chunk_lens[i],
                device=device,
            )
            indices_batch_lst.append(indices_head)
            indices_batch_lst.append(indices_tail)
        indices_batch = torch.cat(indices_batch_lst, dim=0)
        reorder_chunk_ids = generate_reorder_chunk_ids_contiguous(cp_size, device)
        indices_contigious = torch.index_select(
            indices_batch.view(2 * cp_size, -1), dim=0, index=reorder_chunk_ids
        )
        indices_contigious = indices_contigious.view(-1)
        if host_cu_seqlens is not None:
            valid_len = host_cu_seqlens[i + 1] - host_cu_seqlens[i]
            indices_contigious = indices_contigious[:valid_len]

        indices_lst.append(indices_contigious)

    total_indices = torch.cat(indices_lst, dim=0).to(torch.int64)
    return total_indices


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
