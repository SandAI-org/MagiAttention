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

import torch
import torch.distributed as dist
import torch.nn.functional as F
from configuration_llama import LlamaConfig
from llama_pretrain_config import data_config, parallel_config, train_config
from modeling_llama import LlamaDecoderLayer, build_llama3_1b_model
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Shard, distribute_tensor
from torch.optim.lr_scheduler import LinearLR

from magi_attention.api import magi_attn_varlen_dispatch, undispatch
from magi_attention.api.functools import (
    compute_pad_size,
    full_attention_to_varlen_attention,
    squash_batch_dim,
)
from magi_attention.common.enum import AttnOverlapMode
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
    UniformOverlapAlg,
)
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeKey

SEED = 42
CHUNK_SIZE = 512  # chunk_size for magiattention


def _reduce_mean_among_cp(
    partial_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    in_place: bool = False,
) -> torch.Tensor:
    reduced_tensor = partial_tensor.clone().detach()
    reduced_tensor = (
        reduced_tensor.to_local()
        if isinstance(reduced_tensor, DTensor)
        else reduced_tensor
    )

    if device_mesh["cp"].size() > 1:
        reduced_tensor = DTensor.from_local(
            reduced_tensor,
            device_mesh=device_mesh["cp"],
            placements=[Partial(reduce_op="avg")],
        ).full_tensor()

    if in_place:
        if isinstance(partial_tensor, DTensor):
            partial_tensor._local_tensor.copy_(reduced_tensor)
        else:
            partial_tensor.copy_(reduced_tensor)
        return partial_tensor

    return reduced_tensor


def _reduce_mean_among_dp(
    partial_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    in_place: bool = False,
) -> torch.Tensor:
    reduced_tensor = partial_tensor.clone().detach()
    reduced_tensor = (
        reduced_tensor.to_local()
        if isinstance(reduced_tensor, DTensor)
        else reduced_tensor
    )

    if device_mesh["dp"].size() > 1:
        reduced_tensor = DTensor.from_local(
            reduced_tensor,
            device_mesh=device_mesh["dp"],
            placements=[Partial(reduce_op="avg")],
        ).full_tensor()

    if in_place:
        if isinstance(partial_tensor, DTensor):
            partial_tensor._local_tensor.copy_(reduced_tensor)
        else:
            partial_tensor.copy_(reduced_tensor)
        return partial_tensor

    return reduced_tensor


def logger(message: str, rank=None):
    if rank is None:
        print(f"rank{dist.get_rank()}:   " + message)
    else:
        if dist.get_rank() == rank:
            print(f"rank{rank}:   " + message)


def _shard_along_batch_dim_among_dp(
    global_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
) -> torch.Tensor:
    sharded_tensor = global_tensor.clone().detach()

    if device_mesh["dp"].size() > 1:
        sharded_tensor = distribute_tensor(
            sharded_tensor,
            device_mesh["dp"],  # group for fsdp
            placements=[Shard(0)],  # shard batchsize at dim 0
        ).to_local()

    return sharded_tensor


def init_env(backend):
    """
    Init distributed environment
    """
    assert torch.cuda.is_available(), "cuda is not available"
    device_count = torch.cuda.device_count()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    device_id = dist.get_rank() % device_count
    torch.cuda.set_device(device_id)


def build_mesh():
    device_mesh = torch.arange(0, dist.get_world_size()).reshape(
        dist.get_world_size() // parallel_config["context_parallel_size"],  # dp_size
        parallel_config["context_parallel_size"],
    )

    device_mesh = DeviceMesh(
        device_type="cuda",
        mesh=device_mesh,
        mesh_dim_names=("dp", "cp"),  # set dp-cp 2-dim parallel
    )

    # build the dp_cp mesh from flatten dp mesh and cp mesh
    device_mesh["dp", "cp"]._flatten("dp_cp")
    logger(f"{device_mesh=}", rank=0)

    return device_mesh


def apply_fsdp(model, device_mesh):
    """
    apply fsdp2 for llama model.
    """
    for module in model.modules():
        if isinstance(module, LlamaDecoderLayer):
            fully_shard(module, mesh=device_mesh)
    fully_shard(model, mesh=device_mesh)

    return model


def parallize_model(model, device_mesh):
    # pass dp_cp mesh to fsdp, fsdp will handle the gradient sync of both dp and cp.
    apply_fsdp(model, device_mesh["dp_cp"])


def build_optimizer(model, optimizer_config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )

    lr_schedulers = LinearLR(optimizer=optimizer)
    return optimizer, lr_schedulers


def prepare_data(device_mesh, train_iter):
    seqlen = data_config["seqlen"]
    batch_size = data_config["batch_size"]
    vocab_size = LlamaConfig().vocab_size
    dp_size = device_mesh["dp"].size()

    # set different seed for each iter to ensure different random data.
    torch.manual_seed(SEED + train_iter)

    # ---   prepare and shard input data and label   --- #
    global_input = torch.randint(
        size=(batch_size, seqlen),
        high=vocab_size,
        device=torch.cuda.current_device(),
    )

    logger(f"global data: {global_input.shape=}", rank=0)

    global_label = torch.randint_like(
        global_input,
        high=vocab_size,
    )

    local_input = _shard_along_batch_dim_among_dp(global_input, device_mesh)

    local_label = _shard_along_batch_dim_among_dp(global_label, device_mesh)

    logger(f"data after dp shard: {local_input.shape=}", rank=0)

    # ---   prepare data for magi_attention   --- #
    # We do not need to shard data along seqdim among cp manually.
    # magi_attention do not support data with batch dim.
    local_input = squash_batch_dim(local_input)
    cp_size = parallel_config["context_parallel_size"]
    head_dim = LlamaConfig().head_dim

    logger(f"data after squash batch dim: {local_input.shape=}", rank=0)
    # pad seqlen of input data for better performance.
    pad_size = compute_pad_size(local_input.size(0), cp_size, head_dim, CHUNK_SIZE)
    logger(f"{pad_size=}", rank=0)

    cu_seqlens_q, cu_seqlens_k = full_attention_to_varlen_attention(
        batch_size // dp_size, seqlen
    )

    local_label = squash_batch_dim(local_label)

    return local_input, local_label, cu_seqlens_q, cu_seqlens_k, pad_size


def prepare_magi_attention(input, cu_seqlens_q, cu_seqlens_k, pad_size, cp_group):
    # ---   magi_attn_flex_dispatch   --- #
    # an example of distattnconfig
    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        overlap_config=OverlapConfig(
            enable=True,
            mode=AttnOverlapMode.STATIC,
            degree=4,
            min_chunk_size=13,
            max_num_chunks=52,
            alg=UniformOverlapAlg(
                random_costs=True,
                random_seed=42,
            ),
        ),
        high_bandwith_domain_size=8,
        deterministic=False,
    )

    # you can also use fa_varlen-like varlen dispatch interface directly
    x_padded, dist_attn_runtime_key = magi_attn_varlen_dispatch(
        input,
        cu_seqlens_q,
        cu_seqlens_k,
        head_dim=LlamaConfig().head_dim,
        pad_size=pad_size,
        chunk_size=CHUNK_SIZE,
        cp_group=cp_group,
        causal=LlamaConfig().is_causal,
        dist_attn_config=dist_attn_config,
    )

    return x_padded, dist_attn_runtime_key


def loss_func(
    output, label, device_mesh, magi_attention_runtime_key: DistAttnRuntimeKey | None
):
    # since input's dispatched but not label,
    # we need to undispatch the ouput
    # along seqlen among cp group using magi_attention undispatch
    if magi_attention_runtime_key is not None:
        output = undispatch(
            output, magi_attention_runtime_key
        )  # output shape is (s, v)

    vocab_size = LlamaConfig().vocab_size
    loss = F.cross_entropy(
        output.view(-1, vocab_size),
        label.view(-1),
        reduction="mean",
    )

    # ---   reducemean loss and print   --- #
    loss_reduced = _reduce_mean_among_cp(
        loss,
        device_mesh,
    )

    loss_reduced = _reduce_mean_among_dp(
        loss_reduced,
        device_mesh,
    )

    logger(f"{loss_reduced=}", rank=0)

    return loss


def train(model, optimizer, lr_scheduler, device_mesh, train_iter):
    """main training loop"""
    model.train()

    for iter in range(train_iter):
        logger(f" -----   iter{iter}   ---  ", rank=0)
        logger("", rank=0)
        input, label, cu_seqlens_q, cu_seqlens_k, pad_size = prepare_data(
            device_mesh, train_iter
        )

        dist_attn_runtime_key = None

        if (
            parallel_config["context_parallel_size"] > 1
            and parallel_config["context_parallel_backend"] == "magi_attention"
        ):
            # dispatched input
            input, dist_attn_runtime_key = prepare_magi_attention(
                input, cu_seqlens_q, cu_seqlens_k, pad_size, device_mesh.get_group("cp")
            )

        output = model(input, dist_attn_runtime_key)

        loss = loss_func(output, label, device_mesh, dist_attn_runtime_key)

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        logger("", rank=0)


def clean():
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # ---   initialize distributed env   --- #
    init_env(backend="nccl")

    # ---   build device mesh  --- #
    device_mesh = build_mesh()

    # ---   set seed   --- #
    torch.manual_seed(SEED)

    # ---   build llama model  --- #
    model = build_llama3_1b_model()
    model = model.to(torch.cuda.current_device())
    logger(f"Llama model: {model=}", rank=0)

    # --   apply parallisim(fsdp + magi_attention)   --- #
    parallize_model(model, device_mesh)
    logger(f"Llama parallize model: {model=}", rank=0)

    # ---   build optimizer and lr_scheduler   --- #
    optimizer, lr_scheduler = build_optimizer(model, train_config["optimizer_config"])

    # ---   main training loop   --- #
    train(model, optimizer, lr_scheduler, device_mesh, train_config["train_iters"])

    clean()
