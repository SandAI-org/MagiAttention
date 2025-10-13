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

import torch
import torch.distributed as dist
from magi_attention.utils import nvtx
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs

from magi_fsdp._fsdp_api import MixedPrecisionPolicy
from magi_fsdp._fully_shard import fully_shard
from magi_fsdp.testing.common_fsdp import MultiDtypeTransformer

# from torch.distributed.fsdp import fully_shard
# from torch.distributed.fsdp import MixedPrecisionPolicy


def init_distributed(world_size, is_hsdp=False):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    if is_hsdp:
        mesh = torch.arange(0, world_size).reshape(2, 2)
        mesh_dim_names = ["replica", "shard"]
    else:
        mesh = torch.arange(0, world_size)
        mesh_dim_names = None
    deivce_mesh = DeviceMesh("cuda", mesh=mesh, mesh_dim_names=mesh_dim_names)
    return deivce_mesh


if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device_mesh = init_distributed(world_size, is_hsdp=False)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    ENABLE_MAIN_PARAMS = os.environ.get("MAIN_PARAMS", "0") == "1"
    print(f"{ENABLE_MAIN_PARAMS=}")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    vocab_size = 4096
    model_args = ModelArgs(
        n_layers=16,
        n_heads=32,
        dim=3072,
        vocab_size=vocab_size,
        max_seq_len=4096,
        dropout_p=0,
        use_attn_mask=False,
        weight_tying=False,
        checkpoint_activations=False,
    )

    multi_dtype_config = {
        "TransformerBlock": {
            0: torch.float32,
            1: torch.bfloat16,
            2: torch.float32,
            3: torch.float32,
            4: torch.bfloat16,
            5: torch.float16,
            6: torch.float32,
            7: torch.bfloat16,
            8: torch.float32,
            9: torch.bfloat16,
            10: torch.float32,
            11: torch.float16,
            12: torch.bfloat16,
            13: torch.float32,
            14: torch.float16,
            15: torch.float32,
        },
        "LayerNorm": torch.float32,
    }
    model = MultiDtypeTransformer(model_args, multi_dtype_config)  # type: ignore

    # mp_policy = MixedPrecisionPolicy(
    #     param_dtype=torch.bfloat16,
    #     reduce_dtype=torch.float32,
    # )
    mp_policy = MixedPrecisionPolicy()

    layers0 = [model.layers[0], model.layers[1]]  # fp32,bf16
    fully_shard(layers0, mesh=device_mesh, mp_policy=mp_policy)
    layers1 = [model.layers[2]]  # fp32
    fully_shard(layers1, mesh=device_mesh, mp_policy=mp_policy)
    layers2 = [model.layers[3], model.layers[4], model.layers[5]]  # fp32,bf16,fp16
    fully_shard(layers2, mesh=device_mesh, mp_policy=mp_policy)
    layers3 = [
        model.layers[6],
        model.layers[7],
        model.layers[8],
        model.layers[9],
    ]  # fp32,bf16,fp32,bf16
    fully_shard(layers3, mesh=device_mesh, mp_policy=mp_policy)
    layers4 = [
        model.layers[10],
        model.layers[11],
        model.layers[12],
        model.layers[13],
        model.layers[14],
        model.layers[15],
    ]  # fp32,fp16,bf16,fp32,fp16,bf32
    fully_shard(layers4, mesh=device_mesh, mp_policy=mp_policy)

    # layers = [layer for layer in model.layers]
    # fully_shard(layers, mesh=device_mesh, mp_policy=mp_policy)

    fully_shard(model, mesh=device_mesh, mp_policy=mp_policy)

    lr = 1e-3

    if ENABLE_MAIN_PARAMS:
        main_params = [
            p.detach().clone().float().requires_grad_(True) for p in model.parameters()
        ]
        optim = torch.optim.Adam(main_params, lr=lr)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # forward/backward with optimizer
    # -------------------------
    batch_size = 4
    seq_len = 4096
    num_steps = 10
    torch.manual_seed(42 + rank + 1)
    inp = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)

    prof_iters, prof_start_iter, prof_end_iter = 10, 4, 7
    for iter in range(num_steps):
        # init for nvtx
        nvtx.switch_profile(
            iter_id=iter,
            start=prof_start_iter,
            end=prof_end_iter,
            profile_ranks=[0],
        )

        dist.barrier()
        torch.cuda.synchronize()

        optim.zero_grad()
        out = model(inp)
        loss = out.sum()
        loss.backward()

        # cast grad to fp32
        if ENABLE_MAIN_PARAMS:
            for model_p, master_p in zip(model.parameters(), main_params):
                if model_p.grad is not None:
                    if master_p.grad is None:
                        master_p.grad = model_p.grad.detach().float().clone()
                    else:
                        master_p.grad.copy_(model_p.grad.detach().float())
        optim.step()

        # copy main params to model params
        if ENABLE_MAIN_PARAMS:
            for model_p, master_p in zip(model.parameters(), main_params):
                model_p.data.copy_(master_p.data.to(dtype=model_p.dtype))

        # print(f"{torch.isnan(out).any()=}")
