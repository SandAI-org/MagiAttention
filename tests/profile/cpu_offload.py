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

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    TransformerBlock,
)

from magi_fsdp import fully_shard
from magi_fsdp._fsdp_api import CPUOffloadPolicy


class MyTransformer(nn.Module):
    def __init__(self, layer_nums: int = 5):
        super().__init__()
        self.layer_nums = layer_nums
        self.model_args = ModelArgs(
            n_layers=16,
            n_heads=32,
            dim=128,
            vocab_size=4096,
            max_seq_len=4096,
            dropout_p=0,
            use_attn_mask=False,
            weight_tying=False,
            checkpoint_activations=False,
        )
        self.transformer_block = nn.ModuleList(
            [TransformerBlock(self.model_args) for _ in range(layer_nums)]
        )

    def forward(self, x: torch.Tensor):
        for block in self.transformer_block:
            x = block(x)
        return x


torch.cuda.reset_peak_memory_stats()
dist.init_process_group("nccl")
rank = dist.get_rank()
print(f"{rank} ")
torch.cuda.set_device(rank % torch.cuda.device_count())
device = torch.cuda.current_device()

torch.manual_seed(42)
model = MyTransformer()
for i in range(model.layer_nums):
    if i != model.layer_nums - 1:
        fully_shard(
            model.transformer_block[i],
            offload_policy=CPUOffloadPolicy(foreach_offload=False),
        )
    else:
        fully_shard(
            model.transformer_block[i],
            offload_policy=CPUOffloadPolicy(foreach_offload=False),
        )
        model.transformer_block[i].set_offload_activation_after_forward(False)
fully_shard(
    model,
    offload_policy=CPUOffloadPolicy(foreach_offload=False, pin_memory=True),
    reshard_after_forward=True,
)

torch.manual_seed(42)
inp = torch.randn(4, 4096, 128, device=device)

for i in range(10):
    if rank == 0 and i == 6:
        torch.cuda.profiler.start()
        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
    if rank == 0 and i == 8:
        torch.cuda.profiler.stop()

    # -----    barrier at the beginning of each iteration   ---- #

    dist.barrier()
    torch.cuda.synchronize()

    loss = model(inp).sum()
    loss.backward()
