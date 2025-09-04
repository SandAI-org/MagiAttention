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

# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests

from magi_fsdp import fully_shard


class ToyModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim, bias=False),
        )
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        y = self.linears(x)
        y = self.proj(y)
        return y


class ReplicateFullyShardInit(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    def _compare_module(self, mod, replicate_mod):
        local_batch_size = 1
        global_batch_size = self.world_size * local_batch_size
        input = torch.randn(global_batch_size, 2)
        target = torch.randn(global_batch_size, 2)

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        for iteration in range(2):
            step_model(mod, input, target)
            step_model(
                replicate_mod,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            self.assertEqual(
                len(list(mod.parameters())),
                len(list(replicate_mod.parameters())),
            )
            for i, j in zip(mod.parameters(), replicate_mod.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]

    @skip_if_lt_x_gpu(2)
    def test_replicate_fully_shard_init(self):
        self._init_pg()
        torch.cuda.set_device(self.rank)

        dim = 3
        bz = 2
        model = ToyModel(dim).cuda()
        for linear in model.linears:
            fully_shard(linear)
        fully_shard(model.linears)
        replicate(model, device_id=torch.cuda.current_device())
        for linear in model.linears:
            self.assertTrue(isinstance(linear.weight, DTensor))
        inp = torch.rand(bz, dim)
        # trigger lazy init
        model(inp).sum()
        for linear in model.linears:
            self.assertTrue(isinstance(linear.weight, DTensor))


if __name__ == "__main__":
    run_tests()
