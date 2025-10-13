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

import copy
import os

import torch
import torch.distributed as dist
from torch._dynamo.utils import counters
from torch.distributed._composable import replicate
from torch.distributed.tensor import init_device_mesh

from magi_fsdp import fully_shard
from magi_fsdp.testing import parameterize
from magi_fsdp.testing.common_fsdp import FSDPTest

# to log recompiles for torch.compile
os.environ["TORCH_LOGS"] = "recompiles"


class TestMagiFullyShardCompile(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @parameterize("device_type", ["cuda"])
    def test_fully_shard_compile(
        self,
        device_type: str,
    ):
        assert device_type in ("cuda", "cpu"), f"{device_type}"
        torch.manual_seed(42)

        layers = [torch.nn.Linear(4, 4) for _ in range(3)]
        model = torch.nn.Sequential(*layers)
        ref_model = copy.deepcopy(model)
        if device_type == "cuda":
            replicate(ref_model.cuda(), device_ids=[self.rank])
        else:
            gloo_pg = dist.new_group(backend="gloo")
            replicate(ref_model, process_group=gloo_pg)
        mesh = init_device_mesh("cuda", (self.world_size,))

        for i in range(3):
            fully_shard(model[i], mesh=mesh)
        fully_shard(model, mesh=mesh)

        inp = torch.randn(4, 4)
        ref_model = torch.compile(ref_model, backend="aot_eager", fullgraph=False)
        ref_out = ref_model(inp)

        counters.clear()
        torch._dynamo.reset()
        torch.distributed.barrier()
        model = torch.compile(model, backend="aot_eager", fullgraph=False)
        out = model(inp)
        torch.distributed.barrier()

        # NOTE: Since we disable Dynamo for hooks, we cannot use `torch._dynamo.testing.CompileCounterWithBackend` here.
        # Instead, we use `_dynamo.utils.counters`. When recompilation occurs, the values in this counter will keep increasing.
        # In ConvertFrame.__call__, `counters` tracks frame processing:
        #   `frames.total` counts frames attempted for conversion,
        #   `frames.ok` counts successfully converted frames.
        # `graph_break` tracks compile breaks at the _pre_forward hook.
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(counters["frames"]["total"], 1)
        self.assertEqual(counters["frames"]["ok"], 1)
        self.assertEqual(ref_out, out)
