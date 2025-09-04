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
# Owner(s): ["module: fsdp"]

import functools
import os
import unittest

import torch.distributed as dist
from torch._dynamo.test_case import run_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.logging_utils import LoggingTestCase

requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


@skip_if_lt_x_gpu(2)
class LoggingTests(LoggingTestCase):
    @requires_distributed()
    def test_fsdp_logging(self):
        env = dict(os.environ)
        env["TORCH_LOGS"] = "fsdp"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_PORT"] = "34715"
        env["MASTER_ADDR"] = "localhost"
        _, stderr = self.run_process_no_exception(
            """\
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from magi_fsdp import fully_shard
logger = logging.getLogger("magi_fsdp")
logger.setLevel(logging.DEBUG)
device = "cuda"
torch.manual_seed(0)
model = nn.Sequential(*[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)])
for layer in model:
    fully_shard(layer)
fully_shard(model)
x = torch.randn((4, 4), device=device)
model(x).sum().backward()
""",
            env=env,
        )
        self.assertIn("MagiFSDP::root_pre_forward", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::pre_forward (0)", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::pre_forward (1)", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::post_forward (0)", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::post_forward (1)", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::pre_backward (0)", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::pre_backward (1)", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::post_backward (0)", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::post_backward (1)", stderr.decode("utf-8"))
        self.assertIn("MagiFSDP::root_post_backward", stderr.decode("utf-8"))


if __name__ == "__main__":
    run_tests()
