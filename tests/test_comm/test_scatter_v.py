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
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from magi_attention.comm.primitive import scatter_v
from magi_attention.testing.dist_common import DistTestBase, with_comms


class TestScatterV(DistTestBase):
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_scatter_v_equal(self):
        # ----    equal scatter     ---- #

        b, s, h = 16, 1024, 128
        dim = 1
        split_sizes = None

        x = torch.randn((b, s, h), device=self.device)
        chunk_size = s // self.world_size
        x_scatter_v_ref = x[:, chunk_size * self.rank : chunk_size * (self.rank + 1)]

        x_scatter_v = scatter_v(
            x_global=x,
            group=self.process_group,
            dim=dim,
            split_sizes=split_sizes,
        )

        self.assertTrue(torch.equal(x_scatter_v_ref, x_scatter_v))

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_scatter_v_unequal(self):
        # ----    unequal scatter     ---- #

        b, s, h = 16, 1024, 128
        dim = 1
        split_sizes = [s + rank for rank in range(self.world_size)]

        x_split_list = [
            torch.randn((b, split_size, h), device=self.device)
            for split_size in split_sizes
        ]
        x_scatter_v_ref = x_split_list[self.rank]

        x_scatter_v = scatter_v(
            x_global=torch.concat(x_split_list, dim=dim),
            group=self.process_group,
            dim=dim,
            split_sizes=split_sizes,
        )

        self.assertTrue(torch.equal(x_scatter_v_ref, x_scatter_v))


if __name__ == "__main__":
    run_tests()
