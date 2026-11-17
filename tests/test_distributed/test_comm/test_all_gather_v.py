# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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

from magi_attention.comm.primitive._all_gather_v import (
    _split_dims,
    _trans_with_dim0,
    all_gather_v,
)
from magi_attention.testing.dist_common import DistTestBase, with_comms


class TestAllgatherV(DistTestBase):
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
    def test_all_gather_v_equal(self):
        # ----    equal all gather     ---- #
        b, s, h = 16, 1024, 128
        dim = 1
        split_sizes = None

        x = torch.randn((b, s, h), device=self.device)
        x_ = x.transpose(0, 1).contiguous()  # (s, b, h)

        x_gather_list = [
            torch.empty((s, b, h), device=x.device, dtype=x.dtype)
            for _ in range(self.world_size)
        ]

        dist.all_gather(x_gather_list, x_, group=self.process_group)

        x_gather_v_ref = torch.cat(x_gather_list, dim=0).transpose(0, 1)

        x_gather_v = all_gather_v(
            x_local=x,
            group=self.process_group,
            dim=dim,
            split_sizes=split_sizes,
        )

        self.assertTrue(torch.equal(x_gather_v_ref, x_gather_v))

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_all_gather_v_unequal(self):
        # ----    unequal all gather     ---- #

        b, s, h = 16, 1024, 128
        dim = 1
        split_sizes = [s + rank for rank in range(self.world_size)]

        x = torch.randn((b, s + self.rank, h), device=self.device)
        x_ = x.transpose(0, 1).contiguous()  # (s + rank, b, h)

        x_gather_list = [
            torch.empty((s + rank, b, h), device=x.device, dtype=x.dtype)
            for rank in range(self.world_size)
        ]

        dist.all_gather(x_gather_list, x_, group=self.process_group)

        x_gather_v_ref = torch.cat(x_gather_list, dim=0).transpose(0, 1)

        x_gather_v = all_gather_v(
            x_local=x,
            group=self.process_group,
            dim=dim,
            split_sizes=split_sizes,
        )

        self.assertTrue(torch.equal(x_gather_v_ref, x_gather_v))

    def test_trans_with_dim0(self):
        # ----    high-dim tensor     ---- #

        x = torch.arange(2 * 3 * 4).reshape(2, 3, 4).contiguous()

        y0 = _trans_with_dim0(x, dim=0)
        y0_ = _trans_with_dim0(x, dim=-3)
        self.assertTrue(y0.is_contiguous())
        self.assertTrue(torch.equal(y0, x))
        self.assertTrue(torch.equal(y0_, y0))
        self.assertTrue(y0.data_ptr() == x.data_ptr())  # same storage

        y1 = _trans_with_dim0(x, dim=1)
        y1_ = _trans_with_dim0(x, dim=-2)
        self.assertTrue(y1.is_contiguous())
        self.assertTrue(torch.equal(y1, x.transpose(0, 1)))
        self.assertTrue(torch.equal(y1_, y1))
        self.assertFalse(y1.data_ptr() == x.data_ptr())  # different storage

        y2 = _trans_with_dim0(x, dim=2)
        y2_ = _trans_with_dim0(x, dim=-1)
        self.assertTrue(y2.is_contiguous())
        self.assertTrue(torch.equal(y2, x.transpose(0, 2)))
        self.assertTrue(torch.equal(y2_, y2))
        self.assertFalse(y2.data_ptr() == x.data_ptr())  # different storage

        # ----    1-dim tensor     ---- #

        x = torch.arange(12)

        y0 = _trans_with_dim0(x, dim=0)
        self.assertTrue(y0.is_contiguous())
        self.assertTrue(torch.equal(y0, x))
        self.assertTrue(y0.data_ptr() == x.data_ptr())  # same storage

        y1 = _trans_with_dim0(x, dim=-1)
        self.assertTrue(y1.is_contiguous())
        self.assertTrue(torch.equal(y1, x))
        self.assertTrue(y1.data_ptr() == x.data_ptr())  # same storage

    def test_split_dims(self):
        # ----    high-dim shape     ---- #

        x_shape = [2, 3, 4, 5]

        this_dim, other_dims = _split_dims(x_shape, dim=0)
        self.assertEqual(this_dim, 2)
        self.assertEqual(other_dims, [3, 4, 5])

        this_dim, other_dims = _split_dims(x_shape, dim=1)
        self.assertEqual(this_dim, 3)
        self.assertEqual(other_dims, [2, 4, 5])

        this_dim, other_dims = _split_dims(x_shape, dim=2)
        self.assertEqual(this_dim, 4)
        self.assertEqual(other_dims, [3, 2, 5])

        this_dim, other_dims = _split_dims(x_shape, dim=3)
        this_dim_, other_dims_ = _split_dims(x_shape, dim=-1)
        self.assertEqual(this_dim, 5)
        self.assertEqual(other_dims, [3, 4, 2])
        self.assertEqual(this_dim_, this_dim)
        self.assertEqual(other_dims_, other_dims)

        # ----    1-dim shape     ---- #

        x_shape = [12]

        this_dim, other_dims = _split_dims(x_shape, dim=0)
        self.assertEqual(this_dim, 12)
        self.assertEqual(other_dims, [])

        this_dim, other_dims = _split_dims(x_shape, dim=-1)
        self.assertEqual(this_dim, 12)
        self.assertEqual(other_dims, [])

        # ----    invalid dim     ---- #

        x_shape = [2, 3, 4]

        with self.assertRaises(
            AssertionError,
            msg="dim should be in [0, len(x_shape) - 1) or -1",
        ):
            _split_dims(x_shape, dim=4)

        with self.assertRaises(
            AssertionError,
            msg="dim should be in [0, len(x_shape) - 1) or -1",
        ):
            _split_dims(x_shape, dim=-2)


if __name__ == "__main__":
    run_tests()
