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


import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._random as random
from torch.distributed._tensor import DeviceMesh, DTensor, init_device_mesh
from torch.distributed._tensor.api import distribute_tensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor._random import (
    OffsetBasedRNGTracker,
    is_rng_supported_mesh,
)
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.testing._internal.common_utils import TEST_HPU, run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    skip_unless_torch_gpu,
    with_comms,
)

from magi_fsdp import fully_shard

TYPE_DEVICE = "hpu" if TEST_HPU else "cuda"


class DistTensorRandomInitTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _run_init_op(self, init_op, *args, **kwargs):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        input_size = (8, 4)

        # NOTE: currently random initialization on cuda device has different
        # behavior from other devices. Unify the test once the behavior is unified.
        if not is_rng_supported_mesh(device_mesh):
            input_tensor = torch.randn(*input_size, device=self.device_type)
            dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
            local_tensor_clone = torch.clone(input_tensor)
            torch.manual_seed(self.rank)
            local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
            torch.manual_seed(self.rank)
            dtensor = init_op(dtensor, *args, **kwargs)
            self.assertEqual(local_tensor_clone, dtensor.to_local())
        else:
            # create DTensor from Tensor
            _tensor = torch.empty(*input_size, device=TYPE_DEVICE)
            dtensor = distribute_tensor(_tensor, device_mesh, [Shard(1)])

            # DTensor random init
            dtensor = init_op(dtensor, *args, **kwargs)
            local_tensor = dtensor.to_local()

            # compare with local tensors from other ranks
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    slice_idx = [
                        slice(input_size[0]),
                        slice(
                            other_rank * input_size[1], (other_rank + 1) * input_size[1]
                        ),
                    ]
                    # other rank should have a different local tensor
                    self.assertNotEqual(dtensor.full_tensor()[slice_idx], local_tensor)

    @with_comms
    def test_init_ops(self):
        self._run_init_op(
            torch.nn.init.kaiming_uniform_,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        self._run_init_op(torch.nn.init.normal_, mean=1.5, std=0.8)
        self._run_init_op(torch.nn.init.uniform_, a=0, b=1.2)

        for dtype in (torch.float32, torch.float16):
            self._run_init_op(torch.rand_like, dtype=dtype)
            self._run_init_op(torch.randn_like, dtype=dtype)
            self._run_init_op(torch.randint_like, low=0, high=100, dtype=dtype)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_meta_tensor_init(self):
        # test suite sets each rank's seed to the same value but in actual
        # execution the default random seed will be different (a random value).
        # The DTensor random ops will use the same random seed even though the
        # torch random generator keeps different seeds on ranks. This ensures
        # that Replicate DTensor will have the same initialized results
        # across ranks.
        torch.cuda.manual_seed(self.rank)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [1024, 2048]
        meta_dtensor = distribute_tensor(
            torch.empty(*size, device="meta"), device_mesh, [Replicate()]
        )

        # the tensor slice on the current rank
        self_slice = slice(1024 * self.rank, 1024 * self.rank + 1024)

        # Test 1: enable the distribute region for RNG (by default)
        self.assertTrue(meta_dtensor.is_meta)
        # Tensor meta init
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
        dtensor.uniform_()
        # check `distribute_region_enabled` is set to True by default
        self.assertTrue(random._rng_tracker.distribute_region_enabled)

        # allgather the local tensors
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        # compare with local tensors from other ranks
        for other_rank in range(self.world_size):
            # the RNG result on each rank are the same because they're replicated
            if self.rank != other_rank:
                # other rank should have an identical local tensor
                other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                self.assertEqual(
                    local_tensor[self_slice, :], local_tensor[other_slice, :]
                )

        # Test 2: disable the distribute region for RNG
        self.assertTrue(meta_dtensor.is_meta)
        # Tensor meta init
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
        random._rng_tracker.distribute_region_enabled = False
        dtensor.uniform_()
        # check `distribute_region_enabled` is set to False
        self.assertTrue(not random._rng_tracker.distribute_region_enabled)

        # allgather the local tensors
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        # compare with local tensors from other ranks
        for other_rank in range(self.world_size):
            # the RNG result on each rank differs even they're supposed
            # to be replicated
            if self.rank != other_rank:
                other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                self.assertNotEqual(
                    local_tensor[self_slice, :], local_tensor[other_slice, :]
                )

    @with_comms
    @skip_unless_torch_gpu
    def test_tp_model_meta_init(self):
        # initialize the 1-d device mesh for TP
        tp_mesh = init_device_mesh(self.device_type, mesh_shape=(self.world_size,))

        # model meta init
        with torch.device("meta"):
            model = torch.nn.Linear(self.world_size, self.world_size, bias=False)
            self.assertEqual(model.weight.device, torch.device("meta"))
            parallelize_module(model, tp_mesh, ColwiseParallel())
            if random._rng_tracker is not None:
                random._rng_tracker.distribute_region_enabled = True

            self.assertEqual(model.weight.device, torch.device("meta"))

        # actual initialization
        device = torch.device("cuda", torch.cuda.current_device())
        model.to_empty(device=device)
        model.reset_parameters()
        self.assertTrue(
            random._rng_tracker is not None
            and isinstance(random._rng_tracker, OffsetBasedRNGTracker)
        )
        self.assertEqual(model.weight.device, device)
        assert isinstance(model.weight, DTensor)

        # gather all the shards to compare initialization results
        WORLD = torch.distributed.group.WORLD
        assert WORLD is not None
        weight_local = model.weight.to_local()
        weight_gather = funcol.all_gather_tensor(
            weight_local,
            gather_dim=0,
            group=WORLD,
        )

        # verify the weights are initialized differently on all ranks
        for other_rank in range(self.world_size):
            if self.rank != other_rank:
                self.assertNotEqual(
                    weight_local,
                    weight_gather[other_rank : other_rank + 1, :],
                )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_model_meta_init(self):
        # initialize the 2-d device mesh
        global_mesh = init_device_mesh(
            self.device_type,
            mesh_shape=(self.world_size // 2, 2),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]

        # model meta init
        with torch.device("meta"):
            model = torch.nn.Linear(self.world_size, self.world_size, bias=False)
            self.assertEqual(model.weight.device, torch.device("meta"))
            parallelize_module(model, tp_mesh, ColwiseParallel())
            if random._rng_tracker is not None:
                random._rng_tracker.distribute_region_enabled = True

            fully_shard(model, mesh=dp_mesh)
            self.assertEqual(model.weight.device, torch.device("meta"))

        # actual initialization
        device = torch.device("cuda", torch.cuda.current_device())
        model.to_empty(device=device)
        model.reset_parameters()
        self.assertTrue(
            random._rng_tracker is not None
            and isinstance(random._rng_tracker, OffsetBasedRNGTracker)
        )
        self.assertEqual(model.weight.device, device)
        assert isinstance(model.weight, DTensor)

        # gather all the shards to compare initialization results
        WORLD = torch.distributed.group.WORLD
        assert WORLD is not None
        weight_local = model.weight.to_local()
        weight_gather = funcol.all_gather_tensor(
            weight_local,
            gather_dim=0,
            group=WORLD,
        )

        # verify the weights are initialized differently on all ranks
        for other_rank in range(self.world_size):
            if self.rank != other_rank:
                self.assertNotEqual(
                    weight_local,
                    weight_gather[other_rank : other_rank + 1, :],
                )


class DistTensorRandomInitTest3D(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    @skip_if_lt_x_gpu(8)
    def test_hsdp_tp_model_meta_init(self):
        # initialize the 3-d device mesh
        global_mesh = init_device_mesh(
            self.device_type,
            mesh_shape=(self.world_size // 4, 2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )
        tp_mesh = global_mesh["tp"]
        dp_mesh = global_mesh["dp_replicate", "dp_shard"]

        # model meta init
        with torch.device("meta"):
            model = torch.nn.Linear(self.world_size, self.world_size, bias=False)
            self.assertEqual(model.weight.device, torch.device("meta"))
            parallelize_module(model, tp_mesh, ColwiseParallel())
            if random._rng_tracker is not None:
                random._rng_tracker.distribute_region_enabled = True

            fully_shard(model, mesh=dp_mesh)
            self.assertEqual(model.weight.device, torch.device("meta"))

        # actual initialization
        device = torch.device("cuda", torch.cuda.current_device())
        model.to_empty(device=device)
        model.reset_parameters()
        self.assertTrue(
            random._rng_tracker is not None
            and isinstance(random._rng_tracker, OffsetBasedRNGTracker)
        )
        self.assertEqual(model.weight.device, device)
        assert isinstance(model.weight, DTensor)

        # gather all the shards to compare initialization results
        WORLD = torch.distributed.group.WORLD
        assert WORLD is not None
        weight_local = model.weight.to_local()
        weight_gather = funcol.all_gather_tensor(
            weight_local,
            gather_dim=0,
            group=WORLD,
        )

        # verify the weights are initialized differently on all ranks
        shard_dim_0_len = self.world_size // 4
        for other_rank in range(self.world_size):
            other_rank_dim_0_start = other_rank * shard_dim_0_len
            other_rank_dim_0_end = other_rank_dim_0_start + shard_dim_0_len
            if self.rank % 4 != other_rank % 4:
                self.assertNotEqual(
                    weight_local,
                    weight_gather[other_rank_dim_0_start:other_rank_dim_0_end, :],
                )
            else:
                self.assertEqual(
                    weight_local,
                    weight_gather[other_rank_dim_0_start:other_rank_dim_0_end, :],
                )


if __name__ == "__main__":
    run_tests()
