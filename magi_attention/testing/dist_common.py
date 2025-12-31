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

import datetime
from functools import wraps
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    TIMEOUT_OVERRIDE,
    MultiProcessTestCase,
)

from magi_attention.utils import set_random_seed

from .utils import switch_envvar_decorator

NAME = "name"

SKIP_WORLD_SIZE = "skip_world_size"

PROFILE_ONLY = "profile_only"

INTERFACE = "interface"

DEVICE_TYPE = (
    "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
)

PG_DEFAULT_BACKEND = "nccl" if DEVICE_TYPE == "cuda" else "gloo"

NUM_DEVICES = 4

RUN_IN_MP = "MAGI_ATTENTION_PARAMETERIZE_RUN_IN_MP"


# We use this as a proxy for "multiple GPUs exist"
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())


class DistTestBase(MultiProcessTestCase):
    @property
    def seed(self) -> int:
        return 42

    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        return PG_DEFAULT_BACKEND

    def init_pg(self) -> None:
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"nccl backend requires {self.world_size} GPUs, but only {torch.cuda.device_count()} are available"
            )

        if self.backend not in [
            "nccl",
            "gloo",
            "mpi",
            "cpu:gloo,cuda:nccl",
        ]:
            raise RuntimeError(f"Backend {self.backend} not supported!")

        # Initialize the process group
        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",  # noqa
            timeout=datetime.timedelta(minutes=30),
        )

        # Set the device for this process
        if "nccl" in self.backend:
            torch.cuda.set_device(self.rank)

        # Set random seed with rank offset
        self._set_random_seed()

    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if torch.cuda.is_available() else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        dist.barrier()
        dist.destroy_process_group()

    def _set_random_seed(self) -> None:
        seed = self.seed + self.rank
        set_random_seed(seed)

    def setUp(self) -> None:
        super().setUp()

        timeout = getattr(self, "timeout", None)
        if timeout is not None:
            TIMEOUT_OVERRIDE.update({self.id().split(".")[-1]: timeout})

        self._spawn_processes()


TestFunc = Callable[..., Any]


# wrapper to initialize comms (processgroup)
def with_comms(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(self, *args: tuple[object], **kwargs: dict[str, Any]) -> None:
        # if backend not specified, and cuda available, then use nccl, else gloo
        if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        self.init_pg()
        func(self, *args, **kwargs)
        self.destroy_pg()

    return wrapper


def with_run_in_mp(func: TestFunc) -> TestFunc:
    return switch_envvar_decorator(envvar_name=RUN_IN_MP, enable=True)(with_comms(func))
