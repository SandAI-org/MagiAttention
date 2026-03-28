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

import datetime
import os
import sys
from fnmatch import fnmatch
from functools import wraps
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    TEST_SKIPS,
    TIMEOUT_OVERRIDE,
    MultiProcessTestCase,
)

from magi_attention.utils import set_random_seed

from .utils import switch_envvar_decorator

NAME = "name"

SKIP_WORLD_SIZE = "skip_world_size"

PROFILE_ONLY = "profile_only"

INTERFACE = "interface"

TEST_ATTN_CONFIG = "MAGI_ATTENTION_TEST_ATTN_CONFIG"

_TEST_FILTER_ENV_PREFIX = "MAGI_ATTENTION_TEST_"

TEST_WORLD_SIZE = "MAGI_ATTENTION_TEST_WORLD_SIZE"

_TEST_FILTER_ENVVARS: dict[str, str] = {
    "attn_config": "MAGI_ATTENTION_TEST_ATTN_CONFIG",
    "overlap_config": "MAGI_ATTENTION_TEST_OVERLAP_CONFIG",
    "num_heads": "MAGI_ATTENTION_TEST_NUM_HEADS",
    "head_dim": "MAGI_ATTENTION_TEST_HEAD_DIM",
    "dtype": "MAGI_ATTENTION_TEST_DTYPE",
    "random_type_mapping": "MAGI_ATTENTION_TEST_RANDOM_TYPE_MAPPING",
}


def _match_patterns(value_str: str, raw_env: str) -> bool:
    """Check if ``value_str`` matches any comma-separated pattern in ``raw_env``.
    Supports ``fnmatch`` wildcards.
    """
    patterns = [p.strip() for p in raw_env.split(",") if p.strip()]
    return any(fnmatch(value_str, pat) for pat in patterns)


def should_run_world_size(world_size: int) -> bool:
    """Check whether the given *world_size* should be tested.

    Reads ``MAGI_ATTENTION_TEST_WORLD_SIZE`` from the environment.
    When the variable is unset or empty, all world sizes are run.
    Otherwise it is a comma-separated list of integers, and only
    world sizes present in the list will run.

    Usage examples::

        # Only test world_size=2
        MAGI_ATTENTION_TEST_WORLD_SIZE=2 pytest tests/test_pipeline.py

        # Test world_size 2 and 4
        MAGI_ATTENTION_TEST_WORLD_SIZE=2,4 pytest tests/test_pipeline.py
    """
    raw = os.environ.get(TEST_WORLD_SIZE, "").strip()
    if not raw:
        return True
    allowed = {int(s.strip()) for s in raw.split(",") if s.strip()}
    return world_size in allowed


def skip_if_world_size_filtered(func):
    """Subprocess-level skip decorator for ``MAGI_ATTENTION_TEST_WORLD_SIZE``.

    Works the same way as ``skip_if_lt_x_gpu``: when the world size is not in
    the requested set, the subprocess exits with the ``generic`` skip code so
    that ``MultiProcessTestCase`` reports it as *skipped* rather than spawning
    all distributed workers.

    Reads ``self.world_size`` automatically, so a single decorator can be placed
    on the base-class method and inherited by every world-size subclass.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if should_run_world_size(self.world_size):
            return func(self, *args, **kwargs)
        sys.exit(TEST_SKIPS["generic"].exit_code)

    return wrapper


def should_run_attn_config(name: str) -> bool:
    """Check whether the attn_config with the given *name* should be executed.

    Reads ``MAGI_ATTENTION_TEST_ATTN_CONFIG`` from the environment.  When the
    variable is unset or empty every config is run (backward-compatible).
    Otherwise it is treated as a comma-separated list of patterns (``fnmatch``
    wildcards are supported) and the config runs only when *name* matches at
    least one pattern.

    Usage examples::

        # Run a single config
        MAGI_ATTENTION_TEST_ATTN_CONFIG=uneven_full_attn_10k pytest tests/test_pipeline.py

        # Run multiple configs
        MAGI_ATTENTION_TEST_ATTN_CONFIG=uneven_full_attn_10k,full_attn_14k pytest tests/test_pipeline.py

        # Wildcard matching
        MAGI_ATTENTION_TEST_ATTN_CONFIG="uneven_*" pytest tests/test_pipeline.py
        MAGI_ATTENTION_TEST_ATTN_CONFIG="varlen_block_causal_*" pytest tests/test_pipeline.py

        # Or via pytest CLI option
        pytest tests/test_pipeline.py --test-attn-config "uneven_*"
    """
    raw = os.environ.get(TEST_ATTN_CONFIG, "").strip()
    if not raw:
        return True
    return _match_patterns(name, raw)


def should_run_test_case(**parametrize_args: object) -> bool:
    """Generalized filter for any ``@parameterize`` dimension.

    Each keyword argument corresponds to a parametrize dimension name and
    the value currently being tested.  For dict-typed values that contain a
    ``NAME`` key (e.g. ``attn_config``, ``overlap_config``), the ``NAME``
    value is used for matching.  For tuple values, elements are joined with
    underscores (e.g. ``(8, 8)`` becomes ``"8_8"``).  For other types the
    ``str()`` representation is used.

    The matching is controlled by environment variables following the pattern
    ``MAGI_ATTENTION_TEST_<DIMENSION_UPPER>``, where ``<DIMENSION_UPPER>`` is
    the uppercased dimension name (e.g. ``MAGI_ATTENTION_TEST_ATTN_CONFIG``
    for ``attn_config``).  When the variable is unset or empty the dimension
    is unconstrained.  Otherwise it is a comma-separated list of ``fnmatch``
    patterns.

    Returns ``True`` only when **all** constrained dimensions match.

    Usage examples::

        # Filter by attn_config name
        MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_*" pytest tests/test_pipeline.py

        # Filter by overlap_config name
        MAGI_ATTENTION_TEST_OVERLAP_CONFIG=no_overlap pytest tests/test_pipeline.py

        # Filter by head_dim
        MAGI_ATTENTION_TEST_HEAD_DIM=128 pytest tests/test_pipeline.py

        # Filter by num_heads (underscore-separated, e.g. "8_8" for (8, 8))
        MAGI_ATTENTION_TEST_NUM_HEADS=8_8 pytest tests/test_pipeline.py

        # Filter by dtype
        MAGI_ATTENTION_TEST_DTYPE="*float16*" pytest tests/test_pipeline.py

        # Combine multiple filters (AND logic)
        MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_*" \\
        MAGI_ATTENTION_TEST_OVERLAP_CONFIG=no_overlap \\
        MAGI_ATTENTION_TEST_HEAD_DIM=64 \\
            pytest tests/test_pipeline.py
    """
    for dim_name, value in parametrize_args.items():
        envvar = _TEST_FILTER_ENVVARS.get(
            dim_name,
            _TEST_FILTER_ENV_PREFIX + dim_name.upper(),
        )
        raw = os.environ.get(envvar, "").strip()
        if not raw:
            continue

        if isinstance(value, dict) and NAME in value:
            value_str = str(value[NAME])
        elif isinstance(value, tuple):
            value_str = "_".join(str(v) for v in value)
        else:
            value_str = str(value)

        if not _match_patterns(value_str, raw):
            return False

    return True

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
