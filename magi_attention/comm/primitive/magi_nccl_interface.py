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

import importlib

import torch
import torch.distributed as dist
from torch.distributed.c10d_logger import _exception_logger
from torch.distributed.distributed_c10d import (
    ProcessGroupGloo,
    ProcessGroupNCCL,
    _check_single_tensor,
    _ensure_all_tensors_same_dtype,
    _get_default_group,
    _rank_not_in_group,
    _warn_not_in_group,
    is_gloo_available,
    is_nccl_available,
)

# isort: off
# We need to import the CUDA kernels after importing torch
from magi_nccl import MagiNCCLBackend

# isort: on


def _magi_nccl_shutdown_backend(pg) -> None:
    """Try to shut down the backend of a process group.
    Currently, only ProcessGroupNCCL, ProcessGroupGloo, and MagiNCCLBackend is supported.
    No op for other backends.
    """

    backend = None
    try:
        backend = pg._get_backend(torch.device("cuda"))
    except RuntimeError:
        pass
    if is_nccl_available() and isinstance(backend, (ProcessGroupNCCL, MagiNCCLBackend)):
        # explictly call shutdown to ensure that NCCL resources are released
        backend._shutdown()
    elif is_gloo_available() and isinstance(backend, ProcessGroupGloo):
        backend._shutdown()


@_exception_logger
def group_cast(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: dist.Backend = None,
    async_op: bool = False,
    **kwargs,
):
    """Group cast communication based on MagiNCCLBackend"""

    if _rank_not_in_group(group):
        _warn_not_in_group("group_cast")
        return None

    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    _ensure_all_tensors_same_dtype(output, input)

    if input.is_complex():
        input = torch.view_as_real(input)
    if output.is_complex():
        output = torch.view_as_real(output)

    group = group or _get_default_group()

    backend = group._get_backend(torch.device("cuda"))
    assert isinstance(
        backend, MagiNCCLBackend
    ), f"expected MagiNCCLBackend, got {type(backend)=}"

    work = backend.group_cast(
        input,
        output,
        input_split_size_list,
        output_split_size_list,
        dst_indices_list,
        src_index_list,
    )

    if async_op:
        return work
    else:
        work.wait()


@_exception_logger
def group_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: dist.Backend = None,
    async_op: bool = False,
    **kwargs,
):
    """Group reduce communication based on MagiNCCLBackend"""

    if _rank_not_in_group(group):
        _warn_not_in_group("group_reduce")
        return None

    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    _ensure_all_tensors_same_dtype(output, input)

    if input.is_complex():
        input = torch.view_as_real(input)
    if output.is_complex():
        output = torch.view_as_real(output)

    group = group or _get_default_group()

    backend = group._get_backend(torch.device("cuda"))
    assert isinstance(
        backend, MagiNCCLBackend
    ), f"expected MagiNCCLBackend, got {type(backend)=}"

    work = backend.group_reduce(
        input,
        output,
        input_split_size_list,
        output_split_size_list,
        dst_index_list,
        src_indices_list,
    )

    if async_op:
        return work
    else:
        work.wait()


# NOTE: we have to extend the `_shutdown_backend` function to let `destroy_process_group` works for MagiNCCLBackend
torch_distributed_c10d = importlib.import_module("torch.distributed.distributed_c10d")
torch_distributed_c10d._shutdown_backend = _magi_nccl_shutdown_backend  # type: ignore[attr-defined]
