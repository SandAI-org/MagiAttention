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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .triton_impl import on_device_a2av_triton_impl


class OnDeviceA2AVMgr:
    # A symmetric memory holding the maximum input buffer to send
    input_send_buf = None
    # A symmetric memory for exchanges split sizes during both forward and backward
    splits_buf = None
    # Maximum output length (need to be set before use of OnDeviceA2AV)
    max_output_seqlen = None
    # Whether the symmetric memory buffers are all initialized
    is_initialized = False

    @classmethod
    def apply(
        cls,
        input: torch.Tensor,
        input_splits: torch.Tensor | list[int],
        output: torch.Tensor | None = None,
        received_num_tokens: int = -1,
        group: dist.ProcessGroup = dist.group.WORLD,
    ):
        """
        Args:
            input: input tensor with data for all ranks concatenated.
            input_splits: input splits of shape (group.world_size,)
            group: process group to scope the collective.
        """

        assert cls.is_initialized

        # Initialize input splits buffer (one time only)
        if isinstance(input_splits, list):
            input_splits = torch.tensor(
                input_splits, dtype=torch.int64, device=input.device
            )

        if (
            cls.splits_buf is None
            or cls.splits_buf.shape != input_splits.shape  # type: ignore[unreachable]
        ):
            # NOTE: for a2a-v, the shape of input_splits/output_splits is constant to (world_size,)
            cls.splits_buf = symm_mem.empty(
                *input_splits.shape,
                dtype=input_splits.dtype,
                device=input_splits.device,
            )

        # Copy input to the symmetric buffer
        # TODO: is there a way to tell autograd
        # to feed input directly to our symm_mem buffer?
        cls.input_send_buf.narrow(  # type: ignore[attr-defined]
            dim=0, start=0, length=input.shape[0]
        ).copy_(input.view(input.shape[0], -1))

        # Copy input splits to the symmetric buffer
        cls.splits_buf.copy_(input_splits)

        # Allocate output buffer
        if output is None:
            if received_num_tokens == -1:
                output = input.new_empty(cls.max_output_seqlen, *input.shape[1:])  # type: ignore[unreachable]
            else:
                output = input.new_empty(received_num_tokens, *input.shape[1:])
        elif received_num_tokens == -1:
            received_num_tokens = output.shape[0]

        # Allocate output splits tensor
        output_splits = torch.empty_like(input_splits)

        # Shuffle input to output
        # TODO: for now, we only support triton implementation
        on_device_a2av_triton_impl(
            output=output.view(output.shape[0], -1),
            output_splits=output_splits,
            input=cls.input_send_buf,
            input_splits=cls.splits_buf,
            group=group,
        )

        if received_num_tokens == -1:
            received_num_tokens = output_splits.sum().item()
            output = output.narrow(dim=0, start=0, length=received_num_tokens)

        return output, output_splits

    @classmethod
    def initialize(
        cls,
        max_input_seqlen: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        overflow: int = 1,
        async_op: bool = False,
    ):
        assert (
            not async_op
        ), "async mode is not supported now due to some buffer is allocated in self's a2av stream"

        cls.max_output_seqlen = max_input_seqlen * overflow
        cls.input_send_buf = symm_mem.empty(
            max_input_seqlen,
            hidden_size,  # hidden dim
            dtype=dtype,
            device=device,
        )

        cls.is_initialized = True

    @classmethod
    def finalize(cls):
        cls.max_output_seqlen = None
        cls.input_send_buf = None
        cls.splits_buf = None
        cls.grad_output_buf = None

        cls.is_initialized = False


def on_device_a2av(
    input: torch.Tensor,
    input_splits: torch.Tensor,
    output: torch.Tensor | None = None,
    received_num_tokens: int = -1,
    group: dist.ProcessGroup = dist.group.WORLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    return OnDeviceA2AVMgr.apply(
        input,
        input_splits,
        output,
        received_num_tokens,
        group,
    )
