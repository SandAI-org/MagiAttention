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

import functools
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import _get_device_handle

from magi_fsdp._fsdp_common import compiled_autograd_enabled

from ._fsdp_api import CPUOffloadPolicy, OffloadPolicy


class OffloadGlobalContext:
    def lazy_init(self, device: torch.device):
        self.device_handle = _get_device_handle(device.type)
        high_priority = -1
        self.offload_stream: torch.cuda.Stream = self.device_handle.Stream(
            priority=high_priority
        )
        self.device: torch.device = device
        self.saved_activations_order: list[torch.Tensor] = []
        self.h2d_saved_activations_event: list[torch.cuda.Event] = []
        self.backward_calc_finish_event: list[torch.cuda.Event] = []

        # manager order for h2d prefetch
        self.offload_manager_order: list[CPUOffloadManager] = []
        # only use to measure the amount of data offloaded
        self.total_activation_offload: int = 0

    def reset_global_ctx(self):
        self.saved_activations_order.clear()
        self.h2d_saved_activations_event.clear()
        self.backward_calc_finish_event.clear()
        self.offload_manager_order.clear()
        self.total_activation_offload = 0


# FIXME Only support 1 fwd + 1 bwd now
# FIXME support 1 fwd + n bwd and n fwd + 1 bwd case
# FIXME For multiple forward, the activations are overwritten rather than stored for each forward pass.
#       When the computation graph is destroyed, the saved activations are not destroyed along with it.
# FIXME For multiple backward, FSDP cannot access the retain_graph flag.
#       The saved activations are released after the first backward pass.
@dataclass
class CPUOffloadManager:
    offload_global_ctx: OffloadGlobalContext = OffloadGlobalContext()
    offload_policy: OffloadPolicy = field(default_factory=OffloadPolicy)
    saved_activation_idx: list[int] = field(default_factory=list)
    post_forward_indices: list[int] = field(default_factory=list)

    pinned_buffer: torch.Tensor | None = None
    offsets: list[int] = field(default_factory=list)
    orig_dtypes: list[torch.dtype] = field(default_factory=list)
    orig_shapes: list[torch.Size] = field(default_factory=list)
    bytes_per_tensor: list[int] = field(default_factory=list)

    pre_h2d_in_backward: bool = False
    offload_activation_after_forward: bool = True

    def __post_init__(self):
        if (
            isinstance(self.offload_policy, CPUOffloadPolicy)
            and self.offload_policy.offload_activation
        ):
            self.is_activation_offload_policy = True
            self.pin_memory = self.offload_policy.pin_memory
            self.foreach_offload = self.offload_policy.foreach_offload
        else:
            self.is_activation_offload_policy = False

    def pack(self, tensor: torch.Tensor):
        if (
            not self.is_activation_offload_policy
            or isinstance(tensor, nn.Parameter)
            or tensor.device.type == "cpu"
            or not self.offload_activation_after_forward
        ):
            return tensor

        activation_index = len(self.offload_global_ctx.saved_activations_order)
        # save the tensor on gpu
        self.offload_global_ctx.saved_activations_order.append(tensor)
        # add a placeholder for the event
        self.offload_global_ctx.h2d_saved_activations_event.append(None)
        # save the index of the stored GPU tensor
        self.saved_activation_idx.append(activation_index)

        return activation_index

    def unpack(self, packed_data):
        if not self.is_activation_offload_policy or isinstance(
            packed_data, torch.Tensor
        ):
            return packed_data

        # get the index of the GPU tensor and the H2D event
        activation_index = packed_data
        h2d_event = self.offload_global_ctx.h2d_saved_activations_event[
            activation_index
        ]
        # make the compute stream wait for the H2D stream to complete.
        if h2d_event is not None:
            torch.cuda.current_stream().wait_event(h2d_event)
        # return the pointer of the GPU tensor
        gpu_tensor = self.offload_global_ctx.saved_activations_order[activation_index]
        # destroy the pointers to the tensors and events stored in the manager
        self.offload_global_ctx.h2d_saved_activations_event[activation_index] = None
        self.offload_global_ctx.saved_activations_order[activation_index] = None

        return gpu_tensor

    def _register_saved_tensor_hooks(
        self, module: nn.Module, method_name: str = "forward"
    ):
        if not self.is_activation_offload_policy:
            return

        if not hasattr(module, method_name):
            raise ValueError(f"{type(module)} does not have a method {method_name}")
        orig_method = getattr(module, method_name)

        @functools.wraps(orig_method)
        def forward_with_saved_tensor_hooks(ctx, *args, **kwargs):
            with saved_tensors_hooks(self.pack, self.unpack):
                return orig_method(*args, **kwargs)

        setattr(
            module,
            method_name,
            forward_with_saved_tensor_hooks.__get__(  # type:ignore[attr-defined]
                module, type(module)
            ),
        )

    def post_forward(self) -> None:
        self.d2h()
        self.record_post_forward()

    def pre_backward(self, default_prefetch: bool) -> None:
        self.h2d()
        self.record_default_stream()
        if default_prefetch and not compiled_autograd_enabled():
            self._backward_prefetch()

    def record_post_forward(self) -> None:
        post_forward_index = len(self.offload_global_ctx.offload_manager_order)
        self.offload_global_ctx.offload_manager_order.append(self)
        self.post_forward_indices.append(post_forward_index)

    def _backward_prefetch(self) -> None:
        if not self.post_forward_indices:
            # Can be cleared if running multiple `backward`s
            return
        curr_index = self.post_forward_indices.pop()
        if (target_index := curr_index - 1) < 0:
            return
        # Prefetch naively using the reverse post-forward order, which may
        # have mistargeted prefetches if not all modules used in forward
        # are used in this backward
        target_offload_manager = self.offload_global_ctx.offload_manager_order[
            target_index
        ]
        self._prefetch_h2d(target_offload_manager)

    @staticmethod
    def _prefetch_h2d(target_offload_manager: "CPUOffloadManager") -> None:
        target_offload_manager.h2d()

    def init_from_tensors(
        self,
        tensors: list[torch.Tensor],
    ):
        if (
            not self.is_activation_offload_policy
            or not self.offload_activation_after_forward
        ):
            return

        pin_memory = self.pin_memory
        self.placements = [
            tensor.placements if isinstance(tensor, DTensor) else None
            for tensor in tensors
        ]
        self.device_mesh = [
            tensor.device_mesh if isinstance(tensor, DTensor) else None
            for tensor in tensors
        ]
        # convert the DTensor to Tensor to be processed
        for i, tensor in enumerate(tensors):
            if isinstance(tensor, DTensor):
                tensors[i] = tensor.to_local()
        # compute the activations that the current manager should handle.
        self.orig_dtypes = [tensor.dtype for tensor in tensors]
        self.orig_shapes = [tensor.shape for tensor in tensors]
        numels = [tensor.numel() for tensor in tensors]
        bytes_per_element = [tensor.element_size() for tensor in tensors]
        self.bytes_per_tensor = [n * b for n, b in zip(numels, bytes_per_element)]
        self.total_bytes = sum(self.bytes_per_tensor)
        self.offload_global_ctx.total_activation_offload += self.total_bytes
        self.device = self.offload_global_ctx.device
        self.offsets = []

        # If pin_memory is not allocated, or the currently allocated space cannot hold all tensors,
        # allocate an additional block of pin_memory.
        if self.pinned_buffer is None or self.pinned_buffer.numel() < self.total_bytes:
            self.pinned_buffer = torch.empty(
                self.total_bytes, dtype=torch.uint8, device="cpu", pin_memory=pin_memory
            )

        # allocate an additional block of gpu buffer to do d2d operations
        if not self.foreach_offload:
            self.gpu_tensor_buffer = torch.empty(
                self.total_bytes,
                dtype=torch.uint8,
                device=self.device,
            )

        offset = 0
        # calculate the starting position of each tensor within the pinned memory
        for size in self.bytes_per_tensor:
            self.offsets.append(offset)
            offset += size

    def d2h(self):
        if (
            not self.is_activation_offload_policy
            or not self.offload_activation_after_forward
        ):
            return

        # get the list of tensors that the manager should handle.
        global_tensor_list = self.offload_global_ctx.saved_activations_order
        d2h_stream = self.offload_global_ctx.offload_stream
        self.saved_activation_idx = [
            i
            for i in self.saved_activation_idx
            if global_tensor_list[i].untyped_storage().size() != 0
            and (
                not isinstance(global_tensor_list[i], DTensor)
                or global_tensor_list[i].to_local().untyped_storage().size() != 0
            )
        ]
        tensors = [global_tensor_list[i] for i in self.saved_activation_idx]

        # no tensors to offload, return
        if len(tensors) == 0:
            self.total_bytes = 0
            return

        # only use non_blocking when use pin_memory with d2h
        non_blocking = self.pin_memory
        # allocate pinned memory for the tensor and calculate its offset
        self.init_from_tensors(tensors=tensors)
        # destroy the tensor pointers stored in the manager.
        for i in self.saved_activation_idx:
            global_tensor_list[i] = None
        # make the D2H stream wait for the tasks on the compute stream to complete before starting the D2H operation.
        calc_finish_event = torch.cuda.Event()
        calc_finish_event.record()
        d2h_stream.wait_event(calc_finish_event)

        # TODO: Recording a stream will cause the tensor to be released later,
        # TODO: a better mechanism is needed to control the tensor’s release timing.
        with torch.cuda.stream(d2h_stream):
            if self.foreach_offload:
                for gpu_tensor, offset, n_bytes in zip(
                    tensors, self.offsets, self.bytes_per_tensor
                ):
                    # Flatten each tensor, view it as torch.uint8, and copy it into the allocated pinned memory.
                    unit8_gpu_tensor = gpu_tensor.reshape(-1).view(torch.uint8)
                    self.pinned_buffer[offset : offset + n_bytes].copy_(  # type: ignore
                        unit8_gpu_tensor, non_blocking=non_blocking
                    )
                    # Record the tensor on d2h_stream, it will be released only after the copy operation is completed.
                    gpu_tensor.record_stream(d2h_stream)
            else:
                # Flatten each tensor, view it as torch.uint8, and copy it into the allocated pinned memory.
                src_tensor_list = [
                    gpu_tensor.reshape(-1).view(torch.uint8) for gpu_tensor in tensors
                ]
                dst_tensor_list = [
                    self.gpu_tensor_buffer[offset : offset + n_bytes]
                    for offset, n_bytes in zip(self.offsets, self.bytes_per_tensor)
                ]

                # copy the gpu_tensor into a large GPU buffer.
                torch._foreach_copy_(
                    dst_tensor_list, src_tensor_list, non_blocking=True
                )
                # record the tensor on d2h_stream, it will be released only after the copy operation is completed.
                for gpu_tensor in tensors:
                    gpu_tensor.record_stream(d2h_stream)
                # copy the GPU buffer from device to host
                self.pinned_buffer[: self.total_bytes].copy_(  # type: ignore
                    self.gpu_tensor_buffer[: self.total_bytes],
                    non_blocking=non_blocking,
                )
                self.gpu_tensor_buffer.record_stream(d2h_stream)
                del self.gpu_tensor_buffer

    def h2d(self):
        if (
            not self.is_activation_offload_policy
            or not self.offload_activation_after_forward
            or self.pre_h2d_in_backward
            or self.total_bytes == 0
        ):
            return

        # This module has already performed the H2D (host-to-device).
        self.pre_h2d_in_backward = True

        # only use non_blocking when use pin_memory with h2d
        non_blocking = self.pin_memory

        # get the list of tensors that the manager should handle.
        global_tensor_list = self.offload_global_ctx.saved_activations_order
        h2d_stream: torch.cuda.Stream = self.offload_global_ctx.offload_stream

        # During the backward pass, the H2D transfer of this module should overlap with
        # the computation of the previous module. However, if the CPU is too fast,
        # it will issue H2D transfers without limit, so an explicit wait on the compute stream is required.
        backward_calc_finish_idx = (
            len(self.offload_global_ctx.backward_calc_finish_event) - 1
        )
        if backward_calc_finish_idx >= 0:
            h2d_stream.wait_event(
                self.offload_global_ctx.backward_calc_finish_event[
                    backward_calc_finish_idx
                ]
            )

        with torch.cuda.stream(h2d_stream):
            if self.foreach_offload:
                for (
                    offset,
                    n_bytes,
                    shape,
                    dtype,
                    activation_idx,
                    device_mesh,
                    placement,
                ) in zip(
                    self.offsets,
                    self.bytes_per_tensor,
                    self.orig_shapes,
                    self.orig_dtypes,
                    self.saved_activation_idx,
                    self.device_mesh,
                    self.placements,
                ):
                    # get the flattened tensor
                    t_bytes = self.pinned_buffer[offset : offset + n_bytes]
                    # restore the tensor’s original dtype and shape, and perform the H2D operation
                    gpu_tensor = (
                        t_bytes.view(dtype)
                        .view(shape)
                        .to(self.device, non_blocking=non_blocking)
                    )
                    # convert Tensor to DTensor
                    if device_mesh is not None:
                        gpu_tensor = DTensor.from_local(
                            gpu_tensor, device_mesh=device_mesh, placements=placement
                        )
                    global_tensor_list[activation_idx] = gpu_tensor

                    # record an event on the H2D stream so that
                    # operations on the compute stream can only execute after the H2D operation is complete
                    h2d_event = torch.cuda.Event()
                    h2d_event.record(h2d_stream)
                    self.offload_global_ctx.h2d_saved_activations_event[
                        activation_idx
                    ] = h2d_event
            else:
                self.gpu_tensor_buffer = self.pinned_buffer[: self.total_bytes].to(
                    self.device, non_blocking=non_blocking
                )
                h2d_event = torch.cuda.Event()
                h2d_event.record(h2d_stream)

                for (
                    offset,
                    n_bytes,
                    shape,
                    dtype,
                    activation_idx,
                    device_mesh,
                    placement,
                ) in zip(
                    self.offsets,
                    self.bytes_per_tensor,
                    self.orig_shapes,
                    self.orig_dtypes,
                    self.saved_activation_idx,
                    self.device_mesh,
                    self.placements,
                ):
                    # get the flattened tensor
                    t_bytes = self.gpu_tensor_buffer[offset : offset + n_bytes]
                    # restore the tensor’s original dtype and shape, and perform the H2D operation
                    gpu_tensor = t_bytes.view(dtype).view(shape)
                    # convert Tensor to DTensor
                    if device_mesh is not None:
                        gpu_tensor = DTensor.from_local(
                            gpu_tensor, device_mesh=device_mesh, placements=placement
                        )
                    global_tensor_list[activation_idx] = gpu_tensor

                    self.offload_global_ctx.h2d_saved_activations_event[
                        activation_idx
                    ] = h2d_event

                del self.gpu_tensor_buffer

    def record_default_stream(self):
        # record the default stream and add the event to the global array.
        backward_calc_event = torch.cuda.Event()
        backward_calc_event.record()
        self.offload_global_ctx.backward_calc_finish_event.append(backward_calc_event)

    def reset_cpu_offload_manager(self):
        self.saved_activation_idx.clear()
        self.post_forward_indices.clear()
        self.pre_h2d_in_backward = False
