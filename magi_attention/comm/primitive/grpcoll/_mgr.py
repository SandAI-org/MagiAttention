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

from typing import Any, Optional, Union

import torch.distributed as dist

from magi_attention.common.enum import GrpCollBufferName
from magi_attention.utils.metaclass import SingletonMeta

from ._buffer import GrpCollBuffer
from ._config import GrpCollConfig

__all__ = ["grpcoll_buffer_mgr"]


# TODO: make (process_group, buffer_name) pair as the key for grpcoll buffer
class GrpCollBufferMgr(metaclass=SingletonMeta):
    """
    A singleton class to manage GrpCollBuffer instances by name.
    It requires initialization with a ProcessGroup and Config, then supports
    lazy initialization of buffers via get_buffer.
    """

    def __init__(self):
        self._name_to_buffer: dict[str, GrpCollBuffer] = {}

        # State storage for lazy initialization
        self._group: Optional[dist.ProcessGroup] = None
        self._config: Optional[GrpCollConfig] = None
        self._common_args: dict[str, Any] = {}

        self._is_initialized: bool = False

    def initialize(
        self,
        group: dist.ProcessGroup,
        config: GrpCollConfig = GrpCollConfig(),
        **kwargs,
    ):
        """
        Initialize the manager with the default ProcessGroup and Configuration
        that will be used for all lazily created buffers.
        """
        if self._is_initialized:
            # Logic for re-initialization if necessary (e.g. warning or reset)
            pass

        self._group = group
        self._config = config

        buffer_args = config.to_buffer_args()
        buffer_args.update(kwargs)
        self._common_args = buffer_args

        self._is_initialized = True

    def get_buffer(self, buffer_name: Union[str, GrpCollBufferName]) -> GrpCollBuffer:
        """
        Retrieve a buffer by name (or Enum). If it does not exist, it is lazily initialized
        using the group and config provided during `initialize`.
        """
        self.check_initialized()

        # Normalize to string key
        name_key = (
            buffer_name.value
            if isinstance(buffer_name, GrpCollBufferName)
            else buffer_name
        )

        if name_key not in self._name_to_buffer:
            # Lazy initialization
            new_buffer = GrpCollBuffer(
                group=self._group,
                **self._common_args,
            )
            self._name_to_buffer[name_key] = new_buffer

            # Ensure synchronization across the group upon creation to avoid race conditions
            # or usage before peer buffers are ready.
            dist.barrier(self._group)

        return self._name_to_buffer[name_key]

    def release_buffer(self, buffer_name: Union[str, GrpCollBufferName]):
        """
        Release and destroy a specific named buffer.
        """
        self.check_initialized()

        name_key = (
            buffer_name.value
            if isinstance(buffer_name, GrpCollBufferName)
            else buffer_name
        )

        if name_key not in self._name_to_buffer:
            return

        buffer = self._name_to_buffer.pop(name_key)
        buffer.destroy()

        # Ensure synchronization across the group upon destruction
        dist.barrier(self._group)

    def check_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "GrpCollBufferMgr is not initialized. "
                "Please call `grpcoll_buffer_mgr.initialize(group, config)` first."
            )

    def get_config(self) -> GrpCollConfig:
        self.check_initialized()
        assert self._config is not None
        return self._config

    def get_group(self) -> dist.ProcessGroup:
        self.check_initialized()
        assert self._group is not None
        return self._group

    def __del__(self):
        non_released_names = list(self._name_to_buffer.keys())
        for name in non_released_names:
            if name in self._name_to_buffer:
                buffer = self._name_to_buffer.pop(name)
                try:
                    buffer.destroy()
                except Exception:
                    pass


grpcoll_buffer_mgr = GrpCollBufferMgr()
