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

from typing import Any, Union

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
        # key: (ProcessGroup, str)
        self._buffers: dict[tuple[dist.ProcessGroup, str], GrpCollBuffer] = {}

        # group-specific config storage
        self._group_to_config: dict[dist.ProcessGroup, GrpCollConfig] = {}
        self._group_to_common_args: dict[dist.ProcessGroup, dict[str, Any]] = {}

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
        if group in self._group_to_config:
            # Logic for re-initialization if necessary (e.g. warning or reset)
            pass

        buffer_args = config.to_buffer_args()
        buffer_args.update(kwargs)

        self._group_to_config[group] = config
        self._group_to_common_args[group] = buffer_args

    def get_buffer(
        self, group: dist.ProcessGroup, buffer_name: Union[str, GrpCollBufferName]
    ) -> GrpCollBuffer:
        """
        Retrieve a buffer by name (or Enum). If it does not exist, it is lazily initialized
        using the group and config provided during `initialize`.
        """
        self.check_initialized(group=group)

        # Normalize to string key
        name_key = (
            buffer_name.value
            if isinstance(buffer_name, GrpCollBufferName)
            else buffer_name
        )
        key = (group, name_key)

        if key not in self._buffers:
            # Lazy initialization
            new_buffer = GrpCollBuffer(
                group=group,
                **self._group_to_common_args[group],
            )
            self._buffers[key] = new_buffer

            # Ensure synchronization across the group upon creation to avoid race conditions
            # or usage before peer buffers are ready.
            dist.barrier(group)

        return self._buffers[key]

    def release_buffer(
        self, group: dist.ProcessGroup, buffer_name: Union[str, GrpCollBufferName]
    ):
        """
        Release and destroy a specific named buffer for a group.
        """
        self.check_initialized(group=group)

        name_key = (
            buffer_name.value
            if isinstance(buffer_name, GrpCollBufferName)
            else buffer_name
        )
        key = (group, name_key)

        if key not in self._buffers:
            return

        buffer = self._buffers.pop(key)
        buffer.destroy()

        # Ensure synchronization across the group upon destruction
        dist.barrier(group)

    def release_group(self, group: dist.ProcessGroup):
        """
        Release all buffers belonging to a group.
        """

        keys_to_remove = [k for k in self._buffers if k[0] is group]

        for key in keys_to_remove:
            buffer = self._buffers.pop(key)
            try:
                buffer.destroy()
            except Exception:
                pass

        self._group_to_config.pop(group, None)
        self._group_to_common_args.pop(group, None)

    def _is_initialized(self, group: dist.ProcessGroup) -> bool:
        return group in self._group_to_config

    def check_initialized(self, group: dist.ProcessGroup) -> None:
        if group not in self._group_to_common_args:
            raise RuntimeError(
                f"GrpCollBufferMgr for {group=} is not initialized. "
                f"Please call `grpcoll_buffer_mgr.initialize(group, config)` first."
            )

    def get_config(
        self,
        group: dist.ProcessGroup,
    ) -> GrpCollConfig:
        self.check_initialized(group=group)
        return self._group_to_config[group]

    def __del__(self):
        non_released_names = list(self._buffers.keys())
        for name in non_released_names:
            if name in self._buffers:
                buffer = self._buffers.pop(name)
                try:
                    buffer.destroy()
                except Exception:
                    pass

        try:
            self._group_to_config.clear()
            self._group_to_common_args.clear()
        except Exception:
            pass


grpcoll_buffer_mgr = GrpCollBufferMgr()
