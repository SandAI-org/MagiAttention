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

from typing import Any

import torch.distributed as dist

from magi_attention.utils.metaclass import SingletonMeta

from ._buffer import GrpCollBuffer
from ._config import GrpCollConfig

__all__ = ["grpcoll_mgr"]


class GrpCollMgr(metaclass=SingletonMeta):
    """
    A singleton class to manage GrpCollBuffer for each registered ProcessGroup,
    including the initialization, kernel launch and the resource release.
    """

    def __init__(self):
        self._group_to_buffer: dict[dist.ProcessGroup, GrpCollBuffer] = {}
        self._group_to_config: dict[dist.ProcessGroup, GrpCollConfig] = {}
        self._group_to_args: dict[dist.ProcessGroup, dict[str, Any]] = {}

    def register_buffer(
        self,
        group: dist.ProcessGroup,
        config: GrpCollConfig = GrpCollConfig(),
        **kwargs,
    ):
        # TODO: support other group size
        assert group.size() in (
            2,
            4,
            8,
        ), "For now, only support group size in (2, 4, 8)"

        self.check_released(group)

        self._group_to_config[group] = config

        buffer_args: dict[str, Any] = config.to_buffer_args()
        buffer_args.update(kwargs)
        self._group_to_args[group] = buffer_args

        self._group_to_buffer[group] = GrpCollBuffer(
            group=group,
            **buffer_args,
        )
        dist.barrier(group)

    def release_buffer(
        self,
        group: dist.ProcessGroup,
        **kwargs,
    ):
        self.check_registered(group)

        self._group_to_config.pop(group)
        self._group_to_args.pop(group)
        buffer = self._group_to_buffer.pop(group)

        buffer.destroy()
        dist.barrier(group)

    def get_config(self, group: dist.ProcessGroup) -> GrpCollConfig:
        self.check_registered(group)
        return self._group_to_config[group]

    def get_args(self, group: dist.ProcessGroup) -> dict[str, Any]:
        self.check_registered(group)
        return self._group_to_args[group]

    def get_buffer(self, group: dist.ProcessGroup) -> GrpCollBuffer:
        self.check_registered(group)
        return self._group_to_buffer[group]

    def is_registered(self, group: dist.ProcessGroup) -> bool:
        return (
            group in self._group_to_config
            and group in self._group_to_args
            and group in self._group_to_buffer
        )

    def is_released(self, group: dist.ProcessGroup) -> bool:
        return (
            group not in self._group_to_config
            and group not in self._group_to_args
            and group not in self._group_to_buffer
        )

    def check_registered(self, group: dist.ProcessGroup) -> None:
        if not self.is_registered(group):
            raise ValueError(
                f"ProcessGroup {group.group_name} is not registered. "
                "Please call `register_buffer` first."
            )

    def check_released(self, group: dist.ProcessGroup) -> None:
        if not self.is_released(group):
            raise ValueError(
                f"ProcessGroup {group.group_name} is already registered. "
                "Please call `release_buffer` first."
            )

    def __del__(self):
        non_released_groups = list(self._group_to_buffer.keys())
        for group in non_released_groups:
            self.release_buffer(group)


grpcoll_mgr = GrpCollMgr()
