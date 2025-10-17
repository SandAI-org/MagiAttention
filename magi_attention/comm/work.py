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

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeAlias

import torch
from torch.cuda import Event, Stream
from torch.distributed import Work

from magi_attention.utils import wrap_to_list

from .primitive.grpcoll._event import EventOverlap

GeneralWorkItem: TypeAlias = Work | Stream | Event | EventOverlap | None


@dataclass
class GeneralWork:
    work: GeneralWorkItem | Sequence[GeneralWorkItem] | "GeneralWork" = None

    def __post_init__(self):
        self.work_list: list[GeneralWorkItem] = wrap_to_list(self.work)

    def wait(self) -> None:
        for work in self.work_list:
            match work:
                case GeneralWork():  # recursively wait
                    work.wait()
                case Stream():
                    torch.cuda.current_stream().wait_stream(work)
                case Event():
                    torch.cuda.current_stream().wait_event(work)
                case Work():
                    # NOTE: WorkNCCL::wait only blocks the current stream
                    # on the NCCL stream by default
                    # unless in blocking mode, in which it will block CPU as well
                    work.wait()
                case EventOverlap():
                    work.current_stream_wait()
                case None:
                    pass
                case _:
                    raise TypeError(f"Unsupported type: {type(work)=}")


@dataclass
class WorkWithPostProcessFn:
    """A dist work with a post process func to call after waiting

    HACK: this is a hack way to apply async op
        that the user needs to call post process manually after waiting

    NOTE: this is a disposable object,
        once the work + post process fn are both done,
        the user are not supposed to use this object anymore
    """

    work: GeneralWork | None = None
    post_process_fn: Callable = field(
        default_factory=lambda: lambda *args, **kwargs: None
    )
    async_op: bool = False

    def __post_init__(self):
        # the flag to note if
        # the optional work + post process fn are both done
        # to avoid repeatedly calling post process fn
        self._work_done = False

        # if sync mode, the given work needs to wait immediately
        if not self.async_op:
            self._wait_work()

    def wait_post_process(self, *args, **kwargs) -> Any:
        """Wait for the work to be done,
        then call the post process fn with the given args

        NOTE: this is a one-time API
        """
        if self._work_done:
            raise RuntimeError("Work has already been done.")

        self._wait_work()

        return self._apply_post_process(*args, **kwargs)

    def _wait_work(self) -> None:
        if self.work is not None:
            self.work.wait()
            self.work = None

    def _apply_post_process(self, *args, **kwargs) -> Any:
        ret = self.post_process_fn(*args, **kwargs)

        # when work is done, the post process fn is no longer needed
        # so we set it to a no-op, to avoid some long lived objects
        # e.g. some partial funcs might hold on some tensors
        self.post_process_fn = lambda *args, **kwargs: None

        self._work_done = True

        return ret
