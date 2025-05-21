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
from typing import Any, Callable

from torch.distributed import Work

from magi_attention.utils import wrap_to_list


@dataclass
class WorkWithPostProcessFn:
    """A dist work with a post process func to call after waiting
    HACK: this is a hack way to apply async op
        that the user needs to call post process manually after waiting
    NOTE: this is a disposable object,
        once the work + post process fn are both done,
        the user are not supposed to use this object anymore
    """

    work: Work | list[Work] | None = None
    post_process_fn: Callable = field(
        default_factory=lambda: lambda *args, **kwargs: None
    )
    sync: bool = False

    def __post_init__(self):
        # the flag to note if
        # the optional work + post process fn are both done
        # to avoid repeatedly calling post process fn
        self._work_done = False

        if self.work is not None:
            self.work = wrap_to_list(self.work)

        # if sync mode, the given work needs to wait immediately
        if self.sync and self.work is not None:
            for work in self.work:
                work.wait()
            self.work = None

    def wait_post_process(self, *args, **kwargs) -> Any:
        if self._work_done:
            raise RuntimeError("Work has already been done.")

        if self.work is not None:
            for work in self.work:
                work.wait()
            self.work = None

        ret = self.post_process_fn(*args, **kwargs)

        self._work_done = True

        return ret
