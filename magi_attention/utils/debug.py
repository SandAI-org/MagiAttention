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

import os

import debugpy
import torch.distributed as dist


def debugpy_listen():  # pragma: no cover
    ENABLE_REMOTE_DEBUG = os.environ.get(
        "MAGI_ATTENTION_ENABLE_REMOTE_DEBUG", "false"
    ).lower()
    if ENABLE_REMOTE_DEBUG != "false":
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        debug_ranks = []
        if ENABLE_REMOTE_DEBUG == "true":
            debug_ranks = [0]
        elif ENABLE_REMOTE_DEBUG == "all":
            debug_ranks = [i for i in range(world_size)]
        else:
            debug_ranks = [int(i) for i in ENABLE_REMOTE_DEBUG.split(",")]

        if rank in debug_ranks:
            debug_port = 37777 + int(rank)
            print(f"[rank {rank}] Starting remote debug on port {debug_port}")
            debugpy.listen(("127.0.0.1", debug_port))
            debugpy.wait_for_client()
            print(f"[rank {rank}] Remote debug attached")
