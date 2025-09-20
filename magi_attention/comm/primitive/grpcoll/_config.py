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

from dataclasses import dataclass

from magi_attention.magi_attn_comm import grpcoll

__all__ = ["GrpCollConfig"]


@dataclass(frozen=True)
class GrpCollConfig:
    # for kernel performance
    num_sms: int = 24
    nvl_chunk_size: int = 24
    nvl_buffer_size: int = 256
    rdma_chunk_size: int = 12
    rdma_buffer_size: int = 128

    # for buffer initialization
    # TODO: set the best values automatically
    num_nvl_bytes: int = int(2e9)  # default 2GB
    num_rdma_bytes: int = 0  # FIXME: 0 for now since we don't support RDMA

    def __post_init__(self):
        pass

    def to_kernel_config(self) -> grpcoll.Config:
        return grpcoll.Config(
            self.num_sms,  # num_sms, default 20
            self.nvl_chunk_size,  # num_max_nvl_chunked_send_tokens, default 6
            self.nvl_buffer_size,  # num_max_nvl_chunked_recv_tokens, default 256
            self.rdma_chunk_size,  # num_max_rdma_chunked_send_tokens, default 6
            self.rdma_buffer_size,  # num_max_rdma_chunked_recv_tokens, default 256
        )

    def to_buffer_args(self) -> dict:
        return dict(
            num_nvl_bytes=self.num_nvl_bytes,
            num_rdma_bytes=self.num_rdma_bytes,
            num_qps_per_rank=1 if self.num_rdma_bytes == 0 else self.num_sms,
            low_latency_mode=False,
            explicitly_destroy=True,
        )
