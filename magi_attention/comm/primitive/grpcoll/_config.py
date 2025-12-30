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

import torch


class Config:
    ...


is_magi_attn_comm_installed = False
try:
    from magi_attention.magi_attn_comm.grpcoll import (  # type: ignore[no-redef] # noqa
        Config,
    )

    is_magi_attn_comm_installed = True
except ImportError:
    pass


__all__ = ["GrpCollConfig"]


@dataclass(frozen=True)
class GrpCollConfig:
    # for kernel performance
    # TODO: add performance tuning guide
    num_sms: int = 24
    nvl_chunk_size: int = 8
    nvl_buffer_size: int = 256
    rdma_chunk_size: int = 4
    rdma_buffer_size: int = 128

    # for buffer initialization
    num_nvl_bytes: int = int(2e9)  # default ~2GB
    # NOTE: set it to a positive value (e.g. `int(1e9)`)
    # if and only if internode communication is required
    num_rdma_bytes: int = 0

    def __post_init__(self):
        pass

    def to_kernel_config(self) -> Config:
        assert (
            is_magi_attn_comm_installed
        ), "The `magi_attn_comm` extension module is not installed."
        return Config(  # type: ignore[call-arg]
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

    @staticmethod
    def get_default_group_cast_config(num_ranks: int) -> "GrpCollConfig":
        """
        Get a recommended group cast config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: GrpCollConfig(24, 24, 256, 6, 128),
            4: GrpCollConfig(24, 6, 256, 6, 128),
            8: GrpCollConfig(24, 6, 256, 6, 128),
            16: GrpCollConfig(24, 36, 288, 20, 128),
            24: GrpCollConfig(24, 8, 288, 32, 128),
            32: GrpCollConfig(24, 32, 288, 32, 128),
            64: GrpCollConfig(24, 20, 288, 28, 128),
            128: GrpCollConfig(24, 20, 560, 32, 128),
            144: GrpCollConfig(24, 32, 720, 12, 128),
            160: GrpCollConfig(24, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of ranks: {num_ranks}"
        return config_map[num_ranks]

    @staticmethod
    def get_default_group_reduce_config(num_ranks: int) -> "GrpCollConfig":
        """
        Get a recommended group reduce config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: GrpCollConfig(24, 10, 256, 6, 128),
            4: GrpCollConfig(24, 9, 256, 6, 128),
            8: GrpCollConfig(24, 4, 256, 6, 128),
            16: GrpCollConfig(24, 4, 288, 12, 128),
            24: GrpCollConfig(24, 1, 288, 8, 128),
            32: GrpCollConfig(24, 1, 288, 8, 128),
            64: GrpCollConfig(24, 1, 288, 20, 128),
            128: GrpCollConfig(24, 1, 560, 12, 128),
            144: GrpCollConfig(24, 2, 720, 8, 128),
            160: GrpCollConfig(24, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of ranks: {num_ranks}"
        return config_map[num_ranks]

    @staticmethod
    def get_min_num_bytes_intranode(
        num_sms: int,
        num_ranks: int,
        hidden_size: int,
        nvl_buffer_size: int,
        dtype: torch.dtype,
        transfer_lse: bool = False,
        num_heads: int | None = None,
        num_groups: int = 1,
        alignment: int = 128,  # according to `NUM_BUFFER_ALIGNMENT_BYTES`
    ) -> int:
        if transfer_lse:
            assert (
                num_heads is not None
            ), "num_heads must be set when transfer_lse is True"
            assert (
                hidden_size % num_heads == 0
            ), "hidden_size must be divisible by num_heads"
        else:
            num_heads = 0

        assert num_sms % 2 == 0, "num_sms must be even"
        num_channels = num_sms // 2

        # fmt: off
        return (
            # rank prefix matrix
            num_ranks * num_ranks * torch.int32.itemsize
            # channel start/end offset + queue head/tail
            + num_channels * num_ranks * torch.int32.itemsize * 4
            # token data buffer
            + num_channels * num_ranks * nvl_buffer_size * hidden_size * dtype.itemsize * num_groups
            # src idx
            + num_channels * num_ranks * nvl_buffer_size * torch.int32.itemsize
            # lse data buffer
            + num_channels * num_ranks * nvl_buffer_size * num_heads * torch.float32.itemsize
            # max padding bytes to align for vectorized token data buffer (int4)
            + 16 * num_groups
            # align up to `alignment`
            + alignment - 1
        ) // alignment * alignment
        # fmt: on

    @staticmethod
    def get_min_num_bytes_internode(
        num_sms: int,
        num_rdma_ranks: int,
        num_nvl_ranks: int,
        hidden_size: int,
        rdma_buffer_size: int,
        nvl_buffer_size: int,
        dtype: torch.dtype,
        transfer_lse: bool = False,
        num_heads: int | None = None,
        num_groups: int = 1,
        alignment: int = 128,  # according to `NUM_BUFFER_ALIGNMENT_BYTES`
        rdma_decoulped: bool = True,
    ) -> tuple[int, int]:
        if transfer_lse:
            assert (
                num_heads is not None
            ), "num_heads must be set when transfer_lse is True"
            assert (
                hidden_size % num_heads == 0
            ), "hidden_size must be divisible by num_heads"
        else:
            num_heads = 0

        assert num_sms % 2 == 0, "num_sms must be even"
        num_channels = num_sms // 2

        num_data_buffers = 2 if rdma_decoulped else 1

        hidden_bytes_per_token = hidden_size * dtype.itemsize * num_groups
        lse_bytes_per_token = num_heads * torch.float32.itemsize
        src_meta_bytes_per_token = 2 * torch.int32.itemsize
        num_bytes_per_token = (
            hidden_bytes_per_token + lse_bytes_per_token + src_meta_bytes_per_token
        )

        # fmt: off
        return (
            # data buffer (hidden states + lse + src meta)
            num_channels * num_data_buffers * num_rdma_ranks * rdma_buffer_size * num_bytes_per_token
            # meta buffer (queue head/tail per NVL rank + start/end idx for RDMA/NVL)
            + num_channels * num_data_buffers * num_rdma_ranks * (num_nvl_ranks * 2 + 4) * torch.int32.itemsize
            # align up to `alignment`
            + alignment - 1
        ) // alignment * alignment, (
            # data buffer (hidden states + lse + src meta)
            num_channels * num_nvl_ranks * nvl_buffer_size * num_bytes_per_token
            # meta buffer (queue head/tail per RDMA rank + start/end idx for NVL)
            + num_channels * num_nvl_ranks * (num_rdma_ranks * 2 + 2) * torch.int32.itemsize
            # align up to `alignment`
            + alignment - 1
        ) // alignment * alignment
        # fmt: on
