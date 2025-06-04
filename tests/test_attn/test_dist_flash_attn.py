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

import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.nn.functional import all_gather
from torch.nn.functional import scaled_dot_product_attention
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.common.ranges import AttnRanges
from magi_attention.functional.dist_attn import DistFlashAttnRuntime, dist_attn_func
from magi_attention.meta.collection.calc_meta import AttnArg, AttnCalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms


# TODO: add more unitest for dist ffa especially for causal
class TestDistFlashAttn(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend="nccl")
            for _ in range(2)
        ]

        # -----    set up for hier comm   ---- #

        if magi_attention.is_hierarchical_comm_enable():
            assert self.world_size == 4
            world_size_inter_node, world_size_intra_node = 2, 2
            self.device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(world_size_inter_node, world_size_intra_node),
                mesh_dim_names=("inter", "intra"),
            )
        else:
            self.device_mesh = None

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def world_size(self) -> int:
        return 4

    @property
    def seed(self) -> int:
        return 42

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_full_attn(self, dtype):
        device = torch.cuda.current_device()

        attn_calc_meta = AttnCalcMeta(
            local_attn_arg=AttnArg(
                q_ranges=AttnRanges.from_ranges([[0, 128]]),
                k_ranges=AttnRanges.from_ranges([[0, 128]]),
                is_causal_mapping=[False],
                shard_seqlen_q=128,
                total_area=128 * 128,
            ),
            remote_attn_args_list=[
                AttnArg(
                    q_ranges=AttnRanges.from_ranges([[0, 128]]),
                    k_ranges=AttnRanges.from_ranges([[0, 128 * 3]]),
                    is_causal_mapping=[False],
                    shard_seqlen_q=128,
                    total_area=128 * 128 * 3,
                ),
            ],
        )

        comm_meta = CommMeta(
            num_remote_tokens_per_stage=[128 * 3],
            group_collective_args_list=[
                GroupCollectiveArg(
                    input_split_size_list=[128],
                    output_split_size_list=[128, 128, 128],
                    dst_indices_list=[
                        [rank for rank in range(self.world_size) if rank != self.rank]
                    ],
                    src_index_list=[
                        rank for rank in range(self.world_size) if rank != self.rank
                    ],
                    world_size=self.world_size,
                    device_mesh=self.device_mesh,
                )
            ],
        )

        dist_attn_runtime = DistFlashAttnRuntime(
            comm_meta=comm_meta,
            calc_meta=attn_calc_meta,
            cp_group_kv=self.nccl_groups[0],
            cp_group_dkv=self.nccl_groups[1],
            deterministic=False,
        )

        local_q = torch.randn(
            128, 1, 128, device=device, dtype=dtype, requires_grad=True
        )
        local_k = torch.randn(
            128, 1, 128, device=device, dtype=dtype, requires_grad=True
        )
        local_v = torch.randn(
            128, 1, 128, device=device, dtype=dtype, requires_grad=True
        )

        local_out, _ = dist_attn_func(local_q, local_k, local_v, dist_attn_runtime)
        total_out = torch.cat(all_gather(local_out, group=self.nccl_group), dim=0)

        grad_total_out = torch.randn_like(total_out)
        total_out.backward(grad_total_out)
        local_grad_q, local_grad_k, local_grad_v = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )
        local_q.grad, local_k.grad, local_v.grad = None, None, None

        total_q = torch.cat(all_gather(local_q, group=self.nccl_group), dim=0)
        total_k = torch.cat(all_gather(local_k, group=self.nccl_group), dim=0)
        total_v = torch.cat(all_gather(local_v, group=self.nccl_group), dim=0)

        total_out_ref = scaled_dot_product_attention(
            rearrange(total_q, "t h d -> 1 h t d"),
            rearrange(total_k, "t h d -> 1 h t d"),
            rearrange(total_v, "t h d -> 1 h t d"),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        total_out_ref = rearrange(total_out_ref, "1 h t d -> t h d")
        total_out_ref.backward(grad_total_out)
        local_grad_q_ref, local_grad_k_ref, local_grad_v_ref = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )
        local_q.grad, local_k.grad, local_v.grad = None, None, None

        torch.testing.assert_close(total_out, total_out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(local_grad_q, local_grad_q_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(local_grad_k, local_grad_k_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(local_grad_v, local_grad_v_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
