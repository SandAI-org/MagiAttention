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

from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.nn.functional import all_gather
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_mgr
from magi_attention.common.ranges import AttnRanges
from magi_attention.functional.dist_attn import DistAttnRuntime, dist_attn_func
from magi_attention.meta.collection.calc_meta import AttnArg, CalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.precision import EPSILON, assert_close
from magi_attention.testing.utils import switch_envvar_context, switch_envvars


class TestDistAttn(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        self.sdpa_backend_envvar = "MAGI_ATTENTION_SDPA_BACKEND"

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend="nccl")
            for _ in range(2)
        ]

        # -----    set up for hier comm   ---- #

        self.hier_comm_envvar = "MAGI_ATTENTION_HIERARCHICAL_COMM"
        self.switch_hier_comm_context = partial(
            switch_envvar_context, envvar_name=self.hier_comm_envvar
        )

        assert self.world_size == 4
        world_size_inter_node, world_size_intra_node = 2, 2
        self.device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size_inter_node, world_size_intra_node),
            mesh_dim_names=("inter", "intra"),
        )

        # -----    set up for native grpcoll   ---- #

        self.native_grpcoll_envvar = "MAGI_ATTENTION_NATIVE_GRPCOLL"
        self.switch_native_grpcoll_context = partial(
            switch_envvar_context, envvar_name=self.native_grpcoll_envvar
        )

        for nccl_group in self.nccl_groups:
            grpcoll_mgr.register_buffer(
                group=nccl_group,
                config=GrpCollConfig(
                    num_sms=24,
                    nvl_chunk_size=8,
                    nvl_buffer_size=256,
                    rdma_chunk_size=8,
                    rdma_buffer_size=256,
                    num_nvl_bytes=int(1e9),
                    num_rdma_bytes=0,
                ),
            )
            grpcoll_mgr.check_registered(group=nccl_group)

    def destroy_pg(self):
        for nccl_group in self.nccl_groups:
            grpcoll_mgr.release_buffer(group=nccl_group)
            grpcoll_mgr.check_released(group=nccl_group)

        super().destroy_pg()

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def world_size(self) -> int:
        return 4

    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize("seqlen_sink", [0, 4])
    @parameterize("num_heads", [(8, 8), (8, 4)])
    @parameterize("head_dim", [128, 64])
    @parameterize("use_sdpa_backend", [False, True])
    @parameterize("use_hier_comm", [False, True])
    @parameterize("use_native_grpcoll", [False, True])
    @parameterize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_full_attn(
        self,
        seqlen_sink: int,
        num_heads: tuple[int, int],
        head_dim: int,
        use_sdpa_backend: bool,
        use_hier_comm: bool,
        use_native_grpcoll: bool,
        dtype: torch.dtype,
    ):
        # skip when enabling hier comm
        if use_hier_comm:
            # TODO: support hier comm with native grpcoll
            if use_native_grpcoll:
                return

        # switch the env flags
        switch_back = switch_envvars(
            envvar_name_list=[
                self.sdpa_backend_envvar,
                self.hier_comm_envvar,
                self.native_grpcoll_envvar,
            ],
            enable_dict={
                self.sdpa_backend_envvar: use_sdpa_backend,
                self.hier_comm_envvar: use_hier_comm,
                self.native_grpcoll_envvar: use_native_grpcoll,
            },
        )

        # prepare meta and runtime
        # TODO: add more attn masks for dist attn
        nhq, nhk = num_heads
        calc_meta = CalcMeta(
            local_attn_arg=AttnArg(
                q_ranges=AttnRanges.from_ranges([[0, 128]]),
                k_ranges=AttnRanges.from_ranges([[0, 128]]),
                attn_type_map=[0],
                total_area=128 * 128,
            ),
            remote_attn_args_list=[
                AttnArg(
                    q_ranges=AttnRanges.from_ranges([[0, 128]]),
                    k_ranges=AttnRanges.from_ranges([[0, 128 * 3]]),
                    attn_type_map=[0],
                    total_area=128 * 128 * 3,
                ),
            ],
        )
        comm_meta = CommMeta(
            num_remote_kv_tokens_per_stage=[128 * 3],
            kv_group_collective_args_list=[
                GroupCollectiveArg(
                    input_split_size_list=[128],
                    output_split_size_list=[128, 128, 128],
                    dst_indices_list=[
                        [rank for rank in range(self.world_size) if rank != self.rank]
                    ],
                    src_index_list=[
                        rank for rank in range(self.world_size) if rank != self.rank
                    ],
                    rank=self.rank,
                    world_size=self.world_size,
                    group=self.nccl_group,
                    device_mesh=self.device_mesh if use_hier_comm else None,
                )
            ],
            # TODO: support qo comm meta calculation
            num_remote_qo_tokens_per_stage=[0],
            qo_group_collective_args_list=[None],  # type: ignore[list-item]
        )
        dist_attn_runtime = DistAttnRuntime(
            comm_meta=comm_meta,
            calc_meta=calc_meta,
            cp_group_gc=self.nccl_groups[0],
            cp_group_gr=self.nccl_groups[1],
        )

        # prepare data
        local_q = torch.randn(
            128, nhq, head_dim, device=self.device, dtype=dtype, requires_grad=True
        )
        local_k = torch.randn(
            128, nhk, head_dim, device=self.device, dtype=dtype, requires_grad=True
        )
        local_v = torch.randn(
            128, nhk, head_dim, device=self.device, dtype=dtype, requires_grad=True
        )
        total_mask = torch.ones(512, 512, device=self.device).bool()
        if seqlen_sink > 0:
            total_sink = torch.randn(
                seqlen_sink,
                nhq,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            dist.all_reduce(total_sink.data, group=self.nccl_group)
        else:
            total_sink = None

        # run dist attn func
        local_out, local_lse = dist_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dist_attn_runtime=dist_attn_runtime,
            sink=total_sink,
        )
        total_out = torch.cat(all_gather(local_out, group=self.nccl_group), dim=0)
        total_lse = torch.cat(all_gather(local_lse, group=self.nccl_group), dim=0)

        grad_total_out = torch.randn_like(total_out)
        total_out.backward(grad_total_out)
        local_grad_q, local_grad_k, local_grad_v = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )
        local_q.grad, local_k.grad, local_v.grad = None, None, None
        if total_sink is not None:
            total_dsink = total_sink.grad
            total_sink.grad = None
        else:
            total_dsink = None

        total_q = torch.cat(all_gather(local_q, group=self.nccl_group), dim=0)
        total_k = torch.cat(all_gather(local_k, group=self.nccl_group), dim=0)
        total_v = torch.cat(all_gather(local_v, group=self.nccl_group), dim=0)

        # switch the env flags back
        switch_back()

        # run ref attn func
        total_out_ref, total_lse_ref = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=total_mask,
            sink=total_sink,
            layout="thd",
            sink_layout="sh",
            backend="torch" if total_sink is not None else "sdpa",
            high_precision=True,
            return_lse=True,
        )
        total_out_ref.backward(grad_total_out)
        local_grad_q_ref, local_grad_k_ref, local_grad_v_ref = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )
        if total_sink is not None:
            total_dsink_ref = total_sink.grad
            dist.all_reduce(total_dsink_ref.data, group=self.nccl_group)
        else:
            total_dsink_ref = None

        # check results
        assert_close(
            total_out,
            total_out_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.1 if use_sdpa_backend else 0.08,
            test_case="out",
        )
        assert_close(
            total_lse,
            total_lse_ref,
            atol=EPSILON,
            rtol=5e-3,
            mismatch_threshold=0.01,
            test_case="lse",
        )
        assert_close(
            local_grad_q,
            local_grad_q_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.1 if use_sdpa_backend else 0.08,
            test_case="dq",
        )
        assert_close(
            local_grad_k,
            local_grad_k_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.1 if use_sdpa_backend else 0.08,
            test_case="dk",
        )
        assert_close(
            local_grad_v,
            local_grad_v_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.1 if use_sdpa_backend else 0.08,
            test_case="dv",
        )
        if total_sink is not None:
            assert_close(
                total_dsink,
                total_dsink_ref,
                atol=1e-3,
                rtol=0.1,
                mismatch_threshold=max(1 / (seqlen_sink * nhq), 5e-2),
                test_case="dsink",
            )


if __name__ == "__main__":
    run_tests()
