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
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.nn.functional import all_gather
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from magi_attention import env
from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr
from magi_attention.common.enum import MagiAttentionKernelBackend
from magi_attention.common.ranges import AttnRanges
from magi_attention.functional.dist_attn import DistAttnRuntime, dist_attn_func
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_fwd_sm100 import FFAFwdSm100
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.meta.collection.calc_meta import AttnArg, CalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.flag_generator import FlagCombGenerator
from magi_attention.testing.precision import EPSILON, assert_close
from magi_attention.testing.utils import switch_envvar_context, switch_envvars


class TestDistAttn(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        self.kernel_backend_envvar = "MAGI_ATTENTION_KERNEL_BACKEND"

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

        self.native_grpcoll_registered = True
        for nccl_group in self.nccl_groups:
            try:
                grpcoll_buffer_mgr.initialize(
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
            except Exception as e:
                self.native_grpcoll_registered = False
                print(
                    f"The NCCL group {nccl_group} cannot be registered due to error: \n{e}\n"
                )

        self.flag_generator = FlagCombGenerator(
            flags=[
                "seqlen_sink",
                "return_max_logits",
            ],
            options={
                "seqlen_sink": [0, 4],
                "return_max_logits": [False, True],
            },
            defaults={
                "seqlen_sink": 0,
                "return_max_logits": False,
            },
            groups=[],
            strategy="heuristic",
        )
        self.flag_iterator = iter(self.flag_generator)

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def world_size(self) -> int:
        return 4

    @property
    def timeout(self) -> int:
        return 1800

    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    def _full_attn_runtime(
        self, nhq: int, nhk: int, head_dim: int, use_hier_comm: bool
    ) -> DistAttnRuntime:
        """Full attention over 4 x 128 tokens: a host stage on the local KV and
        one remote stage on the other three ranks' KV."""
        # TODO: add more attn masks for dist attn
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
            seqlen_q_shard=128,
            seqlen_k_local=128,
            seqlen_k_per_remote_stage=[128 * 3],
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
            num_heads_q=nhq,
            num_heads_kv=nhk,
            head_dim=head_dim,
        )
        dist_attn_runtime = DistAttnRuntime(
            comm_meta=comm_meta,
            calc_meta=calc_meta,
            cp_group_gc=self.nccl_groups[0],
            cp_group_gr=self.nccl_groups[1],
        )
        return dist_attn_runtime

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize("num_heads", [(8, 8), (8, 4)])
    @parameterize("head_dim", [128, 64])
    @parameterize(
        "backend",
        [
            MagiAttentionKernelBackend.FFA,
            MagiAttentionKernelBackend.SDPA,
            MagiAttentionKernelBackend.CUTEDSL,
        ],
    )
    @parameterize("use_hier_comm", [False, True])
    @parameterize("use_native_grpcoll", [False, True])
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    def test_full_attn(
        self,
        num_heads: tuple[int, int],
        head_dim: int,
        backend: MagiAttentionKernelBackend,
        use_hier_comm: bool,
        use_native_grpcoll: bool,
        dtype: torch.dtype,
    ):
        # FFA is SM90 only; CUTEDSL range kernels are SM100/SM110 only.
        major, minor = torch.cuda.get_device_capability()
        if backend == MagiAttentionKernelBackend.FFA and (major, minor) != (9, 0):
            return
        if backend == MagiAttentionKernelBackend.CUTEDSL and major not in (10, 11):
            return

        flag_comb = next(self.flag_iterator)
        seqlen_sink = flag_comb["seqlen_sink"]
        return_max_logits = flag_comb["return_max_logits"]
        use_native_grpcoll &= self.native_grpcoll_registered

        is_sdpa_backend = backend == MagiAttentionKernelBackend.SDPA

        # skip when enabling hier comm
        if use_hier_comm:
            # TODO: support hier comm with native grpcoll
            if use_native_grpcoll:
                return

        # switch the env flags
        switch_back = switch_envvars(
            envvar_name_list=[
                self.kernel_backend_envvar,
                self.hier_comm_envvar,
                self.native_grpcoll_envvar,
            ],
            enable_dict={
                self.kernel_backend_envvar: True,
                self.hier_comm_envvar: use_hier_comm,
                self.native_grpcoll_envvar: use_native_grpcoll,
            },
            enable_value_dict={self.kernel_backend_envvar: backend.value},
        )

        # prepare meta and runtime
        nhq, nhk = num_heads
        dist_attn_runtime = self._full_attn_runtime(nhq, nhk, head_dim, use_hier_comm)

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
        local_out, meta = dist_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dist_attn_runtime=dist_attn_runtime,
            sink=total_sink,
            return_max_logits=return_max_logits,
        )
        local_lse = meta.lse
        local_max_logits = meta.max_logits
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
        total_out_ref, total_meta_ref = ref_attn_func(
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
            return_max_logits=return_max_logits,
        )
        total_lse_ref = total_meta_ref.lse
        total_max_logits_ref = total_meta_ref.max_logits
        assert total_lse_ref is not None
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
            mismatch_threshold=0.1 if is_sdpa_backend else 0.08,
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
        if return_max_logits:
            assert_close(
                local_max_logits,
                total_max_logits_ref,
                atol=EPSILON,
                rtol=1e-2 if is_sdpa_backend else 1e-3,
                mismatch_threshold=0.01,
                test_case="max_logits",
            )
        assert_close(
            local_grad_q,
            local_grad_q_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.1 if is_sdpa_backend else 0.08,
            test_case="dq",
        )
        assert_close(
            local_grad_k,
            local_grad_k_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.1 if is_sdpa_backend else 0.08,
            test_case="dk",
        )
        assert_close(
            local_grad_v,
            local_grad_v_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.1 if is_sdpa_backend else 0.08,
            test_case="dv",
        )
        if total_sink is not None:
            assert_close(
                total_dsink,
                total_dsink_ref,
                atol=5e-3,
                rtol=0.1,
                mismatch_threshold=max(1 / (seqlen_sink * nhq), 5e-2),
                test_case="dsink",
            )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize("seqlen_sink", [0, 4])
    def test_cutedsl_softcap(self, seqlen_sink: int):
        """Non-zero softcap on the cutedsl backend with GQA: the host stage and
        the remote stage both compute, merge with the same capped LSE, and a
        sink is counted once (on the host stage)."""
        if torch.cuda.get_device_capability()[0] not in (10, 11):
            return
        softcap, nhq, nhk, head_dim, dtype = 30.0, 8, 4, 128, torch.bfloat16
        switch_back = switch_envvars(
            envvar_name_list=[self.kernel_backend_envvar],
            enable_dict={self.kernel_backend_envvar: True},
            enable_value_dict={
                self.kernel_backend_envvar: MagiAttentionKernelBackend.CUTEDSL.value
            },
        )
        dist_attn_runtime = self._full_attn_runtime(nhq, nhk, head_dim, False)

        # Scale q so the scores reach the cap.
        local_q = (
            4.0 * torch.randn(128, nhq, head_dim, device=self.device, dtype=dtype)
        ).requires_grad_()
        local_k, local_v = (
            torch.randn(
                128, nhk, head_dim, device=self.device, dtype=dtype, requires_grad=True
            )
            for _ in range(2)
        )
        total_sink = None
        if seqlen_sink > 0:
            total_sink = torch.randn(
                seqlen_sink, nhq, device=self.device, dtype=torch.float32
            )
            dist.all_reduce(total_sink, group=self.nccl_group)
            total_sink.requires_grad_()

        local_out, meta = dist_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dist_attn_runtime=dist_attn_runtime,
            sink=total_sink,
            softcap=softcap,
        )
        total_out = torch.cat(all_gather(local_out, group=self.nccl_group), dim=0)
        total_lse = torch.cat(all_gather(meta.lse, group=self.nccl_group), dim=0)
        grad_total_out = torch.randn_like(total_out)
        total_out.backward(grad_total_out)
        grads = [t.grad for t in (local_q, local_k, local_v)]
        for t in (local_q, local_k, local_v):
            t.grad = None
        total_dsink = None
        if total_sink is not None:
            total_dsink, total_sink.grad = total_sink.grad, None
        total_q, total_k, total_v = (
            torch.cat(all_gather(t, group=self.nccl_group), dim=0)
            for t in (local_q, local_k, local_v)
        )
        switch_back()

        total_out_ref, meta_ref = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=torch.ones(512, 512, device=self.device).bool(),
            sink=total_sink,
            softcap=softcap,
            layout="thd",
            sink_layout="sh",
            backend="torch",
            high_precision=True,
            return_lse=True,
        )
        total_out_ref.backward(grad_total_out)
        grads_ref = [t.grad for t in (local_q, local_k, local_v)]

        assert_close(
            total_out,
            total_out_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.08,
            test_case="out",
        )
        assert_close(
            total_lse,
            meta_ref.lse,
            atol=EPSILON,
            rtol=5e-3,
            mismatch_threshold=0.01,
            test_case="lse",
        )
        for name, grad, grad_ref in zip(("dq", "dk", "dv"), grads, grads_ref):
            assert_close(
                grad,
                grad_ref,
                atol=EPSILON,
                rtol=5e-2,
                mismatch_threshold=0.08,
                test_case=name,
            )
        if total_sink is not None:
            total_dsink_ref = total_sink.grad
            dist.all_reduce(total_dsink_ref, group=self.nccl_group)
            assert_close(
                total_dsink,
                total_dsink_ref,
                atol=5e-3,
                rtol=0.1,
                mismatch_threshold=max(1 / (seqlen_sink * nhq), 5e-2),
                test_case="dsink",
            )

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_cutedsl_deterministic_mode_reaches_range_kernels(self):
        """MAGI_ATTENTION_DETERMINISTIC_MODE on the cutedsl backend builds the
        deterministic range kernels for both stages; the results match the
        reference, and O, LSE and dQ, which no communication reduces here,
        repeat bit for bit."""
        if torch.cuda.get_device_capability()[0] not in (10, 11):
            return
        nhq, nhk, head_dim, dtype = 8, 4, 128, torch.bfloat16
        switch_back = switch_envvars(
            envvar_name_list=[
                self.kernel_backend_envvar,
                "MAGI_ATTENTION_DETERMINISTIC_MODE",
            ],
            enable_dict={
                self.kernel_backend_envvar: True,
                "MAGI_ATTENTION_DETERMINISTIC_MODE": True,
            },
            enable_value_dict={
                self.kernel_backend_envvar: MagiAttentionKernelBackend.CUTEDSL.value
            },
        )
        dist_attn_runtime = self._full_attn_runtime(nhq, nhk, head_dim, False)
        local_q, local_k, local_v = (
            torch.randn(128, nh, head_dim, device=self.device, dtype=dtype)
            for nh in (nhq, nhk, nhk)
        )
        grad_total_out = torch.randn(
            512, nhq, head_dim, device=self.device, dtype=dtype
        )
        dist.broadcast(grad_total_out, src=0, group=self.nccl_group)

        built: list = []

        def recording(cls):
            init = cls.__init__

            def record(obj, *args, **kwargs):
                init(obj, *args, **kwargs)
                built.append(obj)

            return mock.patch.object(cls, "__init__", record)

        def run():
            leaves = [
                t.detach().clone().requires_grad_() for t in (local_q, local_k, local_v)
            ]
            local_out, meta = dist_attn_func(
                *leaves, dist_attn_runtime=dist_attn_runtime
            )
            total_out = torch.cat(all_gather(local_out, group=self.nccl_group), dim=0)
            total_out.backward(grad_total_out)
            return local_out.detach(), meta.lse, *(t.grad for t in leaves)

        with recording(FFAFwdSm100), recording(FFABwdSm100), mock.patch.object(
            _flex_flash_attn_fwd, "compile_cache", JITCache()
        ), mock.patch.object(_flex_flash_attn_bwd, "compile_cache", JITCache()):
            first = run()
            second = run()
        switch_back()
        self.assertTrue(built and all(kernel.deterministic for kernel in built))
        for name, got, want in zip(("out", "lse", "dq"), second, first):
            self.assertTrue(torch.equal(got, want), f"{name} differs between runs")

        leaves = [
            t.detach().clone().requires_grad_() for t in (local_q, local_k, local_v)
        ]
        total_q, total_k, total_v = (
            torch.cat(all_gather(t, group=self.nccl_group), dim=0) for t in leaves
        )
        total_out_ref, _ = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=torch.ones(512, 512, device=self.device).bool(),
            layout="thd",
            backend="sdpa",
            high_precision=True,
            return_lse=True,
        )
        total_out_ref.backward(grad_total_out)
        total_out = torch.cat(all_gather(first[0], group=self.nccl_group), dim=0)
        assert_close(
            total_out,
            total_out_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.08,
            test_case="out",
        )
        for name, grad, leaf in zip(("dq", "dk", "dv"), first[2:], leaves):
            assert_close(
                grad,
                leaf.grad,
                atol=EPSILON,
                rtol=5e-2,
                mismatch_threshold=0.08,
                test_case=name,
            )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize("no_overlap", [True, False])
    @parameterize("seqlen_sink", [0, 4])
    def test_skipped_host_stage_asymmetric_kv(self, no_overlap: bool, seqlen_sink: int):
        """Skipped host stages allocate O with V's head dim and dK/dV like K/V.

        No rank attends to its local KV, so every overlap host stage is
        skipped; rank 0 attends to nothing at all, so its merged no-overlap
        argument is skipped too. Q/K head dim 192 and V head dim 128.
        """
        nhq, nhk, head_dim, head_dim_v, dtype = 8, 4, 192, 128, torch.bfloat16
        use_sdpa = env.general.kernel_backend() == MagiAttentionKernelBackend.SDPA
        mismatch_threshold = 0.1 if use_sdpa else 0.08
        seqlen, ws = 128, self.world_size
        empty_arg = AttnArg(
            q_ranges=AttnRanges.from_ranges([]),
            k_ranges=AttnRanges.from_ranges([]),
            attn_type_map=[],
            total_area=0,
        )
        remote_arg = (
            empty_arg
            if self.rank == 0
            else AttnArg(
                q_ranges=AttnRanges.from_ranges([[0, seqlen]]),
                k_ranges=AttnRanges.from_ranges([[0, seqlen * (ws - 1)]]),
                attn_type_map=[0],
                total_area=seqlen * seqlen * (ws - 1),
            )
        )
        calc_meta = CalcMeta(
            local_attn_arg=empty_arg,
            remote_attn_args_list=[remote_arg],
            no_overlap=no_overlap,
            headdim=head_dim,
            headdim_v=head_dim_v,
            seqlen_q_shard=seqlen,
            seqlen_k_local=seqlen,
            seqlen_k_per_remote_stage=[seqlen * (ws - 1)],
        )
        others = [rank for rank in range(ws) if rank != self.rank]
        comm_meta = CommMeta(
            num_remote_kv_tokens_per_stage=[seqlen * (ws - 1)],
            kv_group_collective_args_list=[
                GroupCollectiveArg(
                    input_split_size_list=[seqlen],
                    output_split_size_list=[seqlen] * (ws - 1),
                    dst_indices_list=[others],
                    src_index_list=others,
                    rank=self.rank,
                    world_size=ws,
                    group=self.nccl_group,
                )
            ],
            num_remote_qo_tokens_per_stage=[0],
            qo_group_collective_args_list=[None],  # type: ignore[list-item]
            num_heads_q=nhq,
            num_heads_kv=nhk,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
        )
        dist_attn_runtime = DistAttnRuntime(
            comm_meta=comm_meta,
            calc_meta=calc_meta,
            cp_group_gc=self.nccl_groups[0],
            cp_group_gr=self.nccl_groups[1],
        )

        def rand(nh: int, hd: int) -> torch.Tensor:
            return torch.randn(
                seqlen, nh, hd, device=self.device, dtype=dtype, requires_grad=True
            )

        local_q, local_k, local_v = (
            rand(nhq, head_dim),
            rand(nhk, head_dim),
            rand(nhk, head_dim_v),
        )
        if seqlen_sink > 0:
            total_sink = torch.randn(
                seqlen_sink, nhq, device=self.device, requires_grad=True
            )
            dist.all_reduce(total_sink.data, group=self.nccl_group)
        else:
            total_sink = None

        local_out, meta = dist_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dist_attn_runtime=dist_attn_runtime,
            sink=total_sink,
        )
        assert local_out.shape == (seqlen, nhq, head_dim_v)
        total_out = torch.cat(all_gather(local_out, group=self.nccl_group), dim=0)
        total_lse = torch.cat(all_gather(meta.lse, group=self.nccl_group), dim=0)
        grad_total_out = torch.randn_like(total_out)
        total_out.backward(grad_total_out)
        grads = [x.grad for x in (local_q, local_k, local_v)]
        for x in (local_q, local_k, local_v):
            x.grad = None
        total_dsink = None
        if total_sink is not None:
            total_dsink, total_sink.grad = total_sink.grad, None
        assert grads[2].shape == local_v.shape

        # Rank 0's rows attend only to the sinks: O = 0, LSE = lse_sink (or -inf)
        # and they contribute nothing to dK/dV/dsink, so the reference covers
        # the attending rows only.
        rank0_lse = (
            torch.logsumexp(total_sink.detach(), dim=0).expand(seqlen, nhq)
            if total_sink is not None
            else torch.full((seqlen, nhq), float("-inf"), device=self.device)
        )
        assert torch.all(total_out[:seqlen] == 0)
        torch.testing.assert_close(total_lse[:seqlen], rank0_lse)

        total_q, total_k, total_v = (
            torch.cat(all_gather(x, group=self.nccl_group), dim=0)
            for x in (local_q, local_k, local_v)
        )
        # Q block r attends to every K block except its own.
        mask = torch.ones(seqlen * ws, seqlen * ws, device=self.device).bool()
        for rank in range(ws):
            mask[
                rank * seqlen : (rank + 1) * seqlen, rank * seqlen : (rank + 1) * seqlen
            ] = False
        out_ref, meta_ref = ref_attn_func(
            q=total_q[seqlen:],
            k=total_k,
            v=total_v,
            mask=mask[seqlen:],
            sink=total_sink,
            layout="thd",
            sink_layout="sh",
            backend="torch" if total_sink is not None else "sdpa",
            high_precision=True,
            return_lse=True,
        )
        out_ref.backward(grad_total_out[seqlen:])
        grads_ref = [x.grad for x in (local_q, local_k, local_v)]
        if self.rank == 0:
            grads_ref[0] = torch.zeros_like(local_q)
        total_dsink_ref = None
        if total_sink is not None:
            total_dsink_ref = total_sink.grad
            dist.all_reduce(total_dsink_ref.data, group=self.nccl_group)

        assert_close(
            total_out[seqlen:],
            out_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=mismatch_threshold,
            test_case="out",
        )
        assert_close(
            total_lse[seqlen:],
            meta_ref.lse,
            atol=EPSILON,
            rtol=5e-3,
            mismatch_threshold=0.01,
            test_case="lse",
        )
        for name, grad, grad_ref in zip(("dq", "dk", "dv"), grads, grads_ref):
            assert grad.shape == grad_ref.shape, name
            assert_close(
                grad,
                grad_ref,
                atol=EPSILON,
                rtol=5e-2,
                mismatch_threshold=mismatch_threshold,
                test_case=name,
            )
        if total_sink is not None:
            assert_close(
                total_dsink,
                total_dsink_ref,
                atol=5e-3,
                rtol=0.1,
                mismatch_threshold=max(1 / (seqlen_sink * nhq), 5e-2),
                test_case="dsink",
            )


if __name__ == "__main__":
    run_tests()
