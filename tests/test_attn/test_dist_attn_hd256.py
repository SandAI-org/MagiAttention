# Focused CP tests for head_dim=256 with FA4 backend (B300 / sm100).
#
# Run with:
#   MAGI_ATTENTION_KERNEL_BACKEND=fa4 \
#     torchrun --nproc_per_node=4 tests/test_attn/test_dist_attn_hd256.py

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from magi_attention.common.ranges import AttnRanges
from magi_attention.functional.dist_attn import DistAttnRuntime, dist_attn_func
from magi_attention.meta.collection.calc_meta import AttnArg, CalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from magi_attention.testing import ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.precision import EPSILON, assert_close


class TestDistAttnHD256(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend="nccl")
            for _ in range(2)
        ]

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

    def _run_full_attn(self, nhq: int, nhk: int, dtype: torch.dtype) -> None:
        head_dim = 256
        seq = 128

        calc_meta = CalcMeta(
            local_attn_arg=AttnArg(
                q_ranges=AttnRanges.from_ranges([[0, seq]]),
                k_ranges=AttnRanges.from_ranges([[0, seq]]),
                attn_type_map=[0],
                total_area=seq * seq,
            ),
            remote_attn_args_list=[
                AttnArg(
                    q_ranges=AttnRanges.from_ranges([[0, seq]]),
                    k_ranges=AttnRanges.from_ranges([[0, seq * 3]]),
                    attn_type_map=[0],
                    total_area=seq * seq * 3,
                ),
            ],
            seqlen_q_shard=seq,
            seqlen_k_local=seq,
            seqlen_k_per_remote_stage=[seq * 3],
            headdim=head_dim,
            qhead_per_kvhead=nhq // nhk,
        )
        comm_meta = CommMeta(
            num_remote_kv_tokens_per_stage=[seq * 3],
            kv_group_collective_args_list=[
                GroupCollectiveArg(
                    input_split_size_list=[seq],
                    output_split_size_list=[seq] * (self.world_size - 1),
                    dst_indices_list=[
                        [r for r in range(self.world_size) if r != self.rank]
                    ],
                    src_index_list=[
                        r for r in range(self.world_size) if r != self.rank
                    ],
                    rank=self.rank,
                    world_size=self.world_size,
                    group=self.nccl_group,
                    device_mesh=None,
                )
            ],
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

        local_q = torch.randn(
            seq, nhq, head_dim, device=self.device, dtype=dtype, requires_grad=True
        )
        local_k = torch.randn(
            seq, nhk, head_dim, device=self.device, dtype=dtype, requires_grad=True
        )
        local_v = torch.randn(
            seq, nhk, head_dim, device=self.device, dtype=dtype, requires_grad=True
        )
        total_mask = torch.ones(
            seq * self.world_size, seq * self.world_size, device=self.device
        ).bool()

        local_out, meta = dist_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dist_attn_runtime=dist_attn_runtime,
        )
        local_lse = meta.lse
        total_out = torch.cat(all_gather(local_out, group=self.nccl_group), dim=0)
        total_lse = torch.cat(all_gather(local_lse, group=self.nccl_group), dim=0)

        grad_total_out = torch.randn_like(total_out)
        total_out.backward(grad_total_out)
        local_grad_q, local_grad_k, local_grad_v = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )
        local_q.grad = local_k.grad = local_v.grad = None

        total_q = torch.cat(all_gather(local_q, group=self.nccl_group), dim=0)
        total_k = torch.cat(all_gather(local_k, group=self.nccl_group), dim=0)
        total_v = torch.cat(all_gather(local_v, group=self.nccl_group), dim=0)

        total_out_ref, total_meta_ref = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=total_mask,
            layout="thd",
            backend="sdpa",
            high_precision=True,
            return_lse=True,
        )
        total_lse_ref = total_meta_ref.lse
        assert total_lse_ref is not None
        total_out_ref.backward(grad_total_out)
        local_grad_q_ref, local_grad_k_ref, local_grad_v_ref = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )

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
            mismatch_threshold=0.08,
            test_case="dq",
        )
        assert_close(
            local_grad_k,
            local_grad_k_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.08,
            test_case="dk",
        )
        assert_close(
            local_grad_v,
            local_grad_v_ref,
            atol=EPSILON,
            rtol=5e-2,
            mismatch_threshold=0.08,
            test_case="dv",
        )

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_full_attn_hd256_mha(self):
        self._run_full_attn(nhq=8, nhk=8, dtype=torch.bfloat16)

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_full_attn_hd256_gqa(self):
        # GQA exercises the q_stage=2 path in the FA cute backend (which needs
        # `qhead_per_kvhead` plumbed through CalcMeta to align the sparse mask).
        self._run_full_attn(nhq=8, nhk=4, dtype=torch.bfloat16)


if __name__ == "__main__":
    run_tests()
