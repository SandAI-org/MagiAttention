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

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from enum import Enum

import torch
import torch.distributed as dist
from baselines.loongtrain import LoongTrain
from baselines.ring_attn import RingAttnAllGather, RingAttnP2P
from baselines.shard import (
    ParallelMode,
    get_loongtrain_pg,
    get_ring_pg,
    get_ulysess_pg,
    get_usp_pg,
    init_distributed,
    set_seed,
)
from baselines.ulysess import Ulysess
from baselines.usp import USP
from baselines.utils_cp import AttnBackend
from test_utils import (
    generate_attn_cu_seqlens,
    generate_attn_ranges,
    generate_test_data,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase

from magi_attention.common.enum import AttnMaskType
from magi_attention.utils import nvtx

# import sys


# import sys


class AttnImpl(Enum):
    ULYSSESS = 1
    RING_P2P = 2
    RING_ALLGATHER = 3
    USP = 4
    LOONGTRAIN = 5


class MyAttnProfile(MultiProcessTestCase):
    def __init__(self, TO_TEST=AttnImpl.RING_ALLGATHER):
        self.TO_TEST = TO_TEST

    def profile_attn(self):
        # init distributed environment
        set_seed(42)

        # -----    test ring or all-gather   ---- #
        if self.TO_TEST == AttnImpl.RING_ALLGATHER or self.TO_TEST == AttnImpl.RING_P2P:
            cp_pg_meta = {
                ParallelMode.RING: 4,
            }
            world_size = 4
            device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
            cp_group = get_ring_pg(device_shard)

        # -----    test ulysess   ---- #
        elif self.TO_TEST == AttnImpl.ULYSSESS:
            cp_pg_meta = {
                ParallelMode.ULYSESS: 4,
            }
            world_size = 4
            device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
            cp_group = get_ulysess_pg(device_shard)

        # -----    test usp   ---- #
        elif self.TO_TEST == AttnImpl.USP:
            cp_pg_meta = {
                ParallelMode.ULYSESS: 2,
                ParallelMode.RING: 2,
            }
            # ulysess [0,1] or ring [0,1]
            # cp_pg_meta = {
            #     ParallelMode.RING: 2,
            #     ParallelMode.ULYSESS: 2,
            # }
            world_size = 4
            device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
            cp_group = get_usp_pg(device_shard)
        elif self.TO_TEST == AttnImpl.LOONGTRAIN:
            cp_pg_meta = {
                ParallelMode.ULYSESS: 1,
                ParallelMode.RING: 4,
            }
            # cp_pg_meta = {
            #     ParallelMode.RING: 4,
            #     ParallelMode.ULYSESS: 1,
            # }
            world_size = 4
            # NOTE: param for loongtrain double ring-attention
            window_num = 2
            rank = int(os.environ.get("RANK", 0))
            assert world_size % window_num == 0
            device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
            cp_group = get_loongtrain_pg(device_shard, window_num, rank)

        # -----    set test param   ---- #

        device = torch.cuda.current_device()
        batch_size = 50
        total_seqlen = 4096 * 8
        h = 16
        d = 128
        dtype = torch.float16
        # NUM_SAMPLES = batch_size
        qkv_format = "thd"
        deterministic = False
        dropout = 0.0
        attn_mask_type = AttnMaskType.FULL
        # causal = attn_mask_type == AttnMaskType.CAUSAL
        attn_backend = AttnBackend.FA3

        # -----    init attn module   ---- #

        if self.TO_TEST == AttnImpl.RING_ALLGATHER:
            attn = RingAttnAllGather(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )
            cal_runtime_args = [attn_mask_type, device]
        elif self.TO_TEST == AttnImpl.RING_P2P:
            attn = RingAttnP2P(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )
            cal_runtime_args = [attn_mask_type, device]
        elif self.TO_TEST == AttnImpl.ULYSSESS:
            attn = Ulysess(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )
            cal_runtime_args = [device]
        elif self.TO_TEST == AttnImpl.USP:
            attn = USP(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )
            cal_runtime_args = [attn_mask_type, device]
        elif self.TO_TEST == AttnImpl.LOONGTRAIN:
            attn = LoongTrain(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )
            cal_runtime_args = [attn_mask_type, device]

        # -----    init test data   ---- #

        q, k, v, dout, random_cu_seqlens_list, max_seqlen = generate_test_data(
            batch_size, total_seqlen, h, d, dtype, qkv_format, device
        )
        ranges = generate_attn_ranges(random_cu_seqlens_list, total_seqlen)
        cu_seqlens, cu_seqlens_padded = generate_attn_cu_seqlens(
            random_cu_seqlens_list, max_seqlen, batch_size, qkv_format, device
        )
        # print(f"{random_cu_seqlens_list=},{cu_seqlens=},{cu_seqlens_padded=}")

        dist.broadcast(q.data, src=0)
        dist.broadcast(k.data, src=0)
        dist.broadcast(v.data, src=0)
        dist.broadcast(dout.data, src=0)
        dist.broadcast(cu_seqlens.data, src=0)
        dist.broadcast(cu_seqlens_padded.data, src=0)

        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        # print(f"{q.shape=},{q.requires_grad=}")
        # print(f"{k.shape=},{k.requires_grad=}")
        # print(f"{v.shape=},{v.requires_grad=}")

        # -----    dispatch   ---- #

        q_local = attn.dispatch(q, ranges, total_seqlen, "q")
        k_local = attn.dispatch(k, ranges, total_seqlen, "k")
        v_local = attn.dispatch(v, ranges, total_seqlen, "v")
        dout_local = attn.dispatch(dout, ranges, total_seqlen, "dout")

        attn.pre_compute_attn_runtime_meta(*cal_runtime_args)

        prof_iters, prof_start_iter, prof_end_iter = 10, 4, 7
        for iter in range(prof_iters):
            # init for nvtx
            nvtx.switch_profile(
                iter_id=iter,
                start=prof_start_iter,
                end=prof_end_iter,
                profile_ranks=[0],
            )

            dist.barrier()
            torch.cuda.synchronize()

            # -----    forward   ---- #

            out, lse = attn.apply_attn(
                q_local,
                k_local,
                v_local,
                attn_mask_type,
                dropout,
                None,
                deterministic,
            )

            # out_global = attn.undispatch(out, "q")

            # -----    backward   ---- #

            out.backward(dout_local, retain_graph=True)

            # -----    collect global grad   ---- #

        # dq_global = collect_global_grad(attn, q.grad, cu_seqlens, host_cu_seqlens, "dq")
        # dk_global = collect_global_grad(attn, k.grad, cu_seqlens, host_cu_seqlens, "dk")
        # dv_global = collect_global_grad(attn, v.grad, cu_seqlens, host_cu_seqlens, "dv")

        # max_seqlen = get_max_seqlen(host_cu_seqlens)
        # test_out_ref,test_lse_ref,test_dq_ref,test_dk_ref,test_dv_ref = test_fa3_varlen_func(
        #     q,
        #     k,
        #     v,
        #     dout,
        #     cu_seqlens,
        #     cu_seqlens,
        #     cu_seqlens_padded,
        #     cu_seqlens_padded,
        #     max_seqlen,
        #     max_seqlen,
        #     causal,
        #     qkv_format,
        #     deterministic
        # )

        # print(f"max diff out: {torch.abs(test_out_ref - out_global).max()}")
        # print(f"max diff dq: {torch.abs(test_dq_ref - dq_global).max()}")
        # print(f"max diff dk: {torch.abs(test_dk_ref - dk_global).max()}")
        # print(f"max diff dv: {torch.abs(test_dv_ref - dv_global).max()}")


if __name__ == "__main__":
    test = MyAttnProfile(AttnImpl.LOONGTRAIN)
    test.profile_attn()
