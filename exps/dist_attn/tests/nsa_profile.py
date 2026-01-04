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


# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist

from exps.dist_attn.baselines.shard import (
    ParallelMode,
    get_usp_pg,
    init_distributed,
    set_seed,
)
from exps.dist_attn.baselines.usp_nsa import USPAllGatherNSA
from magi_attention.common.ranges import AttnRanges
from magi_attention.utils import nvtx


class NSAAttnProfile:
    def __init__(self):
        pass

    def profile_attn(self):
        set_seed(42)

        cp_pg_meta = {
            ParallelMode.ULYSESS: 2,
            ParallelMode.RING: 2,
        }
        world_size = 4
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_usp_pg(device_shard)

        # -----    set test param   ---- #

        device = torch.cuda.current_device()
        total_seqlen = 4096 * 8
        heads_q = 16
        heads_k = 16
        hidden_dim = 128
        l_cmp = 64
        l_slc = 64
        slc_topk = 16
        stride = 64
        block_size_q = 32
        window_size_left = 32
        window_size_right = 32
        dtype = torch.float16
        deterministic = False

        q_ranges = AttnRanges.from_ranges(
            [
                [0, 4096],
                [4096, 8192],
                [8192, 12288],
                [12288, 16384],
                [16384, 20480],
                [20480, 24576],
                [24576, 28672],
                [28672, 32768],
            ]
        )
        k_ranges = AttnRanges.from_ranges(
            [
                [0, 4096],
                [4096, 8192],
                [8192, 12288],
                [12288, 16384],
                [16384, 20480],
                [20480, 24576],
                [24576, 28672],
                [28672, 32768],
            ]
        )

        # -----    init test data   ---- #

        q = torch.randn(
            (total_seqlen, heads_q, hidden_dim),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        k = torch.randn(
            (total_seqlen, heads_k, hidden_dim),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        v = torch.randn(
            (total_seqlen, heads_k, hidden_dim),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        dout = torch.randn(
            (total_seqlen, heads_q, hidden_dim), device=device, dtype=dtype
        )

        # dist.broadcast(q.data, src=0)
        # dist.broadcast(k.data, src=0)
        # dist.broadcast(v.data, src=0)
        # dist.broadcast(dout.data, src=0)

        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        # -----    init attn module   ---- #

        set_seed(42)
        attn = USPAllGatherNSA(
            cp_group,
            l_cmp,
            l_slc,
            slc_topk,
            stride,
            block_size_q,
            hidden_dim,
            dtype,
            device,
        )

        # -----    dispatch   ---- #

        q_local = attn.dispatch(q, q_ranges)
        k_local = attn.dispatch(k, k_ranges)
        v_local = attn.dispatch(v, k_ranges)
        dout_local = attn.dispatch(dout, q_ranges)

        # -----    pre compute   ---- #

        attn.pre_compute_attn_runtime_meta(
            q_ranges, k_ranges, window_size_left, window_size_right, device
        )

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
                None,  # type: ignore[arg-type]
                deterministic,
            )

            # -----    backward   ---- #

            out.backward(dout_local)


if __name__ == "__main__":
    # -----    nsa profile   ---- #

    test = NSAAttnProfile()
    test.profile_attn()
