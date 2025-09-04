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

# Copyright (c) 2025 DeepSeek. All Rights Reserved.
#
# Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import random
import time
from typing import Callable

import torch
import torch.distributed as dist

from magi_attention.comm.primitive.grpcoll import (
    group_cast_collective,
    group_reduce_collective,
)
from magi_attention.comm.primitive.grpcoll._buffer import Buffer
from magi_attention.comm.primitive.grpcoll._config import Config
from magi_attention.testing.grpcoll_utils import (
    bench,
    bench_kineto,
    calc_diff,
    get_output_split_size_list_and_src_index_list,
    get_random_dst_indices_list,
    get_random_split_size_list,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
    transfer_group_cast_meta_to_dispatch_meta,
)


def setup_dist_env(
    backend: str = "nccl",
    base_seed: int | None = None,
    seed_bias: Callable = lambda rank: 0,
) -> tuple[int, int, int, dist.ProcessGroup, int, int | None, int, int]:
    """set up distributed environment with the specified process group backend,
    NOTE: the test script using this func to set up should be executed through torchrun

    Args:
        backend (str, optional): the process group backend. Defaults to "nccl".
        base_seed (int | None, optional): the base seed. Defaults to None to not set seed.
        seed_bias (Callable, optional): the seed bias func for each rank. Defaults to lambda rank: 0, i.e., no bias.

    Returns:
        num_nodes, num_local_ranks, world_size, rank, local_rank, world_group, device, seed
    """
    num_nodes = int(os.getenv("NNODES"))  # type: ignore[arg-type]
    num_local_ranks = int(os.getenv("NPROC_PER_NODE"))  # type: ignore[arg-type]
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    seed = None
    if base_seed is not None:
        seed = base_seed + seed_bias(rank)
        torch.manual_seed(seed)
        random.seed(seed)

    return (  # type: ignore[return-value]
        num_nodes,
        num_local_ranks,
        world_size,  # num_ranks
        rank,
        local_rank,
        dist.group.WORLD,
        device,
        seed,
    )  # noqa: E231


def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    buffer: Buffer,
    group: dist.ProcessGroup,
    use_topk: bool = True,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts = args.num_experts

    assert num_experts % num_ranks == 0 and num_local_ranks == 8

    num_local_experts = num_experts // num_ranks
    if use_topk:
        assert num_local_experts == num_ranks
    else:
        assert num_local_experts == 1

    num_max_nvl_chunked_send_tokens = 8
    nvl_buffer_size = num_max_nvl_chunked_recv_tokens = (
        720 if num_ranks in (144, 160) else 512
    )

    num_max_rdma_chunked_send_tokens = 16
    rdma_buffer_size = num_max_rdma_chunked_recv_tokens = 128

    if local_rank == 0:
        print(
            (
                f"[config] {num_max_nvl_chunked_send_tokens=} | {num_max_nvl_chunked_recv_tokens=} | {nvl_buffer_size=}\n"
                f"{num_max_rdma_chunked_send_tokens=} | {num_max_rdma_chunked_recv_tokens=} | {rdma_buffer_size=}\n"
            ),
            flush=True,
        )

    # Config
    config = Config(
        num_sms,  # num_sms, default 20
        num_max_nvl_chunked_send_tokens,  # num_max_nvl_chunked_send_tokens (nvl_chunk_size), default 6
        num_max_nvl_chunked_recv_tokens,  # num_max_nvl_chunked_recv_tokens (nvl_buffer_size), default 256
        num_max_rdma_chunked_send_tokens,  # num_max_rdma_chunked_send_tokens, default 6
        num_max_rdma_chunked_recv_tokens,  # num_max_rdma_chunked_recv_tokens, default 256
    )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)

    # Random score
    # scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    # group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    # group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    # masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    # assert torch.equal(scores, masked_scores) # since we guarantee num_nodes == num_topk_groups, thus scores == masked_scores

    # topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    # topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    # topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    # rank_idx = topk_idx // (num_experts // num_ranks)
    # rank_idx.masked_fill_(topk_idx == -1, -1)
    # inplace_unique(rank_idx, num_ranks)
    # print(f"[RANK {rank}]: {rank_idx=} | {rank_idx.shape=}\n", flush=True)

    num_input_splits = 10
    input_split_size_list = get_random_split_size_list(num_tokens, num_input_splits)
    dst_indices_list = get_random_dst_indices_list(
        num_input_splits, num_ranks, min_num_dst_ranks=1
    )  # HACK: for now, empty dst rank is not supported
    (
        output_split_size_list,
        src_index_list,
    ) = get_output_split_size_list_and_src_index_list(
        input_split_size_list, dst_indices_list, group
    )

    # get ref dispatch output by group-cast
    recv_x_gc = torch.empty(
        (sum(output_split_size_list), *x.shape[1:]), dtype=torch.bfloat16, device="cuda"
    )
    work_with_pf_gc = group_cast_collective(
        input=x,
        output=recv_x_gc,
        input_split_size_list=input_split_size_list,
        dst_indices_list=dst_indices_list,
        output_split_size_list=output_split_size_list,
        src_index_list=src_index_list,
        group=group,
    )
    recv_x_gc = work_with_pf_gc.wait_post_process(recv_x_gc)
    print(f"[RANK {rank}]: {recv_x_gc.shape=} | {recv_x_gc=}\n", flush=True)

    # get ref combine output by group-reduce
    combined_x_gr = torch.zeros_like(x)
    work_with_pf_gr = group_reduce_collective(
        input=recv_x_gc,
        output=combined_x_gr,
        input_split_size_list=output_split_size_list,
        dst_index_list=src_index_list,
        output_split_size_list=input_split_size_list,
        src_indices_list=dst_indices_list,
        group=group,
    )
    combined_x_gr = work_with_pf_gr.wait_post_process(combined_x_gr)
    print(f"[RANK {rank}]: {combined_x_gr.shape=} | {combined_x_gr=}\n", flush=True)

    # transfer group-cast meta args to dispatch meta args
    (
        rank_idx,
        rdma_rank_idx,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        topk_idx,
        topk_weights,
        num_tokens_per_expert,
    ) = transfer_group_cast_meta_to_dispatch_meta(
        rank,
        num_ranks,
        num_local_experts,
        input_split_size_list,
        dst_indices_list,
        num_nodes=num_nodes,
        device="cuda",
        use_topk=use_topk,
    )
    if use_topk:
        topk_weights_pure_rand = torch.randn_like(topk_weights)
        rank_idx_ref = topk_idx // num_local_experts
        rank_idx_ref.masked_fill_(topk_idx == -1, -1)
        inplace_unique(rank_idx_ref, num_ranks)
        assert torch.equal(rank_idx, rank_idx_ref), (
            f"[RANK {rank}]: diff for rank_idx and rank_idx_ref\n{rank_idx=}\n"
            f"{rank_idx_ref=}\n"
        )
    else:
        topk_weights_pure_rand = None
    print(
        f"[RANK {rank}]: {input_split_size_list=} | {dst_indices_list=} | "
        f"{output_split_size_list=} | {src_index_list=} | {sum(output_split_size_list)=}\n",
        f"[RANK {rank}]: {topk_idx=} | {topk_weights=}\n",
        f"[RANK {rank}]: {rank_idx=} | {rdma_rank_idx=}\n",
        flush=True,
    )

    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)
    if local_rank == 0:
        print(
            f"{gbl_num_tokens_per_rank=} | {gbl_num_tokens_per_rank.shape=}\n",
            flush=True,
        )
    print(
        f"[RANK {rank}]: {num_tokens_per_rank=} | {num_tokens_per_rank.shape=}\n",
        flush=True,
    )
    print(
        f"[RANK {rank}]: {num_tokens_per_rdma_rank=} | "
        f"{num_tokens_per_rdma_rank.shape=}\n",  # type: ignore[union-attr]
        flush=True,
    )
    # RDMA dispatch counts
    num_rdma_token_sent = num_tokens_per_rdma_rank.sum().item()  # type: ignore[union-attr]

    # Expert meta
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)
    if local_rank == 0:
        print(
            f"{gbl_num_tokens_per_expert=} | {gbl_num_tokens_per_expert.shape=}\n",
            flush=True,
        )
    print(
        f"[RANK {rank}]: {num_tokens_per_expert=} | {num_tokens_per_expert.shape=}\n",
        flush=True,
    )
    if use_topk:
        # get dispatch layout from buffer
        (
            ref_num_tokens_per_rank,
            ref_num_tokens_per_rdma_rank,
            ref_num_tokens_per_expert,
            ref_is_token_in_rank,
            _,  # event_overlap,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)

        # assert close to layout ref
        assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
        assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
        assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
        assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

        # benchmark dispatch layout
        t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
        if local_rank == 0:
            print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
            print("", flush=True)
        group.barrier()
        time.sleep(1)

    # Test dispatch
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (True,):  # (False, True):
        for async_mode in (True,):  # (False, True):
            for current_x in (x,):  # (x_pure_rand, x, x_e4m3):
                for with_topk in (use_topk,):  # (False, True):
                    if local_rank == 0:
                        print(
                            "\n# ------    Test Internode Dispatch   ------ #\n",
                            flush=True,
                        )

                    # prepare dispatch args
                    if local_rank == 0:
                        print(
                            f"[testing] Running with "
                            f'{"FP8" if isinstance(current_x, tuple) else "BF16"}, '
                            f'{"with" if with_topk else "without"} top-k '
                            f"(async={async_mode}, previous={previous_mode}) ...",
                            flush=True,
                            end="",
                        )
                    dispatch_args = {
                        "x": current_x,
                        "num_tokens_per_rank": num_tokens_per_rank,
                        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
                        "is_token_in_rank": is_token_in_rank,
                        "num_tokens_per_expert": num_tokens_per_expert,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        dispatch_args.update(
                            {
                                "topk_idx": topk_idx,
                                "topk_weights": topk_weights_pure_rand
                                if current_x is x_pure_rand
                                else topk_weights,
                            }
                        )
                    if previous_mode:
                        dispatch_args.update({"previous_event": buffer.capture()})

                    # dispatch
                    # recv_x: shape=[num_recv_tokens, hidden_dim]:
                    #   the recv tokens for this rank (in rank order just like a2a output,
                    #   while the boundary is indicated by rank_prefix_matrix)
                    # recv_topk_idx: shape=[num_recv_tokens, topk]:
                    #   the local expert idx for this rank w.r.t.
                    #   each recv token's topk list (-1 means not sent to this rank)
                    # recv_topk_weights: shape=[num_recv_tokens, topk]:
                    #   the corr. weight for each recv token's topk list (if idx = -1, then weight = 0.)
                    # recv_num_tokens_per_expert_list: shape=[num_local_experts,]:
                    #   the number of tokens to recv for each local expert in this rank
                    # handle: the tuple of some meta tensors that will be passed to combine or cached dispatch
                    # handle[0] (is_token_in_rank): shape=[num_tokens, num_ranks]
                    # handle[1] (rdma_channel_prefix_matrix): shape=[num_rdma_ranks, num_channels]:
                    #   rdma_channel_prefix_matrix[r, :]: the prefix sum of send token end idxs sent by
                    #   each send-channel to rdma rank r calculated in notify_dispatch
                    # handle[2] (gbl_channel_prefix_matrix): shape=[num_ranks, num_channels]:
                    #   gbl_channel_prefix_matrix[r, :]: the prefix sum of send token end idxs sent by
                    #   each send-channel to rank r calculated in notify_dispatch
                    # handle[3] (recv_rdma_channel_prefix_matrix): shape=[num_rdma_ranks, num_channels]:
                    #   recv_rdma_channel_prefix_matrix[r, :]: the prefix sum of recv token end idxs recv by
                    #   each recv-channel from rdma rank r
                    # handle[4] (recv_rdma_rank_prefix_sum): shape=[num_rdma_ranks,]:
                    #   the prefix sum of the number of tokens to recv from each rdma rank calculated in notify_dispatch
                    # handle[5] (recv_gbl_channel_prefix_matrix): shape=[num_ranks, num_channels]:
                    #   recv_gbl_channel_prefix_matrix[r, :]: the prefix sum of recv token start idxs recv by
                    #   each recv-channel from global rank r
                    # NOTE: the start idx is a global idx with rank prefix offsets,
                    #   i.e. recv_gbl_channel_prefix_matrix[r, 0] does not start from 0 except for r == 0
                    # handle[6] (recv_gbl_rank_prefix_sum): shape=[num_ranks,]:
                    #   the prefix sum of the number of tokens to recv from each global rank,
                    #   thus recv_gbl_rank_prefix_sum[-1] == num_recv_tokens calculated in notify_dispatch
                    # handle[7] (recv_src_meta): shape=[num_recv_tokens, sizeof(internode::SourceMeta)=8]:
                    #   the source meta for each recv token,
                    #   where a SourceMeta struct object stores the src_rdma_rank
                    #   and the is_token_in_nvl_rank_bits map of this recv token
                    #   where the j-bit of is_token_in_nvl_rank_bits indicates
                    #   whether this recv token needs to be sent to the j-th local rank of this node
                    # handle[8] (send_rdma_head): shape=[num_tokens, num_rdma_ranks]: send_rdma_head[i, r]:
                    #   the offset in the corr. channel of send token i if it needs to be sent to rdma rank r
                    #   since the rdma_tail_idx starts at 0 when token_idx == token_start_idx for the corr. channel
                    #   thus the send_rdma_head[:, r] will be several cu_seqlens like:
                    #   [0, 1, ... channel0_size, 0, 1, ... channel1_size, ...]
                    #   and if all is_token_in_rank[i, r*8:(r+1)*8] == -1, then send_rdma_head[i, r] == -1 as well
                    #   (and should be ignored in the cu_seqlens above)
                    # handle[9] (send_nvl_head): shape=[num_rdma_recv_tokens, num_local_ranks]:
                    #   send_nvl_head[i, r]: the token offset of the ith recv token in the nvl forward "list" for local rank r
                    #   and if this recv token won't be sent to local rank r, then send_nvl_head[i, r] == -1 as well
                    (
                        recv_x,
                        recv_topk_idx,
                        recv_topk_weights,
                        recv_num_tokens_per_expert_list,
                        handle,
                        event,
                    ) = buffer.dispatch(**dispatch_args)

                    # wait
                    event.current_stream_wait() if async_mode else ()

                    # print
                    (
                        is_token_in_rank_handle,  # handle[0]
                        rdma_channel_prefix_matrix,  # handle[1]
                        gbl_channel_prefix_matrix,  # handle[2]
                        recv_rdma_channel_prefix_matrix,  # handle[3]
                        recv_rdma_rank_prefix_sum,  # handle[4]
                        recv_gbl_channel_prefix_matrix,  # handle[5]
                        recv_gbl_rank_prefix_sum,  # handle[6]
                        recv_src_meta,  # handle[7]
                        send_rdma_head,  # handle[8]
                        send_nvl_head,  # handle[9]
                    ) = handle
                    recv_topk_idx_shape = recv_topk_idx.shape if with_topk else None  # type: ignore[union-attr]
                    recv_topk_weights_shape = (
                        recv_topk_weights.shape if with_topk else None  # type: ignore[union-attr]
                    )
                    print(
                        (
                            f"\n[RANK {rank}]: {recv_x.shape=} | {recv_x=}\n"  # type: ignore[union-attr]
                            f"{recv_topk_idx_shape=} | {recv_topk_idx=}\n"  # type: ignore[union-attr]
                            f"{recv_topk_weights_shape=} | {recv_topk_weights=}\n"  # type: ignore[union-attr]
                            f"{len(recv_num_tokens_per_expert_list)=} | {recv_num_tokens_per_expert_list=}\n"
                            f"{is_token_in_rank_handle.shape=} | {is_token_in_rank_handle=}\n"  # handle[0]
                            f"{rdma_channel_prefix_matrix.shape=} | {rdma_channel_prefix_matrix=}\n"  # handle[1]
                            f"{gbl_channel_prefix_matrix.shape=} | {gbl_channel_prefix_matrix=}\n"  # handle[2]
                            f"{recv_rdma_channel_prefix_matrix.shape=} | {recv_rdma_channel_prefix_matrix=}\n"  # handle[3]
                            f"{recv_rdma_rank_prefix_sum.shape=} | {recv_rdma_rank_prefix_sum=}\n"  # handle[4]
                            f"{recv_gbl_channel_prefix_matrix.shape=} | {recv_gbl_channel_prefix_matrix=}\n"  # handle[5]
                            f"{recv_gbl_rank_prefix_sum.shape=} | {recv_gbl_rank_prefix_sum=}\n"  # handle[6]
                            f"{recv_src_meta.shape=} | {recv_src_meta=}\n"  # handle[7]
                            f"After dipatch: {send_rdma_head.shape=} | {send_rdma_head=}\n"  # handle[8]
                            f"After dipatch: {send_nvl_head.shape=} | {send_nvl_head=}\n\n"  # handle[9]
                        ),
                        flush=True,
                    )

                    # cast back from fp8
                    recv_x = (
                        per_token_cast_back(*recv_x)
                        if isinstance(recv_x, tuple)
                        else recv_x
                    )

                    # check
                    assert torch.equal(recv_x, recv_x_gc)
                    assert recv_gbl_rank_prefix_sum[-1].item() == recv_x.size(
                        0
                    ), f"{recv_gbl_rank_prefix_sum[-1].item()} != {recv_x.size(0)}"
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0
                    ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
                    assert (
                        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                        == recv_num_tokens_per_expert_list
                    )
                    if current_x is not x_pure_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (
                            recv_topk_idx.eq(-1)  # type: ignore[union-attr]
                            | (
                                (recv_topk_idx >= 0)  # type: ignore[operator]
                                & (recv_topk_idx < (num_experts // num_ranks))
                            )
                        ).sum().item() == recv_topk_idx.numel()  # type: ignore[union-attr]
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count  # type: ignore[union-attr]

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[  # type: ignore[index]
                                recv_topk_idx.eq(-1)  # type: ignore[union-attr]
                            ] = recv_topk_weights.amax(  # type: ignore[union-attr]
                                dim=1, keepdim=True
                            ).expand_as(  # type: ignore[union-attr]
                                recv_topk_weights
                            )[
                                recv_topk_idx.eq(-1)  # type: ignore[union-attr]
                            ]
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    if local_rank == 0:
                        print(
                            "\n# ------    Test Internode Cached Dispatch   ------ #\n",
                            flush=True,
                        )

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {
                            "x": current_x,
                            "handle": handle,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})
                        recv_cached_x, _, _, _, _, event = buffer.dispatch(
                            **dispatch_args
                        )
                        event.current_stream_wait() if async_mode else ()
                        recv_cached_x = (
                            per_token_cast_back(*recv_cached_x)
                            if isinstance(recv_cached_x, tuple)
                            else recv_cached_x
                        )
                        if current_x is not x_pure_rand:
                            check_data(recv_cached_x, recv_gbl_rank_prefix_sum)

                    if local_rank == 0:
                        print(
                            "\n# ------    Test Internode Combine   ------ #\n",
                            flush=True,
                        )

                    # prepare combine args
                    # bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    # bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    combine_args = {
                        "x": recv_x,
                        "bias": None,
                        "handle": handle,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        combine_args.update({"topk_weights": recv_topk_weights})
                    if previous_mode:
                        combine_args.update({"previous_event": buffer.capture()})

                    # combine
                    # combined_x: shape=[num_tokens, hidden_size]: combined_x[i]:
                    #   the ith token's sum-reduction result of top-k experts
                    #   NOTE: the combined_x is assumed to be already scaled by topk_weights before combining,
                    #   thus in kernel we don't have to multiply topk_weights
                    # combined_topk_weights: shape=[num_tokens, topk]:
                    #   combined_topk_weights[i]: the ith token's sum-reduction weights
                    #   NOTE: the topk_weights might not a valid probability distribution,
                    #       thus here we might need combined_topk_weights to be normalized
                    #   NOTE: the send_rdma_head will be modified in-place in internode::cached_notify
                    #       for the entries == -1 to the position of next valid token (encoded to -p-1)
                    #       since the combine kernel needs to know the channel position when iterating at this token,
                    #       even though it is not sent to the target rdma rank
                    #   NOTE: the send_nvl_head will be modified in-place in internode::cached_notify
                    #       for the entries == -1 to the position of next valid token (encoded to -p-1)
                    #       since the combine kernel needs to know the channel position when iterating at this token,
                    #       even though it is not sent to the target rdma rank
                    combined_x, combined_topk_weights, event = buffer.combine(
                        **combine_args
                    )

                    # wait
                    event.current_stream_wait() if async_mode else ()

                    # print
                    combined_topk_weights_shape = (
                        combined_topk_weights.shape if with_topk else None  # type: ignore[union-attr]
                    )
                    print(
                        (
                            f"\n[RANK {rank}]: {combined_x.shape=} | {combined_x=}\n"
                            f"{combined_topk_weights_shape=} | {combined_topk_weights=}\n"  # type: ignore[union-attr]
                            f"Before combine: {send_rdma_head.shape=} | {send_rdma_head=}\n\n"
                            f"Before combine: {send_nvl_head.shape=} | {send_nvl_head=}\n\n"
                        ),
                        flush=True,
                    )

                    # check
                    assert torch.equal(combined_x, combined_x_gr)
                    # check_x =
                    #   (combined_x.float() - bias_0.float() - bias_1.float())
                    #   / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    check_x = (combined_x.float()) / is_token_in_rank.sum(
                        dim=1
                    ).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = (
                            combined_topk_weights
                            if (current_x is x_pure_rand)
                            else (
                                combined_topk_weights
                                / is_token_in_rank.sum(dim=1).unsqueeze(1)
                            )
                        )
                        ref_topk_weights = (
                            topk_weights_pure_rand
                            if current_x is x_pure_rand
                            else topk_weights
                        )
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    if local_rank == 0:
                        print(" passed", flush=True)
    if local_rank == 0:
        print("", flush=True)

    # sync before tuning
    torch.cuda.synchronize()
    dist.barrier()

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (
            (dispatch_bf16_rdma_send_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_rdma_send_bytes
        )
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in range(4, 45, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = Config(
                    num_sms,
                    nvl_chunk_size,
                    nvl_buffer_size,
                    rdma_chunk_size,
                    rdma_buffer_size,
                )
                tune_args = {"x": current_x, "handle": handle, "config": config}
                t, notify_t = bench_kineto(
                    lambda: buffer.dispatch(**tune_args), ("dispatch", "notify")
                )
                if t < best_time:
                    best_time, best_results = t, (
                        num_sms,
                        nvl_chunk_size,
                        rdma_chunk_size,
                        notify_t,
                    )
                if local_rank == 0:
                    print(
                        f"[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, "
                        f"RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, "
                        f"notify: {notify_t * 1e6:.2f} us, "
                        f"BW: {rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), "
                        f"{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ",
                        flush=True,
                    )
        if local_rank == 0:
            print(
                f"[tuning] Best dispatch "
                f'({"FP8" if isinstance(current_x, tuple) else "BF16"}): '
                f"SMs {best_results[0]}, NVL chunk {best_results[1]}, "  # type: ignore[index]
                f"RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, "  # type: ignore[index]
                f"notify: {best_results[3] * 1e6:.2f} us, "  # type: ignore[index]
                f"BW: {rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), "
                f"{nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)",
                flush=True,
            )
            print("", flush=True)

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor(
                [best_results[0], best_results[1], best_results[2]],  # type: ignore[index]
                dtype=torch.int32,
                device="cuda",
            )
            all_best_fp8_results_list = [
                torch.zeros_like(best_dispatch_results)
                for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(
                all_best_fp8_results_list, best_dispatch_results, group=group
            )
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = Config(
        best_dispatch_results[0],  # type: ignore[index]
        best_dispatch_results[1],  # type: ignore[index]
        nvl_buffer_size,
        best_dispatch_results[2],  # type: ignore[index]
        rdma_buffer_size,
    )

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = Config(
                num_sms,
                nvl_chunk_size,
                nvl_buffer_size,
                rdma_chunk_size,
                rdma_buffer_size,
            )
            tune_args = {"x": recv_x, "handle": handle, "config": config}
            t, notify_t = bench_kineto(
                lambda: buffer.combine(**tune_args), ("combine", "notify")
            )
            if local_rank == 0:
                print(
                    f"[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, "
                    f"RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, "
                    f"notify: {notify_t * 1e6:.2f} us, "
                    f"BW: {combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), "
                    f"{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ",
                    flush=True,
                )
                if t < best_time:
                    best_time, best_results = t, (
                        num_sms,
                        nvl_chunk_size,
                        rdma_chunk_size,
                        notify_t,
                    )

    if local_rank == 0:
        print(
            f"[tuning] Best combine: SMs {best_results[0]}, "  # type: ignore[index]
            f"NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, "  # type: ignore[index]
            f"transmit: {best_time * 1e6:.2f} us, "
            f"notify: {best_results[3] * 1e6:.2f} us, "  # type: ignore[index]
            f"BW: {combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), "
            f"{combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)",
            flush=True,
        )
        print("", flush=True)


def test_loop(args: argparse.Namespace):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    # init dist
    (
        num_nodes,
        num_local_ranks,
        num_ranks,
        rank,
        local_rank,
        group,
        device,
        seed,
    ) = setup_dist_env(base_seed=0, seed_bias=lambda rank: rank)

    # if args.test_ll_compatibility:
    #     ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
    ll_num_experts = 256

    num_sms = 24
    num_qps_per_rank = max(
        num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0
    )
    args.num_topk_groups = num_topk_groups = num_nodes

    num_nvl_bytes = int(2e9)
    num_rdma_bytes = int(1e9)

    # reset for group-collective
    use_topk = False  # NOTE: disable topk to improve bandwidth by saving unused experts
    num_topk = num_ranks
    num_experts = num_ranks * num_ranks if use_topk else num_ranks
    args.num_topk = num_topk
    args.num_experts = num_experts

    if local_rank == 0:
        print(
            (
                f"[config] {num_nvl_bytes=} ({num_nvl_bytes / 1e9:.2f} GB) | "
                f"{num_rdma_bytes=} ({num_rdma_bytes / 1e9:.2f} GB) | "
                f"{num_nodes=} (num_rdma_ranks) | {num_ranks=} | "
                f"{num_local_ranks=} | {group.size()=} | "  # type: ignore[union-attr]
                f" {num_sms=} | {num_qps_per_rank=} | "
                f"{num_tokens=} | {hidden=} | {num_topk=} | "
                f"{num_experts=} | {num_topk_groups=}\n\n\n"
            ),
            flush=True,
        )

    buffer = Buffer(
        group,
        num_nvl_bytes,
        num_rdma_bytes,
        low_latency_mode=args.test_ll_compatibility,
        num_qps_per_rank=num_qps_per_rank,
        explicitly_destroy=True,
    )
    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    for i in (num_sms,):
        test_main(
            args,
            i,
            local_rank,
            num_local_ranks,
            num_ranks,
            num_nodes,
            rank,
            buffer,
            group,
            use_topk=use_topk,
        )
        if local_rank == 0:
            print("", flush=True)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test internode EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk-groups",
        type=int,
        default=None,
        help="Number of top-k groups (default: `min(num_nodes, 4)`)",
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256"
    )
    parser.add_argument(
        "--test-ll-compatibility",
        action="store_true",
        help="whether to test compatibility with low-latency kernels",
    )
    args = parser.parse_args()

    args.test_ll_compatibility = False

    num_processes = args.num_processes

    # torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

    # launch using torchrun
    test_loop(args)
