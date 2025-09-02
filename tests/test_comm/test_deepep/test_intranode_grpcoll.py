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
import random
import time

import deep_ep
import torch
import torch.distributed as dist

from magi_attention.comm.primitive import group_cast_collective, group_reduce_collective

from .grpcoll_utils import (
    bench,
    calc_diff,
    get_output_split_size_list_and_src_index_list,
    get_random_dst_indices_list,
    get_random_split_size_list,
    init_dist,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
    transfer_group_cast_meta_to_dispatch_meta,
)


# noinspection PyShadowingNames
def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_channels = (
        num_sms // 2
    )  # one channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving

    # Re-Settings for group-collective
    use_topk = False  # NOTE: disable topk to improve bandwidth by saving unused experts
    num_topk = num_ranks  # we can assume num_topk == num_ranks
    if use_topk:
        # if using topk, we can assume num_local_experts == num_ranks,
        # thus when we only need to send certain token to one rank,
        # it can be equivalent to send to several "local experts" in that rank
        num_experts = num_ranks * num_ranks
    else:
        # if not, we can further assume num_local_experts == 1
        # thus sending one token to one rank is equivalent to sending to the only one "local expert" in that rank
        num_experts = num_ranks

    num_max_nvl_chunked_send_tokens = 8
    nvl_buffer_size = (
        num_max_nvl_chunked_recv_tokens
    ) = 256  # nvl_buffer_size, since the buffer is stored at the receiver side

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    if use_topk:
        assert num_local_experts == num_ranks
    else:
        assert num_local_experts == 1
    if local_rank == 0:
        print(
            (
                f"[config] {num_sms=} | {num_channels=} | "
                f"{num_experts=} | {num_tokens=} | {hidden=} | "
                f"{num_topk=} | {num_local_experts=}\n"
                f"{nvl_buffer_size} | {num_max_nvl_chunked_send_tokens=} | {num_max_nvl_chunked_recv_tokens=}\n"
            ),
            flush=True,
        )

    # Config
    config = deep_ep.Config(
        num_sms,  # num_sms, default 20
        num_max_nvl_chunked_send_tokens,  # num_max_nvl_chunked_send_tokens (nvl_chunk_size), default 6
        num_max_nvl_chunked_recv_tokens,  # num_max_nvl_chunked_recv_tokens (nvl_buffer_size), default 256
        # num_max_rdma_chunked_send_tokens, default 6
        # num_max_rdma_chunked_recv_tokens, default 256
    )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None

    # Random score (transfered from group-cast meta args)
    # scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    # topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    # topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    # topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    # rank_idx = topk_idx // num_local_experts
    # rank_idx.masked_fill_(topk_idx == -1, -1)
    # inplace_unique(rank_idx, num_ranks)
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
    rank_idx, topk_idx, topk_weights = transfer_group_cast_meta_to_dispatch_meta(
        rank,
        num_ranks,
        num_local_experts,
        input_split_size_list,
        dst_indices_list,
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
        f"{output_split_size_list=} | {src_index_list=} | "
        f"{sum(output_split_size_list)=}\n",
        flush=True,
    )
    print(f"[RANK {rank}]: {topk_idx=} | {topk_weights=}\n", flush=True)
    print(f"[RANK {rank}]: {rank_idx=}\n", flush=True)

    # Rank layout meta
    # num_tokens_per_rank[r]: the number of tokens sent to rank r by this rank
    # gbl_num_tokens_per_rank[r]: the number of tokens sent to rank r by all ranks
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (
            rank_idx == i
        ).sum()  # the number of tokens sent to rank i
        token_sel = (rank_idx == i).max(dim=-1)[
            0
        ]  # token_sel[j]: whether token j is sent to rank i
        count = token_sel.sum().item()  # the number of tokens sent to rank i
        # after this step, all True (tokens[:count])'s token idx will move to the left
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        # after this step, all True (tokens[:count])'s token idx will sort in ascending order
        tokens[:count] = torch.sort(tokens[:count])[0]
        # after this step, token_idx_in_rank[r][j]: for rank r, the order idx of jth token to send (-1 means not sent)
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="cuda"
        )
    # after this step, token_idx_in_rank[j][r]: for jth token, its order idx to send to rank r (-1 means not sent)
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    # after this step, is_token_in_rank[j][r]: whether jth token is sent to rank r
    is_token_in_rank = token_idx_in_rank >= 0
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

    # Expert meta
    # num_tokens_per_expert[e]: the number of tokens sent to expert e by this rank
    # gbl_num_tokens_per_expert[e]: the number of tokens sent to expert e by all ranks
    if use_topk:
        num_tokens_per_expert = torch.zeros(
            (num_experts,), dtype=torch.int, device="cuda"
        )
        for i in range(num_experts):
            num_tokens_per_expert[i] = (topk_idx == i).sum()
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

        # get dispatch layout from buffer
        (
            ref_num_tokens_per_rank,
            ref_num_tokens_per_rdma_rank,
            ref_num_tokens_per_expert,
            ref_is_token_in_rank,
            event_overlap,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)

        # assert close to layout ref
        assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
        assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
        assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

        # benchmark dispatch layout
        t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
        if local_rank == 0:
            print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
            print("", flush=True)
        group.barrier()
        time.sleep(1)
    else:
        num_tokens_per_expert = num_tokens_per_rank
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

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (True,):  # (False, True):
        for async_mode in (True,):  # (False, True):
            for current_x in filter(
                lambda elem: elem is not None, (x,)
            ):  # (x_pure_rand, x, x_e4m3)):
                for with_topk in (use_topk,):  # (False, True):
                    if local_rank == 0:
                        print(
                            "\n# ------    Test Intranode Dispatch   ------ #\n",
                            flush=True,
                        )

                    # prepare dispatch args
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, '
                            f'{"with" if with_topk else "without"} top-k '
                            f"(async={async_mode}, previous={previous_mode}) ...",
                            flush=True,
                            end="",
                        )
                    dispatch_args = {
                        "x": current_x,
                        "num_tokens_per_rank": num_tokens_per_rank,
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
                    #   the corr. weight for each recv token's topk list
                    #   (if idx = -1, then weight = 0.)
                    # recv_num_tokens_per_expert_list: shape=[num_local_experts,]:
                    #   the number of tokens to recv for each local expert in this rank
                    # handle: the tuple of some meta tensors that will be passed to combine or cached dispatch
                    # handle[0] (rank_prefix_matrix): shape=[num_ranks, num_ranks]:
                    #   rank_prefix_matrix[:, r]: the prefix sum of number of tokens (i.e. end idxs)
                    #   sent by each rank to rank r calculated in notify_dispatch
                    # handle[1] (channel_prefix_matrix): shape=[num_ranks, num_channels]:
                    #   channel_prefix_matrix[r, :]: the prefix sum of send token end idxs
                    #   sent by each send-channel to rank r calculated in notify_dispatch
                    # handle[2] (recv_channel_prefix_matrix): shape=[num_ranks, num_channels]:
                    #   recv_channel_prefix_matrix[r, :]: the prefix sum of recv token start idxs
                    #   recv by each recv-channel from rank r
                    # handle[3] (recv_src_idx): shape=[num_recv_tokens,]:
                    #   the original token idx in the sender's buffer of each recv token
                    #   so this is used in combine stage to indicate the original token position
                    #   that each recv token should be reduced to
                    # handle[4] (is_token_in_rank): shape=[num_tokens, num_ranks]
                    # handle[5] (send_head): shape=[num_tokens, num_ranks]:
                    #   send_head[i, r]: the offset in the corr. channel of send token i
                    #   if it needs to be sent to rank r
                    #   since the cached_channel_tail_idx starts at 0
                    #   when token_idx == token_start_idx for the corr. channel
                    #   thus the send_head[:, r] will be several cu_seqlens like:
                    #       [0, 1, ... channel0_size, 0, 1, ... channel1_size, ...]
                    #   and if is_token_in_rank[i, r] == -1, then send_head[i, r] == -1
                    #   as well (and should be ignored in the cu_seqlens above)
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
                        rank_prefix_matrix,  # handle[0]
                        channel_prefix_matrix,  # handle[1]
                        recv_channel_prefix_matrix,  # handle[2]
                        recv_src_idx,  # handle[3]
                        is_token_in_rank_handle,  # handle[4]
                        send_head,  # handle[5]
                    ) = handle
                    if with_topk:
                        print(
                            (
                                f"\n[RANK {rank}]: {recv_x.shape=} | {recv_x=}\n"
                                f"{recv_topk_idx.shape=} | {recv_topk_idx=}\n"
                                f"{recv_topk_weights.shape=} | {recv_topk_weights=}\n"
                                f"{len(recv_num_tokens_per_expert_list)=} | {recv_num_tokens_per_expert_list=}\n"
                                f"{rank_prefix_matrix.shape=} | {rank_prefix_matrix=}\n"  # handle[0]
                                f"{channel_prefix_matrix.shape=} | {channel_prefix_matrix=}\n"  # handle[1]
                                f"{recv_channel_prefix_matrix.shape=} | {recv_channel_prefix_matrix=}\n"  # handle[2]
                                f"{recv_src_idx.shape=} | {recv_src_idx=}\n"  # handle[3]
                                f"{is_token_in_rank_handle.shape=} | {is_token_in_rank_handle=}\n"  # handle[4]
                                f"After dipatch: {send_head.shape=} | {send_head=}\n\n"  # handle[5]
                            ),
                            flush=True,
                        )
                    else:
                        print(
                            (
                                f"\n[RANK {rank}]: {recv_x.shape=} | {recv_x=}\n"
                                f"{recv_topk_idx=}\n"
                                f"{recv_topk_weights=}\n"
                                f"{len(recv_num_tokens_per_expert_list)=} | {recv_num_tokens_per_expert_list=}\n"
                                f"{rank_prefix_matrix.shape=} | {rank_prefix_matrix=}\n"  # handle[0]
                                f"{channel_prefix_matrix.shape=} | {channel_prefix_matrix=}\n"  # handle[1]
                                f"{recv_channel_prefix_matrix.shape=} | {recv_channel_prefix_matrix=}\n"  # handle[2]
                                f"{recv_src_idx.shape=} | {recv_src_idx=}\n"  # handle[3]
                                f"{is_token_in_rank_handle.shape=} | {is_token_in_rank_handle=}\n"  # handle[4]
                                f"After dipatch: {send_head.shape=} | {send_head=}\n\n"  # handle[5]
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
                    assert torch.equal(is_token_in_rank_handle, is_token_in_rank)
                    assert torch.equal(
                        channel_prefix_matrix[:, -1], num_tokens_per_rank
                    )
                    assert torch.equal(
                        recv_channel_prefix_matrix[rank, 1:],
                        channel_prefix_matrix[rank, :-1],
                    )
                    assert torch.all(recv_channel_prefix_matrix[:, 0] == 0)
                    assert torch.all(send_head[is_token_in_rank_handle == -1] == -1)
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0
                    ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
                    assert (
                        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                        == recv_num_tokens_per_expert_list
                    )
                    if current_x is not x_pure_rand:
                        check_data(recv_x, rank_prefix_matrix)
                    recv_topk_weights_clone = None
                    if with_topk:
                        # Check `topk_idx`
                        assert (
                            recv_topk_idx.eq(-1)
                            | (
                                (recv_topk_idx >= 0)
                                & (recv_topk_idx < (num_experts // num_ranks))
                            )
                        ).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()
                        if current_x is not x_pure_rand:
                            recv_topk_weights[
                                recv_topk_idx.eq(-1)
                            ] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(
                                recv_topk_weights
                            )[
                                recv_topk_idx.eq(-1)
                            ]
                            check_data(recv_topk_weights, rank_prefix_matrix)

                    if local_rank == 0:
                        print(
                            "\n# ------    Test Intranode Dispatch with worst tokens   ------ #\n",
                            flush=True,
                        )

                    # Test `num_worst_tokens != 0`
                    if with_topk:
                        num_worst_tokens = num_tokens * num_ranks
                        dispatch_args.update({"num_worst_tokens": num_worst_tokens})

                        # dispatch with `num_worst_tokens != 0`
                        # then all seqlen dim will be equal to num_worst_tokens
                        # where the excess tokens will be empty
                        (
                            recv_worst_x,
                            recv_worst_topk_idx,
                            recv_worst_topk_weights,
                            empty_list,
                            _,
                            event,
                        ) = buffer.dispatch(**dispatch_args)

                        # wait
                        event.current_stream_wait() if async_mode else ()

                        # print
                        print(
                            (
                                f"\n[RANK {rank}]: {recv_worst_x.shape=}\n"
                                f"{recv_worst_topk_idx.shape=} | {recv_worst_topk_idx[0]=}\n"
                                f"{recv_worst_topk_weights.shape=} | {recv_worst_topk_weights[0]=}\n\n"
                            ),
                            flush=True,
                        )

                        # cast back from fp8
                        recv_worst_x = (
                            per_token_cast_back(*recv_worst_x)
                            if isinstance(recv_worst_x, tuple)
                            else recv_worst_x
                        )

                        # check
                        assert len(empty_list) == 0
                        assert num_worst_tokens == recv_worst_x.size(0)
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)
                        assert torch.equal(recv_x, recv_worst_x[: recv_x.size(0)])
                        assert torch.equal(
                            recv_topk_idx, recv_worst_topk_idx[: recv_x.size(0)]
                        )
                        assert torch.equal(
                            recv_topk_weights_clone,
                            recv_worst_topk_weights[: recv_x.size(0)],
                        )
                        assert torch.all(
                            recv_worst_topk_idx[recv_x.size(0) :] == -1
                        ).item()

                    if local_rank == 0:
                        print(
                            "\n# ------    Test Intranode Cached Dispatch   ------ #\n",
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
                        recv_cache_x, _, _, _, _, event = buffer.dispatch(
                            **dispatch_args
                        )
                        event.current_stream_wait() if async_mode else ()
                        recv_cache_x = (
                            per_token_cast_back(*recv_cache_x)
                            if isinstance(recv_cache_x, tuple)
                            else recv_cache_x
                        )
                        if current_x is not x_pure_rand:
                            check_data(recv_cache_x, rank_prefix_matrix)

                    if local_rank == 0:
                        print(
                            "\n# ------    Test Intranode Combine   ------ #\n",
                            flush=True,
                        )

                    # prepare combine args
                    send_head_copy = send_head.clone()
                    combine_args = {
                        "x": recv_x,
                        "handle": handle,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        combine_args.update({"topk_weights": recv_topk_weights})
                    if previous_mode:
                        combine_args.update({"previous_event": buffer.capture()})

                    # combine
                    # combined_x: shape=[num_tokens, hidden_size]:
                    #   combined_x[i]: the ith token's sum-reduction result of top-k experts
                    #   NOTE: the combined_x is assumed to be already scaled by topk_weights before combining,
                    #   thus in kernel we don't have to multiply topk_weights
                    # combined_topk_weights: shape=[num_tokens, topk]:
                    #   combined_topk_weights[i]: the ith token's sum-reduction weights
                    #   NOTE: the topk_weights might not a valid probability distribution,
                    #   thus here we might need combined_topk_weights to be normalized
                    #   NOTE: the send_head will be modified in-place in intranode::cached_notify_combine
                    #   for the entries == -1 to the position p of next valid token (encoded to -p-1)
                    #   since the combine kernel needs to know the channel position when iterating at this token,
                    #   even though it is not sent to the target rank
                    combined_x, combined_topk_weights, event = buffer.combine(
                        **combine_args
                    )

                    # wait
                    event.current_stream_wait() if async_mode else ()

                    # print
                    if with_topk:
                        print(
                            (
                                f"\n[RANK {rank}]: {combined_x.shape=} | {combined_x=}\n"
                                f"{combined_topk_weights.shape=} | {combined_topk_weights=}\n"
                                f"Before combine: {send_head.shape=} | {send_head=}\n\n"
                            ),
                            flush=True,
                        )
                    else:
                        print(
                            (
                                f"\n[RANK {rank}]: {combined_x.shape=} | {combined_x=}\n"
                                f"{combined_topk_weights=}\n"
                                f"Before combine: {send_head.shape=} | {send_head=}\n\n"
                            ),
                            flush=True,
                        )

                    # check
                    assert torch.equal(combined_x, combined_x_gr)
                    assert torch.equal(
                        send_head[send_head_copy != -1],
                        send_head_copy[send_head_copy != -1],
                    )  # cached_notify_combine will modify send_head in-place for any entry == -1
                    check_x = combined_x.float() / is_token_in_rank.sum(
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
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                    if local_rank == 0:
                        print(" passed", flush=True)
    if local_rank == 0:
        print("", flush=True)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0,):
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)
                config = deep_ep.Buffer.get_dispatch_config(num_ranks)
            tune_args = {"x": current_x, "handle": handle, "config": config}
            t = bench(lambda: buffer.dispatch(**tune_args))[0]
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                    f"{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                    flush=True,
                )
        if local_rank == 0:
            print(
                f"[tuning] Best dispatch "
                f'({"FP8" if isinstance(current_x, tuple) else "BF16"}): '
                f"SMs {best_results[0]}, NVL chunk {best_results[1]}, "  # type: ignore[index]
                f"{nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), "
                f"t: {best_time * 1e6:.2f} us",
                flush=True,
            )
            print("", flush=True)

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor(
                [best_results[0], best_results[1]],  # type: ignore[index]
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
    dispatch_config = deep_ep.Config(
        best_dispatch_results[0],  # type: ignore[index]
        best_dispatch_results[1],  # type: ignore[index]
        nvl_buffer_size,
    )

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0,):
        if nvl_chunk_size > 0:
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        else:
            # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)
            config = deep_ep.Buffer.get_combine_config(num_ranks)
        tune_args = {"x": recv_x, "handle": handle, "config": config}
        t = bench(lambda: buffer.combine(**tune_args))[0]
        if local_rank == 0:
            print(
                f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                f"{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                flush=True,
            )
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

    if local_rank == 0:
        print(
            f"[tuning] Best combine: SMs {best_results[0]}, "  # type: ignore[index]
            f"NVL chunk {best_results[1]}: "  # type: ignore[index]
            f"{combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), "
            f"t: {best_time * 1e6:.2f} us",
            flush=True,
        )
        print("", flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # rank: global rank in default group
    # num_ranks: number of ranks in default group
    # group: the default world group

    # init dist
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    test_ll_compatibility, num_rdma_bytes = False, 0
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            ll_num_tokens, ll_hidden, num_ranks, ll_num_experts
        )
        if local_rank == 0:
            print(
                f"Low latency mode: {ll_num_tokens=} | {ll_hidden=} | {ll_num_experts=} | {ll_num_topk=}",
                flush=True,
            )

    # there's two assertion about this num_nvl_bytes:
    # 1. num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes
    # 2. num_ranks * num_ranks * sizeof(int) +  // Size prefix matrix
    #    num_channels * num_ranks * sizeof(int) + // Channel start offset
    #    num_channels * num_ranks * sizeof(int) + // Channel end offset
    #    num_channels * num_ranks * sizeof(int) * 2 + // Queue head and tail
    #    num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() + // Data buffer
    #    num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) + // Source index buffer
    #    num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(int64_t) + // Top-k index buffer
    #    num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) + // Top-k weight buffer
    #    num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales // FP8 scale buffer
    #    <= num_nvl_bytes
    num_nvl_bytes = int(2e9)
    num_qps_per_rank = ll_num_experts // num_ranks if test_ll_compatibility else 1

    if local_rank == 0:
        print(
            (
                f"[config]: {num_ranks=} | {num_local_ranks=} | {group.size()=} | "
                f"{num_nvl_bytes=} ({num_nvl_bytes / 1e9:.2f} GB) | {num_rdma_bytes=} | {num_qps_per_rank=}\n"
            ),
            flush=True,
        )

    buffer = deep_ep.Buffer(
        group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=test_ll_compatibility,
        num_qps_per_rank=num_qps_per_rank,
        explicitly_destroy=True,
    )
    torch.manual_seed(rank)
    random.seed(rank)

    for num_sms in (24,):  # range(16, 33, 4): # [16, 20, 24, 28, 32]
        if local_rank == 0:
            print(
                f"\n\n============================Testing with {num_sms=}============================\n\n",
                flush=True,
            )
        test_main(args, num_sms, local_rank, num_ranks, rank, buffer, group)
        if local_rank == 0:
            print("", flush=True)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test intranode EP kernels")
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
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
