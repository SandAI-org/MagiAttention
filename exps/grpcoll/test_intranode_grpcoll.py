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
import time

import torch
import torch.distributed as dist

from magi_attention.comm.primitive.grpcoll import group_cast, group_reduce
from magi_attention.comm.primitive.grpcoll._buffer import GrpCollBuffer
from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_mgr
from magi_attention.comm.primitive.grpcoll.utils import (
    get_a2av_perm_idxs_from_group_cast_meta,
    get_dispatch_layout_from_group_cast_meta,
    transfer_splits_and_dst_idxs_to_topk_idx,
    unpermute_tensor,
)

# isort: split
from grpcoll_utils import (
    bench,
    calc_diff,
    get_output_split_size_list_and_src_index_list,
    get_random_dst_indices_list,
    get_random_split_size_list,
    init_dist,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
    sim_gemm,
    transfer_group_cast_meta_to_dispatch_meta,
)


def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: GrpCollBuffer,
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
    distinct_token = True
    random_permute_output = True
    sim_gemm_weight = 2.0
    min_num_dst_ranks = 0
    allow_empty_init_out_buf = (
        min_num_dst_ranks > 0
    )  # if every token has at least one dst, we can empty-init
    pass_out_buffer = True
    acc_reduce_out_buffer = True
    acc_reduce_constant = rank
    if acc_reduce_out_buffer:
        assert pass_out_buffer, "acc_reduce_out_buffer requires pass_out_buffer"
    use_a2av_perm_idxs = "inside"  # choose from "no", "outside", "inside"
    assert use_a2av_perm_idxs in ("no", "outside", "inside")

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
    config = GrpCollConfig(
        num_sms=num_sms,  # num_sms, default 20
        nvl_chunk_size=num_max_nvl_chunked_send_tokens,  # num_max_nvl_chunked_send_tokens (nvl_chunk_size), default 6
        nvl_buffer_size=num_max_nvl_chunked_recv_tokens,  # num_max_nvl_chunked_recv_tokens (nvl_buffer_size), default 256
        # num_max_rdma_chunked_send_tokens, default 6
        # num_max_rdma_chunked_recv_tokens, default 256
    )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    if distinct_token:
        x *= torch.arange(
            rank * num_tokens,
            (rank + 1) * num_tokens,
            dtype=torch.bfloat16,
            device="cuda",
        ).view(-1, 1) / (num_ranks * num_tokens)
        print(f"[RANK {rank}]: distinct_input: {x=}\n", flush=True)
    else:
        x *= rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x) if GrpCollBuffer.is_sm90_compiled() else None
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
        num_splits=num_input_splits,
        num_ranks=num_ranks,
        min_num_dst_ranks=min_num_dst_ranks,
        max_num_dst_ranks=None,
    )
    (
        output_split_size_list,
        src_index_list,
    ) = get_output_split_size_list_and_src_index_list(
        input_split_size_list=input_split_size_list,
        dst_indices_list=dst_indices_list,
        group=group,
        random_permute=random_permute_output,
    )

    # get ref dispatch output by group-cast
    recv_x_gc = torch.empty(
        (sum(output_split_size_list), *x.shape[1:]), dtype=torch.bfloat16, device="cuda"
    )
    recv_x_gc_buf = recv_x_gc.clone() if pass_out_buffer else None
    work_with_pf_gc = group_cast(
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
    x_gr = sim_gemm(recv_x_gc, w=sim_gemm_weight)
    combined_x_gr = torch.zeros_like(x)
    if acc_reduce_out_buffer:
        combined_x_gr += acc_reduce_constant
    combined_x_gr_buf = combined_x_gr.clone() if pass_out_buffer else None
    work_with_pf_gr = group_reduce(
        input=x_gr,
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
        _,  # rdma_rank_idx
        num_tokens_per_rank,
        _,  # num_tokens_per_rdma_rank
        is_token_in_rank,
        topk_idx,
        topk_weights,
        num_tokens_per_expert,
        range_gather_post_dispatch_kwargs,
        range_gather_pre_combine_kwargs,
    ) = transfer_group_cast_meta_to_dispatch_meta(
        rank=rank,
        num_ranks=num_ranks,
        num_nodes=1,
        num_local_experts=num_local_experts,
        input_split_size_list=input_split_size_list,
        dst_indices_list=dst_indices_list,
        output_split_size_list=output_split_size_list,
        src_index_list=src_index_list,
        use_topk=use_topk,
        use_a2a_order_output=not random_permute_output,
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
        f"{topk_idx=} | {topk_weights=}\n",
        f"{rank_idx=}\n",
        flush=True,
    )

    # get perm/unperm idxs to/from a2av through group-cast meta args
    # which is used to replace the post-dispatch range_gather and pre-combine range_gather
    (
        unperm_from_a2av_idx,
        perm_to_a2av_idx,
    ) = get_a2av_perm_idxs_from_group_cast_meta(
        output_split_size_list=output_split_size_list,
        src_index_list=src_index_list,
        world_size=num_ranks,
    )
    print(
        f"[RANK {rank}]: {perm_to_a2av_idx=}\n" f"{unperm_from_a2av_idx=}\n",
        flush=True,
    )
    if not random_permute_output:
        arange_idx = torch.arange(
            sum(output_split_size_list),
            dtype=torch.int64,
            device="cuda",
        )
        assert torch.equal(unperm_from_a2av_idx, arange_idx)
        assert torch.equal(perm_to_a2av_idx, arange_idx)

    # Rank layout meta
    # num_tokens_per_rank[r]: the number of tokens sent to rank r by this rank
    # gbl_num_tokens_per_rank[r]: the number of tokens sent to rank r by all ranks
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

    # test get dispatch layout from group cast meta
    (
        ref_num_tokens_per_rank,
        _,  # ref_num_tokens_per_rdma_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
    ) = get_dispatch_layout_from_group_cast_meta(
        input_split_size_list=input_split_size_list,
        dst_indices_list=dst_indices_list,
        group=group,
        num_nodes=1,
    )

    # assert close to layout ref
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

    # get dispatch layout from buffer as reference
    if not use_topk:
        layout_topk_idx = transfer_splits_and_dst_idxs_to_topk_idx(
            split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            num_ranks=num_ranks,
        )
        assert num_experts == num_ranks
    else:
        layout_topk_idx = topk_idx
    (
        ref_num_tokens_per_rank,
        _,  # ref_num_tokens_per_rdma_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,  # event_overlap,
    ) = buffer.get_dispatch_layout(layout_topk_idx, num_experts)

    print(
        f"[RANK {rank}]: {layout_topk_idx.shape=} | {layout_topk_idx=}\n"
        f"{ref_num_tokens_per_rank.shape=} | {ref_num_tokens_per_rank=}\n"
        f"{ref_num_tokens_per_expert.shape=} | {ref_num_tokens_per_expert=}\n"
        f"{ref_is_token_in_rank.shape=} | {ref_is_token_in_rank=}\n",
        flush=True,
    )

    # assert close to layout ref
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

    # benchmark dispatch layout
    t = bench(lambda: buffer.get_dispatch_layout(layout_topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
        print("", flush=True)
    group.barrier()
    time.sleep(1)

    # Test dispatch
    def check_data(check_x, rank_prefix_matrix):
        if distinct_token:
            # distinct token cannot use this check
            return
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
                            "\n# ------    Test Intranode Dispatch   ------ #\n"
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, '
                            f'{"with" if with_topk else "without"} top-k '
                            f"(async={async_mode}, previous={previous_mode}) ...",
                            flush=True,
                        )

                    # prepare dispatch args
                    # x: shape=[num_local_tokens, hidden_dim]
                    # num_tokens_per_rank: shape=[num_ranks]: the number of tokens sent to each rank
                    # num_tokens_per_expert: shape=[num_experts]: the number of tokens sent to each expert
                    # is_token_in_rank: shape=[num_local_tokens, num_ranks]: whether a local token should be sent to a rank
                    # NOTE: if using top-k, the above args can be
                    #   calculated by buffer.get_dispatch_layout(topk_idx, num_experts)
                    #   where the topk_idx: shape=[num_local_tokens, topk]: the global expert idx for each local token
                    #   but we don't have to pass the topk_idx into dispatch kernels
                    dispatch_args = {
                        "x": current_x,
                        "recv_x": recv_x_gc_buf,
                        "is_token_in_rank": is_token_in_rank,
                        "num_tokens_per_rank": num_tokens_per_rank,
                        "num_tokens_per_expert": num_tokens_per_expert,
                        "config": config,
                        "async_finish": async_mode,
                        "post_perm_idx": perm_to_a2av_idx
                        if use_a2av_perm_idxs == "inside"
                        else None,
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
                    #   None if not with_topk
                    # recv_topk_weights: shape=[num_recv_tokens, topk]:
                    #   the corr. weight for each recv token's topk list
                    #   (if idx = -1, then weight = 0.)
                    #   None if not with_topk
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

                    # check in-place
                    if pass_out_buffer:
                        assert recv_x_gc_buf is not None
                        assert recv_x_gc_buf.data_ptr() == recv_x.data_ptr()  # type: ignore[union-attr]

                    # unpermute recv_x to the order indicated by
                    # output_split_size_list and src_index_list
                    if random_permute_output:
                        if use_a2av_perm_idxs == "inside":
                            # already permuted inside
                            pass
                        else:
                            recv_x_from_a2av = recv_x.clone()  # type: ignore[union-attr]
                            if use_a2av_perm_idxs == "outside":
                                recv_x = recv_x[unperm_from_a2av_idx]
                            elif use_a2av_perm_idxs == "no":
                                recv_x = unpermute_tensor(
                                    tensor=recv_x,
                                    unperm_after_a2a_kwargs=range_gather_post_dispatch_kwargs,
                                )
                            assert recv_x_from_a2av.shape == recv_x.shape  # type: ignore[union-attr]

                    # print
                    (
                        rank_prefix_matrix,  # handle[0]
                        channel_prefix_matrix,  # handle[1]
                        recv_channel_prefix_matrix,  # handle[2]
                        recv_src_idx,  # handle[3]
                        is_token_in_rank_handle,  # handle[4]
                        send_head,  # handle[5]
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
                            recv_topk_idx.eq(-1)  # type: ignore[union-attr]
                            | (
                                (recv_topk_idx >= 0)  # type: ignore[operator]
                                & (recv_topk_idx < (num_experts // num_ranks))
                            )
                        ).sum().item() == recv_topk_idx.numel()  # type: ignore[union-attr]
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count  # type: ignore[union-attr]

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()  # type: ignore[union-attr]
                        if current_x is not x_pure_rand:
                            recv_topk_weights[  # type: ignore[union-attr,index]
                                recv_topk_idx.eq(-1)  # type: ignore[union-attr]
                            ] = recv_topk_weights.amax(  # type: ignore[union-attr]
                                dim=1, keepdim=True
                            ).expand_as(  # type: ignore[union-attr]
                                recv_topk_weights
                            )[
                                recv_topk_idx.eq(-1)  # type: ignore[union-attr]
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
                                f"\n[RANK {rank}]: {recv_worst_x.shape=}\n"  # type: ignore[union-attr]
                                f"{recv_worst_topk_idx.shape=} | {recv_worst_topk_idx[0]=}\n"  # type: ignore[union-attr,index]
                                f"{recv_worst_topk_weights.shape=} | "  # type: ignore[union-attr]
                                f"{recv_worst_topk_weights[0]=}\n\n"  # type: ignore[union-attr,index]
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
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)  # type: ignore[union-attr,index]
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)  # type: ignore[union-attr]
                        assert torch.equal(recv_x, recv_worst_x[: recv_x.size(0)])
                        assert torch.equal(
                            recv_topk_idx, recv_worst_topk_idx[: recv_x.size(0)]  # type: ignore[index]
                        )
                        assert torch.equal(
                            recv_topk_weights_clone,
                            recv_worst_topk_weights[: recv_x.size(0)],  # type: ignore[index]
                        )
                        assert torch.all(
                            recv_worst_topk_idx[recv_x.size(0) :] == -1  # type: ignore[index]
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

                    # simulate gemm
                    x_combine = sim_gemm(recv_x, w=sim_gemm_weight)

                    # permute x to the rank order
                    if random_permute_output:
                        if use_a2av_perm_idxs == "inside":
                            # will permute inside
                            pass
                        else:
                            x_combine_before_to_a2av = x_combine.clone()
                            if use_a2av_perm_idxs == "outside":
                                x_combine = x_combine[perm_to_a2av_idx]
                            elif use_a2av_perm_idxs == "no":
                                x_combine = unpermute_tensor(
                                    tensor=x_combine,
                                    unperm_after_a2a_kwargs=range_gather_pre_combine_kwargs,
                                )
                            assert x_combine_before_to_a2av.shape == x_combine.shape

                    # prepare combine args
                    send_head_copy = send_head.clone()
                    combine_args = {
                        "x": x_combine,
                        "combined_x": combined_x_gr_buf,
                        "handle": handle,
                        "config": config,
                        "async_finish": async_mode,
                        "reduce_op": "sum",
                        "acc_reduce": acc_reduce_out_buffer,
                        "allow_empty_init_out_buf": allow_empty_init_out_buf,
                        # NOTE: still perm_to_a2av_idx, instead of unperm_to_a2av_idx
                        "pre_perm_idx": perm_to_a2av_idx
                        if use_a2av_perm_idxs == "inside"
                        else None,
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

                    # check in-place
                    if pass_out_buffer:
                        assert combined_x_gr_buf is not None
                        assert combined_x_gr_buf.data_ptr() == combined_x.data_ptr()

                    # print
                    combined_topk_weights_shape = (
                        combined_topk_weights.shape if with_topk else None  # type: ignore[union-attr]
                    )
                    print(
                        (
                            f"\n[RANK {rank}]: {combined_x.shape=} | {combined_x=}\n"
                            f"{combined_topk_weights_shape=} | {combined_topk_weights=}\n"  # type: ignore[union-attr]
                            f"Before combine: {send_head.shape=} | {send_head=}\n\n"
                        ),
                        flush=True,
                    )

                    # check
                    torch.testing.assert_close(combined_x, combined_x_gr)
                    assert torch.equal(
                        send_head[send_head_copy != -1],
                        send_head_copy[send_head_copy != -1],
                    )  # cached_notify_combine will modify send_head in-place for any entry == -1

                    send_token_nums = is_token_in_rank.sum(dim=1).unsqueeze(1)
                    check_x = combined_x.float() / send_token_nums
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    ref_x = sim_gemm(ref_x, w=sim_gemm_weight)
                    # if acc_reduce, the combined token should add with a constant rank bias
                    if acc_reduce_out_buffer:
                        ref_x += acc_reduce_constant / send_token_nums

                    # if some token is not sent to any rank, the combined token should be 0
                    if min_num_dst_ranks == 0:
                        zero_num_dst_ranks_mask = (send_token_nums == 0.0).expand_as(
                            combined_x
                        )
                        check_x[zero_num_dst_ranks_mask] = (
                            acc_reduce_constant if acc_reduce_out_buffer else 0.0
                        )
                        ref_x[zero_num_dst_ranks_mask] = (
                            acc_reduce_constant if acc_reduce_out_buffer else 0.0
                        )

                    diff = calc_diff(check_x, ref_x)
                    assert diff < 5e-6, f"{check_x} != {ref_x} with ({diff=})"
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
    for current_x in (x,):  # filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0,):
            if nvl_chunk_size > 0:
                config = GrpCollConfig(
                    num_sms=num_sms,
                    nvl_chunk_size=nvl_chunk_size,
                    nvl_buffer_size=nvl_buffer_size,
                )
            else:
                # Test default config as well
                GrpCollBuffer.set_num_sms(num_sms)
                config = GrpCollBuffer.get_dispatch_config(num_ranks)
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
            all_best_results_list = [
                torch.zeros_like(best_dispatch_results)
                for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(all_best_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_results_list[0].tolist()
    dispatch_config = GrpCollConfig(
        num_sms=best_dispatch_results[0],  # type: ignore[index]
        nvl_chunk_size=best_dispatch_results[1],  # type: ignore[index]
        nvl_buffer_size=nvl_buffer_size,
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
    combined_x_buf = torch.zeros_like(x) if pass_out_buffer else None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0,):
        if nvl_chunk_size > 0:
            config = GrpCollConfig(
                num_sms=num_sms,
                nvl_chunk_size=nvl_chunk_size,
                nvl_buffer_size=nvl_buffer_size,
            )
        else:
            # Test default config as well
            GrpCollBuffer.set_num_sms(num_sms)
            config = GrpCollBuffer.get_combine_config(num_ranks)
        tune_args = {
            "x": recv_x,
            "combined_x": combined_x_buf,
            "handle": handle,
            "config": config,
            "reduce_op": "sum",
            "acc_reduce": acc_reduce_out_buffer,
            "allow_empty_init_out_buf": allow_empty_init_out_buf,
        }
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


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # rank: global rank in default group
    # num_ranks: number of ranks in default group
    # group: the default world group

    use_grpcoll_mgr = True

    # init dist
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    test_ll_compatibility, num_rdma_bytes = False, 0
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
        num_rdma_bytes = GrpCollBuffer.get_low_latency_rdma_size_hint(
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

    num_sms = 24
    num_qps_per_rank = ll_num_experts // num_ranks if test_ll_compatibility else 1

    if local_rank == 0:
        print(
            (
                f"[config]: {num_ranks=} | {num_local_ranks=} | {group.size()=} | "
                f"{num_nvl_bytes=} ({num_nvl_bytes / 1e9:.2f} GB) | {num_rdma_bytes=} | {num_qps_per_rank=}\n"
            ),
            flush=True,
        )

    # NOTE: in buffer config, we auto set:
    # low_latency_mode=False,
    # num_qps_per_rank=1 if num_rdma_bytes == 0 else num_sms,
    # explicitly_destroy=True,
    buffer_config = GrpCollConfig(
        num_sms=num_sms,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
    )
    extra_buffer_kwargs = dict(
        low_latency_mode=test_ll_compatibility,
        num_qps_per_rank=num_qps_per_rank,
        explicitly_destroy=True,
    )

    if use_grpcoll_mgr:
        grpcoll_mgr.register_buffer(
            group=group,
            config=buffer_config,
            **extra_buffer_kwargs,
        )
        buffer = grpcoll_mgr.get_buffer(group)
    else:
        buffer_args = buffer_config.to_buffer_args()
        buffer_args.update(extra_buffer_kwargs)
        buffer = GrpCollBuffer(
            group,
            **buffer_args,
        )

    if local_rank == 0:
        print(
            f"\n\n============================Testing with {num_sms=}============================\n\n",
            flush=True,
        )
    test_main(args, num_sms, local_rank, num_ranks, rank, buffer, group)
    if local_rank == 0:
        print("", flush=True)

    # Destroy the buffer runtime
    if use_grpcoll_mgr:
        grpcoll_mgr.release_buffer(group)
    else:
        buffer.destroy()
        dist.barrier()

    # Destroy the process group
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
        # NOTE: the intranode kernel performance is highly dependent on the hidden size
        # hidden_size = 6 * 128 => bandwidth = 90~100 GB/s
        # hidden_size = 12 * 128 => bandwidth = 140~150 GB/s
        # hidden_size = 24 * 128 => bandwidth = 150~180 GB/s
        # hidden_size = 32 * 128 => bandwidth = 220~240 GB/s
        # hidden_size = 48 * 128 => bandwidth = 280~300 GB/s
        # hidden_size = 56 * 128 => bandwidth = 260~280 GB/s
        # hidden_size = 60 * 128 => bandwidth = 230~250 GB/s
        # hidden_size = 62 * 128 => bandwidth = 200~210 GB/s
        "--hidden",
        type=int,
        default=56 * 128,
        help="Hidden dimension size (default: 56x128=7168)",
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
