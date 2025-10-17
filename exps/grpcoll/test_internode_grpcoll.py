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
from magi_attention.comm.primitive.grpcoll._handle import GrpCollInterHandle
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_mgr
from magi_attention.comm.primitive.grpcoll.utils import (
    get_a2av_perm_idxs_from_group_cast_meta,
    get_dispatch_layout_from_group_cast_meta,
    transfer_splits_and_dst_idxs_to_topk_idx,
    unpermute_tensor,
)
from magi_attention.utils import pad_and_pack_tensors, setup_dist_env

# isort: split
from grpcoll_utils import (
    bench,
    bench_kineto,
    calc_diff,
    get_output_split_size_list_and_src_index_list,
    get_random_dst_indices_list,
    get_random_split_size_list,
    per_token_cast_back,
    per_token_cast_to_fp8,
    perm_idxs2unperm_idxs,
    sim_gemm,
    transfer_group_cast_meta_to_dispatch_meta,
)


def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    buffer: GrpCollBuffer,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts = args.num_experts
    distinct_token = True
    random_permute_output = True
    sim_gemm_weight = 2.0
    min_num_dst_ranks = 0
    allow_empty_init_out_buf = (  # if every token has at least one dst, we can empty-init
        min_num_dst_ranks > 0
    )
    pass_out_buffer = True
    acc_reduce_out_buffer = False  # TODO: support acc_reduce for internode_group_reduce
    acc_reduce_constant = rank
    if acc_reduce_out_buffer:
        assert pass_out_buffer, "acc_reduce_out_buffer requires pass_out_buffer"
    use_a2av_perm_idxs = "outside"  # TODO: support a2av_perm_idxs inside
    assert use_a2av_perm_idxs in ("no", "outside", "inside")

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    num_local_experts = num_experts // num_ranks
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
    config = GrpCollConfig(
        num_sms=num_sms,  # num_sms, default 20
        nvl_chunk_size=num_max_nvl_chunked_send_tokens,  # num_max_nvl_chunked_send_tokens (nvl_chunk_size), default 6
        nvl_buffer_size=num_max_nvl_chunked_recv_tokens,  # num_max_nvl_chunked_recv_tokens (nvl_buffer_size), default 256
        rdma_chunk_size=num_max_rdma_chunked_send_tokens,  # num_max_rdma_chunked_send_tokens, default 6
        rdma_buffer_size=num_max_rdma_chunked_recv_tokens,  # num_max_rdma_chunked_recv_tokens, default 256
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
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)

    # Random score
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

    # to device meta
    input_split_sizes = torch.tensor(
        input_split_size_list,
        dtype=torch.int64,
        device="cuda",
    )
    output_split_sizes = torch.tensor(
        output_split_size_list,
        dtype=torch.int64,
        device="cuda",
    )
    dst_indices_tensor_list: list[torch.Tensor] = [
        torch.tensor(
            dst_indices,
            dtype=torch.int64,
            device="cuda",
        )
        for dst_indices in dst_indices_list
    ]
    dst_indices = pad_and_pack_tensors(  # shape: [num_splits, num_ranks]
        tensors=dst_indices_tensor_list,
        target_length=num_ranks,
        padding_value=-1,
        dtype=torch.int64,
        device="cuda",
    )
    src_index = torch.tensor(
        src_index_list,
        dtype=torch.int64,
        device="cuda",
    )

    # get ref dispatch output by group-cast
    recv_x_gc = torch.empty(
        (sum(output_split_size_list), *x.shape[1:]), dtype=torch.bfloat16, device="cuda"
    )
    recv_x_gc_buf = recv_x_gc.clone() if pass_out_buffer else None
    work_with_pf_gc = group_cast(
        input=x,
        output=recv_x_gc,
        input_split_sizes=input_split_size_list,
        output_split_sizes=output_split_size_list,
        dst_indices=dst_indices_list,
        src_index=src_index_list,
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
        input_split_sizes=output_split_size_list,
        output_split_sizes=input_split_size_list,
        dst_index=src_index_list,
        src_indices=dst_indices_list,
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
        range_gather_post_dispatch_kwargs,
        range_gather_pre_combine_kwargs,
    ) = transfer_group_cast_meta_to_dispatch_meta(
        rank=rank,
        num_ranks=num_ranks,
        num_nodes=num_nodes,
        num_local_experts=num_local_experts,
        input_split_size_list=input_split_size_list,
        dst_indices_list=dst_indices_list,
        output_split_size_list=output_split_size_list,
        src_index_list=src_index_list,
        use_topk=False,
        use_a2a_order_output=not random_permute_output,
    )

    print(
        f"[RANK {rank}]: {input_split_size_list=} | {dst_indices_list=} | "
        f"{output_split_size_list=} | {src_index_list=} | {sum(output_split_size_list)=}\n",
        f"[RANK {rank}]: {topk_idx=} | {topk_weights=}\n",
        f"[RANK {rank}]: {rank_idx=} | {rdma_rank_idx=}\n",
        flush=True,
    )

    # get perm/unperm idxs to/from a2av through group-cast meta args
    # which is used to replace the post-dispatch range_gather and pre-combine range_gather

    # use host meta
    perm_to_a2av_idx = get_a2av_perm_idxs_from_group_cast_meta(
        output_split_sizes=output_split_size_list,
        src_index=src_index_list,
        num_ranks=num_ranks,
    )
    unperm_from_a2av_idx = perm_idxs2unperm_idxs(perm_to_a2av_idx)

    # use device meta
    perm_to_a2av_idx_device = get_a2av_perm_idxs_from_group_cast_meta(
        output_split_sizes=output_split_sizes,
        src_index=src_index,
        num_ranks=num_ranks,
    )
    unperm_from_a2av_idx_device = perm_idxs2unperm_idxs(perm_to_a2av_idx_device)

    assert torch.equal(unperm_from_a2av_idx, unperm_from_a2av_idx_device)
    assert torch.equal(perm_to_a2av_idx, perm_to_a2av_idx_device)

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

    # test get dispatch layout from group cast meta

    # use host meta
    (
        ref_num_tokens_per_rank,
        ref_num_tokens_per_rdma_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
    ) = get_dispatch_layout_from_group_cast_meta(
        input_split_sizes=input_split_size_list,
        dst_indices=dst_indices_list,
        group=group,
        num_nodes=num_nodes,
    )

    # use device meta
    (
        ref_num_tokens_per_rank_device,
        ref_num_tokens_per_rdma_rank_device,
        ref_num_tokens_per_expert_device,
        ref_is_token_in_rank_device,
    ) = get_dispatch_layout_from_group_cast_meta(
        input_split_sizes=input_split_sizes,
        dst_indices=dst_indices,
        group=group,
        num_nodes=num_nodes,
    )

    # assert close to layout ref
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    assert torch.allclose(ref_num_tokens_per_rank_device, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank_device, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert_device, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank_device, is_token_in_rank)

    # get dispatch layout from buffer as reference
    assert num_experts == num_ranks

    # use host meta
    layout_topk_idx = transfer_splits_and_dst_idxs_to_topk_idx(
        input_split_sizes=input_split_size_list,
        dst_indices=dst_indices_list,
        num_ranks=num_ranks,
    )

    # use device meta
    layout_topk_idx_device = transfer_splits_and_dst_idxs_to_topk_idx(
        input_split_sizes=input_split_sizes,
        dst_indices=dst_indices,
        num_ranks=num_ranks,
    )

    assert torch.equal(layout_topk_idx, layout_topk_idx_device)

    (
        ref_num_tokens_per_rank,
        ref_num_tokens_per_rdma_rank,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,  # event_overlap,
    ) = buffer.get_dispatch_layout(layout_topk_idx, num_experts)

    print(
        f"[RANK {rank}]: {layout_topk_idx.shape=} | {layout_topk_idx=}\n"
        f"{ref_num_tokens_per_rank.shape=} | {ref_num_tokens_per_rank=}\n"
        f"{ref_num_tokens_per_rdma_rank.shape=} | {ref_num_tokens_per_rdma_rank=}\n"  # type: ignore[union-attr]
        f"{ref_num_tokens_per_expert.shape=} | {ref_num_tokens_per_expert=}\n"
        f"{ref_is_token_in_rank.shape=} | {ref_is_token_in_rank=}\n",
        flush=True,
    )

    # assert close to layout ref
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
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
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        if distinct_token or random_permute_output:
            # distinct token cannot use this check
            return
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (True,):  # (False, True)
        for async_mode in (True,):  # (False, True)
            for current_x in (x,):
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
                        f"(async={async_mode}, previous={previous_mode}) ...",
                        flush=True,
                        end="",
                    )
                dispatch_args = {
                    "x": current_x,
                    "recv_x": recv_x_gc_buf,
                    "num_tokens_per_rank": num_tokens_per_rank,
                    "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
                    "is_token_in_rank": is_token_in_rank,
                    "num_tokens_per_expert": num_tokens_per_expert,
                    "config": config,
                    "async_finish": async_mode,
                    "post_perm_idx": perm_to_a2av_idx
                    if use_a2av_perm_idxs == "inside"
                    else None,
                }
                if previous_mode:
                    dispatch_args.update({"previous_event": buffer.capture()})

                # dispatch
                # recv_x: shape=[num_recv_tokens, hidden_dim]:
                #   the recv tokens for this rank (in rank order just like a2a output,
                #   while the boundary is indicated by rank_prefix_matrix)
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
                    _,  # recv_lse
                    handle,
                    event,
                ) = buffer.group_cast(**dispatch_args)

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
                assert isinstance(handle, GrpCollInterHandle)

                is_token_in_rank_handle = handle.is_token_in_rank
                rdma_channel_prefix_matrix = handle.rdma_channel_prefix_matrix
                gbl_channel_prefix_matrix = handle.gbl_channel_prefix_matrix
                recv_rdma_channel_prefix_matrix = handle.recv_rdma_channel_prefix_matrix
                recv_rdma_rank_prefix_sum = handle.recv_rdma_rank_prefix_sum
                recv_gbl_channel_prefix_matrix = handle.recv_gbl_channel_prefix_matrix
                recv_gbl_rank_prefix_sum = handle.recv_gbl_rank_prefix_sum
                recv_src_meta = handle.recv_src_meta
                send_rdma_head = handle.send_rdma_head
                send_nvl_head = handle.send_nvl_head

                print(
                    (
                        f"\n[RANK {rank}]: {recv_x.shape=} | {recv_x=}\n"  # type: ignore[union-attr]
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
                if current_x is not x_pure_rand:
                    check_data(recv_x, recv_gbl_rank_prefix_sum)

                if local_rank == 0:
                    print(
                        "\n# ------    Test Internode Cached Dispatch   ------ #\n",
                        flush=True,
                    )

                # Test cached dispatch (must without top-k staffs)
                dispatch_args = {
                    "x": current_x,
                    "handle": handle,
                    "config": config,
                    "async_finish": async_mode,
                }
                if previous_mode:
                    dispatch_args.update({"previous_event": buffer.capture()})
                (
                    recv_cached_x,
                    _,  # recv_cached_lse
                    _,  # handle
                    event,
                ) = buffer.group_cast(**dispatch_args)
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
                # bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                # bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                combine_args = {
                    "x": x_combine,
                    "combined_x": combined_x_gr_buf,
                    "bias": None,
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
                if previous_mode:
                    combine_args.update({"previous_event": buffer.capture()})

                # combine
                # combined_x: shape=[num_tokens, hidden_size]: combined_x[i]:
                #   the ith token's sum-reduction result of top-k experts
                #   NOTE: the send_rdma_head will be modified in-place in internode::cached_notify
                #       for the entries == -1 to the position of next valid token (encoded to -p-1)
                #       since the combine kernel needs to know the channel position when iterating at this token,
                #       even though it is not sent to the target rdma rank
                #   NOTE: the send_nvl_head will be modified in-place in internode::cached_notify
                #       for the entries == -1 to the position of next valid token (encoded to -p-1)
                #       since the combine kernel needs to know the channel position when iterating at this token,
                #       even though it is not sent to the target rdma rank
                combined_x, event = buffer.group_reduce(**combine_args)

                # wait
                event.current_stream_wait() if async_mode else ()

                # check in-place
                if pass_out_buffer:
                    assert combined_x_gr_buf is not None
                    assert combined_x_gr_buf.data_ptr() == combined_x.data_ptr()

                # print
                print(
                    (
                        f"\n[RANK {rank}]: {combined_x.shape=} | {combined_x=}\n"
                        f"Before combine: {send_rdma_head.shape=} | {send_rdma_head=}\n\n"
                        f"Before combine: {send_nvl_head.shape=} | {send_nvl_head=}\n\n"
                    ),
                    flush=True,
                )

                # check
                torch.testing.assert_close(combined_x, combined_x_gr)
                # check_x =
                #   (combined_x.float() - bias_0.float() - bias_1.float())
                #   / is_token_in_rank.sum(dim=1).unsqueeze(1)

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

                # For later tuning
                dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

    if local_rank == 0:
        print("passed", flush=True)

    # sync before tuning
    torch.cuda.synchronize()
    dist.barrier()

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x,):  # (x_e4m3, x):
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
                config = GrpCollConfig(
                    num_sms=num_sms,
                    nvl_chunk_size=nvl_chunk_size,
                    nvl_buffer_size=nvl_buffer_size,
                    rdma_chunk_size=rdma_chunk_size,
                    rdma_buffer_size=rdma_buffer_size,
                )
                tune_args = {"x": current_x, "handle": handle, "config": config}
                t, notify_t = bench_kineto(
                    lambda: buffer.group_cast(**tune_args), ("dispatch", "notify")
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

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor(
                [best_results[0], best_results[1], best_results[2]],  # type: ignore[index]
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
        rdma_chunk_size=best_dispatch_results[2],  # type: ignore[index]
        rdma_buffer_size=rdma_buffer_size,
    )

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    (
        recv_x,
        _,  # recv_lse
        handle,
        _,  # event
    ) = buffer.group_cast(
        **dispatch_args
    )  # type: ignore[assignment]

    # Tune combine performance
    best_time, best_results = 1e10, None
    combined_x_buf = torch.zeros_like(x) if pass_out_buffer else None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = GrpCollConfig(
                num_sms=num_sms,
                nvl_chunk_size=nvl_chunk_size,
                nvl_buffer_size=nvl_buffer_size,
                rdma_chunk_size=rdma_chunk_size,
                rdma_buffer_size=rdma_buffer_size,
            )
            tune_args = {
                "x": recv_x,
                "combined_x": combined_x_buf,
                "handle": handle,
                "config": config,
                "reduce_op": "sum",
                "acc_reduce": acc_reduce_out_buffer,
                "allow_empty_init_out_buf": allow_empty_init_out_buf,
            }
            t, notify_t = bench_kineto(
                lambda: buffer.group_reduce(**tune_args), ("combine", "notify")
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
    use_grpcoll_mgr = True

    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    # init dist
    (
        rank,
        local_rank,
        num_ranks,
        num_nodes,
        num_local_ranks,
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
    num_topk = num_ranks
    num_experts = num_ranks
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
        low_latency_mode=False,
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

    assert num_local_ranks == 8 and num_ranks > 8

    test_main(
        args,
        num_sms,
        local_rank,
        num_local_ranks,
        num_ranks,
        num_nodes,
        rank,
        buffer,
        group,
    )
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
        # TODO: find out the relationship between hidden size and bandwidth
        "--hidden",
        type=int,
        default=56 * 128,
        help="Hidden dimension size (default: 56x128=7168)",
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
