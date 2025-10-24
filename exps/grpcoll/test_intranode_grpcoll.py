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

# mypy: disable-error-code="union-attr,index"
import argparse
import time
from typing import Any

import torch
import torch.distributed as dist

from magi_attention.api.functools import pad_at_dim
from magi_attention.comm.primitive.grpcoll import group_cast, group_reduce
from magi_attention.comm.primitive.grpcoll._buffer import GrpCollBuffer
from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
from magi_attention.comm.primitive.grpcoll._handle import GrpCollIntraHandle
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_mgr
from magi_attention.comm.primitive.grpcoll.utils import (
    get_a2av_perm_idxs_from_group_cast_meta,
    get_native_group_cast_meta,
    transfer_splits_and_dst_idxs_to_t2r_idx,
    unpermute_output,
)
from magi_attention.common.enum import GroupReduceOp
from magi_attention.utils import pad_and_pack_tensors

# isort: split
from grpcoll_utils import (
    bench,
    calc_diff,
    get_output_split_size_list_and_src_index_list,
    get_random_dst_indices_list,
    get_random_split_size_list,
    init_dist,
    perm_idxs2unperm_idxs,
    sim_gemm,
    transfer_native_group_cast_meta,
)


def prepare_test_func_kwargs(
    rank: int,
    local_rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: GrpCollBuffer,
    num_tokens: int,
    hidden_size: int,
    num_heads: int,
    num_input_splits: int,
    num_experts: int,
    num_local_experts: int,
    dtype: torch.dtype,
    distinct_token: bool,
    cast_lse: bool,
    reduce_op: GroupReduceOp,
    pass_out_buffer: bool,
    pass_out_lse_buffer: bool,
    pass_padded_out_buffer: bool,
    random_permute_output: bool,
    sim_gemm_weight: float,
    acc_reduce_out_buffer: bool,
    acc_reduce_constant: int,
    min_num_dst_ranks: int,
) -> dict[str, Any]:
    # Random data
    head_dim = hidden_size // num_heads
    x = torch.ones((num_tokens, hidden_size), dtype=dtype, device="cuda")
    if cast_lse:
        assert num_heads > 0
        lse = (
            torch.arange(num_tokens, dtype=torch.float32, device="cuda")
            .repeat_interleave(repeats=num_heads, dim=0)
            .reshape(-1, num_heads)
        )
        lse_shape = lse.shape
    else:
        lse = None
        lse_shape = None

    if distinct_token:
        x *= torch.arange(
            rank * num_tokens,
            (rank + 1) * num_tokens,
            dtype=dtype,
            device="cuda",
        ).view(-1, 1) / (num_ranks * num_tokens)
    else:
        x *= rank

    print(f"[RANK {rank}]: {x.shape=} | {x=}\n" f"{lse_shape=} | {lse=}\n", flush=True)

    # Random score (transfered from group-cast meta args)
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
        (sum(output_split_size_list), *x.shape[1:]), dtype=dtype, device="cuda"
    )
    recv_x_gc_buf = recv_x_gc.clone() if pass_out_buffer else None
    pad_size = num_tokens * num_ranks - sum(output_split_size_list)
    if pass_out_buffer and pass_padded_out_buffer:
        recv_x_gc_buf = pad_at_dim(recv_x_gc_buf, dim=0, pad_size=pad_size, value=-1)

    recv_lse_gc = (
        torch.empty((recv_x_gc.shape[0], lse.shape[1]), dtype=lse.dtype, device="cuda")
        if cast_lse and pass_out_lse_buffer
        else None
    )
    recv_lse_gc_buf = recv_lse_gc.clone() if cast_lse and pass_out_lse_buffer else None
    if cast_lse and pass_out_lse_buffer and pass_padded_out_buffer:
        recv_lse_gc_buf = pad_at_dim(
            recv_lse_gc_buf, dim=0, pad_size=pad_size, value=-1
        )

    work_with_pf_gc = group_cast(
        input=x,
        output=recv_x_gc,
        input_split_sizes=input_split_size_list,
        output_split_sizes=output_split_size_list,
        dst_indices=dst_indices_list,
        src_index=src_index_list,
        group=group,
        async_op=True,
        cast_lse=cast_lse,
        input_lse=lse,
        output_lse=recv_lse_gc,
    )

    if cast_lse:
        recv_x_gc, recv_lse_gc = work_with_pf_gc.wait_post_process(
            recv_x_gc, recv_lse_gc
        )
        recv_lse_gc_shape = recv_lse_gc.shape
    else:
        recv_x_gc = work_with_pf_gc.wait_post_process(recv_x_gc)
        recv_lse_gc_shape = None

    print(
        f"[RANK {rank}]: {recv_x_gc.shape=} | {recv_x_gc=}\n"
        f"{recv_lse_gc_shape=} | {recv_lse_gc=}\n"
        f"{pad_size=}\n",
        flush=True,
    )

    # get ref combine output by group-reduce
    x_gr = sim_gemm(recv_x_gc, w=sim_gemm_weight)
    combined_x_gr = torch.zeros_like(x)
    if acc_reduce_out_buffer:
        combined_x_gr += acc_reduce_constant
    combined_x_gr_buf = combined_x_gr.clone() if pass_out_buffer else None

    lse_gr = recv_lse_gc.clone() if reduce_op == "lse" else None
    combined_lse_gr = (
        torch.full(
            (combined_x_gr.shape[0], lse.shape[1]),
            fill_value=1 / (acc_reduce_constant + 1)
            if acc_reduce_out_buffer
            else float("-inf"),
            dtype=lse.dtype,
            device="cuda",
        )
        if reduce_op == "lse"
        else None
    )
    combined_lse_gr_buf = (
        combined_lse_gr.clone() if reduce_op == "lse" and pass_out_lse_buffer else None
    )

    work_with_pf_gr = group_reduce(
        input=x_gr.view(-1, num_heads, head_dim) if reduce_op == "lse" else x_gr,
        output=combined_x_gr.view(-1, num_heads, head_dim)
        if reduce_op == "lse"
        else combined_x_gr,
        input_split_sizes=output_split_size_list,
        output_split_sizes=input_split_size_list,
        dst_index=src_index_list,
        src_indices=dst_indices_list,
        group=group,
        async_op=True,
        reduce_op=reduce_op,
        input_lse=lse_gr,
        output_lse=combined_lse_gr,
    )

    if reduce_op == "lse":
        combined_x_gr, combined_lse_gr = work_with_pf_gr.wait_post_process(
            combined_x_gr.view(-1, num_heads, head_dim), combined_lse_gr
        )
        combined_lse_gr_shape = combined_lse_gr.shape
        combined_x_gr = combined_x_gr.view(-1, hidden_size)
    else:
        combined_x_gr = work_with_pf_gr.wait_post_process(combined_x_gr)
        combined_lse_gr_shape = None
    print(
        f"[RANK {rank}]: {combined_x_gr.shape=} | {combined_x_gr=}\n"
        f"{combined_lse_gr_shape=} | {combined_lse_gr=}\n",
        flush=True,
    )

    # transfer group-cast meta args to dispatch meta args
    (
        rank_idx,
        _,  # rdma_rank_idx
        num_tokens_per_rank,
        _,  # num_tokens_per_rdma_rank
        is_token_in_rank,
        topk_idx,
        topk_weights,
        _,  # num_tokens_per_expert
        range_gather_post_dispatch_kwargs,
        range_gather_pre_combine_kwargs,
    ) = transfer_native_group_cast_meta(
        rank=rank,
        num_ranks=num_ranks,
        num_nodes=1,
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
        f"{output_split_size_list=} | {src_index_list=} | "
        f"{sum(output_split_size_list)=}\n",
        f"{topk_idx=} | {topk_weights=}\n",
        f"{rank_idx=}\n",
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
        output_seqlen=recv_x_gc_buf.shape[0],
    )
    if pass_padded_out_buffer:
        unperm_from_a2av_idx_device = perm_idxs2unperm_idxs(
            perm_to_a2av_idx_device[: recv_x_gc.shape[0]]
        )
        unperm_from_a2av_idx_device = pad_at_dim(
            unperm_from_a2av_idx_device,
            dim=0,
            pad_size=perm_to_a2av_idx_device.shape[0]
            - unperm_from_a2av_idx_device.shape[0],
            value=-1,
        )
    else:
        unperm_from_a2av_idx_device = perm_idxs2unperm_idxs(perm_to_a2av_idx_device)

    assert torch.equal(
        unperm_from_a2av_idx, unperm_from_a2av_idx_device[: recv_x_gc.shape[0]]
    )
    assert torch.equal(perm_to_a2av_idx, perm_to_a2av_idx_device[: recv_x_gc.shape[0]])

    print(
        f"[RANK {rank}]: {perm_to_a2av_idx_device.shape=} | {perm_to_a2av_idx_device=}\n"
        f"{unperm_from_a2av_idx_device.shape=} | {unperm_from_a2av_idx_device=}\n",
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

    # test get dispatch layout from group cast meta

    # use host meta
    (
        ref_num_tokens_per_rank,
        _,  # ref_num_tokens_per_rdma_rank,
        ref_is_token_in_rank,
    ) = get_native_group_cast_meta(
        input_split_sizes=input_split_size_list,
        dst_indices=dst_indices_list,
        group=group,
        num_nodes=1,
    )

    # use device meta
    (
        ref_num_tokens_per_rank_device,
        _,  # ref_num_tokens_per_rdma_rank_device,
        ref_is_token_in_rank_device,
    ) = get_native_group_cast_meta(
        input_split_sizes=input_split_sizes,
        dst_indices=dst_indices,
        group=group,
        num_nodes=1,
    )

    # assert close to layout ref
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    assert torch.allclose(ref_num_tokens_per_rank_device, num_tokens_per_rank)
    assert torch.allclose(ref_is_token_in_rank_device, is_token_in_rank)

    # get dispatch layout from buffer as reference
    assert num_experts == num_ranks

    # use host meta
    layout_t2r_idx = transfer_splits_and_dst_idxs_to_t2r_idx(
        input_split_sizes=input_split_size_list,
        dst_indices=dst_indices_list,
        num_ranks=num_ranks,
    )

    # use device meta
    layout_t2r_idx_device = transfer_splits_and_dst_idxs_to_t2r_idx(
        input_split_sizes=input_split_sizes,
        dst_indices=dst_indices,
        num_ranks=num_ranks,
    )

    assert torch.equal(layout_t2r_idx, layout_t2r_idx_device)

    (
        ref_num_tokens_per_rank,
        _,  # ref_num_tokens_per_rdma_rank,
        ref_is_token_in_rank,
        _,  # event_overlap,
    ) = buffer.get_group_cast_meta(layout_t2r_idx)

    print(
        f"[RANK {rank}]: {layout_t2r_idx.shape=} | {layout_t2r_idx=}\n"
        f"{ref_num_tokens_per_rank.shape=} | {ref_num_tokens_per_rank=}\n"
        f"{ref_is_token_in_rank.shape=} | {ref_is_token_in_rank=}\n",
        flush=True,
    )

    # assert close to layout ref
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

    # benchmark dispatch layout
    t = bench(lambda: buffer.get_group_cast_meta(layout_t2r_idx))[0]
    if local_rank == 0:
        print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
        print("", flush=True)
    group.barrier()
    time.sleep(1)

    return dict(
        x=x,
        lse=lse,
        recv_x_gc=recv_x_gc,
        recv_x_gc_buf=recv_x_gc_buf,
        recv_lse_gc=recv_lse_gc,
        recv_lse_gc_buf=recv_lse_gc_buf,
        combined_x_gr=combined_x_gr,
        combined_x_gr_buf=combined_x_gr_buf,
        combined_lse_gr=combined_lse_gr,
        combined_lse_gr_buf=combined_lse_gr_buf,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_rank=num_tokens_per_rank,
        perm_to_a2av_idx=perm_to_a2av_idx_device,
        unperm_from_a2av_idx=unperm_from_a2av_idx_device,
        range_gather_post_dispatch_kwargs=range_gather_post_dispatch_kwargs,
        range_gather_pre_combine_kwargs=range_gather_pre_combine_kwargs,
        gbl_num_tokens_per_rank=gbl_num_tokens_per_rank,
    )


def test_func(
    rank: int,
    local_rank: int,
    config: GrpCollConfig,
    buffer: GrpCollBuffer,
    async_mode: bool,
    previous_mode: bool,
    cast_lse: bool,
    reduce_op: GroupReduceOp,
    pass_out_buffer: bool,
    pass_out_lse_buffer: bool,
    pass_padded_out_buffer: bool,
    random_permute_output: bool,
    use_a2av_perm_idxs: str,
    sim_gemm_weight: float,
    acc_reduce_out_buffer: bool,
    acc_reduce_constant: int,
    min_num_dst_ranks: int,
    **kwargs,
) -> dict[str, Any]:
    # fetch kwargs
    x: torch.Tensor = kwargs["x"]
    lse: torch.Tensor | None = kwargs["lse"]
    recv_x_gc: torch.Tensor = kwargs["recv_x_gc"]
    recv_x_gc_buf: torch.Tensor | None = kwargs["recv_x_gc_buf"]
    recv_lse_gc: torch.Tensor | None = kwargs["recv_lse_gc"]
    recv_lse_gc_buf: torch.Tensor | None = kwargs["recv_lse_gc_buf"]
    combined_x_gr: torch.Tensor = kwargs["combined_x_gr"]
    combined_lse_gr: torch.Tensor | None = kwargs["combined_lse_gr"]
    combined_x_gr_buf: torch.Tensor | None = kwargs["combined_x_gr_buf"]
    combined_lse_gr_buf: torch.Tensor | None = kwargs["combined_lse_gr_buf"]
    is_token_in_rank: torch.Tensor = kwargs["is_token_in_rank"]
    num_tokens_per_rank: torch.Tensor = kwargs["num_tokens_per_rank"]
    perm_to_a2av_idx: torch.Tensor = kwargs["perm_to_a2av_idx"]
    unperm_from_a2av_idx: torch.Tensor = kwargs["unperm_from_a2av_idx"]
    range_gather_post_dispatch_kwargs: dict = kwargs[
        "range_gather_post_dispatch_kwargs"
    ]
    range_gather_pre_combine_kwargs: dict = kwargs["range_gather_pre_combine_kwargs"]
    gbl_num_tokens_per_rank: torch.Tensor = kwargs["gbl_num_tokens_per_rank"]

    # prepare dispatch args
    # x: shape=[num_local_tokens, hidden_dim]
    # num_tokens_per_rank: shape=[num_ranks]: the number of tokens sent to each rank
    # is_token_in_rank: shape=[num_local_tokens, num_ranks]: whether a local token should be sent to a rank
    common_dispatch_args = {  # w/o handle tensors
        "x": x,
        "recv_x": recv_x_gc_buf,
        "config": config,
        "async_finish": async_mode,
        "post_perm_idx": perm_to_a2av_idx if use_a2av_perm_idxs == "inside" else None,
        "cast_lse": cast_lse,
        "lse": lse,
        "recv_lse": recv_lse_gc_buf,
    }
    dispatch_args = common_dispatch_args | {
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_rank": num_tokens_per_rank,
    }
    if previous_mode:
        dispatch_args.update({"previous_event": buffer.capture()})

    # --------------      test normal dispatch       -------------- #

    if local_rank == 0:
        print(
            "\n# ------    Test Normal Intranode Dispatch   ------ #\n",
            flush=True,
        )

    # dispatch
    # recv_x: shape=[num_recv_tokens, hidden_dim]:
    #   the recv tokens for this rank (in rank order just like a2a output,
    #   while the boundary is indicated by rank_prefix_matrix)
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
        recv_lse,
        handle,
        event,
    ) = buffer.group_cast(**dispatch_args)

    # wait
    event.current_stream_wait() if async_mode else ()

    # check recv_x in-place
    if pass_out_buffer:
        assert recv_x_gc_buf is not None
        assert recv_x_gc_buf.data_ptr() == recv_x.data_ptr()

    # check recv_lse in-place
    if cast_lse and pass_out_lse_buffer:
        assert recv_lse_gc_buf is not None
        assert recv_lse_gc_buf.data_ptr() == recv_lse.data_ptr()

    # unpermute recv_x to the order indicated by
    # output_split_size_list and src_index_list
    if random_permute_output:
        if use_a2av_perm_idxs == "inside":
            # already permuted inside
            pass
        else:  # "outside" or "no"
            # recv_x
            recv_x_from_a2av = recv_x.clone()
            if use_a2av_perm_idxs == "outside":
                recv_x = recv_x[unperm_from_a2av_idx]
            elif use_a2av_perm_idxs == "no":
                recv_x = unpermute_output(
                    output=recv_x,
                    unperm_after_a2a_kwargs=range_gather_post_dispatch_kwargs,
                )
            assert recv_x_from_a2av.shape == recv_x.shape

            # recv_lse
            if cast_lse:
                recv_lse_from_a2av = recv_lse.clone()
                if use_a2av_perm_idxs == "outside":
                    recv_lse = recv_lse[unperm_from_a2av_idx]
                elif use_a2av_perm_idxs == "no":
                    recv_lse = unpermute_output(
                        output=recv_lse,
                        unperm_after_a2a_kwargs=range_gather_post_dispatch_kwargs,
                    )
                assert recv_lse_from_a2av.shape == recv_lse.shape

    # print
    assert isinstance(handle, GrpCollIntraHandle)

    actual_gc_output_seqlen = recv_x_gc.size(0)
    recv_lse_shape = recv_lse.shape if cast_lse else None
    rank_prefix_matrix = handle.rank_prefix_matrix
    channel_prefix_matrix = handle.channel_prefix_matrix
    recv_channel_prefix_matrix = handle.recv_channel_prefix_matrix
    recv_src_idx = handle.recv_src_idx
    is_token_in_rank_handle = handle.is_token_in_rank
    send_head = handle.send_head

    print(
        (
            f"\n[RANK {rank}]: {recv_x.shape=} | {recv_x=}\n"
            f"{recv_lse_shape=} | {recv_lse=}\n"
            f"{rank_prefix_matrix.shape=} | {rank_prefix_matrix=}\n"
            f"{channel_prefix_matrix.shape=} | {channel_prefix_matrix=}\n"
            f"{recv_channel_prefix_matrix.shape=} | {recv_channel_prefix_matrix=}\n"
            f"{recv_src_idx.shape=} | {recv_src_idx=}\n"
            f"{is_token_in_rank_handle.shape=} | {is_token_in_rank_handle=}\n"
            f"{perm_to_a2av_idx.shape=} | {perm_to_a2av_idx=}\n"
            f"After dipatch: {send_head.shape=} | {send_head=}\n\n"
        ),
        flush=True,
    )

    # check
    if pass_padded_out_buffer:
        assert recv_x.size(0) > actual_gc_output_seqlen
        assert torch.equal(recv_x[:actual_gc_output_seqlen], recv_x_gc)
    else:
        assert torch.equal(recv_x, recv_x_gc)
    if cast_lse:
        if pass_padded_out_buffer:
            assert recv_lse.size(0) > actual_gc_output_seqlen
            assert torch.equal(recv_lse[:actual_gc_output_seqlen], recv_lse_gc)
        else:
            assert torch.equal(recv_lse, recv_lse_gc)
    assert torch.equal(is_token_in_rank_handle, is_token_in_rank)
    assert torch.equal(channel_prefix_matrix[:, -1], num_tokens_per_rank)
    assert torch.equal(
        recv_channel_prefix_matrix[rank, 1:],
        channel_prefix_matrix[rank, :-1],
    )
    assert torch.all(recv_channel_prefix_matrix[:, 0] == 0)
    assert torch.all(send_head[is_token_in_rank_handle == -1] == -1)
    assert perm_to_a2av_idx.size(0) == recv_x.size(0)
    assert gbl_num_tokens_per_rank[rank].item() == actual_gc_output_seqlen

    # specific check for recv_lse
    if cast_lse:
        assert recv_lse.size(0) == recv_src_idx.size(0)
        num_heads = recv_lse.size(1)

        if random_permute_output:
            if use_a2av_perm_idxs == "no":
                permed_recv_src_idx = unpermute_output(
                    output=recv_src_idx,
                    unperm_after_a2a_kwargs=range_gather_post_dispatch_kwargs,
                )
            else:  # "inside" or "outside"
                # NOTE: we won't permute recv_src_idx inside for now
                permed_recv_src_idx = recv_src_idx[unperm_from_a2av_idx]
        else:
            permed_recv_src_idx = recv_src_idx

        repeated_permed_recv_src_idx = (
            permed_recv_src_idx.repeat_interleave(repeats=num_heads, dim=0)
            .reshape(-1, num_heads)
            .to(recv_lse.dtype)
        )

        if pass_padded_out_buffer:
            assert torch.equal(
                recv_lse[:actual_gc_output_seqlen],
                repeated_permed_recv_src_idx[:actual_gc_output_seqlen],
            )
        else:
            assert torch.equal(recv_lse, repeated_permed_recv_src_idx)

    if local_rank == 0:
        print(
            "\n# ------    Normal Intranode Dispatch Passed   ------ #\n",
            flush=True,
        )

    # --------------      test cached dispatch       -------------- #

    if local_rank == 0:
        print(
            "\n# ------    Test Intranode Cached Dispatch   ------ #\n",
            flush=True,
        )

    # Test cached dispatch
    cached_dispatch_args = common_dispatch_args | {
        "handle": handle,
        "recv_x": torch.empty_like(recv_x_gc_buf) if pass_out_buffer else None,
        "recv_lse": torch.empty_like(recv_lse_gc_buf)
        if cast_lse and pass_out_buffer
        else None,
    }
    if previous_mode:
        cached_dispatch_args.update({"previous_event": buffer.capture()})
    (
        recv_cached_x,
        recv_cached_lse,
        _,  # handle
        event,
    ) = buffer.group_cast(**cached_dispatch_args)
    event.current_stream_wait() if async_mode else ()

    # check recv_cache_x
    if pass_padded_out_buffer:
        assert recv_cached_x.size(0) > actual_gc_output_seqlen
        assert torch.equal(
            recv_cached_x[:actual_gc_output_seqlen], recv_x[:actual_gc_output_seqlen]
        )
    else:
        assert torch.equal(recv_cached_x, recv_x)

    # check recv_cache_lse
    if cast_lse:
        if pass_padded_out_buffer:
            assert recv_cached_lse.size(0) > actual_gc_output_seqlen
            assert torch.equal(
                recv_cached_lse[:actual_gc_output_seqlen],
                recv_lse[:actual_gc_output_seqlen],
            )
        else:
            assert torch.equal(recv_cached_lse, recv_lse)

    if local_rank == 0:
        print(
            "\n# ------    Intranode Cached Dispatch Passed   ------ #\n",
            flush=True,
        )

    # --------------      test normal combine       -------------- #

    if local_rank == 0:
        print(
            "\n# ------    Test Normal Intranode Combine   ------ #\n",
            flush=True,
        )

    # simulate gemm
    x_combine = sim_gemm(recv_x, w=sim_gemm_weight)
    lse_combine = recv_lse.clone() if reduce_op == "lse" else None

    # permute x/lse to the rank order
    if random_permute_output:
        if use_a2av_perm_idxs == "inside":
            # will permute inside
            pass
        else:
            x_combine_before_to_a2av = x_combine.clone()
            if use_a2av_perm_idxs == "outside":
                x_combine = x_combine[perm_to_a2av_idx]
            elif use_a2av_perm_idxs == "no":
                x_combine = unpermute_output(
                    output=x_combine,
                    unperm_after_a2a_kwargs=range_gather_pre_combine_kwargs,
                )
            assert x_combine_before_to_a2av.shape == x_combine.shape

            if reduce_op == "lse":
                lse_combine_before_to_a2av = lse_combine.clone()
                if use_a2av_perm_idxs == "outside":
                    lse_combine = lse_combine[perm_to_a2av_idx]
                elif use_a2av_perm_idxs == "no":
                    lse_combine = unpermute_output(
                        output=lse_combine,
                        unperm_after_a2a_kwargs=range_gather_pre_combine_kwargs,
                    )
                assert lse_combine_before_to_a2av.shape == lse_combine.shape

    # prepare combine args
    send_head_copy = send_head.clone()
    combine_args = {
        "x": x_combine,
        "reduced_x": combined_x_gr_buf,
        "handle": handle,
        "config": config,
        "async_finish": async_mode,
        "reduce_op": reduce_op,
        "acc_reduce": acc_reduce_out_buffer,
        # NOTE: still perm_to_a2av_idx, instead of unperm_to_a2av_idx
        "pre_perm_idx": perm_to_a2av_idx if use_a2av_perm_idxs == "inside" else None,
        "lse": lse_combine,
        "reduced_lse": combined_lse_gr_buf,
    }
    if previous_mode:
        combine_args.update({"previous_event": buffer.capture()})

    # combine
    # combined_x: shape=[num_tokens, hidden_size]:
    #   combined_x[i]: the ith token's sum-reduction result of top-k experts
    #   NOTE: the send_head will be modified in-place in intranode::cached_notify_combine
    #   for the entries == -1 to the position p of next valid token (encoded to -p-1)
    #   since the combine kernel needs to know the channel position when iterating at this token,
    #   even though it is not sent to the target rank
    (
        combined_x,
        combined_lse,
        event,
    ) = buffer.group_reduce(**combine_args)

    # wait
    event.current_stream_wait() if async_mode else ()

    # check combined_x in-place
    if pass_out_buffer:
        assert combined_x_gr_buf is not None
        assert combined_x_gr_buf.data_ptr() == combined_x.data_ptr()

    # check combined_lse in-place
    if reduce_op == "lse" and pass_out_lse_buffer:
        assert combined_lse_gr_buf is not None
        assert combined_lse_gr_buf.data_ptr() == combined_lse.data_ptr()

    # print
    combined_lse_shape = combined_lse.shape if reduce_op == "lse" else None
    print(
        (
            f"\n[RANK {rank}]: {combined_x.shape=} | {combined_x=}\n"
            f"{combined_lse_shape=} | {combined_lse=}\n"
            f"Before combine: {send_head.shape=} | {send_head=}\n\n"
        ),
        flush=True,
    )

    # check combined_x
    match x.dtype:
        case torch.bfloat16 | torch.float32 | torch.float64:
            torch.testing.assert_close(combined_x, combined_x_gr)
        case torch.float16:
            torch.testing.assert_close(combined_x, combined_x_gr, atol=1e-8, rtol=5e-3)
        case _:
            raise ValueError("Unsupported dtype")

    # check combined_lse
    if reduce_op == "lse":
        torch.testing.assert_close(combined_lse, combined_lse_gr)

    # check send head
    # cached_notify_combine will modify send_head in-place for any entry == -1
    assert torch.equal(
        send_head[send_head_copy != -1],
        send_head_copy[send_head_copy != -1],
    )

    # specific check for combined_x for specific reduce op
    if reduce_op in ("sum", "avg"):
        send_token_nums = is_token_in_rank.sum(dim=1).unsqueeze(1)
        check_x = combined_x.float()
        ref_x = sim_gemm(x.float(), w=sim_gemm_weight) * send_token_nums

        # if acc_reduce, the combined token should add with a constant rank bias
        if acc_reduce_out_buffer:
            ref_x += acc_reduce_constant

        if reduce_op == "avg":
            ref_x /= send_token_nums + (1 if acc_reduce_out_buffer else 0)

        # if some token is not sent to any rank, the combined token should be 0 or acc_reduce_constant
        if min_num_dst_ranks == 0:
            zero_num_dst_ranks_mask_x = (send_token_nums == 0.0).expand_as(combined_x)
            check_x[zero_num_dst_ranks_mask_x] = (
                acc_reduce_constant if acc_reduce_out_buffer else 0.0
            )
            ref_x[zero_num_dst_ranks_mask_x] = (
                acc_reduce_constant if acc_reduce_out_buffer else 0.0
            )

        diff_x = calc_diff(check_x, ref_x)
        assert diff_x < 5e-6, f"{check_x=} != {ref_x=} with ({diff_x=})"

    if local_rank == 0:
        print(
            "\n# ------    Normal Intranode Combine Passed  ------ #\n",
            flush=True,
        )

    # For later tuning
    combine_nvl_send_bytes = dispatch_nvl_recv_bytes = (
        recv_x_gc.numel() * recv_x_gc.dtype.itemsize
    )

    return dict(
        handle=handle,
        dispatch_nvl_recv_bytes=dispatch_nvl_recv_bytes,
        combine_nvl_send_bytes=combine_nvl_send_bytes,
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
    num_tokens, hidden_size = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_channels = num_sms // 2
    num_heads = 16
    assert hidden_size % num_heads == 0

    # we can assume num_local_experts == 1
    # thus sending one token to one rank is equivalent to sending to the only one "local expert" in that rank
    num_experts = num_ranks
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    assert num_local_experts == 1

    num_max_nvl_chunked_send_tokens = 8
    nvl_buffer_size = num_max_nvl_chunked_recv_tokens = 256

    # choose dtype from {torch.bfloat16, torch.float16, torch.float32, torch.float64}
    dtype = torch.bfloat16  # TODO: make it parameterizable
    assert dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64)

    # Remake the hidden size to control
    # the communication bytes per token the same as bf16/fp16
    hidden_size = hidden_size * 2 // dtype.itemsize

    # Re-Settings for group-collective
    # TODO: make these parameterizable
    num_topk = num_ranks  # we can assume num_topk == num_ranks
    num_input_splits = 10
    distinct_token = True
    random_permute_output = True  # set to False to make the output / input of group-cast / group-reduce in a2a rank order
    sim_gemm_weight = 2.0
    min_num_dst_ranks = 0

    cast_lse = True
    reduce_op: GroupReduceOp = "lse"  # choose from {"sum", "avg", "lse"}
    if reduce_op == "lse":
        assert cast_lse, "we need to cast lse first before reducing"

    pass_out_buffer = True  # for both dispatch and combine
    pass_out_lse_buffer = True  # for both dispatch and combine
    pass_padded_out_buffer = False  # for dispatch output and combine input

    acc_reduce_out_buffer = True
    acc_reduce_constant = rank
    if acc_reduce_out_buffer:
        assert pass_out_buffer, "acc_reduce_out_buffer requires pass_out_buffer"

    # choose from {"no", "outside", "inside"}
    use_a2av_perm_idxs = "inside"
    assert use_a2av_perm_idxs in ("no", "outside", "inside")
    if (
        pass_padded_out_buffer
    ):  # we only allow inside usage when passing padded out buffer
        assert use_a2av_perm_idxs == "inside"

    # Config
    config = GrpCollConfig(
        num_sms=num_sms,  # num_sms, default 20
        nvl_chunk_size=num_max_nvl_chunked_send_tokens,  # num_max_nvl_chunked_send_tokens (nvl_chunk_size), default 6
        nvl_buffer_size=num_max_nvl_chunked_recv_tokens,  # num_max_nvl_chunked_recv_tokens (nvl_buffer_size), default 256
        # num_max_rdma_chunked_send_tokens, default 6
        # num_max_rdma_chunked_recv_tokens, default 256
    )
    min_num_nvl_bytes = GrpCollConfig.get_min_num_nvl_bytes(
        num_sms=num_sms,
        num_ranks=num_ranks,
        hidden_size=hidden_size,
        nvl_buffer_size=nvl_buffer_size,
        dtype=dtype,
        transfer_lse=cast_lse or reduce_op == "lse",
        num_heads=num_heads,
    )

    # print settings
    if local_rank == 0:
        print(
            (
                f"[config] {num_sms=} | {num_channels=} | {min_num_nvl_bytes=} ({min_num_nvl_bytes / 1024**2:.2f} MB)\n"
                f"{num_experts=} | {num_tokens=} | {hidden_size=} | {dtype=} | "
                f"{num_topk=} | {num_local_experts=} | {num_heads=}\n"
                f"{nvl_buffer_size=} | {num_max_nvl_chunked_send_tokens=} | {num_max_nvl_chunked_recv_tokens=}\n"
                f"{distinct_token=} | {random_permute_output=} | {sim_gemm_weight=} | {min_num_dst_ranks=}\n"
                f"{cast_lse=} | {reduce_op=}\n"
                f"{pass_out_buffer=} | {pass_out_lse_buffer=} | {pass_padded_out_buffer=}\n"
                f"{acc_reduce_out_buffer=} | {acc_reduce_constant=} | {use_a2av_perm_idxs=}\n"
            ),
            flush=True,
        )

    # prepare test kwargs
    test_kwargs = prepare_test_func_kwargs(
        rank=rank,
        local_rank=local_rank,
        num_ranks=num_ranks,
        group=group,
        buffer=buffer,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_input_splits=num_input_splits,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        dtype=dtype,
        distinct_token=distinct_token,
        cast_lse=cast_lse,
        reduce_op=reduce_op,
        pass_out_buffer=pass_out_buffer,
        pass_out_lse_buffer=pass_out_lse_buffer,
        pass_padded_out_buffer=pass_padded_out_buffer,
        random_permute_output=random_permute_output,
        sim_gemm_weight=sim_gemm_weight,
        acc_reduce_out_buffer=acc_reduce_out_buffer,
        acc_reduce_constant=acc_reduce_constant,
        min_num_dst_ranks=min_num_dst_ranks,
    )

    # test dispatch / combine
    test_out = test_func(
        rank=rank,
        local_rank=local_rank,
        config=config,
        buffer=buffer,
        # test parameters
        async_mode=True,
        previous_mode=True,
        cast_lse=cast_lse,
        reduce_op=reduce_op,
        pass_out_buffer=pass_out_buffer,
        pass_out_lse_buffer=pass_out_lse_buffer,
        pass_padded_out_buffer=pass_padded_out_buffer,
        random_permute_output=random_permute_output,
        use_a2av_perm_idxs=use_a2av_perm_idxs,
        sim_gemm_weight=sim_gemm_weight,
        acc_reduce_out_buffer=acc_reduce_out_buffer,
        acc_reduce_constant=acc_reduce_constant,
        min_num_dst_ranks=min_num_dst_ranks,
        # kwargs
        **test_kwargs,
    )

    # fetch some constant test kwargs for later usage
    x = test_kwargs["x"]
    num_tokens_per_rank = test_kwargs["num_tokens_per_rank"]
    is_token_in_rank = test_kwargs["is_token_in_rank"]

    # fetch some constant test out for later usage
    handle = test_out["handle"]
    dispatch_nvl_recv_bytes = test_out["dispatch_nvl_recv_bytes"]
    combine_nvl_send_bytes = test_out["combine_nvl_send_bytes"]

    # --------------      tune dispatch       -------------- #

    if local_rank == 0:
        print(
            "\n# ------    Tune Intranode Dispatch   ------ #\n",
            flush=True,
        )

    best_dispatch_results = None
    best_time, best_results = 1e10, None
    nvl_recv_bytes = dispatch_nvl_recv_bytes
    for nvl_chunk_size in tuple(range(4, 33, 2)) + (0,):
        if nvl_chunk_size > 0:
            config = GrpCollConfig(
                num_sms=num_sms,
                nvl_chunk_size=nvl_chunk_size,
                nvl_buffer_size=nvl_buffer_size,
            )
        else:  # Test default config as well
            config = GrpCollConfig.get_default_group_cast_config(num_ranks)
        tune_args = {
            "x": x,
            "handle": handle,
            "config": config,
        }  # TODO: add other flags to tune args
        t = bench(lambda: buffer.group_cast(**tune_args))[0]
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
            f'({"FP8" if isinstance(x, tuple) else "BF16"}): '
            f"SMs {best_results[0]}, NVL chunk {best_results[1]}, "
            f"{nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), "
            f"t: {best_time * 1e6:.2f} us",
            flush=True,
        )
        print("", flush=True)

    # Gather the best config from rank 0 and the first test setting
    if best_dispatch_results is None:
        best_dispatch_results = torch.tensor(
            [best_results[0], best_results[1]],
            dtype=torch.int32,
            device="cuda",
        )
        all_best_results_list = [
            torch.zeros_like(best_dispatch_results)
            for _ in range(torch.distributed.get_world_size())
        ]
        dist.all_gather(all_best_results_list, best_dispatch_results, group=group)
        best_dispatch_results = all_best_results_list[0].tolist()

    # apply dispatch to get handle before combine
    dispatch_config = GrpCollConfig(
        num_sms=best_dispatch_results[0],
        nvl_chunk_size=best_dispatch_results[1],
        nvl_buffer_size=nvl_buffer_size,
    )
    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
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

    # --------------      tune combine       -------------- #

    if local_rank == 0:
        print(
            "\n# ------    Tune Intranode Combine   ------ #\n",
            flush=True,
        )

    best_time, best_results = 1e10, None
    combined_x_buf = torch.zeros_like(x) if pass_out_buffer else None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0,):
        if nvl_chunk_size > 0:
            config = GrpCollConfig(
                num_sms=num_sms,
                nvl_chunk_size=nvl_chunk_size,
                nvl_buffer_size=nvl_buffer_size,
            )
        else:  # Test default config as well
            config = GrpCollConfig.get_default_group_reduce_config(num_ranks)
        tune_args = {
            "x": recv_x,
            "reduced_x": combined_x_buf,
            "handle": handle,
            "config": config,
            "reduce_op": "sum",
            "acc_reduce": acc_reduce_out_buffer,
        }
        t = bench(lambda: buffer.group_reduce(**tune_args))[0]
        if local_rank == 0:
            print(
                f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                f"{combine_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                flush=True,
            )
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

    if local_rank == 0:
        print(
            f"[tuning] Best combine: SMs {best_results[0]}, "
            f"NVL chunk {best_results[1]}: "
            f"{combine_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), "
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

    num_nvl_bytes = int(2e9)  # 2GB
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
