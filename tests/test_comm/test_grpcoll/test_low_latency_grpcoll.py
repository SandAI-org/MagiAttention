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
from functools import partial

import torch
import torch.distributed as dist

from magi_attention.comm.primitive.grpcoll import (
    group_cast_collective,
    group_reduce_collective,
)
from magi_attention.comm.primitive.grpcoll._buffer import Buffer
from magi_attention.testing.grpcoll_utils import (
    bench,
    bench_kineto,
    calc_diff,
    get_output_split_size_list_and_src_index_list,
    get_random_dst_indices_list,
    get_random_split_size_list,
    hash_tensor,
    init_dist,
    per_token_cast_back,
    transfer_group_cast_meta_to_dispatch_meta,
)


def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    use_logfmt: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    # Random data
    # x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    # x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = (
        torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.1
    )
    print(f"[RANK {rank}] {x=} | {x.shape=}\n", flush=True)

    # Random scores
    # scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    # topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    # topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()
    num_input_splits = 4
    input_split_size_list = get_random_split_size_list(num_tokens, num_input_splits)
    dst_indices_list = get_random_dst_indices_list(
        num_input_splits, num_ranks, min_num_dst_ranks=num_ranks
    )  # HACK: since low-latency is expert-level, we hackly let all token broadcast to all ranks
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
        _,
        _,
        _,
        _,
        _,
        topk_idx,
        topk_weights,
        _,
    ) = transfer_group_cast_meta_to_dispatch_meta(
        rank,
        num_ranks,
        num_local_experts,
        input_split_size_list,
        dst_indices_list,
        device="cuda",
        use_topk=True,
    )
    topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    )  # HACK: since combine kernel will multiply by weights, we hackly let all weights to be 1
    print(
        f"[RANK {rank}]: {input_split_size_list=} | {dst_indices_list=} | "
        f"{output_split_size_list=} | {src_index_list=} | "
        f"{sum(output_split_size_list)=}\n",
        flush=True,
    )
    print(f"[RANK {rank}]: {topk_idx=} | {topk_weights=}\n", flush=True)

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for current_x in (x,):  # (x, x_pure_rand):
        for return_recv_hook in (True,):  # (False, True):
            for dispatch_use_fp8 in (False,):  # (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False,):
                    for use_ue8m0 in (False, True) if round_scale else (False,):
                        num_times += 1
                        for i in range(1):  # range((num_times % 2) + 1):
                            if rank == 0:
                                print(
                                    "\n# ------    Test Low Latency Dispatch   ------ #\n",
                                    flush=True,
                                )

                            # prepare
                            cumulative_local_expert_recv_stats = torch.zeros(
                                (num_local_experts,), dtype=torch.int, device="cuda"
                            )

                            # dispatch
                            # packed_recv_x: shape=[num_local_experts, num_max_recv_tokens, hidden]:
                            #   the recv tokens of all local experts
                            #   with a buffer size of num_max_recv_tokens to avoid cpu-gpu sync
                            #   thus not all tokens in packed_recv_x are valid
                            #   (the number of valid tokens is indicated in packed_recv_count)
                            # packed_recv_count: shape=[num_local_experts,]:
                            #   how many tokens are received for each local expert
                            #   thus packed_recv_count[e, :packed_recv_count[e], :]
                            #   are valid for local expert e
                            # handle[0] (packed_recv_src_info): shape=[num_local_experts, num_max_recv_tokens]:
                            #   the token idx in the sender's buffer for each recv token for local expert e
                            #   and only the valid tokens in packed_recv_src_info[e, :packed_recv_count[e]]
                            #   are valid (non-valid entries are empty numbers)
                            # handle[1] (packed_recv_layout_range): shape=[num_local_experts, num_ranks]:
                            #   the recv range from each global expert (expert id = rank id * local expert id)
                            #   where a single recv range is a int64 number,
                            #   which is actually packed from an int-tuple of (num_recv_tokens, recv_token_begin_idx)
                            #   where the recv_token_begin_idx is the token idx in the recv buffer,
                            #   ranging in [0, num_max_recv_tokens)
                            # handle[2] (num_max_dispatch_tokens_per_rank)
                            # handle[3] (hidden_size)
                            # handle[4] (num_experts)
                            # cumulative_local_expert_recv_stats: shape=[num_local_experts,]:
                            #   the same as packed_recv_count, TODO: so why need this ?
                            (
                                packed_recv_x,
                                packed_recv_count,
                                handle,
                                event,
                                hook,
                            ) = buffer.low_latency_dispatch(
                                current_x,
                                topk_idx,
                                num_tokens,
                                num_experts,
                                use_fp8=dispatch_use_fp8,
                                round_scale=round_scale,
                                use_ue8m0=use_ue8m0,
                                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                async_finish=not return_recv_hook,
                                return_recv_hook=return_recv_hook,
                            )

                            # wait (if return_recv_hook, then the hook will be called to
                            # apply the receive stage to wait for all data to be received)
                            hook() if return_recv_hook else event.current_stream_wait()

                        # cast
                        packed_recv_x = (
                            (packed_recv_x[0], packed_recv_x[1].contiguous())
                            if dispatch_use_fp8
                            else packed_recv_x
                        )
                        simulated_gemm_x = (
                            per_token_cast_back(
                                packed_recv_x[0].view(-1, hidden),
                                packed_recv_x[1].view(-1, hidden // 128),
                            ).view(packed_recv_x[0].shape)
                            if dispatch_use_fp8
                            else packed_recv_x.clone()  # type: ignore[attr-defined]
                        )

                        # unpack
                        (
                            packed_recv_src_info,
                            packed_recv_layout_range,
                            num_max_dispatch_tokens_per_rank,
                            hidden_size,
                            num_experts,
                        ) = handle

                        recv_x = torch.sort(packed_recv_x[0], dim=0)[
                            0
                        ]  # HACK: only local expert 0 activated

                        # print
                        print(
                            (
                                f"[RANK {rank}]: {recv_x.shape=} | {recv_x=}\n\n"
                                f"{packed_recv_x=} | {packed_recv_x.shape=}\n"  # type: ignore[attr-defined]
                                f"{packed_recv_count=} | {sum(packed_recv_count)=} | {packed_recv_count.shape=}\n"
                                f"{packed_recv_src_info=} | {packed_recv_src_info.shape=}\n"
                                f"{packed_recv_layout_range=} | {packed_recv_layout_range.shape=}\n"
                                f"{num_max_dispatch_tokens_per_rank=} | {hidden_size=} | {num_experts=}\n"
                                f"{cumulative_local_expert_recv_stats=}\n\n"
                            ),
                            flush=True,
                        )

                        # check
                        assert torch.equal(recv_x, recv_x_gc)
                        assert torch.equal(
                            cumulative_local_expert_recv_stats, packed_recv_count
                        )
                        all_topk_idx = torch.empty(
                            (num_ranks, num_tokens, num_topk),
                            dtype=topk_idx.dtype,  # type: ignore[union-attr]
                            device="cuda",
                        )
                        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = (
                                per_token_cast_back(
                                    packed_recv_x[0][i], packed_recv_x[1][i]
                                )
                                if dispatch_use_fp8
                                else packed_recv_x[i]
                            )
                            recv_count, recv_src_info, recv_layout_range = (
                                packed_recv_count[i],
                                handle[0][i],
                                handle[1][i],
                            )

                            # Check expert indices
                            int_mask = (2**32) - 1
                            num_valid_tokens = recv_count.item()
                            assert (
                                cumulative_local_expert_recv_stats[i].item()
                                == num_valid_tokens
                            ), f"{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}"
                            assert (
                                num_valid_tokens
                                == (recv_layout_range & int_mask).sum().item()
                            ), f"{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()"
                            assert (
                                num_valid_tokens
                                == (all_topk_idx == expert_id).sum().item()
                            ), f"{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}"

                            # Check received data
                            if current_x is not x_pure_rand:
                                recv_x = recv_x[:num_valid_tokens]
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                recv_src_info = recv_src_info[:num_valid_tokens]
                                assert torch.equal(
                                    recv_x_amin, recv_x[:, :-128].amax(dim=-1)
                                )
                                # if round_scale:
                                #     assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
                                # else:
                                #     assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                                # for j in range(num_ranks):
                                #     begin_idx, count = (recv_layout_range[j] >> 32).item(), \
                                #       (recv_layout_range[j] & int_mask).item()
                                #     if not round_scale:
                                #         assert (recv_x_amin == j - rank_offset).sum().item() == \
                                #       (all_topk_idx[j] == expert_id).sum().item()
                                #     assert (recv_x[begin_idx:begin_idx + count][:-128] - j).sum().item() == 0
                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(
                                    packed_recv_x[0][i, :num_valid_tokens]
                                )
                                hash_value ^= hash_tensor(
                                    packed_recv_x[1][i, :num_valid_tokens]
                                )
                            else:
                                hash_value ^= hash_tensor(
                                    packed_recv_x[i, :num_valid_tokens]  # type: ignore[call-overload]
                                )

                        if rank == 0:
                            print(
                                "\n# ------    Test Low Latency Combine   ------ #\n",
                                flush=True,
                            )

                        # Check combine correctness
                        for zero_copy in (
                            True,
                        ):  # (False, ) if use_logfmt else (False, True):
                            # prepare
                            if zero_copy:
                                # if zero_copy, then we need to copy the data into the RDMA buffer outside the kernel
                                buffer.get_next_low_latency_combine_buffer(handle)[
                                    :, :, :
                                ] = simulated_gemm_x
                            # prepare the combined_x buffer outside the kernel
                            out = torch.empty(
                                (num_tokens, hidden),
                                dtype=torch.bfloat16,
                                device="cuda",
                            )

                            # combine
                            # combined_x: shape=[num_tokens, hidden]
                            combined_x, event, hook = buffer.low_latency_combine(
                                simulated_gemm_x,
                                topk_idx,
                                topk_weights,
                                handle,
                                use_logfmt=use_logfmt,
                                async_finish=not return_recv_hook,
                                zero_copy=zero_copy,
                                return_recv_hook=return_recv_hook,
                                out=out,
                            )

                            # wait (if return_recv_hook, then the hook will be called to apply
                            # the receive stage to wait for all data to be received and reduced)
                            hook() if return_recv_hook else event.current_stream_wait()

                            # print
                            print(
                                (
                                    f"[RANK {rank}] {combined_x=} | {combined_x.shape=}\n"
                                    f"{event=} | {hook=}\n\n"
                                ),
                                flush=True,
                            )

                            # checks
                            if do_check:
                                assert torch.equal(combined_x, combined_x_gr)
                                assert torch.equal(out, combined_x)
                                assert torch.equal(simulated_gemm_x, packed_recv_x)
                                diff = calc_diff(
                                    current_x
                                    * topk_weights.masked_fill(topk_idx == -1, 0)
                                    .sum(dim=1)
                                    .view(-1, 1),
                                    combined_x,
                                )
                                assert torch.isnan(combined_x).sum().item() == 0
                                assert diff < (
                                    7e-4 if dispatch_use_fp8 else 1e-5
                                ), f"Error: {diff=}, {zero_copy=}"
                                hash_value ^= hash_tensor(combined_x)

    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x_pure_rand,
            topk_idx,
            num_tokens,
            num_experts,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            use_fp8=True,
            async_finish=False,
            return_recv_hook=return_recv_hook,
        )
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            use_logfmt=use_logfmt,
            return_recv_hook=return_recv_hook,
        )
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()  # type: ignore[index]
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(
        f"[rank {rank}] Dispatch + combine bandwidth: "
        f"{(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, "
        f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, "
        f"max_t={max_t * 1e6:.2f} us",
        flush=True,
    )

    # Separate profiling
    for return_recv_hook in (False,):  # (False, True):
        group.barrier()
        dispatch_t, combine_t = bench_kineto(
            partial(test_func, return_recv_hook=return_recv_hook),
            kernel_names=("dispatch", "combine"),
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
            num_kernels_per_period=2 if return_recv_hook else 1,
        )
        if not return_recv_hook:
            print(
                f"[rank {rank}] Dispatch bandwidth: "
                f"{num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, "
                f"avg_t={dispatch_t * 1e6:.2f} us | "
                f"Combine bandwidth: "
                f"{num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, "
                f"avg_t={combine_t * 1e6:.2f} us",
                flush=True,
            )
        else:
            print(
                f"[rank {rank}] Dispatch send/recv time: "
                f"{dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | "
                f"Combine send/recv time: "
                f"{combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us",
                flush=True,
            )
    return hash_value


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # init dist
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    num_tokens, hidden = args.num_tokens, args.hidden
    num_max_recv_tokens = num_ranks * num_ranks
    num_topk, num_experts = args.num_topk, args.num_experts

    # reset for group-collective
    num_topk = num_ranks
    num_experts = num_ranks * num_ranks

    assert num_topk <= 9  # kNumMaxTopK = 9
    num_local_experts = num_experts // num_ranks
    num_qps_per_rank = num_local_experts
    allow_nvlink = os.environ.get("GRPCOLL_TEST_LOW_LATENCY_ALLOW_NVLINK", "1") == "1"

    num_device_sms = 132  # for Hopper
    num_warp_groups = (num_experts + num_device_sms - 1) // num_device_sms
    num_warps_per_group = 32 // num_warp_groups
    num_warps = num_warp_groups * num_warps_per_group
    assert num_warps <= 32  # kNumMaxWarpGroups = 32
    num_sms = (num_experts + num_warp_groups - 1) // num_warp_groups

    num_nvl_bytes = 0
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    if local_rank == 0:
        print(
            (
                f"[config] {num_nvl_bytes=} | {num_rdma_bytes=} ({num_rdma_bytes / 1e9:.2f} GB) | "
                f"{num_ranks=} | {num_tokens=} (num_max_dispatch_tokens_per_rank) | {num_max_recv_tokens=} | "
                f"{group.size()=} | {hidden=} |"
                f" {num_topk=} | {num_experts=} | {num_local_experts=} | "
                f"{num_qps_per_rank=} | {allow_nvlink=}\n\n"
            ),
            flush=True,
        )

        print(
            (
                f"[kernel config] {num_device_sms=} | {num_warp_groups=} | "
                f"{num_warps_per_group=} | {num_warps=} | {num_sms=}\n\n"
            ),
            flush=True,
        )

    buffer = Buffer(
        group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_qps_per_rank,
        allow_nvlink_for_low_latency_mode=allow_nvlink,
        explicitly_destroy=True,
    )
    test_main(
        num_tokens,
        hidden,
        num_experts,
        num_topk,
        rank,
        num_ranks,
        group,
        buffer,
        use_logfmt=args.use_logfmt,
        seed=0,
    )

    do_pressure_test = args.pressure_test
    for seed in range(int(1e9) if do_pressure_test else 0):
        if local_rank == 0:
            print(f"Testing with seed {seed} ...", flush=True)
        ref_hash = test_main(
            num_tokens,
            hidden,
            num_experts,
            num_topk,
            rank,
            num_ranks,
            group,
            buffer,
            use_logfmt=args.use_logfmt,
            seed=seed,
        )
        for i in range(20):
            assert (
                test_main(
                    num_tokens,
                    hidden,
                    num_experts,
                    num_topk,
                    rank,
                    num_ranks,
                    group,
                    buffer,
                    use_logfmt=args.use_logfmt,
                    seed=seed,
                )
                == ref_hash
            ), f"Error: seed={seed}"

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description="Test low-latency EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=128, help="Number of tokens (default: 128)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=288, help="Number of experts (default: 288)"
    )
    parser.add_argument(
        "--disable-nvlink",
        action="store_true",
        help="Whether to disable NVLink for testing",
    )
    parser.add_argument(
        "--use-logfmt", action="store_true", help="Whether to test LogFMT combine"
    )
    parser.add_argument(
        "--pressure-test", action="store_true", help="Whether to do pressure test"
    )
    args = parser.parse_args()

    # disable pressure test
    args.pressure_test = False

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
