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


import inspect
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist


def get_random_split_size_list(
    total_seqlen: int,
    num_splits: int,
) -> list[int]:
    cu_seqlens = (
        [0]
        + sorted(random.sample(range(1, total_seqlen - 1), num_splits - 1))
        + [total_seqlen]
    )
    seqlens = torch.tensor(cu_seqlens, dtype=torch.int).diff().tolist()
    return seqlens


def get_random_dst_indices_list(
    num_splits: int,
    num_ranks: int,
    min_num_dst_ranks: int = 0,
    max_num_dst_ranks: int | None = None,
) -> list[list[int]]:
    dst_indices_list: list[list[int]] = [[] for _ in range(num_splits)]
    num_dst_ranks_per_split = torch.randint(
        min_num_dst_ranks,
        (num_ranks + 1) if max_num_dst_ranks is None else (max_num_dst_ranks + 1),
        (num_splits,),
    ).tolist()

    for dst_indices, num_dst_ranks in zip(dst_indices_list, num_dst_ranks_per_split):
        dst_indices.extend(sorted(random.sample(range(num_ranks), num_dst_ranks)))
    return dst_indices_list


def get_output_split_size_list_and_src_index_list(
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    group: dist.ProcessGroup,
) -> tuple[list[int], list[int]]:
    my_rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    input_split_size_list_per_rank = [None] * world_size
    dst_indices_list_per_rank = [None] * world_size
    dist.all_gather_object(
        input_split_size_list_per_rank, input_split_size_list, group=group
    )
    dist.all_gather_object(dst_indices_list_per_rank, dst_indices_list, group=group)

    output_split_size_list, src_index_list = [], []

    output_src_rank_map: dict[int, list[int]] = {}
    for src_rank in range(world_size):
        input_split_size_list_this_rank = input_split_size_list_per_rank[src_rank]
        dst_indices_list_this_rank = dst_indices_list_per_rank[src_rank]
        assert len(input_split_size_list_this_rank) == len(dst_indices_list_this_rank)  # type: ignore[arg-type]

        for input_split_size, dst_indices in zip(  # type: ignore[call-overload]
            input_split_size_list_this_rank, dst_indices_list_this_rank
        ):
            if my_rank in dst_indices:
                output_src_rank_map.setdefault(src_rank, []).append(input_split_size)

    for src_rank in range(world_size):
        split_sizes = output_src_rank_map.get(src_rank, [])
        if split_sizes:
            output_split_size_list.extend(split_sizes)
            src_index_list.extend([src_rank] * len(split_sizes))

    return output_split_size_list, src_index_list


def transfer_group_cast_meta_to_dispatch_meta(
    rank: int,
    num_ranks: int,
    num_local_experts: int,
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    num_nodes: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
    use_topk: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    num_local_ranks = num_ranks // num_nodes
    num_experts = num_local_experts * num_ranks
    num_tokens = sum(input_split_size_list)
    num_splits = len(input_split_size_list)
    input_split_size_list_tensor = torch.tensor(
        input_split_size_list, dtype=dtype, device=device
    )

    if use_topk:
        assert num_local_experts == num_ranks
        topk_idx = torch.full(
            (num_splits, num_ranks), fill_value=-1, dtype=dtype, device="cpu"
        )
        # no other meanings, just placeholders,
        # since the topk weights are used outside of grpcoll intranode/internode kernels
        # and for low-latency kernels, due to the incompatibility, we hackly set pure-one topk-weights outside
        topk_weights = (
            torch.ones((num_tokens, num_ranks), dtype=torch.float32, device=device)
            * rank
        )
        for split_idx in range(num_splits):
            num_dst_ranks = len(dst_indices_list[split_idx])
            assert (
                num_dst_ranks > 0
            ), "For now, we only support non-empty dst_indices_list"
            num_dst_local_experts = num_ranks // num_dst_ranks
            num_last_dst_local_experts = num_ranks - num_dst_local_experts * (
                num_dst_ranks - 1
            )
            is_last_dst_rank = (  # noqa: E731
                lambda r: r == dst_indices_list[split_idx][-1]
            )

            start = 0
            for dst_rank in dst_indices_list[split_idx]:
                num = (
                    num_last_dst_local_experts
                    if is_last_dst_rank(dst_rank)
                    else num_dst_local_experts
                )
                end = start + num
                topk_idx[split_idx][start:end] = dst_rank * num_ranks + torch.arange(
                    num, dtype=dtype
                )
                start = end
        topk_idx = topk_idx.to(device).repeat_interleave(
            input_split_size_list_tensor, dim=0, output_size=num_tokens
        )  # shape=(num_tokens, num_ranks)
    else:
        assert num_local_experts == 1
        topk_idx, topk_weights = None, None

    # construct rank_idx
    rank_idx = torch.full(
        (num_splits, num_ranks), fill_value=-1, dtype=dtype, device="cpu"
    )
    for split_idx in range(num_splits):
        num_dst_ranks = len(dst_indices_list[split_idx])
        assert num_dst_ranks > 0, "For now, we only support non-empty dst_indices_list"
        rank_idx[split_idx, :num_dst_ranks] = torch.tensor(
            sorted(dst_indices_list[split_idx], reverse=True)
        )
    rank_idx = rank_idx.to(device).repeat_interleave(
        input_split_size_list_tensor, dim=0, output_size=num_tokens
    )  # shape=(num_tokens, num_ranks)

    # construct rdma_rank_idx
    if num_nodes > 1:
        rdma_rank_idx = rank_idx // num_local_ranks
        rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
        inplace_unique(rdma_rank_idx, num_nodes)
    else:
        rdma_rank_idx = None

    # construct rank layout meta
    # num_tokens_per_rank[r]: the number of tokens sent to rank r by this rank
    # num_tokens_per_rdma_rank[r]: the number of tokens sent to RDMA rank r by this rank
    # is_token_in_rank[j][r]: whether jth token is sent to rank r
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

    if num_nodes > 1:
        num_tokens_per_rdma_rank = torch.empty(
            (num_nodes,), dtype=torch.int, device="cuda"
        )
        for i in range(num_nodes):
            num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    else:
        num_tokens_per_rdma_rank = None

    # construct expert meta
    # num_tokens_per_expert[e]: the number of tokens sent to expert e by this rank
    if use_topk:
        num_tokens_per_expert = torch.zeros(
            (num_experts,), dtype=torch.int, device="cuda"
        )
        for i in range(num_experts):
            num_tokens_per_expert[i] = (topk_idx == i).sum()
    else:
        num_tokens_per_expert = num_tokens_per_rank

    return (
        rank_idx,
        rdma_rank_idx,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        is_token_in_rank,
        topk_idx,
        topk_weights,
        num_tokens_per_expert,
    )


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_local_ranks * num_nodes))),
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(
    scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int
):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        ) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device="cuda"))
                for _ in range(num_tests):
                    fn()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        prof.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert (
            sum([name in line for line in prof_lines]) == 1
        ), f"Errors of the kernel {name} in the profiling table"

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(
                            float(time_str.replace(unit, "")) / scale
                        )
                        break
                break

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [
                event
                for event in profile_data["traceEvents"]
                if f"::{kernel_name}" in event["name"]
            ]
            events = sorted(events, key=lambda event: event["ts"])
            durations = [event["dur"] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [  # type: ignore[call-overload]
                sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                for j in range(num_kernels_per_period)
            ]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int64).sum().item()
