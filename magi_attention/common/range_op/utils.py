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

import torch

from ..ranges import NaiveRanges


def _calc_cu_range_sizes(
    ranges: torch.Tensor | NaiveRanges,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    if isinstance(ranges, torch.Tensor):
        ranges = ranges.tolist()

    cu_range_sizes = [0]
    total_size = 0
    for start, end in ranges:
        total_size += end - start
        cu_range_sizes.append(total_size)

    cu_range_sizes = torch.tensor(cu_range_sizes, dtype=torch.int64, device=device)

    return cu_range_sizes, total_size


def _calc_ranges_row_map(
    ranges: torch.Tensor,
    total_size: int,
) -> torch.Tensor:
    if ranges.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64, device=ranges.device)

    row_map = torch.arange(0, ranges.shape[0], device=ranges.device)
    range_sizes = ranges[:, 1] - ranges[:, 0]
    row_map = torch.repeat_interleave(
        row_map, range_sizes, dim=0, output_size=total_size
    )

    return row_map


def _calc_out2inp_range_map(
    output_ranges: torch.Tensor | NaiveRanges,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if isinstance(output_ranges, torch.Tensor):
        output_ranges = output_ranges.tolist()

    # convert to list of tuple to be hashable
    output_ranges = list(map(tuple, output_ranges))

    unique_ordered_out_ranges = sorted(set(output_ranges))

    out_range2inp_indices_map: dict[tuple[int, int], list[int]] = {}
    for inp_idx, out_range in enumerate(list(output_ranges)):
        out_range2inp_indices_map.setdefault(out_range, []).append(inp_idx)

    max_inp_indices_size = (
        max([len(inp_indices) for inp_indices in out_range2inp_indices_map.values()])
        if out_range2inp_indices_map
        else 0
    )

    out2inp_range_map = []
    for out_range in unique_ordered_out_ranges:
        inp_range_list = out_range2inp_indices_map[out_range]
        inp_range_list = inp_range_list + [-1] * (
            max_inp_indices_size - len(inp_range_list)
        )
        out2inp_range_map.append(inp_range_list)

    out2inp_range_map = torch.tensor(
        out2inp_range_map, dtype=torch.int64, device=device
    )
    unique_ordered_out_ranges = torch.tensor(
        unique_ordered_out_ranges, dtype=torch.int64, device=device
    )

    return out2inp_range_map, unique_ordered_out_ranges, max_inp_indices_size
