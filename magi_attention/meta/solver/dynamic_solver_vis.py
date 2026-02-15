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

from typing import Sequence

import matplotlib.pyplot as plt

from magi_attention.common.rectangles import AttnRectangles


def _iter_all_rects(bucket_per_rank: Sequence[AttnRectangles]):
    for rank, rects in enumerate(bucket_per_rank):
        # AttnRectangles is iterable and yields AttnRectangle
        for rect in rects:
            yield rank, rect


def visualize_buckets(
    bucket_per_rank: Sequence[AttnRectangles],
    title: str = "DynamicAttnSolver buckets",
    max_size: int | None = None,
    save_path: str | None = None,
    before_dispatch: bool = False,
) -> None:
    """
    Visualize attention rectangles for each rank.

    For each `AttnRectangle`, draw one rectangle:
    - Horizontal axis: k (from k.start to k.end)
    - Vertical axis: q (from q.start to q.end)

    Note:
        q increases downwards, so the q-axis needs to be inverted (smaller at the top, larger at the bottom).
    """

    if not bucket_per_rank:
        return

    # Prepare figure
    # Slightly wider to leave space for the legend on the right
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use different colors for different ranks
    num_ranks = len(bucket_per_rank)
    cmap = plt.get_cmap("tab20")

    q_min = None
    q_max = None
    k_min = None
    k_max = None

    for rank, rect in _iter_all_rects(bucket_per_rank):
        q_range = rect.q_range
        k_range = rect.k_range

        # q: vertical (y), k: horizontal (x)
        x = k_range.start
        y = q_range.start
        width = k_range.end - k_range.start
        height = q_range.end - q_range.start

        if width <= 0 or height <= 0:
            continue

        color = cmap(rank % cmap.N)

        rect_patch = plt.Rectangle(
            (x, y),
            width,
            height,
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.8,
        )
        ax.add_patch(rect_patch)

        # Track global bounds
        q_min = y if q_min is None else min(q_min, y)
        q_max = y + height if q_max is None else max(q_max, y + height)
        k_min = x if k_min is None else min(k_min, x)
        k_max = x + width if k_max is None else max(k_max, x + width)

    if q_min is None or q_max is None or k_min is None or k_max is None:
        # Nothing to draw
        plt.close(fig)
        return

    # Optional clipping to max_size (mainly for very long sequences)
    if max_size is not None:
        k_max = min(k_max, k_min + max_size)
        q_max = min(q_max, q_min + max_size)

    ax.set_xlim(k_min, k_max)
    ax.set_ylim(q_min, q_max)

    # Ensure q and k unit lengths are displayed as the same physical length to avoid distortion
    ax.set_aspect("equal", adjustable="box")

    # q increases downwards -> invert y-axis
    ax.invert_yaxis()

    # Move k-axis to the top
    ax.set_xlabel("k (key index)")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("q (query index)")
    ax.set_title(title)

    # Simple legend by color for each rank
    handles = []
    labels = []
    for rank in range(num_ranks):
        color = cmap(rank % cmap.N)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linewidth=4,
            )
        )
        labels.append(f"rank {rank}")
    # Place legend in the upper right, moved outside the plot area to avoid obscuring content
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.5,
        fontsize="small",
    )

    ax.grid(True, linestyle="--", alpha=0.3)
    # Leave fixed space for the legend on the right to avoid squashing the main plot area
    fig.subplots_adjust(right=0.8, top=0.9)

    if save_path is not None:
        print(f"{save_path=}")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
