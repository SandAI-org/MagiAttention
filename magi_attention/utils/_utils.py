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

import functools
import hashlib
import os
import random
import warnings
from contextlib import contextmanager
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Sequence,
    TypeAlias,
    Union,
)

import numpy as np
import torch
import torch.distributed as dist

from . import nvtx

if TYPE_CHECKING:
    from magi_attention.common.ranges import AttnRanges


def deprecated(func: Callable) -> Callable:
    """A decorator for deprecated functions"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"The '{func.__name__}' is deprecated and might be removed in future versions.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


def rprint_rank(
    msg: str,
    rank: int | None = None,
    width: int = 50,
) -> None:  # pragma: no cover
    from rich import print as rprint

    if rank is None or dist.get_rank() == rank:
        rank = dist.get_rank()
        rprint(
            f"\n{'-' * width}{' ' * 5}{rank=}{' ' * 5}{'-' * width}\n\n" + msg,
            flush=True,
        )


def write_rank(
    msg: str,
    path: str,
    rank: int | None = None,
    width: int = 50,
) -> None:  # pragma: no cover
    if rank is None or dist.get_rank() == rank:
        rank = dist.get_rank()
        write_content = (
            f"\n{'-' * width}{' ' * 5}{rank=}{' ' * 5}{'-' * width}\n\n" + msg
        )

        with open(path, "a") as f:
            f.write(write_content)


def setup_dist_env(
    backend: str = "nccl",
    base_seed: int | None = None,
    seed_bias: Callable = lambda rank: 0,
) -> tuple[int, int, int, int, int, dist.ProcessGroup, int, int | None]:
    """set up distributed environment with the specified process group backend,
    NOTE: the test script using this func to set up should be executed through torchrun

    Args:
        backend (str, optional): the process group backend. Defaults to "nccl".
        base_seed (int | None, optional): the base seed. Defaults to None to not set seed.
        seed_bias (Callable, optional): the seed bias func for each rank. Defaults to lambda rank: 0, i.e., no bias.

    Returns:
        rank, local_rank, world_size, num_nodes, num_local_ranks, world_group, device, seed
    """
    # extract the distributed environment info
    num_nodes = int(os.environ.get("NNODES", "1"))
    num_local_ranks = int(os.environ.get("NPROC_PER_NODE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # setup device
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # init process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # set random seed
    seed = None
    if base_seed is not None:
        seed = base_seed + seed_bias(rank)
        set_random_seed(seed)

    return (
        rank,
        local_rank,
        world_size,
        num_nodes,
        num_local_ranks,
        dist.group.WORLD,
        device,
        seed,
    )  # noqa: E231


def clearup_dist_env() -> None:
    dist.destroy_process_group()


NestedIntList: TypeAlias = Union[list[int], tuple[int, ...], Sequence["NestedIntList"]]


def format_list_field(name: str, data_list: list, indent: str) -> str:
    """Helper to format a list field with tree-like structure."""
    if not data_list:
        return f"{indent}    {name}=[]\n"

    list_repr = f"{indent}    {name}=[\n"
    for i, item in enumerate(data_list):
        prefix = "└── " if i == len(data_list) - 1 else "├── "
        item_repr_lines = repr(item).splitlines()
        list_repr += f"{indent}        {prefix}{item_repr_lines[0]}\n"
        # If the item itself has a multi-line repr, indent it further
        for line in item_repr_lines[1:]:
            list_repr += f"{indent}        {' ' * len(prefix)}{line}\n"
    list_repr += f"{indent}    ]\n"
    return list_repr


def format_dict_field(name: str, data_dict: dict, indent: str) -> str:
    """Helper to format a dict field with tree-like structure."""
    if not data_dict:
        return f"{indent}    {name}={data_dict},\n"

    dict_repr = f"{indent}    {name}={{\n"
    keys = list(data_dict.keys())
    for i, key in enumerate(keys):
        value = data_dict[key]
        prefix = "└── " if i == len(keys) - 1 else "├── "
        value_repr_lines = repr(value).splitlines()
        dict_repr += f"{indent}        {prefix}{repr(key)}: {value_repr_lines[0]}\n"
        # Indent subsequent lines of a multi-line value's repr
        for line in value_repr_lines[1:]:
            dict_repr += (
                f"{indent}        {' ' * (len(prefix) + len(repr(key)) + 2)}{line}\n"
            )
    dict_repr += f"{indent}    }},\n"
    return dict_repr


def seqlens2cu_seqlens(seqlens: list[int]) -> list[int]:
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return cu_seqlens


def cu_seqlens2seqlens(cu_seqlens: list[int]) -> list[int]:
    seqlens = []
    for i in range(1, len(cu_seqlens)):
        seqlens.append(cu_seqlens[i] - cu_seqlens[i - 1])
    return seqlens


@nvtx.instrument_nvtx
def flatten_nested_list(nested_list: NestedIntList) -> list[int]:
    # Initialize a stack with the reversed nested list to process elements from left to right
    stack = list(nested_list[::-1])

    # Initialize an empty list to store the flattened elements
    flat_list: list[int] = []

    # Process the stack until all elements are handled
    while stack:
        item = stack.pop()  # Pop the last element from the stack
        if isinstance(item, (list, tuple)):
            # If the element is a list, reverse it and extend the stack with its elements
            stack.extend(item[::-1])
        else:
            # If the element is not a list, add it to the flat list
            flat_list.append(item)  # type: ignore[arg-type]

    return flat_list  # Return the fully flattened list


def perm_idxs2unperm_idxs(perm_idxs: list[int]) -> list[int]:
    if not perm_idxs:
        return []

    unperm_idxs = [0] * len(perm_idxs)

    for i in range(len(perm_idxs)):
        unperm_idxs[perm_idxs[i]] = i

    return unperm_idxs


def wrap_to_list(x: Any, broadcast_to_length: int = 1) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x] * broadcast_to_length


def is_list_value_all(
    _list: list[Any],
    val: Any = None,
    just_same: bool = False,
    allow_empty: bool = False,
) -> bool:
    if len(_list) == 0:
        return allow_empty

    if just_same:
        assert val is None, "val should be None when just_same is True"
        val = _list[0]

    return all(x == val for x in _list)


def is_list_value_any(
    _list: list[Any],
    val: Any = None,
    just_same: bool = False,
    allow_empty: bool = False,
) -> bool:
    if len(_list) == 0:
        return allow_empty

    if just_same:
        assert val is None, "val should be None when just_same is True"
        val = _list[0]

    return any(x == val for x in _list)


def is_list_type_all(
    _list: list[Any],
    _type: Any = None,
    just_same: bool = False,
    allow_empty: bool = False,
) -> bool:
    if len(_list) == 0:
        return allow_empty

    if just_same:
        assert _type is None, "_type should be None when just_same is True"
        _type = type(_list[0])

    return all(isinstance(x, _type) for x in _list)


def pad_and_pack_tensors(
    tensors: list[torch.Tensor],
    target_length: int,
    padding_value: float = 0.0,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Right Pads a list of 1D tensors to a target length and packs them into a 2D tensor.

    Args:
        tensors: A list of 1D torch.Tensor objects.
        target_length: The desired length for each padded tensor (e.g., num_ranks).
        padding_value: The value to use for padding. Defaults to 0.0.
        dtype: The desired data type of the output tensor. If None,
               it will be inferred from the input tensors.
        device: The desired device of the output tensor. If None,
                it will be inferred from the input tensors.

    Returns:
        A 2D torch.Tensor where each row is a padded input tensor.
        The shape will be (len(tensors), target_length).
    """
    if not tensors:
        return torch.empty(0, target_length, dtype=dtype, device=device)

    # Infer dtype and device if not provided
    if dtype is None:
        dtype = tensors[0].dtype
    if device is None:
        device = tensors[0].device

    num_tensors = len(tensors)

    # Create the output 2D tensor initialized with the padding value
    packed_tensor = torch.full(
        (num_tensors, target_length),
        fill_value=padding_value,
        dtype=dtype,
        device=device,
    )

    for i, tensor in enumerate(tensors):
        if tensor.dim() != 1:
            raise ValueError(f"Input tensor at index {i} is not 1D: {tensor.dim()}D")

        current_length = tensor.numel()
        if current_length > target_length:
            raise ValueError(
                f"Tensor at index {i} has length {current_length}, "
                f"which is greater than target_length {target_length}. "
                "Cannot pad to a smaller length."
            )

        # Copy the original tensor into the corresponding row of the packed tensor
        packed_tensor[i, :current_length] = tensor

    return packed_tensor


def transpose_matrix(matrix: list[list[Any]]) -> list[list[Any]]:
    """
    Transposes a 2D list (matrix) where each cell contains custom objects.

    Args:
        matrix (list[list[Any]]): A 2D list to be transposed.

    Returns:
        list[list[Any]]: The transposed 2D list.
    """
    assert matrix and isinstance(matrix[0], list), "Input must be a non-empty 2D list."

    col_size = len(matrix[0])
    assert all(
        len(row) == col_size for row in matrix
    ), "All rows must have the same length to be transposed."

    # transpose the matrix using zip
    transposed = [list(row) for row in zip(*matrix)]

    return transposed


def repr_matrix(matrix: np.ndarray) -> str:  # pragma: no cover
    repr_str = ""
    sep = "    "

    nrows, ncols = matrix.shape[0], matrix.shape[1]
    row_idx_width = len(str(nrows))
    col_idx_width = len(str(ncols))
    to_str = lambda x: f"{x: <{col_idx_width}}"  # noqa

    repr_str += " " * (row_idx_width + 3) + sep.join(map(to_str, range(ncols))) + "\n"
    col_width = len(repr_str)
    repr_str += " " * 3 + "+" + "-" * (col_width - 3) + ">" + "\n"

    for row_idx, row in enumerate(matrix):
        repr_str += (
            f"{row_idx: <{row_idx_width}}" + " | " + sep.join(map(to_str, row)) + "\n"
        )

    return repr_str


def vis_matrix(
    matrix: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    val_ticks: list[float] | None = None,
    format_ticks: Callable | None = None,
    save_path: str | None = None,
    alpha: float = 1.0,
    interpolation_order: int = 1,
    plot_interpolation: str = "nearest",
) -> None:  # pragma: no cover
    """
    Visualizes a 2D matrix as a grayscale image. Supports proportional downsampling.

    Args:
        matrix: The 2D numpy array to visualize.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        val_ticks: Optional list of tick values for the color bar.
        format_ticks: Optional formatting function for the color bar tick labels.
        save_path: Optional path to save the plot. If None, the plot is not saved.
        alpha: Downsampling ratio factor (0 < alpha <= 1.0). Defaults to 1.0 (no downsampling).
            If alpha is less than 1.0, the matrix will be downsampled proportionally before plotting.
        interpolation_order: The order of interpolation used for downsampling
            (0 for nearest, 1 for linear, 2 for quadratic, 3 for cubic).
            Only applies when alpha < 1.0. Defaults to 1 (linear interpolation).
        plot_interpolation: The interpolation method for `imshow`. Common values include "nearest",
            "bilinear", "bicubic". Defaults to "nearest".
    """

    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input 'matrix' must be a 2D numpy array.")

    # Validate alpha parameter
    assert (
        0 < alpha <= 1.0
    ), "'alpha' must be greater than 0 and less than or equal to 1.0"

    cmap = plt.cm.gray
    original_nrows, original_ncols = matrix.shape[0], matrix.shape[1]

    matrix_to_plot = matrix
    current_nrows, current_ncols = original_nrows, original_ncols

    # Downsample if alpha is less than 1.0
    if alpha < 1.0:
        # Calculate new dimensions, ensuring they are at least 1 (unless original dimension is 0)
        new_nrows = max(1, int(original_nrows * alpha)) if original_nrows > 0 else 0
        new_ncols = max(1, int(original_ncols * alpha)) if original_ncols > 0 else 0

        # Only perform zoom if target dimensions are smaller than original and valid
        if (new_nrows > 0 and new_ncols > 0) and (
            new_nrows < original_nrows or new_ncols < original_ncols
        ):
            zoom_factors = (new_nrows / original_nrows, new_ncols / original_ncols)
            try:
                matrix_to_plot = zoom(matrix, zoom_factors, order=interpolation_order)
            except Exception as e:
                print(
                    f"Error during downsampling with scipy.ndimage.zoom: {e}. Original matrix will be used."
                )
                matrix_to_plot = matrix
                new_nrows, new_ncols = original_nrows, original_ncols  # Reset on error
            current_nrows, current_ncols = (
                matrix_to_plot.shape[0],
                matrix_to_plot.shape[1],
            )
        elif original_nrows == 0 or original_ncols == 0:
            # Handle cases where the original matrix is empty
            matrix_to_plot = np.array([])
            current_nrows, current_ncols = 0, 0

    # Set up the figure and axes
    # Adjust figsize based on the aspect ratio of the matrix to be plotted
    # Use a fixed width (e.g., 8 inches) and calculate height to maintain aspect ratio
    # Ensure a minimum size for readability, even for small matrices
    fig_width = 8
    # Calculate aspect ratio, preventing division by zero for empty or single-dimension matrices
    aspect_ratio = current_nrows / current_ncols if current_ncols > 0 else 1.0
    fig_height = max(3.0, fig_width * aspect_ratio)  # Minimum height of 3 inches

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if current_nrows == 0 or current_ncols == 0:
        raise RuntimeError("Matrix is empty or too small after downsampling.")
    else:
        # Plot the (potentially downsampled) matrix
        cax = ax.imshow(matrix_to_plot, cmap=cmap, interpolation=plot_interpolation)

        # Set ticks based on the dimensions of the currently plotted matrix
        # This preserves the original function's intent of labeling each pixel for *display*.
        ax.set_xticks(np.arange(current_ncols))
        ax.set_xticklabels(np.arange(current_ncols))
        ax.set_yticks(np.arange(current_nrows))
        ax.set_yticklabels(np.arange(current_nrows))

        if alpha < 1.0:
            ax.set_title(f"{title} (downsampled ratio: {alpha * 100:.1f}%)")
        else:
            ax.set_title(title)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Ensure cells are square and adjust plot limits for pixel-perfect display
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.5, current_ncols - 0.5)
        ax.set_ylim(
            current_nrows - 0.5, -0.5
        )  # Invert Y-axis to match typical image plotting origin

        # Add color bar
        cbar = plt.colorbar(
            cax, ax=ax, fraction=0.046, pad=0.04
        )  # Attach colorbar to main axes, better sizing

        if val_ticks is not None:
            cbar.set_ticks(val_ticks)
        if format_ticks is not None:
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

    # Save the plot before showing
    if save_path is not None:
        try:
            plt.savefig(
                save_path, bbox_inches="tight", dpi=300
            )  # dpi=300 for higher quality
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    plt.show()
    plt.close(fig)  # Close the figure after showing or saving to free up memory


def vis_attn_mask(
    attn_mask: torch.Tensor,
    alpha: float = 1.0,
    save_path: str | None = None,
) -> None:  # pragma: no cover
    """
    Visualizes a 2D attention mask (torch.Tensor) using the vis_matrix function.

    This function converts a PyTorch attention mask tensor to a NumPy array,
    then calls `vis_matrix` to plot it as a grayscale image, potentially
    downsampling it based on the `alpha` parameter. The attention mask is typically
    expected to contain 0s (for unmasked/allowed attention) and 1s (for masked/disallowed attention).

    Args:
        attn_mask: A 2D torch.Tensor representing the attention mask.
            Expected to have dimensions (sequence_length_q, sequence_length_k).
        alpha: Downsampling ratio factor (0 < alpha <= 1.0). Defaults to 1.0 (no downsampling).
            If alpha is less than 1.0, the attention mask will be downsampled proportionally
            before being passed to `vis_matrix` for plotting.
        save_path: Optional path to save the plot. If None, the plot is not saved.
            This path is passed directly to the `vis_matrix` function.
    """
    vis_matrix(
        attn_mask.cpu().numpy(),  # Convert to NumPy array and move to CPU if necessary
        title="Attention Mask",  # Changed title to be more generic
        xlabel="Key Sequence Index (k)",  # Changed xlabel for clarity
        ylabel="Query Sequence Index (q)",  # Changed ylabel for clarity
        val_ticks=[0, 1],
        format_ticks=lambda x, pos: "Masked"
        if x == 0
        else "Unmasked",  # Adjusted labels
        save_path=save_path,
        alpha=alpha,
    )


def make_slice_mask_from_ffa_attn_type(
    seqlen_q: int,
    seqlen_k: int,
    attn_type_idx: int = 0,
    device: str | int = "cuda",
) -> torch.Tensor:
    """Make the boolean mask tensor of certain AttnSlice from the given attn type index

    Args:
        seqlen_q (int): the seqlen of query in the AttnSlice
        seqlen_k (int): the seqlen of key in the AttnSlice
        attn_type_idx (int, optional): the attn type index. Defaults to ``0``.
        device (str | int, optional): the device. Defaults to "cuda".

    Raises:
        ValueError: the attn type index is invalid

    Returns:
        torch.Tensor: the boolean mask tensor of certain AttnSlice
            where the entries with ``False`` indicate that the corresponding positions are masked
    """
    max_seqlen = max(seqlen_q, seqlen_k)
    latend_square_full_mask = torch.ones(
        (max_seqlen, max_seqlen),
        dtype=torch.bool,
        device=device,
    )

    match attn_type_idx:
        case 0:  # full
            mask = latend_square_full_mask[:seqlen_q, :seqlen_k]
        case 1:  # causal with bottom-right aligned
            mask = torch.tril(latend_square_full_mask)[-seqlen_q:, -seqlen_k:]
        case 2:  # inv-causal with top-left aligned
            mask = torch.triu(latend_square_full_mask)[:seqlen_q, :seqlen_k]
        case 3:  # bi-causal with bottom-right and top-left (bi-directional) aligned
            mask = (
                torch.tril(latend_square_full_mask)[-seqlen_q:, -seqlen_k:]
                & torch.triu(latend_square_full_mask)[:seqlen_q, :seqlen_k]
            )
        case _:
            raise ValueError(f"Invalid {attn_type_idx=}")

    return mask


def make_attn_mask_from_ffa_args(
    q_ranges: "AttnRanges",
    k_ranges: "AttnRanges",
    attn_type_map: list[int],
    total_seqlen_q: int,
    total_seqlen_k: int,
    device: str | int = "cuda",
) -> torch.Tensor:
    """Make the complete latent boolean mask tensor from the FFA arguments

    Args:
        q_ranges (AttnRanges): the query ranges
        k_ranges (AttnRanges): the key ranges
        attn_type_map (list[int]): the mask type list
        total_seqlen_q (int): the total seqlen of query
        total_seqlen_k (int): the total seqlen of key
        device (str | int, optional): the device of the mask tensor.
            Defaults to "cuda".

    Returns:
        torch.Tensor: the boolean mask tensor on the given device
            where the entries with ``False`` indicate that the corresponding positions are masked
    """
    mask = torch.zeros(
        (total_seqlen_q, total_seqlen_k),
        dtype=torch.bool,
        device=device,
    )

    for q_range, k_range, attn_type_idx in zip(q_ranges, k_ranges, attn_type_map):
        slice_mask = make_slice_mask_from_ffa_attn_type(
            seqlen_q=q_range.seqlen,
            seqlen_k=k_range.seqlen,
            attn_type_idx=attn_type_idx,
            device=device,
        )

        mask[
            q_range.start : q_range.end,
            k_range.start : k_range.end,
        ] = slice_mask

    return mask


def is_fp_dtype_at_least(
    tensor: torch.Tensor,
    lowest_precision: torch.dtype,
) -> bool:
    return torch.finfo(tensor.dtype).bits >= torch.finfo(lowest_precision).bits


def to_higher_fp_dtype(
    tensor: torch.Tensor,
    lowest_precision: torch.dtype,
) -> torch.Tensor:
    if not is_fp_dtype_at_least(tensor, lowest_precision):
        return tensor.to(lowest_precision)
    return tensor


def max_fp_dtype(
    *dtypes: torch.dtype,
) -> torch.dtype:
    return max(dtypes, key=lambda dtype: torch.finfo(dtype).bits)


def argmin(iterable: Iterable[Any], key: Callable = lambda x: x) -> int:
    return min(enumerate(iterable), key=lambda x: key(x[1]))[0]


def argmax(iterable: Iterable[Any], key: Callable = lambda x: x) -> int:
    return max(enumerate(iterable), key=lambda x: key(x[1]))[0]


def argsort(iterable: Iterable[Any], key: Callable = lambda x: x) -> list[int]:
    return [i[0] for i in sorted(enumerate(iterable), key=lambda x: key(x[1]))]


def str2seed(s: str) -> int:
    max_value = 2**32 - 1  # numpy max seed
    hash_value = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

    return hash_value % (max_value + 1)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _collect_rng_states() -> dict[str, Any]:
    """Collect the global random state of :mod:`torch`, :mod:`numpy` and Python."""
    return {
        "numpy": np.random.get_state(),
        "python": python_get_rng_state(),
    }


def _set_rng_states(rng_state_dict: dict[str, Any]) -> None:
    """Set the global random state of :mod:`torch`, :mod:`numpy` and Python in the current process."""
    np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))


@contextmanager
def sync_rng(seed: int) -> Generator[None, None, None]:
    """A context manager that syncs the random seed for everything including python, numpy and torch on all devices
    and resets the global random state on exit to what it was before entering.

    Args:
        seed (int): The random seed to set.
    """

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        states = _collect_rng_states()
        set_random_seed(seed)
        yield
        _set_rng_states(states)


def is_same_process_group(
    group1: dist.ProcessGroup | None,
    group2: dist.ProcessGroup | None,
) -> bool:
    """Determine whether two communication groups are the same

    Args:
        group1 (dist.ProcessGroup | None): process group 1
        group2 (dist.ProcessGroup | None): process group 2

    Returns:
        bool: whether two communication groups are the same
    """
    if group1 is None and group2 is None:
        return True
    if not isinstance(group1, dist.ProcessGroup) or not isinstance(
        group2, dist.ProcessGroup
    ):
        return False

    group1_ranks = sorted(dist.get_process_group_ranks(group=group1))
    group2_ranks = sorted(dist.get_process_group_ranks(group=group2))
    if group1_ranks == group2_ranks:
        return True
    return False


def is_same_device_mesh(
    mesh1: dist.device_mesh.DeviceMesh | None,
    mesh2: dist.device_mesh.DeviceMesh | None,
) -> bool:
    """Determine whether two device meshs are the same

    Args:
        mesh1 (dist.device_mesh.DeviceMesh | None): device mesh1
        mesh2 (dist.device_mesh.DeviceMesh | None): device mesh2

    Returns:
        bool: whether two device meshs are the same
    """
    if mesh1 is None and mesh2 is None:
        return True
    if not isinstance(mesh1, dist.device_mesh.DeviceMesh) or not isinstance(
        mesh2, dist.device_mesh.DeviceMesh
    ):
        return False

    return mesh1.device_type == mesh2.device_type and torch.equal(
        mesh1.mesh, mesh2.mesh
    )


def get_calc_cost_factor(
    num_heads_q: int,
    head_dim: int,
    tflops: float,
    mfu: float,
    sec_ratio: float = 1e6,  # default μs, 1s = 1e6 μs
    is_fwd: bool = True,
) -> float:
    """
    calc cost factor = 2 * 2 * nhq * hd / TFLOPS / mfu * sec_ratio *
        (1 if is_fwd else 2.5)
    """

    return (
        (1 if is_fwd else 2.5)  # 5 matmul for bwd
        * 2  # 2 matmul
        * 2  # 2 flops per matmul
        * num_heads_q
        * head_dim
        / tflops
        / mfu
        * sec_ratio
    )


def get_comm_cost_factor(
    num_heads_kv: int,
    head_dim: int,
    bandwidth: float,
    bwu: float,
    corr_factor: float,
    sec_ratio: float = 1e6,  # default μs, 1s = 1e6 μs
) -> float:
    """
    comm cost factor = 2 * nhkv * hd / bandwidth / corr_factor / bwu * sec_ratio
    """
    return (
        2 * num_heads_kv * head_dim / bandwidth / corr_factor / bwu * sec_ratio  # k + v
    )


def get_a2a_corr_factor(world_size: int) -> float:
    return (world_size - 1) / world_size if world_size > 1 else 1.0


def missing_dependency(func_name: str, dep_name: str):
    """
    Return a dummy function that raises ImportError when called.
    """

    def _raise(*args, **kwargs):
        raise ImportError(
            f"`{func_name}` requires optional dependency `{dep_name}`, "
            f"but it is not installed."
        )

    return _raise
