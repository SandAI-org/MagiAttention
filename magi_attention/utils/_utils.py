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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from rich import print as rprint

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
) -> tuple[int, int, int, dist.ProcessGroup, int, int | None]:
    """set up distributed environment with the specified process group backend,
    NOTE: the test script using this func to set up should be executed through torchrun

    Args:
        backend (str, optional): the process group backend. Defaults to "nccl".
        base_seed (int | None, optional): the base seed. Defaults to None to not set seed.
        seed_bias (Callable, optional): the seed bias func for each rank. Defaults to lambda rank: 0, i.e., no bias.

    Returns:
        rank, local_rank, world_size, world_group, device, seed
    """
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

    return (
        rank,
        local_rank,
        world_size,
        dist.group.WORLD,
        device,
        seed,
    )  # noqa: E231


def clearup_dist_env() -> None:
    dist.destroy_process_group()


NestedIntList: TypeAlias = Union[list[int], tuple[int, ...], Sequence["NestedIntList"]]


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
) -> None:  # pragma: no cover
    cmap = plt.cm.gray
    nrows, ncols = matrix.shape[0], matrix.shape[1]

    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap=cmap, interpolation="nearest")

    ax.set_xticks(np.arange(ncols), np.arange(ncols))
    ax.set_yticks(np.arange(nrows), np.arange(nrows))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = plt.colorbar(cax)

    if val_ticks is not None:
        cbar.set_ticks(val_ticks)
    if format_ticks is not None:
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

    plt.show()

    if save_path is not None:
        plt.savefig(save_path)


def vis_attn_mask(
    attn_mask: torch.Tensor,
    save_path: str | None = None,
) -> None:  # pragma: no cover
    vis_matrix(
        attn_mask.cpu().numpy(),
        title="attn_mask",
        xlabel="k",
        ylabel="q",
        val_ticks=[0, 1],
        format_ticks=lambda x, pos: "unmasked" if x == 0 else "unmasked",
        save_path=save_path,
    )


def make_ffa_causal_mask(
    seqlen_q: int,
    seqlen_k: int,
    attn_type_idx: int = 0,
    device: str | int = "cuda",
) -> torch.Tensor:
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


def get_attn_mask_from_ffa_args(
    q_ranges: "AttnRanges",
    k_ranges: "AttnRanges",
    attn_type_map: list[int],
    total_seqlen_q: int,
    total_seqlen_k: int,
    device: str | int = "cuda",
) -> torch.Tensor:
    mask = torch.zeros(
        (total_seqlen_q, total_seqlen_k),
        dtype=torch.bool,
        device=device,
    )

    for q_range, k_range, attn_type_idx in zip(q_ranges, k_ranges, attn_type_map):
        slice_mask = make_ffa_causal_mask(
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


def to_higher_fp_dtype(
    tensor: torch.Tensor,
    lowest_precision: torch.dtype,
) -> torch.Tensor:
    if torch.finfo(tensor.dtype).bits < torch.finfo(lowest_precision).bits:
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
