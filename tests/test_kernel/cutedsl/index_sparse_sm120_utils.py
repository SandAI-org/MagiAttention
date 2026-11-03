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

"""Serve the sm90 IndexSparse forward contract with the SM120 forward.

Shared by ``test_index_sparse_fwd_sm120.py`` and
``exps/attn/cutedsl/run_index_sparse_sm120_benchmark.py`` so the test and the
bench agree on which sm90 inputs SM120 can serve today.

The sm90 contract is described in ``index_sparse_sm100_utils.py``. The SM120
forward (``ffa_fwd_sm120_index``) takes the sm90 layout itself, with the limits
of the SM100 forward: ``head_dim == 128``, ``qbs * G == 128`` for the q-head
group ``G = num_heads_q / num_heads_kv`` (one index row fills one 128-row Q
tile) and ``topk`` a multiple of 4. There is no SM120 index-sparse backward.
"""

from enum import Enum

import torch
import torch.nn.functional as F

from magi_attention.kernel.cutedsl.ffa_fwd_sm120_index import (
    FFAFwdSm120Index,
    ffa_fwd_sm120_index,
)
from tests.test_kernel.cutedsl.index_sparse_sm100_utils import (
    Sm100IndexUnsupportedError,
    index_sparse_ref_fwd,
)

__all__ = [
    "Sm120IndexGap",
    "Sm120IndexUnsupportedError",
    "ffa_fwd_sm120_index_sm90_layout",
    "index_sparse_ref_fwd",
    "is_sm120_family",
    "sm120_index_fwd_gaps",
]


def is_sm120_family() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


class Sm120IndexGap(Enum):
    """sm90 IndexSparse forward features the SM120 forward cannot serve yet.

    Remove a member once the kernel (and ``ffa_fwd_sm120_index_sm90_layout``)
    supports it; the cases that needed it then run the numerical comparison
    instead of reporting a gap.
    """

    HEAD_DIM = "head_dim != 128: FFAFwdSm120Index fixes head_dim at 128"
    SPARSE_K_BLOCK = (
        "sparse_k_block_size > 1: the kernel gathers single K tokens, not K blocks"
    )
    Q_TILE_FILL = (
        "q_block_size * group != 128: FFAFwdSm120Index maps one index row onto "
        "one 128-row Q tile"
    )


class Sm120IndexUnsupportedError(Sm100IndexUnsupportedError):
    """Raised before any launch when the sm90 inputs need an ``Sm120IndexGap``.

    Derives from the SM100 error so the shared case runner of
    ``test_index_sparse_bwd_sm100.py`` reports such a case as GAP.
    """


def sm120_index_fwd_gaps(
    *,
    head_dim: int,
    sparse_k_block_size: int,
    q_block_size: int,
    group: int,
) -> list[Sm120IndexGap]:
    """The ``Sm120IndexGap`` members an sm90 IndexSparse forward call needs.

    ``q_block_size`` is the sm90 one (q tokens per index row) and ``group`` is
    ``num_heads_q / num_heads_kv``.
    """
    gaps = []
    if head_dim != FFAFwdSm120Index.head_dim:
        gaps.append(Sm120IndexGap.HEAD_DIM)
    if sparse_k_block_size != 1:
        gaps.append(Sm120IndexGap.SPARSE_K_BLOCK)
    if q_block_size * group != FFAFwdSm120Index.tile_m:
        gaps.append(Sm120IndexGap.Q_TILE_FILL)
    return gaps


def ffa_fwd_sm120_index_sm90_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    q_block_size: int = 1,
    sparse_k_block_size: int = 1,
    softmax_scale: float | None = None,
    assume_valid_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IndexSparse forward with sm90 tensors, run by ``ffa_fwd_sm120_index``.

    ``q_block_size`` is the sm90 one: q tokens per index row. The tensors reach
    the kernel unchanged, except that an index whose ``topk`` is not a multiple
    of ``FFAFwdSm120Index.topk_align`` is copied with -1 columns appended.
    ``assume_valid_ids`` is forwarded to ``ffa_fwd_sm120_index`` and requires an
    aligned ``topk``, since the appended -1 columns would break its promise.
    Returns (out, lse) in the sm90 layout.

    Raises:
        Sm120IndexUnsupportedError: the call needs an ``Sm120IndexGap``.
    """
    _, num_heads_q, head_dim = q.shape
    _, num_heads_kv, _ = k.shape
    assert num_heads_q % num_heads_kv == 0
    gaps = sm120_index_fwd_gaps(
        head_dim=head_dim,
        sparse_k_block_size=sparse_k_block_size,
        q_block_size=q_block_size,
        group=num_heads_q // num_heads_kv,
    )
    if gaps:
        raise Sm120IndexUnsupportedError(gaps)

    topk_pad = -index_sparse_indices.shape[-1] % FFAFwdSm120Index.topk_align
    if topk_pad:
        assert not assume_valid_ids, (
            f"assume_valid_ids needs topk to be a multiple of "
            f"{FFAFwdSm120Index.topk_align}: aligning it appends -1 ids"
        )
        index_sparse_indices = F.pad(index_sparse_indices, (0, topk_pad), value=-1)
    return ffa_fwd_sm120_index(
        q,
        k,
        v,
        index_sparse_indices,
        softmax_scale=softmax_scale,
        q_block_size=q_block_size,
        assume_valid_ids=assume_valid_ids,
    )
