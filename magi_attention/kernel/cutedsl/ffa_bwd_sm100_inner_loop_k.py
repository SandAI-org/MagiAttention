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

# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.

# pyright: reportInvalidTypeForm=false

import math
from typing import Optional

import torch

# isort: split
from .ffa_utils import MT_MAP
from .flex_flash_attn import _flex_flash_attn_bwd


class FFABwdSm100InnerLoopK:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        tile_m: int = 128,
        tile_n: int = 64,
    ):
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        assert tile_m > 0 and tile_n > 0
        # One load/compute covers the full head: Q is tile_m x head_dim, KV is tile_n x head_dim.
        assert self.tile_hdim == head_dim and self.tile_hdim <= 128, (
            f"head_dim={head_dim} must fit in one tile "
            f"(multiple of {hdim_multiple_of}, at most 128)"
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        mask_types: int | torch.Tensor = MT_MAP.full,
        softmax_scale: Optional[float] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        dq: Optional[torch.Tensor] = None,
        dk: Optional[torch.Tensor] = None,
        dv: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert q.shape[-1] == k.shape[-1] == v.shape[-1] == self.tile_hdim
        assert q.ndim == k.ndim == v.ndim == 3
        assert out.shape == dout.shape == q.shape
        assert q.dtype in (torch.float16, torch.bfloat16)
        assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
        assert q.is_cuda
        assert all(
            t.device == q.device for t in (k, v, out, dout, lse, q_ranges, k_ranges)
        )
        assert q_ranges.shape == k_ranges.shape and q_ranges.shape[-1] == 2
        assert q_ranges.dtype == k_ranges.dtype == torch.int32
        assert lse.shape == (q.shape[0], q.shape[1]) and lse.dtype == torch.float32
        assert lse.stride(-1) == 1
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.tile_hdim)

        # ///////////////////////////////////////////////////////////////////////////////
        # HACK_NON_SWAP_LOOP: non-swap = existing FFABwdSm100 (K outer loop, tile_n=128).
        # DEVIATION: test-only delegate lives in the production entry
        # Reason: the inner-loop-K kernel is not implemented; outer tests need a correct backward
        # Tracking: delete this return when __call__ launches the inner-loop-K kernel
        # ///////////////////////////////////////////////////////////////////////////////
        return _flex_flash_attn_bwd(
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            dout=dout,
            dq=dq,
            dk=dk,
            dv=dv,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            mask_types=mask_types,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
        )


def ffa_bwd_sm100_inner_loop_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    *,
    mask_types: int | torch.Tensor = MT_MAP.full,
    softmax_scale: Optional[float] = None,
    tile_m: int = 128,
    tile_n: int = 64,
    head_dim: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_dim is None:
        head_dim = int(q.shape[-1])
    return FFABwdSm100InnerLoopK(head_dim, tile_m, tile_n)(
        q,
        k,
        v,
        out,
        dout,
        lse,
        q_ranges,
        k_ranges,
        mask_types=mask_types,
        softmax_scale=softmax_scale,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dq=dq,
        dk=dk,
        dv=dv,
    )
