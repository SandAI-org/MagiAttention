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

from typing import Optional

import torch
from einops import rearrange
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from magi_attention.api import calc_attn, get_most_recent_key


# define magi_attn function
def Magi_Attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # (b, num_heads, seq_len, hidden_dim)
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    magi_attn_key = get_most_recent_key()

    dtype = query.dtype
    q, k, v = [
        rearrange(e, "1 nh s hd -> (1 s) nh hd").to(
            torch.bfloat16
        )  # ffa only supports fp16/bf16 for now
        for e in (query, key, value)
    ]

    o = calc_attn(q, k, v, magi_attn_key)[0]
    o = rearrange(o, "(1 s) nh hd -> 1 s (nh hd)").to(dtype)  # assume batch_size is 1

    return o, None


# register Magi_Attention as attn_backend globally.
ALL_ATTENTION_FUNCTIONS.register("Magi_Attention", Magi_Attention_forward)
