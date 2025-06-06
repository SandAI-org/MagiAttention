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

from typing import Optional, TypedDict

import torch
from einops import rearrange
from torch import nn
from transformers import LlamaForCausalLM

# from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

# from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing_extensions import Unpack

from magi_attention.api import calc_attn, squash_batch_dim, undispatch
from magi_attention.api.magi_attn_interface import DistAttnRuntimeDict

# from transformers.utils import LossKwargs


def get_magi_attention_key():
    """get newest magi_attention key"""

    return DistAttnRuntimeDict.get_most_recent_key()


# define magi_attn function and register
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
    magi_attn_key = get_magi_attention_key()

    dtype = query.dtype
    q, k, v = [
        rearrange(e, "b nh s hd -> (b s) nh hd").to(
            torch.bfloat16
        )  # ffa only supports fp16/bf16 for now
        for e in (query, key, value)
    ]

    o = calc_attn(q, k, v, magi_attn_key)[0]
    o = rearrange(o, "(1 s) nh hd -> 1 s (nh hd)").to(dtype)  # assume batch_size is 1

    return o, None


# register Magi_Attention as attn_backend globally.
ALL_ATTENTION_FUNCTIONS.register("Magi_Attention", Magi_Attention_forward)


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cumulative_seqlens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cumulative_seqlens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cumulative_seqlens_q: Optional[torch.LongTensor]
    cumulative_seqlens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


class LossKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the loss function

    Attributes:
        num_items_in_batch (`int`, *optional*):
            Number of items in the batch. It is recommended to pass it when
            you are doing gradient accumulation.
    """

    num_items_in_batch: Optional[int]


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    ...


class MagiLlamaForCasual(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])
        magi_attn_key = get_magi_attention_key()

        logits = self.lm_head(hidden_states)

        if magi_attn_key is not None:
            logits = squash_batch_dim(logits)

            logits = undispatch(logits, magi_attn_key)
            logits = logits.unsqueeze(0)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
