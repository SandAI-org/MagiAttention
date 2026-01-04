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

from functools import partial

import torch
import torch.nn.functional as F
from configuration_llama import LlamaConfig
from einops import rearrange
from torch import nn
from torch.nn.functional import scaled_dot_product_attention as sdpa_func

from magi_attention.api import calc_attn, get_position_ids
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeKey


class LlamaRMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.hp_params_dtype)
        )
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): Input tensor, with shape: [s, h].
        Returns:
            torch.Tensor: Output tensor, with shape: [s, h].
        """

        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


ACT2FN = {
    "gelu": partial(F.gelu, approximate=False),
    "relu": F.relu,
    "silu": F.silu,
}


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias,
            dtype=config.params_dtype,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias,
            dtype=config.params_dtype,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.mlp_bias,
            dtype=config.params_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor, with shape: [s, h].

        Returns:
            torch.Tensor: Output tensor, with shape: [s, h].
        """

        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor.

    Args:
        x(torch.Tensor): input tensor, with shape: [..., s, hd]
        cos(torch.Tensor): cos basis tensor, with shape: [max_seqlen, hd]
        sin(torch.Tensor): sin basis tensor, with shape: [max_seqlen, hd]
        position_ids(torch.Tensor): position id tensor, with shape: [s]

    Returns:
        output(torch.Tensor): embedded output tensor, with shape: [s, hd]
    """
    # with magi_attention
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        cos = cos[: x.shape[-2]]
        sin = sin[: x.shape[-2]]

    input_batch_shape_size = len(x.shape[:-2])

    for _ in range(input_batch_shape_size):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    output = (x * cos) + (rotate_half(x) * sin)
    return output.to(x.dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.max_seqlen = config.max_seqlen
        self.emb_dtype = config.hp_params_dtype

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(
                    dtype=self.emb_dtype
                )
                / self.head_dim
            )
        )
        t = torch.arange(self.max_seqlen, dtype=self.emb_dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb, sin_emb = emb.cos(), emb.sin()  # shape: (s, hd)

        self.register_buffer("cos_emb", cos_emb, persistent=False)
        self.register_buffer("sin_emb", sin_emb, persistent=False)

    def forward(
        self, x: torch.Tensor, dist_attn_runtime_key: DistAttnRuntimeKey | None = None
    ) -> torch.Tensor:
        """Forward pass of the RoPE Module.
        Args:
            x (torch.Tensor): Input tensor, with shape: [..., s, hd].
            DistAttnRuntimeKey (DistAttnRuntimeKey, optional):  DistAttnRuntimeKey. Default None.

        Returns:
            torch.Tensor: Output tensor, with shape: [...,s, hd].
        """
        # get the position_ids after dispatching
        if dist_attn_runtime_key:
            position_ids = get_position_ids(dist_attn_runtime_key)
        else:
            position_ids = None

        x = apply_rotary_pos_emb(
            x,
            cos=self.cos_emb,
            sin=self.sin_emb,
            position_ids=position_ids,
        )

        return x


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = config.is_causal
        self.rope = LlamaRotaryEmbedding(self.config)

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=config.params_dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=config.params_dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=config.params_dtype,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=config.params_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        magi_attention_runtime_key: DistAttnRuntimeKey | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Self Attention Module.

        Args:
            x (torch.Tensor): Input tensor, with shape: [s, h].
            DistAttnRuntimeKey (DistAttnRuntimeKey, optional):  DistAttnRuntimeKey. Default None.

        Returns:
            torch.Tensor: Output tensor, with shape: [s, h].
        """
        # for sdpa
        q, k, v = [
            rearrange(
                e,
                "s (nh hd) -> 1 nh s hd",
                hd=self.head_dim,
            )
            for e in [
                self.q_proj(x),
                self.k_proj(x),
                self.v_proj(x),
            ]
        ]

        o = self.attn_func(q, k, v, magi_attention_runtime_key)

        # apply out proj
        o = self.o_proj(o)

        return o

    def attn_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        magi_attention_runtime_key: DistAttnRuntimeKey | None = None,
    ) -> torch.Tensor:
        # apply rope
        q, k = [self.rope(e, magi_attention_runtime_key) for e in (q, k)]

        # apply inner backend
        if magi_attention_runtime_key is not None:
            return self.magi_attention_func(q, k, v, magi_attention_runtime_key)
        else:
            return self.sdpa_func(q, k, v)

    def sdpa_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        o = sdpa_func(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attention_dropout,
            is_causal=self.is_causal,
            scale=self.scaling,
            enable_gqa=True,
        )
        o = rearrange(o, "1 nh s hd -> s (nh hd)")

        return o

    def magi_attention_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        magi_attention_runtime_key: DistAttnRuntimeKey,
    ) -> torch.Tensor:
        dtype = q.dtype
        # HACK: reshape to (t, nh, nd) since for now magi_attention only supports
        # ffa as the attn backend which only supports (t, nh, nd) as input
        q, k, v = [
            rearrange(e, "1 nh s hd -> (1 s) nh hd").to(
                torch.float16
            )  # ffa only supports fp16/bf16 for now
            for e in (q, k, v)
        ]

        o = calc_attn(q, k, v, magi_attention_runtime_key)[0]
        o = rearrange(o, "(1 s) nh hd -> s (nh hd)").to(dtype)

        return o


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

    def forward(
        self,
        x: torch.Tensor,
        magi_attention_runtime_key: DistAttnRuntimeKey | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Decoder Layer module.

        Args:
            x (torch.Tensor): Input tensor, with shape: [s, h].
            DistAttnRuntimeKey (DistAttnRuntimeKey, optional):  DistAttnRuntimeKey. Default None.

        Returns:
            torch.Tensor: Output tensor, with shape: [s, h].
        """

        # --- attention -- #

        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, magi_attention_runtime_key)
        x = x + residual

        # --- mlp --- #

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x


class LlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.emb_dtype = config.params_dtype
        self.params_dtype = config.params_dtype

        self.vocab_emb = nn.Embedding(
            self.vocab_size, self.hidden_size, dtype=self.emb_dtype
        )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.lm_head = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=False,
            dtype=self.params_dtype,
        )

    def forward(
        self,
        v: torch.LongTensor,
        magi_attention_runtime_key: DistAttnRuntimeKey | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Llama Module.
        Args:
            v (torch.Tensor): Input vocab tensor, with shape: [s, v].
            magi_attention_runtime_key (DistAttnRuntimeKey, optional): magi_attention runtime key. Default None.
        Returns:
            torch.Tensor: Output vocab logits tensor, with shape: [s, v].
        """

        x = self.vocab_emb(v)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            x = decoder_layer(x, magi_attention_runtime_key)

        logits = self.lm_head(x)

        return logits


def build_llama3_1b_model():
    return LlamaModel(LlamaConfig())
