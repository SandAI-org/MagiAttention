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

"""NVIDIA Flex Attention adapters for Magi's interval-mask CP stages."""

import torch

from magi_attention.meta.collection.calc_meta import CudnnAttnArg


def validate_cudnn_inputs(
    q, k, v, *, sink=None, softcap=0.0, return_max_logits=False, deterministic=False
):
    """Validate the supported NVIDIA input contract."""
    if sink is not None or softcap != 0.0 or return_max_logits:
        raise ValueError(
            "cudnn requires sink=None, softcap=0 and return_max_logits=False"
        )
    if any(
        t.ndim != 3 or not t.is_cuda or t.dtype != torch.bfloat16 for t in (q, k, v)
    ):
        raise ValueError("cudnn requires rank-3 CUDA BF16 Q/K/V")
    if any(t.device != q.device for t in (k, v)):
        raise ValueError("Q/K/V must be on the same CUDA device")
    if torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
        raise ValueError("cudnn currently supports SM100/SM103 only")
    if (
        q.shape[-1] != k.shape[-1]
        or k.shape[:2] != v.shape[:2]
        or k.shape[1] <= 0
        or q.shape[1] <= 0
        or q.shape[1] % k.shape[1]
    ):
        raise ValueError("Incompatible Q/K/V head geometry")
    dq, dv = q.shape[-1], v.shape[-1]
    if not (
        (8 <= dq <= 128 and dq % 8 == 0 and 8 <= dv <= 128 and dv % 8 == 0)
        or (dq, dv) in ((192, 128), (256, 256))
    ):
        raise ValueError(f"Unsupported cudnn head dimensions: {(dq, dv)}")


def _canonical(tensor: torch.Tensor, alignment: int = 16) -> torch.Tensor:
    tensor = tensor.contiguous()
    if tensor.data_ptr() % alignment:
        tensor = tensor.clone()
    return tensor


@torch.no_grad()
def build_cudnn_mask_plan(attn_arg, q, k, v):
    """Build immutable stage topology without compiling attention executors."""
    try:
        import magi_to_hstu_cuda
        from cudnn.flex_attention import create_mask_plan
    except ImportError as exc:
        raise ImportError(
            "Install the cudnn backend with scripts/install_cudnn_flex_attn.sh"
        ) from exc

    with torch.cuda.device(q.device), torch.no_grad():
        # NVIDIA consumes the original interval relation, not FFA merged maps.
        func = magi_to_hstu_cuda.magi_to_hstu(
            q_ranges=attn_arg.q_ranges.to_tensor(device=q.device),
            k_ranges=attn_arg.k_ranges.to_tensor(device=q.device),
            mask_types=torch.tensor(
                attn_arg.attn_type_map, dtype=torch.int32, device=q.device
            ),
            seqlen_q=q.shape[1],
            seqlen_k=k.shape[1],
            n_max_func=2 * len(attn_arg.k_ranges) + 1,
        )
        # Sorted, merged intervals are followed by unused zero slots.
        # Prefix maxima replace those slots by empty [last_end,last_end).
        func = torch.cummax(func, dim=0).values.unsqueeze(0).contiguous()
        plan = create_mask_plan(func, q, k, v, build_backward=True)
        ready = torch.cuda.Event()
        ready.record()
        return plan, ready


@torch.no_grad()
def cudnn_fwd(
    q,
    k,
    v,
    attn_arg: CudnnAttnArg,
    softmax_scale=None,
    softcap=0.0,
    sink=None,
    deterministic=False,
    **kwargs,
):
    validate_cudnn_inputs(q, k, v, sink=sink, softcap=softcap)
    if attn_arg.can_skip(False) or not q.numel() or not k.numel():
        out = torch.zeros((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
        lse = torch.full(
            q.shape[:2], -float("inf"), dtype=torch.float32, device=q.device
        )
        return out, lse

    # The pinned NVIDIA execution wrappers own JIT compilation and bounded caches.
    from cudnn.flex_attention.execution import _flex_attention_forward

    q, k, v = (_canonical(t).unsqueeze(0) for t in (q, k, v))
    plan = attn_arg.to_cudnn_args(q, k, v)
    result = _flex_attention_forward(
        q,
        k,
        v,
        mask_plan=plan,
        softmax_scale=softmax_scale,
        return_lse=True,
        stream=torch.cuda.current_stream(q.device),
    )
    return result["o_tensor"].squeeze(0), result["lse_tensor"].squeeze(0).mT


@torch.no_grad()
def cudnn_bwd(
    do,
    q,
    k,
    v,
    o,
    lse,
    attn_arg: CudnnAttnArg,
    softmax_scale=None,
    softcap=0.0,
    sink=None,
    deterministic=False,
    **kwargs,
):
    validate_cudnn_inputs(q, k, v, sink=sink, softcap=softcap)
    if attn_arg.can_skip(False) or not q.numel() or not k.numel():
        return *(torch.zeros_like(t) for t in (q, k, v)), None

    from cudnn.flex_attention.execution import _flex_attention_backward

    alignment = 128 if (q.shape[-1], v.shape[-1]) == (256, 256) else 16
    q, k, v, o, do = (_canonical(t, alignment).unsqueeze(0) for t in (q, k, v, o, do))
    plan = attn_arg.to_cudnn_args(q, k, v)
    result = _flex_attention_backward(
        q,
        k,
        v,
        o,
        do,
        lse.mT.contiguous().unsqueeze(0),
        mask_plan=plan,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        stream=torch.cuda.current_stream(q.device),
    )
    return (
        result["dq_tensor"].squeeze(0),
        result["dk_tensor"].squeeze(0),
        result["dv_tensor"].squeeze(0),
        None,
    )
