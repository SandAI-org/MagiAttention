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

from typing import List

import torch

# fa3
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

from magi_attention.common.ranges import AttnRanges
from magi_attention.testing.precision import ref_attn_func


def fa3_test_func(q, k, v, dout, causal, deterministic, qkv_format="bshd"):
    q_layer, k_layer, v_layer = [x.detach().clone() for x in [q, k, v]]
    q_layer.requires_grad_(True)
    k_layer.requires_grad_(True)
    v_layer.requires_grad_(True)
    dout_layer = dout
    if qkv_format == "sbhd":
        q_layer, k_layer, v_layer, dout_layer = [
            x.transpose(0, 1).contiguous() for x in [q_layer, k_layer, v_layer, dout]
        ]
    q_layer.retain_grad()
    k_layer.retain_grad()
    v_layer.retain_grad()
    softmax_scale = q.shape[-1] ** (-0.5)
    window_size = (-1, 0) if causal else (-1, -1)
    out = flash_attn_func(
        q_layer,
        k_layer,
        v_layer,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )
    out.backward(dout_layer)
    dq, dk, dv = q_layer.grad, k_layer.grad, v_layer.grad
    if qkv_format == "sbhd":
        out, dq, dk, dv = [x.transpose(0, 1).contiguous() for x in [out, dq, dk, dv]]
    return (out, dq, dk, dv)


def generate_unpad_indices(cu_seqlens, cu_seqlens_padded, device):
    batch_size = cu_seqlens.shape[0] - 1
    indices_lst = []
    for i in range(batch_size):
        st = cu_seqlens_padded[i]
        ed = st + (cu_seqlens[i + 1] - cu_seqlens[i])
        indices_lst.append(torch.arange(start=st.item(), end=ed.item(), device=device))
    indices = torch.cat(indices_lst, dim=0)
    return indices


def fa3_varlen_test_func(
    q_,
    k_,
    v_,
    dout_,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    max_seqlen_q,
    max_seqlen_k,
    causal,
    qkv_format,
    deterministic,
):
    q, k, v = [x.detach().clone() for x in [q_, k_, v_]]
    softmax_scale = q.shape[-1] ** (-0.5)
    window_size = (-1, 0) if causal else (-1, -1)

    device = q.device
    restore_shape_q = q.shape
    batch_size = restore_shape_q[0]
    other_shape = q.shape[2:]

    if qkv_format == "sbhd":
        batch_size = restore_shape_q[1]
        q_layer, k_layer, v_layer, dout = [
            x.transpose(0, 1).contiguous() for x in [q, k, v, dout_]
        ]
    else:
        q_layer, k_layer, v_layer, dout = q, k, v, dout_

    if qkv_format != "thd":
        q_layer, k_layer, v_layer, dout = [
            x.view(-1, *other_shape) for x in [q_layer, k_layer, v_layer, dout]
        ]
        restore_shape_q = q_layer.shape
        restore_shape_kv = k_layer.shape
        unpad_indices_q = generate_unpad_indices(
            cu_seqlens_q, cu_seqlens_q_padded, device
        )
        q_part = torch.gather(
            q_layer,
            dim=0,
            index=unpad_indices_q[:, None, None].expand(-1, *other_shape),
        )
        dout_part = torch.gather(
            dout, dim=0, index=unpad_indices_q[:, None, None].expand(-1, *other_shape)
        )
        unpad_indices_kv = generate_unpad_indices(
            cu_seqlens_k, cu_seqlens_kv_padded, device
        )
        k_part = torch.gather(
            k_layer,
            dim=0,
            index=unpad_indices_kv[:, None, None].expand(-1, *other_shape),
        )
        v_part = torch.gather(
            v_layer,
            dim=0,
            index=unpad_indices_kv[:, None, None].expand(-1, *other_shape),
        )
    else:
        q_part = q_layer
        k_part = k_layer
        v_part = v_layer
        dout_part = dout

    q_part.requires_grad_(True)
    k_part.requires_grad_(True)
    v_part.requires_grad_(True)

    out_ = flash_attn_varlen_func(
        q_part,
        k_part,
        v_part,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )
    out_.backward(dout_part)

    dq_ = q_part.grad
    dk_ = k_part.grad
    dv_ = v_part.grad

    if qkv_format != "thd":
        out = torch.zeros(*restore_shape_q, device=device, dtype=out_.dtype)
        dq = torch.zeros(*restore_shape_q, device=device, dtype=dq_.dtype)
        dk = torch.zeros(*restore_shape_kv, device=device, dtype=dk_.dtype)
        dv = torch.zeros(*restore_shape_kv, device=device, dtype=dv_.dtype)

        out.scatter_(0, unpad_indices_q[:, None, None].expand(-1, *other_shape), out_)
        dq.scatter_(0, unpad_indices_q[:, None, None].expand(-1, *other_shape), dq_)
        dk.scatter_(0, unpad_indices_kv[:, None, None].expand(-1, *other_shape), dk_)
        dv.scatter_(0, unpad_indices_kv[:, None, None].expand(-1, *other_shape), dv_)
    else:
        out, dq, dk, dv = out_, dq_, dk_, dv_

    if qkv_format != "thd":
        out = out.view(batch_size, -1, *restore_shape_q[1:])
        dq = dq.view(batch_size, -1, *restore_shape_q[1:])
        dk = dk.view(batch_size, -1, *restore_shape_kv[1:])
        dv = dv.view(batch_size, -1, *restore_shape_kv[1:])

    if qkv_format == "sbhd":
        out, dq, dk, dv = [x.transpose(0, 1).contiguous() for x in [out, dq, dk, dv]]

    print(f"{out.shape=},{dq.shape=},{dk.shape=},{dv.shape=}")
    return (out, dq, dk, dv)


def make_causal_mask(
    seqlen_q: int,
    seqlen_k: int,
    align: str = "bottom-right",
    dtype=torch.int32,
    device: str = "cpu",
) -> torch.Tensor:
    max_seqlen = max(seqlen_q, seqlen_k)
    causal_mask = torch.tril(torch.ones((max_seqlen, max_seqlen))).to(
        dtype=dtype, device=device
    )

    if align == "bottom-right":
        causal_mask = causal_mask[-seqlen_q:, -seqlen_k:]
    elif align == "top-left":
        causal_mask = causal_mask[:seqlen_q, :seqlen_k]
    else:
        raise ValueError(f"Invalid alignment mode: {align}")

    return causal_mask


def get_attn_mask_from_cu_seqlens(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    total_seqlen_q = cu_seqlens_q[-1].item()
    total_seqlen_k = cu_seqlens_kv[-1].item()
    mask = torch.zeros(
        (total_seqlen_q, total_seqlen_k),
        dtype=torch.bool,
        device=torch.cuda.current_device(),
    )
    batch_size = cu_seqlens_q.shape[0] - 1
    # for q_range, k_range, is_causal in zip(q_ranges, k_ranges, is_causal_mapping):
    for i in range(batch_size):
        st_q = cu_seqlens_q[i].item()
        ed_q = cu_seqlens_q[i + 1].item()
        st_k = cu_seqlens_kv[i].item()
        ed_k = cu_seqlens_kv[i + 1].item()
        if is_causal:
            causal_mask = make_causal_mask(
                seqlen_q=ed_q - st_q,
                seqlen_k=ed_k - st_k,
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
            mask[
                st_q:ed_q,
                st_k:ed_k,
            ] = causal_mask
        else:
            mask[
                st_q:ed_q,
                st_k:ed_k,
            ] = True

    return mask


def ref_torch_sdpa_func(q, k, v, grad_total_out, mask, high_precision=False):
    total_q, total_k, total_v = [x.detach().clone() for x in [q, k, v]]
    total_q.requires_grad_(True)
    total_k.requires_grad_(True)
    total_v.requires_grad_(True)

    total_out_ref_high_precision, _ = ref_attn_func(
        q=total_q,
        k=total_k,
        v=total_v,
        mask=mask,
        layout="thd",
        sink_layout="sh",
        high_precision=high_precision,
        return_lse=False,
    )
    total_out_ref_high_precision.backward(grad_total_out)
    (
        grad_total_q_ref_high_precision,
        grad_total_k_ref_high_precision,
        grad_total_v_ref_high_precision,
    ) = (
        total_q.grad,
        total_k.grad,
        total_v.grad,
    )
    return (
        total_out_ref_high_precision,
        grad_total_q_ref_high_precision,
        grad_total_k_ref_high_precision,
        grad_total_v_ref_high_precision,
    )


def gen_init_data(shape, device, dtype, test_bwd):
    # Prepare inputs
    q = torch.randn(
        *shape, device=device, dtype=dtype, requires_grad=True if test_bwd else False
    )
    k = torch.randn(
        *shape, device=device, dtype=dtype, requires_grad=True if test_bwd else False
    )
    v = torch.randn(
        *shape, device=device, dtype=dtype, requires_grad=True if test_bwd else False
    )
    dout = torch.randn(*shape, device=device, dtype=dtype)
    return q, k, v, dout


def generate_test_data(
    batch_size, total_seqlen, heads_num, hidden_dim, dtype, qkv_format, device
):
    # chunk_seq = total_seqlen if qkv_format == "thd" else total_seqlen // batch_size
    # random_cu_seqlens_list, max_seqlen = generate_random_samples(
    #     total_seqlen, batch_size
    # )

    random_cu_seqlens_list = [0, 512, 1024]
    max_seqlen = 512
    if qkv_format == "thd":
        shape = [total_seqlen, heads_num, hidden_dim]
    elif qkv_format == "bshd":
        shape = [batch_size, max_seqlen, heads_num, hidden_dim]
    elif qkv_format == "sbhd":
        shape = [max_seqlen, batch_size, heads_num, hidden_dim]
    q, k, v, dout = gen_init_data(shape, device, dtype, False)

    if qkv_format != "thd":
        q = fill_data_with_pad(q, random_cu_seqlens_list, qkv_format)
        k = fill_data_with_pad(k, random_cu_seqlens_list, qkv_format)
        v = fill_data_with_pad(v, random_cu_seqlens_list, qkv_format)
        dout = fill_data_with_pad(dout, random_cu_seqlens_list, qkv_format)

    return q, k, v, dout, random_cu_seqlens_list, max_seqlen


def generate_attn_ranges(random_cu_seqlens_list, valid_total_seqlen):
    ranges = AttnRanges.from_cu_seqlens(random_cu_seqlens_list, valid_total_seqlen)
    return ranges


def generate_attn_cu_seqlens(
    random_cu_seqlens_list, max_seqlen, batch_size, qkv_format, device
):
    cu_seqlens = torch.tensor(random_cu_seqlens_list, device=device, dtype=torch.int32)
    if qkv_format == "thd":
        cu_seqlens_padded = cu_seqlens
    else:
        cu_seqlens_padded = torch.arange(
            start=0,
            end=max_seqlen * batch_size + 1,
            step=max_seqlen,
            device=device,
        )
    return cu_seqlens, cu_seqlens_padded


def generate_non_pad_cu_seqlens(batch_size, per_seq_len, device):
    cu_seqlens = torch.arange(
        start=0, end=batch_size * per_seq_len + 1, step=per_seq_len, device=device
    )
    return cu_seqlens.to(torch.int32)


# generate random cu_seqlens list
def generate_random_samples(total_seqlen, NUM_SAMPLES):
    random_indices = (torch.randperm(total_seqlen - 1)[: NUM_SAMPLES - 1] + 1).tolist()
    random_indices = sorted(random_indices)
    random_indices = [0] + random_indices + [total_seqlen]

    max_seqlen = 0
    for i in range(len(random_indices) - 1):
        max_seqlen = max(max_seqlen, random_indices[i + 1] - random_indices[i])

    return random_indices, max_seqlen


# generate random cu_seqlens
def generate_random_cu_seqlens(total_seqlen, NUM_SAMPLES, device):
    random_indices = (torch.randperm(total_seqlen - 1)[: NUM_SAMPLES - 1] + 1).tolist()
    random_indices = sorted(random_indices)
    random_indices = [0] + random_indices + [total_seqlen]
    cu_seqlens = torch.tensor(random_indices, device=device, dtype=torch.int32)
    return cu_seqlens


def generate_random_ranges(total_seqlen, NUM_SAMPLES, device):
    random_indices = (torch.randperm(total_seqlen - 1)[: NUM_SAMPLES - 1] + 1).tolist()
    random_indices = sorted(random_indices)
    random_indices = [0] + random_indices + [total_seqlen]
    ranges = AttnRanges.from_cu_seqlens(random_indices, total_seqlen)
    return ranges


# fill pad token 0 for bshd, sbhd format
def fill_data_with_pad(
    data: torch.Tensor,
    hsot_cu_seqlens: List[int],
    qkv_format,
):
    batch_dim = qkv_format.index("b")
    batch_size = data.shape[batch_dim]
    new_data = data.clone()
    for i in range(batch_size):
        seqlen = hsot_cu_seqlens[i + 1] - hsot_cu_seqlens[i]
        if qkv_format == "bshd":
            new_data[i, seqlen:, :, :] = 0
        elif qkv_format == "sbhd":
            new_data[seqlen:, i, :, :] = 0
    return new_data


def collect_global_grad(attn, grad, ranges, valid_total_seqlen, name):
    grad_part = attn.dispatch(grad, ranges, valid_total_seqlen, name)
    grad_global = attn.undispatch(grad_part, name)
    return grad_global
