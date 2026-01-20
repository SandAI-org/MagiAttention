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

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, reduce, repeat
from triton.language.extra import libdevice

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.utils import max_fp_dtype, nvtx, to_higher_fp_dtype, to_triton_dtype


def safe_subtract(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Safely subtracts two tensors,
    where the subtraction results of two `-inf` will be set to `-inf`, instead of `nan`,
    as well as the corr. gradients will be set to zeros
    """
    two_neg_inf_mask = (a == b) & (a == float("-inf"))

    return (a - b).masked_fill(two_neg_inf_mask, float("-inf"))


def safe_lse(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Safely applies row-wise log-sum-exp to the input tensor,
    where all-``-inf`` rows's grad will be set to ``0``, instead of ``nan``

    NOTE: ``torch.logsumexp`` is numerically stabilized according to
    https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch-logsumexp

    and although all-``-inf`` rows will result in ``-inf`` correctly,
    yet the corr. gradients will result in ``nan``, similar to ``torch.nn.functional.softmax``'s backward behavior
    """
    all_neg_inf_mask = (x == float("-inf")).all(dim=dim, keepdim=keepdim)

    def safe_lse_bwd_hook(g):
        g.masked_fill_(all_neg_inf_mask, 0.0)
        return g

    if x.requires_grad:
        x.register_hook(safe_lse_bwd_hook)

    lse = torch.logsumexp(x, dim=dim, keepdim=keepdim)

    return lse.masked_fill(all_neg_inf_mask, float("-inf"))


def safe_softmax(
    x: torch.Tensor,
    lse: torch.Tensor | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Safely applies row-wise softmax to the input tensor,
    where all-``-inf`` rows will be set to all-zero rows
    as well as the corr. gradients will be set to zeros

    NOTE: ``torch.nn.functional.softmax`` has many limitations,
    and besides the safety operations mentioned above, it also has bugs
    when dtype is ``float64``, where the sum cannot be assured to ``1``

    Therefore, we recommend to use this function with ``lse`` given,
    where we will use the safer formula of softmax:
        ``softmax(x) = exp(x - lse)``

    NOTE: if ``lse`` is given, it requires to have the same ``ndim`` as ``x``
    and trivial size ``1`` at ``dim`` to be broadcastable to ``x``
    """

    if lse is not None:
        assert x.ndim == lse.ndim and lse.size(dim) == 1
        return torch.exp(safe_subtract(x, lse))

    all_neg_inf_mask = (x == float("-inf")).all(dim=dim, keepdim=True)

    def safe_softmax_bwd_hook(g):
        g.masked_fill_(all_neg_inf_mask, 0.0)
        return g

    if x.requires_grad:
        x.register_hook(safe_softmax_bwd_hook)

    sm = F.softmax(x, dim=dim)

    sm = sm.masked_fill(all_neg_inf_mask, 0.0)

    return sm


def softmax_bwd(dout: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Standard backward func for `out = softmax(inp)`"""

    diag_out = torch.diag_embed(out)
    outer_out = torch.einsum("...ij, ...ik -> ...ijk", out, out)
    dinp = torch.einsum("...ij, ...ijk -> ...ik", dout, diag_out - outer_out)

    return dinp


def sink_bwd(
    sink: torch.Tensor,
    lse: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    sink_layout: AttnSinkLayout = "sh",
    dsink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the backward pass for the gradient of sink tokens

    Args:
        sink (torch.Tensor): the sink tokens with shape specified by sink_layout:
            sh: [seqlen_sink, num_heads_q]
            shd: [seqlen_sink, num_heads_q, head_dim]
            ssh: [seqlen_q, seqlen_sink, num_heads_q]
        lse (torch.Tensor): log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
        o (torch.Tensor): output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
        do (torch.Tensor): the gradient of output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
        dsink (torch.Tensor | None, optional): the output buffer for the gradient of sink. Defaults to None.
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh".

    Returns:
        torch.Tensor: the gradient of sink with shape specified by sink_layout:
            sh: [seqlen_sink, num_heads_q]
            shd: [seqlen_sink, num_heads_q, head_dim]
            ssh: [seqlen_q, seqlen_sink, num_heads_q]
    """

    # prepare sink, lse
    sink_dtype = sink.dtype
    match sink_layout:
        case "sh":
            assert sink.ndim == 2
            sink = repeat(sink, "s_sink hq -> hq sq s_sink", sq=lse.size(0))
        case "ssh":
            assert sink.ndim == 3 and sink.size(0) == lse.size(0)
            sink = rearrange(sink, "sq s_sink hq -> hq sq s_sink")
        case "shd":
            raise NotImplementedError(f"sink_layout {sink_layout} is not supported yet")
        case _:
            raise ValueError(f"Invalid sink_layout {sink_layout}")
    sink = sink.to(lse.dtype)
    lse = rearrange(lse, "sq hq -> hq sq 1")

    # calculate delta = (o * do).sum(dim=-1)
    # where o.shape = [sq, nhq, d]
    #       do.shape = [sq, nhq, d]
    #       delta.shape = [nhq, sq, 1]
    delta = reduce((o * do).to(lse.dtype), "sq hq d -> hq sq 1", "sum")

    # calculat p_sink = exp(sink - lse)
    # where sink.shape = [nhq, sq, s_sink]
    #       lse.shape = [nhq, sq, 1]
    #       p_sink.shape = [nhq, sq, s_sink]
    p_sink = safe_softmax(sink, lse)

    match sink_layout:
        case "sh":
            # calculate dsink = p_sink.T x -delta
            # where p_sink.shape = [nhq, sq, s_sink]
            #       delta.shape = [[nhq, sq, 1]
            #       dsink.shape = [s_sink, nhq]
            dsink_ = reduce(p_sink * -delta, "nhq sq s_sink -> s_sink nhq", "sum")
        case "ssh":
            # calculate dsink = p_sink * -delta
            # where p_sink.shape = [nhq, sq, s_sink]
            #       delta.shape = [[nhq, sq, 1]
            #       dsink.shape = [nhq, sq, s_sink] -> [sq, s_sink, nhq]
            dsink_ = rearrange(p_sink * -delta, "nhq sq s_sink -> sq s_sink nhq")
        case "shd":
            raise NotImplementedError(f"sink_layout {sink_layout} is not supported yet")
        case _:
            raise ValueError(f"Invalid sink_layout {sink_layout}")

    # Copy to the output buffer if given
    if dsink is None:
        dsink = dsink_.to(sink_dtype)
    else:
        assert dsink.dtype == sink_dtype
        dsink.copy_(dsink_)

    return dsink


sink_bwd_compiled = torch.compile(dynamic=True)(sink_bwd)


def calc_lse_rescale_weight(
    lse_to_rescale: torch.Tensor,
    rescaled_lse: torch.Tensor,
) -> torch.Tensor:
    """Calculate the rescale weight to correct attention output
    given the lse to rescale (i.e. the old original lse)
    and the rescaled lse (i.e. the new lse)

    Args:
        lse_to_rescale (torch.Tensor): the old log-sum-exp tensor to rescale
            with shape: [seqlen_q, num_heads_q]
        rescaled_lse (torch.Tensor): the new rescaled log-sum-exp tensor
            with shape: [seqlen_q, num_heads_q]

    Returns:
        torch.Tensor: the rescale weight with shape: [seqlen_q, num_heads_q, 1]
    """
    # calculate the rescale weight with shape: [sq, nhq, 1]
    # formula: w = exp(lse_old - lse_new)
    return safe_subtract(lse_to_rescale, rescaled_lse).exp().unsqueeze(-1)


def calc_lse_sink(
    sink: torch.Tensor,
    seqlen_q: int,
    sink_layout: AttnSinkLayout = "sh",
) -> torch.Tensor:
    """Calculate the log-sum-exp of the sink tokens
    and repeat it to the given seqlen_q

    Args:
        sink (torch.Tensor): the sink tokens with shape specified by sink_layout:
            sh: [seqlen_sink, num_heads_q]
            shd: [seqlen_sink, num_heads_q, head_dim]
            ssh: [seqlen_q, seqlen_sink, num_heads_q]
        seqlen_q (int): the seqlen of the lse_sink
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh".

    Returns:
        torch.Tensor: the log-sum-exp of the sink tokens with shape: [seqlen_q, num_heads_q]
    """

    match sink_layout:
        case "sh":
            assert sink.ndim == 2
            # calculate lse_sink and repeat to seqlen_q
            lse_sink = (
                # shape: [s_sink, nhq] -> [1, nhq]
                safe_lse(sink, dim=0, keepdim=True)
                # shape: [1, nhq] -> [sq, nhq]
                # NOTE: we had better not use `expand` here
                # to avoid view conflict when involving in-place operations
                # .expand_as(seqlen_q, -1)
                .repeat(seqlen_q, 1)
            )
        case "ssh":
            assert sink.ndim == 3 and sink.size(0) == seqlen_q
            # calculate lse_sink and repeat to seqlen_q
            lse_sink = (
                # shape: [seqlen_q, s_sink, nhq] -> [seqlen_q, nhq]
                safe_lse(sink, dim=1, keepdim=False)
            )
        case "shd":
            raise NotImplementedError(f"{sink_layout} is not supported yet")
        case _:
            raise ValueError(f"Invalid sink_layout: {sink_layout}")

    return lse_sink


calc_lse_sink_compiled = torch.compile(dynamic=True)(calc_lse_sink)


def correct_attn_lse(
    lse1: torch.Tensor,
    lse2: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Corrects the log-sum-exp tensor for online attention.

    Args:
        lse1 (torch.Tensor): log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
        lse2 (torch.Tensor): log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
        inplace (bool, optional): whether to correct ``lse1`` inplace. Defaults to ``False``.

    Returns:
        torch.Tensor: corrected log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
    """

    assert lse1.dtype == lse2.dtype

    min_lse = to_higher_fp_dtype(torch.minimum(lse1, lse2), torch.float32)
    max_lse = to_higher_fp_dtype(torch.maximum(lse1, lse2), torch.float32)

    # formula derivation:
    # lse = log(exp(lse1) + exp(lse2))
    #     = lse1 + log(1 + exp(lse2 - lse1))
    #     = max_lse + log(1 + exp(min_lse - max_lse))
    #     = max_lse + log1p(exp(min_lse - max_lse))
    #     = max_lse + softplus(min_lse - max_lse)
    lse = max_lse + F.softplus(safe_subtract(min_lse, max_lse))

    return lse1.copy_(lse) if inplace else lse.to(lse1.dtype)


correct_attn_lse_compiled = torch.compile(dynamic=True)(correct_attn_lse)


def correct_attn_out(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
    lse: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Corrects the output tensor for online attention.

    Args:
        out1 (torch.Tensor): local output tensor1, with shape: [seqlen_q, num_heads_q, head_dim]
        lse1 (torch.Tensor): local lse for out1, with shape: [seqlen_q, num_heads_q]
        out2 (torch.Tensor): local output tensor2, with shape: [seqlen_q, num_heads_q, head_dim]
        lse2 (torch.Tensor): local lse for out2, with shape: [seqlen_q, num_heads_q]
        lse (torch.Tensor): global lse, with shape: [seqlen_q, num_heads_q]
        inplace (bool, optional): whether to correct ``out1`` inplace. Defaults to ``False``.

    Returns:
        torch.Tensor: corrected global output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
    """
    assert out1.dtype == out2.dtype and lse1.dtype == lse2.dtype == lse.dtype

    # calculate the rescale weight for out1 and out2 resp.
    w1, w2 = [calc_lse_rescale_weight(lsei, lse) for lsei in (lse1, lse2)]

    # correct out
    # with formula: out = w1 * out1 + w2 * out2
    if inplace:
        out1 *= w1
        out = out1.add_(w2 * out2)
    else:
        out = w1 * out1 + w2 * out2
        out = out.to(out1.dtype)

    return out


correct_attn_out_compiled = torch.compile(dynamic=True)(correct_attn_out)


@triton.jit
def _safe_subtract_exp(a, b):
    val = tl.exp(a - b)
    return tl.where(val != val, 0.0, val)  # nan -> 0.0


@triton.jit
def correct_out_lse_kernel(
    input_ptr,
    input_lse_ptr,
    output_ptr,
    output_lse_ptr,
    output_stride_s,
    output_stride_nh,
    output_stride_hd,
    input_stride_s,
    input_stride_nh,
    input_stride_hd,
    output_lse_stride_s,
    output_lse_stride_nh,
    input_lse_stride_s,
    input_lse_stride_nh,
    M,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    reduce_dtype: tl.constexpr,
):
    row_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    curr_m_start = row_block_idx * M_BLOCK
    cols = tl.arange(0, N_BLOCK)[None, :]
    rows = tl.arange(0, M_BLOCK)[:, None]
    row_mask = curr_m_start + rows < M

    out_idx = curr_m_start * output_stride_s + head_idx * output_stride_nh
    curr_out_ptr = output_ptr + out_idx
    out_offs = rows * output_stride_s + cols * output_stride_hd

    inp_idx = curr_m_start * input_stride_s + head_idx * input_stride_nh
    curr_inp_ptr = input_ptr + inp_idx
    inp_offs = rows * input_stride_s + cols * input_stride_hd

    out_lse_idx = curr_m_start * output_lse_stride_s + head_idx * output_lse_stride_nh
    curr_out_lse_ptr = output_lse_ptr + out_lse_idx
    out_lse_offs = rows * output_lse_stride_s

    inp_lse_idx = curr_m_start * input_lse_stride_s + head_idx * input_lse_stride_nh
    curr_inp_lse_ptr = input_lse_ptr + inp_lse_idx
    inp_lse_offs = rows * input_lse_stride_s

    # load output
    out = tl.load(
        curr_out_ptr + out_offs,
        mask=row_mask,
    ).to(reduce_dtype)

    # load output lse
    out_lse = tl.load(
        curr_out_lse_ptr + out_lse_offs,
        mask=row_mask,
    ).to(reduce_dtype)

    # load input
    inp = tl.load(
        curr_inp_ptr + inp_offs,
        mask=row_mask,
    ).to(reduce_dtype)

    # load input lse
    inp_lse = tl.load(
        curr_inp_lse_ptr + inp_lse_offs,
        mask=row_mask,
    ).to(reduce_dtype)

    # correct lse
    # formula derivation:
    # reduced_lse = log(exp(lse1) + exp(lse2))
    #             = lse1 + log(1 + exp(lse2 - lse1))
    #             = max_lse + log(1 + exp(min_lse - max_lse))
    #             = max_lse + log1p(exp(min_lse - max_lse))
    min_lse = tl.minimum(inp_lse, out_lse)
    max_lse = tl.maximum(inp_lse, out_lse)
    reduced_lse = max_lse + libdevice.log1p(_safe_subtract_exp(min_lse, max_lse))

    # get reduce weights: exp(lse_i - reduced_lse)
    out_weight = _safe_subtract_exp(out_lse, reduced_lse)
    inp_weight = _safe_subtract_exp(inp_lse, reduced_lse)

    # reduce output: w1 * out + w2 * inp
    out = out_weight * out + inp_weight * inp

    # reduce output lse
    out_lse = reduced_lse

    # store reduced output
    tl.store(curr_out_ptr + out_offs, out, mask=row_mask)

    # store reduced output lse
    tl.store(curr_out_lse_ptr + out_lse_offs, out_lse, mask=row_mask)


@nvtx.instrument_nvtx
def correct_attn_out_lse(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Corrects the attention result given all of the partial out and lse

    Args:
        out1 (torch.Tensor): the first partial out tensor to be corrected
        lse1 (torch.Tensor): the first partial lse tensor to be corrected
        out2 (torch.Tensor): the second partial out tensor to correct
        lse2 (torch.Tensor): the second partial lse tensor to correct
        inplace (bool, optional):
            whether to reduce the corrected results to ``out1`` and ``lse1`` inplace.
            Defaults to ``False``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the corrected out and lse

    Shape:
        out: [seqlen_q, num_heads_q, head_dim]
        lse: [seqlen_q, num_heads_q]
    """

    # ---   calculate meta   --- #

    # Determine the reduce dtype
    reduce_dtype = max_fp_dtype(lse1.dtype, lse2.dtype, torch.float32)
    reduce_dtype = to_triton_dtype(reduce_dtype)

    # ---   pre-process input/output   --- #

    # Prepare buffer
    output, output_lse = (
        (out1, lse1)
        if inplace
        else (
            out1.clone(),
            lse1.clone(),
        )
    )
    input, input_lse = (out2, lse2)

    # Calculate stride
    output_stride_s, output_stride_nh, output_stride_hd = output.stride()
    output_lse_stride_s, output_lse_stride_nh = output_lse.stride()
    input_stride_s, input_stride_nh, input_stride_hd = input.stride()
    input_lse_stride_s, input_lse_stride_nh = input_lse.stride()

    # ---   calculate grid size   --- #

    M, H, N = input.size()  # seqlen_q, num_heads_q, head_dim
    assert (
        triton.next_power_of_2(N) == N
    ), "head_dim must be power of 2 for triton kernel"

    N_BLOCK = N  # N block size, where one head dim is always a single n block
    M_BLOCK = 128  # M block size to shard seqlen_q
    NUM_M_BLOCKS = triton.cdiv(M, M_BLOCK)  # number of M blocks along seqlen_q

    grid = (NUM_M_BLOCKS, H)

    # ---   launch kernel   --- #

    correct_out_lse_kernel[grid](
        input,
        input_lse,
        output,
        output_lse,
        output_stride_s,
        output_stride_nh,
        output_stride_hd,
        input_stride_s,
        input_stride_nh,
        input_stride_hd,
        output_lse_stride_s,
        output_lse_stride_nh,
        input_lse_stride_s,
        input_lse_stride_nh,
        M,
        M_BLOCK,
        N_BLOCK,
        reduce_dtype,
    )

    return output, output_lse


def correct_attn_lse_with_sink(
    lse: torch.Tensor,
    sink: torch.Tensor,
    sink_layout: AttnSinkLayout = "sh",
    inplace: bool = False,
) -> torch.Tensor:
    """
    Corrects the log-sum-exp tensor with sink tokens

    Args:
        lse (torch.Tensor): log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
        sink (torch.Tensor): the sink tokens with shape specified by sink_layout:
            sh: [seqlen_sink, num_heads_q]
            shd: [seqlen_sink, num_heads_q, head_dim]
            ssh: [seqlen_q, seqlen_sink, num_heads_q]
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh".
        inplace (bool, optional): whether to correct ``lse`` inplace. Defaults to ``False``.


    Returns:
        torch.Tensor: corrected log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
    """
    lse_sink = calc_lse_sink(
        sink=sink,
        seqlen_q=lse.size(0),
        sink_layout=sink_layout,
    )

    return correct_attn_lse(lse, lse_sink, inplace=inplace)


def correct_attn_out_with_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    sink_layout: AttnSinkLayout = "sh",
    inplace: bool = False,
) -> torch.Tensor:
    """
    Corrects the output tensor with sink tokens

    Args:
        out (torch.Tensor): output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
        lse (torch.Tensor): lse tensor, with shape: [seqlen_q, num_heads_q]
        sink (torch.Tensor): the sink tokens with shape specified by sink_layout:
            sh: [seqlen_sink, num_heads_q]
            shd: [seqlen_sink, num_heads_q, head_dim]
            ssh: [seqlen_q, seqlen_sink, num_heads_q]
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh".
        inplace (bool, optional): whether to correct ``out`` inplace. Defaults to ``False``.

    Returns:
        torch.Tensor: corrected output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
    """
    lse_sink = calc_lse_sink(
        sink=sink,
        seqlen_q=lse.size(0),
        sink_layout=sink_layout,
    )

    # calculate the rescale weight with shape: [sq, nhq, 1]
    # formula derivation:
    # w = exp(lse_old - lse_new)
    #   = exp(lse - log(exp(lse) + exp(lse_sink)))
    #   = exp(lse) / (exp(lse) + exp(lse_sink))
    #   = 1 / (1 + exp(lse_sink - lse))
    w = torch.reciprocal(1 + calc_lse_rescale_weight(lse_sink, lse))

    return out.mul_(w) if inplace else (out * w).to(out.dtype)


def correct_attn_out_lse_with_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    sink_layout: AttnSinkLayout = "sh",
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Corrects the output tensor and log-sum-exp tensor with sink tokens

    Args:
        out (torch.Tensor): output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
        lse (torch.Tensor): lse tensor, with shape: [seqlen_q, num_heads_q]
        sink (torch.Tensor): the sink tokens with shape specified by sink_layout:
            sh: [seqlen_sink, num_heads_q]
            shd: [seqlen_sink, num_heads_q, head_dim]
            ssh: [seqlen_q, seqlen_sink, num_heads_q]
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh".
        inplace (bool, optional): whether to correct ``out`` inplace. Defaults to ``False``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: corrected output and lse
            - out: corrected output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
            - lse: corrected lse tensor, with shape: [seqlen_q, num_heads_q]
    """
    lse_sink = calc_lse_sink(
        sink=sink,
        seqlen_q=lse.size(0),
        sink_layout=sink_layout,
    )

    corr_lse = correct_attn_lse(lse, lse_sink, inplace=False)
    w = calc_lse_rescale_weight(lse, corr_lse)

    out = out.mul_(w) if inplace else (out * w).to(out.dtype)
    lse = lse.copy_(corr_lse) if inplace else corr_lse

    return out, lse


correct_attn_out_lse_with_sink_compiled = torch.compile(dynamic=True)(
    correct_attn_out_lse_with_sink
)
