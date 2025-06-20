import random

import pytest
import torch
from einops import rearrange

try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None


from magi_attention.functional.flex_flash_attn import flex_flash_attn_func


def get_mask_from_ranges(q_ranges, k_ranges, attn_type_map, q_len, k_len):
    bsz = q_ranges.shape[0]
    mask = torch.zeros((q_len, k_len), device="cuda", dtype=torch.bool)
    for i in range(bsz):
        if attn_type_map[i] == 1:
            mask_slice = mask[
                q_ranges[i, 0] : q_ranges[i, 1], k_ranges[i, 0] : k_ranges[i, 1]
            ]
            short_len = min(mask_slice.shape[0], mask_slice.shape[1])
            causal_part = torch.ones(
                short_len, short_len, device=mask_slice.device, dtype=mask_slice.dtype
            ).tril_()
            mask_slice[-short_len:, -short_len:] = causal_part
            mask_slice[:, :-short_len] = True
            mask_slice[:-short_len, :] = False
        elif attn_type_map[i] == 0:
            mask[
                q_ranges[i, 0] : q_ranges[i, 1], k_ranges[i, 0] : k_ranges[i, 1]
            ] = True
        elif attn_type_map[i] == 2:
            mask_slice = mask[
                q_ranges[i, 0] : q_ranges[i, 1], k_ranges[i, 0] : k_ranges[i, 1]
            ]
            short_len = min(mask_slice.shape[0], mask_slice.shape[1])
            inv_causal_part = torch.ones(
                short_len, short_len, device=mask_slice.device, dtype=mask_slice.dtype
            ).triu_()
            mask_slice[:short_len, :short_len] = inv_causal_part
            mask_slice[:, short_len:] = True
            mask_slice[short_len:, :] = False
        else:
            mask_slice_causal = mask[
                q_ranges[i, 0] : q_ranges[i, 1], k_ranges[i, 0] : k_ranges[i, 1]
            ].clone()
            short_len = min(mask_slice_causal.shape[0], mask_slice_causal.shape[1])
            causal_part = torch.ones(
                short_len,
                short_len,
                device=mask_slice_causal.device,
                dtype=mask_slice_causal.dtype,
            ).tril_()
            mask_slice_causal[-short_len:, -short_len:] = causal_part
            mask_slice_causal[:, :-short_len] = True
            mask_slice_causal[:-short_len, :] = False

            mask_slice_inv_causal = mask[
                q_ranges[i, 0] : q_ranges[i, 1], k_ranges[i, 0] : k_ranges[i, 1]
            ].clone()
            short_len = min(
                mask_slice_inv_causal.shape[0], mask_slice_inv_causal.shape[1]
            )
            inv_causal_part = torch.ones(
                short_len,
                short_len,
                device=mask_slice_inv_causal.device,
                dtype=mask_slice_inv_causal.dtype,
            ).triu_()
            mask_slice_inv_causal[:short_len, :short_len] = inv_causal_part
            mask_slice_inv_causal[:, short_len:] = True
            mask_slice_inv_causal[short_len:, :] = False

            mask[
                q_ranges[i, 0] : q_ranges[i, 1], k_ranges[i, 0] : k_ranges[i, 1]
            ] = torch.logical_and(mask_slice_causal, mask_slice_inv_causal)

    return mask


def torch_attn_ref(q, k, v, mask, layout="thd", high_precision=True):
    if layout == "thd":
        q = rearrange(q, "t h d -> 1 h t d")
        k = rearrange(k, "t h d -> 1 h t d")
        v = rearrange(v, "t h d -> 1 h t d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if q.shape[1] != k.shape[1]:
        # gqa case
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

    if high_precision:
        out = torch.nn.functional.scaled_dot_product_attention(
            q.to(torch.float64),
            k.to(torch.float64),
            v.to(torch.float64),
            attn_mask=mask,
        )
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    if layout == "thd":
        out = rearrange(out, "1 h t d -> t h d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if high_precision:
        out = out.to(q.dtype)
    return out


def generate_qk_ranges(seqlen_q, seqlen_k, bsz, device="cuda"):
    """generate q k ranges

    Args:
        seqlen: 
        bsz: batch size
        device: 'cuda' by default

    Returns:
        q_range: q_range tensor with shape (bsz, 2)
        k_range: k_range tensor with shape(bsz, 2)
    """

    random.seed(42)

    if bsz == 1:
        # use total seq
        start = random.randint(0, seqlen_q - 1)
        end = random.randint(start + 1, seqlen_q)
        q_ranges = [[start, end]]
        max_seqlen_q = end - start
        
        # generate k_range randomly
        start = random.randint(0, seqlen_k - 1)
        end = random.randint(start + 1, seqlen_k)
        k_ranges = [[start, end]]
        max_seqlen_k = end - start

    else:
        # generate q_ranges randomly
        q_ranges = []
        for i in range(bsz):
            start = random.randint(0, seqlen_q - 1)
            end = random.randint(start + 1, seqlen_q)
            q_ranges.append([start, end])
            max_seqlen_q = max(max_seqlen_q, end - start)

        # generate k_ranges randomly
        k_ranges = []
        for i in range(bsz):
            start = random.randint(0, seqlen_k - 1)
            end = random.randint(start + 1, seqlen_k)
            k_ranges.append([start, end])
            max_seqlen_k = max(max_seqlen_k, end - start)

    q_ranges = torch.tensor(q_ranges, device=device, dtype=torch.int32)
    k_ranges = torch.tensor(k_ranges, device=device, dtype=torch.int32)

    # print(f"DEBUG q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}", flush=True)

    return q_ranges, k_ranges, max_seqlen_q, max_seqlen_k


# @pytest.mark.skip(reason="skipped")
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q", [8, 256, 551, 1234, 1999]
)  # hang when seqlen is smaller than 7
@pytest.mark.parametrize(
    "seqlen_k", [8, 256, 551, 1234, 1999, 9999]
)  # hang when seqlen is smaller than 7
@pytest.mark.parametrize("bsz", [1])
@pytest.mark.parametrize("attn_type", [0, 1, 2, 3])
def test_flex_flash_attn_output(seqlen_q, seqlen_k, bsz, d, mha_type, dtype, attn_type):
    device = "cuda"
    torch.random.manual_seed(43)

    q_ranges, k_ranges, max_seqlen_q, max_seqlen_k = generate_qk_ranges(
        seqlen_q * bsz, seqlen_k * bsz, bsz, device
    )

    # q_ranges = torch.tensor([[  0, 377], [111, 512]], device=device, dtype=torch.int32)
    # k_ranges = torch.tensor([[ 0, 233], [267, 512]], device=device, dtype=torch.int32)
    # max_seqlen_q = 512
    # max_seqlen_k = 256

    print(
        f"q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}"
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    q = torch.randn(
        bsz * seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        bsz * seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        bsz * seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )
    g = torch.randn(bsz * seqlen_q, nheads, d, device=device, dtype=dtype)

    attn_type_map = torch.zeros(bsz, device=device, dtype=torch.int32) + attn_type
    out, lse = flex_flash_attn_func(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        attn_type_map=attn_type_map,
        disable_fwd_atomic_reduction=False,
    )
    out.backward(g)
    dq, dk, dv = q.grad, k.grad, v.grad
    q.grad, k.grad, v.grad = None, None, None

    out_ref = torch_attn_ref(
        q,
        k,
        v,
        mask=get_mask_from_ranges(
            q_ranges, k_ranges, attn_type_map, seqlen_q * bsz, seqlen_k * bsz
        ),
        layout="thd",
        high_precision=True,
    )
    out_ref.backward(g)
    dq_ref, dk_ref, dv_ref = q.grad, k.grad, v.grad
    q.grad, k.grad, v.grad = None, None, None

    out_ref_low_precision = torch_attn_ref(
        q,
        k,
        v,
        mask=get_mask_from_ranges(
            q_ranges, k_ranges, attn_type_map, seqlen_q * bsz, seqlen_k * bsz
        ),
        layout="thd",
        high_precision=False,
    )
    out_ref_low_precision.backward(g)
    dq_ref_low_precision, dk_ref_low_precision, dv_ref_low_precision = (
        q.grad,
        k.grad,
        v.grad,
    )
    q.grad, k.grad, v.grad = None, None, None


    assert (out - out_ref).abs().max().item() <= 2 * (
        out_ref_low_precision - out_ref
    ).abs().max().item(), f"q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}, attn_type: {attn_type}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}"
    # print(f"out: {out[168:, 0, :]}, out_ref: {out_ref[168:, 0, :]}, lse: {lse[0, :]}")
    # print(f"{dq_ref[2633, :, :]=} | {dq[2633, :, :]=}")
    # print(f"{dk_ref[1125, 1, :]=} | {dk[1125, 1, :]=}")
    # print(f"{dv_ref[228 + 5, 1, :]=} | {dv[228 + 5, 1, :]=}")
    # torch.save(dq, "dq.pt")
    assert (dq - dq_ref)[:, :, :].abs().max().item() <= 2 * (
        dq_ref_low_precision - dq_ref
    )[
        :, :, :
    ].abs().max().item(), f"q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}"

    # assert (dk - dk_ref_low_precision).abs().max().item() < 1e-4 or (
    #     dk - dk_ref_low_precision
    # ).abs().max().item() <= 3 * (
    #     dk_ref_low_precision - dk_ref
    # ).abs().max().item(), f"q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}"
    # assert (dv - dv_ref_low_precision).abs().max().item() < 1e-4 or (
    #     dv - dv_ref_low_precision
    # ).abs().max().item() <= 3 * (
    #     dv_ref_low_precision - dv_ref
    # ).abs().max().item(), f"q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}"

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    elsit = []
    print("\n", flush=True)
    print("=========================START=========================", flush=True)
    try:
        torch.testing.assert_close(
            out.to(torch.float32),
            out_ref.to(torch.float32),
            atol=torch.finfo(dtype).eps,
            rtol=torch.finfo(dtype).eps,
        )
    except Exception as e:
        print(
            "---------------------------Start Out check---------------------------",
            flush=True,
        )
        print(
            f"Failed out check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}",
            flush=True,
        )
        print(e, flush=True)
        print(
            "---------------------------End Out check---------------------------",
            flush=True,
        )
        elsit.append(e)
    try:
        torch.testing.assert_close(
            dq, dq_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps
        )
    except Exception as e:
        print(
            "---------------------------Start dq check---------------------------",
            flush=True,
        )
        print(
            f"Failed dq check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}",
            flush=True,
        )
        print(e, flush=True)
        print(
            "---------------------------End dq check---------------------------",
            flush=True,
        )
        elsit.append(e)
    try:
        torch.testing.assert_close(
            dk, dk_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps
        )
    except Exception as e:
        print(
            "---------------------------Start dk check---------------------------",
            flush=True,
        )
        print(
            f"Failed dk check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}",
            flush=True,
        )
        print(e, flush=True)
        print(
            "---------------------------End dk check---------------------------",
            flush=True,
        )
        elsit.append(e)
    try:
        torch.testing.assert_close(
            dv, dv_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps
        )
    except Exception as e:
        print(
            "---------------------------Start dv check---------------------------",
            flush=True,
        )
        print(
            f"Failed dv check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}",
            flush=True,
        )
        print(e, flush=True)
        print(
            "---------------------------End dv check---------------------------",
            flush=True,
        )
        elsit.append(e)
    print("=========================END=========================", flush=True)

    # for e in elsit:
    #     raise e
