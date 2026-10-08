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

"""Agile harness for ``ffa_bwd_sm100_index`` (SM100 DSA backward).

Every q block of 128 tokens attends, per kv head, to its own ``topk`` K tokens
(``index_sparse_indices`` of shape ``(ceil(total_q / 128), num_heads_kv, topk)``,
-1 padded at the tail). ``correct`` runs the fp64 reference forward over the
dense mask built from the indices; ``bench`` uses a chunked gather forward.
Backward always uses ``ffa_bwd_sm100_index``.

Usage (from the repo root):
    python tests/test_kernel/cutedsl/test_ffa_bwd_sm100_index.py smoke
    python tests/test_kernel/cutedsl/test_ffa_bwd_sm100_index.py correct
    python tests/test_kernel/cutedsl/test_ffa_bwd_sm100_index.py bench
"""

import sys
from collections.abc import Callable
from functools import partial

import torch  # noqa: E402

from magi_attention.kernel.cutedsl.ffa_bwd_sm100_index import (  # noqa: E402
    ffa_bwd_sm100_index,
)
from magi_attention.testing.precision import (  # noqa: E402
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.testing.ref_attn import ref_attn_func  # noqa: E402

NUM_HEADS_Q = 64
NUM_HEADS_KV = 8
HEAD_DIM = 128
TILE_M = 128

_GRAD_RTOL = {
    "dq": {torch.bfloat16: 0.3, torch.float16: 0.2},
    "dk": {torch.bfloat16: 0.15, torch.float16: 0.08},
    "dv": {torch.bfloat16: 0.05, torch.float16: 0.05},
}
_GRAD_RTOL_DEFAULT = {"dq": 0.2, "dk": 0.08, "dv": 0.05}


def _num_q_blocks(total_q: int) -> int:
    return (total_q + TILE_M - 1) // TILE_M


def _build_indices(
    total_q: int,
    total_k: int,
    topk: int,
    *,
    var_len: bool,
    seed: int,
    device: str,
) -> torch.Tensor:
    """Distinct sorted K ids per (q block, kv head); ``var_len`` -1 pads a random tail."""
    gen = torch.Generator(device=device).manual_seed(seed)
    num_rows = _num_q_blocks(total_q) * NUM_HEADS_KV
    keys = torch.rand(num_rows, total_k, generator=gen, device=device)
    ids = keys.topk(topk, dim=-1, largest=False).indices.sort(dim=-1).values
    if var_len:
        lens = torch.randint(1, topk + 1, (num_rows, 1), generator=gen, device=device)
        ids = torch.where(torch.arange(topk, device=device) < lens, ids, -1)
    return ids.to(torch.int32).view(-1, NUM_HEADS_KV, topk).contiguous()


def _build_dense_indices(total_q: int, total_k: int, device: str) -> torch.Tensor:
    """Every q block attends to every K token: equivalent to a full mask."""
    ids = torch.arange(total_k, dtype=torch.int32, device=device)
    return ids.expand(_num_q_blocks(total_q), NUM_HEADS_KV, total_k).contiguous()


def _build_mask(indices: torch.Tensor, total_q: int, total_k: int) -> torch.Tensor:
    """(num_heads_q, total_q, total_k) bool mask equivalent to ``indices``."""
    num_q_blocks, num_heads_kv, _ = indices.shape
    # -1 padding scatters into a dropped extra column
    cols = torch.where(indices >= 0, indices, total_k).long()
    mask_blk = torch.zeros(
        num_q_blocks, num_heads_kv, total_k + 1, dtype=torch.bool, device=indices.device
    )
    mask_blk.scatter_(2, cols, True)
    mask_kv = (
        mask_blk[..., :total_k]
        .permute(1, 0, 2)
        .repeat_interleave(TILE_M, dim=1)[:, :total_q]
    )
    return mask_kv.repeat_interleave(NUM_HEADS_Q // num_heads_kv, dim=0)


def _dsa_fwd_gather(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    chunk_blocks: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 forward over the gathered K/V rows, q blocks in chunks; returns (out, lse)."""
    total_q = q.shape[0]
    num_q_blocks, num_heads_kv, _ = indices.shape
    group = NUM_HEADS_Q // num_heads_kv
    out = torch.empty_like(q)
    lse = torch.empty(total_q, NUM_HEADS_Q, dtype=torch.float32, device=q.device)
    head_kv = torch.arange(num_heads_kv, device=q.device)[None, :, None]
    for b0 in range(0, num_q_blocks, chunk_blocks):
        b1 = min(b0 + chunk_blocks, num_q_blocks)
        q0, q1 = b0 * TILE_M, min(b1 * TILE_M, total_q)
        idx = indices[b0:b1]
        valid = idx >= 0
        rows = idx.clamp(min=0).long()
        # k_sel / v_sel: (nb, nhk, topk, d)
        k_sel = k[rows, head_kv].float()
        v_sel = v[rows, head_kv].float()
        # q_blk: (nb, tile_m, nhk, group, d), zero rows past total_q
        q_blk = torch.zeros(
            (b1 - b0) * TILE_M, NUM_HEADS_Q, q.shape[-1], device=q.device
        )
        q_blk[: q1 - q0] = q[q0:q1].float()
        q_blk = q_blk.view(b1 - b0, TILE_M, num_heads_kv, group, -1)
        s = torch.einsum("bmhgd,bhtd->bhgmt", q_blk, k_sel) * softmax_scale
        s = s.masked_fill(~valid[:, :, None, None, :], float("-inf"))
        lse_blk = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse_blk[..., None])
        o = torch.einsum("bhgmt,bhtd->bmhgd", p, v_sel)
        out[q0:q1] = o.reshape(-1, NUM_HEADS_Q, q.shape[-1])[: q1 - q0].to(q.dtype)
        lse[q0:q1] = lse_blk.permute(0, 3, 1, 2).reshape(-1, NUM_HEADS_Q)[: q1 - q0]
    return out, lse


def _ref_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float,
    high_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    out, _meta = ref_attn_func(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        mask=mask,
        softmax_scale=softmax_scale,
        layout="thd",
        backend="sdpa",
        high_precision=high_precision,
    )
    out.backward(do)
    assert q_ref.grad is not None and k_ref.grad is not None and v_ref.grad is not None
    return q_ref.grad, k_ref.grad, v_ref.grad


def _assert_grad(
    name: str,
    actual: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    dtype: torch.dtype,
    test_case: str,
) -> None:
    assert not actual.isnan().any(), f"{test_case} => {name}: NaN in the output"
    rtol = _GRAD_RTOL[name].get(dtype, _GRAD_RTOL_DEFAULT[name])
    atol = EPSILON
    actual_norm = calc_inf_norm(actual, high)
    ref_norm = calc_inf_norm(low, high)
    limit = max(0.0, NORM_RTOL_RATIO * ref_norm)
    if actual_norm > limit:
        raise AssertionError(
            f"{test_case} => {name}: Linf {actual_norm} > "
            f"{NORM_RTOL_RATIO} x low-vs-high Linf {ref_norm}"
        )
    mismatch = extract_mismatch_threshold(
        actual=low,
        expected=high,
        atol=atol,
        rtol=rtol,
        mismatch_thres_ratio=MISMATCH_THRES_RATIO,
        min_mismatch_thres=0.0,
        max_mismatch_thres=MAX_MISMATCH_THRES,
    )
    assert_close(
        actual,
        high,
        atol=atol,
        rtol=rtol,
        mismatch_threshold=mismatch,
        test_case=f"{test_case} => {name}",
    )


def _run_case(
    name: str,
    total_q: int,
    total_k: int,
    topk: int,
    dtype: torch.dtype,
    *,
    var_len: bool = False,
    dense: bool = False,
) -> None:
    """fp64 reference forward, new backward, check grads against the fp64 reference."""
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(total_q, NUM_HEADS_Q, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(total_k, NUM_HEADS_KV, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(total_k, NUM_HEADS_KV, HEAD_DIM, device=device, dtype=dtype)
    do = torch.randn_like(q)
    softmax_scale = HEAD_DIM**-0.5
    if dense:
        indices = _build_dense_indices(total_q, total_k, device)
    else:
        indices = _build_indices(
            total_q, total_k, topk, var_len=var_len, seed=0, device=device
        )
    mask = _build_mask(indices, total_q, total_k)

    out, meta = ref_attn_func(
        q=q,
        k=k,
        v=v,
        mask=mask,
        softmax_scale=softmax_scale,
        layout="thd",
        backend="sdpa",
        high_precision=True,
        return_lse=True,
    )
    assert meta.lse is not None
    # lse of a per-head mask comes out as (1, total_q, num_heads_q)
    lse = meta.lse.reshape(total_q, NUM_HEADS_Q).contiguous()

    # NaN-filled outputs: every element must be overwritten by the backward
    dq, dk, dv = ffa_bwd_sm100_index(
        q,
        k,
        v,
        out,
        do,
        lse,
        indices,
        softmax_scale=softmax_scale,
        dq=torch.full_like(q, float("nan")),
        dk=torch.full_like(k, float("nan")),
        dv=torch.full_like(v, float("nan")),
    )

    dq_hi, dk_hi, dv_hi = _ref_grads(
        q, k, v, do, mask, softmax_scale, high_precision=True
    )
    dq_lo, dk_lo, dv_lo = _ref_grads(
        q, k, v, do, mask, softmax_scale, high_precision=False
    )
    test_case = f"{name} dtype={dtype}"
    _assert_grad("dq", dq, dq_hi, dq_lo, dtype, test_case)
    _assert_grad("dk", dk, dk_hi, dk_lo, dtype, test_case)
    _assert_grad("dv", dv, dv_hi, dv_lo, dtype, test_case)
    print(f"PASS {test_case}")


def smoke() -> None:
    """q 1024 x k 1024, topk 256, bf16. Short functional check."""
    _run_case(
        name="smoke_q1024_k1024_topk256",
        total_q=1024,
        total_k=1024,
        topk=256,
        dtype=torch.bfloat16,
    )


def correct() -> None:
    """Fixed / padded / ragged / dense-equivalent indices, bf16 and fp16."""
    for dtype in (torch.bfloat16, torch.float16):
        # topk a multiple of tile_n, all ids valid
        _run_case("q1024_k1024_topk256", 1024, 1024, 256, dtype)
        # random valid length per row: partial tail tiles, -1 padding
        _run_case("q1024_k1024_topk256_var", 1024, 1024, 256, dtype, var_len=True)
        # partial last q block, topk not a multiple of tile_n, total_k != total_q
        _run_case("q1000_k1500_topk200", 1000, 1500, 200, dtype)
        # indices = arange(total_k): same function as the full mask
        _run_case("q1024_k1024_dense", 1024, 1024, 1024, dtype, dense=True)


def _time_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench() -> None:
    """Time backward only (preprocess + DSA kernel + dK/dV postprocess) and print TFLOPS."""
    device = "cuda"
    dtype = torch.bfloat16
    softmax_scale = HEAD_DIM**-0.5
    warmup = 5
    iters = 10
    print(
        "ffa_bwd_sm100_index: "
        f"heads_q={NUM_HEADS_Q} heads_kv={NUM_HEADS_KV} "
        f"head_dim={HEAD_DIM} dtype={dtype}"
    )
    print(f"{'seqlen':>8} {'topk':>6} {'ms':>10} {'TFLOPS':>10}")
    for seqlen in [4096, 8192, 16384, 32768]:
        for topk in [512, 1024, 2048]:
            torch.manual_seed(0)
            q = torch.randn(seqlen, NUM_HEADS_Q, HEAD_DIM, device=device, dtype=dtype)
            k = torch.randn(seqlen, NUM_HEADS_KV, HEAD_DIM, device=device, dtype=dtype)
            v = torch.randn(seqlen, NUM_HEADS_KV, HEAD_DIM, device=device, dtype=dtype)
            do = torch.randn_like(q)
            indices = _build_indices(
                seqlen, seqlen, topk, var_len=False, seed=0, device=device
            )
            out, lse = _dsa_fwd_gather(q, k, v, indices, softmax_scale)
            run_bwd = partial(
                ffa_bwd_sm100_index,
                q,
                k,
                v,
                out,
                do,
                lse,
                indices,
                softmax_scale=softmax_scale,
            )
            ms = _time_ms(run_bwd, warmup=warmup, iters=iters)
            # 4 * area * heads * head_dim for fwd, x2.5 for bwd (with recompute)
            flops = 2.5 * 4 * seqlen * topk * NUM_HEADS_Q * HEAD_DIM
            tflops = flops / ms * 1e-9
            print(f"{seqlen:8d} {topk:6d} {ms:10.3f} {tflops:10.2f}")


_SCRIPT = "tests/test_kernel/cutedsl/test_ffa_bwd_sm100_index.py"
_MODES = {
    "smoke": smoke,
    "correct": correct,
    "bench": bench,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _MODES:
        print(f"usage: python {_SCRIPT} smoke|correct|bench", file=sys.stderr)
        raise SystemExit(2)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    _MODES[sys.argv[1]]()


if __name__ == "__main__":
    main()
