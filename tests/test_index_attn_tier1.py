#!/usr/bin/env python3
"""Direct (no fork) test of IndexAttn Tier1 configs."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from magi_attention.utils import set_random_seed

from magi_attention.utils.sparse_utils import (
    build_index_attn_indices,
    get_sdpa_mask_from_index_attn_indices,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test_attn"))
from test_index_attn import (
    _run_sparse_attn_and_get_output,
    _compare_against_sdpa,
    SEED,
)

TIER1_CONFIGS = [
    {"name": "mqa128_packgqa", "B": 1, "S": 256, "NHQ": 128, "NHK": 1, "topk": 128, "pack_gqa": True},
    {"name": "mqa64_packgqa",  "B": 1, "S": 256, "NHQ": 64,  "NHK": 1, "topk": 128, "pack_gqa": True},
    {"name": "mqa32_packgqa",  "B": 1, "S": 256, "NHQ": 32,  "NHK": 1, "topk": 128, "pack_gqa": True},
    {"name": "mqa16_packgqa_swapab", "B": 1, "S": 256, "NHQ": 16, "NHK": 1, "topk": 128, "pack_gqa": True, "swap_ab": True},
]

def run_config(cfg):
    set_random_seed(SEED)
    B = cfg["B"]
    S = cfg.get("S", None)
    S_kv = cfg.get("S_kv", S)
    S_q = cfg.get("S_q", min(S_kv, 256))
    NHQ, NHK = cfg["NHQ"], cfg["NHK"]
    D = cfg.get("D", 128)
    topk = cfg["topk"]
    default_max = max(topk) if isinstance(topk, list) else topk
    max_topk = cfg.get("max_topk", default_max)
    pack_gqa = cfg.get("pack_gqa", True)
    swap_ab = cfg.get("swap_ab", False)
    ref_block_size = cfg.get("ref_block_size", None)
    k_block_size = cfg.get("k_block_size", 1)
    dtype = cfg.get("dtype", torch.bfloat16)
    atol = cfg.get("atol", 5e-2)
    device = "cuda"

    t = time.time()
    index_attn_indices = build_index_attn_indices(B, NHK, S_q, S_kv, topk, max_topk, device, k_block_size=k_block_size)
    print(f"    _build_index_attn_indices: {time.time()-t:.2f}s", flush=True)

    q = torch.randn(B, S_q, NHQ, D, dtype=dtype, device=device)
    k = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)
    v = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)

    t = time.time()
    print(f"    _run_sparse_attn_and_get_output ...", flush=True)
    o_ffa = _run_sparse_attn_and_get_output(q, k, v, index_attn_indices, B, S_q, S_kv, NHQ, NHK,
                                             pack_gqa=pack_gqa, swap_ab=swap_ab,
                                             ref_block_size=ref_block_size, k_block_size=k_block_size)
    torch.cuda.synchronize()
    print(f"    _run_sparse_attn_and_get_output: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    sdpa_mask = get_sdpa_mask_from_index_attn_indices(index_attn_indices, B, NHQ, NHK, S_q, S_kv, device, k_block_size=k_block_size)
    print(f"    _build_sdpa_mask: {time.time()-t:.2f}s", flush=True)

    t = time.time()
    test_case = f"[{cfg['name']}]"
    _compare_against_sdpa(o_ffa, q, k, v, sdpa_mask, B, NHQ, NHK, atol, test_case)
    print(f"    _compare_against_sdpa: {time.time()-t:.2f}s", flush=True)

if __name__ == "__main__":
    for cfg in TIER1_CONFIGS:
        t0 = time.time()
        print(f">>> [{cfg['name']}] START", flush=True)
        run_config(cfg)
        print(f">>> [{cfg['name']}] PASSED ({time.time()-t0:.1f}s)", flush=True)
    print("\nAll Tier1 configs PASSED!", flush=True)
