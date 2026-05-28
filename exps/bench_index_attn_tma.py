#!/usr/bin/env python3
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Benchmark: IndexAttn BWD — TMA contiguous vs cp.async scatter-load
#
# Usage:
#   python exps/bench_index_attn_tma.py [--warmup N] [--repeat N] [--seqlen S]

import argparse

import torch
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func


def build_contiguous_indices(B, S, NHK, topk, device):
    total_q = B * S
    indices = torch.full((total_q, NHK, topk), -1, dtype=torch.int32, device=device)
    for b in range(B):
        for qi in range(S):
            row = b * S + qi
            for h in range(NHK):
                ids = torch.arange(topk, device=device) * NHK + h + b * S * NHK
                indices[row, h, :topk] = ids.int()
    return indices


def build_random_indices(B, S, NHK, topk, device):
    total_q = B * S
    indices = torch.full((total_q, NHK, topk), -1, dtype=torch.int32, device=device)
    for b in range(B):
        for qi in range(S):
            row = b * S + qi
            for h in range(NHK):
                perm = torch.randperm(S, device=device)[:topk].sort().values
                ids = (b * S + perm) * NHK + h
                indices[row, h, :topk] = ids.int()
    return indices


def bench_bwd(q, k, v, indices, warmup, repeat):
    torch.cuda.synchronize()
    for _ in range(warmup):
        q.grad = k.grad = v.grad = None
        o, _ = flex_flash_attn_func(
            q,
            k,
            v,
            index_attn_indices=indices,
            q_block_size=1,
            k_block_size=1,
        )
        do = torch.randn_like(o)
        o.backward(do)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        q.grad = k.grad = v.grad = None
        o, _ = flex_flash_attn_func(
            q,
            k,
            v,
            index_attn_indices=indices,
            q_block_size=1,
            k_block_size=1,
        )
        do = torch.randn_like(o)
        torch.cuda.synchronize()
        start_events[i].record()
        o.backward(do)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nheads_q", type=int, default=4)
    parser.add_argument("--nheads_k", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=128)
    args = parser.parse_args()

    B, S, NHQ, NHK, D, topk = (
        args.batch,
        args.seqlen,
        args.nheads_q,
        args.nheads_k,
        args.headdim,
        args.topk,
    )
    device = "cuda"

    print(f"Config: B={B} S={S} NHQ={NHQ} NHK={NHK} D={D} topk={topk}")
    print(f"Warmup={args.warmup} Repeat={args.repeat}")
    print()

    for idx_type, build_fn in [
        ("contiguous", build_contiguous_indices),
        ("random", build_random_indices),
    ]:
        indices = build_fn(B, S, NHK, topk, device)

        torch.cuda.manual_seed(0)
        q = rearrange(
            torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=device),
            "b s (h1 h2) d -> (b s h1) h2 d",
            h1=NHK,
        ).requires_grad_(True)
        torch.cuda.manual_seed(1)
        k = rearrange(
            torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=device),
            "b s h d -> (b s h) 1 d",
        ).requires_grad_(True)
        torch.cuda.manual_seed(2)
        v = rearrange(
            torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=device),
            "b s h d -> (b s h) 1 d",
        ).requires_grad_(True)

        times = bench_bwd(q, k, v, indices, args.warmup, args.repeat)
        avg = sum(times) / len(times)
        mn, mx = min(times), max(times)
        print(f"[{idx_type:>10s}]  BWD avg={avg:.3f}ms  min={mn:.3f}ms  max={mx:.3f}ms")

        print()


if __name__ == "__main__":
    main()
