---
blogpost: true
date: Jan 25, 2026
author: Zewei Tao, Hanwen Sun, Bowen Zeng, Jin Li, Yunpeng Huang
location: China
category: MagiAttention
tags: Sparse Attention, NSA, DSA, Flex-Flash-Attention, Blackwell, CuteDSL
---

# Optimize Sparse Attention in FFA

## Introduction

Sparse attention has become an important technique for accelerating long-context training and inference. By letting each query attend to only a subset of key/value tokens, sparse attention substantially reduces the compute cost of standard softmax attention. Recent architectures such as DeepSeek Sparse Attention (`DSA`) {cite}`deepseekai2025deepseekv32pushingfrontieropen_sparse_attn` have demonstrated the effectiveness of dynamically selected sparse attention. However, turning theoretical FLOP reductions into actual GPU speedups remains challenging. Unlike dense FlashAttention, sparse attention introduces irregular memory accesses and small matrix multiplications that can under-utilize modern Tensor Cores. Efficient execution requires careful design of the sparse representation, work partitioning, and memory pipeline. In this post, we present the design and implementation of `IndexSparse Attention` in `MagiAttention`, and discuss how we optimize its forward, backward, and streaming top-{math}`k` kernels for NVIDIA Blackwell GPUs with CuTe DSL.


## Sparse Patterns and Bottlenecks

Early sparse attention designs fix the attended subset by position — such as sliding windows, strided patterns, and global or sink tokens {cite}`child2019generatinglongsequencessparse_sparse_attn,beltagy2020longformerlongdocumenttransformer_sparse_attn,zaheer2021bigbirdtransformerslonger_sparse_attn`. Recent trainable mechanisms instead select keys dynamically from content in two stages: a lightweight indexer scores the prefix to pick a fixed budget of keys per query, and main attention runs exact softmax attention over the selected keys only. Mainstream designs follow two granularity choices: **token-wise** `DSA` {cite}`deepseekai2025deepseekv32pushingfrontieropen_sparse_attn` picks {math}`k = 2048` individual tokens ({math}`B_k = 1`) on top of `MLA` in MQA mode, where all 128 query heads of one token share a single 512-dim latent KV vector (plus 64-dim RoPE) and one index list; **block-wise** `NSA` {cite}`yuan2025nativesparseattentionhardwarealigned_sparse_attn`, `MoBA` {cite}`lu2025mobamixtureblockattention_sparse_attn`, and `MSA` {cite}`lai2026minimaxsparseattention_sparse_attn` target standard 128-dim `GQA` by selecting contiguous KV blocks per group (or, as in `QSA` {cite}`qwenteam2026qwen38next_sparse_attn`, pooling every 4 keys into micro-blocks before scoring to cut the indexer's {math}`O(N^2)` cost).

Turning dynamic sparsity into real training speedups requires resolving three hardware bottlenecks {cite}`yan2026fsaalternativeefficientimplementation_sparse_attn,lai2026minimaxsparseattention_sparse_attn`:

1. **Tensor Core utilization.** On Blackwell, a single-CTA BF16 `tcgen05.mma` requires {math}`M = 64` or 128 rows. Under standard `GQA`, one query token has only {math}`G = H_q / H_{kv}` heads (typically 4 to 16), so processing tokens one by one leaves most of {math}`M` as wasted padding.
2. **K/V reuse and loop order.** Loading K/V independently per query in a `Q-outer` loop gives an arithmetic intensity of only {math}`\approx G` {cite}`lai2026minimaxsparseattention_sparse_attn`. Grouping work by KV block (`KV-outer`, arithmetic intensity {math}`\approx \frac{2}{3} B_k`) works well for large blocks like `MSA` ({math}`B_k = 128`), whereas at single-token granularity ({math}`B_k = 1`), inverting indices over {math}`N` tokens and splitting sink-token hotspots costs more than attention itself.
3. **`head_dim` vs. backward on-chip memory.** While `MLA`'s 512 head dimension boosts forward arithmetic intensity, backward must simultaneously hold `Q`, `K`, `V`, `dO`, intermediate scores, and `dQ`/`dK`/`dV` accumulators in `TMEM` (512 columns per SM on Blackwell) and `SMEM`. An oversized `head_dim` exhausts `TMEM` and `SMEM`, shrinking tile sizes or pipeline depth and preventing backward from reaching high TFLOPS.


## IndexSparse Attention

**Shared indexing.** In `FFA`, [AttnSlice](./magi_attn.md#attnslice-representation) represents masks as contiguous query–key rectangles, so token-wise selection degenerates into {math}`k` width-1 slices per query. We therefore design **IndexSparse** in the CuTeDSL backend of `FFA` (`SM100` forward and backward, plus `SM120` forward), paired with a streaming top-{math}`k` operator `cutedsl_topk_stream`. To support token-level sparsity on standard `head_dim = 128` across both `MQA` and `GQA`, `IndexSparse` shares one index row across the {math}`G` query heads of a KV group and `q_block_size` consecutive query tokens such that `q_block_size * G = 128` (`q_block_size = 1` for `MQA-128`, `q_block_size = 16` for `8:1 GQA`). Every 128-row Q tile then attends to the exact same K/V rows in a single `Q-outer` pass — filling Tensor Cores and reusing K/V across all 128 rows without union-mask waste. Selections are passed as an `int32` tensor `index_sparse_indices` of shape `(ceil(total_q / q_block_size), num_heads_kv, topk)`, with valid global indices at the front and `-1` padding at the tail when fewer than {math}`k` keys are visible.

:::{todo}
Add overall `IndexSparse` architecture figure here.
:::

```python
from magi_attention.kernel.cutedsl.topk_stream import cutedsl_topk_stream
from magi_attention.kernel.cutedsl.ffa_fwd_sm100_index import ffa_fwd_sm100_index
from magi_attention.kernel.cutedsl.ffa_bwd_sm100_index import ffa_bwd_sm100_index

# MQA-128 example: num_heads_q = 128, num_heads_kv = 1, q_block_size = 1
indices = cutedsl_topk_stream(indexer_scores, k=topk, end=causal_row_ends, out_dtype=torch.int32)
out, lse = ffa_fwd_sm100_index(q, k, v, index_sparse_indices=indices, q_block_size=1)
dq, dk, dv = ffa_bwd_sm100_index(
    q.view(-1, 1, 128), k, v,
    out.view(-1, 1, 128), dout.view(-1, 1, 128), lse.view(-1, 1),
    index_sparse_indices=indices, q_block_size=128,
)
dq = dq.view_as(q)
```

`ffa_bwd_sm100_index` also accepts an optional `dlse` gradient tensor so an auxiliary KL loss on the indexer can backpropagate through `lse` directly.


## Kernel Design

**Forward pass.** Each CTA owns one 128-row Q tile, loads Q contiguously, gathers the {math}`k` scattered K/V rows asynchronously, and reduces `out` and `lse` on chip with online softmax. Early causal positions with fewer than {math}`k` visible keys carry `-1` tail padding that must be masked to {math}`-\infty`. Because valid indices sit at the front and `-1` appears only at the tail, the inner loop traverses K/V tiles **in reverse order from last to first**: partially padded tail tiles are handled up front, and the remaining tiles run without bounds checks or score masking (`assume_valid_ids=True` removes the check entirely at compile time when all rows are full).

**Backward pass.** `ffa_bwd_sm100_index` stays `Q-outer` as well to avoid per-step single-token index inversion and sink-token load skew: `dQ` accumulates on chip and writes back once, while scattered FP32 `dK`/`dV` updates are staged as full rows in shared memory and accumulated via asynchronous whole-row `TMA` `reduce.add`. As context length grows, the random K/V and `dK`/`dV` working set crosses the GPU's L2 capacity {math}`L_2`. In the L2-miss latency window {math}`(0.5\, L_2,\, 4\, L_2]`, the kernel automatically overlaps V gathers with the next tile's K fetch (`gather_v_async`) to double in-flight reads; past {math}`2\, L_2` where DRAM bandwidth saturates, it switches to L2-sized multi-pass execution (`num_k_passes`), bucket-sorting each index row into {math}`\sim 0.8\, L_2` KV chunks so every pass's random reads and reductions stay inside L2.

**Streaming Top-K (`cutedsl_topk_stream`).** Since softmax is monotonic, top-{math}`k` runs directly on raw BF16/FP16 indexer scores and outputs ascending indices {cite}`lai2026minimaxsparseattention_sparse_attn`. At long contexts, {math}`k / N` is tiny (over 99% of entries at {math}`k = 2048, N = 256\text{K}` sit below the threshold), yet multi-pass radix select re-reads the full row from HBM every pass. `cutedsl_topk_stream` ({math}`N \le 2^{21}, k \le 10240`) assigns one CTA per row to stream HBM **only once** in the common case: it first samples strided 64-element blocks across `[0, end)` to estimate a conservative lower-bound threshold on chip, then streams the row once with TMA to filter survivors into an on-chip buffer for a final 2-byte radix select. Mid-scan compaction on ties and a {math}`-\infty` rescan when survivors fall short of {math}`k` guarantee exact results.


## Experiments and Roadmap


## Citation

If you find MagiAttention useful in your research, please cite:

```bibtex
@misc{magiattention2025,
  title={MagiAttention: A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training},
  author={Zewei, Tao and Yunpeng, Huang},
  year={2025},
  howpublished={\url{https://github.com/SandAI-org/MagiAttention/}},
}
```

## References

```{bibliography} refs/sparse_attn.bib
```
