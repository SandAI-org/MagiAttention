---
blogpost: true
date: Jan 25, 2026
author: Jin Li, Hao Chen, Zewei Tao, Hanwen Sun, Bowen Zeng, Yunpeng Huang
location: China
category: MagiAttention
tags: Sparse Attention, NSA, DSA, Flex-Flash-Attention, Blackwell, CuteDSL
---

# Optimize Sparse Attention in FFA

## Introduction

Sparse attention is an important technique for speeding up long-context model training and inference. By letting each query attend to only a subset of key and value tokens, sparse attention cuts the quadratic compute cost of standard softmax attention. Recent models such as DeepSeek Sparse Attention (`DSA`) {cite}`deepseekai2025deepseekv32pushingfrontieropen_sparse_attn` and MiniMax Sparse Attention (`MSA`) {cite}`lai2026minimaxsparseattention_sparse_attn` show that dynamically selecting context from content can match full attention at a small fraction of the FLOPs. Turning those FLOP savings into real wall-clock speedups on GPUs is much harder: sparse attention introduces scattered memory reads and small matrix multiplications that leave modern Tensor Cores underutilized. In this post, we share the design of `IndexSparse Attention` in `MagiAttention` and explain how we built its forward, backward, and streaming top-{math}`k` kernels on NVIDIA Blackwell GPUs with CuTe DSL.


## 1. IndexSparse Attention

### 1.1 Two-Stage Sparse Attention

Trainable dynamic sparse attention runs in two stages: **index selection** followed by **sparse attention**. First, a lightweight indexer scores prior tokens to pick the most relevant KV positions for each query. Given the selected index set {math}`\mathcal{I}_i`, the main branch runs standard softmax attention over just those KV tokens:

```{math}
O_i = \operatorname{softmax}\left(\frac{Q_i K_{\mathcal{I}_i}^\top}{\sqrt{d}}\right) V_{\mathcal{I}_i}.
```

Over a sequence of length {math}`N`, attending to {math}`k` tokens per query lowers main-branch compute from {math}`O(N^2 d)` to {math}`O(N k d)`.

### 1.2 Sparsity Pattern and Shared Indexing

Recent dynamic sparse designs split along selection granularity. Block-sparse methods such as `NSA` {cite}`yuan2025nativesparseattentionhardwarealigned_sparse_attn` and `MSA` {cite}`lai2026minimaxsparseattention_sparse_attn` select whole contiguous KV blocks, keeping memory reads contiguous and GPU execution simple. Token-level methods such as `DSA` {cite}`deepseekai2025deepseekv32pushingfrontieropen_sparse_attn` pick individual KV tokens instead, trading contiguous memory access for finer selection flexibility. `DSA` keeps Tensor Cores busy in the forward pass because it sits on the MQA mode of `MLA`, where all 128 query heads of a token share a single 512-dim KV vector; in the backward pass, however, a 512 `head_dim` quickly fills up on-chip `TMEM` and `SMEM`, limiting achievable TFLOPS.

**IndexSparse Attention** brings arbitrary token-level KV selection to standard 128-dim `GQA` models. On Blackwell GPUs, a single-CTA BF16 `tcgen05.mma` instruction needs 64 or 128 rows in the M dimension, whereas a single query token in standard `GQA` has only {math}`G` query heads per KV head (for example, 8 rows in 8:1 `GQA`). `IndexSparse` bridges that gap by having the {math}`G` query heads of a KV head and {math}`B_Q` consecutive query tokens share one KV index set {math}`\mathcal{I}_b`, with {math}`G \times B_Q = 128` so every tile fills Tensor Cores directly. The selected positions are passed as a 3D `int32` tensor:

```python
index_sparse_indices: [num_q_blocks, num_kv_heads, topk]  # int32
```

Each entry holds a global KV token index. Near the start of a causal sequence where fewer than {math}`k` keys are visible, valid indices come first and `-1` pads the rest of the row. In `FFA`, masks are normally stored as [AttnSlice](./magi_attn.md#attnslice-representation) rectangles; taking an index tensor directly avoids turning token-level selection into {math}`k` width-1 slices per query block.

```{figure} ../../../assets/sparse_attn/indexsparse-attention.png
:name: indexsparse_attention_overview
:align: center
:width: 800px
:alt: Overview of IndexSparse Attention

Overview of `IndexSparse Attention`: Index Selection picks token-level KV indices per query block and KV head, and Sparse Attention computes exact softmax attention over those shared selections.
```

### 1.3 PackGQA Tiling and KV Reuse

Inside the kernel, **PackGQA** folds the {math}`B_Q` consecutive query tokens and {math}`G` query heads that share an index row into one 128-row Q tile. In our primary setup with `G = 8` and `B_Q = 16`, those 128 rows multiply each gathered 128-row K tile as a clean {math}`[128, 128] \times [128, 128]` MMA, and all 128 query rows reuse every gathered KV tile. Because the 16 neighboring tokens share one selection list from the start, the kernel never computes cross-terms for unselected keys only to mask them out before softmax, as a union over separate per-token selections would have to do.

```{figure} ../../../assets/sparse_attn/packgqa-mma-tile.png
:name: packgqa_mma_tile
:align: center
:width: 800px
:alt: PackGQA for IndexSparse Attention

`PackGQA` in `IndexSparse Attention`: 16 consecutive query tokens and 8 query heads pack into one 128-row MMA tile, sharing a single token-index list and multiplying a gathered K tile.
```


## 2. Kernel Design

### 2.1 Loop Order: Q-outer vs. KV-outer

Once the representation is set, we have to choose the loop order across CTAs. **Q-outer** gives each CTA one Q tile, steps through its selected KV tiles, and finishes online softmax on chip. **KV-outer** flips the loop so each CTA holds one KV block and gathers every query across the sequence that selected it; that reuses KV across arbitrary queries, at the price of building an inverted index, splitting popular sink blocks near the start of the sequence across CTAs, and running a separate kernel to merge partial outputs.

`MSA` uses `KV-outer` because each query picks a different subset and its 128-token KV blocks are large enough to amortize the inverted index and merge pass {cite}`yan2026fsaalternativeefficientimplementation_sparse_attn,lai2026minimaxsparseattention_sparse_attn`. At single-token granularity, building an inverted table over {math}`N` individual tokens and splitting sink hotspots costs more than attention itself. Moreover, under `IndexSparse`, all 128 rows of a Q tile already share the same index list, so `Q-outer` forms full 128-row MMA tiles right away and finishes online softmax inside a single CTA with no intermediate buffers or merge kernel.

### 2.2 Forward: Sparse KV Gather and Reverse Traversal

Because selected KV tokens sit at arbitrary positions across the sequence, contiguous TMA loads cannot fetch them directly. Instead, the forward kernel uses asynchronous `cp.async` instructions to gather K and V rows one by one into shared memory from `index_sparse_indices`.

Early query blocks in a causal sequence see fewer than {math}`k` keys, leaving `-1` padding at the end of their index rows that must be masked to {math}`-\infty` before softmax. Because valid indices always sit at the front of the row and `-1` can only appear at the tail, the inner loop walks KV tiles **backward from the last tile to the first**. Any padded tail tiles are handled in the first few iterations; as soon as the loop reaches the first full tile, the rest of the loop runs without bounds checks or score masks.

### 2.3 Backward: Q-outer Traversal and L2 Cache Optimization

Under token-level selection, most ordinary KV tokens are picked by only a handful of query blocks. Switching the backward pass to `KV-outer` would leave most KV positions without enough rows and columns to fill Tensor Cores, on top of needing a single-token inverted index and dealing with sink skew. Our backward kernel therefore stays `Q-outer` as well, reusing the 128-row Q tile to keep Tensor Cores full: `dQ` accumulates on chip and writes out once, while `dK` and `dV` accumulate into global FP32 gradient buffers through `TMA` `reduce.add`.

As sequences grow longer, random K and V reads together with `dK` and `dV` updates span the entire sequence, exceeding L2 cache size and driving down the cache hit rate. To recover L2 locality on long contexts, the backward kernel first groups the selected indices in each row into contiguous KV ranges, then runs one pass per range so each pass touches a KV slice that stays in L2.

### 2.4 Streaming Top-K

Before running sparse attention, we need to pick the largest {math}`k` positions from each length-{math}`N` row of indexer scores. A standard radix select takes multiple histogram passes and re-reads the entire row from HBM on every pass. In long-context workloads, however, almost every score sits well below the cutoff: when picking 2,048 tokens out of 256K, over 99% of entries never contend for the top {math}`k`.

To avoid reading those entries repeatedly, we implemented a streaming kernel, `cutedsl_topk_stream`, where one CTA handles one row. The CTA first takes a small sample of the row to estimate a conservative cutoff on chip, then streams the full row once with TMA, discarding entries below the cutoff on the fly and keeping only the remaining candidates in shared memory for a final exact selection in ascending index order. If a skewed row fills the candidate buffer or leaves fewer than {math}`k` entries, the kernel compacts the buffer in place or re-scans the row as a fallback, keeping the output exact while reading HBM only once in the common case.


## 3. Experiments

We benchmark `IndexSparse Attention` and `cutedsl_topk_stream` on an NVIDIA B300 GPU against full-attention `Dense FFA`. The primary configuration is shown below:

| Parameter | Value |
| --- | --- |
| GPU | NVIDIA B300 (SM103) |
| Head dimension | 128 |
| GQA ratio ({math}`G`) | 8 |
| Query block size ({math}`B_Q`) | 16 |
| MMA tile | 128 × 128 |
| TopK | 4096 |
| Precision | BF16 |

### 3.1 IndexSparse Attention

Fixing `TopK = 4096`, we scale context length from `96K` (`CP = 8`) to `384K` (`CP = 32`) under distributed long-context training shapes. For the forward pass, we report both standalone kernel throughput `IndexSparse (attn)` and end-to-end throughput `IndexSparse (e2e)` with `cutedsl_topk_stream` included.

```{figure} ../../../assets/sparse_attn/bench_index_sparse_fwd.png
:name: bench_index_sparse_fwd
:align: center
:width: 800px
:alt: IndexSparse Attention Forward Throughput on NVIDIA B300

(a) Forward Pass
```

```{figure} ../../../assets/sparse_attn/bench_index_sparse_bwd.png
:name: bench_index_sparse_bwd
:align: center
:width: 800px
:alt: IndexSparse Attention Backward Throughput on NVIDIA B300

(b) Backward Pass

Benchmarking `IndexSparse Attention` against full-attention `Dense FFA` on NVIDIA B300 across `96K–384K` context lengths (`TopK = 4096`).
```

Despite scattered token-level gathers, `IndexSparse` sustains `872–1,074 TFLOPs/s` in the forward pass (`Dense FFA` reaches about `1,545 TFLOPs/s`) and `599–713 TFLOPs/s` in the backward pass (`Dense FFA` reaches about `1,280 TFLOPs/s`). In wall-clock time, as context length grows from `96K` to `384K`, end-to-end forward speedup with `cutedsl_topk_stream` included increases from `12.3x` to `32.0x` (`3.19 ms` vs. `102.11 ms`), while backward speedup increases from `13.4x` to `44.7x` (`6.88 ms` vs. `307.88 ms`). Raising `TopK` to `10,240` at `384K` pushes forward kernel throughput to `961 TFLOPs/s` while maintaining `17.5x` end-to-end forward and `16.1x` backward speedups over `Dense FFA`.

### 3.2 Streaming Top-K

We benchmark `cutedsl_topk_stream` against `torch.topk` at `k = 10,240` (`10K`) on `4,096` `BF16` rows across row lengths from `64K` to `1M`:

```{figure} ../../../assets/sparse_attn/bench_topk_stream.png
:name: bench_topk_stream
:align: center
:width: 800px
:alt: Streaming Top-K Effective Bandwidth vs torch.topk at k=10240 on NVIDIA B300

Effective memory bandwidth of `cutedsl_topk_stream` versus `torch.topk` on NVIDIA B300 (`k = 10,240`).
```

As row length grows from `64K` to `1M`, `torch.topk` stays around `244–408 GB/s` of effective bandwidth. Streaming HBM in a single pass lets `cutedsl_topk_stream` scale up to `4.19 TB/s` at `1M` (`2.09 ms` vs. `36.64 ms` for `torch.topk`), delivering `4.8x–17.5x` speedups across `64K–1M`.


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
