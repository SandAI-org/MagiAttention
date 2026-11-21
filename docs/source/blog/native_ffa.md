---
blogpost: true
date: Nov 14, 2026
author: Yunpeng Huang, Jin Li, Hao Chen
location: China
category: MagiAttention
tags: Flex-Flash-Attention, Flash-Attention, AttnSlice Representation, Hopper, Blackwell, CuTe DSL
---

# Flex-Flash-Attention: The Native Attention Kernel of MagiAttention

## Introduction

`Flex-Flash-Attention` (`FFA`) is the attention kernel underneath `MagiAttention` {cite}`magiattention2025_native_ffa`. It keeps the IO-aware tiling and online softmax of `Flash-Attention` {cite}`dao2022flashattention_native_ffa,dao2023flashattention_native_ffa,shah2024flashattention3fastaccurateattention_native_ffa`, but replaces the batch/varlen abstraction with the [AttnSlice Representation](./magi_attn.md#attnslice-representation): an attention mask is an arbitrary list of `(QRange, KRange, MaskType)` rectangles that may overlap along both the query and the key dimensions. This is what makes the kernel *distributable*: after the dispatch solver and attn solver shard the global mask across context-parallel (CP) ranks, every rank and every overlap stage receives a plain list of slices, and `FFA` computes it without re-encoding.

This post is the entry point for everything about the native `FFA` kernel. We first describe the design shared by all architectures (the slice IR, the scheduling model, and how partial results from overlapping slices are reduced), then go through the Hopper (SM90) and Blackwell (SM100) implementations and where they differ, and finally list the features built on top of the kernel together with the dedicated posts that cover each of them.

`FFA` currently has two native implementations, plus one temporary non-native backend for Blackwell:

| Backend | Target arch | Implementation | Status | Selection |
|---|---|---|---|---|
| `ffa` (native) | SM90 | CUDA C++ / CUTLASS, derived from `FA3` | Default, production | Default, or `MAGI_ATTENTION_KERNEL_BACKEND=ffa` |
| `cutedsl` (native) | SM100 / SM110 | CuTe DSL, derived from the `FA4` CuTe codebase | Work in progress | `MAGI_ATTENTION_KERNEL_BACKEND=cutedsl` |
| `fa4` (`FFA_FA4`) | SM100 / SM80 | Forked `FA4` with an `HSTU Function` mask | Temporary, not native | `MAGI_ATTENTION_KERNEL_BACKEND=fa4` |

```{note}
`FFA_FA4` is **not** the native `FFA` kernel and is **not** the default backend. It converts `AttnSlice` into an `HSTU Function` representation and runs a forked `Flash-Attention 4` kernel, as a stopgap that brought `MagiAttention` to Blackwell before the native SM100 kernel existed. See the separate [blog post](./blackwell_ffa_fa4.md) for details. The rest of this post only covers the two native implementations.
```


## Design Overview

### AttnSlice as the Kernel IR

The kernel consumes three tensors:

- `q_ranges`: `int32[R, 2]`, the `[start, end)` query token range of each slice over the packed `(total_q, num_heads_q, head_dim)` layout.
- `k_ranges`: `int32[R, 2]`, the key token range of each slice over the packed `(total_k, num_heads_kv, head_dim)` layout.
- `attn_type_map`: `int32[R]`, the mask type of each slice: `0 = FULL`, `1 = CAUSAL`, `2 = INV_CAUSAL`, `3 = BI_CAUSAL`.

`CAUSAL` is aligned to the bottom-right corner of the slice, `INV_CAUSAL` to the top-left corner, and `BI_CAUSAL` is the intersection of the two (see [Basic Mask Types in AttnSlice](./magi_attn.md#basic-mask-types-in-attnslice) for figures). With these four types, sliding-window and block-causal patterns stay compact instead of degrading to one slice per row.

Slices are not required to be sorted, disjoint, or aligned to tile boundaries. Several slices may share query rows (for example, a query chunk attending to several remote KV chunks in one CP stage) or share key rows (several query chunks attending to the same KV chunk). Handling those overlaps efficiently and correctly is the main difference between `FFA` and a varlen `Flash-Attention` kernel, and it shows up in three places: tile-range computation, work scheduling, and output reduction.

```{figure} ../../../assets/magi_attn/ffa/attnslice_interpret.png
:name: attnslice_interpret_native_ffa
:align: center
:width: 1000px
:alt: AttnSlice Formulation

An irregular mask decomposed into {math}`\mathrm{AttnSlice}` units. Each unit is one `(QRange, KRange, MaskType)` triplet consumed directly by the `FFA` kernel.
```

### Tile Range per Slice

A forward work tile is `(slice, m_block, head)`. For slice {math}`b` with query length {math}`s_q`, key length {math}`s_k` and the rows {math}`[m_{min}, m_{max}]` of the current `m_block` (relative to the slice), the visible key-block range is:

```{math}
\begin{aligned}
n_{max} &= \begin{cases}
\left\lceil \dfrac{m_{max} + 1 + s_k - s_q}{B_N} \right\rceil & \texttt{CAUSAL},\ \texttt{BI-CAUSAL} \\[2mm]
\left\lceil \dfrac{s_k}{B_N} \right\rceil & \texttt{FULL},\ \texttt{INV-CAUSAL}
\end{cases} \\
n_{min} &= \begin{cases}
\left\lfloor \dfrac{m_{min}}{B_N} \right\rfloor & \texttt{INV-CAUSAL},\ \texttt{BI-CAUSAL} \\[2mm]
0 & \texttt{FULL},\ \texttt{CAUSAL}
\end{cases}
\end{aligned}
```

with both bounds clamped to the slice. Inside {math}`[n_{min}, n_{max})`, only the blocks that intersect a diagonal need element-wise masking; the interior blocks run the unmasked fast path. Both implementations specialize the three regions (causal boundary, unmasked interior, inverse-causal boundary) at compile time, so the common unmasked iterations carry no mask instructions and no `-inf` row checks. The backward pass uses the dual formula to compute the `m_block` range of each `n_block`.

### Slice-Level Parallelism and Scheduling

Each slice is treated like one sequence of a varlen batch, so the tile space is the concatenation of every slice's `m_block` tiles over all heads. Because slice sizes are arbitrary, a static one-tile-per-CTA grid can leave SMs idle at the tail, and a short slice next to a long one causes imbalance. `FFA` therefore uses **persistent CTAs with dynamic tile fetching**: the grid is sized to the number of available SMs, and a CTA grabs its next tile from a global atomic counter, decodes it into `(slice, m_block, head)` with a warp-level prefix sum over slice tile counts, and hands it to its consumer warps through shared memory. When the caller provides `max_seqlen_q` (an upper bound on the longest query range), the decode degrades to an {math}`O(1)` division.

The persistent grid is also how `FFA` cooperates with communication. `sm_margin` reserves SMs that the attention kernel will never occupy, so that the `GroupCast` and `GroupReduce` kernels of the next overlap stage can start immediately instead of queueing behind attention CTAs. See the [kernel overlap](./kernel_overlap.md) post for how `MagiAttention` uses this.

### Reducing Partial Results Across Slices

When several slices write the same output rows, each of them produces a partial output with its own log-sum-exp. Two partial results {math}`(O_1, \mathrm{lse}_1)` and {math}`(O_2, \mathrm{lse}_2)` for the same query row combine exactly as

```{math}
\mathrm{lse} = \log\left(e^{\mathrm{lse}_1} + e^{\mathrm{lse}_2}\right), \qquad
O = e^{\mathrm{lse}_1 - \mathrm{lse}}\, O_1 + e^{\mathrm{lse}_2 - \mathrm{lse}}\, O_2
```

which is the same correction the distributed layer applies between CP stages. Since the merge is not a plain sum, it cannot be done with hardware atomic adds. `FFA` supports three reduction regimes, chosen per call:

1. **Locked LSE-weighted merge** (default, called "atomic reduction" in the API). The output buffer in global memory is itself the accumulator, kept in `float32` by default, and `lse` is initialized to `-inf`. In the epilogue, a CTA acquires a spin lock per `(q_block, head)` with `atomicCAS`; because a slice's query start is generally not aligned to `B_M`, a tile can straddle two physical blocks and needs both locks. Under the lock it reads the previous `lse`, computes the merged `lse`, rescales and adds the previous `O`, writes both back, issues a memory fence, and releases. A forward postprocess kernel then zero-fills rows that no slice covered (their `lse` is still `-inf`), folds in attention sink if present, and casts `O` to the requested dtype.
2. **Direct store**. With `disable_fwd_atomic_reduction=True` (or `disable_bwd_dkv_atomic_reduction=True` in the backward pass), the caller guarantees that the relevant ranges are sorted and pairwise disjoint. Every output row then has a single writer, so the kernel stores results in the native dtype without locks, and uncovered rows are filled by the pre/post-processing kernels.
3. **Range merge**. Slices that share the same `q_range` (forward) or `k_range` (backward) are grouped on the host into a CSR structure: sort by range, collapse consecutive duplicates, and record each group's slice offsets. One work tile then walks the key ranges of all slices in its group inside a single online-softmax loop, so the accumulator never leaves the chip and no global reduction is needed for that group. This is controlled by `range_merge` at the kernel level and by `MAGI_ATTENTION_RANGE_MERGE` in `MagiAttention` (experimental).

The backward pass has the same problem along both dimensions. Here the reductions are plain sums, so they map to hardware: the gradient of the inner loop dimension (`dQ` when the outer loop is over K/V) is accumulated into a `float32` buffer with TMA bulk reduce-add, and the gradient of the outer dimension (`dK`, `dV`) uses reduce-add only when slices share key rows or when GQA folds several query heads onto one KV head. Otherwise it is stored directly.

```{figure} ../../../assets/magi_attn/ffa/ffa_slice_atomic_reduce.png
:name: ffa_slice_atomic_reduce_native_ffa
:align: center
:width: 1000px
:alt: FFA Slice Atomic Reduction

Data loading, on-chip computation, and cross-slice reduction in the `FFA` forward and backward kernels.
```

### Pre- and Post-Processing Kernels

Around the main kernels sit three small kernels, which are also the hooks where several features are implemented without touching the mainloop:

- **Forward postprocess**: zero-fills uncovered rows, merges the attention-sink LSE into `lse` and rescales `O`, and casts the `float32` accumulator.
- **Backward preprocess**: computes {math}`\Delta = \mathrm{rowsum}(O \circ dO)` (named `dPsum` in the code), converts `lse` to the {math}`\log_2` domain used by `exp2`, and computes the sink gradient `dsink` on SM90.
- **Backward postprocess**: applies `softmax_scale` to the accumulated gradients, casts them, and zero-fills key rows that no slice covered.

### Deterministic Mode

The locked merge and the reduce-add accumulation are order-dependent in floating point. On SM90, `deterministic=True` makes the order fixed: every lock carries an `[arrive_num, counter]` pair, the scheduler tracks which slices have already visited each `(block, head)`, and a CTA waits until all slices with a smaller index have committed before it merges. This applies to the forward output, the backward `dQ`, and the backward `dK`/`dV`. The SM100 kernel does not support deterministic mode with ranges yet.


## SM90 Implementation (Hopper)

The SM90 kernel is written in CUDA C++ on top of CUTLASS/CuTe and derives from the `Flash-Attention 3` codebase {cite}`shah2024flashattention3fastaccurateattention_native_ffa`. It is the default backend.

### Warp Specialization and Pipeline

The kernel is a warp-specialized persistent kernel. Warpgroup 0 is the producer, and one to three consumer warpgroups run WGMMA. Register budgets are rebalanced with `setmaxnreg` and are chosen on the Python side per configuration, for example `(producer, consumer) = (56, 256)` with one MMA warpgroup, `(40, 232)` with two, and `(32, 160)` with three. In the TMA path, only one producer warp is needed and the other three exit early. In the sparse scatter path, the whole producer warpgroup issues `cp.async` loads.

Q is loaded once per tile with TMA. K and V each have their own two-stage pipeline, and the shared-memory buffer of `O` aliases the V buffer, since the epilogue only needs it after the last PV GEMM. The PV GEMM uses the register-sourced (RS) WGMMA variant, so `P` stays in registers after softmax and never goes through shared memory.

### Forward Mainloop

Two levels of overlap hide softmax latency behind Tensor Core work:

- **Inter-warpgroup ping-pong**: with two consumer warpgroups, named barriers make one warpgroup run its GEMMs while the other runs softmax.
- **Intra-warpgroup overlap** (on by default): in the steady state, a warpgroup issues `S_i = Q K_i^T`, then issues `O += P_{i-1} V_{i-1}`, waits only for the first GEMM, runs mask and softmax on `S_i`, then waits for the PV GEMM before rescaling `O`. For `head_dim > 128`, `O` is rescaled before the PV GEMM is issued instead of after it completes.

Forward tile sizes are `(B_M, B_N) = (192, 128)` for `head_dim <= 64` and `(128, 128)` for `head_dim <= 128`, and can be overridden with `ref_block_size`.

For very short query ranges (for example the decode-like slices that appear in sparse or inference workloads), `swap_ab=True` computes {math}`S^\mathrm{T} = K Q^\mathrm{T}` and {math}`O^\mathrm{T} = V P^\mathrm{T}` instead. The short query dimension then maps to the `N` dimension of WGMMA, which can be as small as 8, rather than the `M` dimension, which must be a multiple of 64. `ref_block_size` then takes `(8 | 16 | 32 | 64, 64)`.

### Backward Mainloop

The backward kernel offers two loop orders:

- **LoopQ** (default): the outer loop is over K/V tiles, so `dK` and `dV` accumulate in registers, and the inner loop is over Q/dO tiles. `dQ` is written by a dedicated storer warp in the producer warpgroup with `SM90_TMA_REDUCE_ADD` into a `float32` buffer.
- **LoopK** (`swap_bwd_qk_loop=True`): the outer loop is over Q/dO tiles, so `dQ` accumulates in registers, and two storer warps reduce-add `dK`/`dV`. This order suits shapes where few query tiles hit each key tile, such as index-sparse attention with small key blocks, and it is the order that supports `PackGQA` in the backward pass.

Inner-dimension gradients can be written with a 2D TMA tile reduce, a 1D bulk reduce (`cp.reduce.async.bulk.add.f32`, the default for sparse workloads), or scalar `atomicAdd`. The operand swap of each backward GEMM is chosen by `head_dim`: `S`/`dP` are swapped for `head_dim <= 128`, `dK`/`dV` for `head_dim > 128`, and `dQ` for `head_dim > 64`.

### GQA

Without packing, each query head is a separate tile and is mapped to its KV head. With `pack_gqa=True`, the `G = num_heads_q / num_heads_kv` query heads that share a KV head are folded into the `M` dimension through a hierarchical TMA layout `((G, seqlen), head_dim, num_heads_kv)`, so one K/V tile load serves `G` heads. `cat_gqa` is an alternative layout for the backward pass that is mutually exclusive with `pack_gqa`.

### Compilation

Each distinct configuration is rendered from a Jinja template into a single-variant `.cu` file, compiled with `nvcc` through Ninja, and cached on disk under a URI that encodes every specialization (direction, head dim, dtypes, softcap, reduction mode, deterministic, range merge, swap AB, GQA mode, sparse mode, tile size, and so on). Common configurations are pre-built into the wheel with the same machinery. See the [JIT compilation](./jit_compile.md) post for details.

### Constraints

The SM90 path targets `sm_90a` only, computes in `float16` or `bfloat16`, and accepts `head_dim <= 128` (rounded up to 64 or 128). `O` can be written as `float16`, `bfloat16`, or `float32`, and so can `dQ`, `dK`, and `dV`. The build requires CUDA 13 by default, because building with CUDA 12 can cause a significant performance drop. Set `MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=1` to build with CUDA 12 anyway.


## SM100 Implementation (Blackwell)

The SM100 kernel is written in CuTe DSL {cite}`cutlass-pythonDSL-overview_native_ffa`. The codebase started from the `Flash-Attention 4` CuTe module {cite}`dao2025flashattention_cute_native_ffa` and was rebuilt as native `FFA`: the kernels read the `[R, 2]` range tensors directly instead of `cu_seqlens`, accept overlapping query and key ranges, read a per-slice mask type at runtime, and implement the same reduction regimes as the SM90 kernel. This is different from `FFA_FA4`, which translates slices into per-row `HSTU Function` intervals and block-sparse metadata before calling a forked `FA4` kernel.

### Forward Kernel

The forward kernel uses 16 warps with fixed roles:

| Warps | Role |
|---|---|
| 0-3 | `softmax0`: softmax for the first Q tile |
| 4-7 | `softmax1`: softmax for the second Q tile |
| 8-11 | `correction`: `O` rescaling, and the epilogue for ranges |
| 12 | `MMA`: issues all `tcgen05.mma` instructions |
| 13 | `epilogue` (dense path) |
| 14 | `load`: TMA loads of Q, K, V |
| 15 | spare, or the scheduler warp in persistent mode |

Each CTA processes two 128-row Q tiles that share every K/V tile (`q_stage = 2`), and the two softmax warpgroups alternate between them. A single thread of the MMA warp issues every GEMM. It first computes `S_0 = Q_0 K_0^T` and `S_1 = Q_1 K_0^T`, and then, in each iteration, `O += P_{i-1} V_{i-1}` followed by `S = Q K_i^T` for each Q tile. The GEMM descriptors are pre-built in registers, so advancing a pipeline stage costs one integer add. To start the PV GEMM earlier, softmax signals it after the first three quarters of `P` are written, and a second barrier covers the last quarter.

All accumulators live in Tensor Memory (TMEM). The 512 columns are used exactly:

| TMEM columns | Content |
|---|---|
| `[0, 128)` | `S_0`, with `P_0` (bf16) aliased at `[64, 192)` |
| `[128, 256)` | `S_1`, with `P_1` aliased at the same offset |
| `[256, 256 + d_v)` | `O_0` |
| `[256 + d_v, 256 + 2 d_v)` | `O_1` |

Row max and row sum reuse the `S` columns once softmax has consumed them. The PV GEMM reads `P` from TMEM as its A operand, so `P` never goes through shared memory.

Softmax uses two tricks inherited from `FA4`:

- **Conditional rescaling**: the running max is only updated when the new max exceeds the old one by more than a threshold (8 in the {math}`\log_2` domain for 16-bit inputs). Otherwise, the stale max is kept and the rescale factor is exactly 1. The correction warpgroup checks this with a warp ballot and skips rescaling `O` in TMEM when no row needs it.
- **Software `exp2`**: a fraction of the exponentials are computed with a polynomial on the FMA pipes instead of the MUFU unit, to balance the two pipes. The fraction is tuned per `(2-CTA, causal, head_dim, arch)`. On SM103 (B300), whose special function units are fast enough on their own, the emulation is disabled.

For ranges, the correction warpgroup also runs the epilogue, including the locked LSE-weighted merge described above. It takes the lock of the lower physical block with a blocking spin and the lock of the upper block with a try-lock. If the try-lock fails, it releases the lower lock and retries, so a CTA never waits while holding a lock.

Non-causal configurations with `head_dim` 128 or 192 can use 2-CTA MMA (`cta_group::2`, cluster `(2, 1)`), where an MMA tile spans 256 query rows across a CTA pair. Kernels that read per-slice mask types at runtime currently use 1-CTA instructions. The number of K/V pipeline stages is derived from the shared-memory budget, and when `O` is accumulated in `float32` it can reuse K/V slots that have already been consumed, which keeps two Q tiles resident.

### Backward Kernel

The backward kernel also uses 16 warps:

| Warps | Role |
|---|---|
| 0-3 | `reduce`: drains `dQ` from TMEM and reduce-adds it to global memory |
| 4-11 | `compute`: two warpgroups for the `S`/`dP` loads from TMEM, softmax recompute, `dS`, and the `dK`/`dV` epilogue |
| 12 | `MMA` |
| 13 | `load` |
| 14 | `relay`: forwards the peer CTA's "half of `dS` ready" signal to the leader CTA in 2-CTA mode |
| 15 | spare, or the scheduler warp in persistent mode |

The loop order is K-outer, which corresponds to `LoopQ` on SM90. For each K/V tile, the kernel walks the visible Q tiles and issues five GEMMs: {math}`S^\mathrm{T} = K Q^\mathrm{T}`, {math}`dP^\mathrm{T} = V\, dO^\mathrm{T}`, {math}`dV \mathrel{+}= P^\mathrm{T} dO`, {math}`dK \mathrel{+}= dS^\mathrm{T} Q`, and {math}`dQ = dS\, K`. `S`, `dP`, `dV`, `dK`, and `dQ` all live in TMEM. `P` and `dS` alias the left halves of `S` and `dP`. `dQ` aliases `dP` in 1-CTA mode and the right half of `S` in 2-CTA mode, and in both cases the next producer GEMM waits until the reduce warpgroup has drained the previous `dQ`.

`dQ` goes from TMEM to registers, then to shared memory, and is reduce-added to a token-major `float32` buffer with TMA bulk reduce. The unaligned start of a query range is used directly as the TMA coordinate. A postprocess kernel applies `softmax_scale` and casts the result. `dK`/`dV` are stored directly with TMA for MHA with disjoint key ranges. For GQA, or when slices share key rows, they are reduce-added to `float32` buffers and cast afterwards. `head_dim >= 128` can use 2-CTA MMA, and `head_dim = 192` requires it.

### Scheduling

By default, the SM100 forward kernel launches a static grid. Its size is an upper bound on the number of tiles, computed from `total_q + R * (B_M - 1)`, and each CTA decodes its tile from a prefix sum over slice lengths. When query ranges overlap and the kernel is not persistent, the grid becomes `(ceil(max_seqlen_q / B_M), heads, R)` so that each slice gets a fixed quota of CTAs. When `sm_margin > 0`, both passes switch to a dynamic persistent schedule. The grid is capped at `num_SMs - sm_margin` (rounded to the cluster size), and a scheduler warp in the leader CTA fetches tiles from a global atomic counter and broadcasts the tile index to every CTA of the cluster.

### Compilation

Kernels are compiled with `cute.compile` and launched through TVM FFI {cite}`cutlass-pythonDSL-cute-compile-with-tvm-ffi_native_ffa`, which removes most per-call host overhead. Compiled kernels are cached in memory by a compile key and can also be persisted to disk (`MAGI_ATTENTION_FFA_CUTEDSL_CACHE_ENABLED`, `MAGI_ATTENTION_FFA_CUTEDSL_CACHE_DIR`). The disk cache is keyed by a fingerprint of the kernel sources, the `nvidia-cutlass-dsl`, `quack-kernels`, and TVM FFI versions, and the CUDA version. The supported `nvidia-cutlass-dsl` / `quack-kernels` pairs are listed in the [Installation Guide](../user_guide/install.md).

### Constraints

The SM100 backend is still work in progress. It computes in `float16` or `bfloat16` and supports `head_dim <= 128` and `(head_dim, head_dim_v) = (192, 128)`. `O`, `dQ`, `dK`, and `dV` can be written as `float32`, `float16`, or `bfloat16`. Deterministic mode is not supported with ranges, `softcap` is not available on the per-slice-mask path, and `PackGQA` is only used by the forward pass when the output is direct-stored.


## SM90 vs. SM100 at a Glance

| Aspect | SM90 (`ffa`) | SM100 (`cutedsl`) |
|---|---|---|
| Language and origin | CUDA C++ / CUTLASS, from `FA3` | CuTe DSL, from the `FA4` CuTe module |
| MMA | Warpgroup `wgmma`, issued by each consumer warpgroup | `tcgen05.mma`, issued by a single thread, optional 2-CTA |
| Accumulators | Registers (`P` reused as the RS operand) | TMEM (`S`, `P`, `O`, `dK`, `dV`, `dP`, `dQ`) |
| Forward warp layout | 1 producer + 1 to 3 MMA warpgroups | Load / MMA / 2 softmax / correction / epilogue warps |
| Q tiles per CTA | 1 per consumer warpgroup, ping-pong across warpgroups | 2 tiles ping-ponged inside one CTA |
| Softmax latency hiding | Inter- and intra-warpgroup overlap | Conditional rescaling, software `exp2`, split `P` arrival |
| Forward overlapping-Q merge | Locked LSE merge in the consumer epilogue | Locked LSE merge in the correction warpgroup, with try-lock |
| Backward loop order | `LoopQ` (default) or `LoopK` | K-outer |
| `dQ` accumulation | Storer warp, TMA reduce-add into `float32` | Reduce warpgroup, TMA bulk reduce-add into `float32` |
| Scheduling | Always persistent, dynamic | Static by default, dynamic persistent with `sm_margin > 0` |
| Head dims | 64, 128 | `<= 128`, and `(192, 128)` |
| Deterministic | Supported (`LoopQ`) | Not supported with ranges |
| Compilation | Jinja + `nvcc` + Ninja, AOT and JIT | `cute.compile` + TVM FFI, memory and disk cache |


## Features

The following features are built on the kernel. Each has a dedicated post.

| Feature | SM90 | SM100 | Post |
|---|---|---|---|
| Flexible masks (`AttnSlice`, 4 mask types) | Yes | Yes | [MagiAttention](./magi_attn.md) |
| Learnable attention sink | Yes | Yes | [Attention Sink](./attn_sink.md) |
| Muon QK-Clip (`max_logits`) | Yes | Yes | [Muon QK-Clip](./muon_qk_clip.md) |
| Sparse attention | `BlockSparse`, `IndexSparse` | `IndexSparse` (standalone kernels) | [Sparse Attention](./sparse_attn.md) |
| GQA / MQA | `PackGQA`, `CatGQA` | `PackGQA` (forward) | This post |
| Softcap | Yes | Static-mask kernels only | This post |
| Configurable output / gradient dtype | Yes | Yes | This post |
| `sm_margin` for comm overlap | Yes | Yes | [Kernel Overlap](./kernel_overlap.md) |
| Deterministic mode | Yes | Not with ranges | This post |
| JIT compilation | Jinja + `nvcc` | CuTe DSL | [JIT Compilation](./jit_compile.md) |

### Math Foundation

Both implementations follow the `Flash-Attention 2` formulation of the forward and backward passes: online softmax with a running max and sum in the forward pass, and recomputation of `P` from `lse` with {math}`\Delta = \mathrm{rowsum}(O \circ dO)` in the backward pass. See [Flash Attention 2 Math Derivation](./fa2_math_derivation.md) for the full derivation that the kernels implement.

### Learnable Attention Sink

The sink only affects the softmax denominator, so neither implementation touches the mainloop for it. On SM90, the sink LSE is merged into `lse` and `O` is rescaled in the forward postprocess kernel, and `dsink` is computed in the backward preprocess kernel. On SM100, the direct-store path passes {math}`\mathrm{logsumexp}(\mathrm{sink})` to the kernel as one extra logit per head, the locked-merge path folds it in the forward postprocess kernel, and `dsink` comes from a separate kernel in which the last CTA of each head reduces the partial sums in a fixed order. See [Support Learnable Attention Sink](./attn_sink.md) for the math and the distributed implementation.

### Muon QK-Clip

`return_max_logits=True` returns the maximum attention logit per head. On SM90, each CTA reduces row maxima into a shared-memory slot per head with an atomic max and then updates global memory with another atomic max, so `num_heads_q <= 128` is required. On SM100, conditional rescaling means the running max lags the true max, so the softmax warpgroups track an exact row max in a separate register. The global update uses an IEEE bit-pattern trick: a signed integer max for non-negative values and an unsigned integer min for negative values. See [Support Muon QK-Clip](./muon_qk_clip.md) for the distributed reduction across CP ranks and stages.

### Sparse Attention

On SM90, sparse attention runs inside the same kernel family. `BlockSparse` expresses uniformly sized key blocks through ranges, `IndexSparse` reads a `(total_q, num_heads_kv, topk)` index tensor directly, and the producer gathers K/V rows with `cp.async` (or TMA when key blocks are at least 128 tokens). The backward pass picks `LoopK` when key blocks are small. On SM100, `IndexSparse` has dedicated forward and backward kernels built around `PackGQA` tiles of 128 rows with shared token-level indices, together with a top-`k` kernel. These kernels are standalone and not yet wired into the distributed path. See [Optimize Sparse Attention in FFA](./sparse_attn.md) for the design and benchmarks.

### FFA_FA4 Backend

`FFA_FA4` is a temporary Blackwell backend built on a forked `Flash-Attention 4` with an `HSTU Function` mask representation. It is not part of the native `FFA` kernel and is never selected by default. It is described in [Support Blackwell with FFA_FA4 Backend](./blackwell_ffa_fa4.md).


## User Interface

On Hopper, the native kernel is exposed as `flex_flash_attn_func`:

```python
from magi_attention.api import flex_flash_attn_func

out, meta = flex_flash_attn_func(
    q,                # (total_q, num_heads_q, head_dim)
    k,                # (total_k, num_heads_kv, head_dim)
    v,                # (total_k, num_heads_kv, head_dim)
    q_ranges,         # int32 (R, 2)
    k_ranges,         # int32 (R, 2)
    attn_type_map,    # int32 (R,), 0=FULL 1=CAUSAL 2=INV_CAUSAL 3=BI_CAUSAL
    sink=None,
    softcap=0.0,
    sm_margin=0,
    return_max_logits=False,
)
lse, max_logits = meta.lse, meta.max_logits
```

On Blackwell, the CuTe DSL kernel has the same semantics, with keyword-only range arguments and `mask_types` in place of `attn_type_map`:

```python
from magi_attention.kernel.cutedsl import flex_flash_attn_func as cutedsl_ffa_func

out, meta = cutedsl_ffa_func(
    q, k, v,
    q_ranges=q_ranges,
    k_ranges=k_ranges,
    mask_types=attn_type_map,  # int32 (R,) tensor, or a single int (0/1) for a static kernel
    max_seqlen_q=max_seqlen_q, # hard upper bound on the longest q range
    max_seqlen_k=max_seqlen_k,
)
```

```{warning}
`max_seqlen_q` and `max_seqlen_k` are hard upper bounds for the SM100 kernel: launch bounds are sized from them, so an underestimate silently drops work. Omit them to fall back to the safe `total_q` / `total_k` bounds.
```

For distributed attention, `calc_attn` picks the kernel through `MAGI_ATTENTION_KERNEL_BACKEND`: `ffa` (default) on Hopper, or `cutedsl` on Blackwell. The reduction mode, `sm_margin`, and range merge are configured by `MagiAttention` from the attn solver output and the [environment variables](../user_guide/env_variables.md), so `calc_attn` users do not pass them by hand.


## Roadmap

- Promote the SM100 CuTe DSL kernel to the default Blackwell backend and retire `FFA_FA4`.
- Support deterministic mode and softcap with per-slice masks on SM100.


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

```{bibliography} refs/native_ffa.bib
```
