---
blogpost: true
date: Apr 21, 2025
author: Zewei Tao, Yunpeng Huang, Qiangang Wang, Hanwen Sun, Jin Li, Tao Bu, Bowen Zeng
location: China
category: MagiAttention
tags: Attention Slice Representation, Computation Load-Balance, Zero-Redundant Communication, Multi-Stage Overlap, Flex-Flash-Attention, Group Collective, Flash-Attention, Distributed Attention, Context Parallelism
language: English
---

# MagiAttention

**A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training**

## Overview

```{figure} ../../../assets/magi_attn/magiattn_overview.png
:align: center
:width: 1000px
:alt: MagiAttention Overview

Overview of MagiAttention: (1) `FFA` - an optimized kernel based on Flash-Attention 3, further supports flexible mask patterns; (2) The `dispatch solver` shards ultra‑long data and dispatches for load-balanced computation; (3) `GroupCast` and `GroupReduce` primitives eliminate redundant communication; (4) The `overlap solver` adaptively partitons multi-stage computation/communication for optimal overlap; (5) Forward and backward timelines scheduled by MagiAttention. With all components together, MagiAttention enables linear scalability in training with ultra‑long contexts and heterogeneous masks.
```

Training large-scale video‑generation models faces two tightly coupled challenges: (1) ultra‑long contexts—reaching millions of tokens (e.g., **~4M**)—which make attention prohibitively expensive in compute and memory, and (2) highly heterogeneous, irregular attention masks (e.g., block‑causal + Patch‑and‑Pack) that break assumptions of existing kernels and distributed layouts, leading to fragmentation, load imbalance, wasted padding, and large communication overhead.

These same constraints also affect (multimodal) LLMs that aim to support ultra‑long histories and flexible masking for agentic tasks with large retrievals and deep reasoning. <u>Therefore, we require an efficient, mask-flexible, and scalable distributed attention solution</u>.

To address these challenges, we propose [MagiAttention](https://github.com/SandAI-org/MagiAttention), which targets these bottlenecks with **kernel-level flexibility**, while achieving **distributed-level linear scalability** across a broad range of training scenarios, particularly for those involving ultra-long contexts and heterogeneous masks like [Magi-1](https://github.com/SandAI-org/MAGI-1).


## Introduction

Training large-scale autoregressive diffusion models for video generation (e.g., [Magi-1](https://github.com/SandAI-org/MAGI-1)) creates two tightly coupled system challenges. First, training contexts can reach millions of tokens, so naive quadratic attention or inadequately sharded algorithms quickly become infeasible in both compute and memory. Second, practical data pipelines—for example, block‑causal attention combined with Patch‑and‑Pack (PnP) processing {cite}`dehghani2023patchnpacknavit` — produce highly heterogeneous, irregular masks and variable sequence lengths that violate assumptions made by standard attention kernels and distributed layouts. The combined effect is severe fragmentation, imbalanced compute across ranks, excessive padding, and large, often redundant, communication volumes.

Prior context‑parallel solutions {cite}`jacobs2023deepspeed,liu2023ringattentionblockwisetransformers,fang2024uspunifiedsequenceparallelism,gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual` partially mitigate these issues but introduce new limitations: head‑sharded designs impose divisibility constraints and reduce flexibility, ring‑style P2P schemes scale but incur large communication and redundancy under sparse/varlen masks. While recent efforts {cite}`wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm,megatron-lm-hybrid-cp-pr-2054` dynamically adjust CP sizes to avoid unnecessary sharding and redundant communication for shorter sequences, they still incur extra memory overhead for NCCL buffers and involve complex scheduling to balance loads and synchronize across different subsets of ranks.

Crucially, existing methods do not simultaneously (1) provide a unified, distributable representation for a wide class of mask patterns, (2) guarantee balanced compute across context‑parallel (CP) ranks for arbitrarily structured masks, and (3) eliminate unnecessary data movement while enabling robust compute/communication overlap.

MagiAttention addresses these gaps by prioritizing kernel‑level flexibility together with distributed-level scalability, which depends on meeting the following fundamental conditions:

- <b>Linearly Scalable Attention Kernel</b>: The performance of the attention kernel should not degrade as CP size increases. To this end, we introduce [Flex-Flash-Attention](#flex-flash-attn), an extension of FlashAttention-3 (FA3), which natively considers the efficiency impact of attention mask partitioning in distributed environments. It supports distributable mask representations with a tailored kernel implementation to ensure scalability while accommodating a broader range of attention mask types.
- <b>Balanced Computational Workloads</b>: Imbalances in the computational load across CP ranks lead to unavoidable idle bubbles that hinder scalability. MagiAttention is natively designed to ensure [Computation Load Balancing](#comp-load-balance), mitigating such inefficiencies.
- <b>Full Overlap of Communication and Computation</b>: Without sufficient overlap, increasing CP size results in communication-induced idle time on GPUs, impairing scalability. MagiAttention introduces novel [Zero-Redundant Communication Primitives](#zero-redundant-comm) to minimize communication overhead, along with an [Adaptive Multi-Stage Overlap](#multi-stage-overlap) strategy that enables effective communication-computation overlap.

By coordinating a mask‑flexible kernel (FFA), a load‑balancing dispatcher, and zero‑redundancy communication with adaptive overlap, MagiAttention supports a broad spectrum of attention patterns while delivering distributed-level linear scalability across realistic ultra‑long and heterogeneous training workloads.

Below, we briefly review current CP strategies in [Related Work](#related-work), present the key designs in [Methodology](#methodology), and report comprehensive experimental results that validate the approach in [Experiments](#experiments).

We further elaborate upon extended functionalities, optimization techniques, and next-generation design in [Discussion](#discussion), followed by the [Future Work](#future-work) section. Our evolving exploration seeks to broaden the scope and redefine the frontiers of distributed attention, optimizing its performance for large-scale model training and extending its efficacy to inference scenarios in the future.


## Related Work

To handle ultra‑long contexts, context parallelism (CP) is essential, but existing CP strategies do not meet the real-world demanding settings.

DeepSpeed’s `Ulysses` {cite}`jacobs2023deepspeed` uses head-sharded attention with All-to-All transforms; it is easy to integrate but requires the number of heads to be divisible by the CP size, limiting scalability (e.g., GQA and when combined with head-aware tensor parallelism) {cite}`shoeybi2020megatronlm,korthikanti2022reducing`.

`Ring-Attention` {cite}`li2021sequence,liu2023ringattentionblockwisetransformers,wang2024tokenringefficientparallelismframework` keeps sequence-sharded activations and relies on multi-stage ring-style P2P communication for online attention and overlap {cite}`rabe2021self,dao2022flashattention,wang2022overlap`. It scales better than head-sharding but incurs large communication volumes and inefficient P2P primitives as CP size grows. Hybrid 2D schemes like `USP` {cite}`fang2024uspunifiedsequenceparallelism` and `LoongTrain` {cite}`gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual` combine `Ulysses` and `Ring-Attention` to reduce their weaknesses but still lack the fundamental efficiency and scalability needed for ultra‑long contexts.

Irregular masks (e.g., varlen) worsen these issues (see {numref}`ring_attn_load_balance` below). Naive <em>sequential even sharding</em> creates uneven mask-area distribution and imbalanced compute across ranks. Custom <em>zigzag sharding</em> {cite}`ring_flash_attention_issue2` can rebalance specific varlen causal patterns but causes fragmentation, excessive padding, and kernel slowdowns, and it does not generalize to patterns such as the <em>varlen block-causal mask</em> used in autoregressive video generation for [Magi-1](https://github.com/SandAI-org/MAGI-1).

```{figure} ../../../assets/magi_attn/comp/ring_attn_load_balance.png
:name: ring_attn_load_balance
:align: center
:width: 800px
:alt: Ring-Attention Load Balancing

Illustration of `Ring-Attention`'s sharding strategies for load balancing: (a) full mask — sequential sharding across the global mask; (b) causal mask — tailored *zigzag sharding* {cite}`ring_flash_attention_issue2`; (c) varlen full mask — sequential sharding per packed sample; (d) varlen causal mask — per-sample *zigzag sharding*, which increases fragmentation and padding and degrades performance.
```

Second, communication overhead worsens under sparse varlen masks because entire sequence chunks are transferred to all CP ranks—even when many ranks do not need them—yielding over **30% redundant communication**, as shown in [Zero-Redundant Communication Primitives](#zero-redundant-communication-primitives). Third, these inefficiencies undermine pipeline compute–communication overlap: imbalanced workloads and excessive communication make overlap fragile and constrain scalability.

Recent efforts like `DCP` {cite}`wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm` and `Hybrid-CP` {cite}`megatron-lm-hybrid-cp-pr-2054` reduce redundant sharding by dynamically assigning CP group sizes per sample based on sequence length. However, they introduce significant scheduling complexity, frequent cross-group synchronization, and extra NCCL buffer memory, lacking of a bottom-up redesign required for robust, mask-flexible, and scalable distributed attention.


## Methodology

### Flex-Flash-Attention

#### AttnSlice Representation

Flash-Attention {cite}`dao2022flashattention,dao2023flashattention,shah2024flashattention3fastaccurateattention,dao2025flashattention_cute` delivers high throughput, memory efficiency, and native support for varlen-packed inputs, making it a cornerstone for large-scale training. However, its kernels assume regular mask structure and do not handle irregular, rank-distributed masks efficiently—causing fragmentation, load imbalance, excess padding, and higher communication—so a mask‑flexible kernel that preserves Flash‑Attention’s performance is required {cite}`pytorch_sdpa,dong2024flexattentionprogrammingmodel,wang2025flashmaskefficientrichmask`.

Therefore, we introduce Flex-Flash-Attention (FFA), a kernel designed for distributed settings that flexibly handles diverse attention masks. FFA adopts a <b>distributable</b> representation that decomposes an irregular mask into multiple computational units called {math}`\mathrm{AttnSlice}`. Each {math}`\mathrm{AttnSlice}` is the triplet {math}`\mathrm{(QRange, KRange, MaskType)}`, denoting a submask confined to a contiguous 2D query–key region (see {numref}`attnslice_interpret` below).

```{figure} ../../../assets/magi_attn/ffa/attnslice_interpret.png
:name: attnslice_interpret
:align: center
:width: 1000px
:alt: AttnSlice Formulation

Illustration of the {math}`\mathrm{AttnSlice}` formulation for an irregular mask. The mask is decomposed into multiple {math}`\mathrm{AttnSlice}` units, allowing fractal patterns to be re-expressed after redistribution across CP ranks to support distributed attention. Note that computation load balancing across CP ranks is not considered in this illustration.
```

As illustrated in {numref}`mask_with_attn_slice` below, this formulation expresses a wide range of attention masks—including the varlen block-causal mask used in [Magi-1](https://github.com/SandAI-org/MAGI-1)—as compositions of multiple triplets. These representations remain valid after sharding and rearrangement across ranks, making FFA well suited for distributed attention computation.

```{figure} ../../../assets/magi_attn/ffa/mask_with_attn_slice.png
:name: mask_with_attn_slice
:align: center
:width: 1000px
:alt: AttnSlice Mask Patterns

Examples of mask patterns expressed using {math}`\mathrm{AttnSlice}`: (a)–(d) are standard FA3-compatible patterns; (e)–(h) are irregular masks beyond Flash-Attention’s capability—e.g., the varlen block-causal mask—which FFA handles seamlessly while preserving FA3-comparable performance.
```

#### AttnSlice-level Parallelism in FFA

Built on Flash-Attention 3 (FA3) kernels {cite}`shah2024flashattention3fastaccurateattention`, Flex-Flash-Attention (FFA) leverages Hopper GPUs' TMA feature {cite}`nvidia2024accelerating` and implements {math}`\mathrm{AttnSlice}`-level parallelism with atomic operations for correctness (illustrated in {numref}`ffa_slice_atomic_reduce` below). FFA delivers MFU comparable to FA3 while supporting the flexible {math}`\mathrm{AttnSlice}` formulation—see [Attention Kernel Benchmark](cp_benchmark.html#attention-kernel-benchmark) for detailed performance and flexibility comparisons.

```{figure} ../../../assets/magi_attn/ffa/ffa_slice_atomic_reduce.png
:name: ffa_slice_atomic_reduce
:align: center
:width: 1000px
:alt: FFA Slice Atomic Reduction

Illustration of the FFA forward and backward kernels: data loading, on-chip computation, and atomic reduction for slice-level parallelism.
```

#### Basic Mask Types in AttnSlice

Although most mask patterns can be expressed with {math}`\mathrm{AttnSlice}` using the common types {math}`\lbrace\texttt{FULL}, \texttt{CAUSAL}\rbrace`, some patterns—e.g., {math}`\textit{sliding-window}`—become inefficient because they require expressing each row individually. To represent such patterns compactly, we introduce two additional mask types, {math}`\lbrace\texttt{INV-CAUSAL}, \texttt{BI-CAUSAL}\rbrace`. The following {numref}`attn_slice_mask_type_sq=sk`, {numref}`attn_slice_mask_type_sq<sk`, and {numref}`attn_slice_mask_type_sq>sk` illustrate examples of the current {math}`4` supported mask types.

```{figure} ../../../assets/magi_attn/ffa/attn_slice_mask_type_sq=sk.png
:name: attn_slice_mask_type_sq=sk
:align: center
:width: 650px
:alt: AttnSlice Mask Types (seqlen_q = seqlen_k)

Illustrates the four supported mask types for `seqlen_q == seqlen_k`. Note: in this setting, {math}`\texttt{BI-CAUSAL}` reduces to a mask where only the principal diagonal cells are valid.
```

```{figure} ../../../assets/magi_attn/ffa/attn_slice_mask_type_sq<sk.png
:name: attn_slice_mask_type_sq<sk
:align: center
:width: 650px
:alt: AttnSlice Mask Types (seqlen_q < seqlen_k)

Illustration of the four supported mask types when `seqlen_q < seqlen_k`. This configuration commonly occurs when employing {math}`\texttt{INV-CAUSAL}` and {math}`\texttt{BI-CAUSAL}` masks.
```

```{figure} ../../../assets/magi_attn/ffa/attn_slice_mask_type_sq>sk.png
:name: attn_slice_mask_type_sq>sk
:align: center
:width: 650px
:alt: AttnSlice Mask Types (seqlen_q > seqlen_k)

Illustration of the four supported mask types for `seqlen_q > seqlen_k`. Note that {math}`\texttt{BI-CAUSAL}` is empty and contains no valid cells.
```

Using the four supported mask types, we illustrate common {math}`\textit{sliding-window}`-style masks expressed via the {math}`\mathrm{AttnSlice}` formulation (see {numref}`sw_mask_with_slice` below).

```{figure} ../../../assets/magi_attn/ffa/sw_mask_with_slice.png
:name: sw_mask_with_slice
:align: center
:width: 1000px
:alt: Sliding-Window Mask Patterns

Examples of common {math}`\textit{sliding-window}`-style mask patterns formulated by {math}`\mathrm{AttnSlice}`.
```


### Computation Load-Balancing

#### Dispatch Solver

In context-parallel training, heterogeneous attention masks across CP ranks create imbalanced computational workloads. `Ring-Attention` (see [Related Work](#related-work)) uses a partitioning strategy tailored to causal masks and therefore does not generalize to arbitrary patterns. To address this, we propose a generic, efficient `dispatch solver` that balances workload across CP ranks for diverse attention types.

Concretely, we adopt a chunk-wise permutable sharding: partition the global mask evenly along the query dimension into chunks, each associated with a submask area {math}`\lbrace(C_i, \mathrm{Area}(C_i))\rbrace_{i=1}^n`, where {math}`C_i` denotes the i-th chunk, {math}`\mathrm{Area}(C_i)` is its mask area, {math}`n = \frac{seqlen}{\textit{chunk_size}}`, and {math}`\textit{chunk_size}` is a tunable granularity parameter.

These chunks are assigned equally to {math}`\textit{cp_size}` buckets so every bucket contains the same number of chunks (preserving token-level balance for non-attention stages). Each bucket's total mask workload is the summed submask area, written as {math}`\lbrace(B_j, \mathrm{SumArea}(B_j))\rbrace_{j=1}^{\textit{cp_size}}`.

Under this formulation, load balancing reduces to a combinatorial assignment problem: find an optimal mapping {math}`f^*: \lbrace C_i\rbrace_{i=1}^n \rightarrow \lbrace B_j\rbrace_{j=1}^{\textit{cp_size}}` that minimizes the maximum per-bucket area, as shown in the Eq {eq}`eq:comp_load_balance` below.

```{math}
:label: eq:comp_load_balance

\begin{aligned}
  &f^* = \arg \min\limits_{f}\max\limits_{j}\left\{\mathrm{SumArea}(B_j)\right\} \label{eq:comp_load_balance}\\
  &\text{s.t.}\;\;|B_j| = \frac{n}{\textit{cp_size}}, \;\; seqlen \;\%\; (\textit{cp_size} \times \textit{chunk_size}) = 0\nonumber
\end{aligned}
```

Since this problem is NP-hard and mask patterns change across micro-batches, solving it exactly per iteration is impractical. We therefore use a practical greedy Min-Heap algorithm (illustrated in {numref}`min_hp_alg` below) that runs in {math}`O(n\log n)` and yields a fast, effective assignment with minimal runtime overhead.

```{figure} ../../../assets/magi_attn/comp/min_hp_alg.png
:name: min_hp_alg
:align: center
:width: 1000px
:alt: Greedy Load-Balance Dispatch Algorithm

Greedy Load-Balance Dispatch Algorithm via Min-Heap
```

#### Static Attn Solver

...

#### Dynamic Attn Solver

...

### Zero-Redundant Communication Primitives

#### Ring P2P Redundancy Analysis

Ring-style implementations rely on point-to-point (P2P) send/recv primitives that lack fine-grained communication control, causing unnecessary data movement. To quantify this, we record remote key-value ({math}`\mathrm{KV}`) requests and their gradients ({math}`\mathrm{dKV}`) under a causal mask as a simple example shown in {numref}`ring_p2p_redundancy`: in the forward pass {math}`\mathrm{KV}_0` must be sent to all devices via `BroadCast`, while {math}`\mathrm{dKV}_0` requires to be reduced via `AllReduce` during the backward. However, {math}`\mathrm{KV}_7` is required ONLY locally for its host {math}`rank_7` yet still circulates across all devices. This redundant even dissemination—and its cost—becomes more severe for varlen mask patterns.

```{figure} ../../../assets/magi_attn/comm/ring_p2p_redundancy.png
:name: ring_p2p_redundancy
:align: center
:width: 1000px
:alt: Ring P2P Redundant Communication

Examples of redundant communication in Ring P2P with heterogeneous masks: (a) a simple causal mask incurs **25%** redundant communication; (b) irregular masks, e.g., the varlen block-causal mask with the last global block, can exceed **33%** redundancy.
```

#### Group Collective Primitives

To address this, as illustrated in the {numref}`group_gather_reduce_all2allv` below, we introduce two communication primitives: `GroupCast` and `GroupReduce`, which model the communication patterns of low-demand {math}`\mathrm{KV}` and {math}`\mathrm{dKV}`. For example, in the causal mask, {math}`\mathrm{KV}_5` on {math}`\mathrm{rank}_2` is required only by {math}`\{\mathrm{Q}_6,\mathrm{Q}_7\}` and should be sent exclusively to the target ranks {math}`\{\mathrm{rank}_0, \mathrm{rank}_1\}` via `GroupCast`, while the partial {math}`\mathrm{dKV}_5` is collected and reduced back to {math}`\mathrm{rank}_2` via `GroupReduce` accordingly.

```{figure} ../../../assets/magi_attn/comm/group_gather_reduce_all2allv.png
:name: group_gather_reduce_all2allv
:align: center
:width: 1000px
:alt: GroupCast/GroupReduce Primitives

Illustration of `GroupCast/GroupReduce` primitives implemented atop `AlltoAll-v` to achieve zero redundancy, shown using the varlen block-causal mask with the last global block. (a) For forward and backward passes, `GroupCast` builds a transfer table for {math}`\mathrm{KV}` send/receive buffers, invokes `AlltoAll-v`, and uses a custom `Range-Gather` kernel for pre-/post-processing. (b) In the backward pass, `GroupReduce` aggregates partial {math}`\mathrm{dKV}` via `AlltoAll-v`, employing `Range-Gather` for pre-processing and `Range-Scatter-Reduce` for post-processing.
```

#### AlltoAll-v Implementation

As no existing communication kernels support these group collective primitives, we implement them upon `AlltoAll-v` as a prototype, achieving zero-redundant communication in both forward and backward passes. However, this approach introduces extra pre-/post-processing overhead, similar to (un)permutation in expert parallelism (EP) {cite}`gale2022megablocks`. While kernel fusion mitigates the overhead, a dedicated implementation of `GroupCast` and `GroupReduce` remains a key direction for future work.

#### Native Implementation

...

### Multi-Stage Computation/Communication Overlap

Leveraging previous optimizations, we achieve high-performance computation through an efficient kernel and balanced workload dispatch, while minimizing communication overhead with our new primitives. To drive true linear scalability, we further improve end-to-end performance by introducing a multi-stage compute-communication overlap strategy, that effectively hides communication latency and adaptively optimizes overlap through manual or automatic tuning.

Similar to prior works {cite}`liu2023ringattentionblockwisetransformers,zhao2023pytorch,async_tensor_parallelism_in_pytorch`, we schedule pipeline stages to overlap computation with communication for both forward and backward passes, as shown in the following {numref}`multi_stage_overlap_fwd_bwd`. Each {math}`\mathrm{rank}_i` first partitions its remote {math}`\mathrm{KV}`/{math}`\mathrm{dKV}` communication into stages.

```{figure} ../../../assets/magi_attn/mso/multi_stage_overlap_fwd_bwd.png
:name: multi_stage_overlap_fwd_bwd
:align: center
:width: 1000px
:alt: Multi-Stage Overlap Scheduling

Schematic of Magi Attention's multi-stage overlap scheduling. (a) Forward pass: 4-stage scheduling overlaps computation (partial attention outputs and {math}`\textit{lse}` factors) with prefetching of next-stage {math}`\mathrm{KV}` requests (where applicable), hiding all communication overhead with the final stage's computation exposed. (b) Backward pass: 3-stage scheduling overlaps computation (partial {math}`\mathrm{dQ}`, {math}`\mathrm{dKV}`) with prefetching of next-stage {math}`\mathrm{KV}` requests and reduction of prior {math}`\mathrm{dKV}` requests, hiding all communication overhead except the {math}`\mathrm{dKV}` reduction of the final stage.
```

In the forward pass, the scheduler first launches the `GroupCast` kernel to prefetch the next remote {math}`\mathrm{KV}`, then asynchronously executes the FFA kernel for partial attention computation, hiding all communication behind computation. To prevent all SMs from being occupied by the attention kernel, by default, we ensure the communication kernel picked first by setting `CUDA_DEVICE_MAX_CONNECTIONS=1` {cite}`cuda_device_max_connections_issue`. However, we also support relax this constraint by setting an non-zero `sm_margin` argument for the FFA kernel, to preserve some SMs for communication kernels to be launched.

In the backward pass, besides prefetching the next {math}`\mathrm{KV}`, the `GroupReduce` kernel reduces the last {math}`\mathrm{dKV}` in a separate CUDA stream before launching the FFA kernel for the current stage, ensuring communication is overlapped across all stages except the final {math}`\mathrm{dKV}` reduction. Due to PyTorch's one-to-one mapping for process groups and collective communication streams including `AlltoAll-v` {cite}`collectives_nccl_stream_issue`, we internally use an additional CP group for `GroupReduce` to enable full overlap between communication kernels in the backward pass.

To adaptively control overlap granularity, we further introduce a tunable hyperparameter, {math}`\textit{num_stages}`, accounting for varying compute-to-communication ratios across training setups, microbatches, or between forward and backward passes. This parameter can be manually configured or automatically determined by our `overlap solver`, with a simple dynamic search algorithm as shown in the {numref}`dynamic_mso_alg` below.

```{figure} ../../../assets/magi_attn/mso/dynamic_mso_alg.png
:name: dynamic_mso_alg
:align: center
:width: 800px
:alt: Dynamic Overlap Stage Search Algorithm

Dynamic Overlap Stage Search Algorithm
```


## Experiments

### Attention Benchmark


## Discussion

### Attention Sink

Please check this [blog post](https://sandai-org.github.io/MagiAttention/blog/ffa_with_sink) about how to integrate Flex-Flash-Attention, MagiAttention, as well as Flash-Attention, with the learnable attention sink mechanism.

### Muon QK-Clip Max Logits

...


## Future Work

- [ ] **[WIP]** Optimize `Flex-Flash-Attention` kernels on Hopper for improved performance, with emphasis on <u>sparse attention</u> scenarios.
- [ ] **[WIP]** Implement native `GroupCast` and `GroupReduce` communication kernels to reduce communication overhead and lower compute occupancy.
- [ ] **[WIP]** Extend the `DynamicAttnSolver` to better handle dynamic mask patterns (e.g., <u>hybrid attention</u>, <u>sparse attention</u>) for lower communication and improved load balance.
- [ ] Optimize `DistAttnSolver` to cut CPU meta-info overhead and enhance comp/comm overlap.
- [ ] Implement native `Flex-Flash-Attention` kernels on Blackwell to replace the temporary `FFA_FA4` backend.
- [ ] Port `Flex-Flash-Attention` to additional GPU architectures (e.g., Ampere).
- [ ] Expand documentation with more examples and a tuning guide for varied training scenarios.
- [ ] Prepare a standalone technical report/paper detailing MagiAttention.
- [ ] Add support for additional attention patterns, including cross-attention and inference use cases.
- [ ] Upgrade `MagiAttention` to a distributed native `Flex-Flash-Attention` kernel.

<details>
<summary>Done</summary>

- [x] Support MagiAttention on Blackwell with a temporary `FFA_FA4` backend.
- [x] Support `DynamicAttnSolver` with query/output communication pattern to reduce communication in cases where KV-only communication is suboptimal.
- [x] Prototype native `GroupCast` and `GroupReduce` primitives with inter-/intra-node hierarchical optimization based on [DeepEP](https://github.com/deepseek-ai/DeepEP).
- [x] Support learnable attention sink integration with [StreamingLLM](https://arxiv.org/abs/2309.17453).
- [x] Refactor `DistAttnSolver` to support all four mask types and full overlapping strategies.
- [x] Improve the `Dispatch Solver` to reduce communication volume while maintaining compute balance, especially for varlen masks.
- [x] Build a comprehensive `CP Benchmark` validating MagiAttention across mask patterns and training settings.
- [x] Provide `Documentation` covering `Installation`, `QuickStart`, `API reference`, and `Environment Variables`.

</details>

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

```{bibliography} refs/magi_attn.bib
```
