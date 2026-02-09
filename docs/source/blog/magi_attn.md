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

Overview of MagiAttention: (1) FFA - an optimized kernel based on Flash-Attention 3, further supports flexible mask patterns; (2) The dispatch solver shards ultra‑long data and dispatches for load-balanced computation; (3) Group‑Cast and Group‑Reduce primitives eliminate redundant communication; (4) The overlap solver adaptively partitons multi-stage computation/communication for optimal overlap; (5) Forward and backward timelines scheduled by MagiAttention. With all components together, MagiAttention enables linear scalability in training with ultra‑long contexts and heterogeneous masks.
```

Training large-scale video‑generation models faces two tightly coupled challenges: (1) ultra‑long contexts—reaching millions of tokens (e.g., **~4M**)—which make attention prohibitively expensive in compute and memory, and (2) highly heterogeneous, irregular attention masks (e.g., block‑causal + Patch‑and‑Pack) that break assumptions of existing kernels and distributed layouts, leading to fragmentation, load imbalance, wasted padding, and large communication overhead. 

These same constraints also affect (multimodal) LLMs that aim to support ultra‑long histories and flexible masking for agentic tasks with large retrievals and deep reasoning. <u>Therefore, we require an efficient, mask-flexible, and scalable distributed attention solution</u>.

To address these challenges, we propose [MagiAttention](https://github.com/SandAI-org/MagiAttention), which targets these bottlenecks with **kernel-level flexibility**, while achieving **distributed-level linear scalability** across a broad range of training scenarios, particularly for those involving ultra-long contexts and heterogeneous masks like [Magi-1](https://github.com/SandAI-org/MAGI-1).


## Introduction

Training large-scale autoregressive diffusion models for video generation (e.g., [Magi-1](https://github.com/SandAI-org/MAGI-1)) creates two tightly coupled system challenges. First, training contexts can reach millions of tokens, so naive quadratic attention or inadequately sharded algorithms quickly become infeasible in both compute and memory. Second, practical data pipelines—for example, block‑causal attention combined with Patch‑and‑Pack (PnP) processing {cite}`dehghani2023patchnpacknavit` — produce highly heterogeneous, irregular masks and variable sequence lengths that violate assumptions made by standard attention kernels and distributed layouts. The combined effect is severe fragmentation, imbalanced compute across ranks, excessive padding, and large, often redundant, communication volumes.

Prior context‑parallel solutions {cite}`jacobs2023deepspeed,liu2023ringattentionblockwisetransformers,fang2024uspunifiedsequenceparallelism,gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual` partially mitigate these issues but introduce new limitations: head‑sharded designs impose divisibility constraints and reduce flexibility, ring‑style P2P schemes scale but incur large communication and redundancy under sparse/varlen masks. While recent efforts {cite}`wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm` dynamically adjust CP sizes to avoid unnecessary sharding and redundant communication for shorter sequences, they still incur extra memory overhead for NCCL buffers and involve complex scheduling to balance loads and synchronize across different subsets of ranks.

Crucially, existing methods do not simultaneously (1) provide a unified, distributable representation for a wide class of mask patterns, (2) guarantee balanced compute across context‑parallel (CP) ranks for arbitrarily structured masks, and (3) eliminate unnecessary data movement while enabling robust compute/communication overlap.

MagiAttention addresses these gaps by prioritizing kernel‑level flexibility together with distributed-level scalability, which depends on meeting the following fundamental conditions:

- <b>Linearly Scalable Attention Kernel</b>: The performance of the attention kernel should not degrade as CP size increases. To this end, we introduce [Flex-Flash-Attention](#flex-flash-attn), an extension of FlashAttention-3 (FA3), which natively considers the efficiency impact of attention mask partitioning in distributed environments. It supports distributable mask representations with a tailored kernel implementation to ensure scalability while accommodating a broader range of attention mask types.
- <b>Balanced Computational Workloads</b>: Imbalances in the computational load across CP ranks lead to unavoidable idle bubbles that hinder scalability. MagiAttention is natively designed to ensure [Computation Load Balancing](#comp-load-balance), mitigating such inefficiencies.
- <b>Full Overlap of Communication and Computation</b>: Without sufficient overlap, increasing CP size results in communication-induced idle time on GPUs, impairing scalability. MagiAttention introduces novel [Zero-Redundant Communication Primitives](#zero-redundant-comm) to minimize communication overhead, along with an [Adaptive Multi-Stage Overlap](#multi-stage-overlap) strategy that enables effective communication-computation overlap.

By coordinating a mask‑flexible kernel (FFA), a load‑balancing dispatcher, and zero‑redundancy communication with adaptive overlap, MagiAttention supports a broad spectrum of attention patterns while delivering distributed-level linear scalability across realistic ultra‑long and heterogeneous training workloads.

Below, we briefly review current CP strategies in [Related Work](#related-work), present the key designs in [Methodology](#methodology), and report comprehensive experimental results that validate the approach in [Experiment](#experiment).

We further elaborate upon extended functionalities, optimization techniques, and next-generation design in [Discussion](#discussion), followed by the [Future Work](#future-work) section. Our ongoing exploration seeks to broaden the scope and redefine the frontiers of distributed attention, optimizing its performance for large-scale model training and extending its efficacy to inference scenarios in the future.


## Related Work

To tackle the ultra-long context challenge in large-scale model training, the distributed attention mechanism, or context parallelism (CP), is essential.

However, current strategies fall short in our demanding settings. DeepSpeed’s Ulysses {cite}`jacobs2023deepspeed` leverages the multi-head characteristic for head-sharded, sequence-complete propagation in the attention module, and head-complete, sequence-sharded propagation elsewhere, with transformation between parallel placements efficiently handled by All-to-All collective communication. While easy to integrate, it has scalability limitations, requiring the number of heads to be divisible by the CP size, particularly in GQA settings or with tensor parallelism {cite}`shoeybi2020megatronlm,korthikanti2022reducing`. In contrast, Ring-Attention {cite}`li2021sequence,liu2023ringattentionblockwisetransformers,wang2024tokenringefficientparallelismframework` keeps sequence-sharded activations and accesses the complete sequence through multi-stage ring-style point-to-point (P2P) communication to perform online attention propagation {cite}`rabe2021self,dao2022flashattention` and pipeline compute-communication overlap {cite}`wang2022overlap`. Though more scalable, it suffers from significant communication overhead due to large communication volumes and inefficient P2P send/receive primitives over the entire CP group as the CP size increases. Some following works {cite}`fang2024uspunifiedsequenceparallelism,gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual` combine Ulysses and Ring-Attention in a 2D distributed attention approach to mitigate their limitations, yet still lack the efficiency and scalability required for our ultra-long context settings.

Worse still, for irregular attention mask patterns like the aforementioned varlen masks, classic Ring-Attention-based CP strategies are facing more challenges, besides the attention kernel limitations. First, the naive <em>sequential even sharding</em> along the sequence dimension causes uneven distribution of the varlen mask area, leading to imbalanced computational loads across CP ranks. Although the customized <em>zigzag sharding</em> design {cite}`ring_flash_attention_issue2` balances loads for specific (varlen) causal mask patterns in the following figure, it results in kernel performance degradation from fragmented sharding and excessive padding, and does not generalize well to other patterns including the <em>varlen block-causal mask</em> met in autoregressive video generation.

```{figure} ../../../assets/magi_attn/comp/ring_attn_load_balance.png
:align: center
:width: 800px
:alt: Ring-Attention Load Balancing

Illustration of Ring-Attention's customized sharding strategies for load balancing. (a) Full mask uses sequential sharding for the global mask; (b) Causal mask employs tailored *zigzag sharding* {cite}`ring_flash_attention_issue2`; (c) Varlen full mask applies sequential sharding per local mask (one per packed sample); (d) Varlen causal mask uses *zigzag sharding* per local mask, causing performance degradation from fragmentation and padding.
```

Second, the communication overhead issue is exacerbated under sparse varlen mask settings, as entire sequence chunks are still transferred across all CP ranks even when not all ranks require them, might causing over 30% redundant communication costs as illustrated in [Zero-Redundant Comm](#zero-redundant-comm). Third, the former challenges cause the pipeline compute-communication overlap strategy fails more often due to imbalanced loads and large communication overheads, further limiting scalability.

Recent efforts {cite}`wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm` attempt to address these issues by dynamically assigning communication groups of specific CP sizes to different samples based on their sequence lengths, to reduce unnecessary sharding and redundant communication for shorter sequences. However, these methods suffer from extra memory overhead for NCCL buffers and complex scheduling for computation load-balance and synchronization across different sets of ranks.


## Methodology

### Flex-Flash-Attn


Flash Attention {cite}`dao2022flashattention,dao2023flashattention,shah2024flashattention3fastaccurateattention` is foundational in large-scale model training for its superior performance and support for varlen-packed data. However, it offers limited support for irregular attention masks, particularly when such patterns are distributed across CP ranks, resulting in increased complexity and underscoring the need for a more flexible attention kernel {cite}`pytorch_sdpa, dong2024flexattentionprogrammingmodel,wang2025flashmaskefficientrichmask` without compromising performance.

Therefore, we introduce Flex-Flash-Attention (FFA), which is natively designed for distributed scenarios and provides greater flexibility in handling diverse attention mask types. The core idea behind FFA is to generalize a <b>distributable</b> formulation for irregular attention masks by decomposing the entire mask into multiple computational units, each referred to as an {math}`\mathrm{AttnSlice}` . Each {math}`\mathrm{AttnSlice}` is defined by a triplet {math}`\mathrm{(QRange, KRange, MaskType)}`, which specifies a submask with a basic shape bounded by a contiguous 2D query-key region as seen in the figure below.


```{figure} ../../../assets/magi_attn/ffa/attnslice_interpret.png
:align: center
:width: 1000px
:alt: AttnSlice Formulation

Illustration of {math}`\mathrm{AttnSlice}` formulation for some irregular mask. It decomposes the original mask into multiple {math}`\mathrm{AttnSlice}`'s and allows re-expression of fractal masks after rearrangement across CP ranks, making it suitable for distributed attention. Note that computation load balance across CP ranks is not considered in this illustration.
```


Using this formulation, as shown in the figure below, a wide variety of commonly used attention masks, including the varlen block-causal mask for autoregressive video generation, can be expressed as a composition of multiple such triplets even after sharding and rearrrangement in distributed settings, making FFA highly suitable for distributed attention computation.


```{figure} ../../../assets/magi_attn/ffa/mask_with_attn_slice.png
:align: center
:width: 1000px
:alt: AttnSlice Mask Patterns

Examples of mask patterns formulated by {math}`\mathrm{AttnSlice}`. (a)-(d) Standard FA3-compatible patterns; (e)-(h) Irregular masks beyond Flash-Attention's capabilities, including the varlen block-causal mask, which FFA supports seamlessly while maintaining performance comparable to FA3.
```


Built on Flash-Attention 3 (FA3) kernels {cite}`shah2024flashattention3fastaccurateattention`, Flex-Flash-Attention (FFA) leverages Hopper GPUs' TMA feature {cite}`nvidia2024accelerating` and introduces slice-level parallelism with atomic operations for correctness as illustrated in the following figure, achieving comparable MFU to FA3 while supporting the flexible {math}`\mathrm{AttnSlice}` formulation (see [Kernel-Level Experiments](#kernel-level) for FFA performance and flexibility benchmarks compared to other attention kernels).

```{figure} ../../../assets/magi_attn/ffa/ffa_slice_atomic_reduce.png
:align: center
:width: 1000px
:alt: FFA Slice Atomic Reduction

Illustration of the FFA forward and backward kernels' loading and atomic reduction for slice-level parallelism.
```

However, even though we can express most mask patterns using {math}`\mathrm{AttnSlice}` with two common mask type {math}`\lbrace\mathrm{FULL}, \mathrm{CAUSAL}\rbrace`, but when comes to the mask patterns such as {math}`\textit{sliding-window}`, they are quite inefficient (*in such case, we have to express each row one by one*). Therefore, we design two new but a little bit bizarre mask types named {math}`\lbrace\text{INV-CAUSAL}, \text{BI-CAUSAL}\rbrace` to efficiently represent more specific mask patterns, and provide some basic examples about the current {math}`4` mask types we support in the following figures.

```{figure} ../../../assets/magi_attn/ffa/attn_slice_mask_type_sq=sk.png
:align: center
:width: 800px
:alt: AttnSlice Mask Types (seqlen_q = seqlen_k)

Illustration of the four supported mask types when {math}`\text{seqlen}_q = \text{seqlen}_k`. (Note: In this case, {math}`\text{BI-CAUSAL}` represents a mask with only the principal diagonal cells being valid.)
```

```{figure} ../../../assets/magi_attn/ffa/attn_slice_mask_type_sq<sk.png
:align: center
:width: 800px
:alt: AttnSlice Mask Types (seqlen_q < seqlen_k)

Illustration of the four supported mask types when {math}`\text{seqlen}_q < \text{seqlen}_k`. (Note: This is the common case when we adopt {math}`\text{INV-CAUSAL}` and {math}`\text{BI-CAUSAL}`.)
```

```{figure} ../../../assets/magi_attn/ffa/attn_slice_mask_type_sq>sk.png
:align: center
:width: 800px
:alt: AttnSlice Mask Types (seqlen_q > seqlen_k)

Illustration of the four supported mask types when {math}`\text{seqlen}_q > \text{seqlen}_k`. (Note: In this case, {math}`\text{BI-CAUSAL}` represents an empty mask with no valid cells.)
```

Based on the four mask types currently supported, we provide examples of how to express common {math}`\textit{sliding-window}`-style mask patterns using the {math}`\mathrm{AttnSlice}` formulation, as illustrated in the figure below.

```{figure} ../../../assets/magi_attn/ffa/sw_mask_with_slice.png
:align: center
:width: 1000px
:alt: Sliding-Window Mask Patterns

Examples of common {math}`\textit{sliding-window}`-style mask patterns formulated by {math}`\mathrm{AttnSlice}`.
```


### Comp Load-Balance

In context-parallel settings, different CP ranks may be assigned heterogeneous attention masks, resulting in imbalanced computational workloads across ranks. Ring-Attention, as mentioned in [Related Work](#related-work), employs a specialized partitioning strategy designed specifically for causal attention, which limits its applicability to more general attention patterns. To overcome this limitation, we propose a generic and efficient dispatch solver that enables balanced workload distribution across CP ranks for a broad range of attention types.

First, to enable finer-grained control, we propose a chunk-wise permutable sharding strategy as seen in [Overview](#overview). Specifically, the entire mask is evenly partitioned along the query-dimension into chunks, each associated with a submask area: {math}`\lbrace(C_i, \mathrm{Area}(C_i))\rbrace_{i=1}^n`, where {math}`C_i` indicates i-th chunk, {math}`\mathrm{Area}(C_i)` is the mask area of {math}`C_i`, {math}`n` is {math}`\frac{seqlen}{\textit{chunk_size}}`, and {math}`\textit{chunk_size}` is a hyperparameter controlling granularity.

These chunks are then equally assigned to {math}`\textit{cp_size}` buckets, with each bucket containing the exact same number of chunks to ensure token-level load balance in non-attention modules, attaching with a summed submask area, denoted as {math}`\lbrace(B_j, \mathrm{SumArea}(B_j))\rbrace_{j=1}^{\textit{cp_size}}`.


With above strategy, we could fine-grained control the computational workloads of each CP rank, and the load-balancing dispatch becomes a combinatorial optimization problem, defined as finding an optimal mapping function {math}`f^*: \lbrace C_i\rbrace_{i=1}^n \rightarrow \lbrace B_j\rbrace_{j=1}^{\textit{cp_size}}` follows:

```{math}
:label: eq:comp_load_balance

\begin{aligned}
  &f^* = \arg \min\limits_{f}\max\limits_{j}\left\{\mathrm{SumArea}(B_j)\right\} \label{eq:comp_load_balance}\\
  &\text{s.t.}\;\;|B_j| = \frac{n}{\textit{cp_size}}, \;\; seqlen \;\%\; (\textit{cp_size} \times \textit{chunk_size}) = 0\nonumber
\end{aligned}
```

However, this optimization is a known NP-hard problem, making it impractical to find an optimal solution on-the-fly during each training iteration, especially given the varying mask patterns across micro-batches. Thus, we propose an efficient greedy algorithm as shown below that provides a suboptimal yet effective solution within {math}`O(n\log n)` complexity.

```{figure} ../../../assets/magi_attn/comp/min_hp_alg.png
:align: center
:width: 1000px
:alt: Greedy Load-Balance Dispatch Algorithm

Greedy Load-Balance Dispatch Algorithm via Min-Heap
```

### Zero-Redundant Comm

The existing ring-style implementation uses point-to-point send/recv communication primitives, which cannot provide sufficient communication granularity, resulting in redundant communication. Take causal mask as an example, we analyze the redundant communication by recording the distribution of remote key-value ({math}`\mathrm{KV}`) requests and their gradients ({math}`\mathrm{dKV}`) under sparse attention masks. As shown in the following figure, {math}`\mathrm{KV}_0` is required by all queries and should be sent to all devices via Broad-Cast in the forward pass, with {math}`\mathrm{dKV}_0` reduced via All-Reduce in the backward pass. In contrast, {math}`\mathrm{KV}_7` is only needed by its host device but still circulates through all devices, and this redundancy intensifies in varlen scenarios.

```{figure} ../../../assets/magi_attn/comm/ring_p2p_redundancy.png
:align: center
:width: 1000px
:alt: Ring P2P Redundant Communication

Examples illustrating redundant communication in Ring P2P patterns for distributed attention given heterogeneous masks: (a) Even with a simple causal mask, Ring P2P incurs **25%** redundant communication; (b) For irregular mask patterns such as varlen block-causal mask with last global block, Ring P2P results in over **33%** redundancy.
```

To address this, as illustrated in the figure below, we introduce two communication primitives: {math}`\textit{Group-Cast}` and {math}`\textit{Group-Reduce}`, which model the communication patterns of low-demand {math}`\mathrm{KV}` and {math}`\mathrm{dKV}`. For example, in the causal mask, {math}`\mathrm{KV}_5` on {math}`\mathrm{rank}_2` is required only by {math}`\{\mathrm{Q}_6,\mathrm{Q}_7\}` and should be sent exclusively to the target ranks {math}`\{\mathrm{rank}_0, \mathrm{rank}_1\}` via Group-Cast, while the partial {math}`\mathrm{dKV}_5` is collected and reduced back to {math}`\mathrm{rank}_2` via Group-Reduce accordingly.

```{figure} ../../../assets/magi_attn/comm/group_gather_reduce_all2allv.png
:align: center
:width: 1000px
:alt: Group-Cast/Group-Reduce Primitives

Illustration of Group-Cast/Group-Reduce primitives for zero redundancy, using the varlen block-causal mask with the last global block as an example for irregular patterns. (a) In both forward and backward passes, the Group-Cast primitive internally analyzes and generates a transfer table for {math}`\mathrm{KV}` send/receive buffers, and launches the underlying All-to-All-v to complete communication with our custom {math}`\mathrm{Range Gather}` kernel for pre-/post-processing. (b) In the backward pass, Group-Reduce similarly handles the partial {math}`\mathrm{dKV}` communication for reduction, using All-to-All-v with the {math}`\mathrm{Range Gather}` kernel for pre-processing and the {math}`\mathrm{Range Scatter\!-\!Reduce}` kernel for post-processing.
```

As no existing communication kernels support these primitives, we prototype them using All-to-All-v, achieving zero-redundant communication in both forward and backward passes. However, this approach introduces extra pre-/post-processing overhead, similar to (un)permutation in expert parallelism (EP) {cite}`gale2022megablocks`. While kernel fusion mitigates the overhead, a dedicated implementation of Group-Cast and Group-Reduce remains a key direction for future work.


### Multi-Stage Overlap

Leveraging previous optimizations, we achieve high-performance computation through an efficient kernel and balanced workload dispatch, while minimizing communication overhead with our new primitives. To drive true linear scalability, we further improve end-to-end performance by introducing a multi-stage compute-communication overlap strategy, that effectively hides communication latency and adaptively optimizes overlap through manual or automatic tuning.

Similar to prior works {cite}`liu2023ringattentionblockwisetransformers,zhao2023pytorch,async_tensor_parallelism_in_pytorch`, we schedule pipeline stages to overlap computation with communication for both forward and backward passes, as shown in the following figureFig. Each {math}`\mathrm{rank}_i` first partitions its remote {math}`\mathrm{KV}`/{math}`\mathrm{dKV}` communication into stages.

```{figure} ../../../assets/magi_attn/mso/multi_stage_overlap_fwd_bwd.png
:align: center
:width: 1000px
:alt: Multi-Stage Overlap Scheduling

Schematic of Magi Attention's multi-stage overlap scheduling. (a) Forward pass: 4-stage scheduling overlaps computation (partial attention outputs and {math}`\textit{lse}` factors) with prefetching of next-stage {math}`\mathrm{KV}` requests (where applicable), hiding all communication overhead with the final stage's computation exposed. (b) Backward pass: 3-stage scheduling overlaps computation (partial {math}`\mathrm{dQ}`, {math}`\mathrm{dKV}`) with prefetching of next-stage {math}`\mathrm{KV}` requests and reduction of prior {math}`\mathrm{dKV}` requests, hiding all communication overhead except the {math}`\mathrm{dKV}` reduction of the final stage.
```

In the forward pass, the scheduler first launches the Group-Cast kernel to prefetch the next remote {math}`\mathrm{KV}`, then asynchronously executes the FFA kernel for partial attention computation, hiding all communication behind computation. To prevent all SMs from being occupied by the attention kernel, by default, we ensure the communication kernel picked first by setting `CUDA_DEVICE_MAX_CONNECTIONS=1` {cite}`cuda_device_max_connections_issue`. However, we also support relax this constraint by setting an non-zero `sm_margin` argument for the FFA kernel, to preserve some SMs for communication kernels to be launched.


In the backward pass, besides prefetching the next {math}`\mathrm{KV}`, the Group-Reduce kernel reduces the last {math}`\mathrm{dKV}` in a separate CUDA stream before launching the FFA kernel for the current stage, ensuring communication is overlapped across all stages except the final {math}`\mathrm{dKV}` reduction. Due to PyTorch's one-to-one mapping for process groups and collective communication streams including All-to-All-v {cite}`collectives_nccl_stream_issue`, we internally use an additional CP group for Group-Reduce to enable full overlap between communication kernels in the backward pass.

To adaptively control overlap granularity, we further introduce a tunable hyperparameter, {math}`\texttt{num_stages}`, accounting for varying compute-to-communication ratios across training setups, microbatches, or between forward and backward passes. This parameter can be manually configured or automatically determined by our {math}`\textit{overlap solver}`, with a simple dynamic search algorithm as shown below.

```{figure} ../../../assets/magi_attn/mso/dynamic_mso_alg.png
:align: center
:width: 800px
:alt: Dynamic Overlap Stage Search Algorithm

Dynamic Overlap Stage Search Algorithm
```


## Experiment

### Kernel-Level

To demonstrate FFA kernels' state-of-the-art performance and flexibility in handling ultra-long, heterogeneous mask training, we measure the throughput (in {math}`\texttt{TFLOPs/s}`) on Hopper GPUs for both forward and backward passes of prevalent attention kernels across standard and irregular mask patterns.

| settings              | value                                                                          |
|-----------------------|--------------------------------------------------------------------------------|
| batch size (b)        | 1                                                                              |
| number of heads (nh)  | nhq:nhk:nhv = 64:8:8 (GQA)                                                     |
| head dimension (hd)   | 128                                                                            |
| dtype                 | torch.bfloat16                                                                 |
| window size           | 1024 (for sliding window masks only)                                           |

Benchmark settings: for each mask pattern, we vary the sequence length `seqlen` from `4k,8k,16k,...,` up to `128k` (where `seqlen_q==seqlen_k==seqlen`) while measuring the throughput (in {math}`\texttt{TFLOPs/s}`) for forward and backward passes of different attention kernels. Other configurations are fixed using common training settings (see the table above) to focus on the impact of sequence length and mask pattern. For the varlen packed data, we simply follow the variable sequence length distribution in the open-sourced dataset {cite}`xu2024chatqa` illustrated in the following figure, from which we sample to pack and pad to the required `seqlen`.

```{figure} ../../../assets/magi_attn/exp/varlen_seqlen_distribution.png
:align: center
:width: 800px
:alt: Variable-Length Sequence Distribution

Distribution of sequence lengths in the dataset {cite}`xu2024chatqa`, used to sample and construct the variable-length data for both kernel-level and module-level experiments of MagiAttention.
```

Results are reported in the following figures.

```{figure} ../../../assets/magi_attn/exp/kernel/attn_with_full_mask/perf_report_all.png
:align: center
:width: 1000px
:alt: FFA Performance - Full Mask

Benchmarking FFA's performance and flexibility against other leading attention kernels for full mask scenarios.
```

```{figure} ../../../assets/magi_attn/exp/kernel/attn_with_causal_mask/perf_report_all.png
:align: center
:width: 1000px
:alt: FFA Performance - Causal Mask

Benchmarking FFA's performance and flexibility against other leading attention kernels for causal mask scenarios.
```

```{figure} ../../../assets/magi_attn/exp/kernel/attn_with_varlen_full_mask/perf_report_all.png
:align: center
:width: 1000px
:alt: FFA Performance - Varlen Full Mask

Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen full mask scenarios. (Note that: the {math}`\mathbf{E}` symbol indicates the corresponding distributed attention implementation raises *Cuda Out of Memory* error in that specific configuration.)
```

```{figure} ../../../assets/magi_attn/exp/kernel/attn_with_varlen_causal_mask/perf_report_all.png
:align: center
:width: 1000px
:alt: FFA Performance - Varlen Causal Mask

Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen causal mask scenarios. (Note that: the {math}`\mathbf{E}` symbol indicates the corresponding distributed attention implementation raises *Cuda Out of Memory* error in that specific configuration.)
```

```{figure} ../../../assets/magi_attn/exp/kernel/attn_with_sw_causal_mask/perf_report_all.png
:align: center
:width: 1000px
:alt: FFA Performance - Sliding-Window Causal Mask

Benchmarking FFA's performance and flexibility against other leading attention kernels for sliding-window causal mask scenarios. (Note that: the {math}`\mathbf{E}` symbol indicates the corresponding distributed attention implementation raises *Cuda Out of Memory* error in that specific configuration.)
```

```{figure} ../../../assets/magi_attn/exp/kernel/attn_with_varlen_block_causal_mask/perf_report_all.png
:align: center
:width: 1000px
:alt: FFA Performance - Varlen Block Causal Mask

Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen block causal mask scenarios. (Note that: the {math}`\mathbf{E}` symbol indicates the corresponding distributed attention implementation raises *Cuda Out of Memory* error in that specific configuration, while the {math}`\mathbf{X}` symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.)
```


### Module-Level

To validate the scalability of MagiAttention, we assess the throughput (in {math}`\texttt{TFLOPs/s}`) of the attention module propagation as the sequence length and parallel size increases for both forward and backward passes across various mask patterns, and compare it with several state-of-the-art CP strategies.

To validate the scalability of MagiAttention, we assess the per-GPU throughput (in {math}`\texttt{TFLOPs/s/GPU}`) of the attention module during both forward and backward propagation, as the sequence length and parallel size increase. This assessment is compared against common CP strategies including Ring-Attention {cite}`liu2023ringattentionblockwisetransformers` and Ulysses {cite}`jacobs2023deepspeed`. Due to the complexity of supporting irregular masks for baselines, our experiments are limited to the full mask and varlen full mask scenarios. And the distribution of variable sequence lengths still follow the one in [Kernel-level Experiments](#kernel-level).

The experiments are conducted on a large-scale productive GPU cluster<d-footnote>Due to business and confidentiality reasons, specific details about the productive cluster, such as the number and type of GPUs, are withheld.</d-footnote>. We scale the total sequence length `seqlen`, the context-parallel size `cp_size`, and the node size `nnodes` together from (`seqlen=64k,cp_size=1,nnodes=1`), `seqlen=128k,cp_size=2,nnodes=2`, ..., to `seqlen=3072k(3M),cp_size=48,nnodes=48`.

The tensor-parallel size `tp_size` is fixed at {math}`8`, with sequence parallelism (SP) enabled. Other data and model configurations for different mask types are the same as in the table in [Kernel-Level Experiments](#kernel-level).

Therefore, in every training setting, each rank is assigned constantly with `seqlen=64k`, `num_heads_q=8` and `num_heads_k=1` for attention propagation, while the remaining activations stays `seqlen=8k`, `num_heads_q=64` and `num_heads_k=8` with SP enabled. This setup simulates a common training configuration.

The results are presented in the following figures.

:::{subfigure} AB
:layout-sm: A|B
:gap: 8px
:subcaptions: below
:name: fig:magi_attn_tflops_per_gpu_full_mask
:align: center

![](../../../assets/magi_attn/exp/module/full_mask_fwd_per_gpu/flops_report.png)
:alt: (a) Forward Pass

![](../../../assets/magi_attn/exp/module/full_mask_bwd_per_gpu/flops_report.png)
:alt: (b) Backward Pass

Benchmarking MagiAttention's scalability against other leading CP strategies for full mask scenarios: (a) forward pass, (b) backward pass. (Note that: the {math}`\mathbf{X}` symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.)
:::

:::{subfigure} AB
:layout-sm: A|B
:gap: 8px
:subcaptions: below
:name: fig:magi_attn_tflops_per_gpu_varlen_full_mask
:align: center

![](../../../assets/magi_attn/exp/module/varlen_full_mask_fwd_per_gpu/flops_report.png)
:alt: (a) Forward Pass

![](../../../assets/magi_attn/exp/module/varlen_full_mask_bwd_per_gpu/flops_report.png)
:alt: (b) Backward Pass

Benchmarking MagiAttention's scalability against other leading CP strategies for varlen full mask scenarios: (a) forward pass, (b) backward pass. (Note that: the {math}`\mathbf{X}` symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.)
:::


## Discussion

### Attention Sink

Please check this [blog post](https://sandai-org.github.io/MagiAttention/blog/ffa_with_sink) about how to integrate Flex-Flash-Attention, MagiAttention, as well as Flash-Attention, with the learnable attention sink mechanism.


## Future Work

Please check our [Roadmap](https://github.com/SandAI-org/MagiAttention?tab=readme-ov-file#roadmap-%EF%B8%8F).


## FAQ

Q: How to start with MagiAttention ?

A: Please check our [docs](https://sandai-org.github.io/MagiAttention/docs/) for `QuickStart` section.

Q: Does MagiAttention support the GPU architectures other than Hopper ?

A: Currently, MagiAttention only supports the Hopper architecture. But we plan to support more GPU architectures in the future.


## Acknowledgement

We are grateful to the contributors listed below for their valuable contributions during the early stages of MagiAttention.

| Member        | Affiliations                | Email                           | GitHub Account |
| :------------ | :-------------------------- | :------------------------------ | :------------- |
| Zewei Tao     | SandAI                      | <zeweitao@sand.ai>              | littsk         |
| Yunpeng Huang | SandAI                      | <yunpenghuang@sand.ai>          | Strivin0311    |
| Qiangang Wang | SandAI, Nanjing University  | <522024330081@smail.nju.edu.cn> | WT1W           |
| Hanwen Sun    | Peking University           | <sunhanwen@stu.pku.edu.cn>      | hanwen-sun     |
| Jin Li        | SandAI, Tsinghua University | <2609835176@qq.com>             | lijinnn        |
| Tao Bu        | SandAI, Nanjing University  | <502024330002@smail.nju.edu.cn> | Big-TRex       |
| Bowen Zeng    | Zhejiang University         | <zbw.cs@zju.edu.cn>             | KevinZeng08    |
| WenYang Fang  | Nanjing University          | <fwy@smail.nju.edu.cn>          | kagami4243     |
| Siyuang Yan   | Nanjing University          | <siyuanyan@smail.nju.edu.cn>    | FibonaccciYan  |
| Zixu Jiang    | Nanjing University          | <522023330040@smail.nju.edu.cn> | 191220042      |
| Dingkun Xu    | Nanjing University          | <211220090@smail.nju.edu.cn>    | PureDimension  |
| Mingyu Liang  | Nanjing University          | <mingyuliang518@gmail.com>      | gaomusiki      |
| Jingwei Xu    | Nanjing University          | <jingweix@nju.edu.cn>           | paragonlight   |


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
