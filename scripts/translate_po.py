#!/usr/bin/env python3

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

"""Translate Sphinx .po files for MagiAttention from English to Chinese (zh_CN)."""

import os

import polib

BASE = "/Users/bytedance/Desktop/DEV/MagiAttentionInternal/Magi_Attention/docs/locale/zh_CN/LC_MESSAGES/blog"

# ============================================================
# magi_attn.po translations
# ============================================================
magi_attn_translations = {
    # Title
    "MagiAttention": "MagiAttention",
    "**A Distributed Attention Towards Linear Scalability for Ultra-Long "
    "Context, Heterogeneous Mask Training**": "**面向超长上下文、异构掩码训练的线性可扩展分布式注意力**",
    "Overview": "概览",
    "MagiAttention Overview": "MagiAttention 概览",
    "Overview of MagiAttention: (1) `FFA` - an optimized kernel based on "
    "`Flash-Attention 3`, further supports flexible mask patterns; (2) The "
    "`dispatch solver` shards ultra\u2011long data and dispatches for load-balanced"
    " computation; (3) `GroupCast` and `GroupReduce` primitives eliminate "
    "redundant communication; (4) The `overlap solver` adaptively partitons "
    "multi-stage computation/communication for optimal overlap; (5) Forward "
    "and backward timelines scheduled by MagiAttention. With all components "
    "together, MagiAttention enables linear scalability in training with "
    "ultra\u2011long contexts and heterogeneous masks.": "MagiAttention 概览：(1) `FFA` — 基于 `Flash-Attention 3` 的优化内核，进一步支持灵活的掩码模式；"
    "(2) `dispatch solver` 对超长数据进行分片并调度以实现负载均衡的计算；"
    "(3) `GroupCast` 和 `GroupReduce` 原语消除冗余通信；"
    "(4) `overlap solver` 自适应地划分多阶段计算/通信以实现最优重叠；"
    "(5) MagiAttention 调度的前向和反向时间线。所有组件协同工作，MagiAttention 实现了超长上下文和异构掩码训练中的线性可扩展性。",
    "Training large-scale video\u2011generation models faces two tightly coupled "
    "challenges: (1) ultra\u2011long contexts\u2014reaching millions of tokens (e.g., "
    "**~4M**)\u2014which make attention prohibitively expensive in compute and "
    "memory, and (2) highly heterogeneous, irregular attention masks (e.g., "
    "block\u2011causal + Patch\u2011and\u2011Pack) that break assumptions of existing kernels"
    " and distributed layouts, leading to fragmentation, load imbalance, "
    "wasted padding, and large communication overhead.": "训练大规模视频生成模型面临两个紧密耦合的挑战："
    "(1) 超长上下文——可达数百万 token（例如 **~4M**）——使得注意力在计算和内存上代价极高，"
    "(2) 高度异构的不规则注意力掩码（例如 block\u2011causal + Patch\u2011and\u2011Pack）"
    "打破了现有内核和分布式布局的假设，导致碎片化、负载不均衡、填充浪费和大量通信开销。",
    "These same constraints also affect (multimodal) LLMs that aim to support "
    "ultra\u2011long histories and flexible masking for agentic tasks with large "
    "retrievals and deep reasoning. <u>Therefore, we require an efficient, "
    "mask-flexible, and scalable distributed attention solution</u>.": "这些限制同样影响旨在支持超长历史记录和灵活掩码的（多模态）LLM，"
    "以适应涉及大规模检索和深度推理的智能体任务。<u>因此，我们需要一种高效、掩码灵活且可扩展的分布式注意力解决方案</u>。",
    "To address these challenges, we propose "
    "[MagiAttention](https://github.com/SandAI-org/MagiAttention), which "
    "targets these bottlenecks with **kernel-level flexibility**, while "
    "achieving **distributed-level linear scalability** across a broad range "
    "of training scenarios, particularly for those involving ultra-long "
    "contexts and heterogeneous masks like [Magi-1](https://github.com/SandAI-"
    "org/MAGI-1).": "为了应对这些挑战，我们提出了 "
    "[MagiAttention](https://github.com/SandAI-org/MagiAttention)，"
    "它以**内核级灵活性**瞄准这些瓶颈，同时在广泛的训练场景中实现**分布式级线性可扩展性**，"
    "尤其是涉及超长上下文和异构掩码的场景，如 [Magi-1](https://github.com/SandAI-org/MAGI-1)。",
    "Introduction": "引言",
    "Training large-scale autoregressive diffusion models for video generation"
    " (e.g., [Magi-1](https://github.com/SandAI-org/MAGI-1)) creates two "
    "tightly coupled system challenges. First, training contexts can reach "
    "millions of tokens, so naive quadratic attention or inadequately sharded "
    "algorithms quickly become infeasible in both compute and memory. Second, "
    "practical data pipelines\u2014for example, block\u2011causal attention combined "
    "with Patch\u2011and\u2011Pack (PnP) processing {cite}`dehghani2023patchnpacknavit` "
    "\u2014 produce highly heterogeneous, irregular masks and variable sequence "
    "lengths that violate assumptions made by standard attention kernels and "
    "distributed layouts. The combined effect is severe fragmentation, "
    "imbalanced compute across ranks, excessive padding, and large, often "
    "redundant, communication volumes.": "训练大规模自回归扩散视频生成模型（例如 [Magi-1](https://github.com/SandAI-org/MAGI-1)）"
    "带来两个紧密耦合的系统性挑战。首先，训练上下文可达数百万 token，"
    "因此朴素的二次注意力或分片不足的算法在计算和内存上都会迅速变得不可行。"
    "其次，实际的数据流水线——例如 block\u2011causal 注意力与 Patch\u2011and\u2011Pack (PnP) 处理 "
    "{cite}`dehghani2023patchnpacknavit` 的组合——会产生高度异构的不规则掩码和可变序列长度，"
    "违反了标准注意力内核和分布式布局所做的假设。其综合效果是严重的碎片化、"
    "各 rank 间计算不均衡、过度填充以及大量（往往是冗余的）通信量。",
    "Prior context\u2011parallel solutions "
    "{cite}`jacobs2023deepspeed,liu2023ringattentionblockwisetransformers,fang2024uspunifiedsequenceparallelism,gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual`"
    " partially mitigate these issues but introduce new limitations: "
    "head\u2011sharded designs impose divisibility constraints and reduce "
    "flexibility, ring\u2011style P2P schemes scale but incur large communication "
    "and redundancy under sparse/varlen masks. While recent efforts "
    "{cite}`wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm"
    ",megatron-lm-hybrid-cp-pr-2054` dynamically adjust CP sizes to avoid "
    "unnecessary sharding and redundant communication for shorter sequences, "
    "they still incur extra memory overhead for NCCL buffers and involve "
    "complex scheduling to balance loads and synchronize across different "
    "subsets of ranks.": "已有的上下文并行方案 "
    "{cite}`jacobs2023deepspeed,liu2023ringattentionblockwisetransformers,fang2024uspunifiedsequenceparallelism,gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual`"
    " 部分缓解了这些问题，但引入了新的局限性：head 分片设计施加了整除性约束并降低了灵活性，"
    "ring 式 P2P 方案虽然可扩展，但在稀疏/varlen 掩码下会产生大量通信和冗余。"
    "虽然最近的工作 "
    "{cite}`wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm"
    ",megatron-lm-hybrid-cp-pr-2054` 动态调整 CP 大小以避免对较短序列的不必要分片和冗余通信，"
    "但它们仍然会产生额外的 NCCL 缓冲区内存开销，并涉及复杂的调度来平衡负载和在不同 rank 子集之间同步。",
    "Crucially, existing methods do not simultaneously (1) provide a unified, "
    "distributable representation for a wide class of mask patterns, (2) "
    "guarantee balanced compute across context\u2011parallel (CP) ranks for "
    "arbitrarily structured masks, and (3) eliminate unnecessary data movement"
    " while enabling robust compute/communication overlap.": "关键在于，现有方法无法同时：(1) 为广泛的掩码模式提供统一的、可分布式化的表示，"
    "(2) 保证任意结构掩码在上下文并行 (CP) rank 间的计算均衡，"
    "(3) 消除不必要的数据传输，同时实现稳健的计算/通信重叠。",
    "MagiAttention addresses these gaps by prioritizing kernel\u2011level "
    "flexibility together with distributed-level scalability, which depends on"
    " meeting the following fundamental conditions:": "MagiAttention 通过优先考虑内核级灵活性和分布式级可扩展性来解决这些差距，"
    "这依赖于满足以下基本条件：",
    "<b>Linearly Scalable Attention Kernel</b>: The performance of the "
    "attention kernel should not degrade as CP size increases. To this end, we"
    " introduce [Flex-Flash-Attention](#flex-flash-attention), an extension of"
    " FlashAttention-3 (FA3), which natively considers the efficiency impact "
    "of attention mask partitioning in distributed environments. It supports "
    "distributable mask representations with a tailored kernel implementation "
    "to ensure scalability while accommodating a broader range of attention "
    "mask types.": "<b>线性可扩展注意力内核</b>：注意力内核的性能不应随 CP 大小增加而下降。"
    "为此，我们引入 [Flex-Flash-Attention](#flex-flash-attention)，这是 FlashAttention-3 (FA3) 的扩展，"
    "它在设计上原生考虑了分布式环境中注意力掩码分区对效率的影响。"
    "它通过定制的内核实现支持可分布式化的掩码表示，在保证可扩展性的同时适配更广泛的注意力掩码类型。",
    "<b>Balanced Computational Workloads</b>: Imbalances in the computational "
    "load across CP ranks lead to unavoidable idle bubbles that hinder "
    "scalability. MagiAttention is natively designed to ensure [Computation "
    "Load Balancing](#computation-load-balancing), mitigating such "
    "inefficiencies.": "<b>均衡的计算负载</b>：CP rank 间的计算负载不均衡会导致不可避免的空闲气泡，阻碍可扩展性。"
    "MagiAttention 在设计上原生确保[计算负载均衡](#computation-load-balancing)，缓解此类低效问题。",
    "<b>Full Overlap of Communication and Computation</b>: Without sufficient "
    "overlap, increasing CP size results in communication-induced idle time on"
    " GPUs, impairing scalability. MagiAttention introduces novel [Zero-"
    "Redundant Communication Primitives](#zero-redundant-communication-"
    "primitives) to minimize communication overhead, along with an [Adaptive "
    "Multi-Stage Overlap](#multi-stage-computation-communication-overlap) "
    "strategy that enables effective communication-computation overlap.": "<b>通信与计算的完全重叠</b>：如果重叠不充分，增加 CP 大小会导致 GPU 上由通信引起的空闲时间，"
    "损害可扩展性。MagiAttention 引入了新颖的[零冗余通信原语](#zero-redundant-communication-primitives)"
    "以最小化通信开销，并配合[自适应多阶段重叠](#multi-stage-computation-communication-overlap)"
    "策略来实现有效的通信-计算重叠。",
    "By coordinating a mask\u2011flexible kernel, a load\u2011balancing dispatcher, and "
    "zero\u2011redundancy communication with adaptive overlap, MagiAttention "
    "supports a broad spectrum of attention patterns while delivering "
    "distributed-level linear scalability across realistic ultra\u2011long and "
    "heterogeneous training workloads.": "通过协调掩码灵活内核、负载均衡调度器和自适应重叠的零冗余通信，"
    "MagiAttention 支持广泛的注意力模式，同时在真实的超长和异构训练负载上实现分布式级线性可扩展性。",
    "Below, we briefly review current CP strategies in [Related Work"
    "](#related-work), present the key designs in [Methodology](#methodology),"
    " and report comprehensive experimental results that validate the approach"
    " in [Experiments](#experiments).": "下面，我们在[相关工作](#related-work)中简要回顾当前的 CP 策略，"
    "在[方法论](#methodology)中介绍关键设计，并在[实验](#experiments)中报告验证该方法的全面实验结果。",
    "We further elaborate upon preliminaries, extended functionalities, "
    "optimization techniques, and next-generation design in "
    "[Miscellaneous](#miscellaneous), followed by the [Future Work](#future-"
    "work) section. Our evolving exploration seeks to broaden the scope and "
    "redefine the frontiers of distributed attention, optimizing its "
    "performance for large-scale model training and extending its efficacy to "
    "inference scenarios in the future.": "我们在[其他](#miscellaneous)部分进一步阐述预备知识、扩展功能、优化技术和下一代设计，"
    "随后是[未来工作](#future-work)部分。我们持续演进的探索旨在拓宽分布式注意力的范围并重新定义其前沿，"
    "优化其在大规模模型训练中的性能，并在未来将其扩展到推理场景。",
    "Related Work": "相关工作",
    "To handle ultra\u2011long contexts, context parallelism (CP) is essential, but"
    " existing CP strategies do not meet the real-world demanding settings.": "要处理超长上下文，上下文并行 (CP) 至关重要，但现有的 CP 策略无法满足真实世界的苛刻要求。",
    "DeepSpeed's `Ulysses` {cite}`jacobs2023deepspeed` uses head-sharded "
    "attention with All-to-All transforms; it is easy to integrate but "
    "requires the number of heads to be divisible by the CP size, limiting "
    "scalability (e.g., GQA and when combined with head-aware tensor "
    "parallelism) {cite}`shoeybi2020megatronlm,korthikanti2022reducing`.": "DeepSpeed 的 `Ulysses` {cite}`jacobs2023deepspeed` 使用 head 分片注意力和 All-to-All 变换；"
    "它易于集成，但要求 head 数量能被 CP 大小整除，限制了可扩展性"
    "（例如 GQA 以及与 head 感知张量并行结合时）{cite}`shoeybi2020megatronlm,korthikanti2022reducing`。",
    "`Ring-Attention` "
    "{cite}`li2021sequence,liu2023ringattentionblockwisetransformers,wang2024tokenringefficientparallelismframework`"
    " keeps sequence-sharded activations and relies on multi-stage ring-style "
    "P2P communication for online attention and overlap "
    "{cite}`rabe2021self,dao2022flashattention,wang2022overlap`. It scales "
    "better than head-sharding but incurs large communication volumes and "
    "inefficient P2P primitives as CP size grows. Hybrid 2D schemes like `USP`"
    " {cite}`fang2024uspunifiedsequenceparallelism` and `LoongTrain` "
    "{cite}`gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual`"
    " combine `Ulysses` and `Ring-Attention` to reduce their weaknesses but "
    "still lack the fundamental efficiency and scalability needed for "
    "ultra\u2011long contexts.": "`Ring-Attention` "
    "{cite}`li2021sequence,liu2023ringattentionblockwisetransformers,wang2024tokenringefficientparallelismframework`"
    " 保持序列分片的激活并依赖多阶段 ring 式 P2P 通信进行在线注意力和重叠 "
    "{cite}`rabe2021self,dao2022flashattention,wang2022overlap`。"
    "它比 head 分片方案扩展性更好，但随着 CP 大小增加会产生大量通信量和低效的 P2P 原语。"
    "混合 2D 方案如 `USP` {cite}`fang2024uspunifiedsequenceparallelism` 和 `LoongTrain` "
    "{cite}`gu2024loongtrainefficienttraininglongsequence,chen2024longvilascalinglongcontextvisual`"
    " 结合了 `Ulysses` 和 `Ring-Attention` 以减少各自的弱点，但仍缺乏超长上下文所需的根本效率和可扩展性。",
    "Irregular masks (e.g., varlen) worsen these issues (see "
    "{numref}`ring_attn_load_balance` below). Naive <em>sequential even "
    "sharding</em> creates uneven mask-area distribution and imbalanced "
    "compute across ranks. Custom <em>zigzag sharding</em> "
    "{cite}`ring_flash_attention_issue2` can rebalance specific varlen causal "
    "patterns but causes fragmentation, excessive padding, and kernel "
    "slowdowns, and it does not generalize to patterns such as the <em>varlen "
    "block-causal mask</em> used in autoregressive video generation for "
    "[Magi-1](https://github.com/SandAI-org/MAGI-1).": "不规则掩码（例如 varlen）会加剧这些问题（见下方 "
    "{numref}`ring_attn_load_balance`）。朴素的<em>顺序均匀分片</em>"
    "会造成掩码面积分布不均和各 rank 间计算不均衡。"
    "定制的 <em>zigzag 分片</em> {cite}`ring_flash_attention_issue2` "
    "可以重新平衡特定的 varlen causal 模式，但会导致碎片化、过度填充和内核减速，"
    "且无法泛化到诸如 [Magi-1](https://github.com/SandAI-org/MAGI-1) "
    "中用于自回归视频生成的 <em>varlen block-causal 掩码</em> 等模式。",
    "Ring-Attention Load Balancing": "Ring-Attention 负载均衡",
    "Illustration of `Ring-Attention`'s sharding strategies for load "
    "balancing: (a) full mask \u2014 sequential sharding across the global mask; "
    "(b) causal mask \u2014 tailored *zigzag sharding* "
    "{cite}`ring_flash_attention_issue2`; (c) varlen full mask \u2014 sequential "
    "sharding per packed sample; (d) varlen causal mask \u2014 per-sample *zigzag "
    "sharding*, which increases fragmentation and padding and degrades "
    "performance.": "`Ring-Attention` 负载均衡分片策略示意图："
    "(a) full 掩码 — 在全局掩码上顺序分片；"
    "(b) causal 掩码 — 定制的 *zigzag 分片* {cite}`ring_flash_attention_issue2`；"
    "(c) varlen full 掩码 — 按打包样本顺序分片；"
    "(d) varlen causal 掩码 — 按样本 *zigzag 分片*，这会增加碎片化和填充并降低性能。",
    "Second, communication overhead worsens under sparse varlen masks because "
    "entire sequence chunks are transferred to all CP ranks\u2014even when many "
    "ranks do not need them\u2014yielding over **30% redundant communication**, as "
    "shown in [Zero-Redundant Communication Primitives](#zero-redundant-"
    "communication-primitives). Third, these inefficiencies undermine pipeline"
    " compute\u2013communication overlap: imbalanced workloads and excessive "
    "communication make overlap fragile and constrain scalability.": "其次，通信开销在稀疏 varlen 掩码下会恶化，因为整个序列块会被传输到所有 CP rank——"
    "即使许多 rank 并不需要——导致超过 **30% 的冗余通信**，"
    "如[零冗余通信原语](#zero-redundant-communication-primitives)所示。"
    "第三，这些低效问题破坏了流水线计算-通信重叠：不均衡的工作负载和过量通信使重叠变得脆弱并限制了可扩展性。",
    "Recent efforts like `DCP` "
    "{cite}`wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm`"
    " and `Hybrid-CP` {cite}`megatron-lm-hybrid-cp-pr-2054` reduce redundant "
    "sharding by dynamically assigning CP group sizes per sample based on "
    "sequence length. However, they introduce significant scheduling "
    "complexity, frequent cross-group synchronization, and extra NCCL buffer "
    "memory, lacking of a bottom-up redesign required for robust, mask-"
    "flexible, and scalable distributed attention.": "最近的工作如 `DCP` "
    "{cite}`wang2024datacentricheterogeneityadaptivesequenceparallelism,zhang2024dcp,ge2025bytescaleefficientscalingllm`"
    " 和 `Hybrid-CP` {cite}`megatron-lm-hybrid-cp-pr-2054` "
    "通过根据序列长度动态分配每个样本的 CP 组大小来减少冗余分片。"
    "然而，它们引入了显著的调度复杂性、频繁的跨组同步和额外的 NCCL 缓冲区内存，"
    "缺乏稳健、掩码灵活且可扩展的分布式注意力所需的自底向上重新设计。",
    "Methodology": "方法论",
    "Flex-Flash-Attention": "Flex-Flash-Attention",
    "AttnSlice Representation": "AttnSlice 表示",
    "`Flash-Attention` "
    "{cite}`dao2022flashattention,dao2023flashattention,shah2024flashattention3fastaccurateattention,dao2025flashattention_cute`"
    " delivers high throughput, memory efficiency, and native support for "
    "varlen-packed inputs, making it a cornerstone for large-scale training. "
    "However, its kernels assume regular mask structure and do not handle "
    "irregular, rank-distributed masks efficiently\u2014causing fragmentation, load"
    " imbalance, excess padding, and higher communication\u2014so a mask\u2011flexible "
    "kernel that preserves Flash\u2011Attention's performance is required "
    "{cite}`pytorch_sdpa,dong2024flexattentionprogrammingmodel,wang2025flashmaskefficientrichmask`.": "`Flash-Attention` "
    "{cite}`dao2022flashattention,dao2023flashattention,shah2024flashattention3fastaccurateattention,dao2025flashattention_cute`"
    " 提供了高吞吐量、内存效率和对 varlen 打包输入的原生支持，使其成为大规模训练的基石。"
    "然而，其内核假设规则的掩码结构，无法高效处理不规则的、跨 rank 分布的掩码——"
    "导致碎片化、负载不均衡、过度填充和更高的通信开销——"
    "因此需要一个保持 Flash\u2011Attention 性能的掩码灵活内核 "
    "{cite}`pytorch_sdpa,dong2024flexattentionprogrammingmodel,wang2025flashmaskefficientrichmask`。",
    "Therefore, we introduce `Flex-Flash-Attention` (`FFA`), a kernel designed"
    " for distributed settings that flexibly handles diverse attention masks. "
    "`FFA` adopts a <b>distributable</b> representation that decomposes an "
    "irregular mask into multiple computational units called "
    "{math}`\\mathrm{AttnSlice}`. Each {math}`\\mathrm{AttnSlice}` is the "
    "triplet {math}`\\mathrm{(QRange, KRange, MaskType)}`, denoting a submask "
    "confined to a contiguous 2D query\u2013key region (see "
    "{numref}`attnslice_interpret` below).": "因此，我们引入 `Flex-Flash-Attention` (`FFA`)，一种为分布式环境设计的内核，"
    "可灵活处理多种注意力掩码。`FFA` 采用<b>可分布式化</b>的表示方式，"
    "将不规则掩码分解为多个称为 {math}`\\mathrm{AttnSlice}` 的计算单元。"
    "每个 {math}`\\mathrm{AttnSlice}` 是一个三元组 {math}`\\mathrm{(QRange, KRange, MaskType)}`，"
    "表示限定在连续 2D query-key 区域内的子掩码（见下方 {numref}`attnslice_interpret`）。",
    "AttnSlice Formulation": "AttnSlice 公式化表示",
    "Illustration of the {math}`\\mathrm{AttnSlice}` formulation for an "
    "irregular mask. The mask is decomposed into multiple "
    "{math}`\\mathrm{AttnSlice}` units, allowing fractal patterns to be re-"
    "expressed after redistribution across CP ranks to support distributed "
    "attention. Note that computation load balancing across CP ranks is not "
    "considered in this illustration.": "不规则掩码的 {math}`\\mathrm{AttnSlice}` 公式化示意图。"
    "掩码被分解为多个 {math}`\\mathrm{AttnSlice}` 单元，使得分形模式在跨 CP rank 重新分布后"
    "可以重新表达以支持分布式注意力。注意此示意图未考虑跨 CP rank 的计算负载均衡。",
    "As illustrated in {numref}`mask_with_attn_slice` below, this formulation "
    "expresses a wide range of attention masks\u2014including the varlen block-"
    "causal mask used in [Magi-1](https://github.com/SandAI-org/MAGI-1)\u2014as "
    "compositions of multiple triplets. These representations remain valid "
    "after sharding and rearrangement across ranks, making `FFA` well suited "
    "for distributed attention computation.": "如下方 {numref}`mask_with_attn_slice` 所示，该公式化表示能将广泛的注意力掩码——"
    "包括 [Magi-1](https://github.com/SandAI-org/MAGI-1) 中使用的 varlen block-causal 掩码——"
    "表达为多个三元组的组合。这些表示在跨 rank 分片和重排后仍然有效，"
    "使得 `FFA` 非常适合分布式注意力计算。",
    "AttnSlice Mask Patterns": "AttnSlice 掩码模式",
    "Examples of mask patterns expressed using {math}`\\mathrm{AttnSlice}`: "
    "(a)\u2013(d) are standard FA3-compatible patterns; (e)\u2013(h) are irregular masks"
    " beyond Flash-Attention's capability\u2014e.g., the varlen block-causal "
    "mask\u2014which `FFA` handles seamlessly while preserving FA3-comparable "
    "performance.": "使用 {math}`\\mathrm{AttnSlice}` 表达的掩码模式示例："
    "(a)–(d) 为标准的 FA3 兼容模式；(e)–(h) 为超出 Flash-Attention 能力范围的不规则掩码"
    "——例如 varlen block-causal 掩码——`FFA` 可以无缝处理并保持与 FA3 相当的性能。",
    "AttnSlice-level Parallelism in FFA": "FFA 中的 AttnSlice 级并行",
    "Built on `Flash-Attention 3` (`FA3`) kernels "
    "{cite}`shah2024flashattention3fastaccurateattention`, `FFA` leverages "
    "Hopper GPUs' TMA feature {cite}`nvidia2024accelerating` and implements "
    "{math}`\\mathrm{AttnSlice}`-level parallelism with atomic operations for "
    "correctness (illustrated in {numref}`ffa_slice_atomic_reduce` below). "
    "`FFA` delivers MFU comparable to FA3 while supporting the flexible "
    "{math}`\\mathrm{AttnSlice}` formulation\u2014see [Attention Kernel "
    "Benchmark](./cp_benchmark.md#kernel-level) for detailed performance and "
    "flexibility comparisons.": "基于 `Flash-Attention 3` (`FA3`) 内核 "
    "{cite}`shah2024flashattention3fastaccurateattention` 构建，"
    "`FFA` 利用 Hopper GPU 的 TMA 特性 {cite}`nvidia2024accelerating`，"
    "实现了 {math}`\\mathrm{AttnSlice}` 级并行和原子操作以保证正确性"
    "（如下方 {numref}`ffa_slice_atomic_reduce` 所示）。"
    "`FFA` 在支持灵活的 {math}`\\mathrm{AttnSlice}` 公式化的同时，"
    "提供了与 FA3 相当的 MFU——详见[注意力内核基准测试](./cp_benchmark.md#kernel-level)"
    "中的性能和灵活性对比。",
    "FFA Slice Atomic Reduction": "FFA Slice 原子归约",
    "Illustration of the `FFA` forward and backward kernels: data loading, on-"
    "chip computation, and atomic reduction for slice-level parallelism.": "`FFA` 前向和反向内核示意图：数据加载、片上计算和用于 slice 级并行的原子归约。",
    "Basic Mask Types in AttnSlice": "AttnSlice 中的基本掩码类型",
    "Although most mask patterns can be expressed with "
    "{math}`\\mathrm{AttnSlice}` using the common types "
    "{math}`\\lbrace\\texttt{FULL}, \\texttt{CAUSAL}\\rbrace`, some "
    "patterns\u2014e.g., {math}`\\textit{sliding-window}`\u2014become inefficient "
    "because they require expressing each row individually. To represent such "
    "patterns compactly, we introduce two additional mask types, "
    "{math}`\\lbrace\\texttt{INV-CAUSAL}, \\texttt{BI-CAUSAL}\\rbrace`. The "
    "following {numref}`attn_slice_mask_type_sq=sk`, "
    "{numref}`attn_slice_mask_type_sq<sk`, and "
    "{numref}`attn_slice_mask_type_sq>sk` illustrate examples of the current "
    "{math}`4` supported mask types.": "虽然大多数掩码模式可以使用常见类型 "
    "{math}`\\lbrace\\texttt{FULL}, \\texttt{CAUSAL}\\rbrace` 的 "
    "{math}`\\mathrm{AttnSlice}` 来表达，但某些模式——例如 "
    "{math}`\\textit{sliding-window}`——由于需要逐行表达而变得低效。"
    "为了紧凑地表示此类模式，我们引入了两种额外的掩码类型："
    "{math}`\\lbrace\\texttt{INV-CAUSAL}, \\texttt{BI-CAUSAL}\\rbrace`。"
    "以下 {numref}`attn_slice_mask_type_sq=sk`、"
    "{numref}`attn_slice_mask_type_sq<sk` 和 "
    "{numref}`attn_slice_mask_type_sq>sk` 展示了当前 "
    "{math}`4` 种支持的掩码类型示例。",
    "AttnSlice Mask Types (seqlen_q = seqlen_k)": "AttnSlice 掩码类型 (seqlen_q = seqlen_k)",
    "Illustrates the four supported mask types for `seqlen_q == seqlen_k`. "
    "Note: in this setting, {math}`\\texttt{BI-CAUSAL}` reduces to a mask "
    "where only the principal diagonal cells are valid.": "展示 `seqlen_q == seqlen_k` 下四种支持的掩码类型。"
    "注意：在此设置下，{math}`\\texttt{BI-CAUSAL}` 退化为仅主对角线单元有效的掩码。",
    "AttnSlice Mask Types (seqlen_q < seqlen_k)": "AttnSlice 掩码类型 (seqlen_q < seqlen_k)",
    "Illustration of the four supported mask types when `seqlen_q < seqlen_k`."
    " This configuration commonly occurs when employing {math}`\\texttt{INV-"
    "CAUSAL}` and {math}`\\texttt{BI-CAUSAL}` masks.": "`seqlen_q < seqlen_k` 时四种支持的掩码类型示意图。"
    "此配置通常出现在使用 {math}`\\texttt{INV-CAUSAL}` 和 {math}`\\texttt{BI-CAUSAL}` 掩码时。",
    "AttnSlice Mask Types (seqlen_q > seqlen_k)": "AttnSlice 掩码类型 (seqlen_q > seqlen_k)",
    "Illustration of the four supported mask types for `seqlen_q > seqlen_k`. "
    "Note that {math}`\\texttt{BI-CAUSAL}` is empty and contains no valid "
    "cells.": "`seqlen_q > seqlen_k` 时四种支持的掩码类型示意图。"
    "注意 {math}`\\texttt{BI-CAUSAL}` 为空且不包含有效单元。",
    "Using the four supported mask types, we illustrate common {math}`\\textit"
    "{sliding-window}`-style masks expressed via the "
    "{math}`\\mathrm{AttnSlice}` formulation (see {numref}`sw_mask_with_slice`"
    " below).": "利用四种支持的掩码类型，我们展示了通过 {math}`\\mathrm{AttnSlice}` 公式化表达的常见 "
    "{math}`\\textit{sliding-window}` 风格掩码（见下方 {numref}`sw_mask_with_slice`）。",
    "Sliding-Window Mask Patterns": "Sliding-Window 掩码模式",
    "Examples of common {math}`\\textit{sliding-window}`-style mask patterns "
    "formulated by {math}`\\mathrm{AttnSlice}`.": "由 {math}`\\mathrm{AttnSlice}` 公式化的常见 {math}`\\textit{sliding-window}` 风格掩码模式示例。",
    "Computation Load-Balancing": "计算负载均衡",
    "Dispatch Solver": "Dispatch Solver",
    "In context-parallel training, heterogeneous attention masks across CP "
    "ranks create imbalanced computational workloads. `Ring-Attention` (see "
    "[Related Work](#related-work)) uses a partitioning strategy tailored to "
    "causal masks and therefore does not generalize to arbitrary patterns. To "
    "address this, we propose a generic, efficient `dispatch solver` that "
    "balances workload across CP ranks for diverse attention types.": "在上下文并行训练中，跨 CP rank 的异构注意力掩码会造成计算负载不均衡。"
    "`Ring-Attention`（见[相关工作](#related-work)）使用针对 causal 掩码定制的分区策略，"
    "因此无法泛化到任意模式。为此，我们提出了通用且高效的 `dispatch solver`，"
    "可在不同注意力类型下平衡各 CP rank 的工作负载。",
    "Concretely, we adopt a chunk-wise permutable sharding: partition the "
    "global mask evenly along the query dimension into chunks, each associated"
    " with a submask area {math}`\\lbrace(C_i, "
    "\\mathrm{Area}(C_i))\\rbrace_{i=1}^n`, where {math}`C_i` denotes the i-th"
    " chunk, {math}`\\mathrm{Area}(C_i)` is its mask area, {math}`n = "
    "\\frac{seqlen}{\\textit{chunk\\_size}}`, and "
    "{math}`\\textit{chunk\\_size}` is a tunable granularity parameter.": "具体而言，我们采用块级可排列分片：沿 query 维度将全局掩码均匀划分为多个块，"
    "每个块关联一个子掩码面积 {math}`\\lbrace(C_i, \\mathrm{Area}(C_i))\\rbrace_{i=1}^n`，"
    "其中 {math}`C_i` 表示第 i 个块，{math}`\\mathrm{Area}(C_i)` 是其掩码面积，"
    "{math}`n = \\frac{seqlen}{\\textit{chunk\\_size}}`，"
    "而 {math}`\\textit{chunk\\_size}` 是可调的粒度参数。",
    "These chunks are assigned equally to {math}`\\textit{cp\\_size}` buckets "
    "so every bucket contains the same number of chunks (preserving token-"
    "level balance for non-attention stages). Each bucket's total mask "
    "workload is the summed submask area, written as {math}`\\lbrace(B_j, "
    "\\mathrm{SumArea}(B_j))\\rbrace_{j=1}^{\\textit{cp\\_size}}`.": "这些块被均匀分配到 {math}`\\textit{cp\\_size}` 个桶中，"
    "使每个桶包含相同数量的块（为非注意力阶段保持 token 级均衡）。"
    "每个桶的总掩码工作量是其子掩码面积之和，"
    "记为 {math}`\\lbrace(B_j, \\mathrm{SumArea}(B_j))\\rbrace_{j=1}^{\\textit{cp\\_size}}`。",
    "Under this formulation, load balancing reduces to a combinatorial "
    "assignment problem: find an optimal mapping {math}`f^*: \\lbrace "
    "C_i\\rbrace_{i=1}^n \\rightarrow \\lbrace "
    "B_j\\rbrace_{j=1}^{\\textit{cp\\_size}}` that minimizes the maximum per-"
    "bucket area, as shown in the Eq {eq}`eq:comp_load_balance` below.": "在此公式化下，负载均衡归结为一个组合分配问题："
    "找到最优映射 {math}`f^*: \\lbrace C_i\\rbrace_{i=1}^n \\rightarrow \\lbrace "
    "B_j\\rbrace_{j=1}^{\\textit{cp\\_size}}`，使得每桶最大面积最小化，"
    "如下方等式 {eq}`eq:comp_load_balance` 所示。",
    "Since this problem is NP-hard and mask patterns change across micro-"
    "batches, solving it exactly per iteration is impractical. We therefore "
    "use a practical greedy Min-Heap algorithm (illustrated in "
    "{numref}`min_hp_alg` below) that runs in {math}`O(n\\log n)` and yields a"
    " fast, effective assignment with minimal runtime overhead.": "由于该问题是 NP 难的且掩码模式在微批次间变化，每次迭代精确求解不可行。"
    "因此我们使用一种实用的贪心最小堆算法（如下方 {numref}`min_hp_alg` 所示），"
    "运行时间为 {math}`O(n\\log n)`，以最小的运行时开销产生快速有效的分配。",
    "Greedy Load-Balance Dispatch Algorithm": "贪心负载均衡调度算法",
    "Greedy Load-Balance Dispatch Algorithm via Min-Heap": "基于最小堆的贪心负载均衡调度算法",
    "Static Attn Solver": "静态注意力求解器",
    "Upon dispatching tensors along the seqlen dimension into {math}`n` "
    "chunks, the global mask is partitioned into {math}`n^2` submasks and each"
    " CP rank is assigned with {math}`n` submasks. Since each rank can process"
    " only one \u201chost\u201d submask along the principal diagonal of the global mask "
    "using local tensors, the remaining {math}`n\\!-\\!1` \u201cremote\u201d submasks "
    "require communication. This yields two essential but non-trivial meta "
    "structures:": "在沿 seqlen 维度将张量调度为 {math}`n` 个块后，"
    "全局掩码被划分为 {math}`n^2` 个子掩码，每个 CP rank 分配 {math}`n` 个子掩码。"
    "由于每个 rank 只能使用本地张量处理全局掩码主对角线上的一个\u201c宿主\u201d子掩码，"
    "剩余的 {math}`n\\!-\\!1` 个\u201c远程\u201d子掩码需要通信。"
    "这产生了两个重要但非平凡的元结构：",
    "(1) **`CalcMeta`**: Encodes each submask as {math}`\\mathrm{AttnSlice}` "
    "instances per rank (and per stage if using [multi-stage overlap](#multi-"
    "stage-computation-communication-overlap)) and supplies the arguments "
    "required by the `FFA` kernels for calculation.": "(1) **`CalcMeta`**：将每个子掩码编码为每个 rank（以及使用[多阶段重叠](#multi-stage-computation-communication-overlap)时每个阶段）"
    "的 {math}`\\mathrm{AttnSlice}` 实例，并提供 `FFA` 内核计算所需的参数。",
    "(2) **`CommMeta`**: Describes the data exchanges with other CP peers\u2014what"
    " input tensors to fetch for `FFA` and how to reduce partial outputs per "
    "rank (and per stage if using [multi-stage overlap](#multi-stage-"
    "computation-communication-overlap))\u2014producing the arguments for "
    "`GroupCast/GroupReduce` kernels for communication (see [group collective "
    "primitives](#zero-redundant-communication-primitives) for details).": "(2) **`CommMeta`**：描述与其他 CP 对等节点的数据交换——"
    "为 `FFA` 获取哪些输入张量以及如何归约每个 rank"
    "（以及使用[多阶段重叠](#multi-stage-computation-communication-overlap)时每个阶段）"
    "的部分输出——生成 `GroupCast/GroupReduce` 内核通信所需的参数"
    "（详见 [group collective 原语](#zero-redundant-communication-primitives)）。",
    "To produce these, we design the `attn solver` data structure: it consumes"
    " the `dispatch solver` output and emits the `CalcMeta` and `CommMeta` "
    "needed to run distributed attention (forward and backward), i.e., the "
    "argument bundles for `FFA` and `GroupCast/GroupReduce` on each CP rank "
    "and stage. And we initially provide the `static attn solver` "
    "implementation that builds `CalcMeta` and `CommMeta` during the data "
    "preprocessing stage from the `dispatch solver` results, then invokes the "
    "`overlap solver` to derive multi\u2011stage schedules.": "为此，我们设计了 `attn solver` 数据结构：它消费 `dispatch solver` 的输出，"
    "并生成运行分布式注意力（前向和反向）所需的 `CalcMeta` 和 `CommMeta`，"
    "即每个 CP rank 和阶段上 `FFA` 和 `GroupCast/GroupReduce` 的参数束。"
    "我们首先提供了 `static attn solver` 实现，它在数据预处理阶段从 `dispatch solver` 结果中"
    "构建 `CalcMeta` 和 `CommMeta`，然后调用 `overlap solver` 来推导多阶段调度。",
    "However, This `static attn solver` is based on the strong assumption that"
    " the **global mask is static**, i.e. (1) known at the data-processing "
    "stage for each micro-batch and (2) remains unchanged across the whole "
    "forward/backward passes at all attention layers. It also restricts to the"
    " [kv-comm only scheduling](#scheduling-with-kv-comm-only), that only "
    "{math}`\\mathrm{KV}`-related tensors are allowed to be communicated while"
    " {math}`\\mathrm{QO}`-related tensors stay local\u2014limiting scheduling "
    "flexibility and overlap potential.": "然而，此 `static attn solver` 基于一个较强的假设：**全局掩码是静态的**，即 "
    "(1) 在每个微批次的数据处理阶段已知，"
    "(2) 在所有注意力层的整个前向/反向传播过程中保持不变。"
    "它还限制在 [kv-comm only 调度](#scheduling-with-kv-comm-only) 下，"
    "即只允许通信 {math}`\\mathrm{KV}` 相关张量，而 {math}`\\mathrm{QO}` 相关张量保持本地"
    "——限制了调度灵活性和重叠潜力。",
    "Dynamic Attn Solver": "动态注意力求解器",
    "The `static attn solver` handles most standard training cases but is "
    "limited and suboptimal for dynamic mask scenarios\u2014e.g., layer-varying "
    "hybrid attention {cite}`minimax2025minimax01scalingfoundationmodels` or "
    "dynamic sparse masks determined at runtime "
    "{cite}`yuan2025nativesparseattentionhardwarealigned,deepseekai2025deepseekv32pushingfrontieropen`.": "`static attn solver` 可以处理大多数标准训练场景，"
    "但对于动态掩码场景是有限的且次优的——例如逐层变化的混合注意力 "
    "{cite}`minimax2025minimax01scalingfoundationmodels` 或运行时确定的动态稀疏掩码 "
    "{cite}`yuan2025nativesparseattentionhardwarealigned,deepseekai2025deepseekv32pushingfrontieropen`。",
    "To address this, we are developing an experimental `dynamic attn solver` "
    "that dynamically balances computation (*w/o relying on initial dispatch "
    "results by `dispatch solver`*) and minimizes communication **under "
    "general [scheduling with qo-comm enabled](#scheduling-with-qo-comm-"
    "enabled)**, relaxing the heuristics of the current [kv-comm only "
    "scheduling](#scheduling-with-kv-comm-only). Then it will be able to "
    "generate `CalcMeta` and `CommMeta` **on\u2011the\u2011fly** with negligible "
    "overhead during each attention-layer forward pass.": "为了解决这个问题，我们正在开发实验性的 `dynamic attn solver`，"
    "它动态平衡计算（*无需依赖 `dispatch solver` 的初始调度结果*）并在"
    "**通用的 [scheduling with qo-comm enabled](#scheduling-with-qo-comm-enabled)** 下最小化通信，"
    "放宽当前 [kv-comm only 调度](#scheduling-with-kv-comm-only) 的启发式约束。"
    "然后它将能够在每个注意力层前向传播过程中以可忽略的开销**即时**生成 `CalcMeta` 和 `CommMeta`。",
    "See the seperate [blog post](./dynamic_solver.md) for more details about "
    "the motivation, design, implementation, and preliminary results of the "
    "`dynamic attn solver`.": "关于 `dynamic attn solver` 的动机、设计、实现和初步结果的更多详情，请参阅单独的[博客文章](./dynamic_solver.md)。",
    "Zero-Redundant Communication Primitives": "零冗余通信原语",
    "Ring P2P Redundancy Analysis": "Ring P2P 冗余分析",
    "Ring-style implementations rely on point-to-point (P2P) send/recv "
    "primitives that lack fine-grained communication control, causing "
    "unnecessary data movement. To quantify this, we record remote key-value "
    "({math}`\\mathrm{KV}`) requests and their gradients "
    "({math}`\\mathrm{dKV}`) under a causal mask as a simple example shown in "
    "{numref}`ring_p2p_redundancy`: in the forward pass {math}`\\mathrm{KV}_0`"
    " must be sent to all devices via `BroadCast`, while "
    "{math}`\\mathrm{dKV}_0` requires to be reduced via `AllReduce` during the"
    " backward. However, {math}`\\mathrm{KV}_7` is required ONLY locally for "
    "its host {math}`rank_7` yet still circulates across all devices. This "
    "redundant even dissemination\u2014and its cost\u2014becomes more severe for varlen "
    "mask patterns.": "Ring 式实现依赖于缺乏细粒度通信控制的点对点 (P2P) send/recv 原语，导致不必要的数据移动。"
    "为量化这一点，我们记录了 causal 掩码下远程 key-value ({math}`\\mathrm{KV}`) 请求"
    "及其梯度 ({math}`\\mathrm{dKV}`) 作为一个简单示例，"
    "如 {numref}`ring_p2p_redundancy` 所示：在前向传播中 {math}`\\mathrm{KV}_0` "
    "必须通过 `BroadCast` 发送到所有设备，而 {math}`\\mathrm{dKV}_0` 需要在反向传播中"
    "通过 `AllReduce` 归约。然而，{math}`\\mathrm{KV}_7` 仅在其宿主 {math}`rank_7` 本地需要，"
    "却仍在所有设备间循环传播。这种冗余的均匀分发——及其代价——在 varlen 掩码模式下更为严重。",
    "Ring P2P Redundant Communication": "Ring P2P 冗余通信",
    "Examples of redundant communication in Ring P2P with heterogeneous masks:"
    " (a) a simple causal mask incurs **25%** redundant communication; (b) "
    "irregular masks, e.g., the varlen block-causal mask with the last global "
    "block, can exceed **33%** redundancy.": "Ring P2P 中异构掩码的冗余通信示例："
    "(a) 简单的 causal 掩码产生 **25%** 的冗余通信；"
    "(b) 不规则掩码，例如带有最后全局块的 varlen block-causal 掩码，冗余可超过 **33%**。",
    "Group Collective Primitives": "Group Collective 原语",
    "To address this, as illustrated in the "
    "{numref}`group_gather_reduce_all2allv` below, we introduce two "
    "communication primitives: `GroupCast` and `GroupReduce`, which model the "
    "communication patterns of low-demand {math}`\\mathrm{KV}` and "
    "{math}`\\mathrm{dKV}`. For example, in the causal mask, "
    "{math}`\\mathrm{KV}_5` on {math}`\\mathrm{rank}_2` is required only by "
    "{math}`\\{\\mathrm{Q}_6,\\mathrm{Q}_7\\}` and should be sent exclusively "
    "to the target ranks {math}`\\{\\mathrm{rank}_0, \\mathrm{rank}_1\\}` via "
    "`GroupCast`, while the partial {math}`\\mathrm{dKV}_5` is collected and "
    "reduced back to {math}`\\mathrm{rank}_2` via `GroupReduce` accordingly.": "为解决这一问题，如下方 {numref}`group_gather_reduce_all2allv` 所示，"
    "我们引入了两种通信原语：`GroupCast` 和 `GroupReduce`，"
    "它们建模低需求 {math}`\\mathrm{KV}` 和 {math}`\\mathrm{dKV}` 的通信模式。"
    "例如，在 causal 掩码中，{math}`\\mathrm{rank}_2` 上的 {math}`\\mathrm{KV}_5` "
    "仅被 {math}`\\{\\mathrm{Q}_6,\\mathrm{Q}_7\\}` 需要，"
    "应通过 `GroupCast` 仅发送到目标 rank {math}`\\{\\mathrm{rank}_0, \\mathrm{rank}_1\\}`，"
    "而部分 {math}`\\mathrm{dKV}_5` 则通过 `GroupReduce` 相应地收集并归约回 {math}`\\mathrm{rank}_2`。",
    "GroupCast/GroupReduce Primitives": "GroupCast/GroupReduce 原语",
    "Illustration of `GroupCast/GroupReduce` primitives implemented atop "
    "`AlltoAll-v` to achieve zero redundancy, shown using the varlen block-"
    "causal mask with the last global block. (a) For forward and backward "
    "passes, `GroupCast` builds a transfer table for {math}`\\mathrm{KV}` "
    "send/receive buffers, invokes `AlltoAll-v`, and uses a custom `Range-"
    "Gather` kernel for pre-/post-processing. (b) In the backward pass, "
    "`GroupReduce` aggregates partial {math}`\\mathrm{dKV}` via `AlltoAll-v`, "
    "employing `Range-Gather` for pre-processing and `Range-Scatter-Reduce` "
    "for post-processing.": "基于 `AlltoAll-v` 实现零冗余的 `GroupCast/GroupReduce` 原语示意图，"
    "以带有最后全局块的 varlen block-causal 掩码为例。"
    "(a) 在前向和反向传播中，`GroupCast` 为 {math}`\\mathrm{KV}` 发送/接收缓冲区构建传输表，"
    "调用 `AlltoAll-v`，并使用自定义 `Range-Gather` 内核进行预/后处理。"
    "(b) 在反向传播中，`GroupReduce` 通过 `AlltoAll-v` 聚合部分 {math}`\\mathrm{dKV}`，"
    "使用 `Range-Gather` 进行预处理，使用 `Range-Scatter-Reduce` 进行后处理。",
    "AlltoAll-v Implementation": "AlltoAll-v 实现",
    "Since no existing communication kernels support group collectives, we "
    "prototyped `GroupCast` and `GroupReduce` on top of `AlltoAll-v`, "
    "achieving zero-redundant communication in forward and backward passes "
    "(see {numref}`group_gather_reduce_all2allv`). This approach, however, "
    "requires additional pre-/post-processing: `GroupCast` must re-permute "
    "inputs for `AlltoAll-v` and restore outputs (`Range-Gather`), and "
    "`GroupReduce` also performs a reduction on the output (`Range-Scatter-"
    "Reduce`). Although we implemented these steps using optimized Triton "
    "kernels, the extra overhead remains non\u2011negligible and might impact end-"
    "to-end performance.": "由于现有通信内核不支持 group collective，我们在 `AlltoAll-v` 之上原型化了 "
    "`GroupCast` 和 `GroupReduce`，在前向和反向传播中实现了零冗余通信"
    "（见 {numref}`group_gather_reduce_all2allv`）。然而，这种方法需要额外的预/后处理："
    "`GroupCast` 必须为 `AlltoAll-v` 重新排列输入并恢复输出（`Range-Gather`），"
    "`GroupReduce` 还需对输出执行归约（`Range-Scatter-Reduce`）。"
    "虽然我们使用优化的 Triton 内核实现了这些步骤，但额外开销仍然不可忽略，可能影响端到端性能。",
    "Besides the extra pre-/post-processing D2D overhead, another obscure cost"
    " of the `AlltoAll-v` implementation is that it permits only a single "
    "send/recv buffer pair per peer pair and therefore does not natively "
    'support "cast" semantics. Thus, to send a tensor from one rank to a '
    "subset of peers of size {math}`m`, one must allocate {math}`m` separate "
    "send buffers\u2014one per destination\u2014and transfer them individually, even "
    "though the data are identical. This **duplication** incurs substantial "
    "communication overhead, which is particularly severe when the CP group "
    "includes internode peers using `RDMA`, whose bandwidth is much lower than"
    " intranode `NVLink`.": "除了额外的预/后处理 D2D 开销外，`AlltoAll-v` 实现的另一个隐蔽代价是"
    "它每对对等节点仅允许一个 send/recv 缓冲区对，因此不原生支持\u201ccast\u201d语义。"
    "因此，要将一个张量从一个 rank 发送到大小为 {math}`m` 的对等节点子集，"
    "必须分配 {math}`m` 个单独的发送缓冲区——每个目标一个——并逐个传输，"
    "即使数据完全相同。这种**重复**带来了大量通信开销，"
    "当 CP 组包含使用 `RDMA` 的跨节点对等节点时尤为严重，因为其带宽远低于节点内 `NVLink`。",
    "Native Implementation": "原生实现",
    "To mitigate the extra overhead of the `AlltoAll-v` implementation "
    "aforementioned, we develop a native CUDA kernel implementation of group "
    "collectives inspired by DeepEP {cite}`deepep2025`. It not only removes "
    "the pre-/post-processing D2D copies but also significantly improves "
    "efficiency via the optimization of **RDMA transfer de-duplication**, "
    "particularly for hierarchical CP groups spanning internode and intranode "
    "peers.": "为缓解上述 `AlltoAll-v` 实现的额外开销，我们受 DeepEP {cite}`deepep2025` 启发，"
    "开发了 group collective 的原生 CUDA 内核实现。它不仅消除了预/后处理 D2D 拷贝，"
    "还通过 **RDMA 传输去重**优化显著提升了效率，"
    "尤其适用于跨节点和节点内对等节点的分层 CP 组。",
    "Although further optimizations remain, gains are already evident in the "
    "[Attention Benchmark](#attention-benchmark), particularly when scaling up"
    " the hierarchical CP group size. Please see the separate [blog "
    "post](./native_grpcoll.md) for more details about the motivation, design,"
    " implementation, and experimental results of the native implementation of"
    " group collectives.": "虽然仍有进一步优化的空间，但收益在[注意力基准测试](#attention-benchmark)中已经显而易见，"
    "尤其是在扩展分层 CP 组大小时。关于 group collective 原生实现的动机、设计、实现和实验结果的更多详情，"
    "请参阅单独的[博客文章](./native_grpcoll.md)。",
    "Multi-Stage Computation/Communication Overlap": "多阶段计算/通信重叠",
    "Scheduling with KV-Comm Only": "仅 KV 通信调度",
    "Leveraging previous optimizations, we combine an optimized kernel, load-"
    "balanced dispatch, and zero-redundant primitives to minimize "
    "communication overhead and maximize computation throughput individually. "
    "Now, to drive true linear scalability, we introduce an adaptive multi-"
    "stage computation/communication overlap strategy that effectively hides "
    "communication latency and can be tuned manually or automatically.": "利用前述优化，我们将优化内核、负载均衡调度和零冗余原语结合起来，"
    "分别最小化通信开销和最大化计算吞吐量。"
    "现在，为实现真正的线性可扩展性，我们引入自适应多阶段计算/通信重叠策略，"
    "可有效隐藏通信延迟，并可手动或自动调优。",
    "Similar to prior works "
    "{cite}`liu2023ringattentionblockwisetransformers,zhao2023pytorch,async_tensor_parallelism_in_pytorch`,"
    " we schedule pipeline stages to overlap computation and communication in "
    "both forward and backward passes (see "
    "{numref}`multi_stage_overlap_fwd_bwd`). Each {math}`\\mathrm{rank}_i` "
    "partitions its remote {math}`\\mathrm{KV}`/{math}`\\mathrm{dKV}` "
    "exchanges into stages.": "类似于先前的工作 "
    "{cite}`liu2023ringattentionblockwisetransformers,zhao2023pytorch,async_tensor_parallelism_in_pytorch`，"
    "我们调度流水线阶段以在前向和反向传播中重叠计算和通信"
    "（见 {numref}`multi_stage_overlap_fwd_bwd`）。"
    "每个 {math}`\\mathrm{rank}_i` 将其远程 {math}`\\mathrm{KV}`/{math}`\\mathrm{dKV}` 交换划分为多个阶段。",
    "Multi-Stage Overlap Scheduling": "多阶段重叠调度",
    "Illustration of Magi Attention's multi-stage overlap scheduling. (a) "
    "Forward pass \u2014 a 4-stage schedule that overlaps computation (partial "
    "{math}`\\mathrm{O}` and {math}`\\mathrm{LSE}`) with prefetching of next-"
    "stage {math}`\\mathrm{KV}` requests, hiding communication latency except "
    "for the final stage's computation. (b) Backward pass \u2014 a 3-stage schedule"
    " that overlaps computation (partial {math}`\\mathrm{dQ}`, "
    "{math}`\\mathrm{dKV}`), next-stage {math}`\\mathrm{KV}` prefetches, and "
    "reduction of prior {math}`\\mathrm{dKV}` requests, leaving only the final"
    " stage of partial {math}`\\mathrm{dKV}` reduction exposed.": "Magi Attention 多阶段重叠调度示意图。"
    "（a）前向传播 — 4 阶段调度，将计算（部分 {math}`\\mathrm{O}` 和 {math}`\\mathrm{LSE}`）"
    "与下一阶段 {math}`\\mathrm{KV}` 请求的预取重叠，隐藏除最后阶段计算外的所有通信延迟。"
    "（b）反向传播 — 3 阶段调度，将计算（部分 {math}`\\mathrm{dQ}`、{math}`\\mathrm{dKV}`）、"
    "下一阶段 {math}`\\mathrm{KV}` 预取和前一阶段部分 {math}`\\mathrm{dKV}` 归约重叠，"
    "仅暴露最后阶段的部分 {math}`\\mathrm{dKV}` 归约。",
    "In the forward pass, the scheduler launches the `GroupCast` kernel to "
    "prefetch the next {math}`(i\\!+\\!1)`-th stage of remote "
    "{math}`\\mathrm{KV}` while asynchronously executing the current "
    "{math}`i`-th stage of the `FFA` kernel for partial attention. Since "
    "`local qkv` is always available for the initial stage, all communication "
    "latency is fully hidden, leaving only the final remote stage's "
    "computation exposed.": "在前向传播中，调度器启动 `GroupCast` 内核预取下一个第 {math}`(i\\!+\\!1)` 阶段的远程 "
    "{math}`\\mathrm{KV}`，同时异步执行当前第 {math}`i` 阶段的 `FFA` 内核进行部分注意力计算。"
    "由于 `local qkv` 在初始阶段始终可用，所有通信延迟完全被隐藏，仅暴露最后远程阶段的计算。",
    "In the backward pass, the scheduler prefetches the next "
    "{math}`(i\\!+\\!1)`-th stage of {math}`\\mathrm{KV}` and invokes the "
    "`GroupReduce` kernel to reduce the prior {math}`(i\\!-\\!1)`-th stage of "
    "partial {math}`\\mathrm{dKV}` before executing the current {math}`i`-th "
    "attention stage. This overlap conceals communication latency across "
    "stages, exposing only the final stage of partial {math}`\\mathrm{dKV}` "
    "reduction.": "在反向传播中，调度器预取下一个第 {math}`(i\\!+\\!1)` 阶段的 {math}`\\mathrm{KV}` "
    "并调用 `GroupReduce` 内核归约前一个第 {math}`(i\\!-\\!1)` 阶段的部分 {math}`\\mathrm{dKV}`，"
    "然后执行当前第 {math}`i` 阶段的注意力计算。"
    "这种重叠隐藏了跨阶段的通信延迟，仅暴露最后阶段的部分 {math}`\\mathrm{dKV}` 归约。",
    "Scheduling with QO-Comm Enabled": "启用 QO 通信的调度",
    "Initially, we follow the legacy heuristic that only "
    "{math}`\\mathrm{KV}`-related tensors are communicated while "
    "{math}`\\mathrm{QO}`-related tensors remain local, a common practice in "
    "prior works "
    "{cite}`liu2023ringattentionblockwisetransformers,fang2024uspunifiedsequenceparallelism`."
    " This simplifies scheduling and often reduces communication, particularly"
    " in GQA settings where {math}`\\mathrm{KV}` typically has lower volume "
    "than {math}`\\mathrm{QO}`.": "最初，我们遵循传统启发式，仅通信 {math}`\\mathrm{KV}` 相关张量，"
    "而 {math}`\\mathrm{QO}` 相关张量保持本地，这是先前工作 "
    "{cite}`liu2023ringattentionblockwisetransformers,fang2024uspunifiedsequenceparallelism`"
    " 中的常见做法。这简化了调度并通常减少通信，"
    "尤其在 GQA 设置下 {math}`\\mathrm{KV}` 通常比 {math}`\\mathrm{QO}` 体积更小。",
    "However, this heuristic is not fundamental and can be suboptimal for "
    "certain mask patterns and training setups. We therefore support a more "
    "general scheduler that permits communication of {math}`\\mathrm{QO}` when"
    " advantageous. In the forward pass, the scheduler will prefetch the next "
    "stage of remote {math}`\\mathrm{Q}` in addition to remote "
    "{math}`\\mathrm{KV}`, overlapping both of them with the current `FFA` "
    "computation. And a major difference to {math}`\\mathrm{KV}`-only schedule"
    " is that we also need to apply **{math}`\\mathrm{LSE}`-reduction** for "
    "the previous stage's partial {math}`\\mathrm{O,LSE}` while overlapping "
    "with the current stage of computation.": "然而，这种启发式并非根本性的，对于某些掩码模式和训练设置可能是次优的。"
    "因此我们支持更通用的调度器，允许在有利时通信 {math}`\\mathrm{QO}`。"
    "在前向传播中，调度器除了预取远程 {math}`\\mathrm{KV}` 外，还会预取下一阶段的远程 {math}`\\mathrm{Q}`，"
    "将两者与当前 `FFA` 计算重叠。与仅 {math}`\\mathrm{KV}` 调度的主要区别在于，"
    "我们还需要对前一阶段的部分 {math}`\\mathrm{O,LSE}` 应用 **{math}`\\mathrm{LSE}` 归约**，"
    "同时与当前阶段的计算重叠。",
    "In the backward pass, the scheduler will prefetch the next stage of "
    "remote {math}`\\mathrm{KV}` and {math}`\\mathrm{Q,O,dO,LSE}` and "
    "concurrently sum-reduce the prior stage's partial {math}`\\mathrm{dKV}` "
    "and {math}`\\mathrm{dQ}`, overlapping with the current `FFA` backward "
    "computation.": "在反向传播中，调度器将预取下一阶段的远程 {math}`\\mathrm{KV}` 和 "
    "{math}`\\mathrm{Q,O,dO,LSE}`，同时并发对前一阶段的部分 {math}`\\mathrm{dKV}` 和 "
    "{math}`\\mathrm{dQ}` 进行求和归约，与当前 `FFA` 反向计算重叠。",
    "Although the scheduler itself is already supported, enabling this mode "
    "also requires the `dynamic attn solver` to emit the corresponding "
    "`CalcMeta` and `CommMeta` for `FFA` and the group-collective kernels, "
    "which is under active development (see [Dynamic Attn Solver](#dynamic-"
    "attn-solver)). We will release it soon and continue to optimize it for "
    "better performance.": "虽然调度器本身已经支持，但启用此模式还需要 `dynamic attn solver` "
    "为 `FFA` 和 group-collective 内核生成相应的 `CalcMeta` 和 `CommMeta`，"
    "这正在积极开发中（见 [Dynamic Attn Solver](#dynamic-attn-solver)）。"
    "我们将尽快发布并持续优化以获得更好的性能。",
    "How to Ensure Kernels Actually Overlapped": "如何确保内核实际重叠",
    "While the CPU scheduler controls kernel launch order to favor overlap, "
    "the GPU Hyper-Q driver {cite}`bradley2013hyperq` ultimately determines "
    "actual execution order non\u2011deterministically, influenced by transient GPU"
    " resource occupancy as well. Ensuring reliable overlap between "
    "computation and communication kernels is therefore non\u2011trivial.": "虽然 CPU 调度器控制内核启动顺序以促进重叠，但 GPU Hyper-Q 驱动 "
    "{cite}`bradley2013hyperq` 最终以非确定性方式决定实际执行顺序，"
    "同时还受瞬态 GPU 资源占用的影响。因此，确保计算和通信内核之间的可靠重叠并非易事。",
    "See the separate [blog post](./kernel_overlap.md) for practical "
    "techniques and our specific novel approaches.": "关于实用技术和我们的具体新方法，请参阅单独的[博客文章](./kernel_overlap.md)。",
    "Dynamic Overlap Stage Search": "动态重叠阶段搜索",
    "In practice, {math}`\\textit{overlap\\_degree}` is typically tuned "
    "manually in {math}`\\{1,2,3,4\\}`. Automatic search by the `overlap "
    "solver` often underperforms because it requires accurate estimates of <em"
    ">computation-to-communication ratios</em>. We therefore recommend trying "
    "manual tuning for a few iterations to identify a suitable "
    "{math}`\\textit{overlap\\_degree}` before enabling automatic search, "
    "which we will continue to improve for greater robustness.": "在实践中，{math}`\\textit{overlap\\_degree}` 通常在 {math}`\\{1,2,3,4\\}` 中手动调优。"
    "`overlap solver` 的自动搜索往往表现不佳，因为它需要准确估计<em>计算通信比</em>。"
    "因此我们建议先手动调优几次迭代以确定合适的 {math}`\\textit{overlap\\_degree}`，"
    "再启用自动搜索，我们将继续改进以提高鲁棒性。",
    "To control overlap granularity, we introduce the tunable hyperparameter "
    "{math}`\\textit{overlap\\_degree}`, indicating the number of remote "
    "stages to be partitioned, which adapts to varying <em>computation-to-"
    "communication ratios</em> across training setups, microbatches, and "
    "between forward and backward passes. It can be set manually by the user "
    "on their own training setup. Or, we provide an algorithm to choose "
    "automatically by the `overlap solver` using the dynamic search described "
    "in the following {numref}`dynamic_mso_alg`.": "为控制重叠粒度，我们引入可调超参数 {math}`\\textit{overlap\\_degree}`，"
    "指示要划分的远程阶段数，它适应不同训练设置、微批次以及前向和反向传播之间"
    "变化的<em>计算通信比</em>。用户可以在自己的训练设置上手动设定。"
    "或者，我们提供了一种算法，由 `overlap solver` 使用以下 {numref}`dynamic_mso_alg` "
    "中描述的动态搜索自动选择。",
    "Dynamic Overlap Stage Search Algorithm": "动态重叠阶段搜索算法",
    "Experiments": "实验",
    "Attention Benchmark": "注意力基准测试",
    "To evaluate the performance and flexibility of `FFA` kernels and to "
    "validate the distributed scalability of `MagiAttention` for ultra-long, "
    "heterogeneous-mask training, we benchmark throughput on modern GPUs "
    "(e.g., Hopper and Blackwell) for both kernels and distributed attention "
    "modules in forward and backward passes across diverse mask patterns "
    "(standard and irregular), comparing against state-of-the-art kernel- and "
    "distributed-level baselines.": "为评估 `FFA` 内核的性能和灵活性，并验证 `MagiAttention` 在超长、异构掩码训练中的分布式可扩展性，"
    "我们在现代 GPU（如 Hopper 和 Blackwell）上对内核和分布式注意力模块"
    "在前向和反向传播中跨多种掩码模式（标准和不规则）进行吞吐量基准测试，"
    "并与最先进的内核级和分布式级基线进行对比。",
    "We present representative distributed-level benchmarks below for the most"
    " commonly used `varlen causal` mask on both H100 and B200 GPUs, "
    "highlighting MagiAttention's performance and scalability versus other "
    "leading CP strategies.": "我们在下方展示了最常用的 `varlen causal` 掩码在 H100 和 B200 GPU 上的代表性分布式级基准测试结果，"
    "突出展示 MagiAttention 相对于其他领先 CP 策略的性能和可扩展性。",
    "For detailed benchmark settings and results, see the separate [blog "
    "post](./cp_benchmark.md).": "详细的基准测试设置和结果请参阅单独的[博客文章](./cp_benchmark.md)。",
    "H100": "H100",
    "Distributed-Level Throughput - Varlen Causal Mask Forward Pass": "分布式级吞吐量 - Varlen Causal 掩码前向传播",
    "(a) Forward Pass": "（a）前向传播",
    "Distributed-Level Throughput - Varlen Causal Mask Backward Pass": "分布式级吞吐量 - Varlen Causal 掩码反向传播",
    "(b) Backward Pass": "（b）反向传播",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on H100 for the `varlen causal` mask.": "在 H100 上对 `MagiAttention` 的 `varlen causal` 掩码性能和可扩展性进行基线对比基准测试。",
    "B200": "B200",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on B200 for the `varlen causal` mask.": "在 B200 上对 `MagiAttention` 的 `varlen causal` 掩码性能和可扩展性进行基线对比基准测试。",
    "Miscellaneous": "其他",
    "Preliminaries": "预备知识",
    "Flash Attention 2 Math Derivation": "Flash Attention 2 数学推导",
    "See the separate [blog post](./fa2_math_derivation.md) for a detailed "
    "mathematical derivation of the `Flash-Attention 2` forward and backward "
    "passes, which serves as the foundation for our `Flex-Flash-Attention` "
    "kernel design.": "关于 `Flash-Attention 2` 前向和反向传播的详细数学推导，请参阅单独的[博客文章](./fa2_math_derivation.md)，"
    "这是我们 `Flex-Flash-Attention` 内核设计的基础。",
    "Extended Functionalities": "扩展功能",
    "FFA_FA4 Backend for Blackwell": "Blackwell 的 FFA_FA4 后端",
    "Since `FFA` is built on `FA3` kernels that are available only on Hopper, "
    "we provide a temporary `FFA_FA4` backend to enable `MagiAttention` on "
    "Blackwell. `FFA_FA4` implements flexible masking via an `HSTU Function` "
    "representation based on a forked [`Flash-Attention 4` "
    "(`FA4`)](https://github.com/demonatic/flash-"
    "attention/tree/magi_attn_blackwell_support).": "由于 `FFA` 基于仅在 Hopper 上可用的 `FA3` 内核构建，"
    "我们提供了临时的 `FFA_FA4` 后端以在 Blackwell 上启用 `MagiAttention`。"
    "`FFA_FA4` 基于分叉的 [`Flash-Attention 4` (`FA4`)](https://github.com/demonatic/flash-"
    "attention/tree/magi_attn_blackwell_support)，通过 `HSTU Function` 表示实现灵活掩码。",
    "See the separate [blog post](./blackwell_ffa_fa4.md) for design details "
    "and the [Attention Benchmark](./cp_benchmark.md) for Blackwell "
    "performance comparisons for both kernel-level and distributed-level.": "设计详情请参阅单独的[博客文章](./blackwell_ffa_fa4.md)，"
    "Blackwell 内核级和分布式级性能对比请参阅[注意力基准测试](./cp_benchmark.md)。",
    "Attention Sink": "Attention Sink",
    "See the separate [blog post](./attn_sink.md) for a technical description "
    "of how we natively support **learnable attention sink mechanism** in "
    "`Flex-Flash-Attention` (kernel-level), `MagiAttention` (distributed-"
    "level), and `Flash-Attention` (one of the [MagiAttention "
    "Extensions](https://github.com/SandAI-"
    "org/MagiAttention/tree/main/extensions#flashattention-with-attention-"
    "sink-)).": "关于我们如何在 `Flex-Flash-Attention`（内核级）、`MagiAttention`（分布式级）"
    "和 `Flash-Attention`（[MagiAttention 扩展](https://github.com/SandAI-"
    "org/MagiAttention/tree/main/extensions#flashattention-with-attention-sink-) 之一）中"
    "原生支持**可学习 attention sink 机制**的技术描述，请参阅单独的[博客文章](./attn_sink.md)。",
    "Muon QK-Clip": "Muon QK-Clip",
    "See the separate [blog post](./muon_qk_clip.md) for a technical "
    "description of how we natively support **Muon QK-clip technique** in "
    "`Flex-Flash-Attention` (kernel-level) and `MagiAttention` (distributed-"
    "level).": "关于我们如何在 `Flex-Flash-Attention`（内核级）和 `MagiAttention`（分布式级）中"
    "原生支持 **Muon QK-clip 技术**的技术描述，请参阅单独的[博客文章](./muon_qk_clip.md)。",
    "JIT Compilation in FFA": "FFA 中的 JIT 编译",
    "See the separate [blog post](./jit_compile.md) for a technical "
    "description of how we support **Just-In-Time (JIT) compilation** in "
    "`Flex-Flash-Attention`, to reduce pre-building overhead and deliver "
    "optimized kernels for varied attention patterns and training scenarios.": "关于我们如何在 `Flex-Flash-Attention` 中支持**即时 (JIT) 编译**的技术描述，"
    "请参阅单独的[博客文章](./jit_compile.md)，以减少预构建开销并为各种注意力模式和训练场景提供优化内核。",
    "Optimization Techniques": "优化技术",
    "Optimize Sparse Attention in FFA": "优化 FFA 中的稀疏注意力",
    "`Sparse Attention` is a promising research direction to trade model "
    "capacity for sub-quadratic attention cost using (*static/dynamic*) "
    "highly-sparse mask patterns "
    "{cite}`child2019generatinglongsequencessparse,beltagy2020longformerlongdocumenttransformer,zaheer2021bigbirdtransformerslonger,zhang2025spargeattentionaccuratetrainingfreesparse`."
    " Recent works such as `NSA` "
    "{cite}`yuan2025nativesparseattentionhardwarealigned` and `DSA` "
    "{cite}`deepseekai2025deepseekv32pushingfrontieropen` from DeepSeek "
    "introduce novel (*dynamic*) trainable sparse attention mechanisms, "
    "bringing new opportunities for efficient training. Therefore we've been "
    "implementing targeted optimizations on `FFA` for sparse masks to "
    "**natively support (distributed) trainable sparse attention**, and share "
    "our preliminary results in the separate [blog post](./sparse_attn.md).": "`Sparse Attention` 是一个有前景的研究方向，通过（*静态/动态*）高度稀疏的掩码模式"
    "以模型容量换取亚二次注意力代价 "
    "{cite}`child2019generatinglongsequencessparse,beltagy2020longformerlongdocumenttransformer,zaheer2021bigbirdtransformerslonger,zhang2025spargeattentionaccuratetrainingfreesparse`。"
    "近期工作如 DeepSeek 的 `NSA` {cite}`yuan2025nativesparseattentionhardwarealigned` 和 "
    "`DSA` {cite}`deepseekai2025deepseekv32pushingfrontieropen` "
    "引入了新颖的（*动态*）可训练稀疏注意力机制，为高效训练带来了新机遇。"
    "因此我们一直在 `FFA` 上针对稀疏掩码实施定向优化以"
    "**原生支持（分布式）可训练稀疏注意力**，并在单独的[博客文章](./sparse_attn.md)中分享了初步结果。",
    "Next-Generation Design": "下一代设计",
    "Distributed-Native FFA": "分布式原生 FFA",
    "See the separate [blog post](./dist_native.md) for a technical proposal "
    "for the next major version update of `MagiAttention`: a **distributed-"
    "native `FFA` kernel** with fused warp-level communication primitives to "
    "further reduce communication overhead and kernel launch latency.": "关于 `MagiAttention` 下一个主要版本更新的技术提案，请参阅单独的[博客文章](./dist_native.md)："
    "一种**分布式原生 `FFA` 内核**，融合了 warp 级通信原语，进一步减少通信开销和内核启动延迟。",
    "Attention Engine for Inference": "推理用 Attention Engine",
    "See the separate [blog post](./attn_engine.md) for a technical proposal "
    "of the next-generation design named **Attention Engine**, which targets "
    "efficient distributed attention serving for inference scenarios.": "关于名为 **Attention Engine** 的下一代设计技术提案，请参阅单独的[博客文章](./attn_engine.md)，"
    "其目标是为推理场景提供高效的分布式注意力服务。",
    "Future Work": "未来工作",
    "**[WIP]** Optimize `FFA` kernels on Hopper for improved performance, with"
    " emphasis on <u>sparse attention</u> scenarios.": "**[WIP]** 优化 Hopper 上的 `FFA` 内核以提升性能，重点针对<u>稀疏注意力</u>场景。",
    "**[WIP]** Implement native `GroupCast` and `GroupReduce` communication "
    "kernels to reduce communication overhead and lower compute occupancy.": "**[WIP]** 实现原生 `GroupCast` 和 `GroupReduce` 通信内核，以减少通信开销和降低计算占用率。",
    "**[WIP]** Extend the `dynamic attn solver` to better handle dynamic mask "
    "patterns (e.g., <u>hybrid attention</u>, <u>sparse attention</u>) for "
    "lower communication and improved load balance.": "**[WIP]** 扩展 `dynamic attn solver` 以更好地处理动态掩码模式"
    "（例如<u>混合注意力</u>、<u>稀疏注意力</u>），实现更低通信和更好的负载均衡。",
    "Optimize the `static attn solver` to reduce CPU meta-info overhead.": "优化 `static attn solver` 以减少 CPU 元信息开销。",
    "Support individual `OverlapConfig` for forward and backward passes, and "
    "further extend the `overlap solver` to automatically determine optimal "
    "overlap strategies for forward and backward passes separately.": "支持前向和反向传播的独立 `OverlapConfig`，并进一步扩展 `overlap solver` "
    "以自动分别确定前向和反向传播的最优重叠策略。",
    "Implement native `FFA` kernels on Blackwell to replace the temporary "
    "`FFA_FA4` backend.": "在 Blackwell 上实现原生 `FFA` 内核以替代临时的 `FFA_FA4` 后端。",
    "Port `FFA` to additional GPU architectures (e.g., Ampere).": "将 `FFA` 移植到更多 GPU 架构（例如 Ampere）。",
    "Extend attention benchmarking for more GPU architectures beyond H100 and "
    "B200 (e.g., B300 and A100).": "将注意力基准测试扩展到 H100 和 B200 之外的更多 GPU 架构（例如 B300 和 A100）。",
    "Expand documentation with more examples and a tuning guide for varied "
    "training scenarios.": "扩展文档，增加更多示例和针对各种训练场景的调优指南。",
    "Prepare a standalone technical report/paper detailing MagiAttention.": "准备一份独立的技术报告/论文，详细介绍 MagiAttention。",
    "Simplify installation and provide pre-built binaries for common "
    "environments.": "简化安装流程并为常见环境提供预构建二进制文件。",
    "Reduce the configuration space and the number of optional performance-"
    "related environment variables in `MagiAttention` with better defaults and"
    " auto-tuning capabilities.": "通过更好的默认值和自动调优能力，减少 `MagiAttention` 的配置空间"
    "和可选性能相关环境变量的数量。",
    "Add support for additional attention patterns, including cross-attention "
    "and inference use cases.": "增加对更多注意力模式的支持，包括交叉注意力和推理用例。",
    "Upgrade `MagiAttention` to a distributed-native `FFA` kernel with fused "
    "warp-level communication primitives.": "将 `MagiAttention` 升级为具有融合 warp 级通信原语的分布式原生 `FFA` 内核。",
    "Implement `Attention Engine` for distributed attention serving in "
    "inference scenarios.": "实现 `Attention Engine` 以在推理场景中提供分布式注意力服务。",
    "Support MagiAttention on Blackwell with a temporary `FFA_FA4` backend.": "通过临时 `FFA_FA4` 后端在 Blackwell 上支持 MagiAttention。",
    "Support `dynamic attn solver` with query/output communication pattern to "
    "reduce communication in cases where KV-only communication is suboptimal.": "支持 `dynamic attn solver` 的 query/output 通信模式，"
    "以在仅 KV 通信次优的情况下减少通信。",
    "Prototype native `GroupCast` and `GroupReduce` primitives with inter"
    "-/intra-node hierarchical optimization based on "
    "[DeepEP](https://github.com/deepseek-ai/DeepEP).": "基于 [DeepEP](https://github.com/deepseek-ai/DeepEP) 原型化具有跨节点/节点内"
    "分层优化的原生 `GroupCast` 和 `GroupReduce` 原语。",
    "Support learnable attention sink integration with "
    "[StreamingLLM](https://arxiv.org/abs/2309.17453).": "支持与 [StreamingLLM](https://arxiv.org/abs/2309.17453) 集成的可学习 attention sink。",
    "Refactor `dist attn solver` to support all four mask types and full "
    "overlapping strategies.": "重构 `dist attn solver` 以支持所有四种掩码类型和完整的重叠策略。",
    "Improve the `dispatch solver` to reduce communication volume while "
    "maintaining compute balance, especially for varlen masks.": "改进 `dispatch solver` 以在保持计算均衡的同时减少通信量，尤其针对 varlen 掩码。",
    "Build a comprehensive `CP Benchmark` validating MagiAttention across mask"
    " patterns and training settings.": "构建全面的 `CP Benchmark`，在不同掩码模式和训练设置下验证 MagiAttention。",
    "Provide `Documentation` covering `Installation`, `QuickStart`, `API "
    "reference`, and `Environment Variables`.": "提供涵盖 `Installation`、`QuickStart`、`API reference` 和 `Environment Variables` 的 `Documentation`。",
    "Citation": "引用",
    "If you find MagiAttention useful in your research, please cite:": "如果你在研究中发现 MagiAttention 有用，请引用：",
    "References": "参考文献",
}

# ============================================================
# cp_benchmark.po translations
# ============================================================
cp_benchmark_translations = {
    "Long-Context Attention Benchmark": "长上下文注意力基准测试",
    "**From Kernel Efficiency to Distributed Scalability**": "**从内核效率到分布式可扩展性**",
    "To evaluate the performance and flexibility of `Flex-Flash-Attention` "
    "(`FFA`) kernels and to validate the distributed scalability of "
    "`MagiAttention` for ultra-long, heterogeneous-mask training, we benchmark"
    " throughput on modern GPUs (e.g., Hopper and Blackwell) for both kernels "
    "and distributed attention modules in forward and backward passes across "
    "diverse mask patterns (standard and irregular), against state-of-the-art "
    "kernel- and distributed-level baselines.": "为评估 `Flex-Flash-Attention` (`FFA`) 内核的性能和灵活性，"
    "并验证 `MagiAttention` 在超长、异构掩码训练中的分布式可扩展性，"
    "我们在现代 GPU（如 Hopper 和 Blackwell）上对内核和分布式注意力模块"
    "在前向和反向传播中跨多种掩码模式（标准和不规则）进行吞吐量基准测试，"
    "并与最先进的内核级和分布式级基线进行对比。",
    "Benchmark Settings": "基准测试设置",
    "Common Configurations": "通用配置",
    "To focus on the impact of sequence length and mask pattern, we fix other "
    "data and model configurations using common training settings as shown in "
    "the table below.": "为了聚焦序列长度和掩码模式的影响，我们使用如下表所示的常见训练设置固定其他数据和模型配置。",
    "settings": "设置",
    "value": "值",
    "attention type": "注意力类型",
    "self-attention where `seqlen = seqlen_q = seqlen_k`": "自注意力，其中 `seqlen = seqlen_q = seqlen_k`",
    "batch size (b)": "批次大小（b）",
    "1": "1",
    "number of heads (nh)": "头数（nh）",
    "nhq:nhk:nhv = 64:8:8 (GQA)": "nhq:nhk:nhv = 64:8:8 (GQA)",
    "head dimension (hd)": "头维度（hd）",
    "128": "128",
    "dtype": "dtype",
    "`torch.bfloat16`": "`torch.bfloat16`",
    "window size": "窗口大小",
    "1024 (for sliding window masks only)": "1024（仅用于 sliding window 掩码）",
    "Throughput Metrics": "吞吐量指标",
    "Throughput is measured in {math}`\\texttt{TFLOPs/s}` for kernel-level "
    "benchmarks and {math}`\\texttt{TFLOPs/s/GPU}` for distributed benchmarks,"
    " calculated based on the total number of floating-point operations "
    "({math}`\\texttt{FLOPs}`) involved in the attention computation, for both"
    " forward and backward passes respectively.": "吞吐量以 {math}`\\texttt{TFLOPs/s}` 衡量（内核级基准测试）"
    "和 {math}`\\texttt{TFLOPs/s/GPU}`（分布式基准测试），"
    "分别基于注意力计算中前向和反向传播涉及的浮点运算总数 ({math}`\\texttt{FLOPs}`) 计算。",
    "The {math}`\\texttt{FLOPs}` for each {math}`\\mathrm{AttnSlice}` are "
    "computed using the formula below, and the total {math}`\\texttt{FLOPs}` "
    "is the summation of all {math}`\\mathrm{AttnSlice}`:": "每个 {math}`\\mathrm{AttnSlice}` 的 {math}`\\texttt{FLOPs}` 使用以下公式计算，"
    "总 {math}`\\texttt{FLOPs}` 是所有 {math}`\\mathrm{AttnSlice}` 的总和：",
    "And the throughputs are calculated as follows:": "吞吐量按以下方式计算：",
    "Data Distribution and Sampling": "数据分布和采样",
    "To reflect real-world long-context training, we extract the sequence-"
    "length distribution from a representative training dataset and use it to "
    "construct variable-length inputs for both kernel- and distributed-level "
    "experiments (see {numref}`varlen_seqlen_distribution`).": "为反映真实的长上下文训练，我们从代表性训练数据集中提取序列长度分布，"
    "并用于构建内核级和分布式级实验的变长输入（见 {numref}`varlen_seqlen_distribution`）。",
    "Variable-Length Sequence Distribution": "变长序列分布",
    "Distribution of sequence lengths extracted from a real-world dataset, "
    "which is used to sample and construct the variable-length data for both "
    "kernel-level and distributed-level experiments.": "从真实数据集中提取的序列长度分布，用于采样和构建内核级和分布式级实验的变长数据。",
    "We shuffle the dataset, sequentially pack samples into data packs, then "
    "reshuffle those packs to form the final sampling set, where we will fetch"
    " a portion of packs for experiments using `varlen` mask patterns. This "
    "preserves the original token-length distribution so the probability of "
    "tokens from long and short samples within each pack matches the dataset.": "我们对数据集进行洗牌，将样本依次打包成数据包，然后重新洗牌这些数据包以形成最终采样集，"
    "我们将从中获取一部分数据包用于 `varlen` 掩码模式的实验。"
    "这保留了原始 token 长度分布，使每个数据包中来自长短样本的 token 概率与数据集匹配。",
    "To avoid the sampled variable-length data from degenerating into pure "
    "`full/causal` masks to affect the evaluation, we limit each sample's "
    "length at most {math}`\\frac{1}{4}` of the total sequence length (e.g., "
    "no sample exceeds `16K` when measuring with a `64K` total sequence "
    "length).": "为避免采样的变长数据退化为纯 `full/causal` 掩码而影响评估，"
    "我们限制每个样本的长度最多为总序列长度的 {math}`\\frac{1}{4}`"
    "（例如，在使用 `64K` 总序列长度测量时，没有样本超过 `16K`）。",
    "Kernel Baselines": "内核基线",
    "On Hopper, we evaluate our [`FFA`](./magi_attn.md#flex-flash-attention) "
    "kernel against widely used PyTorch's fused `SDPA` "
    "{cite}`pytorch_sdpa_cp_benchmark`, `Flash Attention 2` (`FA2`) "
    "{cite}`dao2023flashattention_cp_benchmark`, `Flash Attention 3` (`FA3`) "
    "{cite}`shah2024flashattention3_cp_benchmark`, NVIDIA's `cuDNN` fused "
    "attention kernel {cite}`nvidia2024accelerating_cp_benchmark` from "
    "[TransformerEngine](https://github.com/NVIDIA/TransformerEngine), as well"
    " as PyTorch's new `FlexAttention` "
    "{cite}`dong2024flexattentionprogrammingmodel_cp_benchmark` and Baidu's "
    "`FlashMask` {cite}`wang2025flashmaskefficientrichmask_cp_benchmark` for "
    "baselines on flexible masks.": "在 Hopper 上，我们将 [`FFA`](./magi_attn.md#flex-flash-attention) 内核与广泛使用的基线进行对比："
    "PyTorch 的融合 `SDPA` {cite}`pytorch_sdpa_cp_benchmark`、"
    "`Flash Attention 2` (`FA2`) {cite}`dao2023flashattention_cp_benchmark`、"
    "`Flash Attention 3` (`FA3`) {cite}`shah2024flashattention3_cp_benchmark`、"
    "来自 [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) 的 NVIDIA `cuDNN` "
    "融合注意力内核 {cite}`nvidia2024accelerating_cp_benchmark`，"
    "以及 PyTorch 的新 `FlexAttention` {cite}`dong2024flexattentionprogrammingmodel_cp_benchmark` "
    "和百度的 `FlashMask` {cite}`wang2025flashmaskefficientrichmask_cp_benchmark` 作为灵活掩码基线。",
    "On Blackwell, we instead evaluate our [`FFA_FA4`](./blackwell_ffa_fa4.md)"
    " kernel against the same baselines, substituting `FA2` and `FA3` with "
    "`Flash Attention 4` (`FA4`) "
    "{cite}`dao2025flashattention_cute_cp_benchmark`, since both `FFA` and "
    "`FA3` are tailored for Hopper and `FA2` does not optimize for SM90+ "
    "architectures. And we don't report the backward performance for `FA4` "
    "since it currently lacks robust support for `varlen` masks, especially on"
    " stable version of `2.8.3`.": "在 Blackwell 上，我们将 [`FFA_FA4`](./blackwell_ffa_fa4.md) 内核与相同基线进行对比，"
    "用 `Flash Attention 4` (`FA4`) {cite}`dao2025flashattention_cute_cp_benchmark` "
    "替代 `FA2` 和 `FA3`，因为 `FFA` 和 `FA3` 都是为 Hopper 定制的，"
    "而 `FA2` 未针对 SM90+ 架构优化。"
    "我们不报告 `FA4` 的反向性能，因为它目前对 `varlen` 掩码缺乏稳健支持，尤其是在稳定版本 `2.8.3` 上。",
    "Distributed Baselines": "分布式基线",
    "We evaluate `MagiAttention` against state-of-the-art distributed "
    "attention mechanisms integrated into [Megatron-"
    "LM](https://github.com/NVIDIA/Megatron-LM) as context-parallel (CP) "
    "backends, including `Ulysess` {cite}`jacobs2023deepspeed_cp_benchmark`, "
    "`Ring P2P` "
    "{cite}`liu2023ringattentionblockwisetransformers_cp_benchmark`, `Ring "
    "AllGather` {cite}`grattafiori2024llama3herdmodels_cp_benchmark`, `USP` "
    "{cite}`fang2024uspunifiedsequenceparallelism_cp_benchmark`, `LoongTrain` "
    "{cite}`gu2024loongtrainefficienttraininglongsequence_cp_benchmark`, and "
    "Megatron `HybridCP` {cite}`megatron-lm-hybrid-cp-pr-2054_cp_benchmark`. "
    "Many of these are discussed in the [Related Work](./magi_attn.md#related-"
    "work) section of the main MagiAttention [blog post](./magi_attn.md).": "我们将 `MagiAttention` 与集成到 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) "
    "中作为上下文并行 (CP) 后端的最先进分布式注意力机制进行对比，包括 "
    "`Ulysess` {cite}`jacobs2023deepspeed_cp_benchmark`、"
    "`Ring P2P` {cite}`liu2023ringattentionblockwisetransformers_cp_benchmark`、"
    "`Ring AllGather` {cite}`grattafiori2024llama3herdmodels_cp_benchmark`、"
    "`USP` {cite}`fang2024uspunifiedsequenceparallelism_cp_benchmark`、"
    "`LoongTrain` {cite}`gu2024loongtrainefficienttraininglongsequence_cp_benchmark` 和 "
    "Megatron `HybridCP` {cite}`megatron-lm-hybrid-cp-pr-2054_cp_benchmark`。"
    "其中许多在 MagiAttention 主[博客文章](./magi_attn.md)的"
    "[相关工作](./magi_attn.md#related-work)部分有讨论。",
    "On Hopper, all baselines use the `FA3` kernel as the attention backend to"
    " ensure a fair comparison with our `FFA` kernel.": "在 Hopper 上，所有基线使用 `FA3` 内核作为注意力后端，以确保与我们的 `FFA` 内核进行公平比较。",
    "On Blackwell, since `FA3` targets Hopper and `FA4` currently lacks robust"
    " backward support for varlen masks on stable version of `2.8.3`, "
    "baselines use the `cuDNN` kernel while we use our `FFA_FA4` backend. "
    "Additionally, Megatron `HybridCP` (which requires `FA3`) is omitted from "
    "Blackwell evaluations.": "在 Blackwell 上，由于 `FA3` 针对 Hopper 而 `FA4` 在稳定版本 `2.8.3` 上"
    "目前缺乏对 varlen 掩码的稳健反向支持，基线使用 `cuDNN` 内核而我们使用 `FFA_FA4` 后端。"
    "此外，Megatron `HybridCP`（需要 `FA3`）在 Blackwell 评估中被省略。",
    "Kernel Level": "内核级别",
    "For kernel-level benchmarking, we evaluate the kernels across 5 common "
    "mask patterns including `full`, `causal`, `varlen full`, `varlen causal` "
    "and `sliding window causal` with one irregular `varlen block causal` mask"
    " used in [Magi-1](https://github.com/SandAI-org/MAGI-1), to assess "
    "performance and flexibility, with the total sequence length varying from "
    "`1K,2K,4K,...,` up to `64K` for both forward and backward passes.": "对于内核级基准测试，我们在 5 种常见掩码模式上评估内核，包括 `full`、`causal`、"
    "`varlen full`、`varlen causal` 和 `sliding window causal`，"
    "以及 [Magi-1](https://github.com/SandAI-org/MAGI-1) 中使用的一种不规则 `varlen block causal` 掩码，"
    "以评估性能和灵活性，总序列长度从 `1K,2K,4K,...,` 变化到 `64K`，涵盖前向和反向传播。",
    "Results are reported in the following figures, while the legend-name "
    "mapping is described below:": "结果在以下图表中报告，图例-名称映射如下所述：",
    "legend": "图例",
    "name": "名称",
    "ffa": "ffa",
    "`FFA`": "`FFA`",
    "fa2 / fa3 / fa4": "fa2 / fa3 / fa4",
    "`FA2` / `FA3` / `FA4`": "`FA2` / `FA3` / `FA4`",
    "cudnn": "cudnn",
    "NVIDIA `cuDNN` fused attention": "NVIDIA `cuDNN` 融合注意力",
    "sdpa": "sdpa",
    "PyTorch's `SDPA`": "PyTorch 的 `SDPA`",
    "flex": "flex",
    "PyTorch's `FlexAttention`": "PyTorch 的 `FlexAttention`",
    "flash_mask": "flash_mask",
    "Baidu's `FlashMask`": "百度的 `FlashMask`",
    "The {math}`\\mathbf{X}` symbol denotes attention kernels unsupported in "
    "that configuration due to kernel limitations or error raised (e.g., `Cuda"
    " Out of Memory`).": "{math}`\\mathbf{X}` 符号表示由于内核限制或引发错误（例如 `Cuda Out of Memory`）"
    "而在该配置中不支持的注意力内核。",
    "For H100": "H100",
    "For B200": "B200",
    "Full Mask": "Full 掩码",
    "Causal Mask": "Causal 掩码",
    "Varlen Full Mask": "Varlen Full 掩码",
    "Varlen Causal Mask": "Varlen Causal 掩码",
    "Sliding Window Causal Mask": "Sliding Window Causal 掩码",
    "Varlen Block Causal Mask 🔥": "Varlen Block Causal 掩码 🔥",
    "Varlen Full Mask 🔥": "Varlen Full 掩码 🔥",
    "Varlen Causal Mask 🔥": "Varlen Causal 掩码 🔥",
    "Kernel-Level Throughput - Full Mask Forward Pass": "内核级吞吐量 - Full 掩码前向传播",
    "Kernel-Level Throughput - Full Mask Backward Pass": "内核级吞吐量 - Full 掩码反向传播",
    "Kernel-Level Throughput - Causal Mask Forward Pass": "内核级吞吐量 - Causal 掩码前向传播",
    "Kernel-Level Throughput - Causal Mask Backward Pass": "内核级吞吐量 - Causal 掩码反向传播",
    "Kernel-Level Throughput - Varlen Full Mask Forward Pass": "内核级吞吐量 - Varlen Full 掩码前向传播",
    "Kernel-Level Throughput - Varlen Full Mask Backward Pass": "内核级吞吐量 - Varlen Full 掩码反向传播",
    "Kernel-Level Throughput - Varlen Causal Mask Forward Pass": "内核级吞吐量 - Varlen Causal 掩码前向传播",
    "Kernel-Level Throughput - Varlen Causal Mask Backward Pass": "内核级吞吐量 - Varlen Causal 掩码反向传播",
    "Kernel-Level Throughput - Sliding Window Causal Mask Forward Pass": "内核级吞吐量 - Sliding Window Causal 掩码前向传播",
    "Kernel-Level Throughput - Sliding Window Causal Mask Backward Pass": "内核级吞吐量 - Sliding Window Causal 掩码反向传播",
    "Kernel-Level Throughput - Varlen Block Causal Mask Forward Pass": "内核级吞吐量 - Varlen Block Causal 掩码前向传播",
    "Kernel-Level Throughput - Varlen Block Causal Mask Backward Pass": "内核级吞吐量 - Varlen Block Causal 掩码反向传播",
    "(a) Forward Pass": "（a）前向传播",
    "(b) Backward Pass": "（b）反向传播",
    "Benchmarking `FFA`'s performance and flexibility against baselines on "
    "H100 for the `full` mask.": "在 H100 上对 `FFA` 的 `full` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA`'s performance and flexibility against baselines on "
    "H100 for the `causal` mask.": "在 H100 上对 `FFA` 的 `causal` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA`'s performance and flexibility against baselines on "
    "H100 for the `varlen full` mask.": "在 H100 上对 `FFA` 的 `varlen full` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA`'s performance and flexibility against baselines on "
    "H100 for the `varlen causal` mask.": "在 H100 上对 `FFA` 的 `varlen causal` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA`'s performance and flexibility against baselines on "
    "H100 for the `sliding window causal` mask.": "在 H100 上对 `FFA` 的 `sliding window causal` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA`'s performance and flexibility against baselines on "
    "H100 for the `varlen block causal` mask.": "在 H100 上对 `FFA` 的 `varlen block causal` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA_FA4`'s performance and flexibility against baselines on"
    " B200 for the `full` mask.": "在 B200 上对 `FFA_FA4` 的 `full` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA_FA4`'s performance and flexibility against baselines on"
    " B200 for the `causal` mask.": "在 B200 上对 `FFA_FA4` 的 `causal` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA_FA4`'s performance and flexibility against baselines on"
    " B200 for the `varlen full` mask.": "在 B200 上对 `FFA_FA4` 的 `varlen full` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA_FA4`'s performance and flexibility against baselines on"
    " B200 for the `varlen causal` mask.": "在 B200 上对 `FFA_FA4` 的 `varlen causal` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA_FA4`'s performance and flexibility against baselines on"
    " B200 for the `sliding window causal` mask.": "在 B200 上对 `FFA_FA4` 的 `sliding window causal` 掩码性能和灵活性进行基线对比基准测试。",
    "Benchmarking `FFA_FA4`'s performance and flexibility against baselines on"
    " B200 for the `varlen block causal` mask.": "在 B200 上对 `FFA_FA4` 的 `varlen block causal` 掩码性能和灵活性进行基线对比基准测试。",
    "Distributed Level": "分布式级别",
    "For distributed-level benchmarking, we evaluate the CP strategies across "
    "4 common mask patterns including `full`, `causal`, `varlen full` and "
    "`varlen causal`, to assess performance and scalability, with the "
    "`cp_size` scaling from `8` up to `64` for both forward and backward "
    "passes.": "对于分布式级基准测试，我们在 4 种常见掩码模式上评估 CP 策略，包括 `full`、`causal`、"
    "`varlen full` 和 `varlen causal`，以评估性能和可扩展性，"
    "`cp_size` 从 `8` 扩展到 `64`，涵盖前向和反向传播。",
    "As for the total sequence length, we scale it linearly together with "
    "`cp_size` and fix the per-device sequence length to reflect the common "
    "training configuration w.r.t. the GPU memory capacity, e.g. `8K` for H100"
    " and `16K` for B200.": "至于总序列长度，我们将其与 `cp_size` 线性扩展，并固定每设备序列长度"
    "以反映与 GPU 显存容量相关的常见训练配置，例如 H100 为 `8K`，B200 为 `16K`。",
    "magi_attn-a2av": "magi_attn-a2av",
    "`MagiAttention` with [`AlltoAll-v`-based group "
    "collectives](./magi_attn.md#alltoall-v-implementation)": "使用[基于 `AlltoAll-v` 的 group collective](./magi_attn.md#alltoall-v-implementation) 的 `MagiAttention`",
    "magi_attn-native": "magi_attn-native",
    "`MagiAttention` with [native group collectives](./magi_attn.md#native-"
    "implementation)": "使用[原生 group collective](./magi_attn.md#native-implementation) 的 `MagiAttention`",
    "ulysses": "ulysses",
    "`Ulysses`": "`Ulysses`",
    "ring_p2p": "ring_p2p",
    "`Ring P2P`": "`Ring P2P`",
    "ring_allgather": "ring_allgather",
    "`Ring AllGather`": "`Ring AllGather`",
    "usp": "usp",
    "`USP`": "`USP`",
    "loongtrain": "loongtrain",
    "`LoongTrain`": "`LoongTrain`",
    "hybrid_dcp": "hybrid_dcp",
    "Megatron `HybridCP`": "Megatron `HybridCP`",
    "For `MagiAttention`, we include two instances with different backends of "
    "group collectives: one using the original `AlltoAll-v`-based "
    "implementation and the other using native kernel based on DeepEP "
    "{cite}`deepep2025_cp_benchmark`, to demonstrate the significant gain from"
    " our new native backend.": "对于 `MagiAttention`，我们包含了两个使用不同 group collective 后端的实例："
    "一个使用原始基于 `AlltoAll-v` 的实现，另一个使用基于 DeepEP "
    "{cite}`deepep2025_cp_benchmark` 的原生内核，以展示新原生后端带来的显著收益。",
    "We've applied some experimental features on `MagiAttention` to further "
    "optimize the performance on benchmarking, which may not be enabled by "
    "default or fully ready for production use yet.": "我们在 `MagiAttention` 上应用了一些实验性功能以进一步优化基准测试性能，"
    "这些功能可能默认未启用或尚未完全准备好用于生产环境。",
    "Therefore, the benchmarking results of `MagiAttention` in this section "
    "are intended to demonstrate the potential performance and scalability of "
    "our design, while the actual performance in production may vary and "
    "require to be tuned specifically.": "因此，本节中 `MagiAttention` 的基准测试结果旨在展示我们设计的潜在性能和可扩展性，"
    "而生产环境中的实际性能可能有所不同并需要针对性调优。",
    "We will continue to optimize and stabilize those features and ease the "
    "adoption in production, and very welcome users to try out those features "
    "and provide feedback to us.": "我们将持续优化和稳定这些功能并简化生产环境的采用，非常欢迎用户试用这些功能并向我们提供反馈。",
    "Distributed-Level Throughput - Full Mask Forward Pass": "分布式级吞吐量 - Full 掩码前向传播",
    "Distributed-Level Throughput - Full Mask Backward Pass": "分布式级吞吐量 - Full 掩码反向传播",
    "Distributed-Level Throughput - Causal Mask Forward Pass": "分布式级吞吐量 - Causal 掩码前向传播",
    "Distributed-Level Throughput - Causal Mask Backward Pass": "分布式级吞吐量 - Causal 掩码反向传播",
    "Distributed-Level Throughput - Varlen Full Mask Forward Pass": "分布式级吞吐量 - Varlen Full 掩码前向传播",
    "Distributed-Level Throughput - Varlen Full Mask Backward Pass": "分布式级吞吐量 - Varlen Full 掩码反向传播",
    "Distributed-Level Throughput - Varlen Causal Mask Forward Pass": "分布式级吞吐量 - Varlen Causal 掩码前向传播",
    "Distributed-Level Throughput - Varlen Causal Mask Backward Pass": "分布式级吞吐量 - Varlen Causal 掩码反向传播",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on H100 for the `full` mask.": "在 H100 上对 `MagiAttention` 的 `full` 掩码性能和可扩展性进行基线对比基准测试。",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on H100 for the `causal` mask.": "在 H100 上对 `MagiAttention` 的 `causal` 掩码性能和可扩展性进行基线对比基准测试。",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on H100 for the `varlen full` mask.": "在 H100 上对 `MagiAttention` 的 `varlen full` 掩码性能和可扩展性进行基线对比基准测试。",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on H100 for the `varlen causal` mask.": "在 H100 上对 `MagiAttention` 的 `varlen causal` 掩码性能和可扩展性进行基线对比基准测试。",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on B200 for the `full` mask.": "在 B200 上对 `MagiAttention` 的 `full` 掩码性能和可扩展性进行基线对比基准测试。",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on B200 for the `causal` mask.": "在 B200 上对 `MagiAttention` 的 `causal` 掩码性能和可扩展性进行基线对比基准测试。",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on B200 for the `varlen full` mask.": "在 B200 上对 `MagiAttention` 的 `varlen full` 掩码性能和可扩展性进行基线对比基准测试。",
    "Benchmarking `MagiAttention`'s performance and scalability against "
    "baselines on B200 for the `varlen causal` mask.": "在 B200 上对 `MagiAttention` 的 `varlen causal` 掩码性能和可扩展性进行基线对比基准测试。",
    "Citation": "引用",
    "If you find MagiAttention useful in your research, please cite:": "如果你在研究中发现 MagiAttention 有用，请引用：",
    "References": "参考文献",
}


def translate_po_file(filepath, translations, name):
    po = polib.pofile(filepath)

    # Remove fuzzy flag from header
    if po.metadata_is_fuzzy:
        po.metadata_is_fuzzy = False

    translated = 0
    skipped = 0
    already_done = 0

    for entry in po:
        if entry.msgstr and entry.msgstr.strip():
            already_done += 1
            continue

        if entry.msgid in translations:
            entry.msgstr = translations[entry.msgid]
            translated += 1
        else:
            skipped += 1

    po.save(filepath)

    print(f"\n{'='*60}")
    print(f"File: {name}")
    print(f"{'='*60}")
    print(f"  Total entries: {len(po)}")
    print(f"  Translated (this run): {translated}")
    print(f"  Already translated: {already_done}")
    print(f"  Skipped (no translation): {skipped}")

    if skipped > 0:
        print("\n  Untranslated entries:")
        for entry in po:
            if not entry.msgstr or not entry.msgstr.strip():
                display = (
                    entry.msgid[:80] + "..." if len(entry.msgid) > 80 else entry.msgid
                )
                print(f"    - {repr(display)}")

    return translated, skipped, already_done


if __name__ == "__main__":
    magi_path = os.path.join(BASE, "magi_attn.po")
    cp_path = os.path.join(BASE, "cp_benchmark.po")

    t1, s1, a1 = translate_po_file(
        magi_path, magi_attn_translations, "blog/magi_attn.po"
    )
    t2, s2, a2 = translate_po_file(
        cp_path, cp_benchmark_translations, "blog/cp_benchmark.po"
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  magi_attn.po: {t1} translated, {s1} skipped, {a1} already done")
    print(f"  cp_benchmark.po: {t2} translated, {s2} skipped, {a2} already done")
    print(f"  Total translated: {t1 + t2}")
