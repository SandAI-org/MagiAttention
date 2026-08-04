---
blogpost: true
date: Jan 21, 2026
author: Jin Li, Zewei Tao, Qiangang Wang, Yunpeng Huang
location: China
category: MagiAttention
tags: Dynamic Load Balance, Sparse Attention, Hybrid Attention, Distributed Attention, Context Parallelism
---

# Dynamic Attention Solver

## Introduction

Context Parallelism (CP) shards sequence activations across distributed devices to overcome memory constraints in long-context training. In MagiAttention, the `static attn solver` has largely solved CP scheduling for standard long-context workloads where masks are static or deterministic prior to the iteration (e.g. causal, causal document, sliding window). By leveraging initial token dispatch before the iteration starts, the static solver ensures compute balance across ranks while restricting data movement to **{math}`\mathrm{KV}`-communication only**, keeping Query/Output ({math}`\mathrm{QO}`) tokens fixed on their host devices.

### Limitations of the Static Solver

To achieve computation load balancing across distributed ranks, the static solver relies on **pre-iteration token dispatch**: it splits the sequence into chunks, estimates the FLOP workload for each chunk, and heuristically dispatches these chunks to CP ranks prior to execution.

Crucially, this approach requires the global attention mask to be **static and fully deterministic before the iteration begins**, so that each chunk's attention FLOP workload can be evaluated in advance to drive the heuristic dispatch algorithm.

However, modern dynamic sparse attention mechanisms break this static assumption:
- **Runtime Dynamic Sparse Attention**: Architectures such as DeepSeek Sparse Attention (DSA) {cite}`deepseekai2025deepseekv32pushingfrontieropen` and Native Sparse Attention (NSA) {cite}`yuan2025nativesparseattentionhardwarealigned` compute sparse {math}`\mathrm{KV}` selections at runtime via lightweight indexers during each layer's forward pass. Because the resulting attention masks depend directly on dynamic activation states and learned parameters, mask structure cannot be known prior to iteration execution.

In these scenarios, pre-iteration token dispatch is completely infeasible, leaving the static solver incapable of balancing computational workloads.

### Core Problem & Design Approach

The core systems challenge in dynamic Context Parallelism is: **under strict activation memory balance constraints (where each rank maintains a uniform sequence shard), how to perform online workload partitioning and compute-communication co-scheduling when token dispatching/reordering is unavailable and attention masks are resolved dynamically at runtime?**

To address this, MagiAttention introduces the **`Dynamic Attention Solver`**. The core idea rests on two key design choices:

1. **Fixed Token Sharding with {math}`\mathrm{QO}`-Comm Enabled**: Without pre-iteration token dispatch, each rank holds a fixed sequence shard. In this setup, communicating KV only strictly fixes each rank's computation workload. This leaves no room for load rebalancing. Therefore, the solver enables communication for both {math}`\mathrm{QO}` and {math}`\mathrm{KV}` (**{math}`\mathrm{QO}`-comm enabled**). Communicating both {math}`\mathrm{QO}` and {math}`\mathrm{KV}` allows any computation workload to be scheduled onto any rank.
2. **Per-Layer Online Solving**: During each attention layer's forward pass, the dynamic solver perceives the current layer's mask structure, calculates the FLOP workload for each attention tile, and dynamically assigns computation tasks to CP ranks. It balances compute FLOPs across devices while minimizing the maximum communication volume among all ranks (i.e., bottleneck minimization), generating execution metadata (`CalcMeta` and `CommMeta`) **in real time**.

In the following sections, we describe the detailed formulation and scheduling algorithms of the `dynamic attn solver` ([Overview](#overview)), present its user interface ([User Interface](#user-interface)), and outline current limitations and future roadmap ([Current Limitations & Future Roadmap](#current-limitations--future-roadmap)).


## Overview





## User Interface




## Current Limitations & Future Roadmap


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

```{bibliography} refs/dynamic_solver.bib
```
