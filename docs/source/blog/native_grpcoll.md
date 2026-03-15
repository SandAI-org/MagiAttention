---
blogpost: true
date: Jan 24, 2026
author: Yunpeng Huang, Zewei Tao
location: China
category: MagiAttention
tags: Group Collective, DeepEP, Collective Communication, Distributed Attention, Context Parallelism
language: English
---

# Support Native Group Collective

## Introduction

With the release of [MagiAttention-v1.1.0](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.1.0), we are excited to announce the support for native group collective CUDA kernels for both intranode and internode communication, based upon the amazing work of DeepEP {cite}`deepep2025_native_grpcoll`.

Compared to the original [`AlltoAll-v` implementation](./magi_attn.md#alltoall-v-implementation), this new approach not only **mitigates the extra pre-/post-processing D2D copies** due to tensor layout transfer by kernel fusion, but also significantly improves efficiency via the optimization of **RDMA transfer de-duplication**, particularly for hierarchical CP groups spanning internode and intranode peers.


## User Interface

### Installation

Installing `MagiAttention` with native group collective support is straightforward. You can follow the standard installation process in the [Installation Guide](../user_guide/install.md#install-magiattention), and the native group collective kernels will be included and built by default, considering your specific GPU architecture and CUDA version automatically.

However, to enable the internode features, you need to ensure that `IBGDA` is properly set up on your bare-metal host machine, which is a prerequisite for utilizing the native group collective kernels when `cp_size > 8` as the communication backend. Please refer to the [Installation Guide](../user_guide/install.md#enable-ibgda-optional) for detailed instructions on how to enable `IBGDA` and verify its functionality.


### Enabling

To enable the native group collective kernels in `MagiAttention`, you can simply set the environment variable `MAGI_ATTENTION_NATIVE_GRPCOLL=1`.

### API

Within `MagiAttention` itself, you don't have to worry about the underlying communication kernels at all, but we will provide a **low-level API for users who want to directly utilize the group collective kernels** for their scenarios involving non-trivial communication patterns.

That's because we believe that, the group collective primitives are **general enough to cover all common communication patterns**, thus can be **widely used and extended beyond the attention mechanism** in modern distributed training scenarios.

:::{todo}
Stay tuned for the upcoming release of the low-level API for group collective kernels, which will be available in the near future.
:::


## Implementation

### Limitations of AlltoAll-v Implementation

Initially, since no existing communication kernels support group collectives, we implemented `GroupCast` and `GroupReduce` on top of `AlltoAll-v` as a prototype, achieving **zero-redundant communication** in forward and backward passes (see {numref}`group_gather_reduce_all2allv_native_grpcoll` below).

```{figure} ../../../assets/magi_attn/comm/group_gather_reduce_all2allv.png
:name: group_gather_reduce_all2allv_native_grpcoll
:align: center
:width: 1000px
:alt: GroupCast/GroupReduce Primitives

Illustration of `GroupCast/GroupReduce` primitives implemented atop `AlltoAll-v` to achieve zero redundancy, shown using the varlen block-causal mask with the last global block. (a) For forward and backward passes, `GroupCast` builds a transfer table for {math}`\mathrm{KV}` send/receive buffers, invokes `AlltoAll-v`, and uses a custom `Range-Gather` kernel for pre-/post-processing. (b) In the backward pass, `GroupReduce` aggregates partial {math}`\mathrm{dKV}` via `AlltoAll-v`, employing `Range-Gather` for pre-processing and `Range-Scatter-Reduce` for post-processing.
```

However, this design introduces **extra pre-/post-processing**: `GroupCast` must re-permute inputs for `AlltoAll-v` and restore outputs (`Range-Gather`), and `GroupReduce` further reduces outputs (`Range-Scatter-Reduce`). Even with optimized Triton kernels, these steps add non‑negligible D2D overhead that can impact end-to-end performance.

Beyond the D2D cost, `AlltoAll-v` permits only a single send/recv buffer pair per peer pair and **does not natively support "cast" semantics**. As a result, sending a tensor from one rank to a subset of peers of size {math}`m` requires allocating {math}`m` separate send buffers and transferring them independently, even though the data are identical. This **duplication** not only leads to **much larger intermediate memory usage**, but also, **causes substantial communication overhead, especially when the CP group spans internode peers over `RDMA`**, where bandwidth is significantly lower than intranode `NVLink`, becoming a **critical bottleneck when `cp_size` scales**.


### Similarities and Differences with EP Dispatch/Combine

Almost at the same time, the DeepEP team released their work {cite}`deepep2025_native_grpcoll` on native kernel implementation of `Dispatch / Combine` communication primitives specific for expert parallelism (EP) scenarios, replacing the traditional `AlltoAll-v`-based implementation with similar pre-/post-processing overhead and RDMA transfer duplication issues.

Inspired by their work, we implemented native `GroupCast / GroupReduce` leveraging the same underlying kernel design of DeepEP's `Dispatch / Combine` respectively and extended it for specific attention communication patterns and beyond.

Specifically, as for `GroupCast`, ...


In this manner, we can fully leverage native group collective kernels, which not only **eliminate the extra D2D copies** by fusing the pre-/post-processing into the communication kernel itself, but also **support native "cast" / "reduce" semantics** by allowing a single send / recv buffer to be sent to / reduced from multiple peers, and substantially **decrease communication overhead over low-bandwidth `RDMA`** by de-duplicating transfers, thus significantly improving communication efficiency and scalability, especially for large CP groups spanning internode peers.


### Other Features and Optimizations



## Experiments

We present representative distributed-level benchmarks below for the most commonly used `varlen causal` mask on both H100 and B200 GPUs, showcasing MagiAttention’s performance and scalability versus other leading CP strategies for both `AlltoAll-v` and native backend, particularly highlighting the performance gain of native group collective kernels when `cp_size > 8` and continues to scale.

For detailed benchmark settings and more benchmarking results, see the separate [blog post](./cp_benchmark.md).

### Kernel Level

:::{todo}
Stay tuned for the upcoming release of the kernel-level benchmarks, which will provide a more fine-grained analysis of the performance improvements brought by the native group collective kernels, including detailed profiling and breakdown of communication overheads.
:::

### Distributed Level

#### H100

```{figure} ../../../assets/magi_attn/exp/distributed/h100/varlen_causal_mask/fwd/flops_report.png
:name: distributed_tflops_per_gpu_h100_varlen_causal_mask_fwd_native_grpcoll
:align: center
:width: 800px
:alt: Distributed-Level Throughput - Varlen Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/distributed/h100/varlen_causal_mask/bwd/flops_report.png
:name: distributed_tflops_per_gpu_h100_varlen_causal_mask_bwd_magi_attn
:align: center
:width: 800px
:alt: Distributed-Level Throughput - Varlen Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `MagiAttention`'s performance and scalability against baselines on H100 for the `varlen causal` mask.
```

#### B200

```{figure} ../../../assets/magi_attn/exp/distributed/b200/varlen_causal_mask/fwd/flops_report.png
:name: distributed_tflops_per_gpu_b200_varlen_causal_mask_fwd_native_grpcoll
:align: center
:width: 800px
:alt: Distributed-Level Throughput - Varlen Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/distributed/b200/varlen_causal_mask/bwd/flops_report.png
:name: distributed_tflops_per_gpu_b200_varlen_causal_mask_bwd_magi_attn
:align: center
:width: 800px
:alt: Distributed-Level Throughput - Varlen Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `MagiAttention`'s performance and scalability against baselines on B200 for the `varlen causal` mask.
```


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

```{bibliography} refs/native_grpcoll.bib
```
