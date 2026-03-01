---
blogpost: true
date: Feb 7, 2026
author: Yunpeng Huang, Yufeng Yang, Jerry Chen, Yujia Liu, Zewei Tao, Qiangang Wang, Kunlun Li
location: China
category: MagiAttention
tags: Blackwell, Flex-Flash-Attention, Flash-Attention, HSTU Function Representation
language: English
---

# Support Blackwell with FFA_FA4 Backend

## Introduction

Before the release of [MagiAttention-v1.1.0](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.1.0), `MagiAttention` had supported only the Hopper GPUs, since the attention kernel backend [`Flex-Flash-Attention` (`FFA`)](./magi_attn.md#flex-flash-attention) is built upon open-sourced `Flash-Attention 3` (`FA3`) {cite}`shah2024flashattention3fastaccurateattention_blackwell_ffa_fa4`, tailored for SM90 compute capability.

To early support the latest `Blackwell` GPUs, instead of natively extending the `FFA` kernels, which is the future plan to deliver utmost flexibility and performance potential, we have been actively collaborating with MINIMAX peers and NVIDIA team and implemented a temporary attention kernel backend named `FFA_FA4`, built upon a forked [`Flash-Attention 4` (`FA4`)](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support) and equipped with flexible mask support via an `HSTU Function` representation.

This allows us to quickly integrate `Blackwell` support into `MagiAttention` and provide users with the opportunity to leverage the enhanced SM100+ capabilities of `Blackwell` for their attention computations, while we continue to work on the native `FFA` extension for `Blackwell` in the background.


## User Interface

### Installation

Installing `MagiAttention` with `FFA_FA4` currently requires additional steps. See the [Installation Guide](../user_guide/install.md#install-flash-attn-cute-optional) for details; we plan to streamline this process in the future.

### Enabling

To enable `FFA_FA4` backend on `Blackwell` GPUs, users can simply set the environment variable `export MAGI_ATTENTION_FA4_BACKEND=1`.

### Pre-Compilation

Since `FFA_FA4` relies on a forked version of `Flash-Attention 4` based on [Cute PythonDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html), it requires JIT-compilation of the attention kernels for different mask patterns, thus we recommend you to pre-compile the common cases for `FFA_FA4` kernels before production usage to avoid runtime JIT re-compilation overhead. See the [Installation Guide](../user_guide/install.md#precompile-ffa-fa4-kernels-optional) for details.


## Implementation



## Experiments

We present representative kernel-level/distributed-level benchmarks below for the most commonly used `varlen causal` mask on B200 GPUs,highlighting MagiAttention’s performance and scalability with the `FFA_FA4` backend versus state-of-the-art context-parallel (CP) strategies and leading attention kernel baselines.

For detailed benchmark settings and more benchmarking results, see the separate [blog post](./cp_benchmark.md).

```{note}
For `FA4` kernel baseline, we don't report the backward performance since it currently lacks robust support for `varlen` masks, especially on stable version of `2.8.3`, which is also the reason why we use `cuDNN` as the kernel backend for most of the CP baselines.
```

### Kernel Level

```{figure} ../../../assets/magi_attn/exp/kernel/b200/varlen_causal_mask/fwd/flops_report.png
:name: kernel_tflops_b200_varlen_causal_mask_fwd_blackwell_ffa_fa4
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/b200/varlen_causal_mask/bwd/flops_report.png
:name: kernel_tflops_b200_varlen_causal_mask_bwd_blackwell_ffa_fa4
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA_FA4`'s performance and flexibility against baselines on B200 for the `varlen causal` mask.
```

### Distributed Level

```{figure} ../../../assets/magi_attn/exp/distributed/b200/varlen_causal_mask/fwd/flops_report.png
:name: distributed_tflops_per_gpu_b200_varlen_causal_mask_fwd_blackwell_ffa_fa4
:align: center
:width: 800px
:alt: Distributed-Level Throughput - Varlen Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/distributed/b200/varlen_causal_mask/bwd/flops_report.png
:name: distributed_tflops_per_gpu_b200_varlen_causal_mask_bwd_blackwell_ffa_fa4
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

```{bibliography} refs/blackwell_ffa_fa4.bib
```
