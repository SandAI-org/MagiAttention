---
blogpost: true
date: Oct 19, 2025
author: Tao Bu, Qiangang Wang, Bowen Zeng, Hanwen Sun, Yunpeng Huang, Zewei Tao
location: China
category: MagiAttention
tags: Benchmark, Distributed Attention, Context Parallelism
language: English
---

# Long-Context Attention Benchmark

**From Kernel Efficiency to Distributed Context Parallelism**

TODO...

## Attention Kernel Benchmark

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

## Distributed Attention Module Benchmark

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

```{bibliography} refs/cp_benchmark.bib
```
