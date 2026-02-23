---
blogpost: true
date: Oct 19, 2025
author: Tao Bu, Qiangang Wang, Bowen Zeng, Hanwen Sun, Yunpeng Huang, Zewei Tao
location: China
category: MagiAttention
tags: Benchmark, Blackwell, Flex-Flash-Attention, Distributed Attention, Context Parallelism
language: English
---

# Long-Context Attention Benchmark

**From Kernel Efficiency to Distributed Scalability**

To evaluate the performance and flexibility of `FFA` kernels and to validate the distributed scalability of `MagiAttention` for ultra-long, heterogeneous-mask training, we benchmark throughput on modern GPUs (e.g., Hopper and Blackwell) for both kernels and distributed attention modules in forward and backward passes across diverse mask patterns (standard and irregular), comparing against state-of-the-art kernel- and distributed-level baselines.


## Benchmark Settings

### Common Configurations

To focus on the impact of sequence length and mask pattern, we fix other data and model configurations using common training settings as shown in the table below.

| settings              | value                                                                            |
|-----------------------|----------------------------------------------------------------------------------|
| attention type        | self-attention where `seqlen = seqlen_q = seqlen_k`                              |
| batch size (b)        | 1                                                                                |
| number of heads (nh)  | nhq:nhk:nhv = 64:8:8 (GQA)                                                       |
| head dimension (hd)   | 128                                                                              |
| dtype                 | `torch.bfloat16`                                                                 |
| window size           | 1024 (for sliding window masks only)                                             |


### Throughput Metrics

Throughput is measured in {math}`\texttt{TFLOPs/s}` for kernel-level benchmarks and {math}`\texttt{TFLOPs/s/GPU}` for distributed benchmarks, calculated based on the total number of floating-point operations ({math}`\texttt{FLOPs}`) involved in the attention computation, for both forward and backward passes respectively. 

The {math}`\texttt{FLOPs}` for each {math}`\mathrm{AttnSlice}` are computed using the formula below, and the total {math}`\texttt{FLOPs}` is the summation of all {math}`\mathrm{AttnSlice}`:

```{math}
:label: flops_calculation

\begin{align}
  \mathrm{FLOPs}^{(fwd)} &= \underbrace{2}_{\text{2 matmul}} \times \underbrace{2}_{\text{2 flops per matmul}} \times\;\; \mathrm{MaskArea}(seqlen, mask\_type) \label{eq:flops_fwd}\\
  &\times batch\_size \times num\_heads\_q \times head\_dim \nonumber\\
  \mathrm{FLOPs}^{(bwd)} &= \underbrace{2.5}_{\text{5 matmul with recomputation}} \times\;\; \mathrm{FLOPs}^{(fwd)} \label{eq:flops_bwd}\\
    where \;\;& \mathrm{MaskArea}(seqlen, full) = seqlen^2, \nonumber\\
     \;\;& \mathrm{MaskArea}(seqlen, causal) = \frac{seqlen(seqlen+1)}{2}, \;\; ...\nonumber
\end{align}
```

And the throughputs are calculated as follows:

```{math}
:label: throughput_calculation

\begin{align}
  \mathrm{TFLOPs/s}^{(wd)} &= \cfrac{\mathrm{FLOPs}^{(wd)}}{\mathrm{Runtime}^{(wd)}}, \quad wd \in \{fwd, bwd\} \\
  \mathrm{TFLOPs/s/GPU}^{(wd)} &= \cfrac{\mathrm{FLOPs}^{(wd)}}{\mathrm{Runtime}^{(wd) }\times cp\_size}, \quad wd \in \{fwd, bwd\}
\end{align}
```

### Data Distribution and Sampling

To better match real-world long-context training scenarios, we select a concrete training dataset (如图, 可删除) and design a dedicated sampling strategy for evaluation. Specifically, we shuffle the entire dataset, pack samples sequentially into data packs, and then shuffle the resulting packs to form the final set of sampling packs. This strategy ensures that, within each data pack, the probability of tokens originating from long and short samples closely matches the distribution of the original dataset.
[图片]

### Kernel Baselines and Mask Patterns

We evaluate the FFA kernel against several mainstream attention kernels across different masking patterns. Different training tasks often correspond to specific mask patterns. Our evaluation is conducted across 12 mask patterns, with 6 regular and 6 heterogeneous masks. (是否需要介绍和图例？)
Hopper GPUs. We evaluate our FFA kernel on Hopper GPUs. For regular masks, the baselines include PyTorch’s fused SDPA, the FlashAttention series (FA2, FA3, FA4), and NVIDIA’s cuDNN fused kernel. For heterogeneous masks, we further include FlexAttention and FlashMask.
Blackwell GPUs. We evaluate our FFA-FA4 kernel on Blackwell GPUs. However, since FA2 and FA3 are specifically designed for the Hopper architecture, we exclude these two baselines from the evaluation on Blackwell GPUs.


### Distributed Baselines and Mask Patterns

We evaluate MagiAttention and several representative distributed attention mechanisms in a distributed setting, focusing on both performance and scalability. The evaluation is carried out across four mask patterns: Full/Causal and Document Full/Causal.
Hopper GPUs.  We evaluate MagiAttention using the FFA kernel. The baselines include Ulysess, Ring P2P, Ring AllGather, USP, LoongTrain, and Megatron HybridCP, with FA3 specified as the attention kernel backend for these distributed attention mechanisms.
Blackwell GPUs. We evaluate MagiAttention using the FFA-FA4 kernel. Similarly, since the FA3 kernel is tailored for the Hopper architecture and FA4 currently does not support backward computation for Document masks, we replace the attention backend of the baselines with cuDNN kernel. Additionally, during the experiments, Megatron HybridCP only supports the FA3 kernel, so we exclude it from the evaluation of Blackwell GPUs.


## Kernel Level

In our experiments, we scale the total sequence length of each data pack from 1K to 64K to evaluate scalability.

### For H100

Benchmark settings: for each mask pattern, we vary the sequence length `seqlen` from `4k,8k,16k,...,` up to `128k` (where `seqlen_q==seqlen_k==seqlen`) while measuring the throughput (in {math}`\texttt{TFLOPs/s}`) for forward and backward passes of different attention kernels. Other configurations are fixed using common training settings (see the table above) to focus on the impact of sequence length and mask pattern. For the varlen packed data, we simply follow the variable sequence length distribution in the open-sourced dataset {cite}`xu2024chatqa` illustrated in the following figure, from which we sample to pack and pad to the required `seqlen`.

```{figure} ../../../assets/magi_attn/exp/varlen_seqlen_distribution.png
:align: center
:width: 800px
:alt: Variable-Length Sequence Distribution

Distribution of sequence lengths in the dataset {cite}`xu2024chatqa`, used to sample and construct the variable-length data for both kernel-level and module-level experiments of MagiAttention.
```

Results are reported in the following figures.

```{figure} ../../../assets/magi_attn/exp/kernel/attn_with_varlen_block_causal_mask/perf_report_all.png
:align: center
:width: 1000px
:alt: FFA Performance - Varlen Block Causal Mask

Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen block causal mask scenarios. (Note that: the {math}`\mathbf{E}` symbol indicates the corresponding distributed attention implementation raises *Cuda Out of Memory* error in that specific configuration, while the {math}`\mathbf{X}` symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.)
```

### For B200


### For B300



## Distributed Level

For distributed experiments, we fix the per-device sequence length at 8K and scale the number of devices from 8 (single node) to 64 (8 nodes), corresponding to a total sequence length per data pack scaling from 64K to 512K. This allows us to evaluate the linear scalability of different distributed attention mechanisms. Additionally, due to Ulysess’s limitations on the number of attention heads, we extend it to MHA during computation to obtain more comparable results.

### For H100

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

### For B200

### For B300

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
