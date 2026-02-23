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

To evaluate the performance and flexibility of `Flex-Flash-Attention` (`FFA`) kernels and to validate the distributed scalability of `MagiAttention` for ultra-long, heterogeneous-mask training, we benchmark throughput on modern GPUs (e.g., Hopper and Blackwell) for both kernels and distributed attention modules in forward and backward passes across diverse mask patterns (standard and irregular), against state-of-the-art kernel- and distributed-level baselines.


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

To reflect real-world long-context training, we extract the sequence-length distribution from a representative training dataset and use it to construct variable-length inputs for both kernel- and distributed-level experiments (see {numref}`varlen_seqlen_distribution`).

```{figure} ../../../assets/magi_attn/exp/varlen_seqlen_distribution.png
:name: varlen_seqlen_distribution
:align: center
:width: 800px
:alt: Variable-Length Sequence Distribution

Distribution of sequence lengths extracted from a real-world dataset, which is used to sample and construct the variable-length data for both kernel-level and distributed-level experiments.
```

We shuffle the dataset, sequentially pack samples into data packs, then reshuffle those packs to form the final sampling set, where we will fetch a portion of packs for experiments using `varlen` mask patterns. This preserves the original token-length distribution so the probability of tokens from long and short samples within each pack matches the dataset.

To avoid the sampled variable-length data from degenerating into pure `full/causal` masks to affect the evaluation, we limit each sample’s length at most {math}`\frac{1}{4}` of the total sequence length (e.g., no sample exceeds `16K` when measuring with a `64K` total sequence length).

### Kernel Baselines

On Hopper, we evaluate our [`FFA`](./magi_attn.md#flex-flash-attention) kernel against widely used PyTorch’s fused `SDPA` {cite}`pytorch_sdpa_cp_benchmark`, `Flash Attention 2` (`FA2`) {cite}`dao2023flashattention_cp_benchmark`, `Flash Attention 3` (`FA3`) {cite}`shah2024flashattention3_cp_benchmark`, NVIDIA’s `cuDNN` fused attention kernel {cite}`nvidia2024accelerating_cp_benchmark` from [TransformerEngine](https://github.com/NVIDIA/TransformerEngine), as well as PyTorch's new `FlexAttention` {cite}`dong2024flexattentionprogrammingmodel_cp_benchmark` and Baidu's `FlashMask` {cite}`wang2025flashmaskefficientrichmask_cp_benchmark` for baselines on flexible masks.

On Blackwell, we instead evaluate our [`FFA_FA4`](./blackwell_ffa_fa4.md) kernel against the same baselines, substituting `FA2` and `FA3` with `Flash Attention 4` (`FA4`) {cite}`dao2025flashattention_cute_cp_benchmark`, since both `FFA` and `FA3` are tailored for Hopper and `FA2` does not optimize for SM90+ architectures.

### Distributed Baselines

We evaluate `MagiAttention` against state-of-the-art distributed attention mechanisms integrated into [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) as context-parallel (CP) backends, including `Ulysess` {cite}`jacobs2023deepspeed_cp_benchmark`, `Ring P2P` {cite}`liu2023ringattentionblockwisetransformers_cp_benchmark`, `Ring AllGather` {cite}`grattafiori2024llama3herdmodels_cp_benchmark`, `USP` {cite}`fang2024uspunifiedsequenceparallelism_cp_benchmark`, `LoongTrain` {cite}`gu2024loongtrainefficienttraininglongsequence_cp_benchmark`, and Megatron `HybridCP` {cite}`megatron-lm-hybrid-cp-pr-2054_cp_benchmark`. Many of these are discussed in the [Related Work](./magi_attn.md#related-work) section of the main MagiAttention [blog post](./magi_attn.md).

On Hopper, all baselines use the `FA3` kernel as the attention backend to ensure a fair comparison with our `FFA` kernel. 

On Blackwell, since `FA3` targets Hopper and `FA4` currently lacks robust backward support for varlen masks, baselines use the `cuDNN` kernel while we use our `FFA_FA4` backend. Additionally, Megatron `HybridCP` (which requires `FA3`) is omitted from Blackwell evaluations.

## Kernel Level

In our experiments, we evaluate the kernels across 5 common mask patterns including `full`, `causal`, `varlen full`, `varlen causal` and `sliding window causal` with one irregular `varlen block causal` mask used in [Magi-1](https://github.com/SandAI-org/MAGI-1), to assess both performance and flexibility, with the total sequence length varying from `1K,2K,4K,...,` up to `64K` for both forward and backward passes.

Results are reported in the following figures.

```{note}
The {math}`\mathbf{X}` symbol denotes attention kernels unsupported in that configuration due to kernel limitations or error raised (e.g., `Cuda Out of Memory`).
```

### For H100

#### Full Mask

```{figure} ../../../assets/magi_attn/exp/kernel/h100/full_mask/fwd/flops_report.png
:name: kernel_flops_h100_full_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Full Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/h100/full_mask/bwd/flops_report.png
:name: kernel_flops_h100_full_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Full Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA`'s performance against baselines on H100 for the `full` mask.
```

#### Causal Mask

```{figure} ../../../assets/magi_attn/exp/kernel/h100/causal_mask/fwd/flops_report.png
:name: kernel_flops_h100_causal_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/h100/causal_mask/bwd/flops_report.png
:name: kernel_flops_h100_causal_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA`'s performance against baselines on H100 for the `causal` mask.
```

#### Varlen Full Mask

```{figure} ../../../assets/magi_attn/exp/kernel/h100/varlen_full_mask/fwd/flops_report.png
:name: kernel_flops_h100_varlen_full_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Full Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/h100/varlen_full_mask/bwd/flops_report.png
:name: kernel_flops_h100_varlen_full_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Full Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA`'s performance against baselines on H100 for the `varlen full` mask.
```

#### Varlen Causal Mask

```{figure} ../../../assets/magi_attn/exp/kernel/h100/varlen_causal_mask/fwd/flops_report.png
:name: kernel_flops_h100_varlen_causal_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/h100/varlen_causal_mask/bwd/flops_report.png
:name: kernel_flops_h100_varlen_causal_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA`'s performance against baselines on H100 for the `varlen causal` mask.
```

#### Sliding Window Causal Mask

```{figure} ../../../assets/magi_attn/exp/kernel/h100/sw_causal_mask/fwd/flops_report.png
:name: kernel_flops_h100_sw_causal_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Sliding Window Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/h100/sw_causal_mask/bwd/flops_report.png
:name: kernel_flops_h100_sw_causal_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Sliding Window Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA`'s performance against baselines on H100 for the `sliding window causal` mask.
```

#### Varlen Block Causal Mask

```{figure} ../../../assets/magi_attn/exp/kernel/h100/varlen_block_causal_mask/fwd/flops_report.png
:name: kernel_flops_h100_varlen_block_causal_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Block Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/h100/varlen_block_causal_mask/bwd/flops_report.png
:name: kernel_flops_h100_varlen_block_causal_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Block Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA`'s performance against baselines on H100 for the `varlen block causal` mask.
```


### For B200

#### Full Mask

```{figure} ../../../assets/magi_attn/exp/kernel/b200/full_mask/fwd/flops_report.png
:name: kernel_flops_b200_full_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Full Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/b200/full_mask/bwd/flops_report.png
:name: kernel_flops_b200_full_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Full Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA_FA4`'s performance against baselines on B200 for the `full` mask.
```

#### Causal Mask

```{figure} ../../../assets/magi_attn/exp/kernel/b200/causal_mask/fwd/flops_report.png
:name: kernel_flops_b200_causal_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/b200/causal_mask/bwd/flops_report.png
:name: kernel_flops_b200_causal_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA_FA4`'s performance against baselines on B200 for the `causal` mask.
```

#### Varlen Full Mask

```{figure} ../../../assets/magi_attn/exp/kernel/b200/varlen_full_mask/fwd/flops_report.png
:name: kernel_flops_b200_varlen_full_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Full Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/b200/varlen_full_mask/bwd/flops_report.png
:name: kernel_flops_b200_varlen_full_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Full Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA_FA4`'s performance against baselines on B200 for the `varlen full` mask.
```

#### Varlen Causal Mask

```{figure} ../../../assets/magi_attn/exp/kernel/b200/varlen_causal_mask/fwd/flops_report.png
:name: kernel_flops_b200_varlen_causal_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/b200/varlen_causal_mask/bwd/flops_report.png
:name: kernel_flops_b200_varlen_causal_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA_FA4`'s performance against baselines on B200 for the `varlen causal` mask.
```

#### Sliding Window Causal Mask

```{figure} ../../../assets/magi_attn/exp/kernel/b200/sw_causal_mask/fwd/flops_report.png
:name: kernel_flops_b200_sw_causal_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Sliding Window Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/b200/sw_causal_mask/bwd/flops_report.png
:name: kernel_flops_b200_sw_causal_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Sliding Window Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA_FA4`'s performance against baselines on B200 for the `sliding window causal` mask.
```

#### Varlen Block Causal Mask

```{figure} ../../../assets/magi_attn/exp/kernel/b200/varlen_block_causal_mask/fwd/flops_report.png
:name: kernel_flops_b200_varlen_block_causal_mask_fwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Block Causal Mask Forward Pass

(a) Forward Pass
```

```{figure} ../../../assets/magi_attn/exp/kernel/b200/varlen_block_causal_mask/bwd/flops_report.png
:name: kernel_flops_b200_varlen_block_causal_mask_bwd
:align: center
:width: 800px
:alt: Kernel-Level Throughput - Varlen Block Causal Mask Backward Pass

(b) Backward Pass

Benchmarking `FFA_FA4`'s performance against baselines on B200 for the `varlen block causal` mask.
```

## Distributed Level

For distributed experiments, we fix the per-device sequence length at 8K and scale the number of devices from 8 (single node) to 64 (8 nodes), corresponding to a total sequence length per data pack scaling from 64K to 512K. This allows us to evaluate the linear scalability of different distributed attention mechanisms. Additionally, due to Ulysess’s limitations on the number of attention heads, we extend it to MHA during computation to obtain more comparable results.

### For H100

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
