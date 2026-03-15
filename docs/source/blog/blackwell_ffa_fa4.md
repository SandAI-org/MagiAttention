---
blogpost: true
date: Feb 7, 2026
author: Yunpeng Huang, Yufeng Yang, Jerry Chen, Yujia Liu, Zewei Tao, Kunlun Li
location: China
category: MagiAttention
tags: Blackwell, Flex-Flash-Attention, Flash-Attention, HSTU Function Representation
language: English
---

# Support Blackwell with FFA_FA4 Backend

## Introduction

Before the release of [MagiAttention-v1.1.0](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.1.0), `MagiAttention` had supported only the Hopper GPUs, since the attention kernel backend [`Flex-Flash-Attention` (`FFA`)](./magi_attn.md#flex-flash-attention) is built upon open-sourced `Flash-Attention 3` (`FA3`) {cite}`shah2024flashattention3fastaccurateattention_blackwell_ffa_fa4`, tailored for SM90 compute capability.

To early support the latest `Blackwell` GPUs, instead of natively extending the `FFA` kernels, which is the future plan to deliver utmost flexibility and performance potential, we have been actively collaborating with MINIMAX peers and NVIDIA team and implemented a temporary attention kernel backend named `FFA_FA4`, built upon a forked [`Flash-Attention 4` (`FA4`)](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support) and equipped with flexible mask support via an [`HSTU Function` representation](#hstu-function-representation).

This allows us to quickly integrate `Blackwell` support into `MagiAttention` and provide users with the opportunity to leverage the enhanced SM100+ capabilities of `Blackwell` for their attention computations, while we continue to work on the native `FFA` extension for `Blackwell` in the background.


## User Interface

### Installation

Installing `MagiAttention` with `FFA_FA4` currently requires additional steps. See the [Installation Guide](../user_guide/install.md#install-flash-attn-cute-optional) for details; we plan to streamline this process in the future.

### Enabling

To enable `FFA_FA4` backend on `Blackwell` GPUs, users can simply set the environment variable `export MAGI_ATTENTION_FA4_BACKEND=1`.

### Pre-Compilation

Since `FFA_FA4` relies on a forked version of `Flash-Attention 4` based on `Cute PythonDSL` {cite}`cutlass-pythonDSL-overview`, it requires JIT-compilation of the attention kernels for different mask patterns, thus we recommend you to pre-compile the common cases for `FFA_FA4` kernels before production usage to avoid runtime JIT re-compilation overhead. See the [Installation Guide](../user_guide/install.md#precompile-ffa-fa4-kernels-optional) for details.


## Implementation

### HSTU Function Representation

In `FFA`, we introduce a novel [AttnSlice Representation](./magi_attn.md#attnslice-representation) of attention masks, which enables efficient kernel execution with distributable and flexible mask support. However, it requires a major modification, including [AttnSlice-level Parallelism](./magi_attn.md#attnslice-level-parallelism-in-ffa), upon `FA3` kernels that are currently only available on Hopper, and cannot be easily and directly applied to `FA4` kernels on Blackwell.

To early support flexible masking on Blackwell, NVIDIA team and us introduce the `HSTU Function` representation, which allows us to handle various mask patterns without extensive changes to the underlying `FA4` kernels.

Specifically, we represent the attention mask as a boolean matrix with the shape `(seqlen_q, seqlen_k)`, where each row of shape `(seqlen_k,)` corresponds to a query token about which key tokens it can attend to. Then for {math}`i`-th row, instead of directly storing the boolean values, we represent it as several segments of consecutive `True` values, and the {math}`j`-th segment's start / end token index formed as a {math}`[start, end)` token range, can be mapped by `HSTU Function` that takes coordinate {math}`(i,2j-1)` / {math}`(i,2j)` as input, where the {math}`0`-th segment's start token index is always `0` so can be omitted in the function representation.

Therefore, each row can be represented by the `HSTU Function` as follows:

```{figure} ../../../assets/magi_attn/ffa/hstu_func_vs_attn_slice.png
:name: hstu_func_vs_attn_slice
:align: center
:width: 800px
:alt: HSTU Function Representation vs AttnSlice Representation

Example of `HSTU Function` representation compared to `AttnSlice` representation for an irregular attention mask pattern.
```


### Flash-Attention 4 Modifications

To fully leverage the `HSTU Function` for flexible masking on Blackwell, we implemented several critical modifications to the base `FA4` kernels, focusing on block-level sparsity generation, memory efficiency, and low-level instruction optimization.

#### Efficient Block Sparsity Generation

Since Flash-Attention operates on a block-wise computation by tiling the entire attention mask, to skip unnecessary computations, we categorize each block into one of three states: **Full** (no masking needed), **Partial** (masking required), or **Empty** (completely masked and can be skipped).

While `Flex-Attention` in PyTorch provides a mechanism to generate block-sparse information, its naive implementation {cite}`PyTorchFlexAttention2025` relies on intermediate tensors that materialize the complete attention mask, which easily results in OOM errors and introduces significant latency for long sequences. To address this, we developed a high-performance **`create_block_mask`** kernel that parses the `HSTU Function` directly {cite}`flash-attn-cute-magi-attn-block-mask`.

This kernel includes both `q2k` (forward) and `k2q` (backward) implementations. In the forward pass, we employ a specific optimization: if an `n`-block is out-of-bounds only in the `q` direction, we treat it as a **Full** block. Since out-of-bound data does not affect the valid computation result and is not written back, this heuristic reduces the number of **Partial** blocks, thereby improving the throughput of the attention kernel.

#### CSR Compression for Sparsity Metadata

The original `FA4` sparsity metadata structure uses fixed-size tensors to store block indices, which scales poorly with sequence length and sparsity. Specifically, it stores `full_block_cnt`, `full_block_idx`, `mask_block_cnt`, and `mask_block_idx`. For highly sparse masks, the `idx` tensors (of shape `[batch, head, m_block_num, n_block_num]`) waste significant memory.

We refactored this into a **Compressed Sparse Row (CSR)** format, consisting of six components:
*   `full_block_cnt` / `mask_block_cnt`
*   `full_block_offset` / `mask_block_offset`
*   `full_block_idx` / `mask_block_idx` (compacted)

By using offsets to locate the valid `n`-block indices for each `m`-block, we only store the indices of non-empty blocks. This transition to CSR significantly reduces the memory footprint of metadata, allowing `FFA_FA4` to scale to ultra-long contexts.

#### Instruction-Level Predication Optimization (R2P)

For the **Partial** blocks, we optimized the masking logic using the **R2P (Register-to-Predicate)** technique in the forward pass. Instead of performing element-wise validity checks, we process 24 elements as a batch. We use an `int32` bitmask where 24 bits represent the validity of 24 consecutive elements, then utilize the `R2P` instruction to bulk-set hardware predicates.

This approach drastically reduces the instruction count:
*   **Standard Implementation:** Requires approximately {math}`128 \times n\_{func}` `ISETP.LE` (Integer Compare and Set Predicate) instructions, 127 `UIADD3` instructions for coordinate calculation, and {math}`128 \times (n\_{func}/2 + 1)` `SEL` (Select) instructions per tile.
*   **R2P Optimization:** Eliminates most comparison and coordinate addition instructions. The number of `SEL` instructions is reduced to 128, and the bitmask generation logic (using `clamp`, shifts, and XOR/OR) scales at only {math}`O(\lceil 128/24 \rceil \times n\_{func}/2)`.

The performance gains from this optimization become increasingly significant as the complexity of the mask (represented by {math}`n\_{func}`) grows.

#### Runtime and Kernel Launch Optimization

Since `FA4` utilizes the `Cute PythonDSL` {cite}`cutlass-pythonDSL-overview`, kernel launching can be expensive due to metadata conversion. We integrated the **`tvm_ffi`** library to streamline the interface between PyTorch and the DSL {cite}`cutlass-pythonDSL-cute-compile-with-tvm-ffi`.

For each unique `compile_key`, we perform an explicit `torch.Tensor` to `DSL.Tensor` conversion only during the first call. Subsequent executions bypass repeated `from_dlpack` calls, significantly reducing the host-side overhead during the launch phase and ensuring that the kernel execution remains the primary bottleneck rather than the Python-to-CUDA bridge.


### Integration into MagiAttention

To convert the mask representation of {math}`\mathrm{AttnSlice}` into the `HSTU Function` format required by `FFA_FA4`, we implement an efficient CUDA kernel **`magi_to_hstu`** {cite}`flash-attn-cute-magi-attn-magi_to_hstu`, which allows us to leverage the new backend with minimal changes and seamlessly integrate into the existing `MagiAttention` framework.

It is note-worthy that `FFA_FA4` takes the maximum number of segments (i.e., {math}`n\_func`) as a template parameter, so instead of using a large fixed value to cover all cases, which is both unrealistic and detrimental to kernel performance, we dynamically choose the minimum `n_func` based on the actual mask pattern within the **`magi_to_hstu`** kernel, thus achieving better performance and memory efficiency.

However, this introduces the risk of runtime JIT compilation for unseen `n_func` values, which is why we recommend custom [pre-compilation](#pre-compilation) for specific user cases.

```{note}
While `FFA_FA4` provides a pathway to support flexible masking on Blackwell, it is a temporary solution. The long-term plan is to extend the native `FFA` kernels to Blackwell, which will unlock the full potential of the architecture and provide even better performance and flexibility under more complex scenarios like (distributed) sparse attention.
```


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
