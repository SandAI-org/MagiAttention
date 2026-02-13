# MagiAttention

<p align="center">
    <a href="https://arxiv.org/pdf/2505.13211"><img alt="paper" src="https://img.shields.io/badge/Paper-Magi_1-red"></a>
    <a href="https://SandAI-org.github.io/MagiAttention/docs/"><img alt="docs" src="https://img.shields.io/badge/Docs-MagiAttention-green"></a>
    <a href="https://SandAI-org.github.io/MagiAttention/docs/main/blog/"><img alt="blog" src="https://img.shields.io/badge/Blog-MagiAttention-purple"></a>
    <a href="https://github.com/SandAI-org/MagiAttention/releases"><img alt="license" src="https://img.shields.io/badge/Release-v1.1.0-blue"></a>
</p>

<p align="center">
    <a href="https://sand.ai"><img alt="blog" src="https://img.shields.io/badge/Sand%20AI-Homepage-333333.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjgwMCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMjI3IDIyNS4wODVDMjI3IDIwMi4zMDMgMjI3IDE5MC45MTIgMjMxLjQzNyAxODIuMjExQzIzNS4zMzkgMTc0LjU1NyAyNDEuNTY2IDE2OC4zMzQgMjQ5LjIyNiAxNjQuNDM0QzI1Ny45MzMgMTYwIDI2OS4zMzIgMTYwIDI5Mi4xMjkgMTYwSDUwNy44NzFDNTA5LjI5NSAxNjAgNTEwLjY3NiAxNjAgNTEyLjAxNCAxNjAuMDAxQzUzMi4wODIgMTYwLjAxNyA1NDIuNjExIDE2MC4yNzcgNTUwLjc3NCAxNjQuNDM0QzU1OC40MzQgMTY4LjMzNCA1NjQuNjYxIDE3NC41NTcgNTY4LjU2MyAxODIuMjExQzU3MyAxOTAuOTEyIDU3MyAyMDIuMzAzIDU3MyAyMjUuMDg1VjI1Ni41NThDNTczIDI5MS4zMTkgNTczIDMwOC43IDU2NS4wMzUgMzIzLjI3OUM1NTguNzU2IDMzNC43NzIgNTQzLjU2NSAzNDYuMTEgNTIzLjA3OCAzNTkuNjA1QzUxNC42NzQgMzY1LjE0MSA1MTAuNDcyIDM2Ny45MDkgNTA1LjYzOSAzNjcuOTM2QzUwMC44MDYgMzY3Ljk2NCA0OTYuNTAzIDM2NS4yIDQ4Ny44OTYgMzU5LjY3MUw0ODcuODk2IDM1OS42N0w0NjYuNDY5IDM0NS45MDVDNDU2Ljg3NSAzMzkuNzQyIDQ1Mi4wNzggMzM2LjY2IDQ1Mi4wNzggMzMyLjIxOEM0NTIuMDc4IDMyNy43NzcgNDU2Ljg3NSAzMjQuNjk1IDQ2Ni40NjkgMzE4LjUzMUw1MjYuNzgyIDI3OS43ODVDNTM1LjI5MSAyNzQuMzE5IDU0MC40MzUgMjY0LjkwMyA1NDAuNDM1IDI1NC43OTRDNTQwLjQzNSAyMzguMzg2IDUyNy4xMjUgMjI1LjA4NSA1MTAuNzA1IDIyNS4wODVIMjg5LjI5NUMyNzIuODc1IDIyNS4wODUgMjU5LjU2NSAyMzguMzg2IDI1OS41NjUgMjU0Ljc5NEMyNTkuNTY1IDI2NC45MDMgMjY0LjcwOSAyNzQuMzE5IDI3My4yMTggMjc5Ljc4NUw1MTMuMTggNDMzLjk0MUM1NDIuNDQxIDQ1Mi43MzggNTU3LjA3MSA0NjIuMTM3IDU2NS4wMzUgNDc2LjcxNkM1NzMgNDkxLjI5NCA1NzMgNTA4LjY3NSA1NzMgNTQzLjQzNlY1NzQuOTE1QzU3MyA1OTcuNjk3IDU3MyA2MDkuMDg4IDU2OC41NjMgNjE3Ljc4OUM1NjQuNjYxIDYyNS40NDQgNTU4LjQzNCA2MzEuNjY2IDU1MC43NzQgNjM1LjU2NkM1NDIuMDY3IDY0MCA1MzAuNjY4IDY0MCA1MDcuODcxIDY0MEgyOTIuMTI5QzI2OS4zMzIgNjQwIDI1Ny45MzMgNjQwIDI0OS4yMjYgNjM1LjU2NkMyNDEuNTY2IDYzMS42NjYgMjM1LjMzOSA2MjUuNDQ0IDIzMS40MzcgNjE3Ljc4OUMyMjcgNjA5LjA4OCAyMjcgNTk3LjY5NyAyMjcgNTc0LjkxNVY1NDMuNDM2QzIyNyA1MDguNjc1IDIyNyA0OTEuMjk0IDIzNC45NjUgNDc2LjcxNkMyNDEuMjQ0IDQ2NS4yMjIgMjU2LjQzMyA0NTMuODg2IDI3Ni45MTggNDQwLjM5MkMyODUuMzIyIDQzNC44NTYgMjg5LjUyNSA0MzIuMDg4IDI5NC4zNTcgNDMyLjA2QzI5OS4xOSA0MzIuMDMyIDMwMy40OTQgNDM0Ljc5NyAzMTIuMSA0NDAuMzI2TDMzMy41MjcgNDU0LjA5MUMzNDMuMTIyIDQ2MC4yNTQgMzQ3LjkxOSA0NjMuMzM2IDM0Ny45MTkgNDY3Ljc3OEMzNDcuOTE5IDQ3Mi4yMiAzNDMuMTIyIDQ3NS4zMDEgMzMzLjUyOCA0ODEuNDY1TDMzMy41MjcgNDgxLjQ2NUwyNzMuMjIgNTIwLjIwOEMyNjQuNzA5IDUyNS42NzUgMjU5LjU2NSA1MzUuMDkxIDI1OS41NjUgNTQ1LjIwMkMyNTkuNTY1IDU2MS42MTIgMjcyLjg3NyA1NzQuOTE1IDI4OS4yOTkgNTc0LjkxNUg1MTAuNzAxQzUyNy4xMjMgNTc0LjkxNSA1NDAuNDM1IDU2MS42MTIgNTQwLjQzNSA1NDUuMjAyQzU0MC40MzUgNTM1LjA5MSA1MzUuMjkxIDUyNS42NzUgNTI2Ljc4IDUyMC4yMDhMMjg2LjgyIDM2Ni4wNTNDMjU3LjU2IDM0Ny4yNTYgMjQyLjkyOSAzMzcuODU3IDIzNC45NjUgMzIzLjI3OUMyMjcgMzA4LjcgMjI3IDI5MS4zMTkgMjI3IDI1Ni41NThWMjI1LjA4NVoiIGZpbGw9IiNGRkZGRkYiLz4KPC9zdmc+Cg=="></a>
    <a href="https://magi.sand.ai"><img alt="product" src="https://img.shields.io/badge/Magi-Product-logo.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjgwMCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNNDY5LjAyNyA1MDcuOTUxVjE4MC4zNjRDNDY5LjAyNyAxNjguNDE2IDQ2OS4wMjcgMTYyLjQ0MiA0NjUuMjQ0IDE2MC41MTlDNDYxLjQ2MSAxNTguNTk2IDQ1Ni42NTkgMTYyLjEzIDQ0Ny4wNTYgMTY5LjE5OEwzNjEuMDQ4IDIzMi40OTZDMzQ2LjI5NiAyNDMuMzUzIDMzOC45MjEgMjQ4Ljc4MSAzMzQuOTQ3IDI1Ni42NUMzMzAuOTczIDI2NC41MTggMzMwLjk3MyAyNzMuNjk1IDMzMC45NzMgMjkyLjA0OVY2MTkuNjM2QzMzMC45NzMgNjMxLjU4NCAzMzAuOTczIDYzNy41NTggMzM0Ljc1NiA2MzkuNDgxQzMzOC41MzkgNjQxLjQwNCAzNDMuMzQxIDYzNy44NyAzNTIuOTQ0IDYzMC44MDJMNDM4Ljk1MiA1NjcuNTA0QzQ1My43MDQgNTU2LjY0OCA0NjEuMDggNTUxLjIxOSA0NjUuMDUzIDU0My4zNUM0NjkuMDI3IDUzNS40ODIgNDY5LjAyNyA1MjYuMzA1IDQ2OS4wMjcgNTA3Ljk1MVpNMjg3LjkwNyA0OTQuMTU1VjIyMS45M0MyODcuOTA3IDIxNC4wMDIgMjg3LjkwNyAyMTAuMDM5IDI4NS4zOTQgMjA4Ljc1NEMyODIuODgxIDIwNy40NyAyNzkuNjg0IDIwOS44MDEgMjczLjI5MiAyMTQuNDYyTDIwOS40MjEgMjYxLjAzMkMxOTguMjYyIDI2OS4xNjggMTkyLjY4MyAyNzMuMjM2IDE4OS42NzUgMjc5LjE2QzE4Ni42NjcgMjg1LjA4NCAxODYuNjY3IDI5Mi4wMDMgMTg2LjY2NyAzMDUuODQxVjU3OC4wNjdDMTg2LjY2NyA1ODUuOTk0IDE4Ni42NjcgNTg5Ljk1OCAxODkuMTggNTkxLjI0MkMxOTEuNjkzIDU5Mi41MjYgMTk0Ljg4OSA1OTAuMTk2IDIwMS4yODIgNTg1LjUzNUwyNjUuMTUyIDUzOC45NjVDMjc2LjMxMSA1MzAuODI5IDI4MS44OSA1MjYuNzYxIDI4NC44OTkgNTIwLjgzN0MyODcuOTA3IDUxNC45MTMgMjg3LjkwNyA1MDcuOTk0IDI4Ny45MDcgNDk0LjE1NVpNNjEzLjMzMyAyMjEuOTNWNDk0LjE1NUM2MTMuMzMzIDUwNy45OTQgNjEzLjMzMyA1MTQuOTEzIDYxMC4zMjUgNTIwLjgzN0M2MDcuMzE3IDUyNi43NjEgNjAxLjczOCA1MzAuODI5IDU5MC41NzkgNTM4Ljk2NUw1MjYuNzA4IDU4NS41MzVDNTIwLjMxNiA1OTAuMTk2IDUxNy4xMTkgNTkyLjUyNiA1MTQuNjA2IDU5MS4yNDJDNTEyLjA5MyA1ODkuOTU4IDUxMi4wOTMgNTg1Ljk5NCA1MTIuMDkzIDU3OC4wNjdWMzA1Ljg0MUM1MTIuMDkzIDI5Mi4wMDMgNTEyLjA5MyAyODUuMDg0IDUxNS4xMDIgMjc5LjE2QzUxOC4xMSAyNzMuMjM2IDUyMy42ODkgMjY5LjE2OCA1MzQuODQ4IDI2MS4wMzJMNTk4LjcxOSAyMTQuNDYyQzYwNS4xMTEgMjA5LjgwMSA2MDguMzA3IDIwNy40NyA2MTAuODIgMjA4Ljc1NEM2MTMuMzMzIDIxMC4wMzkgNjEzLjMzMyAyMTQuMDAyIDYxMy4zMzMgMjIxLjkzWiIgZmlsbD0iI0ZGRkZGRiIgc2hhcGUtcmVuZGVyaW5nPSJjcmlzcEVkZ2VzIi8+Cjwvc3ZnPgo=&color=DCBE7E"></a>
    <a href="https://huggingface.co/sand-ai"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Sand AI-ffc107?color=ffc107&logoColor=white"/></a>
     <a href="https://x.com/SandAI_HQ"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-Sand%20AI-white?logo=x&logoColor=white"/></a>
    <a href="https://discord.gg/hgaZ86D7Wv"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-Sand%20AI-7289da?logo=discord&logoColor=white&color=7289da"/></a>
    <a href="https://github.com/SandAI-org/Magi/LICENSE"><img alt="license" src="https://img.shields.io/badge/License-Apache2.0-green?logo=Apache"></a>
</p>


<h4 align="center">
A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training
</h4>

<div align="center">
  <img src="assets/magi_attn/magiattn_overview.png" alt="MagiAttention Overview" width="100%">
</div>


## Latest News 🔥

- [2026/02] 🎉 We release [MagiAttention-v1.1.0](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.1.0) to: (1) add early support for **Blackwell** via a new attention kernel backend `ffa_fa4` using forked [Flash-Attention 4](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support); (2) provide full support for **native group collective kernels for both intranode and internode communication** based upon [DeepEP](https://github.com/deepseek-ai/DeepEP); (3) update the [MagiAttention Blog](https://SandAI-org.github.io/MagiAttention/docs/main/blog/magi_attn.html) with comprehensive [Benchmark Experiments](https://SandAI-org.github.io/MagiAttention/docs/main/blog/magi_attn.html/#experiment) on H100 and B200, demonstrating SOTA performance and linear scalability.

<details>
<summary>2025 News</summary>

- [2025/11] 🚀 We release [MagiAttention-v1.0.5](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.5) with native support for **(distributed) learnable attention sink** mechanism in both Flex-Flash-Attention and MagiAttention, plus a drop-in integration for Flash-Attention via our [Extensions](https://github.com/SandAI-org/MagiAttention/tree/v1.0.5/extensions#flashattention-with-attention-sink), alongside which we provide a [blog post](https://sandai-org.github.io/MagiAttention/blog/ffa_with_sink) that shares our design insights and implementation details. Furthermore, we support **native group collective kernels for intranode communication** based on [DeepEP](https://github.com/deepseek-ai/DeepEP) as an experimental feature.
- [2025/09] 📌 We release [MagiAttention-v1.0.4](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.4) to update the API, **support compilable and jit-built FFA**, optimize the performance for sparse scenarios, reduce the workspace memory usage, and engage some experimental features in progress.
- [2025/07] 🚀 We release [MagiAttention-v1.0.3](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.3) with improvements including [documentation](https://SandAI-org.github.io/MagiAttention/docs/), **support for all four mask types with arbitary overlapping**, deterministic mode, API updates, FFA performance enhancements with bug fixes, optimized dispatch solvers, hierarchical-comm support, and example codes to train Llama-3 1B model with MagiAttention + FSDP / Transformers.
- [2025/06] 📌 We release [MagiAttention-v1.0.2](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.2) to provide the example code to **integrate Megatron-LM with MagiAttention** with several training convergence experiments (*see [here](./examples/megatron/README.md) for more details*), with some bug fixes and a roadmap added.
- [2025/05] 📌 We release [MagiAttention-v1.0.1](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.1) to support overlapped q_ranges when all mask types are `FULL`, with some code cleanup and bug fixes.
- [2025/04] 🎉 We release [MagiAttention-v1.0.0](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.0) with its [blog](https://SandAI-org.github.io/MagiAttention/blog/): a distributed attention towards linear scalability for ultra-long context, heterogeneous mask training.

</details>

# About

MagiAttention is a next‑generation distributed attention mechanism—commonly called context‑parallel (CP)—that offers kernel‑level flexibility for diverse attention‑mask patterns while delivering linear scalability across distributed training setups. It is especially well suited for workloads involving <u><em>ultra-long contexts and heterogeneous masks</em></u>, e.g., autoregressive video generation with [Magi-1](https://github.com/SandAI-org/MAGI-1).

Additionally, it integrates easily with mainstream training frameworks such as [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [Pytorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [HuggingFace Transformers](https://github.com/huggingface/transformers); see [QuickStart](https://sandai-org.github.io/MagiAttention/docs/main/user_guide/quickstart.html) for usage.

We are committed to continually improving the performance and generality of MagiAttention for the broader research community. 

Stay tuned for exciting enhancements and new features on the horizon! Any feedback or contributions are very welcome!


## Key Designs ✨

To achieve linear scalability in distributed attention, we implemented the following key design innovations:

- **Flexible Flash Attention Kernel**. We introduce a generalized attention mask formulation namely `AttnSlice` with a tailed kernel<em>Flex‑Flash‑Attention (FFA)</em>—natively designed to enable compact expression of diverse mask types and make distributed mask partitioning tractable, with performance comparable to [Flash-Attention 3](https://arxiv.org/abs/2407.08608) on Hopper GPUs, and preliminary support for Blackwell via a forked [Flash-Attention 4](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support).
- **Computation Load Balancing**. With a fine-grained chunk‑level sharding strategy, we elaborate an efficient <em>dispatch solver</em> that ensures balanced computational workloads across each CP rank.
- **Zero-Redundant Communication**. Instead of adopting the common Ring-style P2P communication pattern, we ropose two novel communication primitives, <em>GroupCast</em> and <em>GroupReduce</em>, realizing zero-redundant communication volume for both forward and backward passes.
- **Adaptive Multi-Stage Overlap**. Leveraging the above enhancements, we further implement an adaptive multi-stage overlap strategy that schedules computation and communication to effectively hide latency and maximize utilization via either manual or automatic tuning.

If you are interested in the detailed methodology and implementation, please check our [blog](https://SandAI-org.github.io/MagiAttention/docs/main/blog/magi_attn.html#methodology) for more information.


## Documentation 📚

We provide comprehensive documentation [here](https://SandAI-org.github.io/MagiAttention/docs/) for MagiAttention, including installation instructions, API references, and usage examples, tuning guides, technical blogs, performance benchmarks, etc.


## Installation ⚙️

Please refer to our [Installation](https://SandAI-org.github.io/MagiAttention/docs/main/user_guide/install.html) documentation for detailed instructions on how to install MagiAttention from source.


## Quick Start 🚀

Please refer to our [QuickStart](https://SandAI-org.github.io/MagiAttention/docs/main/user_guide/quickstart.html) documentation on how to get started with MagiAttention, with simple code snippets for basic usage and examples for integrating with popular training frameworks like [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [Pytorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [HuggingFace Transformers](https://github.com/huggingface/transformers).


## Extensions 💡

We provide additional [magi_attn_extensions](https://github.com/SandAI-org/MagiAttention/blob/main/extensions/README.md) to offer supplementary utilities based on `magi_attention`, such as [FlashAttention with Attention Sink](https://github.com/SandAI-org/MagiAttention/blob/main/extensions/README.md#flashattention-with-attention-sink).


## Future Work ⛏️

Please refer to our [Future Work](https://SandAI-org.github.io/MagiAttention/docs/main/blog/magi_attn.html#future-work) documentation for upcoming features and improvements.


## Benchmarks 📊

### Kernel-Level Performance and Flexibility

To demonstrate FFA kernels' state-of-the-art performance and flexibility in handling ultra-long, heterogeneous mask training, we measure the computing power (in $\texttt{TFLOPs/s}$) on Hopper GPUs for both forward and backward passes of prevalent attention kernels across standard and irregular mask patterns.

| settings              | value                                                                          |
|-----------------------|-----------------------------------------------------------------------------|
| batch size (b)        | 1                                                                            |
| number of heads (nh)  | nhq:nhk:nhv = 64:8:8 (GQA)                                    |
| head dimension (hd)   | 128                                                                           |
| dtype                 | torch.bfloat16                                                               |
| dropout probability   | 0.0                                                                          |
| window size           | 1024 (for sliding window masks only)                        |

Benchmark settings: for each mask pattern, we vary the sequence length `seqlen` from $4k,8k,16k,...,$ up to $128k$ (`seqlen_q = seqlen_k = seqlen`) while measuring computation power (in $\texttt{TFLOPs/s}$) for forward and backward passes of different attention kernels. Other configurations are fixed using common training settings (see the table above) to focus on the impact of sequence length and mask pattern. For the varlen packed data, we simply follow the variable sequence length distribution in the open-sourced dataset [ChatQA2-Long-SFT-data](https://huggingface.co/datasets/nvidia/ChatQA2-Long-SFT-data), from which we sample to pack and pad to the required `seqlen`.

Some Results are reported in the following figures, see more in our [blog](https://SandAI-org.github.io/MagiAttention/blog/#kernel-level).


<div align="center">
  <img src="assets/magi_attn/exp/kernel/attn_with_full_mask/perf_report_all.png" alt="full mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for full mask scenarios.</div>
</div>

<div align="center">
  <img src="assets/magi_attn/exp/kernel/attn_with_causal_mask/perf_report_all.png" alt="causal mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for causal mask scenarios.</div>
</div>

<div align="center">
  <img src="assets/magi_attn/exp/kernel/attn_with_varlen_full_mask/perf_report_all.png" alt="varlen full mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen full mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>E</b> symbol indicates the corresponding distributed attention implementation raises <em>Cuda Out of Memory</em> error in that specific configuration.</div>
</div>

<div align="center">
  <img src="assets/magi_attn/exp/kernel/attn_with_varlen_causal_mask/perf_report_all.png" alt="varlen causal mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen causal mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>E</b> symbol indicates the corresponding distributed attention implementation raises <em>Cuda Out of Memory</em> error in that specific configuration.</div>
</div>

<div align="center">
  <img src="assets/magi_attn/exp/kernel/attn_with_sw_causal_mask/perf_report_all.png" alt="sliding-window causal mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for sliding-window causal mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>E</b> symbol indicates the corresponding distributed attention implementation raises <em>Cuda Out of Memory</em> error in that specific configuration.</div>
</div>

<div align="center">
  <img src="assets/magi_attn/exp/kernel/attn_with_varlen_block_causal_mask/perf_report_all.png" alt="varlen block causal mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen block causal mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>E</b> symbol indicates the corresponding distributed attention implementation raises <em>Cuda Out of Memory</em> error in that specific configuration, while the <b>X</b> symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.</div>
</div>


### Module-Level Scalability


To validate the scalability of MagiAttention, we assess the per-GPU computing power (in $\texttt{TFLOPs/s/GPU}$) of the attention module during both forward and backward propagation, as the sequence length and parallel size increase. This assessment is compared against common CP strategies including [Ring-Attention](https://arxiv.org/abs/2310.01889) and [Ulysses](https://arxiv.org/abs/2309.14509). Due to the complexity of supporting irregular masks for baselines, our experiments are limited to the full mask and varlen full mask scenarios. And the distribution of variable sequence lengths still follow the one in [Kernel-Level Experiments](#kernel-level-performance-and-flexibility).

The experiments are conducted on a large-scale productive GPU cluster (<em>Due to business and confidentiality reasons, specific details about the productive cluster, such as the number and type of GPUs, are withheld.</em>). We scale the total sequence length `seqlen`, the context-parallel size `cp_size`, and the node size `nnodes` together from `seqlen:64k, cp_size:1, nnodes:1`, `seqlen:128k, cp_size:2, nnodes:2`, ..., to `seqlen:3072k (3M), cp_size:48, nnodes:48`.

The tensor-parallel size `tp_size` is fixed at 8, with sequence-parallel enabled. Other data and model configurations for different mask types are the same as in the table in [Kernel-Level Experiments](#kernel-level-performance-and-flexibility).

Therefore, in every training setting, each rank is assigned constantly with `seqlen=64k`, `num_heads_q = 8` and `num_heads_k = 1` for attention propagation, while the remaining activations stays `seqlen=8k`, `num_heads_q = 64` and `num_heads_k = 8` with SP enabled. This setup simulates a common training configuration.

Some of the results are presented in the following figures, see more in our [blog](https://SandAI-org.github.io/MagiAttention/blog/#module-level).

As demonstrated, MagiAttention exhibits linear scalability as the context length and CP size increase, in both full mask and varlen full mask configurations, for both forward and backward passes. In contrast, baseline methods either face strict limitations in scaling up or experience performance degradation with ultra-long contexts, which worsens with varlen mask patterns.


<div align="center">
  <img src="assets/magi_attn/exp/module/full_mask_fwd_per_gpu/flops_report.png" alt="full mask magi_attention fwd" width="49%">
  <img src="assets/magi_attn/exp/module/full_mask_bwd_per_gpu/flops_report.png" alt="full mask magi_attention bwd" width="49%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking MaiAttention's scalability against other leading CP strategies for full mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>X</b> symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.</div>
</div>

<div align="center">
  <img src="assets/magi_attn/exp/module/varlen_full_mask_fwd_per_gpu/flops_report.png" alt="varlen full mask magi_attention fwd" width="49%">
  <img src="assets/magi_attn/exp/module/varlen_full_mask_bwd_per_gpu/flops_report.png" alt="varlen full mask magi_attention bwd" width="49%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking MaiAttention's scalability against other leading CP strategies for varlen full mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>X</b> symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.</div>
</div>


## Contributing 🤝

We welcome and value any contributions and collaborations. Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.


## License ⚖️

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Citation 📝

If you find MagiAttention useful in your research, please cite:

```bibtex
@misc{magiattention2025,
  title={MagiAttention: A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training},
  author={Zewei, Tao and Yunpeng, Huang},
  year={2025},
  howpublished={\url{https://github.com/SandAI-org/MagiAttention/}},
}
```

## Acknowledgement ❤️

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


## Star History ⭐

<div align="center">
  <a href="https://star-history.com/#SandAI-org/MagiAttention&Date">
    <img src="https://api.star-history.com/svg?repos=SandAI-org/MagiAttention&type=Date" alt="Star History Chart" style="max-width: 60%; height: auto;"/>
  </a>
</div>
