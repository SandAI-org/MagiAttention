# MagiAttention

<p align="center">
    <a href="https://static.magi.world/static/files/MAGI_1.pdf"><img alt="paper" src="https://img.shields.io/badge/Paper-Magi_1-red"></a>
    <a href="https://SandAI-org.github.io/MagiAttention/"><img alt="blog" src="https://img.shields.io/badge/Blog-MagiAttention-purple"></a>
    <a href="https://github.com/SandAI-org/MagiAttention/releases"><img alt="license" src="https://img.shields.io/badge/Release-v1.0.0-blue"></a>
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
  <img src="./assets/magiattn_overview.png" alt="MaiAttnOverview" width="100%">
</div>


## Latest News 🔥

- [2025/5] We support overlapped q_ranges when all mask types are `FULL` (see [v1.0.1 release note](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.1) for more details), and release the example code to **integrate Megatron with MagiAttention** with several training convergence experiments (see [here](./example/megatron/README.md) for more details).
- [2025/4] 🎉 We release [MagiAttention-v1.0.0](https://github.com/SandAI-org/MagiAttention/tree/v1.0.0) with its [blog](https://SandAI-org.github.io/MagiAttention/): a distributed attention towards linear scalability for ultra-long context, heterogeneous mask training.


# About

MagiAttention is a distributed attention mechanism, or context-parallel (CP) strategy, which aims to support a wide variety of attention mask types with **kernel-level flexibility**, while achieving **linear scalability** with respect to context-parallel (CP) size across a broad range of scenarios, particularly suitable for training tasks involving <u><em>ultra-long, heterogeneous mask</em></u> training like video-generation for [Magi-1](https://github.com/SandAI-org/MAGI-1).

Additionally, it can be easily integrated into prevalent training frameworks such as [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), Pytorch's native [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [transformers](https://github.com/huggingface/transformers), as illustrated in [QuickStart](#quick-start-).

We are committed to continually improving the performance and generality of MagiAttention for the broader research community. Stay tuned for exciting enhancements and new features on the horizon!


## Key Features ✨

To realize linear scalability for distributed attention, we implement and introduce key designs as follows.

For implementation details, more experimental results and future works, please visit our [blog](https://SandAI-org.github.io/MagiAttention/#methodology).

- **Flexible Flash Attention Kernel**. We introduce a generalized formulation for irregular attention mask patterns and implement a flexible flash attention kernel (FFA). It is natively designed for distribution scenarios and provides greater flexibility in handling diverse attention mask types, with performance comparable to [Flash-Attention 3](https://arxiv.org/abs/2407.08608) on Hopper GPUs.
- **Computation Load-Balance**. With a fine-grained sharding strategy, we elaborate an efficient <em>dispatch solver</em> that ensures balanced attention computational loads across each CP rank in every training iteration.
- **Zero-Redundant Communication**. Instead of adopting the common Ring-style P2P communication pattern in CP, we propose two novel communication primitives, <em>GroupCast</em> and <em>GroupReduce</em>, built upon All-to-All-v as a prototypal implementation, enabling zero-redundant communication volume for both forward and backward passes.
- **Adaptive Multi-Stage Overlap**. Leveraging the above enhancements, we further implement a multi-stage compute-communication overlap strategy that effectively hides communication latency and adaptively optimizes overlap through manual or automatic tuning.


## Roadmap ⛏️

- [ ] Optimize `Flex-Flash-Attention` kernels to improve performance and better support sparse attention (*such as [NSA](https://arxiv.org/pdf/2502.11089)*)
- [ ] Support native `GroupCast` and `GroupReduce` kernels and hierarchical communication optimization (*similar to [DeepEP](https://github.com/deepseek-ai/DeepEP)*)
- [ ] Refactor `Distributed Attention Solver` as well as `Flex-Flash-Attention` kernel arguments to support all mask types with all kinds of overlap, and reduce CPU overhead for meta info calculation
- [ ] Improve `Dispatch Solver` to reduce necessary communication volumn while remaining balance in computation (*especially for varlen mask patterns*)
- [ ] Build a comprehensive `CP Benchmark` to better compare the performance of different context parallel strategies under various mask patterns and other training configurations
- [ ] Provide `Documentation` including `API reference` and `User Guide`, with a more detailed technical blog


## Installation ⚙️

### Step1: Activate an NGC pytorch docker container

* release note: [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-02.html#rel-25-02)
* docker image version: nvcr.io/nvidia/pytorch:25.02-py3
* docker run command:

    ```bash
    docker run --name {container_name} -v {host_mnt_root}:{container_mnt_root} -it -d --privileged --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:25.02-py3 /bin/bash
    ```

* docker exec command:

    ```bash
    docker exec -it {container_name} /bin/bash
    ```

### Step2: Install required packages

* command:

    ```bash
    pip install -r requirements.txt
    ```


#### Step3: Install MagiAttention from source

* command:

  ```bash
  git clone https://github.com/SandAI-org/MagiAttention.git

  cd MagiAttention

  git submodule update --init --recursive

  pip install --no-build-isolation .
  ```



## Quick Start 🚀

> [!WARNING]
> MagiAttention currently only supports Hopper GPUs.
> We intend to broaden this support in upcoming updates.


### Basic Usage

We provide basic example code below of how to use `flex_flash_attention` (*non-distributed attention function*) and `magi_attention` (*distributed attention mechanism*), respectively.

For more usage instructions, you can refer to `magi_attention/functional/flex_flash_attn.py` and `magi_attention/api/magi_attn_interface.py`, respectively.

<details>
<summary>Basic Usage</summary>

- **flex_flash_attention**:
  ```python
  import torch
  from magi_attention.api import flex_flash_attn_func

  # --- Define attention config --- #

  total_seqlen = 2048    # 2k tokens
  num_heads_q = 8        # number of attention (query) heads
  num_heads_kv = 2       # number of key/value heads (GQA)
  head_dim = 128         # dimension of each attention head
  dtype = torch.bfloat16 # attention activation / computation dtype (while the reduction dtype is always fp32 for ffa right now)
  device = "cuda"

  # --- Initialize QKV tensor --- #

  q = torch.randn(total_seqlen, num_heads_q, head_dim, dtype=dtype, device=device)
  k = torch.randn(total_seqlen, num_heads_kv, head_dim, dtype=dtype, device=device)
  v = torch.randn(total_seqlen, num_heads_kv, head_dim, dtype=dtype, device=device)

  # --- Initialize FFA meta args for customized attention mask --- #

  # the following customized attention mask looks like (`*` for unmasked, `0` for masked):
  #     - - - - - - - - -> (k)
  #   | * * * * 0 0 0 0
  #   | * * * * 0 0 0 0
  #   | * * * * 0 0 0 0
  #   | * * * * 0 0 0 0
  #   | * * * * * 0 0 0
  #   | * * * * * * 0 0
  #   | * * * * * * * 0
  #   | * * * * * * * *
  #   V
  #  (q)
  q_ranges_tensor = torch.tensor([[0, 1024], [1024, 2048]], dtype=torch.int32, device=device)
  k_ranges_tensor = torch.tensor([[0, 1024], [0, 2048]], dtype=torch.int32, device=device)
  attn_type_map_tensor = torch.tensor([0, 1], dtype=torch.int32, device=device) # full mask for 1st slice, causal mask for 2nd

  max_seqlen_q = 1024 # Max length of all q_ranges (2048 - 1024 = 1024)
  max_seqlen_k = 2048 # Max length of all k_ranges (2048 - 0 = 2048)

  # --- Attention computation --- #

  out, _ = flex_flash_attn_func( # the second return value is `lse` (log-sum-exp), known as the online-softmax correction factor
      q, k, v,
      q_ranges=q_ranges_tensor,
      k_ranges=k_ranges_tensor,
      max_seqlen_q=max_seqlen_q,
      max_seqlen_k=max_seqlen_k,
      attn_type_map=attn_type_map_tensor,
      softmax_scale=None, # defaults to 1/sqrt(head_dim)
  )
  ```

- **magi_attention**: (*NOTE: You should run the following examples in a distributed environment, e.g. using the common `torchrun` script*)
  ```python
  import torch
  import torch.nn as nn
  from magi_attention.api import (
      magi_attn_flex_dispatch, calc_attn, undispatch, # interface functions
      compute_pad_size, # helper functions
  )
  from magi_attention.common import AttnRanges
  from magi_attention.common.enum import AttnMaskType
  from magi_attention.utils import setup_dist_env, clearup_dist_env

  # --- Set up distributed environment --- #

  rank, local_rank, world_size, world_group, device, seed = setup_dist_env()

  # --- Define attention config --- #

  total_seqlen = 32 * 1024   # 32k tokens, if we dispatch it to 8 GPUs, then each GPU holds 4k tokens
  num_heads_q = 48           # number of attention (query) heads
  num_heads_kv = 8           # number of key/value heads (GQA)
  head_dim = 128             # dimension of each attention head
  dtype = torch.bfloat16     # attention activation / computation dtype (while the reduction dtype for partial attention outputs is always fp32 for magi_attention right now)

  # --- Initialize token embedding tensor --- #

  embed_dim = 4096
  x = torch.randn(total_seqlen, embed_dim, device=device, dtype=dtype, requires_grad=True)

  # --- Initialize MagiAttention meta configs for customized attention mask --- #

  # the following customized attention mask is known as `block-causal` mask where `block_size` = 4096 (4k),
  # which looks like (`*` for unmasked, `0` for masked):
  #     - - - - - - - - -> (k)
  #   | * * 0 0 0 0 0 0
  #   | * * 0 0 0 0 0 0
  #   | * * * * 0 0 0 0
  #   | * * * * 0 0 0 0
  #   | * * * * * * 0 0
  #   | * * * * * * 0 0
  #   | * * * * * * * *
  #   | * * * * * * * *
  #   V
  #  (q)
  q_ranges = AttnRanges.from_ranges(
      [
          [0, 4096], # 0~4k
          [4096, 8192], # 4k~8k
          [8192, 12288], # 8k~12k
          [12288, 16384], # 12k~16k
          [16384, 20480], # 16k~20k
          [20480, 24576], # 20k~24k
          [24576, 28672], # 24k~28k
          [28672, 32768], # 28k~32k
      ]
  )
  k_ranges = AttnRanges.from_ranges(
      [
          [0, 4096], # 0~4k
          [0, 8192], # 0~8k
          [0, 12288], # 0~12k
          [0, 16384], # 0~16k
          [0, 20480], # 0~20k
          [0, 24576], # 0~24k
          [0, 28672], # 0~28k
          [0, 32768], # 0~32k
      ]
  )
  attn_mask_type = [AttnMaskType.FULL] * len(q_ranges)
  total_seqlen_q = total_seqlen_k = total_seqlen
  pad_size, _ = compute_pad_size( # pad embeds along seqlen dim for better performance
    total_seqlen_q=total_seqlen_q,
    cp_size=world_size, # assuming we only have 1-dim context parallelism (cp)
    head_dim=head_dim,
  )

  # --- Dispatch token embedding tensor along seqlen dim to multiple ranks --- #

  # NOTE:
  # 1. the dispatched local token embedding may be shuffled along seqlen dim,
  #    so it's safe for token-wise operations such as matmul, layer-norm, etc
  #    while for sample-wise operations like RoPE, you might need to be more careful
  # 2. the `magi_runtime_key` holds some inner meta data as one argument for many other magi_attention APIs,
  #    about which the users may have no bother to care
  local_x, magi_attn_runtime_key = magi_attn_flex_dispatch(
      x,
      q_ranges=q_ranges,
      k_ranges=k_ranges,
      attn_mask_type=attn_mask_type,
      total_seqlen_q=total_seqlen_q,
      total_seqlen_k=total_seqlen_k,
      head_dim=head_dim,
      pad_size=pad_size,
      cp_group=world_group, # assuming we only have 1-dim context parallelism (cp)
  )

  # --- Simulate QKV projection --- #

  q_proj = nn.Linear(embed_dim, num_heads_q * head_dim, dtype=dtype, device=device)
  k_proj = nn.Linear(embed_dim, num_heads_kv * head_dim, dtype=dtype, device=device)
  v_proj = nn.Linear(embed_dim, num_heads_kv * head_dim, dtype=dtype, device=device)

  local_q = q_proj(local_x).view(-1, num_heads_q, head_dim)
  local_k = k_proj(local_x).view(-1, num_heads_kv, head_dim)
  local_v = v_proj(local_x).view(-1, num_heads_kv, head_dim)

  # --- Distributed attention computation --- #

  local_out, _ = calc_attn( # the second return value is `local_lse` (log-sum-exp), known as the online-softmax correction factor
    q=local_q,
    k=local_k,
    v=local_v,
    key=magi_attn_runtime_key,
  )

  # --- Undispatch the output tensor along seqlen dim from multiple ranks and unpad --- #

  # NOTE: the undispatch API may not be used until the moment you need the seqlen dimension to be compelete and ordered,
  # e.g. for either aforementioned sample-wise operations, or loss computation
  total_out = undispatch(
    x=local_out,
    key=magi_attn_runtime_key,
  )

  # --- Clear up distributed environment --- #

  clearup_dist_env()
  ```

</details>


### Example to integrate with FSDP2

We provide an example of how to integrate magi_attention with fsdp2 in `example/torch_native`. You can use `bash run.sh` to run the example.

In this example, we build a llama-1b model and apply fsdp2 with magi_attention as the parallelism strategy.

- `example/torch_native/modeling_llama.py`: build llama model and integrate with magi_attention.
- `example/torch_native/main.py`: main training loop.

</details>


### Example to integrate with Megatron-LM

We create a new repository [Megatron-LM-MagiAttention](https://github.com/SandAI-org/Megatron-LM-MagiAttention/tree/magi_attention), forked from [Megatron-LM v0.11.0](https://github.com/NVIDIA/Megatron-LM/tree/v0.11.0), to provide an example of training the llama-1B model with Megatron-LM + MagiAttention. Furthermore, we conducted an experiment training llama-3-1B model from scratch to verify the convergence of magiattention.

For more information, you can refer to `example/megatron/README.md`.

### Example to integrate with transformers

We provide an example of how to integrate magi_attention with transformers in `example/transformers`. Furthermore, we conducted a continue-training experiment on llama-3-1B model to verify the convergence of magiattention.

For more information, you can refer to `example/transformers/README.md`.

## Documentation

Coming soon ...


## Performance Benchmarks 📊


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

Some Results are reported in the following figures, see more in our [blog](https://SandAI-org.github.io/MagiAttention/#kernel-level).


<div align="center">
  <img src="assets/ffa_exp/attn with fulll mask/perf_report_all.png" alt="full mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for full mask scenarios.</div>
</div>

<div align="center">
  <img src="assets/ffa_exp/attn with causal mask/perf_report_all.png" alt="causal mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for causal mask scenarios.</div>
</div>

<div align="center">
  <img src="assets/ffa_exp/attn with varlen full mask/perf_report_all.png" alt="varlen full mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen full mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>E</b> symbol indicates the corresponding distributed attention implementation raises <em>Cuda Out of Memory</em> error in that specific configuration.</div>
</div>

<div align="center">
  <img src="assets/ffa_exp/attn with varlen causal mask/perf_report_all.png" alt="varlen causal mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen causal mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>E</b> symbol indicates the corresponding distributed attention implementation raises <em>Cuda Out of Memory</em> error in that specific configuration.</div>
</div>

<div align="center">
  <img src="assets/ffa_exp/attn with sw causal mask/perf_report_all.png" alt="sliding-window causal mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for sliding-window causal mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>E</b> symbol indicates the corresponding distributed attention implementation raises <em>Cuda Out of Memory</em> error in that specific configuration.</div>
</div>

<div align="center">
  <img src="assets/ffa_exp/attn with varlen block causal mask/perf_report_all.png" alt="varlen block causal mask ffa" width="100%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking FFA's performance and flexibility against other leading attention kernels for varlen block causal mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>E</b> symbol indicates the corresponding distributed attention implementation raises <em>Cuda Out of Memory</em> error in that specific configuration, while the <b>X</b> symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.</div>
</div>


### Module-Level Scalability


To validate the scalability of MagiAttention, we assess the per-GPU computing power (in $\texttt{TFLOPs/s/GPU}$) of the attention module during both forward and backward propagation, as the sequence length and parallel size increase. This assessment is compared against common CP strategies including [Ring-Attention](https://arxiv.org/abs/2310.01889) and [Ulysses](https://arxiv.org/abs/2309.14509). Due to the complexity of supporting irregular masks for baselines, our experiments are limited to the full mask and varlen full mask scenarios. And the distribution of variable sequence lengths still follow the one in [Kernel-Level Experiments](#kernel-level-performance-and-flexibility).

The experiments are conducted on a large-scale productive GPU cluster (<em>Due to business and confidentiality reasons, specific details about the productive cluster, such as the number and type of GPUs, are withheld.</em>). We scale the total sequence length `seqlen`, the context-parallel size `cp_size`, and the node size `nnodes` together from `seqlen:64k, cp_size:1, nnodes:1`, `seqlen:128k, cp_size:2, nnodes:2`, ..., to `seqlen:3072k (3M), cp_size:48, nnodes:48`.

The tensor-parallel size `tp_size` is fixed at 8, with sequence-parallel enabled. Other data and model configurations for different mask types are the same as in the table in [Kernel-Level Experiments](#kernel-level-performance-and-flexibility).

Therefore, in every training setting, each rank is assigned constantly with `seqlen=64k`, `num_heads_q = 8` and `num_heads_k = 1` for attention propagation, while the remaining activations stays `seqlen=8k`, `num_heads_q = 64` and `num_heads_k = 8` with SP enabled. This setup simulates a common training configuration.

Some of the results are presented in the following figures, see more in our [blog](https://SandAI-org.github.io/MagiAttention/#module-level).

As demonstrated, MagiAttention exhibits linear scalability as the context length and CP size increase, in both full mask and varlen full mask configurations, for both forward and backward passes. In contrast, baseline methods either face strict limitations in scaling up or experience performance degradation with ultra-long contexts, which worsens with varlen mask patterns.


<div align="center">
  <img src="./assets/magi_attention_exp/full_mask_fwd_per_gpu/flops_report.png" alt="full mask magi_attention fwd" width="49%">
  <img src="./assets/magi_attention_exp/full_mask_bwd_per_gpu/flops_report.png" alt="full mask magi_attention bwd" width="49%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking MaiAttention's scalability against other leading CP strategies for full mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>X</b> symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.</div>
</div>

<div align="center">
  <img src="./assets/magi_attention_exp/varlen_full_mask_fwd_per_gpu/flops_report.png" alt="varlen full mask magi_attention fwd" width="49%">
  <img src="./assets/magi_attention_exp/varlen_full_mask_bwd_per_gpu/flops_report.png" alt="varlen full mask magi_attention bwd" width="49%">
  <div style="font-style: italic; margin-top: 5px;">Benchmarking MaiAttention's scalability against other leading CP strategies for varlen full mask scenarios.</div>
  <div style="font-style: italic; margin-top: 5px;">Note that: the <b>X</b> symbol indicates the corresponding distributed attention implementation is not supported in that specific configuration.</div>
</div>


## Contributing 🤝

We welcome and value any contributions and collaborations. Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.


## License ⚖️

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Citation 📝

If you use MagiAttention in your research, please cite:

```bibtex
@misc{magiattention2025,
  title={MagiAttention: A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training},
  author={Zewei, Tao and Yunpeng, Huang},
  year={2025},
  howpublished={\url{https://github.com/SandAI-org/MagiAttention/}},
}
```

## Acknowledgement

We are grateful to the contributors listed below for their valuable contributions during the early stages of MagiAttention.

| Member   | Affiliations         | Email                        | GitHub Account    |
|:-----------|:-------------|:----------------------------|:---------------|
| Zewei Tao    | SandAI       | zeweitao@sand.ai            | littsk         |
| Yunpeng Huang    | SandAI, Nanjing University       | yunpenghuang@sand.ai,hyp@smail.nju.edu.cn       | Strivin0311    |
| Qiangang Wang    | Nanjing University | 522024330081@smail.nju.edu.cn | WT1W           |
| Hanwen Sun   | SandAI, Peking University |  sunhanwen@stu.pku.edu.cn |  hanwen-sun  |
| Tao Bu      | Nanjing University | 502024330002@smail.nju.edu.cn | Big-TRex       |
| WenYang Fang    | Nanjing University | fwy@smail.nju.edu.cn        | kagami4243     |
| Siyuang Yan    | Nanjing University | siyuanyan@smail.nju.edu.cn  | FibonaccciYan  |
| Zixu Jiang     | Nanjing University | 522023330040@smail.nju.edu.cn | 191220042      |
| Dingkun Xu    | Nanjing University | 211220090@smail.nju.edu.cn  | PureDimension  |
| Mingyu Liang    | Nanjing University |   mingyuliang518@gmail.com     | gaomusiki      |
| Jingwei Xu    | Nanjing University | jingweix@nju.edu.cn | paragonlight   |


## Star History

<div align="center">
  <a href="https://star-history.com/#SandAI-org/MagiAttention&Date">
    <img src="https://api.star-history.com/svg?repos=SandAI-org/MagiAttention&type=Date" alt="Star History Chart" style="max-width: 60%; height: auto;"/>
  </a>
</div>
