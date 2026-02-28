---
blogpost: true
date: Feb 15, 2026
author: Yunpeng Huang
location: China
category: MagiAttention
tags: Computation-Communication Overlap, Distributed Attention, Context Parallelism
language: English
---

# How to Ensure Kernels Actually Overlapped

## Challenges

While the CPU scheduler controls kernel launch order to favor overlap, the GPU Hyper-Q driver {cite}`bradley2013hyperq` ultimately determines actual execution order non‑deterministically, influenced by transient GPU resource occupancy as well.

Therefore, ensuring reliable overlap between computation and communication kernels is non‑trivial, particularly when compute kernels saturate GPU resources or when communication kernels leverage SM90+ cluster features that constrain concurrency.


## Approaches

### Single Max Connection

Previous work such as `Tensor-Parallelism` (`TP`) enforces GPU kernel pick order the same as the CPU launch order by setting the environment variable `CUDA_DEVICE_MAX_CONNECTIONS=1` {cite}`cuda_device_max_connections_issue`. This guarantees communication kernels are picked before long-running compute kernels, preventing them from being blocked, but it also limits concurrency across independent GPU streams thus degrading end-to-end throughput; therefore this approach is not recommended.

### SM Margin Reservation

A common approach specific for **persistent compute kernels** like [`FFA`](./magi_attn.md#flex-flash-attention) is to explicitly reserve a subset of SMs (the `sm_margin`) so communication kernels can run concurrently with ongoing computation. But choosing `sm_margin` is a trade-off: setting it too large reduces compute throughput, while too small may prevent effective overlap.

Empirically, for [`AlltoAll-v`-based group collectives](./magi_attn.md#alltoall-v-implementation) with `NCCL_CGA_CLUSTER_SIZE={0,1}`, we observe full overlap with `sm_margin` set to only `4~8`, which is smaller than the SM count used by the NCCL kernels. By contrast, when `NCCL_CGA_CLUSTER_SIZE>1` or when using the [native implementation](./magi_attn.md#native-implementation) that leverages SM90+ cluster features and cooperative launch, communication kernels require a substantially larger `sm_margin` to overlap if not picked first — *no less than the number of SMs used by them*.

```{note}
For `FFA` kernels, you have to two methods to set `sm_margin`:

1. If you are using the `flex_flash_attn_func` interface, you can simply pass the optional argument `sm_margin` to it, which will be forwarded to the underlying `FFA` kernels for both forward and backward passes.

2. If you are using the `calc_attn` interface for distributed attention, you can set the environment variable `MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN` and `MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN` to specify the `sm_margin` for underlying forward and backward kernels respectively.
```

### High Priority Stream

Another simple approach specific for **non-persistent compute kernels** like [`FFA_FA4`](./blackwell_ffa_fa4.md) is to assign communication kernels on a high-priority CUDA stream, which will encourage the GPU scheduler to pick them first before compute kernels, or even (probably) preempt running compute kernels during their wave quantization {cite}`nvidia_mm_background_guide_wave_quant` phase. However, this approach is not always effective and varies by architecture - it is less reliable on Hopper while we observe higher success on Blackwell (*further investigation needed in future work*).

```{note}
For `NCCL` communication kernels with PyTorch interfaces, you can simply just set the environment variable `TORCH_NCCL_HIGH_PRIORITY=1` to assign them to a high-priority stream.
```

### Kernel Barrier

:::{todo}
The upcoming section will be released in the near future. Stay tuned!
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

```{bibliography} refs/kernel_overlap.bib
```
