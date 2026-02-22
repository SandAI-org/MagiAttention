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

While the CPU scheduler controls kernel launch order to favor overlap, the GPU Hyper-Q driver {cite}`bradley2013hyperq` ultimately determines actual execution order non-deterministically. Therefore, with high-occupancy compute kernels like `FFA` that saturate GPU resources, ensuring actual overlap is non-trivial.

Previous work such as `Tensor-Parallelism` (`TP`) enforces GPU kernel pick order the same as the CPU launch order by setting `CUDA_DEVICE_MAX_CONNECTIONS=1` {cite}`cuda_device_max_connections_issue`. This guarantees communication kernels are picked before long-running compute kernels, preventing them from being blocked, but it also limits concurrency across independent GPU streams thus degrading end-to-end throughput; therefore this approach is not recommended.

An alternative specific for *persistent kernels* like `FFA` is to explicitly reserve a subset of SMs (the `sm_margin`) so communication kernels can run concurrently with ongoing computation. But choosing `sm_margin` is a trade-off: setting it too large reduces compute throughput, while too small may prevent effective overlap.

Empirically, for [`AlltoAll-v`-based group collectives](#alltoall-v-implementation) with `NCCL_CGA_CLUSTER_SIZE={0,1}`, we observe full overlap with `sm_margin` set to only `4~8`, which is smaller than the SM count used by the NCCL kernels. By contrast, when `NCCL_CGA_CLUSTER_SIZE>1` or when using the [native implementation](#native-implementation) that leverages SM90+ cluster features and cooperative launch, communication kernels require a substantially larger `sm_margin` to overlap if not picked first — *no less than the number of SMs used by them*.

To address 

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
