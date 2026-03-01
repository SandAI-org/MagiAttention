---
blogpost: true
date: Feb 15, 2026
author: Yunpeng Huang, Zewei Tao
location: China
category: MagiAttention
tags: Computation-Communication Overlap, Distributed Attention, Context Parallelism
language: English
---

# How to Ensure Kernels Actually Overlap

## Challenges

While the CPU scheduler controls the kernel launch order to favor overlapping, the GPU's Hyper-Q driver {cite}`bradley2013hyperq` ultimately dictates the actual execution order. This process is inherently non-deterministic and heavily influenced by transient GPU resource occupancy.

Consequently, **ensuring reliable overlap between computation and communication kernels is non‑trivial**, primarily due to a notorious dilemma in CUDA programming known as **SM Starvation**.

Compute kernels—such as massive `bwd_partial_attn` operations in backpropagation—are inherently "greedy." When dispatched, they can instantaneously saturate all available Streaming Multiprocessors (SMs) on the GPU. Even if a communication kernel (e.g., a network transfer) is dispatched to a parallel CUDA stream immediately afterward, it finds no idle SMs to execute on. The communication kernel is left perpetually waiting in the queue until the heavy computation finishes.

As a result, operations that are asynchronous on the CPU degrade into strict serialization on the GPU hardware—compute first, then communicate—shattering any illusion of overlap. This starvation problem is further exacerbated when communication kernels utilize SM90+ cluster features that inherently constrain concurrency.

## Approaches

### Single Max Connection

Previous frameworks, such as Tensor Parallelism (`TP`), attempt to enforce a strict FIFO GPU kernel scheduling by setting the environment variable `CUDA_DEVICE_MAX_CONNECTIONS=1` {cite}`cuda_device_max_connections_issue`. This guarantees that the GPU driver picks communication kernels in the exact order they were launched by the CPU, preventing them from being blocked by long-running compute kernels. However, this approach severely limits concurrency across independent GPU streams, ultimately degrading end-to-end throughput. Therefore, this method is generally not recommended.

### SM Margin Reservation

A common strategy tailored for **persistent compute kernels**—such as [`FFA`](./magi_attn.md#flex-flash-attention)—is to explicitly reserve a subset of Streaming Multiprocessors (SMs), known as the `sm_margin`. This reservation leaves enough room for communication kernels to execute concurrently alongside ongoing computation. However, configuring the `sm_margin` involves a delicate trade-off: setting it too high sacrifices compute throughput, while setting it too low risks failing to achieve meaningful overlap.

Empirically, for [`AlltoAll-v`-based group collectives](./magi_attn.md#alltoall-v-implementation) with `NCCL_CGA_CLUSTER_SIZE={0,1}`, we observe full overlap with `sm_margin` set to only `4~8`, which is smaller than the SM count used by the NCCL kernels. By contrast, when `NCCL_CGA_CLUSTER_SIZE>1` or when using the [native implementation](./magi_attn.md#native-implementation) that leverages SM90+ cluster features and cooperative launch, communication kernels require a substantially larger `sm_margin` to overlap if not picked first — *no less than the number of SMs used by them*.

```{note}
For `FFA` kernels, you have two methods to set `sm_margin`:

1. If you are using the `flex_flash_attn_func` interface, you can simply pass the optional argument `sm_margin` to it, which will be forwarded to the underlying `FFA` kernels for both forward and backward passes.

2. If you are using the `calc_attn` interface for distributed attention, you can set the environment variables `MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN` and `MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN` to specify the `sm_margin` for the underlying forward and backward kernels, respectively.
```

### High Priority Stream

For **non-persistent compute kernels**, such as [`FFA_FA4`](./blackwell_ffa_fa4.md), another straightforward approach is to assign communication kernels to a high-priority CUDA stream. This encourages the GPU scheduler to dispatch them ahead of compute kernels, or even potentially preempt running compute kernels during their wave quantization phase {cite}`nvidia_mm_background_guide_wave_quant`. However, the effectiveness of this approach varies significantly across GPU architectures. For instance, we have observed it to be less reliable on Hopper architectures, but much more successful on Blackwell (*this warrants further investigation in future work*).

```{note}
For `NCCL` communication kernels with PyTorch interfaces, you can simply set the environment variable `TORCH_NCCL_HIGH_PRIORITY=1` to assign them to a high-priority stream.
```

### Kernel Barrier

To definitively overcome the **SM starvation** problem outlined above, `MagiAttention` introduces a lightweight, device-side fine-grained synchronization primitive: **`KernelBarrier`**.

Unlike traditional `cudaEvent` mechanisms that typically synchronize based on kernel *completion* ("wait for communication to finish"), `KernelBarrier` utilizes fine-grained locks to synchronize based on kernel *launch* ("wait for communication to start"). This subtle yet powerful semantic shift ensures that communication kernels safely secure their required SMs before the heavy compute "beast" is unleashed.

#### 1. Lifecycle and Memory Management
The `KernelBarrier` is elegantly managed on the host side by leveraging PyTorch's RAII mechanics. It allocates a single `Int32` scalar tensor in CUDA memory to act as the counter. This design ensures automatic memory reclamation tied to the PyTorch Tensor's lifecycle, entirely eliminating the need for manual memory freeing ([`kernel_barrier.cu`](https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/csrc/extensions/kernel_barrier.cu)).

#### 2. The `Arrive` Signal in Communication Kernels
When a communication kernel (such as `group_cast` for fetching Remote KV and QO) is dispatched, a POD view of the barrier, `KernelBarrierView`, is passed into it. At the absolute beginning of the communication kernel's execution—strictly limited to the first thread of the first block—it triggers the `arrive()` function ([`kernel_barrier.cuh`](https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/csrc/extensions/kernel_barrier.cuh)):

```cpp
// Executed at the very top of the communication kernel
if constexpr (kHasKernelBarrier) {
  kernel_barrier_view.arrive();
}
```
This atomic increment serves as a clear hardware-level signal: *"I have successfully acquired my SM resources and started running."*

#### 3. The `Wait` Spin-lock in the Compute Stream
Before queuing the massive compute kernel onto the Compute Stream, the Python host code explicitly enforces synchronization by invoking `kernel_barrier_fetch.synchronize()` ([`dist_attn.py`](https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/functional/dist_attn.py)).

Rather than blocking the CPU host, this injects a microscopic `wait_kernel` (comprising exactly 1 Block and 1 Thread) into the compute stream. This tiny kernel executes a volatile `while` loop (a spin-lock), patiently waiting until the target number of communication kernels have signaled their arrival.

#### Perfect Overlap Execution Flow
This combination orchestrates a perfect scheduling dance on the GPU:
1. The `wait_kernel` is scheduled onto the Compute Stream, occupying merely a fraction of a single SM. The greedy compute kernel sits idle in the queue immediately *behind* it.
2. Because the Compute Stream is intentionally stalled without hoarding hardware resources, the GPU scheduler readily dispatches the communication kernels from the Communication Stream to the remaining idle SMs.
3. The communication kernels lock in their required SMs, begin execution, and instantly trigger `arrive()`, incrementing the shared counter.
4. Once the target arrival count is met (e.g., both Remote KV and QO fetch kernels have launched), the `wait_kernel` breaks out of its spin-lock and exits.
5. The heavy compute kernel is immediately released from the queue, instantly saturating and monopolizing all *remaining* idle SMs.

By applying this fine-grained, device-side scheduling trick, MagiAttention safely and deterministically overlaps both streams on the hardware. This effectively eliminates the SM starvation deadlocks that frequently plague deep learning engine optimization.


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
