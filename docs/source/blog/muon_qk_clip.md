---
blogpost: true
date: Feb 4, 2026
author: Jin Li, Yunpeng Huang
location: China
category: MagiAttention
tags: Muon, QK-Clip, Flex-Flash-Attention, Flash-Attention
language: English
---

# Support Muon QK-Clip

## Introduction

Muon optimizer {cite}`jordan2024muon` based on matrix orthogonalization has demonstrated faster convergence than traditional optimizers such as Adam {cite}`kingma2017adammethodstochasticoptimization,loshchilov2019decoupledweightdecayregularization` in training small-scale language models, and then rapidly adopted and proved to be scalable for large language models by Kimi {cite}`liu2025muonscalablellmtraining`. 

To address training instability when scaling Muon, Kimi has introduced many experimental tricks based on careful theoretical analysis {cite}`liu2025muonscalablellmtraining,kimiteam2026kimik2openagentic`, among which the `QK-Clip` technique introduced in Kimi K2 {cite}`kimiteam2026kimik2openagentic` is a critical component to prevent loss spikes and divergence caused by exploding attention logits.

However, `QK-Clip` requires tracking the maximum attention logits (`max_logits`) across the entire attention matrix {math}`S := QK^\mathrm T`, which is **non-trivial to access** since we don't usually materialize the full attention matrix for memory efficiency, following the standard practice in `Flash Attention` implementations {cite}`dao2022flashattention_muon_qk_clip,dao2023flashattention_muon_qk_clip`, not to mention the additional complexity of distributed training with context parallelism (CP) enabled where the attention matrix might be partitioned across CP ranks.

To natively support (distributed) Muon `QK-Clip`, we have implemented it at both the kernel level in `Flex-Flash-Attention` (`FFA`) and the distributed level in `MagiAttention`, and share ours easy-to-use interface, technical details and simple experiment results in the rest of this blog post.


## User Interface

Previously, the APIs of `flex_flash_attn_func` and `calc_attn` returned a tuple of `(out, lse)`, following `Flash Attention` style. To support (distributed) Muon `QK-Clip` and maybe other features in the future, we generalize the interface to return a tuple of `(out, meta)`, where the `meta` is an instance of dataclass `AttnForwardMeta`, containing the fields that are useful but non-trivial to access out of the core-attention forward pass, such as `lse` and `max_logits`.

As shown in the following code snippets, With this return type, you can access the original `lse` tensor easily as `meta.lse`, and optionally the maximum logits tensor as `meta.max_logits` if you set the argument `return_max_logits=True` (defaults to `False` to return `None`). And we might add more fields to `meta` for new features without breaking existing code.

```{warning}
Enabling `return_max_logits=True` for the first time will trigger a Just-In-Time (JIT) compilation since it is not included in the pre-built kernels of `FFA`, which may cause a one-time delay. Subsequent calls will use the cached kernel and run at full speed.

See more details about JIT compilation in `FFA` in the separate [blog post](./jit_compile.md).
```

* For `flex_flash_attn_func`:

  ```python
  out, meta = flex_flash_attn_func(
      q, 
      k, 
      v,
      q_ranges, 
      k_ranges,
      attn_type_map,
      return_max_logits=True
  )

  lse = meta.lse # shape = (seqlen_q, num_heads_q), dtype=float32
  max_logits = meta.max_logits # shape = (num_heads_q,), dtype=float32, or None if return_max_logits=False
  ```

* For `calc_attn`:

  ```python
  out, meta = calc_attn(
      q,
      k,
      v,
      key,
      return_max_logits=True
  )

  local_lse = meta.lse # shape = (local_seqlen_q, num_heads_q), dtype=float32
  global_max_logits = meta.max_logits # shape = (num_heads_q,), dtype=float32, or None if return_max_logits=False
  ```


## Implementation

The max_logits feature computes the maximum attention score (QK^T) with flexible attention masking for each attention head using a two-level reduction strategy:
- Intra-block reduction: Within each CUDA block, after each worktile computation completes in the epilogue phase, threads perform thread-level reduction to compute the maximum across rows they process. Warp-level reduction then aggregates these values using shuffle operations to obtain a single maximum per warp. The first thread in each warp atomically updates the shared memory buffer smem_max_logits[head_idx] using a lock-free atomic maximum operation. For PackGQA mode where multiple query heads share the same key-value heads, each row's maximum is directly atomically updated to the corresponding Q head entry in shared memory.
- Inter-block reduction: When a block completes processing all its worktiles, all threads synchronize to ensure all intra-block reductions are complete. Threads then read the block-level maximum from shared memory and atomically update the global memory buffer max_logits[head_idx]. Before updating global memory, the block-level maximum is multiplied by softmax_scale to ensure consistency with the scaled attention scores used in softmax computation.
- Memory layout: Each block allocates a shared memory buffer smem_max_logits with size equal to the number of attention heads (currently limited to 128), initialized to negative infinity at kernel launch. The global memory buffer max_logits has one float32 value per query head, also initialized to negative infinity.
- Atomic operation: A lock-free atomic maximum operation using compare-and-swap ensures thread-safe updates without locks, handling concurrent updates from multiple threads within a block and from multiple blocks across different streaming multiprocessors; when updating the maximum value, if another thread has already written a larger value, the current thread can immediately stop without waiting, which further reduces unnecessary contention.


## Experiments

We benchmarked max_logits against baseline flexible flash attention under full, causal, and variable-length settings across sequence lengths up to 16K. On full and causal attention, ffa_max_logits stays within roughly 1–2.5% of the baseline throughput, and within about 2–3.5% in the more challenging variable-length cases. 
[图片]


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

```{bibliography} refs/muon_qk_clip.bib
```
