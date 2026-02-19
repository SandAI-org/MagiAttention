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

In large-scale language model pre-training, maintaining numerical stability is critical to avoiding training instabilities and loss spikes.  Following the methodology introduced in the Kimi K2 technical report (https://arxiv.org/pdf/2507.20534), we have integrated the calculation of max_logits within our distributed attention framework to support QK-Clip.
As highlighted in the paper, "to address the training instability while enjoying advanced token efficiency, we propose the QK-Clip technique...  [which] prevents the logits in the attention mechanism from exploding."  By tracking the global maximum logit across distributed sequence partitions, the framework can effectively clip or scale attention scores before the softmax operation.  This implementation ensures that the model can be trained on trillions of tokens without catastrophic divergence, providing the foundational stability required for long-context distributed training.


## User Interface
Previously, the attention functions returned a tuple of (out, lse). The interface has been updated to return (out, meta), where meta is an instance of AttnForwardMeta.To enable this feature, simply set return_max_logits=True. You can then access the result from the returned metadata:

For flex_flash_attn_func or dist_attn_runtime_mgr.calc_attn
out, meta = flex_flash_attn_func(
    q, k, v, q_ranges, k_ranges, 
    return_max_logits=True
)


Access max_logits from meta
max_logits = meta.max_logits  # Shape: (num_heads_q)
Important Note
This feature is not included in the pre-built kernels. Enabling return_max_logits=True for the first time will trigger a Just-In-Time (JIT) compilation, which may cause a one-time delay. Subsequent calls will use the cached kernel and run at full speed.


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
