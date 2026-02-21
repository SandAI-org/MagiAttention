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

The Muon optimizer {cite}`jordan2024muon`, which leverages matrix orthogonalization, has shown faster convergence than traditional optimizers such as Adam {cite}`kingma2017adammethodstochasticoptimization,loshchilov2019decoupledweightdecayregularization` on smaller language models and was subsequently demonstrated to scale to large models by Kimi {cite}`liu2025muonscalablellmtraining`. 

To mitigate training instability when scaling Muon, Kimi proposed several theoretically motivated techniques {cite}`liu2025muonscalablellmtraining,kimiteam2026kimik2openagentic`; among them, the `QK-Clip` method from Kimi K2 {cite}`kimiteam2026kimik2openagentic` is essential for preventing loss spikes and divergence caused by exploding attention logits.

`QK-Clip` requires tracking the maximum attention logits (`max_logits`) over the entire attention matrix {math}`S := QK^\mathrm T`, which is non-trivial because implementations based on `Flash Attention` typically avoid materializing the full attention matrix for memory efficiency {cite}`dao2022flashattention_muon_qk_clip,dao2023flashattention_muon_qk_clip`. This challenge is compounded in distributed setups with context parallelism (CP), where the attention matrix may be partitioned across CP ranks.

We address these challenges by adding native support for (distributed) Muon `QK-Clip` at both the kernel level in `Flex-Flash-Attention` (`FFA`) and the distributed level in `MagiAttention`, and present a concise API, implementation details, and empirical results below.


## User Interface

Previously, the APIs of `flex_flash_attn_func` and `calc_attn` returned a tuple of `(out, lse)`, following `Flash Attention` style. To support (distributed) Muon `QK-Clip` and maybe other features in the future, we generalize the interface to return a tuple of `(out, meta)`, where the `meta` is an instance of dataclass `AttnForwardMeta`, containing the fields that are useful but non-trivial to access out of the core-attention forward pass, such as `lse` and `max_logits`.

As shown in the following code snippets, With this return type, you can access the original `lse` tensor easily as `meta.lse`, and optionally the maximum logits tensor as `meta.max_logits` if you set the argument `return_max_logits=True` (defaults to `False` to return `None`). This `meta`-based design allows adding new fields for new features without breaking existing code.

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

### Kernel-Level Implementation in FFA

To compute the maximum attention logits:

```{math}
\mathrm{max\_logits} := \max\limits_{i\in [0,sq),j\in [0,sk)} \{S_{i,j}\}, \quad S := QK^\mathrm T \cdot \mathrm{softmax\_scale} + \mathrm{bias}
```

with flexible attention masking for each attention head in the `FFA` forward kernel, we adopt a two-level reduction strategy:

- **Intra-block Reduction**: Within each CUDA block, after each worktile epilogue, threads perform a thread-level reduction to compute the `max_logits` over their assigned rows. Warp-level shuffle reduction aggregates per-warp maxima, and the first lane in each warp atomically updates the shared buffer `smem_max_logits[head_q_idx]` using a lock-free atomic-max. In `PackGQA` mode, where multiple query heads share key-value heads, each row’s max is atomically written directly to the corresponding `smem_max_logits[head_q_idx]`.

- **Inter-block Reduction**: Once a block has processed all its worktiles, threads synchronize to ensure intra-block reductions are complete, read the block-reduced `max_logits` from shared memory, multiply it by `softmax_scale` for consistency with scaled attention scores, and atomically update the global buffer `gmem_max_logits[head_q_idx]`.

- **Memory Allocation**: Each block allocates a shared buffer `smem_max_logits` sized to the number of attention heads (<em>currently limited up to `128`</em>), initialized to `-inf`. The global buffer `gmem_max_logits` has shape `(num_heads_q,)`, dtype `float32`, and is also initialized to `-inf`.

- **Atomic Maximum**: Updates use a lock-free compare-and-swap atomic-max to ensure thread-safe, lockless updates across threads and blocks. If a larger value is already present, the updating thread can exit immediately, minimizing contention.


### Distributed-Level Implementation in MagiAttention

To compute the global maximum attention logits from the partial results computed on each CP rank for each stage:

```{math}
\mathrm{global\_max\_logits} := \max\limits_{r\in [0,cp\_size),k\in [0,num\_stages)} \{\mathrm{partial\_max\_logits}_{r,k}\}
```

we also need to adopt a two-level reduction strategy:

- **Inter-stage Reduction**: On each CP rank, allocate a per-rank accumulative buffer `partial_max_logits` and pass it into the `FFA` forward kernel for every stage to accumulate stage-level `max_logits` per attention head.

- **Inter-rank Reduction**: After stage accumulation, perform an `AllReduce` with `reduce_op=max` across CP ranks to obtain the final `global_max_logits`, and write it into `meta.max_logits` in the `calc_attn` return value for user access.


## Experiments

We benchmark `FFA` with `max_logits` enabled against the original implementation (without it) across `full`, `causal`, and `varlen full/causal` mask patterns for sequence lengths up to `16k`.

As shown in the {numref}`muon_qk_clip_max_logits` below, throughput with `max_logits` remains close to the baseline: roughly `1%~2.5%` overhead for `full` and `causal` masks, and about `2%~3.5%` for the more challenging `varlen full/causal` cases, indicating a **negligible runtime impact** from computing and returning `max_logits`.

```{figure} ../../../assets/magi_attn/ffa/muon_qk_clip_max_logits.png
:name: muon_qk_clip_max_logits
:align: center
:width: 800px
:alt: Muon QK-Clip Max Logits Performance in FFA

Benchmark results of `FFA` with `max_logits` enabled against the original implementation (without it) across `full`, `causal`, and `varlen full/causal` mask patterns for sequence lengths up to `16k`.
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

```{bibliography} refs/muon_qk_clip.bib
```
