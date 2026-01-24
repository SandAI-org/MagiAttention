## Sparse Attention Toturials

Usage Guide for MagiAttention in Sparse Attention Scenarios.
For now, we only support uniform block sparse(q_block_size and k_block_size are the same for each block) and topk indices as input.

### examples
```python
from magi_attention.functional import block_sparse_attn

out, lse = block_sparse_attn(q, k, v, q_block_size, k_block_size, topk_indices)
```

### Tuning Guide


| ref_block_size(q_block_size, k_block_size) | qhead_per_k_head | swapab | packgqa | sparse_load | tile_shape | tflops |
|--------------------------------------------|------------------|-------|---------|-------------|------------|--------|
| (1, 64)                                    | 8                | True  | True    | False       | (8, 64)    | 56     |
| (8, 64)                                    | 8                | False | True    | False       | (64, 64)   | 355    |
| (16, 64)                                   | 8                | False | True    | False       | (128, 64)  | 468    |
| (64, 64)                                   | 8                | False | True    | False       | (128, 64)  | 480    |
| (128, 128)                                 | 8                | False | False   | False       | (128, 128) | 585    |
| (384, 384)                                 | 8                | False | False   | False       | (128, 128) | 633    |
| (128, 1)                                   | 8                | False | False   | True        | (128, 128) | 388    |


**Explanation:**

- **swapab**: Enable swap_ab mode for optimizing performance when q_block_size is small (<= 16). Cannot be enabled together with sparse_load.

- **packgqa**: Group query heads sharing the same KV head into a single computation block tile. Significantly improves computational efficiency when q_block_size is small. Recommended for GQA scenarios.

- **sparse_load**: Enable sparse load mode for optimizing performance when k_block_size is small (<= 64). Must be used together with auto_range_merge=True. Cannot be enabled together with swapab.

- **tile_shape**: Internal tile size configuration for FFA kernel (M, N). Automatically selected based on ref_block_size, or can be manually specified via sparse_args.



### Blog
TODO: add details of implementation and experiments.
