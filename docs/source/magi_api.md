# API Reference

```{eval-rst}
.. py:module:: magi_attention.api
```

```{contents}
:local: true
```

## Flexible Flash Attention

To support computing irregular-shaped masks, we implemented a `flexible_flash_attention` kernel, which can be invoked through the following interface.

```{eval-rst}
.. currentmodule:: magi_attention.functional.flex_flash_attn
```

```{eval-rst}
.. autofunction:: flex_flash_attn_func
```


## Dispatch

### Varlen Dispatch

If you're using a mask defined by `cu_seqlens`, such as a varlen full or varlen causal mask, we've designed a similar interface inspired by FlashAttention's API, making it easy for you to get started quickly. In the function named `magi_attn_varlen_dispatch`, you can obtain the dispatched `x` and `key`.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: magi_attn_varlen_dispatch
```

The logic of the `magi_attn_varlen_dispatch` function mainly consists of two parts: it first calls `magi_attn_varlen_key` to compute a key value, and then uses this key to dispatch the input x. The description of `magi_attn_varlen_key` is as follows.

```{eval-rst}
.. autofunction:: magi_attn_varlen_key
```

### Flexible Dispatch

If the masks you're using are not limited to varlen full or varlen causal, but also include sliding window masks or other more diverse types, we recommend using the following API. By calling `magi_attn_flex_dispatch`, you can obtain the dispatched x and key.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: magi_attn_flex_dispatch
```

Similar to the logic of `magi_attn_varlen_dispatch`, `magi_attn_flex_dispatch` first calls `magi_attn_flex_key` to obtain a key, and then uses this key to dispatch x. The description of `magi_attn_flex_key` is as follows.

```{eval-rst}
.. autofunction:: magi_attn_flex_key
```

### Dispatch Function

If you already have the key, you can call `dispatch` function to get the padded and dispatched local tensor.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: dispatch
```

## Calculate Attention

After dispatch and projection, you should obtain the query, key, and value needed for computation. Using the key obtained from the dispatch function mentioned above, you can perform the computation by calling `calc_attn`, which returns the results out and lse. The description of calc_attn is as follows.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: calc_attn
```

## Undispatch


### Undispatch Function

When you need to recover the complete global tensor from the local tensor like computing the loss, you can call `undispatch` function to unpad and undispatch the local tensor along the seqlen dim.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: undispatch
```


## Utility Functions

### Compute Pad Size and Padding

During the use of MagiAttention, we divide the `total_seqlen` into multiple chunks of size `chunk_size` and evenly distribute them across multiple GPUs. To ensure that `total_seqlen` is divisible by `chunk_size` and that each GPU receives the same number of chunks, we need to pad the original input. You can call `compute_pad_size` to calculate the required padding length, and use this value as a parameter in subsequent functions.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: compute_pad_size
```

After obtaining `pad_size`, you can use `pad_at_dim` and `unpad_at_dim` function to pad and unpad the tensor.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: pad_at_dim
```

```{eval-rst}
.. autofunction:: unpad_at_dim
```

Similarly, you can use `pad_size` along with `total_seqlen` and other related information to apply padding to a (q_ranges, k_ranges, masktypes) tuple using `apply_padding` function. This function fills the padding region with invalid slices.

```{eval-rst}
.. autofunction:: apply_padding
```


### Get Position Ids

Since MagiAttention needs to permute the input tensor along the seqlen dim, some token-aware ops might be affected, such as RoPE. Therefore, we provide a function `get_position_ids` to get the position ids of the input tensor similar to Llama.


```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: get_position_ids
```


### Get Most Recent Key

If you have trouble accessing the meta key, and meanwhile you need to get the most recent key, then you can call `get_most_recent_key` to get it. However, we strongly recommend you to access the key passed through the arguments, in case of unexpected inconsistency.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: get_most_recent_key
```


### Infer Varlen Masks

If you want to use a varlen mask where each segment has the same length, we provide a `infer_varlen_mask_from_batch` function that generates the corresponding cu_seqlens tensors for you.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: infer_varlen_mask_from_batch
```

During the use of varlen mask, it is often necessary to reshape a tensor of shape `[batch_size × seq_len, ...]` into `[batch_size × seq_len, ...]`. To facilitate the use of the above APIs, we provide the `squash_batch_dim` function to merge the tensor dimensions.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: squash_batch_dim
```

### Infer Sliding Window Masks

In the design of `MagiAttention`, we use a (q_range, k_range, masktype) tuple to represent a slice. For sliding window masks, we do not provide a dedicated masktype to represent them directly. However, a sliding window mask can be decomposed into a combination of existing masktypes such as `full`, `causal`, `inv_causal`, and `bi_causal`. If you're unsure how to perform this decomposition, we provide `infer_attn_mask_from_sliding_window` function to handle this process for you.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: infer_attn_mask_from_sliding_window
```
