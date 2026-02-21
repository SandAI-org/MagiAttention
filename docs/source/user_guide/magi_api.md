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

If you're using a mask defined by `cu_seqlens`, such as a varlen full or varlen causal mask, we've designed a similar interface `magi_attn_varlen_key` inspired by FlashAttention's API as follows, making it easy for you to get started quickly.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: magi_attn_varlen_key
```

If you want to apply more than one masks within the same training pass, you can use `make_varlen_key_for_new_mask_after_dispatch` to make a new key for the new mask, given the mask arguments specific for varlen mask in flash-attn-varlen style and the existing key used for dispatch.

Then the new mask will reuse the same dispatch solution as the mask used for dispatch, but with different meta arguments for computation and communication.

```{eval-rst}
.. autofunction:: make_varlen_key_for_new_mask_after_dispatch
```


### Flexible Dispatch

If the masks you're using are not limited to varlen full or varlen causal, but also include sliding window masks or other more diverse types, we recommend using the `magi_attn_flex_key` as follows.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: magi_attn_flex_key
```

If you want to apply more than one varlen masks within the same training pass, you can use `make_flex_key_for_new_mask_after_dispatch` to make a new key for the new mask, given the mask arguments and the existing key used for dispatch.

Then the new mask will reuse the same dispatch solution as the mask used for dispatch, but with different meta arguments for computation and communication.

```{eval-rst}
.. autofunction:: make_flex_key_for_new_mask_after_dispatch
```

### Dispatch Function

When you get the dist attn runtime key, you can call `dispatch` function to dispatch the global input tensor(s) to get the padded local tensor(s) along the seqlen dim.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: dispatch
```

## Calculate Attention

After dispatch and QKV projection, you should obtain the local query, key, and value. Then you can calculate the distributed attention by calling `calc_attn` with the dist attn runtime key to get the local attention output tensor.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: calc_attn
```

## Undispatch


### Undispatch Function

When you need to recover the global output tensor(s) from the local one(s), to compute the loss or some reason else, you can call `undispatch` function to undispatch the padded local ouput tensor(s) back to the unpadded global tensor along the seqlen dim.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: undispatch
```


## Utility Functions

### Compute Pad Size and Padding

During the use of MagiAttention, we divide the `total_seqlen` into multiple chunks of size `chunk_size` and evenly distribute them across multiple GPUs. To ensure that `total_seqlen` is divisible by `chunk_size` and that each GPU receives the same number of chunks, we need to pad the original input.

You can call `compute_pad_size` to calculate the required padding length, and use this value as a parameter in subsequent functions.

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

Similarly, you can use `pad_size` along with `total_seqlen` and other related information to apply padding to a `(q_ranges, k_ranges, mask_types)` tuple using `apply_padding` function.

This function fills the padding region with invalid slices.

```{eval-rst}
.. autofunction:: apply_padding
```


### Get Position Ids

Since MagiAttention needs to permute the input tensor along the seqlen dim, some token-aware ops might be affected, such as RoPE.

Therefore, we provide a function `get_position_ids` to get the position ids of the input tensor similar to Llama.


```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: get_position_ids
```


### Get Most Recent Key

If you have trouble accessing the meta key, and meanwhile you need to get the most recent key for certain `cp_group`, then you can call `get_most_recent_key` to get it by specifying the `cp_group`.

However, we strongly recommend you to access the key passed through the arguments, in case of unexpected inconsistency.

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

During the use of varlen mask, it is often necessary to reshape a tensor of shape `[batch_size × seq_len, ...]` into `[batch_size × seq_len, ...]`.

To facilitate the use of the above APIs, we provide the `squash_batch_dim` function to merge the tensor dimensions.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: squash_batch_dim
```

Moreover, if you have already computed the ``cu_seqlens`` tensor and want to generate a varlen mask based on it, we provide the ``infer_attn_mask_from_cu_seqlens`` function. This function can create three types of masks—varlen full, varlen causal, and varlen sliding window—according to `cu_seqlens`, `causal`, and `window_size`, and returns the result in the form of a `(q_ranges, k_ranges, mask_types, total_seqlen_q, total_seqlen_k)`.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: infer_attn_mask_from_cu_seqlens
```

### Infer Sliding Window Masks

In the design of `MagiAttention`, we use a (q_range, k_range, masktype) tuple to represent a slice.

For sliding window masks, we do not provide a dedicated masktype to represent them directly.

However, a sliding window mask can be decomposed into a combination of existing masktypes such as `full`, `causal`, `inv_causal`, and `bi_causal`.

If you're unsure how to perform this decomposition, we provide `infer_attn_mask_from_sliding_window` function to handle this process for you.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: infer_attn_mask_from_sliding_window
```
