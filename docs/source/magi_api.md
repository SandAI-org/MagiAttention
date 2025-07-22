# MagiAttention API

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


## Compute Pad Size

During the use of MagiAttention, we divide the `total_seqlen` into multiple chunks of size `chunk_size` and evenly distribute them across multiple GPUs. To ensure that `total_seqlen` is divisible by `chunk_size` and that each GPU receives the same number of chunks, we need to pad the original input. You can call `compute_pad_size` to calculate the required padding length, and use this value as a parameter in subsequent functions.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: compute_pad_size
```

## Dispatch

### Dispatch for varlen masks

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

### Dispatch for flexible masks

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

## Calculate Attention

After dispatch and projection, you should obtain the query, key, and value needed for computation. Using the key obtained from the dispatch function mentioned above, you can perform the computation by calling `calc_attn`, which returns the results out and lse. The description of calc_attn is as follows.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: calc_attn
```

## Undispatch

After the attention computation, communication is needed to gather the results back to all GPUs. We provide an API to perform the undispatch process.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: undispatch
```
