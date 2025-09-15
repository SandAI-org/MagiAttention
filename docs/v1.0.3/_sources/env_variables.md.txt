# Environment Variables

In **MagiAttention**, many features need to be configured through environment variables. Below are some environment variables that can be set, along with their descriptions.


## For Performance

**MAGI_ATTENTION_HIERARCHICAL_COMM**

Toggling `MAGI_ATTENTION_HIERARCHICAL_COMM` env variable to `1` to enable hierarchical group-collective comm within 2-dim cp group (inter_node group + intra_node group).

```{note}
This is for now a temporary solution to reduce the redundant inter-node communication and might be removed or updated in the future.
```

**MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN**

The sm margin number of ffa forward kernel saved for comm kernels.

**MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN**

The sm margin number of ffa backward kernel saved for comm kernels.


**MAGI_ATTENTION_FFA_FORWARD_INPLACE_CORRECT**

Toggling this env variable to `1` can enable inplace-correct for out and lse in ffa forward to avoid the storage of partial results and the memory-bound `result_correction` as a forward post process.

```{note}
This feature will be enabled by default as long as it's stable (i.e. no effect on accuracy or performance).
```

**MAGI_ATTENTION_FFA_BACKWARD_HIGH_PRECISION_REDUCE**

Toggling this env variable to `1` can enable high-precision (fp32) reduce for dkv among ranks in ffa backward to increase the precision at the cost of double comm overhead。

```{note}
Inside the ffa backward kernel, we always use high-precision (fp32) accumulation for partial dkv.
However, by default we will downcast it to kv dtype before reducing among ranks to decrease comm overhead.
```


## For Debug

**MAGI_ATTENTION_SANITY_CHECK**

Toggling `MAGI_ATTENTION_SANITY_CHECK` env variable to `1` can enable many sanity check codes inside magi_attention.

```{note}
This is only supposed to be used for testing or debugging, since the extra sanity-check overhead might be non-negligible.
```

**MAGI_ATTENTION_SDPA_BACKEND**

Toggling `MAGI_ATTENTION_SDPA_BACKEND` env variable to `1` can switch the attn kernel backend from ffa to sdpa-math, to support higher precision like `fp32`, `fp64`.

```{note}
This is only supposed to be used for testing or debugging, since the performance is not acceptable.
```

**MAGI_ATTENTION_DETERMINISTIC_MODE**

Toggle `MAGI_ATTENTION_DETERMINISTIC_MODE` env variable to `1` to enable deterministic mode to use deterministic algorithms for all magi_attention kernels.
