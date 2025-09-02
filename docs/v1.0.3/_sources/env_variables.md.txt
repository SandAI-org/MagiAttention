# Environment Variables

In **MagiAttention**, many features need to be configured through environment variables. Below are some environment variables that can be set, along with their descriptions.


## For Performance

**MAGI_ATTENTION_HIERARCHICAL_COMM**

Toggle this env variable to `1` to enable hierarchical group-collective comm within 2-dim cp group (inter_node group + intra_node group). The default value is `0`.

```{note}
This is for now a temporary solution to reduce the redundant inter-node communication and might be removed or updated in the future.
```

**MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN**

Set the value of this env variable to control the number of SMs of the ffa forward kernel saved for comm kernels. The default value is `4` if `CUDA_DEVICE_MAX_CONNECTIONS` > `1`, otherwise `0`.

**MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN**

Set the value of this env variable to control the number of SMs of the ffa backward kernel saved for comm kernels. The default value is `4` if `CUDA_DEVICE_MAX_CONNECTIONS` > `1`, otherwise `0`.


**MAGI_ATTENTION_FFA_FORWARD_INPLACE_CORRECT**

Toggle this env variable to `1` can enable inplace-correct for out and lse in ffa forward to avoid the storage of partial results and the memory-bound `correct_attn_fwd_result` as a forward post process. The default value is `0`.

```{note}
This feature will be enabled by default as long as it's stable (i.e. no effect on accuracy or performance).
```

**MAGI_ATTENTION_FFA_BACKWARD_HIGH_PRECISION_REDUCE**

Toggle this env variable to `1` can enable high-precision (fp32) reduce for dkv among ranks in ffa backward to increase the precision at the cost of double comm overhead. The default value is `0`.

```{note}
Inside the ffa backward kernel, we always use high-precision (fp32) accumulation for partial dkv.
However, by default we will downcast it to kv dtype before reducing among ranks to decrease comm overhead.
```

**MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE**

Set the value of this env variable to control the size of `dist_attn_runtime_dict`. The default value is `100`.

## For Debug

**MAGI_ATTENTION_SANITY_CHECK**

Toggle this env variable to `1` can enable many sanity check codes inside magi_attention. The default value is `0`.

```{note}
This is only supposed to be used for testing or debugging, since the extra sanity-check overhead might be non-negligible.
```

**MAGI_ATTENTION_SDPA_BACKEND**

Toggle this env variable to `1` can switch the attn kernel backend from ffa to sdpa-math, to support higher precision like `fp32` or `fp64`. The default value is `0`.

```{note}
This is only supposed to be used for testing or debugging, since the performance is not acceptable.
```

**MAGI_ATTENTION_DETERMINISTIC_MODE**

Toggle this env variable to `1` to enable deterministic mode to use deterministic algorithms for all magi_attention kernels. The default value is `0`.


## For Build

**MAGI_ATTENTION_PREBUILD_FFA**

Toggle this env variable to `1` can enable pre-build ffa kernels for some common options with `ref_block_size=None` and leave others built in jit mode. The default value is `1`.


**MAGI_ATTENTION_PREBUILD_FFA_JOBS**

Set the value of this env variable to control the number of jobs used to pre-build ffa kernels. The default value is `256`.


**MAGI_ATTENTION_SKIP_FFA_UTILS_BUILD**

Toggle this env variable to `1` can skip building `flexible_flash_attention_utils_cuda`. The default value is `0`.


**MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD**

Toggle this env variable to `1` can skip building `magi_attn_ext`. The default value is `0`.
