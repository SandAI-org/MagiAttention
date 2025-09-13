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

**CUDA_DEVICE_MAX_CONNECTIONS**

This environment variable defines the number of hardware queues that CUDA streams can utilize. Increasing this value can improve the overlap of communication and computation, but may also increase PCIe traffic.

**MAGI_ATTENTION_QO_COMM**

Toggle this env variable to `1` to enable query/output communication, including fetching remote q (fwd), reducing partial out and lse (fwd), fetching remote q,o,lse,do (bwd), reducing partial dq (bwd), to eliminate the restriction that communication is limited solely to key/value. The default value is `0`.

```{note}
This feature is experimental and under development for now, which dose NOT support neither multi-stage overlap nor hierarchical comm.
```


**MAGI_ATTENTION_FFA_FORWARD_HIGH_PRECISION_REDUCE**

Toggle this env variable to `1` to enable high-precision (fp32) reduce for partial out during dist-attn forward
to trade-off double comm overhead for increased precision and less dtype-cast overhead. The default value is `0`.

```{note}
1. Inside the ffa forward kernel, we always use high-precision (fp32) accumulation for partial out.

2. We always use high-precision (fp32) lse everywhere.

3. This feature works for out only when enabling qo comm.
```


**MAGI_ATTENTION_FFA_BACKWARD_HIGH_PRECISION_REDUCE**

Toggle this env variable to `1` to enable high-precision (fp32) reduce for partial dq,dk,dv during dist-attn backward
to trade-off double comm overhead for increased precision and less dtype-cast overhead. The default value is `0`.

```{note}
1. Inside the ffa backward kernel, we always use high-precision (fp32) accumulation for partial dq,dk,dv.

2. This feature works for dq only when enabling qo comm.
```

**MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE**

Set the value of this env variable to control the size of `dist_attn_runtime_dict`. The default value is `100`.


## For Debug

**MAGI_ATTENTION_SANITY_CHECK**

Toggle this env variable to `1` to enable many sanity check codes inside magi_attention. The default value is `0`.

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

### JIT

**MAGI_ATTENTION_WORKSPACE_BASE**

Specifies the base directory for the Magi Attention workspace, which includes cache and generated source files. If not set, it defaults to the user's home directory (`~`).

**MAGI_ATTENTION_BUILD_VERBOSE**

Toggle this env variable to `1` to enable verbose output during the JIT compilation process, showing the full ninja build commands being executed. The default value is `0`.

**MAGI_ATTENTION_BUILD_DEBUG**

Toggle this env variable to `1` to enable debug flags for the C++/CUDA compiler. This includes options like `-g` (debugging symbols) and other flags to get more detailed information, such as register usage. The default value is `0`.

**NVCC_THREADS**

Sets the number of threads for `nvcc`'s `--split-compile` option, which can speed up the JIT compilation of CUDA kernels. The default value is `4`.

### AOT

**MAGI_ATTENTION_PREBUILD_FFA**

Toggle this env variable to `1` to enable pre-build ffa kernels for some common options with `ref_block_size=None` and leave others built in jit mode. The default value is `1`.


**MAGI_ATTENTION_PREBUILD_FFA_JOBS**

Set the value of this env variable to control the number of jobs used to pre-build ffa kernels. The default value is `256`.


**MAGI_ATTENTION_SKIP_FFA_UTILS_BUILD**

Toggle this env variable to `1` can skip building `flexible_flash_attention_utils_cuda`. The default value is `0`.


**MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD**

Toggle this env variable to `1` can skip building `magi_attn_ext`. The default value is `0`.
