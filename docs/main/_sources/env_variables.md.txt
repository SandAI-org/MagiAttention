# Environment Variables

In **MagiAttention**, many features need to be configured through environment variables. Below are some environment variables that can be set, along with their descriptions.


## For Correctness

**MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE**

Toggle this env variable to `1` to enable high-precision (fp32) reduce for partial out during dist-attn forward
to trade-off double comm overhead for increased precision and less dtype-cast overhead. The default value is `0`.

```{note}
1. Inside the ffa forward kernel, we always use high-precision (fp32) accumulation for partial out.

2. We always use high-precision (fp32) lse everywhere.

3. This feature works for out only when enabling qo comm.
```


**MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE**

Toggle this env variable to `1` to enable high-precision (fp32) reduce for partial dq,dk,dv during dist-attn backward
to trade-off double comm overhead for increased precision and less dtype-cast overhead. The default value is `0`.

```{note}
1. Inside the ffa backward kernel, we always use high-precision (fp32) accumulation for partial dq,dk,dv.

2. This feature works for dq only when enabling qo comm.
```


**MAGI_ATTENTION_DSINK_ALL_REDUCE_OP**

Set the value of this env variable to control the all-reduce op for sink gradients within `dist_attn_func` when involving attention sink. The default value is `none`. And options are within {`none`, `sum`, `avg`}.


```{note}
For now we only accept global replicated sink tensor as input to feed into `dist_attn_func`, and the gradients of sink in each cp rank are partial and requires to be sum-reduced across cp ranks.

However, since sink tensor is learnable, it will be considered as a regular parameter in the model similar to `bias` in `nn.Linear` layer.

So under some popular training frameworks, such as Megatron-LM, FSDP, the sum-reduction across cp ranks of the partial gradients of sink might be automatically applied within the whole `dp x cp` mesh.

To avoid repeated reduction, we provide this environment variable to specify the all-reduce op for sink gradients within `dist_attn_func`, whose default value is `none` to NOT apply any reduction to sink gradients by `dist_attn_func` and let the framework handle it.

However, under the scenarios w/o any framework mechanism to reduce parameters across cp ranks, you have to specify this environment variable to `sum`.

And sometimes, `avg` might also be an option when you need to scale the sink gradients by `1/cp`.
```


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
This feature is experimental and under early development for now, and not compatible with many other features,
thus please do NOT enable it unless you know exactly what you are doing.
```

**MAGI_ATTENTION_NATIVE_GRPCOLL**

Toggle this env variable to `1` to enable native kernel implementation for group collective comm. The default value is `0`.

```{note}
This feature is experimental and under early development for now, and not compatible with many other features,
thus please do NOT enable it unless you know exactly what you are doing.
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


**MAGI_ATTENTION_PROFILE_MODE**

Toggle this env variable to `1` to enable profiling mode to profile all magi_attention kernels, by now mainly for ffa kernels (*see [here](https://github.com/SandAI-org/MagiAttention/tree/main/exps/attn/profile_ffa) for more details*). The default value is `0`.

```{note}
This is only supposed to be used for development. Please do NOT enable it in production.
```

## For Build

### JIT

**MAGI_ATTENTION_WORKSPACE_BASE**

Specifies the base directory for the Magi Attention workspace, which includes cache and generated source files. If not set, it defaults to the user's home directory (`~`).

**MAGI_ATTENTION_BUILD_VERBOSE**

Toggle this env variable to `1` to enable verbose output during the JIT compilation process, showing the full ninja build commands being executed. The default value is `0`.

**MAGI_ATTENTION_BUILD_DEBUG**

Toggle this env variable to `1` to enable debug flags for the C++/CUDA compiler. This includes options like `-g` (debugging symbols) and other flags to get more detailed information, such as register usage. The default value is `0`.


**MAGI_ATTENTION_NO_BUILD_CACHE**

Toggle this env variable to `1` to disable caching for built ffa kernels. The default value is `0`.

**MAGI_ATTENTION_FORCE_JIT_BUILD**

Toggle this env variable to `1` to force building FFA in JIT mode, even the pre-built AOT `libs` exists. The default value is `0`.

**NVCC_THREADS**

Sets the number of threads for `nvcc`'s `--split-compile` option, which can speed up the JIT compilation of CUDA kernels. The default value is `4`.


### AOT

**MAGI_ATTENTION_PREBUILD_FFA_JOBS**

Set the value of this env variable to control the number of parallel compilation jobs used to pre-build ffa kernels. The default value is the ceiling of `90%` of the available CPU cores (i.e. `ceil(num_cpu_cores * 0.9)`).

**MAX_JOBS**

Set the value of this env variable to control the number of parallel compilation jobs used to build the extension modules other than ffa. The default value is the ceiling of `90%` of the available CPU cores (i.e. `ceil(num_cpu_cores * 0.9)`).

**MAGI_ATTENTION_PREBUILD_FFA**

Toggle this env variable to `1` to enable pre-build ffa kernels for some common options with `ref_block_size=None` and leave others built in jit mode. The default value is `1`.


**MAGI_ATTENTION_SKIP_FFA_UTILS_BUILD**

Toggle this env variable to `1` can skip building `flexible_flash_attention_utils_cuda`. The default value is `0`.


**MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD**

Toggle this env variable to `1` can skip building `magi_attn_ext`. The default value is `0`.

**MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD**

Toggle this env variable to `1` can skip building `magi_attn_comm`. The default value is `0`.

**NVSHMEM_DIR**

Set this env variable to the path of the custom `nvshmem` installation directory.

If not set, it defaults to find the system module `nvidia-nvshmem-cu12` as listed in `requirements.txt`.

If not found anywhere, all relative features used in native group collective comm kernels are disabled.
