# Environment Variables

Below are some environment variables that can be set, along with their descriptions.
All of the MagiAttention-specific ones are prefixed with `MAGI_ATTENTION_`.

:::{note}
Since MagiAttention is actively evolving, many advanced but experimental features will be released but enabled through environment variables.
:::

```{contents}
:local: true
```


## For Correctness

**MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.comm.is_fwd_high_precision_reduce_enable`

Toggle this env variable to `1` to enable high-precision (fp32) reduce for partial out during dist-attn forward
to trade-off double comm overhead for increased precision and less dtype-cast overhead.

```{note}
1. Inside the ffa forward kernel, we always use high-precision (fp32) accumulation for partial out.

2. We always use high-precision (fp32) lse everywhere.

3. This feature works for out only when enabling qo comm.
```


**MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.comm.is_bwd_high_precision_reduce_enable`

Toggle this env variable to `1` to enable high-precision (fp32) reduce for partial dq,dk,dv during dist-attn backward
to trade-off double comm overhead for increased precision and less dtype-cast overhead.

```{note}
1. Inside the ffa backward kernel, we always use high-precision (fp32) accumulation for partial dq,dk,dv.

2. This feature works for dq only when enabling qo comm.
```


**MAGI_ATTENTION_DSINK_ALL_REDUCE_OP**

- **Defaults to:** `none` (options: `none`, `sum`, `avg`)
- **Used by:** `magi_attention.env.comm.dsink_all_reduce_op`

Set the value of this env variable to control the all-reduce op for sink gradients within `dist_attn_func` when involving attention sink.


```{note}
For now we only accept global replicated sink tensor as input to feed into `dist_attn_func`, and the gradients of sink in each cp rank are partial and requires to be sum-reduced across cp ranks.

However, since sink tensor is learnable, it will be considered as a regular parameter in the model similar to `bias` in `nn.Linear` layer.

So under some popular training frameworks, such as Megatron-LM, FSDP, the sum-reduction across cp ranks of the partial gradients of sink might be automatically applied within the whole `dp x cp` mesh.

To avoid repeated reduction, we provide this environment variable to specify the all-reduce op for sink gradients within `dist_attn_func`, whose default value is `none` to NOT apply any reduction to sink gradients by `dist_attn_func` and let the framework handle it.

However, under the scenarios w/o any framework mechanism to reduce parameters across cp ranks, you have to specify this environment variable to `sum`.

And sometimes, `avg` might also be an option when you need to scale the sink gradients by `1/cp`.
```


## For Runtime

**MAGI_ATTENTION_LOG_LEVEL**

- **Defaults to:** `WARN`
- **Used by:** `magi_attention.env.general.log_level`

Set this env variable to control the logging verbosity of the entire `magi_attention` package.
Valid values (case-insensitive) are `DEBUG`, `INFO`, `WARN`, `WARNING`, `ERROR`, `CRITICAL`.

**MAGI_ATTENTION_KERNEL_BACKEND**

- **Defaults to:** `ffa`
- **Used by:** `magi_attention.env.general.kernel_backend`

Set this env variable to choose the attn kernel backend. Valid values are:

- `ffa`: flex-flash-attention (default, high-performance persistent kernel).
- `sdpa`: offline SDPA implementation (for testing / high precision like `fp32`/`fp64`).
- `sdpa_ol`: online (block-wise) SDPA implementation (for testing, lower memory than `sdpa`).
- `fa4`: Flash-Attention 4 monkey-patch (workaround for Blackwell GPUs).

```{note}
This supersedes the legacy `MAGI_ATTENTION_SDPA_BACKEND=1` and `MAGI_ATTENTION_FA4_BACKEND=1`
toggles, which are still supported but must NOT be set together with `MAGI_ATTENTION_KERNEL_BACKEND`.
```

**MAGI_ATTENTION_PRECISION**

- **Defaults to:** unset (use the input dtype as-is)
- **Used by:** `magi_attention.env.general.precision`

Set this env variable to override the compute dtype for attention kernels. Valid values are
`bf16`, `fp16`, `fp32`, `fp64`. When set, input Q/K/V are cast to the specified dtype before
attention computation, and the output is cast back to the original input dtype.


## For Performance

**MAGI_ATTENTION_FA4_BACKEND**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.kernel_backend`

Toggle this env variable to `1` to switch the attn kernel backend from `FFA` to `FFA_FA4`, a monkey patch version of Flash-Attention 4, to temporarily support arbitrary mask on Blackwell GPUs.

```{note}
This is a legacy toggle kept for backward compatibility; prefer `MAGI_ATTENTION_KERNEL_BACKEND=fa4`.
This is for now a workaround solution might be removed or updated in the future.
```

**MAGI_ATTENTION_NATIVE_GRPCOLL**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.comm.is_native_grpcoll_enable`

Toggle this env variable to `1` to enable native kernel implementation for group collective comm.

```{note}
This feature is experimental and under active development for now, and not compatible with many other features,
thus please do NOT enable it unless you know exactly what you are doing.
```

**MAGI_ATTENTION_QO_COMM**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.comm.is_qo_comm_enable`

Toggle this env variable to `1` to enable query/output communication, including fetching remote q (fwd), reducing partial out and lse (fwd), fetching remote q,o,lse,do (bwd), reducing partial dq (bwd), to eliminate the restriction that communication is limited solely to key/value.

```{note}
This feature is experimental and under active development for now, and not compatible with many other features,
thus please do NOT enable it unless you know exactly what you are doing.
```

**MAGI_ATTENTION_FLATTEN_HEAD_GROUPS**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.is_flatten_head_groups_enable`

Toggle this env variable to `1` to flatten head groups within GQA/MQA attention to optimize dynamic solver performance.

```{note}
This feature is experimental and under active development for now, and not compatible with many other features,
thus please do NOT enable it unless you know exactly what you are doing.
```

**MAGI_ATTENTION_AUTO_RANGE_MERGE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.is_auto_range_merge_enable`

Toggle this env variable to `1` to enable automatic range merging for flex-flash-attention,
to improve performance by reducing the number of attention ranges.

```{note}
This feature is experimental and under active development for now,
thus please do NOT enable it unless you know exactly what you are doing.
```

**MAGI_ATTENTION_CATGQA**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.is_cat_gqa_enable`

Toggle this env variable to `1` to enable CatGQA mode for flex-flash-attention backward, to further optimize the performance under GQA settings by concatenating multiple Q heads sharing the same KV head.

```{note}
This feature is experimental and under active development for now,
thus please do NOT enable it unless you know exactly what you are doing.
```

**MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.dist_attn_backward_hide_tail_reduce`

Toggle this env variable to `1` to trade saving the last remote `kv` activation for reordering overlap stages during backward, hiding the final remote `group_reduce` with the host FFA stage.

```{note}
This feature is experimental and under active development for now, and not compatible with many other features like qo comm,
thus please do NOT enable it unless you know exactly what you are doing.
```

**MAGI_ATTENTION_MIN_CHUNKS_PER_RANK**

- **Defaults to:** `8`
- **Used by:** `magi_attention.env.general.min_chunks_per_rank`

Set the value of this env variable to control the minimum number of chunks per context parallel rank, to control the granularity of computational load-balance.

**MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN**

- **Defaults to:** `4` if `CUDA_DEVICE_MAX_CONNECTIONS` > `1`, otherwise `0`
- **Used by:** `magi_attention.env.comm.ffa_fwd_sm_margin_save_for_comm`

Set the value of this env variable to control the number of SMs of the ffa forward kernel saved for comm kernels.

**MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN**

- **Defaults to:** `4` if `CUDA_DEVICE_MAX_CONNECTIONS` > `1`, otherwise `0`
- **Used by:** `magi_attention.env.comm.ffa_bwd_sm_margin_save_for_comm`

Set the value of this env variable to control the number of SMs of the ffa backward kernel saved for comm kernels.

**MAGI_ATTENTION_CPP_BACKEND**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.is_cpp_backend_enable`

Toggle this env variable to `1` to enable C++ backend for core data structures (`AttnRange`, `AttnMaskType`, etc.) to avoid Python overhead.

```{note}
This feature is experimental and under active development for now.
If the C++ extension is not found or this variable is set to `0`, it will fall back to the Python implementation.
```

**MAGI_ATTENTION_HIERARCHICAL_COMM**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.comm.is_hierarchical_comm_enable`

Toggle this env variable to `1` to enable hierarchical group-collective comm within 2-dim cp group (inter_node group + intra_node group).

```{note}
This is for now a temporary solution to reduce the redundant inter-node communication and might be removed or updated in the future.
```

**MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE**

- **Defaults to:** `1000`
- **Used by:** `magi_attention.env.general.dist_attn_runtime_dict_size`

Set the value of this env variable to control the maximum LRU cache size of `dist_attn_runtime_dict_mgr`.

**CUDA_DEVICE_MAX_CONNECTIONS**

- **Defaults to:** `8`
- **Used by:** `magi_attention.env.general.is_cuda_device_max_connections_one`

This environment variable defines the number of hardware queues that CUDA streams can utilize. Increasing this value can improve the overlap of communication and computation, but may also increase PCIe traffic.


## For Debug

**MAGI_ATTENTION_SANITY_CHECK**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.is_sanity_check_enable`

Toggle this env variable to `1` to enable many sanity check codes inside magi_attention.

```{note}
This is only supposed to be used for testing or debugging, since the extra sanity-check overhead might be non-negligible.
```

**MAGI_ATTENTION_SDPA_BACKEND**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.kernel_backend`

Toggle this env variable to `1` can switch the attn kernel backend from ffa to sdpa-math, to support higher precision like `fp32` or `fp64`.

```{note}
This is a legacy toggle kept for backward compatibility; prefer `MAGI_ATTENTION_KERNEL_BACKEND=sdpa`.
This is only supposed to be used for testing or debugging, since the performance is not acceptable.
```

**MAGI_ATTENTION_DETERMINISTIC_MODE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.is_deterministic_mode_enable`

Toggle this env variable to `1` to enable deterministic mode to use deterministic algorithms for all magi_attention kernels.


**MAGI_ATTENTION_PROFILE_MODE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.general.is_profile_mode_enable`

Toggle this env variable to `1` to enable profiling mode to profile all magi_attention kernels, currently mainly for ffa kernels (*see [here](https://github.com/SandAI-org/MagiAttention/tree/main/exps/attn/profile_ffa) for more details*).

```{note}
This is only supposed to be used for development. Please do NOT enable it in production.
```

**MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.kernel.cutedsl`

Toggle this env variable to `1` to enable debug-time checks and diagnostics for the CuteDSL FFA kernels.

## For Build

### JIT

**MAGI_ATTENTION_WORKSPACE_BASE**

- **Defaults to:** `$HOME`
- **Used by:** `magi_attention.env.build.workspace_base_dir`

Specifies the base directory for the MagiAttention workspace, which includes cache and generated source files.

**MAGI_ATTENTION_BUILD_VERBOSE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.build.is_build_verbose`

Toggle this env variable to `1` to enable verbose output during the JIT compilation process, showing the full ninja build commands being executed.

**MAGI_ATTENTION_BUILD_DEBUG**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.build.is_build_debug`

Toggle this env variable to `1` to enable debug flags for the C++/CUDA compiler. This includes options like `-g` (debugging symbols) and other flags to get more detailed information, such as register usage.

**MAGI_ATTENTION_NO_BUILD_CACHE**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.build.is_no_build_cache`

Toggle this env variable to `1` to disable caching for built ffa kernels.

**MAGI_ATTENTION_FORCE_JIT_BUILD**

- **Defaults to:** `0`
- **Used by:** `magi_attention.env.build.is_force_jit_build`

Toggle this env variable to `1` to force building FFA in JIT mode, even the pre-built AOT `libs` exists.

**NVCC_THREADS**

- **Defaults to:** `4`
- **Used by:** `magi_attention.env.build.nvcc_threads`

Sets the number of threads for `nvcc`'s `--split-compile` option, which can speed up the JIT compilation of CUDA kernels.

**MAGI_ATTENTION_FFA_CUTEDSL_CACHE_ENABLED**

- **Defaults to:** `0`
- **Used by:** `magi_attention.kernel.cutedsl.cache_utils`

Toggle this env variable to `1` to enable on-disk caching of compiled CuteDSL FFA kernels.

**MAGI_ATTENTION_FFA_CUTEDSL_CACHE_DIR**

- **Defaults to:** unset (use the default cache location)
- **Used by:** `magi_attention.kernel.cutedsl.cache_utils`

Set this env variable to specify the directory for the CuteDSL FFA kernel cache.


### AOT

**MAGI_ATTENTION_PREBUILD_FFA_JOBS**

- **Defaults to:** `ceil(num_cpu_cores * 0.9)`
- **Used by:** `setup.py`

Set the value of this env variable to control the number of parallel compilation jobs used to pre-build ffa kernels.

**MAX_JOBS**

- **Defaults to:** `ceil(num_cpu_cores * 0.9)`
- **Used by:** `setup.py`

Set the value of this env variable to control the number of parallel compilation jobs used to build the extension modules other than ffa.

**MAGI_ATTENTION_PREBUILD_FFA**

- **Defaults to:** `1`
- **Used by:** `setup.py`

Toggle this env variable to `1` to enable pre-build ffa kernels for some common options with `ref_block_size=None` and leave others built in jit mode.

**MAGI_ATTENTION_SKIP_CUDA_BUILD**

- **Defaults to:** `0`
- **Used by:** `setup.py`

Toggle this env variable to `1` to skip building all the CUDA extension modules entirely (e.g. for a Python-only / docs build).

**MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD**

- **Defaults to:** `0`
- **Used by:** `setup.py`

Toggle this env variable to `1` can skip building `magi_attn_ext`.

**MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD**

- **Defaults to:** `0`
- **Used by:** `setup.py`

Toggle this env variable to `1` can skip building `magi_attn_comm`.

**MAGI_ATTENTION_FORCE_CXX11_ABI**

- **Defaults to:** `0`
- **Used by:** `setup.py`

Toggle this env variable to `1` to force building the extension modules with the C++11 ABI (`_GLIBCXX_USE_CXX11_ABI=1`), to match a PyTorch build compiled with the new ABI.

**MAGI_ATTENTION_DISABLE_SM90_FEATURES**

- **Defaults to:** `0`
- **Used by:** `setup.py`

Toggle this env variable to `1` to disable the SM90 (Hopper)-specific features when building the extension modules, e.g. when targeting a toolchain without SM90 support.

**MAGI_ATTENTION_FFA_FA4_CACHE_DIR**

- **Defaults to:** `magi_attention/lib/ffa_fa4_cache/`
- **Used by:** `magi_attention.functional.fa4_utils`

Set this env variable to specify the cache directory for pre-compiled `FFA_FA4` kernels.

**MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY**

- **Defaults to:** unset (auto-detect the current GPU device)
- **Used by:** `setup.py`

Set this env variable to specify the compute capability used to build MagiAttention extension modules (affects `magi_attn_comm` and `create_block_mask`). Supports comma-separated values for multi-arch builds, e.g. `MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90,100`. If not set, we will try to detect the compute capability of the current GPU device, and raise an error if detection fails.

**MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12**

- **Defaults to:** `0`
- **Used by:** `setup.py`

Set this env variable to `1` to allow building MagiAttention extension modules with `CUDA-12`, which might cause significant performance degradation compared to `CUDA-13+`. When `0`, we will raise an error and abort the installation if the CUDA version is lower than `13.0`.

**NVSHMEM_DIR**

- **Defaults to:** the system module `nvidia-nvshmem-cu12` (as listed in `requirements.txt`)
- **Used by:** `setup.py`

Set this env variable to the path of the custom `nvshmem` installation directory.

If not found anywhere, all relative features used in native group collective comm kernels are disabled.


## For Testing

**MAGI_ATTENTION_PARAMETERIZE_RUN_IN_MP**

- **Defaults to:** `0`
- **Used by:** `magi_attention.testing` (`with_run_in_mp`)

Whether to run parameterized distributed test cases in a multiprocessing context. This is set internally by the `@with_run_in_mp` decorator rather than being configured by users directly.

**MAGI_ATTENTION_COPYRIGHT_TEST_YEAR**

- **Defaults to:** current calendar year
- **Used by:** `tools/codestyle/copyright.py`

Overrides the "current year" used by the copyright-header check. Useful for reproducible testing of the copyright tooling.
