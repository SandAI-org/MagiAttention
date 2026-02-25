## Major Features

1. **Early Support Blackwell**
   - Added early support for **Blackwell** via a new attention kernel backend `FFA_FA4` using forked [Flash-Attention 4](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support).
   - It can be enabled via `export MAGI_ATTENTION_FA4_BACKEND=1` (*see our [docs](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/env_variables.html) for more details*).
   - It requires specific installation process including the dependent `flash_attn_cute` module, described in our [docs](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/user_guide/install.html).
   - For more technical details, see our [blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html#native-implementation).
    - For Blackwell benchmark results, see our blog for both [kernel-level](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html#for-b200) and [distributed-level](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html#id26).
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/190, https://github.com/SandAI-org/MagiAttention/pull/206, https://github.com/SandAI-org/MagiAttention/pull/209, https://github.com/SandAI-org/MagiAttention/pull/225.

2. **Fully Support Native Group Collective for Intranode/Internode**
    - Supported native kernel implementation of `group collective` primitives for both intranode and internode communication based on [DeepEP](https://github.com/deepseek-ai/DeepEP), to achieve better performance than the original `all-to-all-v` implementation.
    - It can be enabled via `export MAGI_ATTENTION_NATIVE_GRPCOLL=1` (*see our [docs](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/env_variables.html) for more details*).
    - It requires specific installation process including enabling `IBGDA`, described in our [docs](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/user_guide/install.html).
    - For more technical details, see our [blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html#native-implementation).
    - For benchmark results, see our [blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html#distributed-level).
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/182, https://github.com/SandAI-org/MagiAttention/pull/223, https://github.com/SandAI-org/MagiAttention/pull/228, https://github.com/SandAI-org/MagiAttention/pull/229, https://github.com/SandAI-org/MagiAttention/pull/230, https://github.com/SandAI-org/MagiAttention/pull/233, https://github.com/SandAI-org/MagiAttention/pull/241, https://github.com/SandAI-org/MagiAttention/pull/249, https://github.com/SandAI-org/MagiAttention/pull/253.

 3. **Support (Distributed) Muon QK-Clip for Max Logits**
    - Supported (distributed) Muon QK-clip technique introduced in [Kimi K2](https://arxiv.org/pdf/2507.20534), by returning the `max_logits` from the FFA forward kernel, and handling its (distributed) reduction for `calc_attn`.
    - It can be accessed from the returned `meta.max_logits` by just passing the optional argument `return_max_logits=True` for both `flex_flash_attn_func` and `calc_attn` APIs.
    - For more technical details, see our [blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/muon_qk_clip.html).
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/237, https://github.com/SandAI-org/MagiAttention/pull/239, https://github.com/SandAI-org/MagiAttention/pull/245, https://github.com/SandAI-org/MagiAttention/pull/247.

## Developing Features

1. **Optimize Sparse Attention in FFA**
    - Supported `SwapAB` feature for the FFA forward kernel to reduce `kBlockM` size for less wgmma instruction waste under sparse attention scenarios.
    - Supported `PackGQA` feature for the FFA forward kernel to gather the `q` sharing the same head of `kv` for better performance under sparse attention scenarios.
    - Supported individual `tile_scheduler` for the FFA forward kernel to reduce its meta load/store and calculation overhead, with the help of optional argument `max_seqlen_q`.
    - Supported `SparseLoad` feature for the FFA forward kernel to use `cp.async` to load instead of `TMA` to make sparse data in global memory become dense in shared memory under sparse attention scenarios.
    - Supported `SwapBwdQKLoop` feature for the FFA backward kernel to allow `q-outer-loop + kv-inner-loop`, to lay a foundation for future backward features development under sparse attention scenarios.
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/158, https://github.com/SandAI-org/MagiAttention/pull/185, https://github.com/SandAI-org/MagiAttention/pull/204, https://github.com/SandAI-org/MagiAttention/pull/207, https://github.com/SandAI-org/MagiAttention/pull/214, https://github.com/SandAI-org/MagiAttention/pull/224.
 
2. **Optimize Dynamic Attention Solver**
    - Optimized the algorithms of dynamic solver and benchmarked under full, causal, full document, causal document mask patterns.
    - Optimized the solving overhead of dynamic solver with c++ basic data structure backend supported and `OMP` parallelism.
    - Supported `flatten_head_group` feature to further improve the results from dynamic solver, enabled via `export MAGI_ATTENTION_FLATTEN_HEAD_GROUPS=1` (*see our [docs](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/env_variables.html) for more details*).
    - Supported `cpp_backend_data_structure` feature to reduce the overhead of dynamic solver, enabled via `export MAGI_ATTENTION_CPP_BACKEND=1` (*see our [docs](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/env_variables.html) for more details*).
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/183, https://github.com/SandAI-org/MagiAttention/pull/210, https://github.com/SandAI-org/MagiAttention/pull/220.

3. **(Experimental) Optimize DistAttn**
    - Supported `AutoRangeMerge` feature to reduce the fragmented `AttnSlices` due to partition, enabled via `export MAGI_ATTENTION_AUTO_RANGE_MERGE=1`.
    - Supported `CatGQA` feature to optimize FFA backward kernel performance under `GQA` settings by concatenating multiple Q heads sharing the same KV head, enabled via `export MAGI_ATTENTION_CATGQA=1`.
    - Supported `HideTailReduce` feature to trade-off saving last stage of remote `kv` as extra activation during forward to reverse the order of overlap stages during backward to hide last remote stage of `group_reduce` with host stage of `ffa`, enabled via `export MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE=1`.
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/244, https://github.com/SandAI-org/MagiAttention/pull/256.

## Architecture Refactoring

1. **API Update**
   - Updated the return values for both `flex_flash_attn_func` and `calc_attn` APIs, from `(out, lse)` to `(out, meta)`, where you can access (local) `lse` by `meta.lse`, as well as (global) `max_logits` by `meta.max_logits` if passing the optional argument `return_max_logits=True`, w.r.t. the PRs: https://github.com/SandAI-org/MagiAttention/pull/237, https://github.com/SandAI-org/MagiAttention/pull/247.
   - Added `num_heads_q`, `num_heads_kv` and `head_dim` into required arguments for `magi_attn_flex_key` and `magi_attn_varlen_key` APIs, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/236.
   - Marked `magi_attn_varlen_dispatch` and `magi_attn_flex_dispatch` as deprecated APIs, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/236.
   - Updated `dist_attn_runtime_dict` to be instantiated per cp_group, thus `get_most_recent_key` now requires pass in the `cp_group` argument, w.r.t. the PRs: https://github.com/SandAI-org/MagiAttention/pull/226, https://github.com/SandAI-org/MagiAttention/pull/232.

5. **MagiAttention Extensions**
    - Support the [magi_attn_extensions](https://github.com/SandAI-org/MagiAttention/tree/v1.1.0/extensions) as a single module to be installed with `pip`, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/176.

## Bug Fixes

1. **Flash Attention with Attention sink**
   - Detached the `sink` tensor from computation graph for `fa_interface_with_sink
`, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/155.

2. **Native Group Collective**
    - Fixed GPU-CPU notify mechanism bug of native group collective for both intranode and internode, by disabling the read/write for the pinned counters when no GPU-CPU sync is required, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/200.

## Testing Enhancements

1. **UniTest Update**
   - Integated `coverage` into the unitest CI, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/178.
   - Supported `ref_attn_func` with online-softmax implementation for "torch" backend to reduce memory overhead, simulating `Flash Attention 2`, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/174.
   - Updated the unitest CI from `pull_request`mode to `pull_request_target` mode, to allow the outside developer being able to run CI with any of the reviewers' permission, w.r.t. the PRs: https://github.com/SandAI-org/MagiAttention/pull/212, https://github.com/SandAI-org/MagiAttention/pull/215, https://github.com/SandAI-org/MagiAttention/pull/216, https://github.com/SandAI-org/MagiAttention/pull/217.

## Others

1. **Refine CP Benchmark**
   - Refined CP benchmark with more features and better pipeline, including the figure drawing, profiling, flag combination switching, etc (*see the [README](https://github.com/SandAI-org/MagiAttention/tree/v1.1.0/exps/README.md#distributed-attention-module-benchmark) for more details*), w.r.t. the PRs: https://github.com/SandAI-org/MagiAttention/pull/177, https://github.com/SandAI-org/MagiAttention/pull/197, https://github.com/SandAI-org/MagiAttention/pull/205, https://github.com/SandAI-org/MagiAttention/pull/221, https://github.com/SandAI-org/MagiAttention/pull/222, https://github.com/SandAI-org/MagiAttention/pull/227. 
   - Extended the CP baseline with Megatron [HybridCP](https://github.com/NVIDIA/Megatron-LM/pull/2054), w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/187.
   - Added an individual [blog post](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html) for CP Benchmark, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/247.

2. **Refactor Docs with new Blogs**
   - Refactored the [User Guide](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/user_guide/) in docs, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/247.
   - Moved and refactored the [Blogs](https://sandai-org.github.io/MagiAttention/docs/main/blog/) in docs, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/247.
   
 3. **Update Copyright**
     - Updated the copyright header comments from `2025` to `2025~2026`, w.r.t. the PR: https://github.com/SandAI-org/MagiAttention/pull/203.