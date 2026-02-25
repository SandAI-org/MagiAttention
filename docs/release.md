## Major Features

1. **Early Support Blackwell**
    - Added preliminary Blackwell support via a new attention kernel backend `FFA_FA4` using a fork of [Flash-Attention 4](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support).
    - Enable with `export MAGI_ATTENTION_FA4_BACKEND=1` (*see our [docs](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/env_variables.html)*).
    - Installation requires the `flash_attn_cute` module; see the [install guide](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/user_guide/install.html).
    - Technical details and implementation notes are in our [blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html#native-implementation).
    - Blackwell benchmarks are available for both [kernel-level](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html#for-b200) and [distributed-level](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html#id26).
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/190, https://github.com/SandAI-org/MagiAttention/pull/206, https://github.com/SandAI-org/MagiAttention/pull/209, https://github.com/SandAI-org/MagiAttention/pull/225.

2. **Full Native Group Collective (Intra/Internode)**
    - Implemented native kernel implementation of `group collective` primitives for intranode and internode communication using [DeepEP](https://github.com/deepseek-ai/DeepEP), improving performance over the prior `all-to-all-v` approach, especially with pre-/post-processing elimination and internode volume de-duplication.
    - Enable with `export MAGI_ATTENTION_NATIVE_GRPCOLL=1` (*see our [docs](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/env_variables.html)*).
    - Installation requires enabling `IBGDA`; see the [install guide](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/user_guide/install.html).
    - Implementation notes: [blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html#native-implementation).
    - Distributed benchmark results: [blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html#distributed-level).
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/182, https://github.com/SandAI-org/MagiAttention/pull/223, https://github.com/SandAI-org/MagiAttention/pull/228, https://github.com/SandAI-org/MagiAttention/pull/229, https://github.com/SandAI-org/MagiAttention/pull/230, https://github.com/SandAI-org/MagiAttention/pull/233, https://github.com/SandAI-org/MagiAttention/pull/241, https://github.com/SandAI-org/MagiAttention/pull/249, https://github.com/SandAI-org/MagiAttention/pull/253.

3. **(Distributed) Muon QK-Clip for Max Logits**
    - Added support for the (distributed) Muon QK-clip technique from [Kimi K2](https://arxiv.org/pdf/2507.20534) by returning `max_logits` from the FFA forward kernel and performing its distributed reduction in `calc_attn`.
    - Access `meta.max_logits` by passing `return_max_logits=True` to `flex_flash_attn_func` and `calc_attn`.
    - Implementation details: [blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/muon_qk_clip.html).
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/237, https://github.com/SandAI-org/MagiAttention/pull/239, https://github.com/SandAI-org/MagiAttention/pull/245, https://github.com/SandAI-org/MagiAttention/pull/247.

## Developing Features

1. **Optimize Sparse Attention in FFA**
    - Added `SwapAB` to the FFA forward kernel to reduce kBlockM and reduce wgmma waste under sparse attention.
    - Added `PackGQA` to the FFA forward kernel to gather Q heads sharing a KV head for GQA settings under sparse attention.
    - Added specific `tile_scheduler` to the FFA forward kernel to lower meta overhead by passing optional `max_seqlen_q` under sparse attention.
    - Implemented `SparseLoad` to use `cp.async` instead of `TMA`, making sparse global memory access dense in shared memory under sparse attention.
    - Added `SwapBwdQKLoop` in the backward kernel to enable `q-outer-loop + kv-inner-loop`, preparing for future backward optimizations under sparse attention.
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/158, https://github.com/SandAI-org/MagiAttention/pull/185, https://github.com/SandAI-org/MagiAttention/pull/204, https://github.com/SandAI-org/MagiAttention/pull/207, https://github.com/SandAI-org/MagiAttention/pull/214, https://github.com/SandAI-org/MagiAttention/pull/224.

2. **Optimize Dynamic Attention Solver**
    - Improved dynamic solver algorithms and benchmarked across full/causal and document mask patterns.
    - Reduced solver overhead via a C++-based data structure backend and OMP parallelism.
    - Added `flatten_head_group` (enable with `export MAGI_ATTENTION_FLATTEN_HEAD_GROUPS=1`) for additional speedups.
    - Added `cpp_backend_data_structure` (enable with `export MAGI_ATTENTION_CPP_BACKEND=1`) to lower solver overhead.
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/183, https://github.com/SandAI-org/MagiAttention/pull/210, https://github.com/SandAI-org/MagiAttention/pull/220.

3. **Optimize DistAttn**
    - Added `AutoRangeMerge` to reduce fragmented `AttnSlices` after partitioning (enable with `export MAGI_ATTENTION_AUTO_RANGE_MERGE=1` and require JIT building).
    - Added `CatGQA` to improve FFA backward kernels performance under GQA by concatenating Q heads sharing the same KV head (enable with `export MAGI_ATTENTION_CATGQA=1` and require JIT building).
    - Added `HideTailReduce` to trade saving the last remote `kv` activation for reordering overlap stages during backward, hiding the final remote `group_reduce` with the host FFA stage (enable with `export MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE=1`).
    - Related PRs: https://github.com/SandAI-org/MagiAttention/pull/244, https://github.com/SandAI-org/MagiAttention/pull/256.

## Architecture Refactoring

1. **API Update**
    - Changed return values of `flex_flash_attn_func` and `calc_attn` from `(out, lse)` to `(out, meta)`. Access `lse` at `meta.lse` and, if requested, `max_logits` at `meta.max_logits` via `return_max_logits=True` (see PRs: https://github.com/SandAI-org/MagiAttention/pull/237, https://github.com/SandAI-org/MagiAttention/pull/247).
    - Added required arguments `num_heads_q`, `num_heads_kv`, and `head_dim` to `magi_attn_flex_key` and `magi_attn_varlen_key` (see PR: https://github.com/SandAI-org/MagiAttention/pull/236).
    - Deprecated `magi_attn_varlen_dispatch` and `magi_attn_flex_dispatch` (see PR: https://github.com/SandAI-org/MagiAttention/pull/236).
    - Updated `dist_attn_runtime_dict` to be instantiated per `cp_group`; `get_most_recent_key` now requires the `cp_group` argument (see PRs: https://github.com/SandAI-org/MagiAttention/pull/226, https://github.com/SandAI-org/MagiAttention/pull/232).

5. **MagiAttention Extensions**
    - Packaged [magi_attn_extensions](https://github.com/SandAI-org/MagiAttention/tree/v1.1.0/extensions) as a single pip-installable module (see PR: https://github.com/SandAI-org/MagiAttention/pull/176).

## Bug Fixes

1. **Flash Attention with Attention sink**
    - Detached the `sink` tensor from the computation graph in `fa_interface_with_sink`, fixing backprop issues (see PR: https://github.com/SandAI-org/MagiAttention/pull/155).

2. **Native Group Collective**
    - Fixed the GPU-CPU notification bug in the native group collective by disabling pinned counter read/write when GPU-CPU sync is unnecessary (see PR: https://github.com/SandAI-org/MagiAttention/pull/200).

## Testing Enhancements

1. **UniTest Update**
    - Integrated `coverage` into unit-test CI (see PR: https://github.com/SandAI-org/MagiAttention/pull/178).
    - Added online-softmax `ref_attn_func` for the "torch" backend to simulate Flash Attention 2 and reduce memory usage (see PR: https://github.com/SandAI-org/MagiAttention/pull/174).
    - Migrated unit-test CI from `pull_request` to `pull_request_target` to allow external contributors to run CI with reviewer permissions (see PRs: https://github.com/SandAI-org/MagiAttention/pull/212, https://github.com/SandAI-org/MagiAttention/pull/215, https://github.com/SandAI-org/MagiAttention/pull/216, https://github.com/SandAI-org/MagiAttention/pull/217).

## Others

1. **Refine CP Benchmark**
    - Enhanced CP benchmark tooling and pipeline (figures, profiling, flag combinations, etc.). See the CP benchmark [README](https://github.com/SandAI-org/MagiAttention/tree/v1.1.0/exps/README.md#distributed-attention-module-benchmark) for details (see PRs: https://github.com/SandAI-org/MagiAttention/pull/177, https://github.com/SandAI-org/MagiAttention/pull/197, https://github.com/SandAI-org/MagiAttention/pull/205, https://github.com/SandAI-org/MagiAttention/pull/221, https://github.com/SandAI-org/MagiAttention/pull/222, https://github.com/SandAI-org/MagiAttention/pull/227).
    - Extended CP baseline with Megatron [HybridCP](https://github.com/NVIDIA/Megatron-LM/pull/2054) (see PR: https://github.com/SandAI-org/MagiAttention/pull/187).
    - Published a dedicated [blog post](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html) for the CP benchmark (see PR: https://github.com/SandAI-org/MagiAttention/pull/247).

2. **Refactor Docs with new Blogs**
    - Reworked the [User Guide](https://sandai-org.github.io/MagiAttention/docs/v1.1.0/user_guide/) and reorganized [Blogs](https://sandai-org.github.io/MagiAttention/docs/main/blog/) to reflect new content and deep dives (see PR: https://github.com/SandAI-org/MagiAttention/pull/247).

3. **Update Copyright**
    - Updated copyright header comments from `2025` to `2025~2026` (see PR: https://github.com/SandAI-org/MagiAttention/pull/203).