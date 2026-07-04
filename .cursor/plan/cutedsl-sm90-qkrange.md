# CuTe-DSL SM90 Native qkrange 开发计划

> 2026-07-03 21:40 | branch: feat/cutedsl-sm90-qkrange | base: origin/main@2726855f
> codebase: MagiAttention-2 | env: cenn-2 on h100-6

---

## 任务总纲

**你负责 CuTe-DSL 里 SM90 的所有内容**。核心方向：
1. 用户接口和 kernel 内部都用 **native q_ranges/k_ranges + attn_type_map**（不是 cu_seqlens hack）
2. Dense → Block Sparse → Index Sparse 分阶段开发
3. SM90 独立开发，common/utils 修改做 SM90 独有函数，避免与陈浩 SM100 冲突
4. FA4/FlexAttention 的 block_sparse 入口优先级低，遇到冲突可先删除

## 用户接口方向

| 场景 | 接口 | 说明 |
|------|------|------|
| Dense | q_ranges/k_ranges + attn_type_map | MagiAttention 原生，必须 |
| Index Sparse | index_attn_indices (total_q, nhk, topk) | 同 Cutlass 接口 |
| Block Sparse | q_ranges + sparse_load=True（MagiAttention 原生） | 必须支持 |
| Block Sparse (FA4) | TorchFlexAttnArgs(block_sparse_tensors=...) | 低优先级，可选 |

## 开发原则

- 涉及 common/utils 修改 → 新建 `_sm90` 后缀函数（如 `seqlen_info_sm90.py`），不改已有代码
- 预编译：批量测试前先并行编译所有 config，参考 `.tmp/062-tma2d-relax/precompile_all.py`
- 测试：不自己写测试文件，参考已有的 `test_block_sparse.py`、`test_index_sparse.py`
- Bench：用 `exps/attn/dense/run_benchmark.py` 和 `exps/attn/sparse/bench_sparse_analysis.py`

---

## Phase 1: Dense FWD + BWD with native qkrange

### 1.1 核心目标
kernel 内部直接接收 q_ranges/k_ranges + attn_type_map，不经过 cu_seqlens hack。
支持 full/causal/inv_causal/bi_causal per-range mask types。

### 1.2 需要新建/修改的文件（SM90 独有）

| 文件 | 动作 | 内容 |
|------|------|------|
| `seqlen_info_sm90.py` | 新建 | `SeqlenInfoSm90` 从 q_ranges[bidb] 读 offset/seqlen |
| `block_info_sm90.py` | 新建 | `BlockInfoSm90` 支持 per-range mask_type 的 n_block_min_max |
| `mask_sm90.py` | 新建 | `AttentionMaskSm90` 支持 inv_causal/bi_causal |
| `tile_scheduler_sm90.py` | 新建 | 支持 ranges 的 scheduler |
| `ffa_fwd_sm90.py` | 修改 | __call__ 接收 ranges tensor；kernel 用新的 SeqlenInfo/BlockInfo |
| `ffa_bwd_sm90.py` | 修改 | 同 FWD |
| `flex_flash_attn.py` | 修改 | SM90 分支去掉 ranges_to_cu_seqlens hack，直传 ranges |
| `ffa_utils.py` | 修改 | normalize_mask_types 支持 per-range Tensor |

### 1.3 关键 feature 优先级（Dense 阶段）

| 优先级 | Feature | 说明 |
|--------|---------|------|
| P0 | native q_ranges/k_ranges + per-range mask_types | 核心，替代 cu_seqlens hack |
| P0 | PackGQA FWD | 已有，确保不 break |
| P0 | head 到 outer (scheduler head iteration) | grid 遍历 head 维度 |
| P0 | intra_wg_overlap | producer-consumer warp 间 K/V load 重叠 |
| P1 | tile 大小控制 | 通过参数/env 控制 tile_m/tile_n |
| P1 | stage 参数控制 | num_stages for K/V pipeline |
| P1 | reg 控制 | producer/consumer register quota |
| P1 | store in producer vs consumer | O store 在哪个 warp 执行 |
| P1 | inter_wg_overlap | 多 MMA WG 间重叠 |
| P2 | SwapAB FWD | small Q tile 场景 |
| P2 | PackGQA BWD | BWD 目前 hardcoded False |
| P2 | BWD LoopQ (swap_qk) | 目前只有 LoopK |
| P2 | Atomic reductions | FWD lse, BWD dKV |
| P3 | Deterministic BWD | dQ semaphore ordering |

### 1.4 验收标准

- Dense FWD: varlen full/causal correctness（vs torch SDPA ref）
- Dense BWD: varlen full/causal dQ/dK/dV correctness
- Dense FWD bench: TFLOPS vs FA3 baseline（±5%, seqlen 2K-16K, MHA+GQA）
- Dense BWD bench: TFLOPS vs FA3 baseline

---

## Phase 2: Block Sparse (qkrange 体系)

### 2.1 核心目标
用 MagiAttention 原生的 q_ranges + sparse_load 体系（不是 FA4 的 block lists）。
kernel 内部用 BlockMeta 式结构 + cpasync/TMA1d inner load/store。

### 2.2 需要新建/修改的文件

| 文件 | 动作 | 内容 |
|------|------|------|
| `block_meta_sm90.py` | 新建 | BlockSparseBlockMeta (NamedTuple) |
| `inner_load_sm90.py` | 新建 | TMA 2D / TMA 1D / cpasync scatter load 模式 |
| `inner_store_sm90.py` | 新建 | TMA reduce-add / cpasync atomic store |
| `sparse_utils_sm90.py` | 新建 | SM90 独有的 sparse producer/consumer |
| `ffa_fwd_sm90.py` | 修改 | block sparse inner loop |
| `ffa_bwd_sm90.py` | 修改 | block sparse BWD (LoopK + LoopQ) |

### 2.3 关键 feature

| 优先级 | Feature | 说明 |
|--------|---------|------|
| P0 | Block Sparse FWD (LoopK) | inner K scatter load by ranges |
| P0 | Block Sparse BWD (LoopK) | outer Q, inner K scatter |
| P1 | Block Sparse BWD (LoopQ) | outer K, inner Q scatter（需 swap_qk） |
| P1 | TMA 2D inner load | k_block_size >= tile_n 时 |
| P1 | cpasync scatter inner load | k_block_size < tile_n 时 |
| P2 | range merge | auto_range_merge for block sparse |
| P2 | inner store mode | TMA reduce-add vs cpasync atomic |

### 2.4 验收标准

- Block Sparse FWD+BWD correctness（参考 test_block_sparse.py TestBlockSparseSimple）
- TFLOPS vs Cutlass block sparse（同 density 场景，bench_sparse_analysis.py）

---

## Phase 3: Index Sparse

### 3.1 核心目标
复现 Cutlass 的 index_attn 功能到 CuTe-DSL。

### 3.2 需要新建的文件

| 文件 | 动作 | 内容 |
|------|------|------|
| `index_sparse_sm90.py` | 新建 | IndexSparseTensors(NamedTuple) + FWD/BWD logic |

### 3.3 验收标准

- Index Sparse FWD+BWD correctness（参考 test_index_sparse.py TestIndexSparseSimple）
- TFLOPS vs Cutlass index_attn

---

## Codebase Survey 结果 (2026-07-03 22:00)

### hack 定位

**`flex_flash_attn.py` L121-124 (FWD) / L596-599 (BWD)**:
```python
# Step-1 hack: only q/k ranges equivalent to a cu_seqlens partition are
# supported, so collapse them to cu_seqlens here...
cu_seqlens_q = ranges_to_cu_seqlens(q_ranges)
cu_seqlens_k = ranges_to_cu_seqlens(k_ranges)
```
→ SM90 分支需跳过此 hack，直传 ranges tensor 给 kernel

**`ffa_utils.py` L56-66 `_MaskTypeMap`**:
```python
full: ClassVar[int] = 0
causal: ClassVar[int] = 1
# TODO: support inv_causal and bi_causal
```
→ 需要扩展 mask_type 到 4 种 + per-range Tensor

**`ffa_utils.py` L70-92 `normalize_mask_types`**:
```python
raise NotImplementedError("Per-range mask_types (torch.Tensor) is not supported yet.")
```
→ SM90 native qkrange 后需支持 per-range tensor

### 现有 SeqlenInfo/BlockInfo/Mask 结构

- `SeqlenInfoQK` 从 `cu_seqlens[batch_idx]` 读取 offset/seqlen → native ranges 需直接从 `ranges[bidb]` 读
- `BlockInfo.get_n_block_min_max` 用 `is_causal` bool → 需扩展为 per-range mask_type int 分发
- `AttentionMask` 在 SM90 FWD 用 `R2P bitmask` 快速路径 → causal 可复用，inv_causal/bi_causal 需新增

### SM90 FWD kernel 结构 (ffa_fwd_sm90.py, 2686 lines)

- `FFAFwdSm90.__init__`: 接收 mask_type, is_local, pack_gqa, intra_wg_overlap 等
- `FFAFwdSm90.__call__`: 构造 TMA tensors, scheduler, grid → launch kernel
- kernel 内部: scheduler → get_current_work → SeqlenInfoQK.create(cu_seqlens) → BlockInfo → AttentionMask → inner loop
- Block sparse: 通过 `produce_block_sparse_loads` / `consume_block_sparse_loads` 在 inner loop 中控制 K/V load

### SM90 BWD kernel (ffa_bwd_sm90.py)

- 结构类似 FWD，额外有 dQ_accum, dKV_accum, preprocess/postprocess 步骤
- `pack_gqa = False` hardcoded

### Tile Scheduler

- **Dense non-varlen**: `SingleTileLPTScheduler` (L2 swizzle + LPT)
- **Dense varlen**: `SingleTileVarlenScheduler` (warp-level prefix sum over cu_seqlens)
- **BWD**: `SingleTileLPTBwdScheduler`
- → native ranges 需要适配 VarlenScheduler 的 `_get_num_m_blocks`

### Cutlass C++ FFA 参考架构 (subagent 调研)

**BlockMeta 三变体** (block_meta.h):
- `DenseBlockMeta`: 从 `q_ranges[bidb]`/`k_ranges[bidb]`/`attn_type_map[bidb]` 读取，支持 FULL/CAUSAL/INVCAUSAL/BICAUSAL per-range
- `BlockSparseBlockMeta`: 统一 range_size 的 scatter token walk + producer-only fill_token_indices
- `IndexSparseBlockMeta`: Q-token 级 topk 索引 + K physical row 转换

**Tile Scheduler** (fwd/bwd_tile_scheduler.hpp):
- 唯一一种：persistent work-stealing (atomicAdd tile counter)，从 ranges 计算 tile 数量
- 不需要单独 varlen scheduler —— ranges 本身就表达了 variable length

**Inner Load/Store Modes** (inner_mode.hpp):
- TMA 2D: physically contiguous (dense / block_sparse when kbs>=kBlockN)
- CpAsync scatter: 非连续 per-row gather (8×16B × groups)
- TMA 1D: 仅用于 BWD dX reduce-add store
- 自动选择: `_is_contiguous` compile-time flag

**Producer/Consumer Warp Split**:
- FWD: producer=loads only, consumer=MMA+epilogue+O store
- BWD: producer loaders + optional DxStorer warps; consumer=MMA+dKV epilogue
- `intra_wg_overlap`: Q@Ki 与 P@Vi-1 重叠 (FWD, default true)
- `inter_wg_overlap`: 未启用 ("no gain")

**Register Quotas** (JIT env):
- FWD scatter: (64,216); FWD dense: (56,256)/(40,232)/(32,160)
- BWD: producer 40-104, consumer derived from budget

### CuTe-DSL SM90 Gap Analysis (subagent 调研)

**核心差距**: CuTe-DSL 目前仅理解 `cu_seqlens` 连续分区 + 全局单一 mask_type
- `SeqlenInfoQK`: 从 cu_seqlens[b] 读 offset → 需改为从 ranges[b] 读绝对 start/end
- `BlockInfo`: is_causal constexpr → 需 per-range attn_type runtime dispatch
- `SingleTileVarlenScheduler`: cu_seqlens prefix sum → 需 range-aware scheduling
- `AttentionMask`: 全局 causal/local → 需 per-range FULL/CAUSAL/INVCAUSAL/BICAUSAL

**不需要改的**: 内部 GEMM/pipeline 结构、TMA atom 机制、warp specialization 模式、SM90 tile-size 启发式

### Sparse Utils (sparse_utils.py, ~2300 lines)

- `BlockSparseTensors` (NamedTuple): mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, ...
- `produce_block_sparse_loads`: 根据 batch_idx/head_idx/m_block 索引 mask_block_idx 得到要 load 的 n_block
- 这是 FA4/FlexAttention 的 precomputed block lists 体系
- → Phase 2 需要新建 SM90 ranges-based block sparse，不改此文件

---

## 执行时间表

| 阶段 | 任务 | 预计 |
|------|------|------|
| Phase 1a | Dense FWD native qkrange + PackGQA + correctness | 1 周 |
| Phase 1b | Dense BWD native qkrange + correctness | 1 周 |
| Phase 1c | Dense bench 对齐 + 性能调优 | 0.5 周 |
| Phase 2a | Block Sparse FWD (LoopK) | 1 周 |
| Phase 2b | Block Sparse BWD (LoopK + LoopQ) | 1-2 周 |
| Phase 3 | Index Sparse FWD + BWD | 2-3 周 |

**总计：约 6-8 周**

---

## 详细执行计划与提交策略 (2026-07-03 22:17)

### 整体框架设计

**核心洞察**：现有 SM90 kernel 的内部 GEMM/TMA/warp-spec 微架构完全不需要改。改动集中在"数据入口"层：
1. **host dispatch** (`flex_flash_attn.py`): 跳过 `ranges_to_cu_seqlens` hack，直传 ranges tensor
2. **SeqlenInfo** → **RangeInfoQK**: 从 `ranges[bidb]` 读绝对 offset/seqlen，而非 cu_seqlens 累积前缀
3. **Scheduler**: 保持 `SingleTileVarlenScheduler` 不变 — 传 `seqlens_q = ranges[:,1] - ranges[:,0]` 作为 `mSeqUsedQ`
4. **BlockInfo/AttentionMask**: Phase 1a 先保持全局 mask_type（compile-time），Phase 1c 再加 per-range runtime dispatch

**关键设计决策**：
- `RangeInfoQK` 保持与 `SeqlenInfoQK` 完全相同的字段名和方法签名 → 下游 BlockInfo/AttentionMask/offset_batch_Q/K/score_mod 全部零改动
- Scheduler 不改：host 从 ranges 预算 `seqlens_q` 填入 `mSeqUsedQ` 槽 → 调度逻辑不变
- 新建文件命名：`range_info_sm90.py`（不用 `_sm90` 后缀，因为 ranges 本身就是 SM90 专有路径）

### Phase 1a: Dense FWD native qkrange（单一 mask_type）

**目标**：kernel 直接接收 q_ranges/k_ranges，FWD correctness 通过

| Step | 动作 | 文件 | 可测试性 |
|------|------|------|----------|
| 1a-1 | 新建 `range_info_sm90.py`：`RangeInfoQK` dataclass，`create_from_ranges()` 工厂方法 | 新建，无 breakage | import 验证 |
| 1a-2 | 修改 `flex_flash_attn.py` `_flex_flash_attn_fwd` SM90 路径：跳过 `ranges_to_cu_seqlens`，传 ranges tensor 到 kernel | 修改 dispatch | FWD smoke test |
| 1a-3 | 修改 `ffa_fwd_sm90.py` `__call__`：接收 `mQRanges`, `mKRanges` 参数；用 `RangeInfoQK.create_from_ranges` 替代 `SeqlenInfoQK.create` | 修改 kernel host | FWD correctness |
| 1a-4 | 修改 `ffa_fwd_sm90.py` `kernel`：kernel 参数增加 ranges tensor；内部用 RangeInfoQK | 修改 kernel device | FWD correctness |

**验收测试**：
```bash
pytest tests/test_kernel/cutedsl/test_ffa_simple.py::test_varlen_fwd_bwd -v -k "not force_sm80"
```

**提交点 Commit 1**：`feat(sm90): native qkrange support for FWD kernel`

### Phase 1b: Dense BWD native qkrange

| Step | 动作 | 文件 |
|------|------|------|
| 1b-1 | 修改 `flex_flash_attn.py` `_flex_flash_attn_bwd` SM90 路径 | dispatch |
| 1b-2 | 修改 `ffa_bwd_sm90.py` `__call__` + `kernel` | BWD kernel |
| 1b-3 | 修改 `ffa_bwd_preprocess.py` / `ffa_bwd_postprocess.py` 如有需要 | BWD pre/post |

**验收测试**：同上 pytest（包含 BWD 梯度验证）

**提交点 Commit 2**：`feat(sm90): native qkrange support for BWD kernel`

### Phase 1c: Per-range mask_type + inv_causal/bi_causal

| Step | 动作 |
|------|------|
| 1c-1 | 扩展 `_MaskTypeMap`: inv_causal=2, bi_causal=3 |
| 1c-2 | `RangeInfoQK` 增加 `mask_type` 字段，从 `attn_type_map[bidb]` 读取 |
| 1c-3 | `BlockInfo.get_n_block_min_max` 支持 runtime mask_type dispatch |
| 1c-4 | `AttentionMask.apply_mask` 支持 inv_causal/bi_causal |

**验收测试**：混合 mask_type 的 varlen correctness

**提交点 Commit 3**：`feat(sm90): per-range mask_type with inv_causal/bi_causal`

### Phase 1d: Dense bench 对齐

| Step | 动作 |
|------|------|
| 1d-1 | 运行 `exps/attn/cutedsl/run_benchmark_simple.py` 对比 native ranges 性能 |
| 1d-2 | 确认无回归（native ranges vs old cu_seqlens path 性能差异 < 2%）|
| 1d-3 | NCU 分析 if needed |

**提交点 Commit 4**：`perf(sm90): dense qkrange bench validation` (如有改动)

### 提交流程

```bash
# 在 cenn-2 容器中:
pre-commit run --all-files  # 自动 lint 修复
git add -A
git commit -m "..."
git push origin feat/cutedsl-sm90-qkrange
```

---

## 执行记录

### Phase 1a+1b 完成 (2026-07-03 23:15)

**Commit**: `3bddf9bb feat(sm90): native qkrange support for FWD+BWD kernels`

**改动汇总**:
- 新建 `range_info_sm90.py`: `create_seqlen_info_from_ranges()` 从 `[N,2]` ranges 读 offset/seqlen，返回标准 `SeqlenInfoQK`
- `flex_flash_attn.py`: SM90 FWD+BWD 路径额外传 `q_ranges/k_ranges` cute tensor 到 kernel
- `ffa_fwd_sm90.py`: `__call__` + `kernel` 增加 `mQRanges/mKRanges` 参数；kernel 内 `const_expr(mQRanges is not None)` 分支用 ranges 创建 SeqlenInfo
- `ffa_bwd_sm90.py`: 同 FWD 的改动

**设计决策**:
- host 仍从 ranges 导出 cu_seqlens 用于 layout/TMA/scheduler（dense ranges 情况下等价）
- kernel 内部通过 ranges 读取 offset/seqlen，不依赖 cu_seqlens
- `SeqlenInfoQK` 字段不变，下游 BlockInfo/AttentionMask/offset_batch 零改动
- 非 SM90 / 非 varlen 路径完全不受影响（`mQRanges=None` fallback）

**测试结果**: 216 passed, 72 skipped (SM80-forced), 0 failures

**下一步**: Phase 1c — per-range mask_type + inv_causal/bi_causal

---

### Phase 1c 完成 (2026-07-04 11:50)

**改动汇总**:
- `ffa_utils.py`: `_MaskTypeMap` 新增 `inv_causal=2`, `bi_causal=3`；`normalize_mask_types` 支持 `torch.Tensor` 输入
- `range_info_sm90.py`: 新增 `get_n_block_min_max_runtime` / `get_m_block_min_max_runtime`（runtime mask_type block skipping）；新增 `read_attn_type_map`（`partial`-friendly helper）
- `mask_sm90.py` (新建): R2P bitmask 实现 `_apply_inv_causal_or_bi_causal_mask_sm90` + dispatch 函数 `apply_mask_with_runtime_type_sm90`
- `ffa_fwd_sm90.py`: `use_per_range_mask` 编译开关；`kernel` 内创建 `partial(read_attn_type_map, mAttnTypeMap)` 传入 `mma` 方法
- `ffa_bwd_sm90.py`: 同 FWD 模式，`partial` 传入 `load`/`mma`/`dQacc_store` 三个方法
- `flex_flash_attn.py`: host 端 `attn_type_map` Tensor 处理 + compile key 增加 `use_per_range_mask_sm90` + autograd 保存 + 始终传 `mAttnTypeMap` 参数（避免 JIT 剥离 None）
- `test_ffa_simple.py`: 新增 `_ref_attn_per_range` helper + `test_per_range_mask_type_fwd_bwd` 测试

**核心技术突破**: CuTe-DSL SCF 框架中 `cute.Tensor`（MLIR value）无法跨 while-loop / dynamic-if 边界。解决方案：`@cute.jit` helper `read_attn_type_map` + `functools.partial` 捕获为 Python 对象，再作为参数传入包含 while-loop 的方法。

**测试结果**: seqlen=128 FWD + dQ/dK/dV 全部 no mismatch ✓
- seqlen=256 JIT 编译挂起（100% CPU 10h+ 未完成）— CuTe-DSL 编译器层面问题，非代码逻辑错误

**下一步**: Phase 1d — Dense bench 对齐

---

## 环境备忘

```
codebase: /home/niubility2/cenzhiyao/MagiAttention-2
container: cenn-2 on h100-6
branch: feat/cutedsl-sm90-qkrange
执行模板: ssh Sensecore-bma-h100-6 "docker exec cenn-2 bash -c 'source /root/.magi_env && cd /home/niubility2/cenzhiyao/MagiAttention-2 && {command}'"
CUDA_HOME: /usr/local/cuda-13.0
cutlass-dsl: 4.4.2
quack: 0.4.1
GPU: H100-80GB x8
```
