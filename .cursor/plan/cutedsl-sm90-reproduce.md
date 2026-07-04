# CuTe-DSL SM90 Reproduce Plan

> 2026-07-03 14:30 | main@2726855f

## 1. Intent Analysis

### Division of Labor
| Role | Phase 1 | Phase 2 |
|------|---------|---------|
| Chenhao | SM100 general features (native ranges, reductions, GQA) | SM90 advanced features |
| You | SM90 reproduce (reimplement Cutlass SM90 features in CuTe-DSL) | SM100 sparse attn |

- reproduce = dense + sparse functional parity, bench TFLOPS alignment
- advanced = new features NOT in Cutlass SM90 (attn_sink, muon, etc.)

## 2. Feature Gap: CuTe-DSL SM90 vs Cutlass SM90

| Feature | Cutlass SM90 | CuTe-DSL SM90 | Gap |
|---------|-------------|---------------|-----|
| Mask | full/causal/inv_causal/bi_causal+per-range | full/causal only | BIG |
| Sequence | native q_ranges/k_ranges+attn_type_map | hack to cu_seqlens | BIG |
| GQA FWD | PackGQA | PackGQA | none |
| GQA BWD | PackGQA+FlattenGQA | pack_gqa=False hardcoded | MED |
| Block Sparse FWD | sparse_load | FA4-style block_sparse | different system |
| Block Sparse BWD | sparse_load LoopK | FA4-style | different system |
| Index Sparse | FWD+BWD index_attn | NONE | HUGE |
| BWD LoopQ | swap_bwd_qk_loop | NONE | BIG |
| Deterministic | yes | yes | none |
| Softcap | yes | yes (score_mod) | none |
| score_mod/mask_mod | no (C++) | yes | CuTe-DSL stronger |
| Reductions | FWD atomic lse, BWD atomic dKV | no atomic | MED |
| Range merging | auto_range_merge | none | LOW |

## 3. Two Sparse Systems Are Incompatible

- Cutlass: ranges+attn_type_map based (MagiAttention native block_meta.h)
- CuTe-DSL: FA4-style BlockMask (mask_block_cnt/idx from TorchFlexAttn)
- Index Sparse exists ONLY in Cutlass; zero infrastructure in CuTe-DSL

## 4. qkrange: Keep and Implement Natively (Not Overthrow)

- PR327 TODO first item = "Native ranges + mask types in-kernel"
- Interface already accepts q_ranges/k_ranges; current hack is temporary
- This is general feature for Chenhao Phase1; you wait and sync to SM90

## 5. blockmeta: Not Present in CuTe-DSL

- CuTe-DSL uses FA4-fork BlockSparseTensors (kv_num_blocks/kv_indices)
- No BlockSparseBlockMeta or IndexSparseBlockMeta equivalent exists

## 6. Priority

- P0: Dense bench alignment (0.5d) - ffa vs fa3 regression
- P1: Block sparse SM90 verify + bench (1-2d)
- P2: GQA BWD pack_gqa (2-3d)
- P3: BWD LoopQ + Index Sparse (Phase1 skip, 3-5 weeks total)

Phase 1 deliverable: Dense + Block Sparse bench report + GQA BWD (about 1-2 weeks)

## 7. Open Questions

1. Sparse system: FA4-style or port native sparse_load/index_attn?
2. Index Sparse: Phase1 must or Phase2?
3. Native ranges timeline: wait for Chenhao SM100?
4. Bench alignment threshold: plus-minus 5 percent?

---

## Appendix: Deep Analysis (2026-07-03 15:21)

综合三路并行调研结果，补充细节。

### A. "reproduce" 范围精确推断

**结论：reproduce = dense功能对齐 + sparse功能移植，bench TFLOPS为验收标准。**

推理链：
1. 领导说"cutlass FFA index-attn on SM90基本收了之后" → 你在 Cutlass 做的 SM90 工作**包含 index_attn**
2. "从cutedsl FFA的sm90 reproduce开始" = 把 Cutlass SM90 做过的搬到 CuTe-DSL
3. 但 CuTe-DSL 的 block sparse 是 FA4 风格，与 Cutlass 的 sparse_load 体系完全不同
4. Index Sparse 在 CuTe-DSL 中零基础设施（搜索 IndexSparse/index_sparse/index_attn 结果为空）
5. **Phase 1 策略：先对齐 dense，验证 FA4 block sparse，GQA BWD。Index Sparse 工作量太大（3-5周），留后面。**

### B. "advanced feature" 精确定义

PR #327 TODO 中明确标注为 "future" 的特性：
- attn_sink (learnable sink, SSH layout)
- Muon qk-clip (return_max_logits)
- configurable output dtype
- deterministic + block-sparse BWD
- cat_gqa BWD
- torch.compile custom ops integration

这些在 Cutlass SM90 上**部分已有**（sink, muon, deterministic），但大部分在 CuTe-DSL 上都没实现。
"advanced" = 这些前沿/边缘特性，陈浩 Phase 2 去 SM90 做。
**你 Phase 1 只需追平核心功能路径（dense + sparse + GQA BWD），不需要做 advanced。**

### C. 两套 Sparse 体系对比（关键架构问题）

| 维度 | Cutlass FFA (block_meta.h) | CuTe-DSL (sparse_utils.py) |
|------|---------------------------|---------------------------|
| **数据来源** | MagiAttention 自研 | FA4/FlexAttention fork |
| **Block Sparse 表示** | q_ranges + k_ranges + attn_type_map → BlockSparseBlockMeta (token_indices scatter) | mask_block_cnt + mask_block_idx (precomputed block lists) |
| **Index Sparse 表示** | IndexSparseBlockMeta (total_q, nhk, max_topk) | **完全不存在** |
| **Inner load** | TMA 2D / cp.async scatter (k_block_size决定) | TMA atom (block sparse 继承 FA4 路径) |
| **Range merge** | auto_range_merge via C++ ext | 无 |
| **BWD loop** | LoopK (default) / LoopQ (swap_bwd_qk_loop) | LoopK only (BWD outer=K, inner=Q) |
| **GQA BWD** | PackGQA + FlattenGQA + CatGQA | pack_gqa=False 硬编码 |

**核心差异**：Cutlass 用 ranges 做 runtime scatter（fill_token_indices → smem），CuTe-DSL 用预计算 block lists（无 runtime scatter）。

**开放问题**：是在 CuTe-DSL 上移植 Cutlass 的 ranges-scatter 体系？还是继续用 FA4 的 precomputed block lists？
- 如果要做 Index Sparse → 必须引入类似 ranges-scatter 的机制（FA4 的 block lists 不支持 token-level 粒度）
- 如果只做 Block Sparse → FA4 的方案可能够用，但需要验证性能

### D. qkrange 设计分析

**现状**：
- `flex_flash_attn.py` 接收 `q_ranges/k_ranges`
- `ranges_to_cu_seqlens()` 将其降级为 cu_seqlens（要求连续、非重叠、从0开始）
- kernel 层（ffa_fwd_sm90.py, ffa_bwd_sm90.py）只看 `mCuSeqlensQ/K`，完全不知道 ranges
- `normalize_mask_types()` 对 Tensor 类型的 mask_types 直接 raise NotImplementedError

**领导意图推断**：
- PR #327 TODO 第一条 = "Native ranges + mask types in-kernel"
- 接口已预留 ranges 参数 + "Step-1 hack" 注释 → **设计意图是最终支持**
- 这属于 general feature → 陈浩 Phase 1 在 SM100 先做，你后续同步到 SM90
- **不会推翻**——只会扩展 kernel 让它直接感知 ranges

**Native ranges 在 CuTe-DSL 中需要做什么**：
1. 停止在 host 端降级 → 直接传 q_ranges/k_ranges tensor 到 kernel
2. SeqlenInfoQK 从 ranges[bidb] 读取 offset/seqlen（参考 Cutlass seqlen.h）
3. 支持 per-range mask_type（读 attn_type_map[bidb]）
4. Tile scheduler 支持任意 ranges（含 range merge batch loop）

### E. Cutlass SM90 BWD 的关键架构（reproduce 需理解）

**LoopK vs LoopQ**：
- LoopQ (default, swap_bwd_qk_loop=False): outer=K, inner=Q → dK/dV 直接写
- LoopK (swap_bwd_qk_loop=True): outer=Q, inner=K → dQ 直接写
- Index Sparse BWD **两种都需要**：LoopQ 重建 inner_indices，LoopK 复用 FWD indices

**Register allocation**：
- Cutlass 有 `_ffa_register_quota()` 集中管理 producer/consumer registers
- CuTe-DSL 无等价机制（但 kernel 可通过 cute 框架设置）

**Inner load modes**（与 k_block_size 绑定）：
- k_block_size >= kBlockN → TMA 2D inner load
- k_block_size < kBlockN → cp.async scatter inner load
- CuTe-DSL 的 TMA atom 是否支持 2D scatter 需要调研

### F. 执行计划（修订版）

| # | 任务 | 优先级 | 前置 | 工作量 |
|---|------|--------|------|--------|
| 1 | Dense regression bench: ffa vs fa3, seqlen 1K-32K, MHA+GQA, FWD+BWD | P0 | 无 | 0.5天 |
| 2 | FA4-style block sparse SM90 correctness 验证 (构造 BlockMask 测试) | P1 | 无 | 1天 |
| 3 | Block sparse bench 对比 (CuTe-DSL vs Cutlass, 同 density 场景) | P1 | #2 | 0.5天 |
| 4 | GQA BWD pack_gqa 实现 (修改 ffa_bwd_sm90.py) | P2 | 无 | 2-3天 |
| 5 | Atomic reductions (FWD lse, BWD dKV) | P2 | 无 | 1-2天 |
| 6 | inv_causal / bi_causal mask types | P2 | 无 | 1天 |
| 7 | 输出：特性对齐报告 + bench 数据 | - | #1-6 | 0.5天 |
| 8 | BWD LoopQ / swap_qk | P3 | #4 | 5-7天 |
| 9 | Index Sparse FWD+BWD | P3 | #8 + native ranges | 2-4周 |
| 10 | Native ranges (同步陈浩 SM100 设计) | P4 | 陈浩完成 | 1-2周 |

**Phase 1 目标**（约2周）：#1-7 完成 → Dense+BlockSparse bench对齐 + GQA BWD + 基础特性补齐
**Phase 2 预备**：#8-10 → 为 SM100 sparse attn 打基础

---

## Appendix B: Block Sparse 表达体系 + CuTe-DSL 自定义结构可行性 (2026-07-03 18:02)

### B1. FA4/FlexAttention BlockMask 是标准表达——有据可查

CuTe-DSL 的 `BlockSparseTensorsTorch(mask_block_cnt, mask_block_idx, ...)` 格式确实是 **FA4 + PyTorch FlexAttention 的标准**，不是自创的。

**证据链**：

1. **PyTorch 官方 `BlockMask` 类**（`torch.nn.attention.flex_attention.BlockMask`）：
   - 核心字段：`kv_num_blocks`(B,H,M), `kv_indices`(B,H,M,N), `full_kv_num_blocks`, `full_kv_indices`
   - 用户通过 `create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN)` 从一个 `mask_mod` 函数自动生成
   - 源码：https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py

2. **FA4（Dao-AILab/flash-attention）的 `BlockSparseTensors`**：
   - 字段名：`mask_block_cnt`(=kv_num_blocks), `mask_block_idx`(=kv_indices), `full_block_cnt`, `full_block_idx`
   - 就是 PyTorch BlockMask 换了个名字，语义完全一致
   - 源码：`flash_attn/cute/block_sparsity.py`

3. **我们的 CuTe-DSL FFA**：
   - 直接 fork 自 FA4，`BlockSparseTensorsTorch` 就是 FA4 的 `BlockSparseTensors` + torch 版
   - 命名和语义完全一致

**FlexAttention 的工作流**（标准用法）：
```
用户定义 mask_mod(b, h, q_idx, kv_idx) → bool
    ↓ create_block_mask(mask_mod, B, H, Q, K)
    ↓ 在 GPU 上对每个 block 采样 mask_mod，分类 full/partial/empty
    ↓ 输出 BlockMask(kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices)
    ↓ flex_attention(q, k, v, block_mask=block_mask)
```

**关键认知**：这套体系的设计哲学是"block-level sparsity"——以 tile_m × tile_n 为粒度，预计算哪些 block 需要算。不支持 token-level 粒度（Index Sparse 需要的）。

### B2. 与 MagiAttention 原生体系的根本差异

| 维度 | FlexAttention/FA4 (block lists) | MagiAttention (ranges+blockmeta) |
|------|--------------------------------|----------------------------------|
| 粒度 | block-level（tile_m × tile_n = 128×128） | range-level（任意长度区间）+ token-level（Index Sparse） |
| 输入 | mask_mod 函数 → 自动生成 block lists | q_ranges/k_ranges + attn_type_map |
| Sparse 表示 | cnt/idx 数组（每个 Q-block 对应一组 K-block 列表） | BlockMeta 结构体（含 ranges、indices、merge info） |
| Index Sparse | 不支持 | index_attn_indices (total_q, nhk, topk) |
| 实现路径 | producer 按 block list 遍历，TMA 加载整 block | producer 按 ranges scatter，支持 TMA 2D / cpasync |

### B3. CuTe-DSL 自定义结构体能力分析

**CuTe-DSL 支持两种自定义数据结构方式**：

**方式一：`@cute.struct`（设备端共享内存结构体）**
用于定义 SMEM 布局，类似 C++ struct。已在 FFA 中广泛使用：
```python
@cute.struct
class SharedStorageQKV:
    mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, Q_stage * 2]
    sQ: cute.struct.Align[cute.struct.MemRange[dtype, cosize], 128]
    sK: ...
```
限制：这是共享内存分配器，不是通用的数据容器。字段必须是标量/数组类型。

**方式二：`NamedTuple` + `__new_from_mlir_values__`（kernel 参数传递）**
用于将多个 tensor 打包传入 kernel。CuTe-DSL 的 JIT 编译器会自动展开 NamedTuple 的字段作为 kernel 参数：
```python
class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None = None
    ...
    def __new_from_mlir_values__(self, values):
        # 处理 None 字段的重建逻辑
```
**这就是 CuTe-DSL 传递自定义数据结构到 kernel 的标准方式**。

**方式三：`cpasync.CopyG2SOp` + `cp_async_commit_group/wait_group`（非 TMA 的异步拷贝）**
SM90 FWD kernel 中 pack_gqa 的 Q 加载回退路径已经在用：
```python
atom_async_copy = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    dtype, num_bits_per_copy=128,
)
# ... 用 cute.copy() 发起 → cp_async_commit_group() 提交 → 同步
```
**cpasync scatter load 在 CuTe-DSL 中完全可行**——已有工作代码。

### B4. 在 CuTe-DSL 中实现 BlockMeta 式结构——可行性评估

**问题**：能否在 CuTe-DSL 的 kernel 里实现类似 Cutlass 的 `BlockSparseBlockMeta` / `IndexSparseBlockMeta`？

**回答：完全可行，且已有先例。**

`BlockSparseTensors(NamedTuple)` 就是一个 BlockMeta 的等价物：
- Cutlass 的 `BlockSparseBlockMeta` 通过 C++ template 参数传入 kernel → CuTe-DSL 用 NamedTuple + `@cute.jit` 参数传入
- Cutlass 的 `fill_token_indices()` 将 range indices 写入 SMEM → CuTe-DSL 可以用同样的逻辑（`@cute.struct` 分配 SMEM + cpasync/TMA 写入）
- Cutlass 的 inner loop 按 `token_indices` scatter load K/V → CuTe-DSL 可以用 `cpasync.CopyG2SOp` 做同样的 scatter

**具体来说，要实现 Index Sparse，需要**：
1. 定义 `IndexSparseTensors(NamedTuple)` — 包含 `indices`, `topk_cnt`, `inv_indices` 等 cute.Tensor
2. Kernel 参数传递 — 和 BlockSparseTensors 一样通过 `@cute.jit` 自动展开
3. Inner loop scatter load — 用 `cpasync.CopyG2SOp` 按 token index 加载 K/V 到 SMEM（已有 pack_gqa Q load 的先例）
4. BWD inner store — 用 `cpasync.CopyReduceBulkTensorTileS2GOp` (TMA reduce-add) 或 scalar atomicAdd

**技术风险点**：
- `cpasync.CopyG2SOp` 的 scatter 模式性能如何？需要 bench
- TMA 2D descriptor 在 CuTe-DSL 中的 scatter 支持需要验证（可能需要 `update_tma_descriptor` 动态更新 base ptr）

### B5. 结论与建议

1. **CuTe-DSL 的 block sparse 是 FA4 标准——不是自创**。FlexAttention BlockMask 是业界标准，我们 fork 使用没有问题。

2. **MagiAttention 的 ranges+blockmeta 体系和 FA4 的 block lists 体系可以共存**。host 端做转换（ranges → block lists 或反向），kernel 端用 NamedTuple 传入即可。

3. **在 CuTe-DSL 中实现类 BlockMeta 的自定义结构完全可行**：
   - NamedTuple + `__new_from_mlir_values__` 用于 kernel 参数传递
   - `@cute.struct` 用于 SMEM 布局
   - cpasync scatter load 已有工作代码

4. **Index Sparse 的实现路径是通的**，技术上没有不可逾越的障碍。关键是工作量大（3-5周）。

5. **建议路径**：
   - Phase 1 保持 FA4 标准的 block sparse → 验证性能
   - Index Sparse 用 NamedTuple 自定义结构 + cpasync scatter → Phase 2
   - 不需要推翻 FA4 体系，两套可以共存

---

## 2025-07-04 14:28 — IndexSparseBlockMeta 模板参数重命名 + SparseKBlockSize 校验

### 分析：kInnerBlockSize 与 KBlockSize 的实际使用场景

| SparseKBlockSize (kbs) | InnerBlockSize (tile) | 触发路径 | 场景 |
|---|---|---|---|
| **1** | kBlockN (128) | FWD scatter CpAsync; BWD InnerLoopK scatter | token-level IndexSparse (主力场景) |
| **≥kBlockN (128)** | kBlockN (128) | FWD TMA 2D contiguous; BWD InnerLoopQ TMA 2D | block-level IndexSparse / BlockSparse |
| **1<kbs<kBlockN (e.g. 8,16,32,64)** | kBlockN (128) | ⚠ 代码编译通过但**语义不安全** | 当前无测试无校验 |

### 问题：1<kbs<kBlockN 的灰色地带

1. **FWD**：`_is_contiguous = KBlockSize >= kBlockN`。kbs=32 → scatter CpAsync。**编译通过、可运行，但 fill_token_indices 的 block-level 分支假设 kbs 整除 kInnerBlockSize，若 kbs=32<kInnerBlockSize=128 则 `tiles_per_kblock=0` → 除零 UB**。
2. **BWD InnerLoopQ**：`SwapBwdQKLoop && KBlockSize >= kBlockN` 才启用 TMA。kbs=32 → CpAsync scatter fallback。`get_n_block_abs()` 有 `static_assert(kKBlockSize >= kInnerBlockSize)` 保护。但 InnerLoopQ 的 fill_token_indices (Q 侧) 不依赖 kbs，所以**不会触发 UB**。
3. **BWD InnerLoopK**：kbs>1 时走 `fill_token_indices` 的 block-level 分支。同样 `tiles_per_kblock = kbs / kInnerBlockSize`，kbs<kInnerBlockSize → 除零。
4. **Python 侧**：完全没有 `k_block_size` 的 power-of-2/alignment 校验。`k_block_size=7` 会一路透传到 kernel，产生 UB。

### 结论：当前只有两种安全值：kbs=1 或 kbs≥kBlockN(128)

中间值 (2..127) 会在 C++ 的 `fill_token_indices` block-level 分支和 `get_n_block_abs` 中除零。需要**加入显式校验**。

### 执行计划

**P1（高，本轮执行）：模板参数重命名**
- `kInnerBlockSize_` → `InnerBlockSize_`，静态成员 → `InnerBlockSize`（与 BlockSparseBlockMeta 一致）
- `KBlockSize_` → `SparseKBlockSize_`，静态成员 → `SparseKBlockSize`
- 修改 block_meta.h、mainloop_fwd、mainloop_bwd 的实例化点
- 同步修改注释和 static_assert 消息

**P2（高，本轮执行）：Python 侧 k_block_size 校验**
- `flex_flash_attn.py` 入口处加 assert：
  ```python
  assert k_block_size >= 1
  assert k_block_size == 1 or k_block_size >= tile_size, (
      f"k_block_size must be 1 (token-level) or >= tile_size ({tile_size}), "
      f"got {k_block_size}. Values in (1, tile_size) cause undefined behavior."
  )
  ```
- 同时保证 `k_block_size` 是 2 的幂次（`k_block_size & (k_block_size - 1) == 0`）

**P3（低，后续）：支持中间 kbs (8/16/32/64)**
- 需要让 fill_token_indices 在 kbs<kBlockN 时走 scatter 分支而非 block-level 分支
- 需要单独的 tile packing 逻辑（一个 tile 跨多个 K block）
- 复杂度高，优先级低——当前 kbs=1 和 kbs=128 已覆盖实际需求
