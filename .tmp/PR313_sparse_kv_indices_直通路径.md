# PR #313: sparse_kv_indices 直通 Kernel 路径

## 核心目标

用户直接传入 `sparse_kv_indices: (total_q, num_kv_heads, max_topk)` → kernel 直读，**消除整条 `q_ranges → k_ranges → flat_token_ids` 间接链路**。

## 数据流对比

```
旧：block_mask → Python 预处理(ranges/merge/metadata CUDA kernel) → 6个中间tensor → kernel 反查 k_range 定位 token
新：sparse_kv_indices → Python view(-1, max_topk) → kernel 尾部扫 -1 算 loop_count → 滑动指针直读全局行号
```

## 改动结构（28 文件，净减 ~1900 行）

### 1. Python 接口

**`flex_flash_attn.py`**：新增 `sparse_kv_indices` 参数（与 `q_ranges` 互斥），内部仅 `view` + 对齐 assert（2 行），删除整个 `compute_sparse_load_metadata` 调用链。

### 2. C++ 参数透传（5 层统一动作）

`flash.h` → `flex_flash_common` → `flex_flash_fwd` → `jinja` → `launch_template`

每层：删 3 个旧参数（`loop_count / invalid_count / equal_k_range_size`），加 2 个新参数（`sparse_kv_indices: int*` + `sparse_kv_max_topk: int`）。`q_ranges`/`k_ranges` 改为 optional。

### 3. Kernel 核心（mainloop，最大改动）

- **SparseBlockMeta 重写**：旧的 2 个 struct（~180 行，含 k_range 反查 + 7-way switch-case）→ 统一为 `template<IsProducer> SparseBlockMeta`（~50 行），构造时线性扫 `-1` 算 `actual_topk → loop_count`，`prefetch()` 靠指针滑动一步到位
- **mma 主循环统一**：旧的 `mma()` + `sparse_mma()`（两份 ~200 行重复代码）→ 单一 `mma()`，`if constexpr(SparseKV)` 分支
- **producer/consumer 统一**（`flash_fwd_kernel_sm90.h`）：旧的两套 producer 分支合并，consumer 统一调 `mainloop.mma()`
- **IsProducer 模板优化**：consumer warp group 的 `token_indices[8]` 等数组编译为 size=0，省 ~68 字节寄存器
- **INT32 溢出修复**：`unique_idx * max_topk` 在大序列时溢出，改为 `int64_t` 转换

### 4. 辅助改动

- **Epilogue**：参数从 `SeqlenInfo&` → `(offset_q, seqlen_q)`，sparse 时 `seqlen_q=1`
- **Scheduler**：新增 `SparseKV` 模板参数，`seqlen` 硬编码为 1，跳过 ranges 读取
- **Mask**：`apply_sparse_load` → `apply_sparse_kv`（纯重命名）
- 全局重命名 `SparseLoad` → `SparseKV`（C++ 模板/Python flag/JIT key）

### 5. 删除

- `preprocess_sparse_load.cu`（178 行，旧 CUDA 预处理 kernel）
- `test_block_sparse_attn.py`（1540 行，旧 sparse 测试）
- `compute_sparse_load_metadata` 的 pybind 绑定和类型注解

### 6. 新增测试

`test_sparse_kv_indices_attn.py`（543 行），按维度分层：

| Tier | 覆盖 |
|------|------|
| CI 必跑 | MQA(128/64/32) + pack_gqa |
| Slow | cross-batch 变 topk、head_dim(D=32/64/128)、长序列(S=65536 回归)、SwapAB |
