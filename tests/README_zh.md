# 测试指南

## Kernel Backend

`test_pipeline.py` 对**全部四种 kernel backend**（`ffa`、`sdpa`、`sdpa_ol`、`fa4`）进行参数化测试。每个 `attn_config` 通过 `BACKENDS` 标签标注适用的 backend 集合，不兼容的 (backend, config) 组合会自动跳过。

- `ffa` / `fa4` -- 使用大 seqlen 配置 + fp16/bf16，采用 norm + mismatch 容差检查。
- `sdpa` / `sdpa_ol` -- 使用小 seqlen 配置 + fp64，采用精确（EPSILON）容差检查。

backend 通过 `MAGI_ATTENTION_KERNEL_BACKEND` 环境变量控制（见下文）。

## 测试用例过滤

运行测试时，可以通过环境变量精细控制哪些参数化测试用例需要执行。所有过滤条件之间是 **AND** 关系——只有同时满足所有过滤条件的用例才会执行。

### 环境变量

| 环境变量 | 过滤维度 | 匹配目标 |
|---------|---------|---------|
| `MAGI_ATTENTION_TEST_WORLD_SIZE` | `world_size` | 逗号分隔的整数列表，如 `2` 或 `2,4` |
| `MAGI_ATTENTION_TEST_ATTN_CONFIG` | `attn_config` | `NAME` 字段 |
| `MAGI_ATTENTION_TEST_NUM_HEADS` | `num_heads` | 下划线分隔，如 `8_8` 表示 `(8, 8)` |
| `MAGI_ATTENTION_TEST_HEAD_DIM` | `head_dim` | 数值的字符串表示，如 `64` |
| `MAGI_ATTENTION_TEST_DTYPE` | `dtype` | 字符串表示，如 `torch.float16` |
| `MAGI_ATTENTION_TEST_BACKEND` | `backend` | 枚举值，如 `MagiAttentionKernelBackend.FFA` |

对于 dict 类型的参数（如 `attn_config`），过滤器匹配其 `NAME` 字段的值。对于其他类型的参数，使用 `str()` 转换后的字符串进行匹配。

> **注意：** `overlap_config` 和 `random_type_mapping` 已经从 `@parameterize` 维度移入 `FlagCombGenerator` 的 flag 组合中，由 heuristic 策略自动选取。因此 `MAGI_ATTENTION_TEST_OVERLAP_CONFIG` 和 `MAGI_ATTENTION_TEST_RANDOM_TYPE_MAPPING` 环境变量不再生效。

环境变量的值是**逗号分隔的 fnmatch 模式列表**，支持 `*`、`?` 等通配符。

### 使用示例

```bash
# 只运行 world_size=2
MAGI_ATTENTION_TEST_WORLD_SIZE=2 pytest tests/test_pipeline.py

# 只运行 world_size=2 和 4
MAGI_ATTENTION_TEST_WORLD_SIZE=2,4 pytest tests/test_pipeline.py

# 只运行某个 attn_config
MAGI_ATTENTION_TEST_ATTN_CONFIG=full_attn_14k pytest tests/test_pipeline.py

# 通配符匹配多个 attn_config
MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_*" pytest tests/test_pipeline.py

# 只运行 head_dim=128
MAGI_ATTENTION_TEST_HEAD_DIM=128 pytest tests/test_pipeline.py

# 只运行 MHA (8,8)
MAGI_ATTENTION_TEST_NUM_HEADS=8_8 pytest tests/test_pipeline.py

# 只运行 float16
MAGI_ATTENTION_TEST_DTYPE="*float16*" pytest tests/test_pipeline.py

# 只运行 sdpa_ol backend 的测试
MAGI_ATTENTION_TEST_BACKEND="*SDPA_OL*" pytest tests/test_pipeline.py

# 组合过滤：full_attn 配置 + head_dim=64
MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_*" \
MAGI_ATTENTION_TEST_HEAD_DIM=64 \
    pytest tests/test_pipeline.py

# 逗号分隔匹配多个值
MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_14k,uneven_full_attn_10k" \
    pytest tests/test_pipeline.py
```

## Flag 锁定（用户预设环境变量）

如果用户在运行测试前已经设置了某个 flag 对应的环境变量，测试框架会**尊重用户的设置**，将该 flag 锁定为用户设置的值，`FlagCombGenerator` 不会覆盖它。

### 使用示例

```bash
# 强制所有测试使用 qo_comm=1
MAGI_ATTENTION_QO_COMM=1 pytest tests/test_pipeline.py

# 强制使用 deterministic mode
MAGI_ATTENTION_DETERMINISTIC_MODE=1 pytest tests/test_pipeline.py

# 强制使用 native grpcoll + high precision reduce
MAGI_ATTENTION_NATIVE_GRPCOLL=1 \
MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE=1 \
    pytest tests/test_pipeline.py

# 强制 CUDA_DEVICE_MAX_CONNECTIONS=1
CUDA_DEVICE_MAX_CONNECTIONS=1 pytest tests/test_pipeline.py

# 组合使用：锁定 flag + 过滤测试用例
MAGI_ATTENTION_QO_COMM=1 \
MAGI_ATTENTION_TEST_ATTN_CONFIG=full_attn_14k \
    pytest tests/test_pipeline.py
```

### 支持的 flag 环境变量

| 环境变量 | 对应 flag | 值类型 |
|---------|----------|-------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `device_max_connections` | 整数（如 `1` 或 `8`）|
| `MAGI_ATTENTION_KERNEL_BACKEND` | kernel backend | `ffa` / `sdpa` / `sdpa_ol` / `fa4` |
| `MAGI_ATTENTION_DETERMINISTIC_MODE` | `deterministic_mode` | `0` / `1` |
| `MAGI_ATTENTION_HIERARCHICAL_COMM` | `enable_hier_comm` | `0` / `1` |
| `MAGI_ATTENTION_QO_COMM` | `enable_qo_comm` | `0` / `1` |
| `MAGI_ATTENTION_NATIVE_GRPCOLL` | `enable_native_grpcoll` | `0` / `1` |
| `MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE` | `fwd_hp_reduce` | `0` / `1` |
| `MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE` | `bwd_hp_reduce` | `0` / `1` |
| `MAGI_ATTENTION_FLATTEN_HEAD_GROUPS` | `flatten_head_groups` | `0` / `1` |
| `MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE` | `bwd_hide_tail_reduce` | `0` / `1` |

> **注意：** 旧版环境变量 `MAGI_ATTENTION_SDPA_BACKEND` 和 `MAGI_ATTENTION_FA4_BACKEND` 仍然支持以兼容旧代码，但**不能**与 `MAGI_ATTENTION_KERNEL_BACKEND` 同时设置。

## Flag 组合生成器

测试使用 `FlagCombGenerator` 自动生成 flag 的组合，包括环境变量 flag（如 `deterministic_mode`、`enable_qo_comm`）和测试参数 flag（`overlap_config`、`random_type_mapping`）。

这些 flag 使用 `heuristic` 策略生成组合，而不是笛卡尔积全展开，大幅减少测试组合数。

### 上下文感知过滤

`FlagCombGenerator.get_next_valid_comb(test_config, is_valid_fn)` 方法会根据当前测试上下文自动过滤不合法的 flag 组合。约束规则包括：

- `no_overlap` 模式下不允许 `qo_comm=True`
- `qo_comm=True` 时只允许 `disable_mso` 或 `no_overlap` 的 overlap 配置
- `qo_comm=True` 时不允许 `hier_comm=True` 或 `bwd_hide_tail_reduce=True`
- `native_grpcoll=True` 时不允许 `hier_comm=True`
- `flatten_head_groups=True` 必须配合 `qo_comm=True`，且不兼容 sink 和 `return_max_logits`
- `fa4` backend 不允许 `deterministic`、`fwd_hp_reduce`、`bwd_hp_reduce`、`qo_comm`、`sink`、`bwd_hide_tail_reduce`
- `sdpa` / `sdpa_ol` backend 不允许 `native_grpcoll`
- 等等

不合法的组合不会被浪费——它们会被延迟到其他合法的测试上下文中重新使用。如果所有可能的组合都不合法，会快速抛出 `RuntimeError` 而不是无限等待。

## OverlapConfig degree 语义

`OverlapConfig` 通过 `degree` 参数统一控制 overlap 行为：

| degree | 含义 |
|--------|------|
| `0` | **no overlap** -- blocking 通信 + 合并 attn_arg，无 LSE reduce 精度损失 |
| `1` | local + 1 remote stage，不分 chunk（无多阶段重叠） |
| `N (N>=2)` | local + N remote stages（静态多阶段重叠） |
| `None` | 动态模式（solver 自动决定最优 degree） |

```python
# no overlap（最高精度）
OverlapConfig(degree=0)

# 不使用多阶段重叠
OverlapConfig(degree=1)

# 4 阶段静态重叠
OverlapConfig(degree=4, mode=AttnOverlapMode.STATIC)

# 动态重叠
OverlapConfig(degree=None, mode=AttnOverlapMode.DYNAMIC)
```
