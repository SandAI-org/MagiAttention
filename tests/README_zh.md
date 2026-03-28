# 测试指南

## 测试用例过滤

运行测试时，可以通过环境变量精细控制哪些参数化测试用例需要执行。所有过滤条件之间是 **AND** 关系——只有同时满足所有过滤条件的用例才会执行。

### 环境变量

| 环境变量 | 过滤维度 | 匹配目标 |
|---------|---------|---------|
| `MAGI_ATTENTION_TEST_WORLD_SIZE` | `world_size` | 逗号分隔的整数列表，如 `2` 或 `2,4` |
| `MAGI_ATTENTION_TEST_ATTN_CONFIG` | `attn_config` | `NAME` 字段 |
| `MAGI_ATTENTION_TEST_OVERLAP_CONFIG` | `overlap_config` | `NAME` 字段 |
| `MAGI_ATTENTION_TEST_NUM_HEADS` | `num_heads` | 字符串表示，如 `(8, 8)` |
| `MAGI_ATTENTION_TEST_HEAD_DIM` | `head_dim` | 数值的字符串表示，如 `64` |
| `MAGI_ATTENTION_TEST_DTYPE` | `dtype` | 字符串表示，如 `torch.float16` |
| `MAGI_ATTENTION_TEST_RANDOM_TYPE_MAPPING` | `random_type_mapping` | `True` 或 `False` |

对于 dict 类型的参数（如 `attn_config`、`overlap_config`），过滤器匹配其 `NAME` 字段的值。对于其他类型的参数，使用 `str()` 转换后的字符串进行匹配。

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

# 只运行 no_overlap 的 overlap_config
MAGI_ATTENTION_TEST_OVERLAP_CONFIG=no_overlap pytest tests/test_pipeline.py

# 只运行 head_dim=128
MAGI_ATTENTION_TEST_HEAD_DIM=128 pytest tests/test_pipeline.py

# 只运行 MHA (8,8)
MAGI_ATTENTION_TEST_NUM_HEADS="(8, 8)" pytest tests/test_pipeline.py

# 只运行 float16
MAGI_ATTENTION_TEST_DTYPE="*float16*" pytest tests/test_pipeline.py

# 组合过滤：full_attn 配置 + no_overlap + head_dim=64
MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_*" \
MAGI_ATTENTION_TEST_OVERLAP_CONFIG=no_overlap \
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
MAGI_ATTENTION_TEST_OVERLAP_CONFIG=disable_mso \
    pytest tests/test_pipeline.py
```

### 支持的 flag 环境变量

| 环境变量 | 对应 flag | 值类型 |
|---------|----------|-------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `device_max_connections` | 整数（如 `1` 或 `8`）|
| `MAGI_ATTENTION_DETERMINISTIC_MODE` | `deterministic_mode` | `0` / `1` |
| `MAGI_ATTENTION_HIERARCHICAL_COMM` | `enable_hier_comm` | `0` / `1` |
| `MAGI_ATTENTION_QO_COMM` | `enable_qo_comm` | `0` / `1` |
| `MAGI_ATTENTION_NATIVE_GRPCOLL` | `enable_native_grpcoll` | `0` / `1` |
| `MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE` | `fwd_hp_reduce` | `0` / `1` |
| `MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE` | `bwd_hp_reduce` | `0` / `1` |
| `MAGI_ATTENTION_FLATTEN_HEAD_GROUPS` | `flatten_head_groups` | `0` / `1` |
| `MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE` | `bwd_hide_tail_reduce` | `0` / `1` |

## Flag 组合生成器

测试使用 `FlagCombGenerator` 自动生成环境变量 flag 的组合。

### 上下文感知过滤

`FlagCombGenerator.get_next_valid_comb(test_config, is_valid_fn)` 方法会根据当前测试上下文自动过滤不合法的 flag 组合。约束规则包括：

- `no_overlap` 模式下不允许 `qo_comm=True`
- `qo_comm=True` 时只允许 `disable_mso` 或 `no_overlap` 的 overlap 配置
- `qo_comm=True` 时不允许 `hier_comm=True` 或 `bwd_hide_tail_reduce=True`
- `native_grpcoll=True` 时不允许 `hier_comm=True`
- `flatten_head_groups=True` 必须配合 `qo_comm=True`，且不兼容 sink 和 `return_max_logits`
- 等等

不合法的组合不会被浪费——它们会被延迟到其他合法的测试上下文中重新使用。

## no_overlap 模式

`overlap_config` 中的 `no_overlap` 选项使用 blocking 通信 + 合并 attn_arg 的方式执行分布式注意力，完全避免 LSE reduce 带来的精度损失。

```python
OverlapConfig(no_overlap=True)
```
