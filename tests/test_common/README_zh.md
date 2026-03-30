# test_common — 核心数据结构单元测试

本目录包含 `magi_attention.common` 中核心数据结构的单元测试，包括 `AttnRange`、`AttnRanges`、`AttnRectangle`、`AttnRectangles`、`AttnMaskType` 和 `AttnMask`。

## 双 Backend 测试

每个测试通过 `conftest.py` 中的 fixture 自动在 **Python 和 C++ (pybind11)** 两个 backend 下各运行一次，确保两个 backend 可互换。

### 工作原理

- `conftest.py` 定义了一个 `backend` fixture，`params=["python", "cpp"]`。
- 每个测试运行前，fixture 会设置 `MAGI_ATTENTION_CPP_BACKEND` 环境变量并重新加载 `magi_attention.common` 模块。
- 接受 `backend` 参数的测试方法会被采集两次——每个 backend 一次。
- **不接受** `backend` 参数的测试方法只运行一次，使用默认的 Python backend。

### 运行测试

```bash
# 运行所有测试（两个 backend 各一遍）
pytest tests/test_common/ -v

# 只运行 Python backend
pytest tests/test_common/ -v -k "python"

# 只运行 C++ backend
pytest tests/test_common/ -v -k "cpp"

# 运行单个文件
pytest tests/test_common/test_attn_range.py -v

# 运行单个测试方法
pytest tests/test_common/test_attn_range.py::TestAttnRange::test_simple_properties -v
```

### 输出示例

使用 `-v` 时，每个测试会显示 backend 后缀：

```
test_attn_range.py::TestAttnRange::test_simple_properties[python] PASSED
test_attn_range.py::TestAttnRange::test_simple_properties[cpp]    PASSED
```

## 测试文件说明

| 文件 | 测试内容 |
|------|----------|
| `test_attn_range.py` | `AttnRange` — 范围创建、集合运算、校验、序列化 |
| `test_attn_ranges.py` | `AttnRanges` — 范围列表、合并、排序、分块、本地映射 |
| `test_rectangle.py` | `AttnRectangle` — 矩形几何、切割、收缩、mask 类型判断 |
| `test_rectangles.py` | `AttnRectangles` — 集合操作、from_ranges、切割、面积 |
| `test_attn_mask.py` | `AttnMask` — mask 构造、sub-mask 提取（仅 Python backend） |
| `test_protocol_conformance.py` | 验证两个 backend 都满足 Protocol 接口约束 |
| `conftest.py` | 共享的 `backend` fixture，实现双 backend 参数化 |

## 新增测试指南

1. 测试方法接受 `backend` 参数即可自动双 backend 运行：
   ```python
   class TestMyFeature:
       def test_something(self, backend):
           from magi_attention.common import AttnRange
           r = AttnRange(0, 10)
           assert r.seqlen == 10
   ```

2. 在测试方法**内部**导入类（fixture 切换 backend 之后），始终通过 `magi_attention.common` 导入——不要直接从 `magi_attention.common.range` 等子模块导入。

3. 如果测试仅适用于 Python backend（例如测试没有 C++ 对应的 `AttnMask`），可以在文件中覆盖 fixture：
   ```python
   @pytest.fixture(params=["python"])
   def backend(request):
       yield request.param
   ```
