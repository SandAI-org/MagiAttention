# test_common — Unit Tests for Core Data Structures

This directory contains unit tests for the core data structures in `magi_attention.common`, including `AttnRange`, `AttnRanges`, `AttnRectangle`, `AttnRectangles`, `AttnMaskType`, and `AttnMask`.

## Dual-Backend Testing

Every test automatically runs against **both** the Python and C++ (pybind11) backends via a `conftest.py` fixture. This ensures the two backends remain interchangeable.

### How It Works

- `conftest.py` defines a `backend` fixture with `params=["python", "cpp"]`.
- Before each test, the fixture sets `MAGI_ATTENTION_CPP_BACKEND` and reloads the `magi_attention.common` modules.
- Test methods that accept a `backend` parameter will be collected twice — once per backend.
- Test methods **without** the `backend` parameter run once with the default (Python) backend.

### Running Tests

```bash
# Run all tests (both backends)
pytest tests/test_common/ -v

# Run only Python backend
pytest tests/test_common/ -v -k "python"

# Run only C++ backend
pytest tests/test_common/ -v -k "cpp"

# Run a specific test file
pytest tests/test_common/test_attn_range.py -v

# Run a specific test method
pytest tests/test_common/test_attn_range.py::TestAttnRange::test_simple_properties -v
```

### Output

With `-v`, each test shows its backend suffix:

```
test_attn_range.py::TestAttnRange::test_simple_properties[python] PASSED
test_attn_range.py::TestAttnRange::test_simple_properties[cpp]    PASSED
```

## Test Files

| File | What It Tests |
|------|---------------|
| `test_attn_range.py` | `AttnRange` — range creation, set operations, validation, serialization |
| `test_attn_ranges.py` | `AttnRanges` — list of ranges, merge, sort, chunk, local mapping |
| `test_rectangle.py` | `AttnRectangle` — rectangle geometry, cut, shrink, mask type detection |
| `test_rectangles.py` | `AttnRectangles` — collection operations, from_ranges, cut, area |
| `test_attn_mask.py` | `AttnMask` — mask construction, sub-mask extraction (Python-only) |
| `test_protocol_conformance.py` | Verifies both backends satisfy the Protocol contracts |
| `conftest.py` | Shared `backend` fixture for dual-backend parametrization |

## Adding New Tests

1. Create test methods that accept `backend` as a parameter:
   ```python
   class TestMyFeature:
       def test_something(self, backend):
           from magi_attention.common import AttnRange
           r = AttnRange(0, 10)
           assert r.seqlen == 10
   ```

2. Import classes **inside** the test method body (after the fixture has switched the backend), always through `magi_attention.common` — never directly from submodules like `magi_attention.common.range`.

3. If a test only applies to the Python backend (e.g., testing `AttnMask` which has no C++ counterpart), override the fixture locally:
   ```python
   @pytest.fixture(params=["python"])
   def backend(request):
       yield request.param
   ```
