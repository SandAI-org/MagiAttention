# Test Guide

## Test Case Filtering

You can use environment variables to precisely control which parameterized test cases are executed. All filter conditions use **AND** logic -- only test cases that satisfy all active filters will run.

### Environment Variables

| Environment Variable | Parametrize Dimension | Match Target |
|---------------------|----------------------|-------------|
| `MAGI_ATTENTION_TEST_WORLD_SIZE` | `world_size` | Comma-separated integers, e.g. `2` or `2,4` |
| `MAGI_ATTENTION_TEST_ATTN_CONFIG` | `attn_config` | `NAME` field |
| `MAGI_ATTENTION_TEST_OVERLAP_CONFIG` | `overlap_config` | `NAME` field |
| `MAGI_ATTENTION_TEST_NUM_HEADS` | `num_heads` | Underscore-separated, e.g. `8_8` for `(8, 8)` |
| `MAGI_ATTENTION_TEST_HEAD_DIM` | `head_dim` | String repr, e.g. `64` |
| `MAGI_ATTENTION_TEST_DTYPE` | `dtype` | String repr, e.g. `torch.float16` |
| `MAGI_ATTENTION_TEST_RANDOM_TYPE_MAPPING` | `random_type_mapping` | `True` or `False` |

For dict-typed parameters (e.g. `attn_config`, `overlap_config`), the filter matches against the `NAME` field value. For other types, the `str()` representation is used for matching.

Values are **comma-separated fnmatch pattern lists**, supporting `*`, `?` and other glob wildcards.

### Usage Examples

```bash
# Run only world_size=2
MAGI_ATTENTION_TEST_WORLD_SIZE=2 pytest tests/test_pipeline.py

# Run only world_size 2 and 4
MAGI_ATTENTION_TEST_WORLD_SIZE=2,4 pytest tests/test_pipeline.py

# Run only a specific attn_config
MAGI_ATTENTION_TEST_ATTN_CONFIG=full_attn_14k pytest tests/test_pipeline.py

# Wildcard matching for multiple attn_configs
MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_*" pytest tests/test_pipeline.py

# Run only no_overlap overlap_config
MAGI_ATTENTION_TEST_OVERLAP_CONFIG=no_overlap pytest tests/test_pipeline.py

# Run only head_dim=128
MAGI_ATTENTION_TEST_HEAD_DIM=128 pytest tests/test_pipeline.py

# Run only MHA (8,8)
MAGI_ATTENTION_TEST_NUM_HEADS=8_8 pytest tests/test_pipeline.py

# Run only float16
MAGI_ATTENTION_TEST_DTYPE="*float16*" pytest tests/test_pipeline.py

# Combined filters: full_attn configs + no_overlap + head_dim=64
MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_*" \
MAGI_ATTENTION_TEST_OVERLAP_CONFIG=no_overlap \
MAGI_ATTENTION_TEST_HEAD_DIM=64 \
    pytest tests/test_pipeline.py

# Comma-separated to match multiple values
MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_14k,uneven_full_attn_10k" \
    pytest tests/test_pipeline.py
```

## Flag Pinning (User-Preset Environment Variables)

If the user has already set a flag's environment variable before running tests, the test framework will **respect the user's setting** and lock that flag to the user-specified value. `FlagCombGenerator` will never override it.

### Usage Examples

```bash
# Force all tests to use qo_comm=1
MAGI_ATTENTION_QO_COMM=1 pytest tests/test_pipeline.py

# Force deterministic mode
MAGI_ATTENTION_DETERMINISTIC_MODE=1 pytest tests/test_pipeline.py

# Force native grpcoll + high precision reduce
MAGI_ATTENTION_NATIVE_GRPCOLL=1 \
MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE=1 \
    pytest tests/test_pipeline.py

# Force CUDA_DEVICE_MAX_CONNECTIONS=1
CUDA_DEVICE_MAX_CONNECTIONS=1 pytest tests/test_pipeline.py

# Combine: pin flags + filter test cases
MAGI_ATTENTION_QO_COMM=1 \
MAGI_ATTENTION_TEST_OVERLAP_CONFIG=disable_mso \
    pytest tests/test_pipeline.py
```

### Supported Flag Environment Variables

| Environment Variable | Flag Name | Value Type |
|---------------------|-----------|-----------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `device_max_connections` | Integer (e.g. `1` or `8`) |
| `MAGI_ATTENTION_DETERMINISTIC_MODE` | `deterministic_mode` | `0` / `1` |
| `MAGI_ATTENTION_HIERARCHICAL_COMM` | `enable_hier_comm` | `0` / `1` |
| `MAGI_ATTENTION_QO_COMM` | `enable_qo_comm` | `0` / `1` |
| `MAGI_ATTENTION_NATIVE_GRPCOLL` | `enable_native_grpcoll` | `0` / `1` |
| `MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE` | `fwd_hp_reduce` | `0` / `1` |
| `MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE` | `bwd_hp_reduce` | `0` / `1` |
| `MAGI_ATTENTION_FLATTEN_HEAD_GROUPS` | `flatten_head_groups` | `0` / `1` |
| `MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE` | `bwd_hide_tail_reduce` | `0` / `1` |

## Flag Combination Generator

Tests use `FlagCombGenerator` to automatically produce combinations of environment variable flags.

### Context-Aware Filtering

`FlagCombGenerator.get_next_valid_comb(test_config, is_valid_fn)` filters out illegal flag combinations based on the current test context. Constraint rules include:

- `no_overlap` mode disallows `qo_comm=True`
- `qo_comm=True` only allows `disable_mso` or `no_overlap` overlap configs
- `qo_comm=True` disallows `hier_comm=True` or `bwd_hide_tail_reduce=True`
- `native_grpcoll=True` disallows `hier_comm=True`
- `flatten_head_groups=True` requires `qo_comm=True`, and is incompatible with sink and `return_max_logits`
- etc.

Invalid combinations are not wasted -- they are deferred and retried in future calls with different test contexts.

## no_overlap Mode

The `no_overlap` option in `overlap_config` uses blocking communication with a merged attn_arg to perform distributed attention, completely avoiding precision loss from LSE reduce.

```python
OverlapConfig(no_overlap=True)
```
