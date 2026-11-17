# Testing

## Test organization

Tests follow the layer they exercise. Keep distributed integration separate from
local contracts, even when both exercise the same public function.

| Directory | Responsibility |
| --- | --- |
| `tests/test_common` | Core values, protocols, and host range metadata |
| `tests/test_meta` | Containers, collections, and local solver contracts |
| `tests/test_utils` | Production utilities and environment option parsing |
| `tests/test_testing` | Test helpers and reference implementations |
| `tests/test_functional` | Attention wrappers, autograd, and local runtime contracts |
| `tests/test_kernel/cuda` | Native CUDA kernel contracts |
| `tests/test_kernel/cutedsl` | CuTeDSL kernels and architecture-specific behavior |
| `tests/test_kernel/triton` | Triton kernels, including range operations |
| `tests/test_distributed` | Process groups, collectives, and distributed API/runtime integration |
| `tests/test_api` | Local public API and mask construction contracts |
| `tests/test_pipeline.py` | End-to-end attention pipeline |

A worker-based kernel sweep can still belong in `test_kernel`: spawning workers
alone does not make the behavior under test a communication integration test.
Inspect its test base and required world size before running it.

## Running a focused test

Run commands from the repository root with the dependencies from the
[installation guide](../user_guide/install.md). Select enough GPUs for the test's
world size. The following commands use one selected GPU for local tests:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY python -m pytest -q tests/test_common
CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY python -m pytest -q tests/test_meta
CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY python -m pytest -q tests/test_functional/test_dist_attn_runtime.py
CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=LAZY python -m pytest -q tests/test_kernel/triton/test_range_ops
```

Use `python -m pytest --collect-only -q tests` to inspect collection without
claiming execution. `magi_attention.testing.parameterize` runs combinations
inside each test method, so the pytest item count is not the parameter count.
Distributed tests use the library's process launcher; do not also launch the
same test with `torchrun` unless its entry point explicitly requires it.

## Writing tests

Use `unittest.TestCase` and `unittest.main()` for ordinary CPU or single-GPU
contracts. Use the existing distributed test base and its `run_tests()` entry
point when process-group or worker lifecycle is required. Prefer the library's
`parameterize`, environment-switching helpers, and tensor assertions; keep
exact equality distinct from numerical tolerance checks.

Use `magi_attention.utils.arch` for architecture conditions and the version
helpers for dependency gates. Do not skip GPU tests merely because CUDA is
unavailable: a required runtime failure should be visible. Keep architecture,
version, rank-count, and explicit profiling restrictions meaningful.

Helpers shared by independent tests belong in `magi_attention.testing`; small
case-specific helpers stay in their test module. Do not import a test module
from another test module. Keep profiling workloads explicitly opt-in.

## Python and C++ common backends

Common container tests select implementation types inside each file. The
`TestAttnRange`, `TestAttnRanges`, rectangle, and protocol classes exercise Python;
their `TestCpp*` subclasses repeat the contracts with native types. These tests
do not change the package's backend or reload modules. `AttnMask` is Python-only.
An unavailable native extension skips its direct implementation tests.

```bash
python -m pytest -q tests/test_common/test_attn_range.py::TestAttnRange
python -m pytest -q tests/test_common/test_attn_range.py::TestCppAttnRange
```

The self-attention metadata and ground-truth dispatcher suites also check
import-time routing. Each file launches its own Python and C++ suites with
`multiprocessing` spawn, setting `MAGI_ATTENTION_CPP_BACKEND` before the child
imports the package. It restores the parent's environment immediately after
startup and checks child failures and timeouts. There is no recursive pytest
wrapper or separate CI coverage mode. Keep backend setup local to these tests.

## Native kernel precompilation

Test classes that exercise native FFA variants declare
`precompile_kernel_specs()`, ideally using the same constants as their parameter
matrix. Register their module in `magi_attention.testing.precompile` when adding
or moving a suite. Inspect the requested kernels without compiling them:

```bash
python -m magi_attention.testing.precompile --list
```

This registry feeds the CI prebuild as well as manual precompilation. Keep it
synchronized with the real test modules; do not include removed extension APIs.

## Kernel Backend

`test_pipeline.py` parameterizes over **all five kernel backends** (`ffa`, `sdpa`, `sdpa_ol`, `fa4`, `cutedsl`). Each `attn_config` is tagged with the set of backends it applies to, so incompatible (backend, config) pairs are automatically skipped.

- `ffa` / `fa4` / `cutedsl` -- run large-seqlen configs with fp16/bf16, using norm-based + mismatch-based tolerance checks.
- `sdpa` / `sdpa_ol` -- run small-seqlen configs with fp64, using exact (EPSILON) tolerance checks.

The backend is controlled by `MAGI_ATTENTION_KERNEL_BACKEND` (see below).

## Test Case Filtering

You can use environment variables to precisely control which parameterized test cases are executed. All filter conditions use **AND** logic -- only test cases that satisfy all active filters will run.

### Environment Variables

| Environment Variable | Parametrize Dimension | Match Target |
|---------------------|----------------------|-------------|
| `MAGI_ATTENTION_TEST_WORLD_SIZE` | `world_size` | Comma-separated integers, e.g. `2` or `2,4` |
| `MAGI_ATTENTION_TEST_ATTN_CONFIG` | `attn_config` | `NAME` field |
| `MAGI_ATTENTION_TEST_NUM_HEADS` | `num_heads` | Underscore-separated, e.g. `8_8` for `(8, 8)` |
| `MAGI_ATTENTION_TEST_HEAD_DIM` | `head_dim` | String repr, e.g. `64` |
| `MAGI_ATTENTION_TEST_DTYPE` | `dtype` | String repr, e.g. `torch.float16` |
| `MAGI_ATTENTION_TEST_BACKEND` | `backend` | Canonical `.value`, e.g. `ffa` / `sdpa` / `sdpa_ol` / `fa4` / `cutedsl` |

For dict-typed parameters (e.g. `attn_config`), the filter matches against the `NAME` field value. For `Enum` parameters (e.g. `backend`), it matches the canonical `.value` (`ffa`), the member `.name` (`FFA`), and `str()` (`MagiAttentionKernelBackend.FFA`). For other types, the `str()` representation is used.

> **Note:** `overlap_config` and `random_type_mapping` have been moved from `@parameterize` dimensions into `FlagCombGenerator` flags and are selected automatically by the heuristic strategy. The env vars `MAGI_ATTENTION_TEST_OVERLAP_CONFIG` and `MAGI_ATTENTION_TEST_RANDOM_TYPE_MAPPING` no longer take effect.

> **Note:** `num_heads`, `head_dims` and `dtype` are one `@parameterize` dimension (`shape_cfg`, 16 triples). The list is split by world-size parity: odd world sizes run 8 triples and even world sizes run the other 8, so every triple runs on four world sizes. The `MAGI_ATTENTION_TEST_NUM_HEADS` / `MAGI_ATTENTION_TEST_HEAD_DIM` / `MAGI_ATTENTION_TEST_DTYPE` filters match the unpacked values and are unaffected.

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

# Run only head_dim=128
MAGI_ATTENTION_TEST_HEAD_DIM=128 pytest tests/test_pipeline.py

# Run only MHA (8,8)
MAGI_ATTENTION_TEST_NUM_HEADS=8_8 pytest tests/test_pipeline.py

# Run only float16
MAGI_ATTENTION_TEST_DTYPE="*float16*" pytest tests/test_pipeline.py

# Run only sdpa_ol backend tests
MAGI_ATTENTION_TEST_BACKEND=sdpa_ol pytest tests/test_pipeline.py

# Combined filters: full_attn configs + head_dim=64
MAGI_ATTENTION_TEST_ATTN_CONFIG="full_attn_*" \
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
MAGI_ATTENTION_TEST_ATTN_CONFIG=full_attn_14k \
    pytest tests/test_pipeline.py
```

### Supported Flag Environment Variables

| Environment Variable | Flag Name | Value Type |
|---------------------|-----------|-----------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | `device_max_connections` | Integer (e.g. `1` or `8`) |
| `MAGI_ATTENTION_KERNEL_BACKEND` | kernel backend | `ffa` / `sdpa` / `sdpa_ol` / `fa4` / `cutedsl` |
| `MAGI_ATTENTION_DETERMINISTIC_MODE` | `deterministic_mode` | `0` / `1` |
| `MAGI_ATTENTION_HIERARCHICAL_COMM` | `enable_hier_comm` | `0` / `1` |
| `MAGI_ATTENTION_QO_COMM` | `enable_qo_comm` | `0` / `1` |
| `MAGI_ATTENTION_NATIVE_GRPCOLL` | `enable_native_grpcoll` | `0` / `1` |
| `MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE` | `fwd_hp_reduce` | `0` / `1` |
| `MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE` | `bwd_hp_reduce` | `0` / `1` |
| `MAGI_ATTENTION_FLATTEN_HEAD_GROUPS` | `flatten_head_groups` | `0` / `1` |
| `MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE` | `bwd_hide_tail_reduce` | `0` / `1` |

> **Note:** The legacy env vars `MAGI_ATTENTION_SDPA_BACKEND` and `MAGI_ATTENTION_FA4_BACKEND` are still supported for backward compatibility, but must **not** be set together with `MAGI_ATTENTION_KERNEL_BACKEND`.

## Flag Combination Generator

Tests use `FlagCombGenerator` to automatically produce combinations of flags, including environment variable flags (e.g. `deterministic_mode`, `enable_qo_comm`) and test parameter flags (`overlap_config`, `random_type_mapping`).

These flags are combined using a `heuristic` strategy instead of full Cartesian product, significantly reducing the total number of test combinations.

### Context-Aware Filtering

`FlagCombGenerator.get_next_valid_comb(test_config, is_valid_fn)` filters out illegal flag combinations based on the current test context. Constraint rules include:

- `no_overlap` mode disallows `qo_comm=True`
- `qo_comm=True` only allows `disable_mso` or `no_overlap` overlap configs
- `qo_comm=True` disallows `hier_comm=True` or `bwd_hide_tail_reduce=True`
- `native_grpcoll=True` disallows `hier_comm=True`; with `head_dim != head_dim_v` it also requires `num_heads_kv * (head_dim + head_dim_v)` to be a multiple of the hidden-size alignment (256 for fp16/bf16)
- `flatten_head_groups=True` requires `qo_comm=True`, and is incompatible with sink and `return_max_logits`
- `fa4` backend disallows `deterministic`, `fwd_hp_reduce`, `bwd_hp_reduce`, `qo_comm`, `sink`, `bwd_hide_tail_reduce`
- `sdpa` / `sdpa_ol` backends disallow `native_grpcoll`
- etc.

Invalid combinations are not wasted -- they are deferred and retried in future calls with different test contexts. If all possible combinations are invalid, a `RuntimeError` is raised immediately instead of hanging indefinitely.

## OverlapConfig degree Semantics

`OverlapConfig` uses the `degree` parameter to control overlap behavior:

| degree | Meaning |
|--------|---------|
| `0` | **no overlap** -- blocking communication + merged attn_arg, no LSE reduce precision loss |
| `1` | local + 1 remote stage, no chunking (no multi-stage overlap) |
| `N (N>=2)` | local + N remote stages (static multi-stage overlap) |
| `None` | Dynamic mode (solver determines optimal degree automatically) |

```python
# no overlap (highest precision)
OverlapConfig(degree=0)

# no multi-stage overlap
OverlapConfig(degree=1)

# 4-stage static overlap
OverlapConfig(degree=4, mode=AttnOverlapMode.STATIC)

# dynamic overlap
OverlapConfig(degree=None, mode=AttnOverlapMode.DYNAMIC)
```

## Architecture-independent coverage

Use `@pytest.mark.arch_independent` on reviewed CPU/Python contracts whose
coverage does not depend on GPU architecture. Class markers apply to every
method, so reconsider the marker before adding GPU cases to a marked class.
Do not mark CUDA numerics, gradients, precision/RNG, compiler, or collective
coverage merely because it is slow or shared by both CI platforms.

The root collection hook skips these items only when `TASK_CI_PLATFORM` is
`b300` or `b300_fa4` and the actual device is Blackwell. Hopper CI owns the
shared coverage. Without that CI profile, local Blackwell runs remain complete;
`--skip-slow`, architecture restrictions, and optional-dependency skips retain
their existing behavior. The marker changes execution, not collection.

The portable runner discovers the root hook when testing absolute paths from
its temporary working directory with `--import-mode=append`; it continues to
import the installed wheel. The tracked root `conftest.py` and tests already
participate in the main package's trigger/portable input digests, so changing
this policy invalidates previous validation without a cache schema change.

## Memory ledger reconciliation

The analytical memory-budget tests run without launching a large attention case.
For an explicit native FFA memory check on Hopper, use
`python -m exps.attn.ffa_mem_budget --help` to inspect the opt-in experiment.
