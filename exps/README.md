# Benchmark Experiments for MagiAttention


## Attention Kernel Benchmark


### Baseline Tests for Correctness

basic command:

```bash
pytest exps/attn/tests
```

### Normal Attention Performance and Flexibility

TODO ... (add more instructions to reproduce the experiments)

basic command:

```bash
cd exps/attn

# NOTE: since transformer-engine has its own customized way to install flash-attention-3
# which has conflict with the original repo
# you need to get into the package directory and manually copy the `flash_attn_interface.py` file into `flash_attn_3/` subdirectory
bash run_benchmark.sh
```

the command to draw from existed csv files:

```bash
cd exps/dist_attn

python draw_benchmark.py
```

### Block Sparse Attention Performance and Flexibility

This benchmark supports performance profiling for uniform and variable block sparse attention.

#### Quick Start

```bash
cd exps/attn
bash run_block_sparse_benchmark.sh
```

#### Configuring Block Sizes

Edit `run_block_sparse_benchmark.py` to configure different Q/K block size combinations, used for uniform block sparse mask.

**1. Small K Block**
```python
q_block_sizes = [64, 64, 64, 64, 64]
k_block_sizes = [64, 32, 16, 8, 1]
```
Use case: Token-level sparse attention in DSA, like 64x1 block size.

**2. Small Q Block**
```python
q_block_sizes = [64, 32, 16, 8]
k_block_sizes = [64, 64, 64, 64]
```
Use case: NSA (Native Sparse Attention) in GQA scenarios, like 16x128 block size, where fewer queries attend to large K/V blocks.

**3. Large Q and K Blocks**
```python
q_block_sizes = [64, 128]
k_block_sizes = [64, 128]
```
Use case: Traditional block sparse patterns like 64x64 or 128x128 block sizes.


## Distributed Attention Module Benchmark


### Baseline Tests for Correctness

basic command:

```bash
pytest exps/dist_attn/tests
```

### Performance and Scalability

TODO ... (add more instructions to reproduce the experiments)

#### Guide to use context-parallel benchmark:

basic command:

```bash
cd exps/dist_attn

bash run_benchmark.sh --config config_file --profile output_name
```

bench with custom config file:

```bash
cd exps/dist_attn

# 1. Use the default config file (`exps/dist_attn/benchmark_conf.py`)
bash run_benchmark.sh

# 2. Use the specific config file
bash run_benchmark.sh --config config_file

# 3. Use the specific config file
bash run_benchmark.sh --config=config_file
```

bench with nsys profiler command:

```bash
cd exps/dist_attn

# 1. Enable profiling with default output name (cp_benchmark)
bash run_benchmark.sh --profile

# 2. Enable profiling and specify an output name
bash run_benchmark.sh --profile output_name

# 3. Equivalent syntax using '='
bash run_benchmark.sh --profile=output_name

# 4. Disable profiling by default
bash run_benchmark.sh
```

When benchmarking with profiling, user can set env vars `PROFILE_ITER` and `PROFILE_WARMUP` to additionally control the number of iterations and warmups.

custom bench configuration:

The default configuration file `exps/dist_attn/benchmark_conf.py` defines all necessary params for the benchmark, making it easy to adapt the setup to different environments or experiment settings, including:

- SEED
- BENCH_CONFIG (how to bench):
    - bench metrics config (see `magi_attention/benchmarking/bench.py` for details):
        - quantiles: quantile points to report results.
        - bench_flops / bench_mem: Whether to evaluate FLOPs or memory.
        - bench_mode: statistic mode (mean, median, min, max).
        - iteration / warmup: number of iterations and warmups for each run.
        - output_path: directory to save bench results.
    - dist_attn_impl: all distributed attn to evaluate, as x-vals.
    - bench sweep config:
        - mask_pattern: all mask patterns to evaluate。 Options: [full, causal, varlen-full, varlen-causal]
        - workload: all pipeline modes to evaluate. Options: [fwd, bwd, 1f1b]
- SAMPLE_CONFIG:
    - defines how to sample datasets to simulate real training scenarios, see `benchmark_conf.py` for details.
- DATA_CONFIG:
    - defines how to generate data to run the bench, see `benchmark_conf.py` for details.
- ATTN_CONFIG:
    - defines how to configure the attention mechanisms, see `benchmark_conf.py` for details.

#### Guide to run context-parallel profile:

basic command:

```bash
cd exps/dist_attn

bash run_profile.sh
```


## Communication Kernel Benchmark

### Native Group Collective Integration Tests and Performance Tuning

basic command:

```bash
cd exps/grpcoll

# by default, this test only runs the intra-node mode
# and if you want to test other modes,
# you can create a `.env` file
# and add `TEST_MODE=xxx` into it, where xxx is one of `intra_node`, `inter_node`, `low_latency`

# As for testing inter-node mode,
# you also need to add your own master node IP `MASTER_ADDR=xxx` into the `.env` file
# and pass the node rank as the first argument of the following command, e.g. `bash run_grpcoll_test.sh $NODE`

bash run_grpcoll_test.sh
```


### Device All-to-All-V PoC Test

TODO ... (add more instructions to reproduce the experiments)

basic command:

```bash
cd exps/device_a2av

bash run.sh
```
