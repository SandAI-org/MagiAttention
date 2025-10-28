# Benchmark Experiments for MagiAttention


## Attention Kernel Benchmark


### Baseline Tests for Correctness

basic running command:

```bash
pytest exps/attn/tests
```

### Normal Attention Performance and Flexibility

TODO ... (add more instructions to reproduce the experiments)

basic running command:

```bash
cd exps/attn

bash run_benchmark.sh
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

basic running command:

```bash
pytest exps/dist_attn/tests
```

### Performance and Scalability

TODO ... (add more instructions to reproduce the experiments)

basic running command:

```bash
cd exps/dist_attn

bash run_benchmark.sh
```


## Communication Kernel Benchmark

### Native Group Collective Integration Tests and Performance Tuning

TODO ... (add more instructions to reproduce the experiments)

basic running command:

```bash
cd exps/grpcoll

bash run_grpcoll_test.sh
```
