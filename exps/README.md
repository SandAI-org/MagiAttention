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

```bash
cd exps/attn

bash run_block_sparse_benchmark.sh
```

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
