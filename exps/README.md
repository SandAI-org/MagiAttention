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

basic running command:

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

basic running command:

```bash
cd exps/device_a2av

bash run.sh
```
