# Benchmark Experiments for MagiAttention


## Kernel-Level Attention Performance and Flexibility

TODO ... (add more instructions to reproduce the experiments)

basic running command:

```bash
cd exps/attn

bash run_benchmark.sh
```


## Module-Level Distributed Attention Performance and Scalability

TODO ... (add more instructions to reproduce the experiments)

basic running command:

```bash
cd exps/dist_attn

export PYTHONPATH="${PYTHONPATH}:/path/to/MagiAttention/"

bash run_benchmark.sh
```

## Baseline Tests for Correctness

basic running command:

```bash
cd exps/dist_attn/tests

export PYTHONPATH="${PYTHONPATH}:/path/to/MagiAttention/"

pytest test_baseline_attn.py
```
