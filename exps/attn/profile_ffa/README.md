## How to profile ffa

### Test settings
For now, we test for dense and block sparse scenerias.

model_configs:
- nhq: 64
- nhk: 8
- headdim: 128
- dtype: torch.bfloat16

You can change the model-related settings in `common_params` within `ffa_benchmark.py`.

Dense:
- seqlens_to_test = [8192]
- mask_types_to_test = ["full", "causal", "varlen_full", "varlen_causal"]

You can change the dense-related settings in `run_dense_tests` within `ffa_benchmark.py`.

Block sparse:
- seqlens_to_test = [49152]
- sparsity_ratios_to_test = [0.1, 0.2, 0.5]
- block_sizes_to_test = [64, 128]

You can change the block_sparse-related settings in `run_block_sparse_tests` within `ffa_benchmark.py`.


### Basic file usage
- `ffa_benchmark.py`: run ffa for dense/block sparse mask.
    ```shell
    PYTHONPATH=../../../ python ffa_benchmark.py --test_type dense/block_sparse --o output.csv
    ```

- `compare_ffa_results.py`: compare two output csv with same mask type.
    ```shell
    python compare_ffa_results.py base_output.csv target_output.csv compare_result.csv
    ```

### Shells
- run_branch_profile.sh: profile dense and block_sparse mask for current branch. Generate profile_dense_branch_name.csv and profile_block_sparse_branch_name.csv in output_dir.
```shell
bash run_branch_profile.sh current_branch_name output_dir
```

- profile_ffa.sh: profile dense and block_sparse mask for base and target branch, generate base_target_dense/block_sparse.csv in optimize_ffa/benchmark_results_time dir.
```shell
bash profile_ffa.sh base_branch_name target_branch_name
```

### Example
**Banch**
- base branch: main
- target branch: optimize_ffa

**run shell**
- bash profile_ffa.sh main profile_ffa >& output.txt

**Results**

In dir `optimize_ffa/benchmark_results_time`
- `profile_dense_main.csv`:   dense mask result for main branch.
- `profile_dense_optimize_ffa.csv`:   dense mask result for optimize_ffa branch.
- `profile_block_sparse_main.csv`: block sparse mask result for main branch.
- `profile_block_sparse_optimize_ffa.csv`: block sparse mask results for optimize_ffa branch.
- `compare_main_optimize_ffa_dense.csv`: compare results for dense mask.
- `compare_main_optimize_ffa_block_sparse.csv`: compare results for block sparse mask.
- `output.txt`: Containing Intermediate outputs. At the end of output.txt, warnings will be issued for any cases with a TFLOPs variation greater than 1.5%.



Testing Config: {'seqlen': 8192, 'mask_type': 'full'}
**FORWARD PERFORMANCE**
- Total Runtime (ms): 3.5861
- Achieved TFLOP/s: 613.2028

**Internal Timing Breakdown**
| Operation   | Time (ms) | Description             |
|-------------|-----------|-------------------------|
| range_merge | -1.0000   | RangeMerge              |
| Prepare     | 0.0153    | prepare_mha_forward     |
| Run         | 3.4125    | run_mha_forward         |
| Fill        | 0.0119    | Fast_zero_fill          |
| to          | 0.0032    | cast output to qdtype   |


**BACKWARD PERFORMANCE**
- Total Runtime (ms)   | 10.0102
- Achieved TFLOP/s     | 549.1948

**Internal Timing Breakdown**
| Operation  | Time (ms) | Description             |
|------------|-----------|-------------------------|
| range_merge| -1.0000   | RangeMerge              |
| Prepare    | 0.1265    | prepare_mha_backward    |
| Preprocess | 0.1050    | bwd_preprocess          |
| Run        | 9.3409    | run_mha_backward        |
| to         | 0.1765    | cast dq, dk, dv         |
