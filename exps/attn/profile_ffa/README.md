## How to profile ffa
### Basic files
- `ffa_benchmark.py`: run ffa for dense/block sparse mask.
    ```shell
    PYTHONPATH=../../../ python --test_type dense/block_sparse --o output.csv ffa_benchmark.py
    ```

- `compare_ffa_results.py`: compare two output csv with same mask type.
    ```shell
    python base_output.csv target_output.csv compare_result.csv
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
- `main_optimize_ffa_dense.csv`: compare results for dense mask.
- `main_optimize_ffa_block_sparse.csv`: compare results for block sparse mask.
- `output.txt`: Containing Intermediate outputs. At the end of output.txt, warnings will be issued for any cases with a TFLOPs variation greater than 1.5%.
