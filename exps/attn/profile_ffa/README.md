## How to profile ffa

### Settings

For now, we test only for dense and block sparse scenerias.

#### Model Config

- nhq: [64]
- nhk: [64]   # change nhk to test differnt packgqa settings, for ffa backward of block sparse, gqa performance is bad now.
- headdim: [128]
- dtype: [torch.bfloat16]

You can change the model-related settings in `common_params` within `ffa_benchmark.py`.

#### Dense Config

- seqlens_to_test = [8192]
- mask_types_to_test = ["full", "causal", "varlen_full", "varlen_causal"]

You can change the dense-related settings in `run_dense_tests` within `ffa_benchmark.py`.

#### Block sparse Config

- seqlens_to_test = [49152]
- sparsity_ratios_to_test = [0.05, 0.1, 0.2, 0.5, 1.0]
- q_block_sizes = [64, 128]
- k_block_sizes = [64, 128]
- pack_gqa_options = [False]
- swap_ab_options = [False]

You can change the block_sparse-related settings in `run_block_sparse_tests` within `ffa_benchmark.py`.


### Basic Usage

#### Python Scripts

- `ffa_benchmark.py`: run ffa for dense/block sparse mask.
    ```shell
    export MAGI_ATTENTION_PROFILE_MODE=1
    # NOTE: enabling profile mode will enforce ffa to build in JIT mode
    # thus here we toggle this on by default to show the verbose building process
    # instead of waiting w/o any output
    export MAGI_ATTENTION_BUILD_VERBOSE=1

    OUT_DIR="outs"
    TEST_TYPE="dense" # choose from {"dense", "block_sparse"}
    OUTPUT_NAME="output"

    # you can add --fwd or --bwd to run fwd or bwd only.
    # by default we run both fwd and bwd.
    PYTHONPATH=../../../ python ffa_benchmark.py --test_type ${TEST_TYPE} --o ${OUT_DIR}/${OUTPUT_NAME}.csv

    # you can enable ncu profile for fwd/bwd pass.
    # PYTHONPATH=../../../ ncu -f --set full --nvtx --nvtx-include backward_pass -o ncu_output_name \
    # python ffa_benchmark.py --test_type ${TEST_TYPE} --bwd --o ${OUT_DIR}/${OUTPUT_NAME}.csv
    ```

- `compare_ffa_results.py`: compare two output csv with same mask type.
    ```shell
    export MAGI_ATTENTION_PROFILE_MODE=1
    # NOTE: enabling profile mode will enforce ffa to build in JIT mode
    # thus here we toggle this on by default to show the verbose building process
    # instead of waiting w/o any output
    export MAGI_ATTENTION_BUILD_VERBOSE=1

    OUT_DIR="outs"
    BASE_OUTPUT_NAME="base_output"
    TARGET_OUTPUT_NAME="target_output"
    COMPARE_RESULT_NAME="compare_result"

    python compare_ffa_results.py ${OUT_DIR}/${BASE_OUTPUT_NAME}.csv ${OUT_DIR}/${TARGET_OUTPUT_NAME}.csv ${OUT_DIR}/${COMPARE_RESULT_NAME}.csv
    ```

#### Shell Scripts

- `run_branch_profile.sh`: profile dense and block_sparse mask for current branch. Generate `profile_dense_branch_name.csv` and `profile_block_sparse_branch_name.csv` in `output_dir`.
    ```shell

    CURRENT_BRANCH_NAME="main"
    OUTPUT_DIR="outs"

    bash run_branch_profile.sh ${CURRENT_BRANCH_NAME} ${OUTPUT_DIR}
    ```

- `profile_ffa.sh`: profile dense and block_sparse mask for base and target branch, generate `base_target_dense/block_sparse.csv` in `optimize_ffa/benchmark_results_time` dir.
    ```shell

    BASE_BRANCH_NAME="main"
    TARGET_BRANCH_NAME="optimize_ffa"

    bash profile_ffa.sh ${BASE_BRANCH_NAME} ${TARGET_BRANCH_NAME}
    ```

### Example

#### Steps

**Pick Branches**

- base branch: `main`
- target branch: `optimize_ffa`

**Run Script**

```shell
bash profile_ffa.sh main profile_ffa >& output.txt
```

**View Output**

In dir `optimize_ffa/benchmark_results_time`
- `profile_dense_main.csv`:   dense mask result for main branch.
- `profile_dense_optimize_ffa.csv`:   dense mask result for optimize_ffa branch.
- `profile_block_sparse_main.csv`: block sparse mask result for main branch.
- `profile_block_sparse_optimize_ffa.csv`: block sparse mask results for optimize_ffa branch.
- `compare_main_optimize_ffa_dense.csv`: compare results for dense mask.
- `compare_main_optimize_ffa_block_sparse.csv`: compare results for block sparse mask.
- `output.txt`: Containing Intermediate outputs. At the end of output.txt, warnings will be issued for any cases with a TFLOPs variation greater than 1.5%.


#### Results

> [!NOTE]
> Testing Config: {'seqlen': 8192, 'mask_type': 'full'}

**Forward Performance**

- Total Runtime (ms): `3.5861`
- Achieved TFLOP/s: `613.2028`

**Internal Timing Breakdown**

| Operation   | Time (ms) | Description             |
|-------------|-----------|-------------------------|
| range_merge | -1.0000   | RangeMerge              |
| Prepare     | 0.0153    | prepare_ffa_forward     |
| Run         | 3.4125    | run_ffa_forward         |
| Postprocess | 0.0119    | fwd_postprocess         |
| to          | 0.0032    | cast output to qdtype   |


**Backward Performance**

- Total Runtime (ms): `10.0102`
- Achieved TFLOP/s: `549.1948`

**Internal Timing Breakdown**

| Operation  | Time (ms) | Description             |
|------------|-----------|-------------------------|
| range_merge| -1.0000   | RangeMerge              |
| Prepare    | 0.1265    | prepare_ffa_backward    |
| Preprocess | 0.1050    | bwd_preprocess          |
| Run        | 9.3409    | run_ffa_backward        |
| to         | 0.1765    | cast dq, dk, dv         |


NOTE: For more detailed and accurate performance Info, please use ncu.
