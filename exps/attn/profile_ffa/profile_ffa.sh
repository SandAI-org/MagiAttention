#!/bin/bash

# Copyright (c) 2025 SandAI. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Script Description ---
# Purpose: Runs benchmarks for a base and target branch, compares the results,
#          and stores all outputs directly in a new timestamped directory.
# Arguments:
#   $1: The name of the base branch (e.g., 'main').
#   $2: The name of the target branch (e.g., 'develop' or a feature branch).
# Usage: ./run_comparison.sh <base_branch> <target_branch>

# 1. Check arguments and create the output directory
if [ "$#" -ne 2 ]; then
    echo "Error: This script requires exactly two arguments."
    echo "Usage: $0 <base_branch> <target_branch>"
    exit 1
fi

BASE_BRANCH=$1
TARGET_BRANCH=$2
OUTPUT_DIR="benchmark_results_$(date +'%Y%m%d-%H%M%S')"

echo "================================================="
echo "Starting benchmark comparison for:"
echo "  Base Branch:   $BASE_BRANCH"
echo "  Target Branch: $TARGET_BRANCH"
echo "---"
echo "All results will be saved in: $OUTPUT_DIR"
echo "================================================="

mkdir -p "$OUTPUT_DIR"

# 3. Run the benchmark for the BASE branch, passing the output directory
echo ""
echo "--- Running benchmarks for BASE branch: $BASE_BRANCH ---"
git checkout $BASE_BRANCH
bash run_branch_profile.sh "$BASE_BRANCH" "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Benchmark run failed for the base branch ($BASE_BRANCH)."
    exit 1
fi

# 4. Run the benchmark for the TARGET branch, passing the output directory
echo ""
echo "--- Running benchmarks for TARGET branch: $TARGET_BRANCH ---"
bash run_branch_profile.sh "$TARGET_BRANCH" "$OUTPUT_DIR"
git checkout $TARGET_BRANCH
if [ $? -ne 0 ]; then
    echo "Error: Benchmark run failed for the target branch ($TARGET_BRANCH)."
    exit 1
fi

# 5. Compare the generated results.
echo ""
echo "--- Comparing benchmark results ---"

# Define full paths for the comparison script's input files
BASE_DENSE_CSV="$OUTPUT_DIR/profile_dense_${BASE_BRANCH}.csv"
TARGET_DENSE_CSV="$OUTPUT_DIR/profile_dense_${TARGET_BRANCH}.csv"
COMPARE_DENSE_CSV="$OUTPUT_DIR/compare_${BASE_BRANCH}_${TARGET_BRANCH}_dense.csv"

if [ -f "$BASE_DENSE_CSV" ] && [ -f "$TARGET_DENSE_CSV" ]; then
    echo "Comparing dense results into: $COMPARE_DENSE_CSV"
    python compare_ffa_results.py "$BASE_DENSE_CSV" "$TARGET_DENSE_CSV" "$COMPARE_DENSE_CSV"
else
    echo "Warning: One or both dense CSV files not found. Skipping dense comparison."
fi

# Define full paths for the comparison script's input files
BASE_SPARSE_CSV="$OUTPUT_DIR/profile_block_sparse_${BASE_BRANCH}.csv"
TARGET_SPARSE_CSV="$OUTPUT_DIR/profile_block_sparse_${TARGET_BRANCH}.csv"
COMPARE_SPARSE_CSV="$OUTPUT_DIR/compare_${BASE_BRANCH}_${TARGET_BRANCH}_block_sparse.csv"

if [ -f "$BASE_SPARSE_CSV" ] && [ -f "$TARGET_SPARSE_CSV" ]; then
    echo "Comparing block_sparse results into: $COMPARE_SPARSE_CSV"
    python compare_ffa_results.py "$BASE_SPARSE_CSV" "$TARGET_SPARSE_CSV" "$COMPARE_SPARSE_CSV"
else
    echo "Warning: One or both block_sparse CSV files not found. Skipping block_sparse comparison."
fi

# REMOVED the cleanup step as it's no longer needed.

echo ""
echo "================================================="
echo "Comparison run finished successfully!"
echo "All results are located in the directory: $OUTPUT_DIR"
echo "================================================="
