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
# Purpose: Runs benchmark tests for both 'dense' and 'block_sparse' types.
# Arguments:
#   $1: The required <branch_name> to generate unique output filenames.
#   $2: The required <output_dir> where the files will be saved.
# Usage: ./run_benchmark.sh your_branch_name path/to/output_dir

export MAGI_ATTENTION_PROFILE_MODE=1
# NOTE: enabling profile mode will enforce ffa to build in JIT mode
# thus here we toggle this on by default to show the verbose building process
# instead of waiting w/o any output
export MAGI_ATTENTION_BUILD_VERBOSE=1

# 1. Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Error: This script requires exactly two arguments."
    echo "Usage: $0 <branch_name> <output_dir>"
    exit 1
fi

# 2. Assign the arguments to more readable variables
BRANCH_NAME=$1
OUTPUT_DIR=$2

echo "================================================="
echo "Received Branch Name: $BRANCH_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "================================================="

# --- ADDED: Ensure the output directory exists before trying to write to it ---
echo "Ensuring output directory '$OUTPUT_DIR' exists..."
mkdir -p "$OUTPUT_DIR"
# ---------------------------------------------------------------------------

# 3. Run the dense type test
#    The output path is now constructed with the directory prefix.
OUTPUT_DENSE="${OUTPUT_DIR}/profile_dense_${BRANCH_NAME}.csv"
echo "" # Print a blank line for spacing
echo "Running dense test, outputting to file: $OUTPUT_DENSE ..."
PYTHONPATH=../../../ python ffa_benchmark.py --test_type dense -o "$OUTPUT_DENSE"

# 4. Run the block_sparse type test
#    The output path is now constructed with the directory prefix.
OUTPUT_BLOCK_SPARSE="${OUTPUT_DIR}/profile_block_sparse_${BRANCH_NAME}.csv"
echo "" # Print a blank line for spacing
echo "Running block_sparse test, outputting to file: $OUTPUT_BLOCK_SPARSE ..."
PYTHONPATH=../../../ python ffa_benchmark.py --test_type block_sparse -o "$OUTPUT_BLOCK_SPARSE"

echo ""
echo "All profile have completed!"
