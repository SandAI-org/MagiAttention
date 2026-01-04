#!/bin/bash

# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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

TEST_ROOT=.
LOG_ROOT=${TEST_ROOT}/outs
SRC_NAME="test"

mkdir -p ${LOG_ROOT}

export PYTHONPATH=$PYTHONPATH:.

export NNODES=1
# export NNODES=2 # ngc-2505 version of torch does not support inter-node comm with symm-mem

if [[ $NNODES -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export MASTER_ADDR="localhost"
    export RANK=0
    LOG_NAME=${SRC_NAME}
else
    export CUDA_VISIBLE_DEVICES="0,1"
    export MASTER_ADDR=10.119.210.141 # replace with your own master node IP

    if [ -z "$1" ]; then
        echo "Error: Please specify the rank of this node."
        echo "Usage: ./run_distributed.sh <rank>"
        echo "Example: ./run_distributed.sh 0  (for master node 0)"
        exit 1
    else
        echo "Launch with node rank: $1"
    fi
    export RANK=$1
    LOG_NAME=${SRC_NAME}_n${RANK}
fi

LOG_PATH=${LOG_ROOT}/${LOG_NAME}.log

export OMP_NUM_THREADS=1
export NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export MASTER_PORT=23457

export TEST_PROFILE_MODE=0
export TEST_USE_NCU_FOR_PROFILE=0
export TEST_SANITIZER_MODE=0

# export TEST_CASE="naive_a2a"
export TEST_CASE="naive_a2a_v"


CMD="torchrun \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ${SRC_NAME}.py
"

echo "Logging to ${LOG_PATH} ..."

if [[ $TEST_PROFILE_MODE == "1" ]]; then
    if [[ $TEST_USE_NCU_FOR_PROFILE == "1" ]]; then
        ncu \
            --target-processes all \
            --set full \
            --kernel-name device_kernel \
            -f -o ${SRC_NAME}.ncu-rep \
            $CMD > ${LOG_PATH} 2>&1
    else
        nsys profile \
            --force-overwrite true \
            -o ${SRC_NAME}.nsys-rep \
            --capture-range=cudaProfilerApi \
            $CMD > ${LOG_PATH} 2>&1
    fi
elif [[ $TEST_SANITIZER_MODE == "1" ]]; then
    compute-sanitizer --tool memcheck $CMD > ${LOG_PATH} 2>&1
else
    $CMD > ${LOG_PATH} 2>&1
fi
