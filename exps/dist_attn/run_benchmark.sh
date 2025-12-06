#! /bin/bash

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

export NNODES=${NNODES:-1}
export GPUS_PER_NODE=1
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
export NODE_RANK=${RANK:-0}
# export MAGI_ATTENTION_HIERARCHICAL_COMM=${MAGI_ATTENTION_HIERARCHICAL_COMM:-0}

if [[ $NNODES -eq 1 ]]; then # single-node
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-16985}
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

export PYTHONPATH=../../

if [ "${MAGI_ATTENTION_HIERARCHICAL_COMM}" == "1" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=8
    echo "set CUDA_DEVICE_MAX_CONNECTIONS=8"
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    echo "set CUDA_DEVICE_MAX_CONNECTIONS=1"
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo $DISTRIBUTED_ARGS

TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS run_benchmark.py"

CMD="nsys profile \
    --force-overwrite true \
    -o magi_dynamic_valen_full_16k_ws8.nsys-rep \
    --capture-range=cudaProfilerApi \
    $TORCHRUN_CMD"

# TORCHRUN_CMD="nsys profile \
#     --force-overwrite true \
#     -o ring_allgather.nsys-rep \
#     --capture-range=cudaProfilerApi \
#     $CMD"

$TORCHRUN_CMD
# $CMD
