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

export WORLD_SIZE=${WORLD_SIZE:-8}
export GPUS_PER_NODE=${WORLD_SIZE}
export NNODES=1
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-16988}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo $DISTRIBUTED_ARGS

# TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS run_benchmark.py"
CMD="torchrun $DISTRIBUTED_ARGS run_benchmark.py"
TORCHRUN_CMD="nsys profile \
    --force-overwrite true \
    -o usp_4_2.nsys-rep \
    --capture-range=cudaProfilerApi \
    $CMD"
$TORCHRUN_CMD
