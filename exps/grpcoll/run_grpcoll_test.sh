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

if [[ -f .env ]]; then
    source .env # maybe put your own master node IP here
fi

TEST_ROOT=.
LOG_ROOT=${TEST_ROOT}/outs
TEST_MODE=${TEST_MODE:-"intra_node"} # intra_node | low_latency | internode

mkdir -p ${LOG_ROOT}

export PYTHONPATH=$PYTHONPATH:.

# grpcoll test will set the env vars in the script
# export NVSHMEM_IB_ENABLE_IBGDA=1
# export NVSHMEM_IBGDA_NIC_HANDLER=gpu
# export NVSHMEM_DISABLE_P2P=0 # set to 0 to enable NVLink in low-latency mode
# export NVSHMEM_SYMMETRIC_SIZE=2**30 # default: 1GB


# ----- test-intranode ----- #

if [[ $TEST_MODE == "intra_node" ]]; then
    LOG_PATH=${LOG_ROOT}/test_intranode_grpcoll.log
    echo "Logging to ${LOG_PATH} ..."
    python ${TEST_ROOT}/test_intranode_grpcoll.py > ${LOG_PATH} 2>&1; exit 0
fi

# ----- test-low-latency ----- #

if [[ $TEST_MODE == "low_latency" ]]; then
    LOG_PATH=${LOG_ROOT}/test_low_latency_grpcoll.log
    echo "Logging to ${LOG_PATH} ..."
    python ${TEST_ROOT}/test_low_latency_grpcoll.py > ${LOG_PATH} 2>&1; exit 0
fi

# ----- test-internode ----- #

if [[ $TEST_MODE != "inter_node" ]]; then
    echo "Error: Unknown TEST_MODE=$TEST_MODE"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Error: Please specify the rank of this node."
    echo "Usage: ./run_distributed.sh <rank>"
    echo "Example: ./run_distributed.sh 0  (for master node 0)"
    exit 1
else
    echo "Launch with node rank: $1"
fi

# init dist env vars
export OMP_NUM_THREADS=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} # replace with your own master node IP
export MASTER_PORT=23457
export NNODES=2 # in deepep internode kernels, it will check num_ranks > NUM_MAX_NVL_PEERS, which equals to 8 by default
export NPROC_PER_NODE=8
export RANK=$1

if [[ $RANK -ge $NNODES ]]; then
    echo "Error: RANK=$RANK, but NNODES=$NNODES"
    exit 1
fi

# self-added env variable to control low-latency mode for test_internode.py
export GRPCOLL_TEST_INTERNODE_LL_COMPATIBILITY=0

CMD="torchrun \
--nproc_per_node=$NPROC_PER_NODE \
--nnodes=$NNODES \
--node_rank=$RANK \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
${TEST_ROOT}/test_internode_grpcoll.py
"

LOG_PATH=${LOG_ROOT}/test_internode_grpcoll_n${RANK}.log
echo "Logging to ${LOG_PATH} ..."
$CMD > ${LOG_PATH} 2>&1
