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

if [[ -f .env ]]; then
    source .env # maybe put your own master node IP here
fi

TEST_ROOT=.
LOG_ROOT=${TEST_ROOT}/outs
TEST_MODE=${TEST_MODE:-"intra_node"} # intra_node | low_latency | internode

mkdir -p ${LOG_ROOT}

# Set common env vars
export PYTHONPATH=$PYTHONPATH:.
export OMP_NUM_THREADS=1
# export CUDA_LAUNCH_BLOCKING=1

# Set nccl env vars
# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=${SOCKET_IFNAME:-"bond0"}

# Set nvshmem env vars
# export NVSHMEM_DEBUG=INFO
# export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
# export NVSHMEM_IBGDA_ENABLE_MULTI_PORT=1
# export NVSHMEM_HCA_LIST=mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17
# export NVSHMEM_IB_ADDR_FAMILY=AF_INET
# export NVSHMEM_IB_ADDR_RANGE=0.0.0.0/0
# export NVSHMEM_IB_GID_INDEX=3
# export NVSHMEM_IB_TRAFFIC_CLASS=128
# export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NCCL_SOCKET_IFNAME}


# ----- test-intranode ----- #

if [[ $TEST_MODE == "intra_node" ]]; then
    LOG_PATH=${LOG_ROOT}/test_intranode_grpcoll.log
    echo "Logging to ${LOG_PATH} ..."
    python ${TEST_ROOT}/test_intranode_grpcoll.py > ${LOG_PATH} 2>&1
    exit $?
fi

# ----- test-low-latency ----- #

if [[ $TEST_MODE == "low_latency" ]]; then
    LOG_PATH=${LOG_ROOT}/test_low_latency_grpcoll.log
    echo "Logging to ${LOG_PATH} ..."
    python ${TEST_ROOT}/test_low_latency_grpcoll.py > ${LOG_PATH} 2>&1
    exit $?
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

# Init multi-node dist env vars
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} # replace with your own master node IP
export MASTER_PORT=23457
export NNODES=2 # in deepep internode kernels, it will check num_ranks > NUM_MAX_NVL_PEERS, which equals to 8 by default
export NPROC_PER_NODE=8
export RANK=$1

if [[ $RANK -ge $NNODES ]]; then
    echo "Error: RANK=$RANK, but NNODES=$NNODES"
    exit 1
fi

echo "Multi-Node Distributed settings: MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT, NNODES=$NNODES, NPROC_PER_NODE=$NPROC_PER_NODE, RANK=$RANK"

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
