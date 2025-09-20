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

TEST_ROOT=.
LOG_ROOT=${TEST_ROOT}/outs

export PYTHONPATH=$PYTHONPATH:.

# grpcoll test will set the env vars in the script
# export NVSHMEM_IB_ENABLE_IBGDA=1
# export NVSHMEM_IBGDA_NIC_HANDLER=gpu
# export NVSHMEM_DISABLE_P2P=0 # set to 0 to enable NVLink in low-latency mode
# export NVSHMEM_SYMMETRIC_SIZE=2**30 # default: 1GB


# ----- test-intranode ----- #

# self-added env variable to control low-latency mode for test_intranode.py
# FIXME: enable this wll raise the error:
#   assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
export GRPCOLL_TEST_INTRANODE_LOW_LATENCY=0

# LOG_PATH=${LOG_ROOT}/test_intranode_grpcoll.log
# echo "Logging to ${LOG_PATH} ..."
# python ${TEST_ROOT}/test_intranode_grpcoll.py > ${LOG_PATH} 2>&1; exit 0

# ----- test-low-latency ----- #

# self-added env variable to control allow-nvlink mode for test_low_latency.py
export GRPCOLL_TEST_LOW_LATENCY_ALLOW_NVLINK=1

# LOG_PATH=${LOG_ROOT}/test_low_latency_grpcoll.log
# echo "Logging to ${LOG_PATH} ..."
# python ${TEST_ROOT}/test_low_latency_grpcoll.py > ${LOG_PATH} 2>&1; exit 0


# ----- test-internode ----- #

if [ -z "$1" ]; then
    echo "Error: Please specify the rank of this node."
    echo "Usage: ./run_distributed.sh <rank>"
    echo "Example: ./run_distributed.sh 0  (for master node 0)"
    exit 1
else
    echo "Launch with node rank: $1"
fi

# init dist env vars

if [[ -f .env ]]; then
    source .env # maybe put your own master node IP here
fi

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
