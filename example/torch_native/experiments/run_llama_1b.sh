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

set -ex

echo "Running a native torch job ..."

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=9010

BS=1
SEQLEN=8192
PRECISION="bf16=true"
#FSDP_CONFIG="llama3_fsdp_native.json"
#JOB_NAME="LLAMA3_FSDP_NATIVE_GPU${NPROC_PER_NODE}_BS${BS}_SEQLEN${SEQLEN}_BF16_FA"
cp_size=1
NPROC_PER_NODE=8
export cp_size=$cp_size

export WANDB_PROJECT="test"

torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    run_magi_clm.py \
    --num_train_epochs 2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --config_name /home/niubility2/sunhanwen/Megatron-LM/checkpoints/Llama-3.2-1b/ \
    --tokenizer_name /home/niubility2/sunhanwen/Megatron-LM/checkpoints/Llama-3.2-1b/ \
    --trust_remote_code true \
    --cache_dir ./cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --$PRECISION \
    --report_to wandb \
    --run_name magi-cp$cp_size \
    #--fsdp "auto_wrap" \
    #--fsdp_config $FSDP_CONFIG 2>&1 | tee ./$JOB_NAME.log
