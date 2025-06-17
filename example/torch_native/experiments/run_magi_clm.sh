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

[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=29533

BS=1
SEQLEN=8192
PRECISION="bf16=true"
CP_SIZE=$1
GA=$2
NPROC_PER_NODE=8
export cp_size=$CP_SIZE

export WANDB_PROJECT="your_wandb_project"
MODEL_PATH="your_model_path"

torchrun --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    run_magi_clm.py \
    --num_train_epochs 2 \
    --dataset_name openwebtext \
    --use_fast_tokenizer false \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --do_train \
    --output_dir your_checkpoint_path \
    --overwrite_output_dir \
    --config_name $MODEL_PATH \
    --tokenizer_name $MODEL_PATH \
    --model_name_or_path $MODEL_PATH \
    --trust_remote_code true \
    --cache_dir ./cache \
    --block_size $SEQLEN \
    --optim adamw_torch \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --save_strategy no \
    --logging_strategy steps \
    --gradient_checkpointing no \
    --gradient_accumulation_steps $GA \
    --logging_steps 1 \
    --$PRECISION \
    --report_to wandb \
    --max_steps 2000 \
    --run_name magi-cp$CP_SIZE-ga$GA \
