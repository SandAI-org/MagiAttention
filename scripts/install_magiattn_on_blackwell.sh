# !/bin/bash

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

# --- Step0. Clone the MagiAttention repository and navigate into it
# which is skipped by default since you have probably already have the repo.

# git clone https://github.com/SandAI-org/MagiAttention && cd MagiAttention

# --- Step1. Checkout the branch for Blackwell support
# which is skipped by default since the working branch might have already been merged into main.

# git checkout optim_blackwell

# --- Step2. Initialize and update submodules

git submodule update --init --recursive

# --- Step3. Install flash_attn_cute as a core dependency

bash scripts/install_flash_attn_cute.sh

# --- Step4. Install other dependencies

pip install -r requirements.txt

# --- Step5. Set environment variables to skip building FFA
# which only supports up to NVIDIA Hopper architecture

export MAGI_ATTENTION_PREBUILD_FFA=0

# --- Step6. Install MagiAttention in editable mode

pip install -e . -v --no-build-isolation

# --- Step7. For now, to use magi_attention as usual on Blackwell, 
# you might need to set some extra environment variables as follows:

export MAGI_ATTENTION_FA4_BACKEND=1

# which we will work on improving the user experience in the future releases.
