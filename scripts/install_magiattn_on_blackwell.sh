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

# --- Step1. Clone the MagiAttention repository and navigate into it
# which is skipped by default since you have probably already have the repo.

# git clone https://github.com/SandAI-org/MagiAttention && cd MagiAttention

# --- Step2. Initialize and update submodules

git submodule update --init --recursive

# --- Step3. Install flash_attn_cute as a core dependency
# for FA4 backend of MagiAttention on Blackwell

bash scripts/install_flash_attn_cute.sh

# --- Step4. Install other dependencies

pip install -r requirements.txt

# --- Step5. Set environment variables to skip building FFA
# which only supports up to Hopper

export MAGI_ATTENTION_PREBUILD_FFA=0

# --- Step6. Install MagiAttention in editable mode

pip install -e . -v --no-build-isolation

# --- Step7. Enable FA4 backend to use FFA_FA4 kernels

export MAGI_ATTENTION_FA4_BACKEND=1

# --- Step8. (Optional) Run pre-compilation of FFA_FA4 kernels 
# to avoid runtime re-compilation overheads

python tools/precompile_ffa_fa4.py
