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

# Example:
#   bash scripts/install_flash_attn_cute.sh "sm80,sm90,sm100"

# Install cute version of ffa-fa4 for Blackwell support

ARCH_ARG="$1"

if [[ -z "$ARCH_ARG" ]]; then
	echo "Usage: $0 \"<arch_list>\" (e.g., \"sm80,sm90,sm100\")"
	exit 1
fi

cd magi_attention/functional/flash-attention

echo "[magiattn] Installing cute ffa-fa4 (Blackwell support)"
bash install.sh

# Install cutlass version of ffa-fa4 for Ampere/Hopper support

if [[ "$ARCH_ARG" == *sm80* || "$ARCH_ARG" == *sm90* ]]; then

	# Patch create_block_mask setup.py to support headless builds (no GPU).
	# The submodule's setup.py calls torch.cuda.is_available() to detect arch,
	# which fails on SCM build machines with no GPU. We patch it to fall back
	# to reading MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY env var instead.
	BLOCK_MASK_SETUP="csrc/utils/create_block_mask/setup.py"
	if [[ -f "$BLOCK_MASK_SETUP" ]] && ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
		echo "[magiattn] Patching create_block_mask setup.py for headless build..."
		SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
		python3 "${SCRIPT_DIR}/patch_create_block_mask.py" "$BLOCK_MASK_SETUP"
	fi

	cd hopper/

	# NOTE: see `Makefile` under this directory for required build options/flags
    # for example, NUM_FUNC=1,3 can only support the standard masks including full,causal,varlen-full,varlen-casual,sliding-window, etc
	if [[ "$ARCH_ARG" == *sm80* ]]; then
		echo "[magiattn] Installing cutlass ffa-fa4 for Ampere (SM8X=1)"
		make install ARBITRARY=1 NUM_FUNC=1,3 HDIM128=1 SM8X=1
	fi
	if [[ "$ARCH_ARG" == *sm90* ]]; then
		echo "[magiattn] Installing cutlass ffa-fa4 for Hopper (SM90=1)"
		make install ARBITRARY=1 NUM_FUNC=1,3 HDIM128=1 SM90=1
	fi
fi
