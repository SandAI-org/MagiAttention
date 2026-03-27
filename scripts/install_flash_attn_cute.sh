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

REPO_ROOT="$(pwd)"
PATCH_SCRIPT="${REPO_ROOT}/scripts/patch_create_block_mask.py"
BLOCK_MASK_SETUP="magi_attention/functional/flash-attention/csrc/utils/create_block_mask/setup.py"

# Patch create_block_mask setup.py before any install step that might trigger it.
# The submodule's setup.py calls torch.cuda.is_available() to detect the GPU arch,
# which fails on SCM build machines (no GPU). Patch it to fall back to
# MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY env var.
if [[ -f "$BLOCK_MASK_SETUP" ]] && ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
	echo "[magiattn] Patching create_block_mask setup.py for headless build..."
	python3 "$PATCH_SCRIPT" "$BLOCK_MASK_SETUP"
fi

FA_DIR="magi_attention/functional/flash-attention"
cd "$FA_DIR"

echo "[magiattn] Installing cute ffa-fa4 (Blackwell support)"
bash install.sh

# Install cutlass version of ffa-fa4 for Ampere/Hopper support

if [[ "$ARCH_ARG" == *sm80* || "$ARCH_ARG" == *sm90* ]]; then
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

	cd "$REPO_ROOT"
fi

# Collect sub-package wheels for SCM distribution if MAGI_WHEEL_DIR is set.
# Build artifacts from the install steps above are reused (no recompilation).
if [[ -n "$MAGI_WHEEL_DIR" ]]; then
	echo "[magiattn] Collecting sub-package wheels into $MAGI_WHEEL_DIR..."
	for src_dir in \
		"${FA_DIR}/csrc/utils/magi_to_hstu" \
		"${FA_DIR}/csrc/utils/create_block_mask" \
		"${FA_DIR}/flash_attn" \
		"${FA_DIR}/hopper"; do
		if [[ -d "${REPO_ROOT}/${src_dir}" ]]; then
			echo "[magiattn] Building wheel from ${src_dir}..."
			pip wheel --no-deps --no-build-isolation --wheel-dir "$MAGI_WHEEL_DIR" "${REPO_ROOT}/${src_dir}" \
				|| echo "[magiattn] WARNING: Could not build wheel from ${src_dir}, skipping"
		fi
	done
fi
