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
		SM8X_FLAG=1
		[[ "$ARCH_ARG" != *sm80* ]] && SM8X_FLAG=0
		echo "[magiattn] Installing cutlass ffa-fa4 for Hopper (SM90=1, SM8X=${SM8X_FLAG})"
		make install ARBITRARY=1 NUM_FUNC=1,3 HDIM128=1 SM90=1 SM8X=${SM8X_FLAG}
	fi

	cd "$REPO_ROOT"
fi

# Collect sub-package wheels for SCM distribution if MAGI_WHEEL_DIR is set.
if [[ -n "$MAGI_WHEEL_DIR" ]]; then
	echo "[magiattn] Collecting sub-package wheels into $MAGI_WHEEL_DIR..."

	for src_dir in \
		"${FA_DIR}/hopper" \
		"${FA_DIR}/csrc/utils/magi_to_hstu" \
		"${FA_DIR}/csrc/utils/create_block_mask" \
		"${FA_DIR}/flash_attn"; do
		if [[ -d "${REPO_ROOT}/${src_dir}" ]]; then
			echo "[magiattn] Building wheel from ${src_dir}..."
			pip wheel --no-deps --no-build-isolation --wheel-dir "$MAGI_WHEEL_DIR" "${REPO_ROOT}/${src_dir}" \
				|| echo "[magiattn] WARNING: Could not build wheel from ${src_dir}, skipping"
		fi
	done
fi
