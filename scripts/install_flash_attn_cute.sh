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
#   bash scripts/install_flash_attn_cute.sh
#
# Installs the JIT-compiled cute FA-4 backend (Blackwell / sm100) used by the
# magi_attention FA4 kernel path. head_dim 64/128/192/256 are all supported
# automatically by the JIT cute kernel (no compile-time flag needed).
#
# Only the sm100 path is installed here; sm80/sm90 cutlass C++ builds are not
# part of this script.

set -e

REPO_ROOT="$(pwd)"
FA_DIR="magi_attention/functional/flash-attention"

if [[ ! -d "$FA_DIR" ]]; then
	echo "Error: $FA_DIR not found. Run this script from the MagiAttention repo root." >&2
	exit 1
fi

cd "$FA_DIR"

# 1) Install flash-attn-4 (cute backend, supports head_dim=256 on sm100).
echo "[magiattn] Installing flash-attn-4 (cute backend)"
pip install -e flash_attn/cute --no-build-isolation

# 2) Install create_block_mask_cuda (CUDA helper that builds the CSR block-sparse
#    mask consumed by the FA4 kernel).
echo "[magiattn] Installing create_block_mask_cuda"
pip install -e csrc/utils/create_block_mask --no-build-isolation

cd "$REPO_ROOT"

# 3) Sanity check: magi_attention's FA4 path also depends on magi_to_hstu_cuda.
#    Its source no longer ships with this flash-attention fork; it must be
#    supplied by the docker base image (or built from a separate source repo).
if ! python -c "import magi_to_hstu_cuda" 2>/dev/null; then
	echo ""
	echo "[magiattn] WARNING: magi_to_hstu_cuda is not importable."
	echo "  The FA4 backend will fail at FA4AttnArg.__post_init__ without it."
	echo "  Source no longer lives in flash-attention; install from your"
	echo "  docker base image or a separate source repo."
	echo ""
fi

# 4) Optional: collect sub-package wheels for SCM distribution.
if [[ -n "$MAGI_WHEEL_DIR" ]]; then
	echo "[magiattn] Collecting sub-package wheels into $MAGI_WHEEL_DIR..."

	PLAT_OPT=""
	if [[ -n "$MAGI_WHEEL_PLAT_NAME" ]]; then
		PLAT_OPT="--plat-name=$MAGI_WHEEL_PLAT_NAME"
	fi

	for src_dir in \
		"${FA_DIR}/csrc/utils/create_block_mask" \
		"${FA_DIR}/flash_attn/cute"; do
		if [[ -d "${REPO_ROOT}/${src_dir}" ]]; then
			echo "[magiattn] Building wheel from ${src_dir}..."
			(cd "${REPO_ROOT}/${src_dir}" \
				&& python setup.py bdist_wheel $PLAT_OPT \
				&& cp -f dist/*.whl "$MAGI_WHEEL_DIR/") \
				|| echo "[magiattn] WARNING: Could not build wheel from ${src_dir}, skipping"
		fi
	done
fi
