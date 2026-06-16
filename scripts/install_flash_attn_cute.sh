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
#   bash scripts/install_flash_attn_cute.sh                    # sm100-only (default)
#   bash scripts/install_flash_attn_cute.sh "sm80,sm90,sm100"  # also build cutlass FA for Ampere/Hopper
#
# Always installs the JIT-compiled cute FA-4 backend (Blackwell / sm100) used
# by the magi_attention FA4 kernel path. head_dim 64/128/192/256 are all
# supported automatically by the JIT cute kernel (no compile-time flag needed).
#
# If the optional first arg contains "sm80" or "sm90", this script also
# builds the cutlass ffa-fa4 C++ kernels under flash-attention/hopper for the
# matching architectures (mirrors the v1.1.1 behaviour). The default empty
# arg installs sm100 only, keeping docker build time low for Blackwell
# images.

set -e

ARCH_ARG="${1:-}"

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
#
#    Its setup.py auto-detects the target arch via torch.cuda.get_device_capability(),
#    which fails during docker build (no GPU exposed). When CUDA isn't available we
#    patch the setup.py to use $MAGI_ATTENTION_CUDA_ARCH (default 100, i.e. sm_100)
#    instead.
if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
	ARCH="${MAGI_ATTENTION_CUDA_ARCH:-100}"
	echo "[magiattn] No GPU detected; targeting sm_${ARCH} for create_block_mask_cuda"
	sed -i.bak \
		"s|raise RuntimeError(\"CUDA is not available, cannot determine target architecture\")|return [\"-gencode\", \"arch=compute_${ARCH},code=sm_${ARCH}\"]|" \
		csrc/utils/create_block_mask/setup.py
fi

echo "[magiattn] Installing create_block_mask_cuda"
pip install -e csrc/utils/create_block_mask --no-build-isolation

# 3) Install magi_to_hstu_cuda (CUDA helper that converts MagiAttention masks
#    into the HSTU format consumed by the FA4 backend).
#
#    setup.py hard-codes -gencode for sm_80/90/100, so it builds fine without
#    a GPU exposed at docker-build time — no patching needed.
echo "[magiattn] Installing magi_to_hstu_cuda"
pip install -e csrc/utils/magi_to_hstu --no-build-isolation

# 3b) Install cutlass ffa-fa4 for Ampere/Hopper if explicitly requested via
#     the ARCH_ARG (mirrors the v1.1.1 install_flash_attn_cute.sh behaviour).
#     Skipped when ARCH_ARG is empty or only requests sm100 — keeping the
#     default docker build time low for Blackwell-only images.
if [[ "$ARCH_ARG" == *sm80* || "$ARCH_ARG" == *sm90* ]]; then
	cd hopper/

	# NOTE: see `Makefile` under this directory for required build options/flags
	# for example, NUM_FUNC=1,3 can only support the standard masks including full,causal,varlen-full,varlen-casual,sliding-window, etc
	if [[ "$ARCH_ARG" == *sm80* ]]; then
		echo "[magiattn] Installing cutlass ffa-fa4 for Ampere (SM8X=1)"
		make install ARBITRARY=1 NUM_FUNC=1,3,5,7 HDIM128=1 SM8X=1
	fi
	if [[ "$ARCH_ARG" == *sm90* ]]; then
		SM8X_FLAG=1
		[[ "$ARCH_ARG" != *sm80* ]] && SM8X_FLAG=0
		echo "[magiattn] Installing cutlass ffa-fa4 for Hopper (SM90=1, SM8X=${SM8X_FLAG})"
		make install ARBITRARY=1 NUM_FUNC=1,3,5,7 HDIM128=1 SM90=1 SM8X=${SM8X_FLAG}
	fi

	cd ..
fi

cd "$REPO_ROOT"

# 4) Sanity check: magi_attention's FA4 path depends on magi_to_hstu_cuda
#    being importable. The install above should have made it available; warn
#    loudly if not (e.g. the build silently failed).
if ! python -c "import magi_to_hstu_cuda" 2>/dev/null; then
	echo ""
	echo "[magiattn] WARNING: magi_to_hstu_cuda is not importable after install."
	echo "  The FA4 backend will fail at FA4AttnArg.__post_init__ without it."
	echo "  Check the pip install output above for build errors."
	echo ""
fi

# 5) Optional: collect sub-package wheels for SCM distribution.
if [[ -n "$MAGI_WHEEL_DIR" ]]; then
	echo "[magiattn] Collecting sub-package wheels into $MAGI_WHEEL_DIR..."

	PLAT_OPT=""
	if [[ -n "$MAGI_WHEEL_PLAT_NAME" ]]; then
		PLAT_OPT="--plat-name=$MAGI_WHEEL_PLAT_NAME"
	fi

	for src_dir in \
		"${FA_DIR}/csrc/utils/create_block_mask" \
		"${FA_DIR}/csrc/utils/magi_to_hstu" \
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
