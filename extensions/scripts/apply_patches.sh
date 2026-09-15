#!/usr/bin/env bash

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

# Apply tools/patches/*.patch to the installed flash_attn package. These
# temporary patches provide features consumed by magi_attn_extensions, such as
# return_fp32_out and save_fp32_out_for_bwd.
#
# Usage:
#   bash scripts/apply_patches.sh
#   bash scripts/apply_patches.sh --dry-run
#   bash scripts/apply_patches.sh --reverse
#   bash scripts/apply_patches.sh --reverse --dry-run
#
# Environment variables:
#   MAGI_ATTN_EXTENSIONS_APPLY_PATCHES=1  enable patching (default)
#   MAGI_ATTN_EXTENSIONS_APPLY_PATCHES=0  skip patching
#   PYTHON=python3                         interpreter containing flash_attn
#
# Exit codes:
#   0  all patches applied/reversed, already in the requested state, or skipped
#   1  invalid arguments or a patch cannot be applied/reversed cleanly
#   2  flash_attn cannot be located in the selected Python environment

set -euo pipefail

DRY_RUN=0
REVERSE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --reverse) REVERSE=1 ;;
        *)
            echo "[magi_attn_extensions.apply_patches] Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

APPLY_PATCHES="${MAGI_ATTN_EXTENSIONS_APPLY_PATCHES:-1}"
if [[ "$APPLY_PATCHES" != "1" ]]; then
    echo "[magi_attn_extensions.apply_patches] MAGI_ATTN_EXTENSIONS_APPLY_PATCHES=$APPLY_PATCHES — skipping."
    exit 0
fi

PYTHON_BIN="${PYTHON:-python3}"
FLASH_ATTN_PKG_DIR="$($PYTHON_BIN -c '
import importlib.util
import pathlib

spec = importlib.util.find_spec("flash_attn")
if spec is None or spec.origin is None:
    raise SystemExit(1)
print(pathlib.Path(spec.origin).resolve().parent)
' 2>/dev/null)" || true
if [[ -z "$FLASH_ATTN_PKG_DIR" || ! -d "$FLASH_ATTN_PKG_DIR" ]]; then
    echo "[magi_attn_extensions.apply_patches] ERROR: could not locate flash_attn with '$PYTHON_BIN'." >&2
    echo "[magi_attn_extensions.apply_patches] Install flash-attn4 or set PYTHON to the correct interpreter." >&2
    exit 2
fi
echo "[magi_attn_extensions.apply_patches] flash_attn package dir: $FLASH_ATTN_PKG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTENSIONS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCHES_DIR="$EXTENSIONS_ROOT/tools/patches"

if [[ ! -d "$PATCHES_DIR" ]]; then
    echo "[magi_attn_extensions.apply_patches] No patches directory at $PATCHES_DIR — nothing to do."
    exit 0
fi

shopt -s nullglob
PATCH_FILES=("$PATCHES_DIR"/*.patch)
shopt -u nullglob
if [[ ${#PATCH_FILES[@]} -eq 0 ]]; then
    echo "[magi_attn_extensions.apply_patches] No *.patch files in $PATCHES_DIR — nothing to do."
    exit 0
fi

# Patch paths are rooted at a/flash_attn/...; strip both components and apply
# relative to the imported flash_attn package directory.
PATCH_BASE_FLAGS=(
    --batch
    --strip=2
    --directory="$FLASH_ATTN_PKG_DIR"
    --reject-file=-
    --no-backup-if-mismatch
)

probe_forward() {
    patch --dry-run --forward "${PATCH_BASE_FLAGS[@]}" < "$1" >/dev/null 2>&1
}

probe_reverse() {
    patch --dry-run --reverse --forward "${PATCH_BASE_FLAGS[@]}" < "$1" >/dev/null 2>&1
}

CHANGED=0
SKIPPED=0
FAILED=0

for patchfile in "${PATCH_FILES[@]}"; do
    pname="$(basename "$patchfile")"

    if [[ "$REVERSE" -eq 1 ]]; then
        if probe_reverse "$patchfile"; then
            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "[magi_attn_extensions.apply_patches] DRY RUN: would reverse $pname"
            else
                echo "[magi_attn_extensions.apply_patches] Reversing $pname ..."
                patch --reverse --forward "${PATCH_BASE_FLAGS[@]}" < "$patchfile"
            fi
            (( CHANGED++ )) || true
        elif probe_forward "$patchfile"; then
            echo "[magi_attn_extensions.apply_patches] SKIP (already reversed): $pname"
            (( SKIPPED++ )) || true
        else
            echo "[magi_attn_extensions.apply_patches] FAILED: cannot reverse $pname cleanly" >&2
            (( FAILED++ )) || true
        fi
    else
        if probe_forward "$patchfile"; then
            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "[magi_attn_extensions.apply_patches] DRY RUN: would apply $pname"
            else
                echo "[magi_attn_extensions.apply_patches] Applying $pname ..."
                patch --forward "${PATCH_BASE_FLAGS[@]}" < "$patchfile"
            fi
            (( CHANGED++ )) || true
        elif probe_reverse "$patchfile"; then
            echo "[magi_attn_extensions.apply_patches] SKIP (already applied): $pname"
            (( SKIPPED++ )) || true
        else
            echo "[magi_attn_extensions.apply_patches] FAILED: cannot apply $pname cleanly" >&2
            (( FAILED++ )) || true
        fi
    fi
done

# The FP32-O kernel patch is not sufficient by itself: the Python dispatcher
# must also expose out_dtype and forward the FP32 mode into the SM100 kernel.
# Fail during setup rather than several minutes later in the first train step.
if [[ "$REVERSE" -eq 0 && "$DRY_RUN" -eq 0 && "$FAILED" -eq 0 ]]; then
    if ! "$PYTHON_BIN" -c '
import inspect

from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.interface import _flash_attn_fwd

fwd_params = inspect.signature(_flash_attn_fwd).parameters
kernel_params = inspect.signature(FlashAttentionForwardSm100).parameters
required_fwd = {"out_dtype"}
required_kernel = {"fp32_O", "use_block_sparsity", "has_page_table"}
missing = (required_fwd - fwd_params.keys()) | (required_kernel - kernel_params.keys())
if missing:
    raise SystemExit(f"missing patched FA4 parameters: {sorted(missing)}")
'; then
        echo "[magi_attn_extensions.apply_patches] FAILED: patched FA4 FP32-O interface validation failed" >&2
        (( FAILED++ )) || true
    else
        echo "[magi_attn_extensions.apply_patches] Verified FA4 FP32-O interface."
    fi
fi

if [[ "$REVERSE" -eq 1 ]]; then
    requested_state="reversed"
else
    requested_state="applied"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
    changed_label="would_change"
else
    changed_label="changed"
fi

echo
echo "[magi_attn_extensions.apply_patches] Done. requested=$requested_state $changed_label=$CHANGED skipped=$SKIPPED failed=$FAILED"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
