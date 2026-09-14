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


set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
cd "$repo_root"
"$python_bin" -m pip install -r requirements.txt \
    "nvidia-cudnn-frontend[cutedsl] @ git+https://github.com/NVIDIA/cudnn-frontend.git@705e9ca89b9b73f7594551e745d96790dd227e7e"
# Install the interval-mask converter shared with FFA_FA4.
"$python_bin" -m pip install --no-build-isolation \
    ./magi_attention/functional/flash-attention/csrc/utils/magi_to_hstu
MAGI_ATTENTION_PREBUILD_FFA=0 "$python_bin" -m pip install -e . --no-build-isolation
printf '%s\n' 'Select this backend with MAGI_ATTENTION_KERNEL_BACKEND=cudnn.'
printf '%s\n' 'Unset MAGI_ATTENTION_FA4_BACKEND and MAGI_ATTENTION_SDPA_BACKEND first.'
