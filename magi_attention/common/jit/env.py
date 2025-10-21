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

# Copyright (c) 2024 by FlashInfer team.
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

# NOTE(lequn): Do not "from .jit.env import xxx".
# Do "from .jit import env as jit_env" and use "jit_env.xxx" instead.
# This helps AOT script to override envs.

import os
import pathlib
import re
import warnings

from torch.utils.cpp_extension import _get_cuda_arch_flags

MAGI_ATTENTION_BASE_DIR = pathlib.Path(
    os.getenv("MAGI_ATTENTION_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

MAGI_ATTENTION_CACHE_DIR = MAGI_ATTENTION_BASE_DIR / ".cache" / "magi_attention"


def _get_workspace_dir_name() -> pathlib.Path:
    try:
        with warnings.catch_warnings():
            # Ignore the warning for TORCH_CUDA_ARCH_LIST not set
            warnings.filterwarnings(
                "ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch"
            )
            flags = _get_cuda_arch_flags()
        arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
    except Exception:
        arch = "noarch"
    # e.g.: $HOME/.cache/MAGI/75_80_89_90/
    return MAGI_ATTENTION_CACHE_DIR / arch


# use pathlib
MAGI_ATTENTION_WORKSPACE_DIR = _get_workspace_dir_name()
MAGI_ATTENTION_JIT_DIR = MAGI_ATTENTION_WORKSPACE_DIR / "cached_ops"
MAGI_ATTENTION_GEN_SRC_DIR = MAGI_ATTENTION_WORKSPACE_DIR / "generated"
_package_root = pathlib.Path(__file__).resolve().parents[2]

MAGI_ATTENTION_CSRC_DIR = _package_root / "csrc"
FLEXIBLE_FLASH_ATTENTION_CSRC_DIR = MAGI_ATTENTION_CSRC_DIR / "flexible_flash_attention"
MAGI_ATTENTION_CSRC_UTILS_DIR = MAGI_ATTENTION_CSRC_DIR / "utils"
MAGI_ATTENTION_INCLUDE_DIR = MAGI_ATTENTION_CSRC_DIR / "common"
# MAGI_SRC_DIR = _package_root / "data" / "src"
MAGI_ATTENTION_AOT_DIR = _package_root / "lib"
CUTLASS_INCLUDE_DIRS = [
    _package_root / "csrc" / "cutlass" / "include",
    _package_root / "csrc" / "cutlass" / "tools" / "util" / "include",
]
