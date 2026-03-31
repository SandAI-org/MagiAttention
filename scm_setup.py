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

import subprocess
import sys
from glob import glob

from setuptools import setup


def install_wheel(wheel_path):
    """Install the provided wheel using pip."""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            wheel_path,
            "--force-reinstall",
            "--no-deps",
        ]
    )


# Install all wheel files: magi_attention main wheel + sub-packages
# (flash_attn_cute, ffa_fa3, create_block_mask_cuda, magi_to_hstu_cuda, etc.)
wheel_files = sorted(glob("*.whl"))

if not wheel_files:
    raise FileNotFoundError("No wheel file found in package.")

# Installation order matters:
#   1. flash_attn_cute must be installed BEFORE ffa_fa3, because ffa_fa3
#      installs into flash_attn_cute.ffa_fa3 (a sub-package).
#   2. magi_attention must be installed last.
magi_wheels = [w for w in wheel_files if "magi_attention" in w]
ffa_fa3_wheels = [
    w for w in wheel_files if "ffa_fa3" in w and "magi_attention" not in w
]
other_wheels = [
    w for w in wheel_files if w not in magi_wheels and w not in ffa_fa3_wheels
]

for whl in other_wheels + ffa_fa3_wheels + magi_wheels:
    print(f"Installing {whl}...")
    install_wheel(whl)

setup(
    name="magi-attention-scm-installer",
    version="1.0.5",
    packages=[],
    install_requires=[],
    scripts=[],
    description="A fake package just using setup.py for installing wheels",
)
