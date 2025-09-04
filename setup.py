# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import os

from setuptools import find_packages, setup

this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "magi_fsdp"

with open(os.path.join(this_dir, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=[
            "build",
            "tests",
            "dist",
            "docs",
            "tools",
            "assets",
        ],
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    # ext_modules=[],
    # cmdclass={},
)
