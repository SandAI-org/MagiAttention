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

import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "magi_attn_extensions"

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    return str(public_version)


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "dist",
            "tests",
        )
    ),
    description="Extensions to provide supplementary utilities based on MagiAttention.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "magi_attention",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
