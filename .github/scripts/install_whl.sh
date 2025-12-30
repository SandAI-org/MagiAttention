#!/bin/bash

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

# Ensure a package name is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide the package name as the first argument."
    exit 1
fi

echo "=== Package Install Operation ==="
echo "📦 Package Name: $1"
echo "-----------------------------"

echo "🔍 Getting git commit ID..."
# Extracts the first 7 characters of the SHA to match the folder name created in build_whl.sh
git_commit_id=${GITHUB_SHA:0:7}

# Define the source directory based on the build script's naming convention
SOURCE_DIR="/workspace/${1}_${git_commit_id}"

echo "📂 Locating source directory: ${SOURCE_DIR}..."

# Check if the directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Directory not found at ${SOURCE_DIR}"
    exit 1
fi

# Check if a .whl file exists in that directory
count=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.whl" | wc -l)
if [ "$count" -eq 0 ]; then
    echo "❌ Error: No .whl files found in ${SOURCE_DIR}"
    exit 1
fi

echo "🛠️  Installing wheel package..."
# Install the wheel found in the directory.
# --force-reinstall ensures the new version is installed even if the version number hasn't changed.
pip install --force-reinstall "${SOURCE_DIR}"/*.whl

echo "✅ Install operation completed!"
