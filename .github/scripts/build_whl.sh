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

echo "=== Package Build Operation ==="
echo "📦 Package Name: $1"
echo "---------------------------"

echo "🔍 Getting git commit ID..."
git_commit_id=${GITHUB_SHA:0:7}

echo "📁 Creating workspace directory..."
mkdir -p /workspace/${1}_${git_commit_id}

echo "🗑️  Cleaning existing files..."
rm -rf /workspace/${1}_${git_commit_id}/*

echo "🛠️  Building wheel package..."
python -m build --wheel --no-isolation -v

echo "📋 Copying wheel files..."
cp dist/*.whl /workspace/${1}_${git_commit_id}/

echo "✅ Build operation completed!"
