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

echo "=== Requirements Installation ==="

unset http_proxy
unset https_proxy

if [ -f requirements.txt ]; then
    echo "📦 Installing requirements.txt..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
    echo "✅ Base requirements installed successfully!"
else
    echo "⚠️  No requirements.txt found. Skipping."
fi

if [ -f requirements_dev.txt ]; then
    echo "🛠️  Installing development requirements..."
    pip install -r requirements_dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
    echo "✅ Development requirements installed successfully!"
else
    echo "⚠️  No requirements_dev.txt found. Skipping."
fi
