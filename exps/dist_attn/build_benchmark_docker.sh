#! /bin/bash

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

# NOTE: This assumes that the base image already has MagiAttention installed.

# Make sure to log in before pushing, if necessary.
# docker login

docker build --network host -f ./Dockerfile.benchmark --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t magi-attn-benchmark:25.10.4 .

docker push magi-attn-benchmark:25.10.4
