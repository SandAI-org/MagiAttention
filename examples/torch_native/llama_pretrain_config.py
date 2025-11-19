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

data_config = dict(
    seqlen=2048,
    batch_size=8,
)

train_config = dict(
    train_iters=10,
    optimizer_config=dict(
        learning_rate=1e-3,
        weight_decay=1e-2,
        max_clip_grad_norm=1.0,
        grad_rescale_factor=1e2,
    ),
)

parallel_config = dict(
    context_parallel_size=2, context_parallel_backend="magi_attention"
)
