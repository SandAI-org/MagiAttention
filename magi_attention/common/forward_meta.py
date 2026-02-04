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

from dataclasses import dataclass

import torch


@dataclass
class AttnForwardMeta:
    """Attention forward metadata.

    Attributes:
        lse: Log-sum-exp of the attention weights. In a distributed setting, this is a
            local tensor where each device holds the LSE computed from its local query
            shards.
        max_logits: Maximum logits per query head. In a distributed setting,
            this is a replicated tensor where each device holds the global maximum
            computed across the entire sequence, ensuring consistency across all devices.
    """

    lse: torch.Tensor | None
    max_logits: torch.Tensor | None
