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

import torch

__version__ = "1.0.0"


def get_num_sms() -> int:
    """Return the number of SMs on the current GPU."""
    return torch.cuda.get_device_properties(0).multi_processor_count


def get_dev_cap_num() -> int:
    """Return the SM version of the current GPU (e.g. 90 for Hopper, 100 for Blackwell)."""
    major, minor = torch.cuda.get_device_capability(0)
    return major * 10 + minor  # e.g. 90 for SM90, 100 for SM100


def get_dev_cap_str(upper: bool = False) -> str:
    """Return the SM version of the current GPU as a string (e.g. "sm90" for Hopper, "sm100" for Blackwell)."""
    capability = get_dev_cap_num()
    s = f"sm{capability}"
    return s.upper() if upper else s  # e.g. "sm90" or "SM90" (if upper)


def is_ampere() -> bool:
    """Return True iff the current CUDA device is Ampere (SM80+) but not Hopper (SM90+) or newer."""
    capability = get_dev_cap_num()
    return capability >= 80 and capability < 90


def is_hopper() -> bool:
    """Return True iff the current CUDA device is Hopper (SM90+) but not Blackwell (SM100+)."""
    capability = get_dev_cap_num()
    return capability >= 90 and capability < 100


def is_blackwell() -> bool:
    """Return True iff the current CUDA device is Blackwell (SM100+) or newer."""
    capability = get_dev_cap_num()
    return capability >= 100
