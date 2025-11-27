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

from .fa2_interface_with_sink import (
    fa2_func_with_sink,
    fa2_kvpacked_func_with_sink,
    fa2_qkvpacked_func_with_sink,
    fa2_varlen_func_with_sink,
    fa2_varlen_kvpacked_func_with_sink,
    fa2_varlen_qkvpacked_func_with_sink,
)
from .fa3_interface_with_sink import (
    fa3_func_with_sink,
    fa3_qkvpacked_func_with_sink,
    fa3_varlen_func_with_sink,
)

__all__ = [
    "fa2_func_with_sink",
    "fa2_qkvpacked_func_with_sink",
    "fa2_kvpacked_func_with_sink",
    "fa2_varlen_func_with_sink",
    "fa2_varlen_qkvpacked_func_with_sink",
    "fa2_varlen_kvpacked_func_with_sink",
    "fa3_func_with_sink",
    "fa3_varlen_func_with_sink",
    "fa3_qkvpacked_func_with_sink",
]

__version__ = "1.0.0"
