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

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# SM120 reuses the SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has a
# smaller shared memory capacity (99 KB vs 163 KB on SM80). This module simply
# subclasses the forked SM80 forward kernel (FFAFwdSm80) and overrides the SMEM
# capacity check accordingly.
#
# NOTE: SM120 is currently unverified (no hardware available). This is intentionally
# kept as a thin framework over the SM80 path; only the capacity check differs.

from .ffa_fwd_sm80 import FFAFwdSm80


class FFAFwdSm120(FFAFwdSm80):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.debug_print:
            print("[fwd_sm120_init] Using FFAFwdSm120 (SM80 MMA + SM120 SMEM capacity)")

    @property
    def smem_capacity_arch(self) -> str:
        """SM120 has a smaller SMEM budget (99 KB vs 163 KB on SM80)."""
        return "sm_120"
