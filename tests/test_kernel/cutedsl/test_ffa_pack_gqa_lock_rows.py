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

"""Range-lock arithmetic of the packed atomic fwd, checked on the CPU.

Mirrors the SM100 range atomic fwd: a stage tile ``m_tile`` of a relation
starting at token ``q_start`` locks the blocks of global packed rows
``[q_start * G + m_tile * tile_m, ... + tile_m)``, and the host allocates
``ceil(total_q * G / tile_m) + 1`` blocks per kv head. Imports no kernel
module, so it runs without a GPU.
"""

import unittest

_TILE_M = 128


class TestFfaPackGqaLockRows(unittest.TestCase):
    def test_packed_rows_stay_inside_their_stage_locks(self):
        """Every valid packed row of a stage tile maps to a physical
        (token, q head) whose lock block is one of the (at most two) blocks
        the stage takes, and those blocks fit the host lock array."""
        total_q = 1000
        for group in (1, 2, 4, 8, 16, 32, 64, 128):
            num_lock_blocks = (total_q * group + _TILE_M - 1) // _TILE_M + 1
            for q_start in (0, 1, 3, 7, 64, 127, 333, 999):
                for len_q in (1, 2, 7, 63, 64, 65, 200, total_q - q_start):
                    len_q = min(len_q, total_q - q_start)
                    num_rows = len_q * group
                    for m_tile in range((num_rows + _TILE_M - 1) // _TILE_M):
                        lock_row = q_start * group + m_tile * _TILE_M
                        # Stage tiles start on a token boundary.
                        self.assertEqual(lock_row % group, 0)
                        blocks = {
                            lock_row // _TILE_M,
                            (lock_row + _TILE_M - 1) // _TILE_M,
                        }
                        self.assertLess(max(blocks), num_lock_blocks)
                        for tidx in range(min(_TILE_M, num_rows - m_tile * _TILE_M)):
                            p = m_tile * _TILE_M + tidx
                            token, g = q_start + p // group, p % group
                            self.assertLess(token, q_start + len_q)
                            self.assertIn((group * token + g) // _TILE_M, blocks)


if __name__ == "__main__":
    unittest.main()
