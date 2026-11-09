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
from torch.testing._internal.common_utils import TestCase, run_tests

from magi_attention.kernel.cutedsl.index_sparse_rows import (
    MAX_CHUNKS,
    count_index_sparse_valid_ids,
    partition_index_sparse_rows_by_k_chunk,
)
from magi_attention.testing import parameterize

# (num_q_blocks, nhk, topk, total_k, num_chunks)
_full_row_cases = [
    (64, 1, 2048, 70000, 8),
    (40, 2, 300, 5000, 3),
    (8, 1, 4096, 524288, 16),
    (12, 1, 3000, 400000, 6),
    (6, 2, 5000, 2**20, MAX_CHUNKS),
]
_padded_row_cases = [
    (100, 2, 300, 5000, 5),
    (7, 3, 1, 10, 4),
    (16, 1, 5000, 400000, 6),
    (9, 2, 3000, 400000, 1),
]
# topk past 2**16
_long_row_cases = [
    (3, 1, 70000, 2**20, 3),
    (3, 1, 70000, 2**20, 5),
]


def _make_indices(num_q_blocks, nhk, topk, total_k, padded):
    """Unsorted rows with repeated ids; padded rows end in a -1 tail of any length."""
    indices = torch.randint(
        0, total_k, (num_q_blocks, nhk, topk), device="cuda", dtype=torch.int32
    )
    if padded:
        valid = torch.randint(0, topk + 1, (num_q_blocks, nhk, 1), device="cuda")
        col = torch.arange(topk, device="cuda")
        indices = torch.where(col < valid, indices, -1).int()
    return indices


def _partition_reference(indices, chunk_size, num_chunks):
    """Stable sort of every row by chunk id, the -1 tail sorting last."""
    key = torch.where(indices >= 0, indices // chunk_size, num_chunks)
    partitioned = indices.gather(-1, torch.sort(key, dim=-1, stable=True).indices)
    lens = torch.stack([(key == c).sum(-1) for c in range(num_chunks)]).int()
    starts = (torch.cumsum(lens, dim=0) - lens).int()
    return partitioned, starts, lens


class TestPartitionIndexSparseRowsByKChunk(TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("the partition kernel needs a GPU")
        torch.manual_seed(0)

    def _check(self, case, padded):
        num_q_blocks, nhk, topk, total_k, num_chunks = case
        chunk_size = (total_k + num_chunks - 1) // num_chunks
        indices = _make_indices(num_q_blocks, nhk, topk, total_k, padded)
        out = partition_index_sparse_rows_by_k_chunk(indices, chunk_size, num_chunks)
        ref = _partition_reference(indices, chunk_size, num_chunks)
        for name, a, b in zip(("partitioned", "starts", "lens"), out, ref):
            self.assertEqual(a.dtype, torch.int32, name)
            self.assertTrue(torch.equal(a, b), name)

    @parameterize("case", _full_row_cases)
    def test_groups_ids_by_chunk_keeping_row_order(self, case):
        self._check(case, padded=False)

    @parameterize("case", _padded_row_cases)
    def test_keeps_minus_one_tail(self, case):
        self._check(case, padded=True)

    @parameterize("case", _long_row_cases)
    def test_long_rows(self, case):
        self._check(case, padded=False)
        self._check(case, padded=True)

    def test_single_chunk_is_identity(self):
        indices = torch.randint(0, 1000, (16, 1, 128), device="cuda", dtype=torch.int32)
        partitioned, starts, lens = partition_index_sparse_rows_by_k_chunk(
            indices, 1000, 1
        )
        self.assertTrue(torch.equal(partitioned, indices))
        self.assertTrue(torch.equal(starts, torch.zeros_like(starts)))
        self.assertTrue(torch.equal(lens, torch.full_like(lens, 128)))


class TestCountIndexSparseValidIds(TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("the count kernel needs a GPU")
        torch.manual_seed(0)

    @parameterize("shape", [(100, 2, 300), (7, 3, 1), (16, 1, 5000), (2, 1, 70000)])
    def test_counts_ids_before_minus_one_tail(self, shape):
        num_q_blocks, nhk, topk = shape
        indices = _make_indices(num_q_blocks, nhk, topk, 2**31 - 1, padded=True)
        lens = count_index_sparse_valid_ids(indices)
        self.assertEqual(lens.dtype, torch.int32)
        self.assertTrue(torch.equal(lens, (indices >= 0).sum(-1, dtype=torch.int32)))


if __name__ == "__main__":
    run_tests()
