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

import unittest
from unittest import TestCase

import torch

from magi_attention.utils.index_sparse_utils import (
    _partition_index_sparse_rows_by_k_chunk_launch,
    partition_index_sparse_rows_by_k_chunk,
)


def _partition_reference(
    indices: torch.Tensor, chunk_size: int, num_chunks: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable sort of every row by chunk id, the -1 tail sorting last."""
    key = torch.where(indices >= 0, indices // chunk_size, num_chunks)
    partitioned = indices.gather(-1, torch.sort(key, dim=-1, stable=True).indices)
    lens = torch.stack([(key == c).sum(-1) for c in range(num_chunks)]).int()
    starts = (torch.cumsum(lens, dim=0) - lens).int()
    return partitioned, starts, lens


@unittest.skipUnless(torch.cuda.is_available(), "the partition kernel is Triton")
class TestPartitionIndexSparseRowsByKChunk(TestCase):
    def _check(self, num_q_blocks, nhk, topk, total_k, num_chunks, padded, tile=0):
        """``tile=0``: the public dispatch; ``None`` / an int force a kernel."""
        torch.manual_seed(0)
        chunk_size = (total_k + num_chunks - 1) // num_chunks
        indices = torch.randint(
            0, total_k, (num_q_blocks, nhk, topk), device="cuda", dtype=torch.int32
        )
        if padded:
            valid = torch.randint(0, topk + 1, (num_q_blocks, nhk, 1), device="cuda")
            col = torch.arange(topk, device="cuda")
            indices = torch.where(col < valid, indices, -1).int()
        if tile == 0:
            out = partition_index_sparse_rows_by_k_chunk(
                indices, chunk_size, num_chunks
            )
        else:
            out = _partition_index_sparse_rows_by_k_chunk_launch(
                indices, chunk_size, num_chunks, tile
            )
        ref = _partition_reference(indices, chunk_size, num_chunks)
        for name, a, b in zip(("partitioned", "starts", "lens"), out, ref):
            self.assertTrue(torch.equal(a, b), name)

    def test_groups_ids_by_chunk_keeping_row_order(self):
        """Full rows, one or several kv heads, chunk counts with and without a ragged group of 4."""
        for num_q_blocks, nhk, topk, total_k, num_chunks in (
            (64, 1, 2048, 70000, 8),
            (40, 2, 300, 5000, 3),
            (8, 1, 4096, 524288, 16),
        ):
            with self.subTest(topk=topk, num_chunks=num_chunks):
                self._check(num_q_blocks, nhk, topk, total_k, num_chunks, padded=False)

    def test_topk_past_16_bit_counts(self):
        """Rows of 2**16 ids and more, a chunk alone holding 2**16 ids or more."""
        for topk, total_k, num_chunks in (
            (2**16, 2**19, 3),
            (70000, 2**19, 5),
            (2**17, 2**19, 16),
            (400000, 400000, 2),
        ):
            with self.subTest(topk=topk, num_chunks=num_chunks):
                self._check(4, 1, topk, total_k, num_chunks, padded=False)
        self._check(3, 2, 70000, 2**19, 4, padded=True)

    def test_tiled_kernel(self):
        """Small tiles: ragged last tile, rows under a tile, ragged group of 4."""
        for num_q_blocks, nhk, topk, total_k, num_chunks, tile in (
            (40, 2, 300, 5000, 3, 128),
            (16, 1, 1000, 70000, 8, 256),
            (8, 1, 4096, 524288, 16, 512),
            (9, 3, 777, 3000, 17, 64),
            (5, 1, 100, 1000, 5, 128),
            (16, 1, 128, 1000, 1, 32),
        ):
            for padded in (False, True):
                with self.subTest(topk=topk, num_chunks=num_chunks, tile=tile):
                    self._check(
                        num_q_blocks, nhk, topk, total_k, num_chunks, padded, tile
                    )

    def test_sorted_rows(self):
        """Ascending rows come back as they are, the chunk bounds binary searched."""
        for num_q_blocks, nhk, topk, total_k, num_chunks in (
            (40, 2, 300, 5000, 3),
            (9, 3, 777, 3000, 17),
            (16, 1, 128, 1000, 1),
            (4, 1, 70000, 2**19, 6),
            (2, 1, 400000, 400000, 2),
        ):
            for padded in (False, True):
                with self.subTest(topk=topk, num_chunks=num_chunks, padded=padded):
                    torch.manual_seed(0)
                    chunk_size = (total_k + num_chunks - 1) // num_chunks
                    indices = torch.randint(
                        0, total_k, (num_q_blocks, nhk, topk), device="cuda"
                    )
                    indices = indices.sort(dim=-1).values.int()
                    if padded:
                        valid = torch.randint(
                            0, topk + 1, (num_q_blocks, nhk, 1), device="cuda"
                        )
                        col = torch.arange(topk, device="cuda")
                        indices = torch.where(col < valid, indices, -1).int()
                    out = partition_index_sparse_rows_by_k_chunk(
                        indices, chunk_size, num_chunks, rows_sorted=True
                    )
                    ref = _partition_reference(indices, chunk_size, num_chunks)
                    self.assertEqual(out[0].data_ptr(), indices.data_ptr())
                    for name, a, b in zip(("partitioned", "starts", "lens"), out, ref):
                        self.assertTrue(torch.equal(a, b), name)

    def test_keeps_minus_one_tail(self):
        """Rows with any number of valid ids, including none and all."""
        self._check(100, 2, 300, 5000, 5, padded=True)
        self._check(7, 3, 1, 10, 4, padded=True)

    def test_single_chunk_is_identity(self):
        indices = torch.randint(0, 1000, (16, 1, 128), device="cuda", dtype=torch.int32)
        partitioned, starts, lens = partition_index_sparse_rows_by_k_chunk(
            indices, 1000, 1
        )
        self.assertTrue(torch.equal(partitioned, indices))
        self.assertTrue(torch.equal(starts, torch.zeros_like(starts)))
        self.assertTrue(torch.equal(lens, torch.full_like(lens, 128)))


if __name__ == "__main__":
    unittest.main()
