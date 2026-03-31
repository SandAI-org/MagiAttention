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
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import DispatchConfig, MinHeapDispatchAlg
from magi_attention.functional.dispatch import dispatch_func, undispatch_func
from magi_attention.functional.roll import roll_p2p as roll_func
from magi_attention.functional.roll import roll_simple_p2p as roll_simple_func
from magi_attention.meta import make_dispatch_meta_from_qk_ranges
from magi_attention.testing.dist_common import DistTestBase, with_comms

WORLD_SIZE = 4
SEED = 42


class _RollTestMixin:
    """Shared helpers for roll test classes.  Subclasses set ``roll_fn``."""

    roll_fn = None  # override in subclass

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    @property
    def seed(self) -> int:
        return SEED

    def _make_meta(
        self, total_seqlen: int, chunk_size: int, uneven_shard: bool = False
    ):
        rank = self.rank
        cp_size = self.world_size

        q_ranges = AttnRanges.from_ranges([(0, total_seqlen)])
        k_ranges = AttnRanges.from_ranges([(0, total_seqlen)])
        attn_mask_type = [AttnMaskType.CAUSAL]

        dispatch_config = DispatchConfig(alg=MinHeapDispatchAlg())

        meta_q, _ = make_dispatch_meta_from_qk_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen,
            total_seqlen_k=total_seqlen,
            chunk_size=chunk_size,
            cp_rank=rank,
            cp_size=cp_size,
            dispatch_config=dispatch_config,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            uneven_shard=uneven_shard,
        )
        return meta_q

    def _run_roll_test(
        self, total_seqlen, chunk_size, hidden, shift, seq_dim=0, uneven_shard=False
    ):
        device = torch.cuda.current_device()
        torch.manual_seed(SEED)

        meta = self._make_meta(total_seqlen, chunk_size, uneven_shard=uneven_shard)
        group = self.process_group

        x_global = torch.randn(total_seqlen, hidden, device=device)

        x_local = dispatch_func(
            x_global=x_global, group=group, meta=meta, seq_dim=seq_dim
        )

        rolled_local = self.roll_fn(
            x_local=x_local, shift=shift, meta=meta, group=group, seq_dim=seq_dim
        )

        rolled_global = undispatch_func(
            x_local=rolled_local, meta=meta, group=group, seq_dim=seq_dim
        )

        ref_global = torch.roll(x_global, shifts=shift, dims=seq_dim)

        self.assertTrue(
            torch.equal(rolled_global, ref_global),
            f"shift={shift}: max diff={torch.max(torch.abs(rolled_global - ref_global)).item()}",
        )


class _RollTestCases:
    """All roll test methods.  Mixed into concrete test classes."""

    # ------------------------------------------------------------------
    # shift = 0 (no-op)
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_zero(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=0)

    # ------------------------------------------------------------------
    # shift is an exact multiple of chunk_size (whole-chunk transfer)
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_chunk_aligned(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=4)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_multi_chunk_aligned(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=12)

    # ------------------------------------------------------------------
    # shift is NOT a multiple of chunk_size (partial chunk transfer)
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_non_aligned(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=3)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_non_aligned_large(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=13)

    # ------------------------------------------------------------------
    # negative shift
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_negative(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=-5)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_negative_chunk_aligned(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=-8)

    # ------------------------------------------------------------------
    # shift >= total_seqlen (wraps around)
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_full_wrap(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=32)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_more_than_seqlen(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=37)

    # ------------------------------------------------------------------
    # shift = 1 (smallest non-trivial)
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_shift_one(self):
        self._run_roll_test(total_seqlen=32, chunk_size=4, hidden=8, shift=1)

    # ------------------------------------------------------------------
    # larger sequence with bigger chunk_size
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_large_sequence(self):
        self._run_roll_test(total_seqlen=256, chunk_size=16, hidden=32, shift=23)

    # ------------------------------------------------------------------
    # backward correctness
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_backward(self):
        device = torch.cuda.current_device()
        total_seqlen = 32
        hidden = 8
        chunk_size = 4
        shift = 5
        seq_dim = 0

        torch.manual_seed(SEED)
        meta = self._make_meta(total_seqlen, chunk_size)
        group = self.process_group

        x_global = torch.randn(total_seqlen, hidden, device=device)

        x_local = dispatch_func(
            x_global=x_global, group=group, meta=meta, seq_dim=seq_dim
        )
        x_local_for_grad = x_local.detach().clone().requires_grad_(True)

        rolled = self.roll_fn(
            x_local=x_local_for_grad,
            shift=shift,
            meta=meta,
            group=group,
            seq_dim=seq_dim,
        )

        loss = rolled.sum()
        loss.backward()

        grad = x_local_for_grad.grad
        self.assertIsNotNone(grad)
        expected_grad = torch.ones_like(x_local)
        self.assertTrue(
            torch.equal(grad, expected_grad),
            f"backward max diff={torch.max(torch.abs(grad - expected_grad)).item()}",
        )

    # ==================================================================
    # Uneven shard: total_seqlen % chunk_size != 0 (last chunk smaller)
    # ==================================================================

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_shift_zero(self):
        self._run_roll_test(
            total_seqlen=30, chunk_size=4, hidden=8, shift=0, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_shift_chunk_aligned(self):
        self._run_roll_test(
            total_seqlen=30, chunk_size=4, hidden=8, shift=4, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_shift_non_aligned(self):
        self._run_roll_test(
            total_seqlen=30, chunk_size=4, hidden=8, shift=3, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_cross_last_chunk(self):
        self._run_roll_test(
            total_seqlen=42, chunk_size=10, hidden=8, shift=3, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_shift_negative(self):
        self._run_roll_test(
            total_seqlen=30, chunk_size=4, hidden=8, shift=-5, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_shift_wrap(self):
        self._run_roll_test(
            total_seqlen=30, chunk_size=4, hidden=8, shift=37, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_shift_one(self):
        self._run_roll_test(
            total_seqlen=30, chunk_size=4, hidden=8, shift=1, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_last_chunk_size_one(self):
        self._run_roll_test(
            total_seqlen=33, chunk_size=4, hidden=8, shift=7, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_large_sequence(self):
        self._run_roll_test(
            total_seqlen=250, chunk_size=16, hidden=32, shift=23, uneven_shard=True
        )

    # ------------------------------------------------------------------
    # chunk_size = 1 (each chunk holds exactly 1 token)
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_chunk_size_one(self):
        self._run_roll_test(total_seqlen=16, chunk_size=1, hidden=8, shift=3)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_chunk_size_one_negative(self):
        self._run_roll_test(total_seqlen=16, chunk_size=1, hidden=8, shift=-5)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_roll_chunk_size_one_wrap(self):
        self._run_roll_test(total_seqlen=16, chunk_size=1, hidden=8, shift=19)

    # ------------------------------------------------------------------
    # backward correctness with uneven shard
    # ------------------------------------------------------------------
    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_uneven_roll_backward(self):
        device = torch.cuda.current_device()
        total_seqlen = 30
        hidden = 8
        chunk_size = 4
        shift = 5
        seq_dim = 0

        torch.manual_seed(SEED)
        meta = self._make_meta(total_seqlen, chunk_size, uneven_shard=True)
        group = self.process_group

        x_global = torch.randn(total_seqlen, hidden, device=device)

        x_local = dispatch_func(
            x_global=x_global, group=group, meta=meta, seq_dim=seq_dim
        )
        x_local_for_grad = x_local.detach().clone().requires_grad_(True)

        rolled = self.roll_fn(
            x_local=x_local_for_grad,
            shift=shift,
            meta=meta,
            group=group,
            seq_dim=seq_dim,
        )

        loss = rolled.sum()
        loss.backward()

        grad = x_local_for_grad.grad
        self.assertIsNotNone(grad)
        expected_grad = torch.ones_like(x_local)
        self.assertTrue(
            torch.equal(grad, expected_grad),
            f"backward max diff={torch.max(torch.abs(grad - expected_grad)).item()}",
        )


class TestRollP2P(_RollTestMixin, _RollTestCases, DistTestBase):
    roll_fn = staticmethod(roll_func)


class TestRollSimpleP2P(_RollTestMixin, _RollTestCases, DistTestBase):
    roll_fn = staticmethod(roll_simple_func)


if __name__ == "__main__":
    run_tests()
