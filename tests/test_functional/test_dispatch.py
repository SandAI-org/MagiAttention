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
from magi_attention.meta import make_dispatch_meta_from_qk_ranges
from magi_attention.testing.dist_common import DistTestBase, with_comms

WORLD_SIZE = 4
SEED = 42


class TestDispatch(DistTestBase):

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    @property
    def seed(self) -> int:
        return SEED

    def _make_meta(self, total_seqlen: int, chunk_size: int, uneven_shard: bool = False):
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

    # ------------------------------------------------------------------
    # Forward round-trip: dispatch -> undispatch == identity
    # ------------------------------------------------------------------

    def _run_forward_roundtrip(
        self, total_seqlen, chunk_size, hidden, seq_dim=0, uneven_shard=False
    ):
        device = torch.cuda.current_device()
        torch.manual_seed(SEED)

        meta = self._make_meta(total_seqlen, chunk_size, uneven_shard=uneven_shard)
        group = self.process_group

        x_global = torch.randn(total_seqlen, hidden, device=device)

        x_local = dispatch_func(x_global=x_global, group=group, meta=meta, seq_dim=seq_dim)
        x_roundtrip = undispatch_func(x_local=x_local, meta=meta, group=group, seq_dim=seq_dim)

        self.assertTrue(
            torch.equal(x_roundtrip, x_global),
            f"roundtrip max diff={torch.max(torch.abs(x_roundtrip - x_global)).item()}",
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_forward_roundtrip(self):
        self._run_forward_roundtrip(total_seqlen=32, chunk_size=4, hidden=8)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_forward_roundtrip_large(self):
        self._run_forward_roundtrip(total_seqlen=256, chunk_size=16, hidden=32)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_forward_roundtrip_uneven(self):
        self._run_forward_roundtrip(
            total_seqlen=30, chunk_size=4, hidden=8, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_forward_roundtrip_uneven_large(self):
        self._run_forward_roundtrip(
            total_seqlen=250, chunk_size=16, hidden=32, uneven_shard=True
        )

    # ------------------------------------------------------------------
    # Backward: is_partial_grad=False (default)
    #   dispatch(x).sum().backward() should give grad=1 everywhere on x
    # ------------------------------------------------------------------

    def _run_backward_default(
        self, total_seqlen, chunk_size, hidden, seq_dim=0, uneven_shard=False
    ):
        device = torch.cuda.current_device()
        torch.manual_seed(SEED)

        meta = self._make_meta(total_seqlen, chunk_size, uneven_shard=uneven_shard)
        group = self.process_group

        x_global = torch.randn(total_seqlen, hidden, device=device, requires_grad=True)

        x_local = dispatch_func(x_global=x_global, group=group, meta=meta, seq_dim=seq_dim)
        x_roundtrip = undispatch_func(
            x_local=x_local, meta=meta, group=group, seq_dim=seq_dim, is_partial_grad=False
        )

        loss = x_roundtrip.sum()
        loss.backward()

        expected_grad = torch.ones_like(x_global)
        self.assertTrue(
            torch.allclose(x_global.grad, expected_grad),
            f"default backward max diff={torch.max(torch.abs(x_global.grad - expected_grad)).item()}",
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_backward_default(self):
        self._run_backward_default(total_seqlen=32, chunk_size=4, hidden=8)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_backward_default_large(self):
        self._run_backward_default(total_seqlen=256, chunk_size=16, hidden=32)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_backward_default_uneven(self):
        self._run_backward_default(
            total_seqlen=30, chunk_size=4, hidden=8, uneven_shard=True
        )

    # ------------------------------------------------------------------
    # Backward: is_partial_grad=True
    #
    # Scenario: each rank holds x_local, undispatch gathers to x_global,
    # then a function f(x_global) produces a loss.  With is_partial_grad,
    # the backward of undispatch uses reduce_scatter.
    #
    # To test correctness we compare against a manual reference:
    #   1. compute grad_global on each rank (all ranks share the same
    #      full grad thanks to the broadcast-like undispatch forward),
    #   2. the reference backward for each rank's x_local is:
    #      select_local_chunks(grad_global) — i.e. what the default
    #      backward does.
    #   With is_partial_grad=True the backward should give the same
    #   result *if every rank's grad_global is identical* (which it is
    #   when the forward output is the same all-gathered tensor).
    #   In that case reduce_scatter(grad, world_size) == grad / 1
    #   only if the partial contributions sum to the full grad.
    #
    # A cleaner test: each rank fabricates a *different* partial gradient
    # (simulating partial attention). We verify that the backward output
    # equals the sum of all ranks' select_local_chunks of their partials.
    # ------------------------------------------------------------------

    def _run_backward_partial_grad(
        self, total_seqlen, chunk_size, hidden, seq_dim=0, uneven_shard=False
    ):
        device = torch.cuda.current_device()
        rank = self.rank
        world_size = self.world_size
        torch.manual_seed(SEED)

        meta = self._make_meta(total_seqlen, chunk_size, uneven_shard=uneven_shard)
        group = self.process_group

        x_global = torch.randn(total_seqlen, hidden, device=device)
        x_local = dispatch_func(x_global=x_global, group=group, meta=meta, seq_dim=seq_dim)
        x_local_param = x_local.detach().clone().requires_grad_(True)

        x_gathered = undispatch_func(
            x_local=x_local_param,
            meta=meta,
            group=group,
            seq_dim=seq_dim,
            is_partial_grad=True,
        )

        torch.manual_seed(SEED + rank)
        partial_grad = torch.randn_like(x_gathered)

        x_gathered.backward(partial_grad)
        actual_grad = x_local_param.grad.clone()

        all_partial_grads = [torch.zeros_like(partial_grad) for _ in range(world_size)]
        dist.all_gather(all_partial_grads, partial_grad, group=group)
        summed_grad = sum(all_partial_grads)

        from magi_attention.functional.dispatch import _select_local_chunks

        expected_grad = _select_local_chunks(summed_grad, meta, rank, seq_dim)

        self.assertTrue(
            torch.allclose(actual_grad, expected_grad, atol=1e-5, rtol=1e-5),
            f"partial_grad backward max diff="
            f"{torch.max(torch.abs(actual_grad - expected_grad)).item()}",
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_backward_partial_grad(self):
        self._run_backward_partial_grad(total_seqlen=32, chunk_size=4, hidden=8)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_backward_partial_grad_large(self):
        self._run_backward_partial_grad(total_seqlen=256, chunk_size=16, hidden=32)

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_backward_partial_grad_uneven(self):
        self._run_backward_partial_grad(
            total_seqlen=30, chunk_size=4, hidden=8, uneven_shard=True
        )

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_backward_partial_grad_uneven_large(self):
        self._run_backward_partial_grad(
            total_seqlen=250, chunk_size=16, hidden=32, uneven_shard=True
        )

    # ------------------------------------------------------------------
    # Backward: is_partial_grad=True, uniform partial grad (all ones)
    #   When every rank passes grad=ones, reduce_scatter sums world_size
    #   copies → each rank's local grad should be world_size * ones_local.
    # ------------------------------------------------------------------

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_backward_partial_grad_uniform(self):
        device = torch.cuda.current_device()
        total_seqlen = 32
        chunk_size = 4
        hidden = 8
        seq_dim = 0

        torch.manual_seed(SEED)
        meta = self._make_meta(total_seqlen, chunk_size)
        group = self.process_group

        x_global = torch.randn(total_seqlen, hidden, device=device)
        x_local = dispatch_func(x_global=x_global, group=group, meta=meta, seq_dim=seq_dim)
        x_local_param = x_local.detach().clone().requires_grad_(True)

        x_gathered = undispatch_func(
            x_local=x_local_param,
            meta=meta,
            group=group,
            seq_dim=seq_dim,
            is_partial_grad=True,
        )

        grad_out = torch.ones_like(x_gathered)
        x_gathered.backward(grad_out)

        expected = torch.full_like(x_local_param, fill_value=float(self.world_size))
        self.assertTrue(
            torch.allclose(x_local_param.grad, expected, atol=1e-5, rtol=1e-5),
            f"uniform partial_grad backward max diff="
            f"{torch.max(torch.abs(x_local_param.grad - expected)).item()}",
        )


if __name__ == "__main__":
    run_tests()
