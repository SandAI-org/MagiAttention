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

"""Device-side probes of the deterministic q/k-range protocol, for tests.

The probes patch, at trace time, the Python functions the SM100 kernels
call (``range_chain`` and the DYNAMIC scheduler), so production modules
carry no test branch. A patched kernel differs from the normal one while
its compile key does not, so every probe context swaps in empty in-memory
compile caches: a probed kernel is never served to, or loaded from, a
normal launch.

- :func:`delay_injection` sleeps at one protocol step, to reorder the
  claims, arrivals, publications and conflict scans of the clusters.
- :func:`record_visits` counts, per CTA tile index, the warps that consume
  a valid DYNAMIC claim, to check that every claimed unit runs once on
  every CTA of one cluster.
"""

import enum
from contextlib import ExitStack, contextmanager
from typing import Iterator
from unittest import mock

import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from magi_attention.kernel.cutedsl import range_chain
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.tile_scheduler import (
    DynamicState,
    SingleTileVarlenScheduler,
)


class DelayPoint(enum.Enum):
    """Protocol step a :func:`delay_injection` sleeps at."""

    # Every CTA sleeps longer the lower its block index before the first
    # claim, so physical clusters claim out of launch order.
    START = "start"
    # Between claiming a ticket and publishing it to the cluster.
    CLAIM = "claim"
    # Before an arrival, on odd block indices: the second CTA of a 2-CTA
    # cluster, or every other CTA of a 1-CTA grid, arrives late.
    PEER_ARRIVE = "peer_arrive"
    # After the count completed and was reset, before the writer number is
    # published.
    PUBLISH = "publish"
    # Before a conflict-scan step, per lane.
    SCAN = "scan"


def _asm(asm: str, constraints: str, operands: list, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def random_sleep(salt: Int32, seed: int, max_ns: int, *, loc=None, ip=None) -> None:
    """Sleep, with probability 1/2, a pseudo-random time below ``max_ns``
    drawn from ``%globaltimer``, the block index, ``salt`` and ``seed``."""
    _asm(
        "{\n\t.reg .u32 t, h, ns;\n\t"
        "mov.u32 t, %globaltimer_lo;\n\t"
        "mov.u32 h, %ctaid.x;\n\t"
        "mad.lo.u32 h, h, 0x9E3779B1, t;\n\t"
        f"xor.b32 h, h, {seed & 0x7FFFFFFF};\n\t"
        "add.u32 h, h, $0;\n\t"
        "mul.lo.u32 h, h, 0x85EBCA6B;\n\t"
        "shr.u32 t, h, 13;\n\t"
        "xor.b32 h, h, t;\n\t"
        "mul.lo.u32 h, h, 0xC2B2AE35;\n\t"
        "shr.u32 t, h, 16;\n\t"
        "xor.b32 h, h, t;\n\t"
        f"rem.u32 ns, h, {max_ns};\n\t"
        "shr.u32 t, h, 31;\n\t"
        "mul.lo.u32 ns, ns, t;\n\t"
        "nanosleep.u32 ns;\n\t}",
        "r",
        [Int32(salt).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def odd_block_sleep(seed: int, max_ns: int, *, loc=None, ip=None) -> None:
    """:func:`random_sleep` on odd block indices only."""
    _asm(
        "{\n\t.reg .u32 t, h, ns;\n\t.reg .pred p;\n\t"
        "mov.u32 h, %ctaid.x;\n\t"
        "and.b32 t, h, 1;\n\t"
        "setp.ne.u32 p, t, 0;\n\t"
        "mov.u32 t, %globaltimer_lo;\n\t"
        "mad.lo.u32 h, h, 0x9E3779B1, t;\n\t"
        f"xor.b32 h, h, {seed & 0x7FFFFFFF};\n\t"
        "mul.lo.u32 h, h, 0x85EBCA6B;\n\t"
        "shr.u32 t, h, 15;\n\t"
        "xor.b32 h, h, t;\n\t"
        f"rem.u32 ns, h, {max_ns};\n\t"
        "@p nanosleep.u32 ns;\n\t}",
        "",
        [],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def launch_order_sleep(step_ns: int, *, loc=None, ip=None) -> None:
    """Sleep ``(gridDim.x - blockIdx.x) * step_ns``: low blocks start last."""
    _asm(
        "{\n\t.reg .u32 a, b;\n\t"
        "mov.u32 a, %nctaid.x;\n\t"
        "mov.u32 b, %ctaid.x;\n\t"
        "sub.u32 a, a, b;\n\t"
        f"mul.lo.u32 a, a, {step_ns};\n\t"
        "nanosleep.u32 a;\n\t}",
        "",
        [],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def count_visit(
    visits_addr: int,
    clusters_addr: int,
    tile_idx: Int32,
    valid: Int32,
    cluster_size: int,
    *,
    loc=None,
    ip=None,
) -> None:
    """Lane 0 of a warp holding a valid tile adds 1 to ``visits[tile_idx]``
    and its cluster index + 1 to ``clusters[tile_idx]``."""
    _asm(
        "{\n\t.reg .u32 lane, c;\n\t.reg .u64 off, a;\n\t.reg .pred p, q;\n\t"
        "mov.u32 lane, %laneid;\n\t"
        "setp.eq.u32 p, lane, 0;\n\t"
        "setp.ne.s32 q, $1, 0;\n\t"
        "and.pred p, p, q;\n\t"
        "mul.wide.s32 off, $0, 4;\n\t"
        "add.u64 a, off, $2;\n\t"
        "@p red.relaxed.gpu.global.add.s32 [a], 1;\n\t"
        "mov.u32 c, %ctaid.x;\n\t"
        f"div.u32 c, c, {cluster_size};\n\t"
        "add.u32 c, c, 1;\n\t"
        "add.u64 a, off, $3;\n\t"
        "@p red.relaxed.gpu.global.add.s32 [a], c;\n\t}",
        "r,r,l,l",
        [
            Int32(tile_idx).ir_value(loc=loc, ip=ip),
            Int32(valid).ir_value(loc=loc, ip=ip),
            Int64(visits_addr).ir_value(loc=loc, ip=ip),
            Int64(clusters_addr).ir_value(loc=loc, ip=ip),
        ],
        loc=loc,
        ip=ip,
    )


@contextmanager
def fresh_compile_caches() -> Iterator[None]:
    """Empty in-memory compile caches for both hosts within the context."""
    with mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ), mock.patch.object(_flex_flash_attn_bwd, "compile_cache", JITCache()):
        yield


# Per-event sleep bound of each point (START: per block index below the
# grid size). The claim and start points sleep once per tile or launch, so
# they take longer sleeps to reorder anything; nanosleep caps one sleep at
# about 1 ms.
_DEFAULT_MAX_NS = {
    DelayPoint.START: 3_000,
    DelayPoint.CLAIM: 200_000,
    DelayPoint.PEER_ARRIVE: 20_000,
    DelayPoint.PUBLISH: 20_000,
    DelayPoint.SCAN: 20_000,
}


@contextmanager
def delay_injection(
    point: DelayPoint, seed: int, max_ns: int | None = None
) -> Iterator[None]:
    """Kernels traced within the context sleep at ``point``."""
    max_ns = _DEFAULT_MAX_NS[point] if max_ns is None else max_ns
    stack = ExitStack()
    stack.enter_context(fresh_compile_caches())
    if point is DelayPoint.START:
        claim_first = SingleTileVarlenScheduler.claim_first_work

        def delayed_claim_first(self, *, loc=None, ip=None):
            launch_order_sleep(max_ns, loc=loc, ip=ip)
            claim_first(self, loc=loc, ip=ip)

        stack.enter_context(
            mock.patch.object(
                SingleTileVarlenScheduler, "claim_first_work", delayed_claim_first
            )
        )
    elif point is DelayPoint.CLAIM:
        publish_unit = DynamicState.publish

        def delayed_publish_unit(self, unit, cluster_size):
            random_sleep(unit, seed, max_ns)
            publish_unit(self, unit, cluster_size)

        stack.enter_context(
            mock.patch.object(DynamicState, "publish", delayed_publish_unit)
        )
    elif point is DelayPoint.PEER_ARRIVE:
        arrive = range_chain.arrive

        def delayed_arrive(mChain, block, head, writer, weight, cluster_size=1):
            odd_block_sleep(seed, max_ns)
            arrive(mChain, block, head, writer, weight, cluster_size)

        stack.enter_context(mock.patch.object(range_chain, "arrive", delayed_arrive))
    elif point is DelayPoint.PUBLISH:
        publish = range_chain.publish

        def delayed_publish(mChain, block, head, writer):
            random_sleep(writer, seed, max_ns)
            publish(mChain, block, head, writer)

        stack.enter_context(mock.patch.object(range_chain, "publish", delayed_publish))
    else:
        assert point is DelayPoint.SCAN
        scan = range_chain.scan_conflicts

        def delayed_scan(
            mRanges, mConflict, slot, range_from, range_to, block_size, last_writer
        ):
            random_sleep(range_from, seed, max_ns)
            scan(
                mRanges, mConflict, slot, range_from, range_to, block_size, last_writer
            )

        stack.enter_context(
            mock.patch.object(range_chain, "scan_conflicts", delayed_scan)
        )
    with stack:
        yield


@contextmanager
def record_visits(
    num_tiles: int, cluster_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Kernels traced within the context count their valid DYNAMIC claims.

    Yields ``(visits, clusters)``, int32 ``[num_tiles]``: per CTA tile index
    ``unit * cluster_size + rank``, the number of warps that consumed it as
    a valid tile, and the sum of their cluster indices + 1. Tile indices at
    or above ``num_tiles`` must not be valid.
    """
    visits = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    clusters = torch.zeros_like(visits)
    consume = SingleTileVarlenScheduler._consume_published_unit

    def counted_consume(self, *, loc=None, ip=None):
        work = consume(self, loc=loc, ip=ip)
        count_visit(
            visits.data_ptr(),
            clusters.data_ptr(),
            self._tile_idx,
            Int32(work.is_valid_tile),
            cluster_size,
        )
        return work

    with fresh_compile_caches(), mock.patch.object(
        SingleTileVarlenScheduler, "_consume_published_unit", counted_consume
    ):
        yield visits, clusters
