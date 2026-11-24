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

import enum
from contextlib import ExitStack, contextmanager
from typing import Iterator
from unittest import mock

import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from magi_attention.kernel.cutedsl import range_write_order
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.tile_scheduler import SingleTileVarlenScheduler


class DelayPoint(enum.Enum):
    START = "start"
    CHAIN_WAIT = "chain_wait"
    PAIR_ARRIVE = "pair_arrive"


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
def odd_block_sleep(seed: int, max_ns: int, *, loc=None, ip=None) -> None:
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
    # Isolate patched kernels from normal compile caches (same compile key).
    with mock.patch.object(
        _flex_flash_attn_fwd, "compile_cache", JITCache()
    ), mock.patch.object(_flex_flash_attn_bwd, "compile_cache", JITCache()):
        yield


@contextmanager
def delay_injection(point: DelayPoint, seed: int, max_ns: int) -> Iterator[None]:
    # PTX nanosleep.u32 caps each sleep at ~1 ms.
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
    elif point is DelayPoint.PAIR_ARRIVE:
        pair_handoff_arrive = range_write_order.pair_handoff_arrive

        def delayed_pair_handoff_arrive(mbar, handoff, peer):
            odd_block_sleep(seed, max_ns)
            pair_handoff_arrive(mbar, handoff, peer)

        stack.enter_context(
            mock.patch.object(
                range_write_order, "pair_handoff_arrive", delayed_pair_handoff_arrive
            )
        )
    else:
        assert point is DelayPoint.CHAIN_WAIT
        wait_published = range_write_order.wait_published

        def delayed_wait_published(mChain, block, head, writer):
            odd_block_sleep(seed, max_ns)
            wait_published(mChain, block, head, writer)

        stack.enter_context(
            mock.patch.object(
                range_write_order, "wait_published", delayed_wait_published
            )
        )
    with stack:
        yield


@contextmanager
def record_visits(
    num_tiles: int, cluster_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    # clusters[tile] accumulates (cluster_idx + 1) per visiting warp.
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
