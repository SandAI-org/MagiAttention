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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl

from .triton_utils import get_flat_bid, get_flat_tid

# ".reg .u32   %tmp32_<1>;":
#   allocate a 32-bit register named %tmp32_0
# ".reg .pred  %p<1>;":
#   allocate a predicate register named %p0
# "send_signal:":
#   the loop label used for bra instruction
# "atom.global.{order}.sys.cas.b32 %tmp32_0, [$1], 0, 1;":
#   atomicCAS($1, 0, 1) in the given memory order, choosing from "relaxed" or "release",
#   where $1 indicates the input address, and the returned old value is stored in %tmp32_0
# "setp.eq.u32 %p0, %tmp32_0, 0;":
#   set %p0 to true if %tmp32_0 is equal to 0
#   i.e. when the input address is set to 1, the %tmp32_0 will be 0, then the predicate is true
# "@!%p0 bra send_signal;":
#   if the predicate is false, branch to the loop label
#   i.e. when the input address is set to 1, the predicate is true, then break the loop,
#   otherwise, continue the loop
put_signal_asm_template = """
{{
    .reg .u32   %tmp32_<1>;
    .reg .pred  %p<1>;

    send_signal:
        atom.global.{order}.sys.cas.b32 %tmp32_0, [$1], 0, 1;
        setp.eq.u32 %p0, %tmp32_0, 0;
        @!%p0 bra send_signal;
}}
"""

relaxed_put_signal_asm = triton.language.constexpr(
    put_signal_asm_template.format(order="relaxed")
)
release_put_signal_asm = triton.language.constexpr(
    put_signal_asm_template.format(order="release")
)


@triton.jit
def put_signal(addrs, sem: tl.constexpr):
    """put the signal into the remote addresses
    with the given semantics to decide the memory order
    """

    if sem == "relaxed":
        tl.inline_asm_elementwise(
            relaxed_put_signal_asm,  # relaxed order
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            release_put_signal_asm,  # release order for store
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


# ".reg .u32   %tmp32_<1>;":
#   allocate a 32-bit register named %tmp32_0
# ".reg .pred  %p<1>;":
#   allocate a predicate register named %p0
# "wait_signal:":
#   the loop label used for bra instruction
# "atom.global.{order}.sys.cas.b32 %tmp32_0, [$1], 1, 0;":
#   atomicCAS($1, 1, 0) in the given memory order, choosing from "relaxed" or "acquire",
#   where $1 indicates the input address, and the returned old value is stored in %tmp32_0
# "setp.eq.u32 %p0, %tmp32_0, 1;":
#   set %p0 to true if %tmp32_0 is equal to 1
#   i.e. when the input address is reset to 0, the %tmp32_0 will be 1, then the predicate is true
# "@!%p0 bra wait_signal;":
#   if the predicate is false, branch to the loop label
#   i.e. when the input address is reset to 0, the predicate is true, then break the loop,
#   otherwise, continue the loop
wait_signal_asm_template = """
{{
    .reg .u32   %tmp32_<1>;
    .reg .pred  %p<1>;

    wait_signal:
        atom.global.sys.{order}.cas.b32 %tmp32_0, [$1], 1, 0;
        setp.eq.u32 %p0, %tmp32_0, 1;
        @!%p0 bra wait_signal;
}}
"""

relaxed_wait_signal_asm = triton.language.constexpr(
    wait_signal_asm_template.format(order="relaxed")
)
acquire_wait_signal_asm = triton.language.constexpr(
    wait_signal_asm_template.format(order="acquire")
)


@triton.jit
def wait_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            relaxed_wait_signal_asm,  # relaxed order
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            acquire_wait_signal_asm,  # acquire order
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


@triton.jit
def blockwise_barrier(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    """
    Synchronizes blocks with matching block_id across participating devices.

    Note: the function itself is not a system level barrier/fence. It is a
    building block for expressing different synchronization patterns.

    Pattern 0: Ensures that all writes to symm_mem buffers from previous
    kernels across all devices are visible to the current kernel:

        blockwise_barrier(..., sem="relaxed")
        sync_threads()

    Pattern 1: Ensures that all writes to symm_mem buffers from the current
    block are visible to all remote blocks with matching blockIdx:

        sync_threads()
        blockwise_barrier(..., sem="acq_rel")
        sync_threads()

    Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
    for writing by subsequent kernels across all devices.

        sync_threads()
        blockwise_barrier(..., sem="relaxed")

    CUDA graph friendliness:

        This barrier operates through atomic operations on a zero-filled signal pad,
        which resets to a zero-filled state after each successful synchronization.
        This design eliminates the need for incrementing a flag from host.
    """

    flat_tid = get_flat_tid()
    if flat_tid >= world_size:
        # only each block's first world_size threads
        # are responsible for signal synchronization for this block
        return

    block_id = get_flat_bid() if block_id is None else block_id
    block_offset = block_id * world_size
    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))

    # get the signal ptrs array for all the remote ranks, with shape [world_size, num_blocks, world_size]
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )

    # get the signal ptrs array for the current rank, with shape [num_blocks, world_size]
    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )

    # get the sending signal ptrs of all the remote ranks for this rank w.r.t. this block
    # i.e. remote_signal_pad_addrs[:, block_id, rank]
    send_addrs = remote_signal_pad_addrs + block_offset + rank

    # get the waiting signal ptrs of this rank for all the remote ranks w.r.t. this block
    # i.e. local_signal_pad_addr[block_id, :]
    wait_addrs = local_signal_pad_addr + block_offset + remote_ranks

    # put the signal into other remote ranks for this rank w.r.t. this block
    # with the given semantics, choosing from "relaxed" or "acq_rel"
    # and if sem is "acq_rel", the signal will be put in "release" order
    put_signal(send_addrs, sem)

    # wait the signal of this rank from other remote ranks w.r.t. this block
    # with the given semantics, choosing from "relaxed" or "acq_rel"
    # and if sem is "acq_rel", the signal will be put in "acquire" order
    wait_signal(wait_addrs, sem)


@triton.jit
def sync_threads():
    """Sync the threads in the block,
    simulating `__syncthreads()` in CUDA.
    """

    tl.inline_asm_elementwise(
        # use bar.sync with the 0th flag to sync the entire block
        "bar.sync 0;",  # asm
        "=r",  # output constraints, which means the output will be stored in a regular register
        [],  # input args list
        dtype=tl.int32,  # the dtype of each output
        is_pure=False,  # not a "pure" op, i.e. it has side effects
        pack=1,
    )
