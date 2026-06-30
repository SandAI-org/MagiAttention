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

# mypy: disable-error-code="misc"


from typing import Callable, Optional, Type

import cutlass
import cutlass.cute as cute


def get_smem_layout_atom(
    dtype: Type[cutlass.Numeric], k_dim: int
) -> cute.ComposedLayout:
    dtype_byte = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)

    # one smem row is at most 32banks x 4B/bank = 128B
    smem_k_block_size = (
        cutlass.const_expr(
            128
            if bytes_per_row % 128 == 0
            else (
                64
                if bytes_per_row % 64 == 0
                else (32 if bytes_per_row % 32 == 0 else 16)
            )
        )
        // dtype_byte
    )

    # B = log2(blk // 8) => TODO(REVIEW): why ?
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )

    # M = S => TODO(REVIEW): why ?
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)

    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),  # SW<B,M,S>
        0,
        cute.make_ordered_layout(  # (8, blk)
            (8 if cutlass.const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
    A_in_regs: cutlass.Constexpr[bool] = False,
    B_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    """Issue an SM80 (Ampere) warp-level tiled MMA `acc += A @ B.T`, with the
    operands' S2R (smem-to-register) copies software-pipelined against the
    MMA along the K (contraction) dimension.

    The K dimension is iterated in `MMA_K = cute.size(tCsA.shape[2])` steps.
    Before the first `cute.gemm`, the k=0 operand tile is loaded from smem to
    rmem; then within the loop, the k+1 tile is prefetched (S2R copied) while
    the k-th `cute.gemm` is issued, so the next operand load overlaps with the
    current MMA. An operand already living in registers can skip its S2R copy
    via `A_in_regs`/`B_in_regs`.

    Args:
        tiled_mma: The tiled MMA describing the warp-level MMA atom and the
            thread/value partitioning of A, B and the accumulator.
        acc: The accumulator fragment in registers (the MMA's C operand). It is
            read-modify-written in place: on return `acc += A @ B`. Must be
            pre-initialized (e.g. zero-filled) by the caller.
        tCrA: A-operand register fragment, partitioned for the MMA as
            `(MMA_ATOM, MMA_M, MMA_K)`. This is the tensor fed to `cute.gemm`.
        tCrB: B-operand register fragment, partitioned for the MMA as
            `(MMA_ATOM, MMA_N, MMA_K)`. This is the tensor fed to `cute.gemm`.
        tCsA: A-operand source view in shared memory, partitioned by
            `smem_thr_copy_A` and indexed per K-step `[None, None, k]` for the
            S2R copy into `tCrA`. Ignored when `A_in_regs=True`.
        tCsB: B-operand source view in shared memory, partitioned by
            `smem_thr_copy_B` and indexed per K-step `[None, None, k]` for the
            S2R copy into `tCrB`. Ignored when `B_in_regs=True`.
        smem_thr_copy_A: The thread-sliced smem->rmem tiled copy used to stage
            A from `tCsA` into `tCrA` (via `retile`). Encodes the S2R copy atom
            (e.g. ldmatrix) and the per-thread layout for A.
        smem_thr_copy_B: The thread-sliced smem->rmem tiled copy used to stage
            B from `tCsB` into `tCrB`. Same role as `smem_thr_copy_A` but for B.
        hook_fn: Optional callback invoked exactly once, right after the k=0
            `cute.gemm` is issued. Used to overlap unrelated work with the MMA
            pipeline, e.g. committing the cp.async group that loads the next
            Q/K/V tile for the following mainloop iteration.
        A_in_regs: If True, A already resides in registers (`tCrA`); skip its
            S2R copy and ignore `tCsA`/`smem_thr_copy_A`. Used for RS-MMA where
            the A operand is produced in registers by a previous MMA.
        B_in_regs: If True, B already resides in registers (`tCrB`); skip its
            S2R copy and ignore `tCsB`/`smem_thr_copy_B`.
        swap_AB: If True, compute `acc += B @ A.T` instead by swapping the A and B
            operands (and their copies / in-regs flags) before recursing. Used
            to map a logically transposed MMA onto the same physical atom (e.g.
            the `*_swapAB` variants of the SdP/dKV/dQ MMAs).

    Returns:
        None. The result is accumulated in place into `acc`.
    """
    if cutlass.const_expr(swap_AB):
        gemm(
            tiled_mma,
            acc,
            tCrB,
            tCrA,
            tCsB,
            tCsA,
            smem_thr_copy_B,
            smem_thr_copy_A,
            hook_fn,
            A_in_regs=B_in_regs,
            B_in_regs=A_in_regs,
            swap_AB=False,
        )
    else:
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
        if cutlass.const_expr(not A_in_regs):
            cute.copy(
                smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0]
            )
        if cutlass.const_expr(not B_in_regs):
            cute.copy(
                smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0]
            )
        for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
            if k < cute.size(tCsA.shape[2]) - 1:
                if cutlass.const_expr(not A_in_regs):
                    cute.copy(
                        smem_thr_copy_A,
                        tCsA[None, None, k + 1],
                        tCrA_copy_view[None, None, k + 1],
                    )
                if cutlass.const_expr(not B_in_regs):
                    cute.copy(
                        smem_thr_copy_B,
                        tCsB[None, None, k + 1],
                        tCrB_copy_view[None, None, k + 1],
                    )
            cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            if cutlass.const_expr(k == 0 and hook_fn is not None):
                hook_fn()


@cute.jit
def gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
) -> None:
    """Issue an SM80 (Ampere) warp-level tiled MMA `acc += A @ B.T`, the
    register-stationary (RS) variant of `gemm`.

    Unlike `gemm`, the A operand is assumed to already live in registers
    (`tCrA`), so only the B operand is software-pipelined from smem to rmem
    along the K (contraction) dimension. The K loop iterates
    `MMA_K = cute.size(tCrA.shape[2])` steps: the k=0 B tile is S2R-copied
    before the loop, then each k+1 B tile is prefetched while the k-th
    `cute.gemm` is issued, overlapping the next B load with the current MMA.

    This is used when A is produced directly in registers by a preceding MMA
    (e.g. P/dS feeding the dV/dK MMA), avoiding a round-trip through smem.

    Args:
        tiled_mma: The tiled MMA describing the warp-level MMA atom and the
            thread/value partitioning of A, B and the accumulator.
        acc: The accumulator fragment in registers (the MMA's C operand). It is
            read-modify-written in place: on return `acc += A @ B.T`. Must be
            pre-initialized (e.g. zero-filled) by the caller.
        tCrA: A-operand register fragment, partitioned for the MMA as
            `(MMA_ATOM, MMA_M, MMA_K)`. Already resident in registers; fed
            directly to `cute.gemm` with no S2R copy.
        tCrB: B-operand register fragment, partitioned for the MMA as
            `(MMA_ATOM, MMA_N, MMA_K)`. Filled per K-step by the S2R copy from
            `tCsB` and then fed to `cute.gemm`.
        tCsB: B-operand source view in shared memory, partitioned by
            `smem_thr_copy_B` and indexed per K-step `[None, None, k]` for the
            S2R copy into `tCrB`.
        smem_thr_copy_B: The thread-sliced smem->rmem tiled copy used to stage
            B from `tCsB` into `tCrB` (via `retile`). Encodes the S2R copy atom
            (e.g. ldmatrix) and the per-thread layout for B.
        hook_fn: Optional callback invoked exactly once, right after the k=0
            `cute.gemm` is issued. Used to overlap unrelated work with the MMA
            pipeline, e.g. committing the cp.async group that loads the next
            tile for the following mainloop iteration.

    Returns:
        None. The result is accumulated in place into `acc`.
    """
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if cutlass.const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(
                smem_thr_copy_B,
                tCsB[None, None, k + 1],
                tCrB_copy_view[None, None, k + 1],
            )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        if cutlass.const_expr(k == 0 and hook_fn is not None):
            hook_fn()
