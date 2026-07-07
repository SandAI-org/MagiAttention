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

from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist

from .precision import assert_close
from .utils import poll_cuda_event

# ---------------------------------------------------------------------------
# Compute / NCCL overlap safety harness
# ---------------------------------------------------------------------------


class NcclOverlayState:
    """NCCL all_reduce traffic on a side stream for compute/comm overlap tests."""

    def __init__(self, *, device: torch.device, nccl_mb: int) -> None:
        self.comm_stream = torch.cuda.Stream(device=device)
        numel = (nccl_mb * 1024 * 1024) // 4
        self.buf = torch.ones(numel, device=device, dtype=torch.float32)

    def launch(self, *, repeats: int) -> None:
        with torch.cuda.stream(self.comm_stream):
            torch.cuda.nvtx.range_push("nccl_overlay")
            for _ in range(repeats):
                dist.all_reduce(self.buf, op=dist.ReduceOp.SUM, async_op=False)
            torch.cuda.nvtx.range_pop()

    def synchronize(self) -> None:
        self.comm_stream.synchronize()


def _normalize_fn_outputs(outputs: Any) -> tuple[torch.Tensor, ...]:
    """Coerce the return value of a user-supplied fn to a flat tuple of tensors.
    Non-tensor elements are silently dropped."""
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    return tuple(v for v in outputs if isinstance(v, torch.Tensor))


def _run_fn_with_optional_nccl(
    fn: Callable[[], Any],
    *,
    nccl: NcclOverlayState | None,
    nccl_repeats: int,
) -> tuple[torch.Tensor, ...]:
    """Run *fn* on the default CUDA stream, optionally overlapping with NCCL
    all-reduce traffic launched on a side stream."""
    if nccl is not None:
        dist.barrier()
        nccl.launch(repeats=nccl_repeats)

    torch.cuda.nvtx.range_push("overlap_safe_compute")
    outputs = _normalize_fn_outputs(fn())
    torch.cuda.nvtx.range_pop()

    torch.cuda.current_stream().synchronize()
    if nccl is not None:
        nccl.synchronize()
    return outputs


def assert_overlap_safe(
    fn: Callable[[], Any],
    *,
    device: torch.device = torch.device("cuda"),
    nccl_mb: int = 512,
    nccl_repeats: int = 32,
    overlap_iters: int = 50,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mismatch_threshold: float = 0.0,
    output_names: Sequence[str] | None = None,
    test_case: str = "",
    print_rank: int = 0,
    log_interval: int = 10,
    hang_timeout: float = 30.0,
) -> None:
    """Assert that *fn* is safe to run concurrently with NCCL all-reduce ops.

    Runs three phases:

    1. Baseline - execute ``fn()`` once *without* any NCCL traffic and
       clone the tensor outputs.
    2. Correctness - execute ``fn()`` once *with* NCCL all-reduce traffic
       on a side stream and compare each tensor output against the baseline via
       :func:`assert_close`.  Non-tensor return values are silently ignored.
       ``atol``, ``rtol`` and ``mismatch_threshold`` allow for acceptable
       non-determinism.
    3. Hang stress - execute ``fn()`` with NCCL traffic for
       ``overlap_iters`` more iterations, inserting a
       :func:`torch.distributed.barrier` after every iteration so that a hang
       on any rank surfaces as a timeout rather than silently succeeding.

    Args:
        fn (Callable[[], Any]): Zero-argument callable that runs the kernel
            under test. It may return a single :class:`torch.Tensor`,
            a sequence/tuple of values (non-tensor elements are ignored), or ``None``.
            Bind inputs via a closure or :func:`functools.partial`.
        device (torch.device): CUDA device for this rank.  Used to allocate
            the NCCL buffer and side communication stream inside :class:`NcclOverlayState`.
        nccl_mb (int, optional): Size in MiB of the all-reduce buffer on the
            communication side stream.  Defaults to ``512``.
        nccl_repeats (int, optional): Number of consecutive all-reduce calls
            launched per invocation of ``fn``.  Defaults to ``32``.
        overlap_iters (int, optional): Number of additional overlap iterations
            run in the hang stress phase. Defaults to ``50``.
        atol (float, optional): Absolute tolerance for output comparison.
            Defaults to ``1e-5``.
        rtol (float, optional): Relative tolerance for output comparison.
            Defaults to ``1e-5``.
        mismatch_threshold (float, optional): Fraction of elements (in
            ``[0, 1]``) allowed to exceed ``atol``/``rtol``.  Defaults to
            ``0.0``.
        output_names (Sequence[str] | None, optional): Names for each output
            tensor, used in log messages and assertion failure messages.
            Defaults to ``"out_0"``, ``"out_1"``, …
        test_case (str, optional): Human-readable label included in log
            messages and assertion errors.  Defaults to ``""``.
        print_rank (int, optional): Rank that prints progress messages.  Pass
            ``-1`` to print from all ranks.  Defaults to ``0``.
        log_interval (int, optional): Print a progress message every this many
            stress iterations.  Defaults to ``10``.
        hang_timeout (float, optional): Per-rank CUDA event poll timeout in
            seconds.  If the GPU work of a single iteration does not complete
            within this window, a :class:`RuntimeError` is raised immediately
            rather than waiting for the distributed process-group timeout.
            Defaults to ``30.0``.

    Examples:
        >>> import torch
        >>> from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
        >>> from magi_attention.testing import DistTestBase, with_comms
        >>> from magi_attention.testing.template import assert_overlap_safe
        >>>
        >>> class TestMyKernelOverlap(DistTestBase):
        ...     @skip_if_lt_x_gpu(2)
        ...     @with_comms
        ...     def test_my_kernel_with_nccl_overlap(self) -> None:
        ...         device = torch.device("cuda", self.rank)
        ...         x = torch.randn(4096, 512, device=device, dtype=torch.bfloat16)
        ...
        ...         def run():
        ...             return my_kernel(x)  # single Tensor or tuple
        ...
        ...         assert_overlap_safe(
        ...             run,
        ...             device=device,
        ...             nccl_mb=512,
        ...             nccl_repeats=32,
        ...             overlap_iters=50,
        ...             atol=1e-2,
        ...             rtol=1e-2,
        ...             mismatch_threshold=2e-2,
        ...             output_names=["out"],
        ...             test_case=f"my_kernel rank={self.rank}",
        ...             print_rank=0,
        ...         )
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_print_rank = print_rank == -1 or rank == print_rank

    # ── Phase 1: baseline (no NCCL) ──────────────────────────────────────────
    baseline_outputs = tuple(
        t.detach().clone()
        for t in _run_fn_with_optional_nccl(fn, nccl=None, nccl_repeats=nccl_repeats)
    )
    n_outputs = len(baseline_outputs)
    names: list[str] = (
        list(output_names)
        if output_names is not None
        else [f"out_{i}" for i in range(n_outputs)]
    )

    # ── Phase 2: one overlap run + correctness check ──────────────────────────
    nccl = NcclOverlayState(device=device, nccl_mb=nccl_mb)
    overlap_outputs = _run_fn_with_optional_nccl(
        fn, nccl=nccl, nccl_repeats=nccl_repeats
    )

    for name, overlap_out, baseline_out in zip(
        names, overlap_outputs, baseline_outputs
    ):
        assert_close(
            overlap_out,
            baseline_out.to(overlap_out.dtype),
            atol=atol,
            rtol=rtol,
            mismatch_threshold=mismatch_threshold,
            test_case=f"{test_case} [{name}] overlap-vs-baseline",
            print_rank=print_rank,
        )

    # ── Phase 3: hang stress test ─────────────────────────────────────────────
    if is_print_rank:
        print(
            f"{test_case}: starting hang stress " f"({overlap_iters} iters)",
            flush=True,
        )

    for iter_idx in range(overlap_iters):
        _run_fn_with_optional_nccl(fn, nccl=nccl, nccl_repeats=nccl_repeats)
        poll_cuda_event(
            timeout=hang_timeout,
            error_msg=(
                f"{test_case}: hang detected at stress iter "
                f"{iter_idx + 1}/{overlap_iters} (timeout={hang_timeout}s)"
            ),
        )
        if is_print_rank and iter_idx % log_interval == 0:
            print(
                f"{test_case}: stress " f"{iter_idx + 1}/{overlap_iters}",
                flush=True,
            )

    if is_print_rank:
        print(
            f"{test_case}: hang stress finished " f"{overlap_iters} iterations",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Determinism harness
# ---------------------------------------------------------------------------


def assert_deterministic(
    fn: Callable[[], Any],
    *,
    repeats: int = 200,
    output_names: Sequence[str] | None = None,
    test_case: str = "",
    print_rank: int = -1,
    log_interval: int = 50,
) -> None:
    """Assert that *fn* produces bitwise-identical tensor outputs on every call.

    Executes ``fn()`` once to capture a reference, then calls it ``repeats``
    more times and checks that every output tensor is identical to the reference
    via :func:`torch.equal`.

    Args:
        fn (Callable[[], Any]): Zero-argument callable whose outputs are
            checked for determinism.  Non-tensor return values are silently
            dropped.  Bind inputs via a closure or :func:`functools.partial`.
        repeats (int, optional): Number of additional calls after the
            reference run.  Defaults to ``200``.
        output_names (Sequence[str] | None, optional): Names for each output
            tensor, used in assertion failure messages.  Defaults to
            ``"out_0"``, ``"out_1"``, …
        test_case (str, optional): Human-readable label included in assertion
            error messages.  Defaults to ``""``.
        print_rank (int, optional): Rank that prints progress messages.  Pass
            ``-1`` to print from all ranks.  Defaults to ``-1``.
        log_interval (int, optional): Print a progress message every this many
            repeats.  Defaults to ``50``.

    Examples:
        >>> import torch
        >>> from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
        >>> from magi_attention.testing.template import assert_deterministic
        >>>
        >>> class TestMyKernelDeterminism(DistTestBase):
        ...     @with_run_in_mp
        ...     def test_my_kernel_deterministic(self) -> None:
        ...         x = torch.randn(1024, 512, device="cuda", dtype=torch.float32)
        ...
        ...         def run():
        ...             return my_kernel(x)  # single Tensor or tuple
        ...
        ...         assert_deterministic(
        ...             run,
        ...             repeats=200,
        ...             output_names=["out"],
        ...             test_case=f"my_kernel rank={self.rank}",
        ...             print_rank=-1,
        ...         )
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_print_rank = print_rank == -1 or rank == print_rank

    reference = tuple(t.detach().clone() for t in _normalize_fn_outputs(fn()))
    names: list[str] = (
        list(output_names)
        if output_names is not None
        else [f"out_{i}" for i in range(len(reference))]
    )

    if is_print_rank:
        print(
            f"{test_case}: starting determinism check ({repeats} repeats)", flush=True
        )

    for run_idx in range(1, repeats + 1):
        outputs = _normalize_fn_outputs(fn())
        for name, out, ref in zip(names, outputs, reference):
            assert torch.equal(
                out, ref
            ), f"{test_case}: run={run_idx} non-deterministic {name}"
        if is_print_rank and run_idx % log_interval == 0:
            print(f"{test_case}: determinism {run_idx}/{repeats}", flush=True)

    if is_print_rank:
        print(f"{test_case}: determinism check finished {repeats} repeats", flush=True)
