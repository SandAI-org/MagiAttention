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

"""Analytical GPU-memory ledger for the FFA (flexible flash attention) kernels.

This module answers, without touching a GPU: "for a given (direction, seqlen,
heads, head_dim, dtype) configuration, how many bytes does one FFA fwd/bwd
call need at peak, item by item?" — so OOM limits like the canonical
524K-token bwd wall can be derived (and re-derived after every allocator
change) from a single formula instead of bisection runs.

The model mirrors the allocation sites in
``magi_attention/functional/flex_flash_attn.py``:

forward (training, atomic reduction on)::

    resident inputs : q, k, v                                   (io dtype)
    + out accumulator (fp32)        [_flex_flash_attn_forward]
    + lse (fp32 [sq, nhq])
    + out cast copy (io dtype)      [fwd_cast; coexists with fp32 out]

backward (entered with q, k, v, out, lse, dout resident)::

    + dq accumulator (fp32)         [_flex_flash_attn_backward]
    + dk/dv accumulators (fp32)
    + dPsum + lse_log2 (fp32 [4, sq_rounded, nhq] each, kernel-internal)
    + dq/dk/dv cast copies (io)     [bwd_cast; coexist with fp32 grads]

Two planned peak-reduction knobs are modeled ahead of implementation so their
projected ceilings can be compared (see ``.tmp/058-fwd-tokenidx/NOTES.md`` P2):

- ``bwd_free_saved_out``: dPsum is precomputed once, then the saved ``out``
  storage is released before the dq accumulator is allocated.
- ``bwd_num_qhead_chunks``: bwd runs per q-head chunk; the fp32 dq accumulator
  and the sliced q/out/dout copies shrink by the chunk count, with each chunk
  cast to the io dtype immediately.
"""

from dataclasses import dataclass, field

__all__ = [
    "FFAMemBudgetConfig",
    "FFAMemBudget",
    "ffa_memory_budget",
    "ffa_max_total_seqlen",
]

# dPsum / LSE_log2 are padded to [4, round_up(total_q, 128), nhq] fp32 buffers
# (see `total_q_rounded` in flex_flash_bwd.hpp).
_DPSUM_PAD_FACTOR = 4
_DPSUM_ROW_ROUNDING = 128
_FP32_BYTES = 4


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class FFAMemBudgetConfig:
    """Shape/dtype configuration for one FFA fwd or bwd call.

    The canonical sparse benchmark config (MQA 128/1, hd=128, bf16) is the
    default everywhere except the seqlens, which are mandatory.
    """

    direction: str  # "fwd" | "bwd"
    total_seqlen_q: int
    total_seqlen_k: int
    num_heads_q: int = 128
    num_heads_k: int = 1
    head_dim: int = 128
    io_dtype_bytes: int = 2  # bf16/fp16 inputs and returned grads
    accum_dtype_bytes: int = 4  # fp32 atomic-reduction accumulators

    # -- fwd variant knobs --
    # False (default) = atomic reduction on: out is allocated in fp32 and cast
    # back to the io dtype afterwards.
    disable_fwd_atomic_reduction: bool = False

    # -- bwd variant knobs (planned peak reductions, see module docstring) --
    bwd_free_saved_out: bool = False
    bwd_num_qhead_chunks: int = 1

    def __post_init__(self) -> None:
        assert self.direction in ("fwd", "bwd"), self.direction
        assert self.total_seqlen_q > 0 and self.total_seqlen_k > 0
        assert self.num_heads_q % self.num_heads_k == 0
        assert self.bwd_num_qhead_chunks >= 1
        assert self.num_heads_q % self.bwd_num_qhead_chunks == 0


@dataclass(frozen=True)
class FFAMemBudget:
    """Itemized byte ledger plus the modeled allocator peak."""

    config: FFAMemBudgetConfig
    # name -> bytes for every modeled allocation, in allocation order;
    # transient items (freed before peak or allocated after others are freed)
    # are still listed so the ledger is auditable against memory_history.
    items: dict[str, int] = field(default_factory=dict)
    peak_bytes: int = 0

    def format(self) -> str:
        gib = 1024**3
        lines = [f"[ffa_memory_budget] {self.config}"]
        lines += [
            f"  {name:<24} {nbytes / gib:8.3f} GiB"
            for name, nbytes in self.items.items()
        ]
        lines.append(f"  {'PEAK':<24} {self.peak_bytes / gib:8.3f} GiB")
        return "\n".join(lines)


def ffa_memory_budget(config: FFAMemBudgetConfig) -> FFAMemBudget:
    """Compute the itemized memory ledger and peak for one FFA call.

    The peak models the worst simultaneous-resident set along the call's
    allocation timeline (inputs included, allocator fragmentation excluded —
    real usage is a few percent higher).
    """
    sq, sk = config.total_seqlen_q, config.total_seqlen_k
    nhq, nhk, hd = config.num_heads_q, config.num_heads_k, config.head_dim
    io, acc = config.io_dtype_bytes, config.accum_dtype_bytes

    q_bytes = sq * nhq * hd * io
    kv_bytes = sk * nhk * hd * io  # per tensor (k or v)
    lse_bytes = sq * nhq * _FP32_BYTES

    items: dict[str, int] = {}

    if config.direction == "fwd":
        items["q"] = q_bytes
        items["k"] = kv_bytes
        items["v"] = kv_bytes
        out_acc_bytes = (
            q_bytes if config.disable_fwd_atomic_reduction else sq * nhq * hd * acc
        )
        items["out_accum"] = out_acc_bytes
        items["lse"] = lse_bytes
        if not config.disable_fwd_atomic_reduction:
            # fwd_cast: io-dtype copy coexists with the fp32 accumulator
            items["out_cast"] = q_bytes
        peak = sum(items.values())
        return FFAMemBudget(config=config, items=items, peak_bytes=peak)

    # ---- bwd ----
    chunks = config.bwd_num_qhead_chunks

    items["q"] = q_bytes
    items["k"] = kv_bytes
    items["v"] = kv_bytes
    items["out_saved"] = q_bytes
    items["lse"] = lse_bytes
    items["dout"] = q_bytes
    items["dk_accum"] = sk * nhk * hd * acc
    items["dv_accum"] = sk * nhk * hd * acc

    # dPsum / lse_log2 live only inside the extension call (allocated per
    # call, freed at op return), so they overlap the kernel phase but not
    # the cast phase. With chunking they shrink with the per-call head count.
    dpsum_call_bytes = (
        _DPSUM_PAD_FACTOR
        * _round_up(sq, _DPSUM_ROW_ROUNDING)
        * (nhq // chunks)
        * _FP32_BYTES
    )
    items["dpsum (per call)"] = dpsum_call_bytes
    items["lse_log2 (per call)"] = dpsum_call_bytes

    resident = sum(items.values()) - 2 * dpsum_call_bytes

    if config.bwd_free_saved_out:
        # out is only needed to compute dPsum; after the (full-head) dPsum
        # precompute its storage is released before any dq allocation.
        resident -= items["out_saved"]
        items["out_saved (freed early)"] = -items["out_saved"]

    dq_acc_full = sq * nhq * hd * acc

    if chunks == 1:
        items["dq_accum"] = dq_acc_full
        # bwd_cast: dq io-dtype copy coexists with the fp32 accumulator
        # (dk/dv casts are strictly smaller and happen after dq fp32 is
        # already freed in program order, so dq dominates the cast phase)
        items["dq_cast"] = q_bytes
        kernel_phase = resident + 2 * dpsum_call_bytes + dq_acc_full
        cast_phase = resident + dq_acc_full + q_bytes
        peak = max(kernel_phase, cast_phase)
        return FFAMemBudget(config=config, items=items, peak_bytes=peak)

    # chunked bwd: full io-dtype dq is allocated upfront, then per chunk we
    # hold contiguous q/out/dout head-slices and a 1/chunks fp32 dq
    # accumulator; the chunk cast writes into a view of the full dq
    # (no extra allocation) before the chunk buffers are freed.
    items["dq_out (full, io dtype)"] = q_bytes
    chunk_slice_srcs = 2 if config.bwd_free_saved_out else 3  # q, dout[, out]
    chunk_bytes = chunk_slice_srcs * (q_bytes // chunks) + dq_acc_full // chunks
    items[f"per-chunk buffers (x1/{chunks})"] = chunk_bytes
    peak = resident + q_bytes + chunk_bytes + 2 * dpsum_call_bytes
    return FFAMemBudget(config=config, items=items, peak_bytes=peak)


def ffa_max_total_seqlen(
    budget_bytes: int,
    config: FFAMemBudgetConfig,
    seqlen_step: int = 1024,
) -> int:
    """Largest total_seqlen (q == k, scaled together) whose peak fits budget.

    Bisects on the seqlen (peak is monotonic in S); returns a multiple of
    ``seqlen_step``. The seqlens inside ``config`` are ignored.
    """

    def peak_at(s: int) -> int:
        cfg = FFAMemBudgetConfig(
            direction=config.direction,
            total_seqlen_q=s,
            total_seqlen_k=s,
            num_heads_q=config.num_heads_q,
            num_heads_k=config.num_heads_k,
            head_dim=config.head_dim,
            io_dtype_bytes=config.io_dtype_bytes,
            accum_dtype_bytes=config.accum_dtype_bytes,
            disable_fwd_atomic_reduction=config.disable_fwd_atomic_reduction,
            bwd_free_saved_out=config.bwd_free_saved_out,
            bwd_num_qhead_chunks=config.bwd_num_qhead_chunks,
        )
        return ffa_memory_budget(cfg).peak_bytes

    lo, hi = seqlen_step, seqlen_step
    while peak_at(hi) <= budget_bytes:
        hi *= 2
    while hi - lo > seqlen_step:
        mid = (lo + hi) // 2 // seqlen_step * seqlen_step
        if peak_at(mid) <= budget_bytes:
            lo = mid
        else:
            hi = mid
    return lo
