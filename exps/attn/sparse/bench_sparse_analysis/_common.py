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

"""Shared constants, GPU selection, JSON persistence, timing, and tensor helpers."""

import datetime
import gc
import json
import os

# ── Global config ──────────────────────────────────────────────
NHQ, NHK, HD, KBS = 128, 1, 128, 128
S_FULL = 32768
TOPK_VALS = [32768, 16384, 8192, 4096, 2048]
WARMUP, ITERS = 8, 20

# ── Video-production scenarios (shared by Phase 6 & 8) ────────
# qseqlen = kvseqlen/64, topk = kvseqlen/8
VIDEO_SCENARIOS = [
    # (kvseqlen, qseqlen, topk)
    (32768, 512, 4096),
    (65536, 1024, 8192),
    (131072, 2048, 16384),
    (262144, 4096, 32768),
    (524288, 8192, 65536),
]

# ── Plot style (consistent across all phases) ─────────────────
# Our kernels: warm/saturated colors
COLOR_INDEX_SPARSE = (0.77, 0.34, 0.49)
COLOR_BLOCK_SPARSE = (0.29, 0.57, 0.60)
# External baselines: gray/dark
COLOR_FLEXATTN = (0.45, 0.45, 0.45)
COLOR_TRITON = (0.65, 0.65, 0.65)
COLOR_TRITON_LOOPQ = (0.80, 0.80, 0.80)

PLOT_BAR_WIDTH_RATIO = 0.8
PLOT_BAR_ALPHA = 0.85
PLOT_VALUE_FONTSIZE = 6
PLOT_YLIM = (0, 750)
PLOT_DPI_SAVE = 180
PLOT_SUBPLOT_WIDTH = 8
PLOT_SUBPLOT_HEIGHT = 7

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_OUT = os.path.join(_SCRIPT_DIR, "outs", "sparse_analysis")

PHASES = [
    "0-method-parity",
    "1-topk-sweep",
    "2-kbs-compare",
    "3-l2-inflection",
    "4-loopk-debug",
    "4_1-skip-ablation",
    "4_2-iss-double-buffer",
    "5-scaling",
    "6-video-production",
    "7-outer-store-mode",
    "8-baseline-comparison",
]


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _out_dir(phase):
    return os.path.join(_BASE_OUT, phase)


def _results_path(phase):
    return os.path.join(_out_dir(phase), "results.json")


# ── GPU selection ──────────────────────────────────────────────
def _find_free_gpu():
    import torch

    n = torch.cuda.device_count()
    if n <= 1:
        return 0
    best_idx, best_used = 0, float("inf")
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / (1024**2)
        if used < best_used:
            best_idx, best_used = i, used
    return best_idx


def _check_gpu_isolation(gpu: int) -> bool:
    """Check that the selected GPU has no other compute processes. Returns True if clean."""
    import subprocess
    import warnings

    clean = True
    try:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            physical_gpu = cvd.split(",")[gpu]
        else:
            physical_gpu = str(gpu)

        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
                f"--id={physical_gpu}",
            ],
            text=True,
            timeout=5,
        ).strip()

        our_pid = os.getpid()
        other_procs = []
        for line in procs.splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            pid = int(parts[0].strip())
            if pid != our_pid:
                other_procs.append(line.strip())

        if other_procs:
            warnings.warn(
                f"\n⚠️  GPU {physical_gpu} has other compute processes: "
                f"{other_procs}. Benchmark will be inaccurate!",
                stacklevel=2,
            )
            clean = False

    except Exception:
        pass
    return clean


def _set_gpu():
    import torch

    gpu = _find_free_gpu()
    torch.cuda.set_device(gpu)
    if not _check_gpu_isolation(gpu):
        print(
            f"  [WARN] GPU {gpu} is NOT fully idle. "
            f"Use CUDA_VISIBLE_DEVICES=<free_gpu> for stable results.",
            flush=True,
        )
    return gpu


# ── JSON persistence ──────────────────────────────────────────
def _load_results(phase):
    path = _results_path(phase)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_results(phase, results):
    path = _results_path(phase)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    if os.path.getsize(tmp) > 0:
        os.replace(tmp, path)


def _has_entry(results, key, topk):
    d = results.get(key, {})
    if not d or "topk" not in d:
        return False
    try:
        idx = d["topk"].index(topk)
        return d["tflops"][idx] is not None
    except (ValueError, IndexError):
        return False


def _set_entry(results, key, topk, tflops, ms):
    if key not in results:
        results[key] = {"topk": [], "tflops": [], "ms": []}
    d = results[key]
    if topk in d["topk"]:
        idx = d["topk"].index(topk)
        d["tflops"][idx] = tflops
        d["ms"][idx] = ms
    else:
        d["topk"].append(topk)
        d["tflops"].append(tflops)
        d["ms"].append(ms)


def _parse_rerun(s):
    if not s:
        return None
    entries = set()
    for item in s.split(","):
        parts = item.strip().split("/")
        if len(parts) == 2:
            for tk in TOPK_VALS:
                entries.add((parts[0], parts[1], tk))
        elif len(parts) == 3:
            entries.add((parts[0], parts[1], int(parts[2])))
    return entries or None


# ── Timing infrastructure ─────────────────────────────────────
def _bench_kernel(run_fn, flops, device):
    import torch

    for _ in range(WARMUP):
        run_fn()
    torch.cuda.synchronize(device)
    l2_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        l2_flush.zero_()
        starts[i].record()
        run_fn()
        ends[i].record()
    torch.cuda.synchronize(device)
    del l2_flush
    times = sorted([s.elapsed_time(e) for s, e in zip(starts, ends)])
    ms = times[len(times) // 2]
    return flops / ms * 1e-9, ms


# ── Tensor & index helpers ────────────────────────────────────
def _make_tensors(S, device, dtype, grad=False):
    import torch

    q = torch.randn(S, NHQ, HD, dtype=dtype, device=device)
    k = torch.randn(S, NHK, HD, dtype=dtype, device=device)
    v = torch.randn(S, NHK, HD, dtype=dtype, device=device)
    if grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def _make_tensors_kv_short(S, topk, device, dtype, grad=False):
    import torch

    q = torch.randn(S, NHQ, HD, dtype=dtype, device=device)
    k = torch.randn(topk, NHK, HD, dtype=dtype, device=device)
    v = torch.randn(topk, NHK, HD, dtype=dtype, device=device)
    if grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def _calc_flops(S, topk, is_bwd):
    fwd = 4 * S * topk * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _build_idx_kbs128(S, topk, device):
    import torch

    n_total, n_topk = S // KBS, topk // KBS
    if n_topk >= n_total:
        idx = torch.arange(n_total, dtype=torch.int32, device=device)
        return idx.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()
    gen = torch.Generator().manual_seed(42)
    rand_vals = torch.rand(S, n_total, generator=gen)
    perms = rand_vals.argsort(dim=1)[:, :n_topk].sort(dim=1).values
    return (
        perms.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(dtype=torch.int32, device=device)
        .contiguous()
    )


def _build_idx_kbs128_cpasync(S, topk, device):
    """Build kbs=128 indices expanded to per-token (kbs=1) for CpAsync path."""
    import torch

    block_idx = _build_idx_kbs128(S, topk, device)
    B_q, nhk, n_blocks = block_idx.shape
    expanded = block_idx.unsqueeze(-1) * KBS + torch.arange(
        KBS, device=device, dtype=torch.int32
    )
    return expanded.reshape(B_q, nhk, n_blocks * KBS).contiguous()


def _build_idx_kbs1(S, topk, device):
    import torch

    if topk >= S:
        idx = torch.arange(S, dtype=torch.int32, device=device)
        return idx.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()
    gen = torch.Generator().manual_seed(42)
    rand_vals = torch.rand(S, S, generator=gen)
    idx = rand_vals.argsort(dim=1)[:, :topk].sort(dim=1).values
    return (
        idx.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(dtype=torch.int32, device=device)
        .contiguous()
    )


def _indices_to_ranges(indices, S):
    import torch

    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    ia_3d = indices.permute(1, 0, 2).contiguous()
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        ia_3d, block_m=1, block_n=KBS, num_k_blocks=S // KBS
    )
    atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=indices.device)
    return q_ranges, k_ranges, atm


# ── Common bench wrapper ──────────────────────────────────────
def _bench_ffa(S, topk, pass_type, kw, device):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    is_bwd = pass_type != "fwd"
    if not is_bwd:
        kw.setdefault("disable_fwd_atomic_reduction", True)
    if "q_ranges" in kw or "index_sparse_indices" in kw:
        q, k, v = _make_tensors(S, device, torch.bfloat16, grad=is_bwd)
    else:
        q, k, v = _make_tensors_kv_short(S, topk, device, torch.bfloat16, grad=is_bwd)
    o, *_ = flex_flash_attn_func(q, k, v, **kw)
    flops = _calc_flops(S, topk, is_bwd)

    if not is_bwd:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)  # noqa: F821

    else:
        do = torch.randn_like(o)

        def run_fn():  # noqa: F811
            o.backward(do, retain_graph=True)  # noqa: F821

    tf, ms = _bench_kernel(run_fn, flops, device)
    q = k = v = o = None
    gc.collect()
    torch.cuda.empty_cache()
    return tf, ms


# ── Shared plot helper ────────────────────────────────────────
def plot_grouped_bars(
    ax,
    x,
    methods,
    data_fn,
    *,
    title="",
    xlabel="",
    ylabel="TFLOPS",
    x_labels=None,
    ylim=None,
):
    """Draw grouped bars on *ax* with unified style.

    Parameters
    ----------
    ax : matplotlib Axes
    x : np.ndarray of x-tick positions
    methods : list of (method_id, label, color)
        Ordered left-to-right. Put baselines first (grey), our kernels last.
    data_fn : callable(method_id) -> list[float]
        Returns TFLOPS values for each x position (0 means missing).
    """

    n_m = len(methods)
    bw = PLOT_BAR_WIDTH_RATIO / n_m
    for i, (mid, lbl, col) in enumerate(methods):
        vals = data_fn(mid)
        off = (i - n_m / 2 + 0.5) * bw
        bars = ax.bar(
            x + off,
            vals,
            width=bw,
            label=lbl,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            alpha=PLOT_BAR_ALPHA,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=PLOT_VALUE_FONTSIZE,
                    fontweight="bold",
                )

    ax.set_title(title, fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    if x_labels is not None:
        ax.set_xticklabels(x_labels, fontsize=9)
    ax.tick_params(axis="y", labelsize=11)
    effective_ylim = ylim if ylim is not None else PLOT_YLIM
    ax.set_ylim(*effective_ylim)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
