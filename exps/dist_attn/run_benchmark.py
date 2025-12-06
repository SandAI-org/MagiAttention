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

import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import magi_attention
from exps.attn.baselines.utils import calculate_attn_flops
from exps.dist_attn.baselines.interface import AttnImpl
from exps.dist_attn.baselines.loongtrain import LoongTrain
from exps.dist_attn.baselines.ring_attn import RingAttnAllGather, RingAttnP2P
from exps.dist_attn.baselines.shard import (
    ParallelMode,
    get_loongtrain_pg,
    get_ring_pg,
    get_ulysess_pg,
    get_usp_pg,
    init_distributed,
    set_seed,
)
from exps.dist_attn.baselines.ulysess import Ulysess
from exps.dist_attn.baselines.usp import USP
from exps.dist_attn.baselines.utils_cp import AttnBackend
from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.mask import MaskIterator
from exps.dist_attn.benchmark_conf import (
    ATTN_CONFIG,
    BENCH_CONFIG,
    DATA_CONFIG,
    SAMPLE_CONFIG,
    SEED,
)
from magi_attention.api import calc_attn, compute_pad_size, magi_attn_flex_dispatch
from magi_attention.benchmarking.bench import Benchmark, do_bench, perf_report
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import DistAttnConfig
from magi_attention.meta.solver.dispatch_solver import DispatchConfig
from magi_attention.meta.solver.overlap_solver import OverlapConfig, UniformOverlapAlg

already_known_oom_before_run = False
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
TOTAL_SEQLEN = DATA_CONFIG.seqlen_per_rank * WORLD_SIZE
OUTLIER = int(-1e10)

CP_GROUP_META = {
    AttnImpl.ULYSSES: {
        ParallelMode.ULYSESS: WORLD_SIZE,
    },
    AttnImpl.RING_P2P: {
        ParallelMode.RING: WORLD_SIZE,
    },
    AttnImpl.RING_ALLGATHER: {
        ParallelMode.RING: WORLD_SIZE,
    },
    AttnImpl.USP: {
        # inter-node
        ParallelMode.RING: max(1, WORLD_SIZE // 8),
        # intra-node
        ParallelMode.ULYSESS: min(WORLD_SIZE, 8),
    },
    AttnImpl.LOONGTRAIN: {
        # inter-node
        ParallelMode.RING: max(1, WORLD_SIZE // 8),
        # intra-node
        ParallelMode.ULYSESS: min(WORLD_SIZE, 8),
    },
    AttnImpl.MAGI_ATTENTION: {
        # inter-node
        ParallelMode.INTER_WINDOW: max(1, WORLD_SIZE // 8),
        # intra-node
        ParallelMode.INTRA_WINDOW: min(WORLD_SIZE, 8),
    },
}


CP_GROUP = {  # type: ignore[var-annotated]
    AttnImpl.ULYSSES: {},
    AttnImpl.RING_P2P: {},
    AttnImpl.RING_ALLGATHER: {},
    AttnImpl.USP: {},
    AttnImpl.LOONGTRAIN: {},
    AttnImpl.MAGI_ATTENTION: {},
}


def init_dist_environment(
    attn_impl: AttnImpl,
    world_size: int,
    cp_pg_meta,
):
    global CP_GROUP
    if attn_impl in CP_GROUP:
        if world_size in CP_GROUP[attn_impl].keys():
            return CP_GROUP[attn_impl][world_size]
    rank = int(os.environ.get("RANK", 0))

    # -----    init ring or all-gather   ---- #
    if attn_impl == AttnImpl.RING_ALLGATHER or attn_impl == AttnImpl.RING_P2P:
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_ring_pg(device_shard)

    # -----    init ulysess   ---- #
    elif attn_impl == AttnImpl.ULYSSES:
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_ulysess_pg(device_shard)

    # -----    init usp   ---- #
    elif attn_impl == AttnImpl.USP:
        device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
        cp_group = get_usp_pg(device_shard)

    # -----    init loongtrain   ---- #
    elif attn_impl == AttnImpl.LOONGTRAIN:
        # NOTE: init window num for loongtrain double ring-attention
        ring_size = cp_pg_meta[ParallelMode.RING]
        for i in range(1, ring_size + 1):
            if ring_size % i != 0:
                continue
            # balance double ring with intra-ring size >= inter ring size
            if i > ring_size // i:
                break
            window_num = i
        if window_num >= ring_size:
            window_num = 1
        assert ring_size % window_num == 0
        device_shard = init_distributed(world_size=world_size, pg_meta=None)
        cp_group = get_loongtrain_pg(cp_pg_meta, window_num, rank)
    elif attn_impl == AttnImpl.MAGI_ATTENTION:
        _ = init_distributed(world_size=world_size, pg_meta=None)
        inter_size, intra_size = (
            cp_pg_meta[ParallelMode.INTER_WINDOW],
            cp_pg_meta[ParallelMode.INTRA_WINDOW],
        )
        if (
            magi_attention.comm.is_hierarchical_comm_enable()
            and intra_size == 8
            and (inter_size == 1 or inter_size % 2 == 0)
        ):
            # NOTE: init hierarchical device_mesh for magi
            cp_group = init_device_mesh(
                device_type="cuda",
                mesh_shape=(
                    cp_pg_meta[ParallelMode.INTER_WINDOW],
                    cp_pg_meta[ParallelMode.INTRA_WINDOW],
                ),
                mesh_dim_names=("inter", "intra"),
            )
        else:
            assert (
                not magi_attention.comm.is_hierarchical_comm_enable()
            ), "A 2D cp_mesh must be provided when hierarchical comm is enabled, instead of a single cp_group."
            cp_group = dist.new_group(list(range(world_size)), backend="nccl")
    # avoid repeated init cp group
    CP_GROUP[attn_impl][world_size] = cp_group
    return cp_group


def run_dist_attn(
    total_seqlen: int,
    embed_dim: int,
    q_heads: int,
    kv_heads: int,
    hidden_size: int,
    dtype,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    dropout: float,
    softmax_scale: float,
    deterministic: bool,
    world_size: int,
    attn_mask_type: AttnMaskType,
    attn_impl: AttnImpl,
    attn_backend: AttnBackend,
    cp_group,
    wd: str,
):
    device = torch.cuda.current_device()

    # -----    init attn module   ---- #

    if attn_impl == AttnImpl.RING_ALLGATHER:
        attn = RingAttnAllGather(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.RING_P2P:
        attn = RingAttnP2P(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.ULYSSES:
        attn = Ulysess(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [device]
    elif attn_impl == AttnImpl.USP:
        attn = USP(cp_process_group=cp_group, qkv_format="thd", backend=attn_backend)  # type: ignore[assignment]
        cal_runtime_args = [attn_mask_type, device]
    elif attn_impl == AttnImpl.LOONGTRAIN:
        attn = LoongTrain(  # type: ignore[assignment]
            cp_process_group=cp_group, qkv_format="thd", backend=attn_backend
        )
        cal_runtime_args = [attn_mask_type, device]

    # -----    init test data   ---- #

    x = torch.randn(total_seqlen, embed_dim, dtype=dtype, device=device)
    q_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )
    k_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    v_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    dout_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )

    # -----    dispatch   ---- #

    # NOTE: dispatch only support (t, h, d)
    x = x.view(total_seqlen, 1, embed_dim)
    x_local = attn.dispatch(x, q_ranges, total_seqlen, ["q", "dout"])
    _ = attn.dispatch(x, k_ranges, total_seqlen, ["k", "v"])
    x_local = x_local.view(-1, embed_dim)

    # -----   qkv do proj ----- #

    q_local = q_proj(x_local).view(-1, q_heads, hidden_size)
    k_local = k_proj(x_local).view(-1, kv_heads, hidden_size)
    v_local = v_proj(x_local).view(-1, kv_heads, hidden_size)
    dout_local = dout_proj(x_local).view(-1, q_heads, hidden_size)

    q_local.requires_grad_(True)
    k_local.requires_grad_(True)
    v_local.requires_grad_(True)

    if attn_impl == AttnImpl.ULYSSES:
        assert world_size % kv_heads == 0 or kv_heads % world_size == 0
        H = world_size // kv_heads
        if H > 1:
            k_local = torch.repeat_interleave(k_local, H, dim=1)
            v_local = torch.repeat_interleave(v_local, H, dim=1)

    # -----   pre_compute ---- #

    attn.pre_compute_attn_runtime_meta(*cal_runtime_args)

    # -----   attn func ---- #

    def fn():
        return attn.apply_attn(
            q_local,
            k_local,
            v_local,
            attn_mask_type,
            dropout,
            softmax_scale,
            deterministic,
        )

    if wd == "bwd":
        try:
            out, _ = fn()
        except Exception as e:
            if "CUDA out of memory" not in str(e):
                print(
                    f"Error occured before running {attn_impl} with {attn_mask_type} mask "
                    f"when {total_seqlen=}, {q_heads=} during {wd}: {e=}"
                )
                raise e
            global already_known_oom_before_run
            already_known_oom_before_run = True

        def fn():
            out.backward(dout_local, retain_graph=True)

    elif wd == "1f1b":

        def fn():
            out, _ = attn.apply_attn(
                q_local,
                k_local,
                v_local,
                attn_mask_type,
                dropout,
                softmax_scale,
                deterministic,
            )
            out.backward(dout_local, retain_graph=True)

    return fn


def run_magi_attn(
    total_seqlen: int,
    embed_dim: int,
    q_heads: int,
    kv_heads: int,
    hidden_size: int,
    dtype: torch.dtype,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    world_size: int,
    chunk_size: int,
    attn_mask_type: list[AttnMaskType],
    cp_group_or_mesh,
    wd: str,
):
    device = torch.cuda.current_device()

    # -----    init test data   ---- #

    x = torch.randn(total_seqlen, embed_dim, dtype=dtype, device=device)

    q_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )
    k_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    v_proj = torch.nn.Linear(
        embed_dim, kv_heads * hidden_size, dtype=dtype, device=device
    )
    dout_proj = torch.nn.Linear(
        embed_dim, q_heads * hidden_size, dtype=dtype, device=device
    )

    # -----   init dispatch mata ----- #

    pad_size = compute_pad_size(
        total_seqlen_q=total_seqlen,
        cp_size=world_size,
        chunk_size=chunk_size,
    )

    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(alg=ATTN_CONFIG.dispatch_alg()),  # type: ignore[arg-type]
        overlap_config=OverlapConfig(
            enable=ATTN_CONFIG.enable_overlap,
            mode=ATTN_CONFIG.overlap_mode,
            degree=ATTN_CONFIG.degree,
            min_chunk_size=ATTN_CONFIG.min_chunk_size,
            max_num_chunks=ATTN_CONFIG.max_num_chunks,
            alg=UniformOverlapAlg(
                random_costs=True,
                random_seed=42,
            ),
        ),
    )

    # -----    dispatch   ---- #

    (
        x_local,
        magi_attn_runtime_key,
    ) = magi_attn_flex_dispatch(  # local_x with shape (total_seqlen_q + pad_size) / cp_size, h)
        x,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen,
        total_seqlen_k=total_seqlen,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_group_or_mesh,
        dist_attn_config=dist_attn_config,
        num_heads_q=q_heads,
        num_heads_kv=kv_heads,
    )

    # -----   projection  ----- #

    q_local = q_proj(x_local).view(-1, q_heads, hidden_size)
    k_local = k_proj(x_local).view(-1, kv_heads, hidden_size)
    v_local = v_proj(x_local).view(-1, kv_heads, hidden_size)
    dout_local = dout_proj(x_local).view(-1, q_heads, hidden_size)

    q_local.requires_grad_(True)
    k_local.requires_grad_(True)
    v_local.requires_grad_(True)

    # -----   attn func ---- #

    def fn():
        return calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)

    if wd == "bwd":
        try:
            out, _ = fn()
        except Exception as e:
            if "CUDA out of memory" not in str(e):
                print(
                    f"Error occured before running magi-attention with {attn_mask_type} mask "
                    f"when {total_seqlen=}, {q_heads=} during {wd}: {e=}"
                )
                raise e
            global already_known_oom_before_run
            already_known_oom_before_run = True

        def fn():
            out.backward(dout_local, retain_graph=True)

    elif wd == "1f1b":

        def fn():
            out, _ = calc_attn(q_local, k_local, v_local, magi_attn_runtime_key)
            out.backward(dout_local, retain_graph=True)

    return fn


x_vals = [dist_attn for dist_attn in BENCH_CONFIG.dist_attn_impl]
x_names = ["attn_impl" for _ in x_vals]
dist_attn_benchmarks = [
    Benchmark(
        x_names=x_names,
        x_vals=x_vals,
        x_log=False,
        line_arg="seqlen",
        line_vals=[TOTAL_SEQLEN],
        line_names=[str(TOTAL_SEQLEN)],
        styles=[  # Line styles.
            ("green", "--"),
            ("orange", "--"),
            ("steelblue", "--"),
            ("red", "-"),
        ],
        ylabel={  # Label name for the y-axis.
            "flops": "Throughout (TFLOPs/s)",
            "mem": "Peak Memory (GB)",
        },
        # Name for the plot. Used also as a file name for saving the plot.
        plot_name=f"Total_seqlen_of_{TOTAL_SEQLEN}_{wd}_with_{mask.value}",
        args={  # Values for function arguments not in `x_names` and `y_name`.
            "mask_nums": SAMPLE_CONFIG.pack_num,
            "mask_type": mask,
            "seed": SEED,
            "wd": wd,
        },
    )
    for mask in BENCH_CONFIG.mask_pattern
    for wd in BENCH_CONFIG.workload
]


def extend_bench_results(
    perf_dict_total: Dict[str, Any],
    result_info: Dict[str, Any],
    return_mode: str,
    quantiles: List[float] | None = None,
    return_flops: bool = True,
    return_mem: bool = True,
):
    if quantiles is not None:
        for i in range(len(quantiles)):
            if return_flops:
                result_info["tflops-" + str(quantiles[i])] = perf_dict_total["flops"][i]
            if return_mem:
                result_info["mem-" + str(quantiles[i])] = perf_dict_total["mem"][i]
    else:
        if return_flops:
            result_info["tflops-" + return_mode] = perf_dict_total["flops"][0]
        if return_mem:
            result_info["mem-" + return_mode] = perf_dict_total["mem"][0]
    return result_info


@perf_report(dist_attn_benchmarks)
def run_benchmark(
    mask_nums: int,
    mask_type: FlashMaskType,
    seed: int = 42,
    seqlen: int = 0,
    attn_impl: AttnImpl = AttnImpl.RING_P2P,
    wd: str = "fwd",
):
    set_seed(seed)

    # -----    init mask iterator with dataset sampler   ---- #
    mask_iterator = MaskIterator(
        num_iterations=mask_nums,
        mask_type=mask_type,
        total_seqlen=TOTAL_SEQLEN,
        data_path=SAMPLE_CONFIG.dataset_path,
        chunk_ratio=SAMPLE_CONFIG.chunk_ratio,
        is_binned=SAMPLE_CONFIG.is_binned,
        to_attn_ranges=SAMPLE_CONFIG.to_attn_ranges,
        seed=seed,
    )

    # -----    init dist environment   ---- #
    cp_pg_meta = CP_GROUP_META[attn_impl]
    cp_group = init_dist_environment(
        attn_impl=attn_impl,
        world_size=WORLD_SIZE,
        cp_pg_meta=cp_pg_meta,
    )

    output_n = len(BENCH_CONFIG.quantiles) if BENCH_CONFIG.quantiles else 1
    perf_dict_total = {
        "flops": [0] * output_n,
        "mem": [0] * output_n,
    }

    # generate mask_nums pack
    for q_ranges, k_ranges, attn_mask_type, _ in mask_iterator:
        global already_known_oom_before_run
        already_known_oom_before_run = False

        # q_ranges = AttnRanges.from_ranges([[i * 8192, (i + 1) * 8192] for i in range(8)])
        # k_ranges = AttnRanges.from_ranges([[i * 8192, (i + 1) * 8192] for i in range(8)])
        # attn_mask_type = [AttnMaskType.FULL] * 8
        # print(q_ranges)
        # print(k_ranges)
        # print(attn_mask_type)

        if attn_impl != AttnImpl.MAGI_ATTENTION:
            fn = run_dist_attn(
                total_seqlen=seqlen,
                embed_dim=DATA_CONFIG.embed_dim,
                q_heads=DATA_CONFIG.heads_q,
                kv_heads=DATA_CONFIG.heads_kv,
                hidden_size=DATA_CONFIG.hidden_size,
                dtype=DATA_CONFIG.dtype,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                dropout=ATTN_CONFIG.dropout,
                softmax_scale=ATTN_CONFIG.softmax_scale,  # type: ignore
                deterministic=ATTN_CONFIG.deterministic,
                world_size=WORLD_SIZE,
                attn_mask_type=attn_mask_type[0],
                attn_impl=attn_impl,
                attn_backend=ATTN_CONFIG.attn_backend,
                cp_group=cp_group,
                wd=wd,
            )
        else:
            assert attn_impl is AttnImpl.MAGI_ATTENTION
            fn = run_magi_attn(
                total_seqlen=TOTAL_SEQLEN,
                embed_dim=DATA_CONFIG.embed_dim,
                q_heads=DATA_CONFIG.heads_q,
                kv_heads=DATA_CONFIG.heads_kv,
                hidden_size=DATA_CONFIG.hidden_size,
                dtype=DATA_CONFIG.dtype,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                world_size=WORLD_SIZE,
                chunk_size=ATTN_CONFIG.chunk_size,
                attn_mask_type=attn_mask_type,
                cp_group_or_mesh=cp_group,
                wd=wd,
            )

        if already_known_oom_before_run:
            perf_dict = {
                "flops": [OUTLIER] * output_n,
                "mem": [OUTLIER] * output_n,
            }
            perf_dict_total = {
                "flops": [OUTLIER] * output_n,
                "mem": [OUTLIER] * output_n,
            }
            break

        attn_flops_dict = calculate_attn_flops(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=seqlen,
            num_heads_q=DATA_CONFIG.heads_q,
            head_dim=DATA_CONFIG.hidden_size,
        )
        attn_flops = attn_flops_dict[wd]

        try:
            perf_dict = do_bench(
                fn,
                quantiles=BENCH_CONFIG.quantiles,
                mem_record_mode="peak",
                return_mode=BENCH_CONFIG.bench_mode,
                return_flops=BENCH_CONFIG.bench_flops,
                return_mem=BENCH_CONFIG.bench_mem,
                warmup=BENCH_CONFIG.warmup,
                rep=BENCH_CONFIG.iteration,
            )

            # post process the perf_dict
            def ms_to_tflops(ms: float) -> float:
                return attn_flops / ms * 1e-9

            def mem_to_gb(mem: int) -> float:
                if mem <= 0:
                    return mem
                return mem / (1024**3)

            if BENCH_CONFIG.bench_flops:
                flops = perf_dict["flops"]
                flops = torch.tensor(
                    flops, dtype=torch.float32, device=torch.cuda.current_device()
                )
                dist.all_reduce(flops, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
                perf_dict["flops"] = flops.tolist()  # type: ignore
                perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))  # type: ignore
                perf_dict_total["flops"] = [
                    perf_dict_total["flops"][i] + perf_dict["flops"][i]
                    for i in range(len(perf_dict_total["flops"]))
                ]
            if BENCH_CONFIG.bench_mem:
                mem = perf_dict["mem"]
                mem = torch.tensor(
                    mem, dtype=torch.float32, device=torch.cuda.current_device()
                )
                dist.all_reduce(mem, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
                perf_dict["mem"] = [m // WORLD_SIZE for m in mem.tolist()]  # type: ignore
                perf_dict["mem"] = list(map(mem_to_gb, perf_dict["mem"]))  # type: ignore
                perf_dict_total["mem"] = [
                    perf_dict_total["mem"][i] + perf_dict["mem"][i]
                    for i in range(len(perf_dict_total["mem"]))
                ]

        except Exception as e:
            if "CUDA out of memory" not in str(e):
                raise e
            # negative indicates oom
            perf_dict = {
                "flops": [OUTLIER] * output_n,
                "mem": [OUTLIER] * output_n,
            }
            perf_dict_total = {
                "flops": [OUTLIER] * output_n,
                "mem": [OUTLIER] * output_n,
            }
            break

    # avg results
    perf_dict_total["flops"] = [
        metric / mask_nums for metric in perf_dict_total["flops"]  # type: ignore
    ]
    perf_dict_total["mem"] = [metric / mask_nums for metric in perf_dict_total["mem"]]  # type: ignore
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        result_info = {
            "baseline": attn_impl.value,
            "masktype": mask_type.value,
            "world_size": WORLD_SIZE,
            "ulysses": cp_pg_meta.get(ParallelMode.ULYSESS, -1),
            "ring": cp_pg_meta.get(ParallelMode.RING, -1),
        }
        result_info = extend_bench_results(
            perf_dict_total,
            result_info,
            BENCH_CONFIG.bench_mode,
            BENCH_CONFIG.quantiles,
            BENCH_CONFIG.bench_flops,
            BENCH_CONFIG.bench_mem,
        )

        df_new = pd.DataFrame([result_info])
        output_file = os.path.join(
            BENCH_CONFIG.output_path,
            "output-"
            + str(WORLD_SIZE)
            + "-"
            + str(mask_type.value)
            + "-"
            + wd
            + ".csv",
        )

        if not os.path.exists(output_file):
            df_new.to_csv(output_file, index=False, header=True)
        else:
            df_new.to_csv(output_file, mode="a", index=False, header=False)

    return perf_dict_total


if __name__ == "__main__":
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")

    os.makedirs(BENCH_CONFIG.output_path, exist_ok=True)
    run_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=BENCH_CONFIG.output_path
    )

    # destroy cp comm group
    for cp_groups in CP_GROUP.values():
        for cp_group in cp_groups.values():
            if isinstance(cp_group, dist.ProcessGroup):
                try:
                    dist.destroy_process_group(cp_group)
                except Exception:
                    pass
            elif isinstance(cp_group, dict):
                for pg in cp_group.values():
                    if isinstance(pg, dist.ProcessGroup):
                        try:
                            dist.destroy_process_group(pg)
                        except Exception:
                            pass
