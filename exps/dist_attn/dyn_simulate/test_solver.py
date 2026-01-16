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

import argparse
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, List

import torch.distributed as dist
from pydantic import TypeAdapter

# Add project root to PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import magi_attention  # noqa: E402
from exps.dist_attn.benchmark.enums import FlashMaskType, MetricsType  # noqa: E402
from exps.dist_attn.benchmark.mask import MaskIterator  # noqa: E402
from exps.dist_attn.benchmark.metric import (  # noqa: E402
    MetricData,
    MetricDataCalculator,
)
from magi_attention.common.enum import AttnRole, AttnType  # noqa: E402

# from magi_attention.meta.algorithms import NCQDynamicAttnAlgorithm
from magi_attention.meta.algorithms import (  # noqa: E402
    BinaryGreedyParallelDynamicAttnAlgorithm,
)
from magi_attention.meta.collection.dispatch_meta import DispatchMeta  # noqa: E402
from magi_attention.meta.solver.dynamic_attn_solver import (  # noqa: E402
    DynamicAttnSolver,
)

# usage:
# python exps/dist_attn/dyn_simulate/test_solver.py --config exps/dist_attn/benchmark_conf.py --world-size 8 > test.log

# Global configuration variables
ATTN_CONFIG: Any = None
BENCH_CONFIG: Any = None
DATA_CONFIG: Any = None
ENVVAR_CONFIG: Any = None
SAMPLE_CONFIG: Any = None
SEED: Any = None
TOTAL_SEQLEN: Any = None
WORLD_SIZE: int = 8  # Default value, can be overridden by command line argument


# Simulate distributed ProcessGroup
class MockProcessGroup:
    def __init__(self, rank, world_size):
        self._rank = rank
        self._world_size = world_size

    def rank(self):
        return self._rank

    def size(self):
        return self._world_size

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size


# Simulate dist-related global functions
def mock_all_gather_object(object_list, obj, group=None):
    pass


def mock_get_world_size(group=None):
    if group is not None:
        if hasattr(group, "size"):
            return group.size()
        return group.get_world_size()
    return WORLD_SIZE


def mock_get_rank(group=None):
    if group is not None:
        if hasattr(group, "rank"):
            return group.rank()
        return group.get_rank()
    return 0


# Replace dist methods
dist.all_gather_object = mock_all_gather_object
dist.get_world_size = mock_get_world_size
dist.get_rank = mock_get_rank


def load_py_as_dict(config_path: str) -> dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_path = os.path.abspath(config_path)
    module_name = "conf"
    spec = spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config file: {config_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    raw_config = {
        k: v
        for k, v in module.__dict__.items()
        if (not k.startswith("__") and k[0].isupper())
    }
    try:
        return TypeAdapter(dict[str, Any]).validate_python(raw_config)
    except Exception as e:
        raise ValueError(f"Failed to validate config: {str(e)}")


def load_bench_config(config_path):
    global BENCH_CONFIG, ATTN_CONFIG, DATA_CONFIG, ENVVAR_CONFIG, SAMPLE_CONFIG, SEED, TOTAL_SEQLEN
    config_dict = load_py_as_dict(config_path)
    BENCH_CONFIG = config_dict["BENCH_CONFIG"]
    ATTN_CONFIG = config_dict["ATTN_CONFIG"]
    DATA_CONFIG = config_dict["DATA_CONFIG"]
    ENVVAR_CONFIG = config_dict["ENVVAR_CONFIG"]
    SAMPLE_CONFIG = config_dict["SAMPLE_CONFIG"]
    SEED = config_dict["SEED"]
    TOTAL_SEQLEN = DATA_CONFIG.seqlen_per_rank * WORLD_SIZE


def create_sequential_dispatch_meta_list(
    total_seqlen: int,
    cp_size: int,
    chunk_size: int,
    attn_role: AttnRole,
) -> List[DispatchMeta]:
    num_chunks = total_seqlen // chunk_size
    chunks_per_rank = num_chunks // cp_size

    all_partitions = []
    for r in range(cp_size):
        partition = list(range(r * chunks_per_rank, (r + 1) * chunks_per_rank))
        all_partitions.append(partition)

    metas = []
    for r in range(cp_size):
        meta = DispatchMeta(
            attn_role=attn_role,
            attn_type=AttnType.SELF_ATTN,
            total_seqlen=total_seqlen,
            shard_seqlen=total_seqlen // cp_size,
            max_valid_ids=total_seqlen,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            cp_rank=r,
            cp_size=cp_size,
            partitions=all_partitions,
            partitions_perm_idxs=list(range(num_chunks)),
            partitions_unperm_idxs=list(range(num_chunks)),
        )
        metas.append(meta)
    return metas


def simulate_solver_and_measure_cost():
    global WORLD_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--world-size", type=int, default=8, help="World size (number of processes)"
    )
    args = parser.parse_args()

    # Set WORLD_SIZE from command line argument
    WORLD_SIZE = args.world_size

    load_bench_config(args.config)

    print("=" * 40)
    print("MagiAttention Solver Partition Simulation (Config-Driven)")
    print(f"Total Seqlen: {TOTAL_SEQLEN}")
    print(f"World Size:   {WORLD_SIZE}")
    print(f"Chunk Size:   {ATTN_CONFIG.chunk_size}")
    print(f"Heads (Q/KV): {DATA_CONFIG.heads_q}/{DATA_CONFIG.heads_kv}")
    print(f"Head Dim:     {DATA_CONFIG.hidden_size}")
    print("=" * 40)

    # Iterate over all mask patterns in configuration
    for mask_type in BENCH_CONFIG.mask_pattern:
        print(f"\n>>> Testing Mask Pattern: {mask_type.value}")

        # Determine iteration count (reference run_benchmark.py)
        mask_nums = (
            SAMPLE_CONFIG.pack_num
            if mask_type in [FlashMaskType.FULL_DOCUMENT, FlashMaskType.CAUSAL_DOCUMENT]
            else 1
        )

        mask_iterator = MaskIterator(
            num_iterations=mask_nums,
            mask_type=mask_type,
            total_seqlen=TOTAL_SEQLEN,
            data_path=SAMPLE_CONFIG.dataset_path,
            chunk_ratio=SAMPLE_CONFIG.chunk_ratio,
            is_binned=SAMPLE_CONFIG.is_binned,
            to_attn_ranges=SAMPLE_CONFIG.to_attn_ranges,
            seed=SEED,
            drop_thres=SAMPLE_CONFIG.drop_thres,
        )
        # Simulate generating dispatch meta
        metas_q = create_sequential_dispatch_meta_list(
            TOTAL_SEQLEN, WORLD_SIZE, ATTN_CONFIG.chunk_size, AttnRole.QUERY
        )
        metas_k = create_sequential_dispatch_meta_list(
            TOTAL_SEQLEN, WORLD_SIZE, ATTN_CONFIG.chunk_size, AttnRole.KEY
        )

        for mask_idx, (q_ranges, k_ranges, attn_mask_type, _) in enumerate(
            mask_iterator
        ):
            print(f"  Iteration {mask_idx}:")
            comm_meta_list = [None] * WORLD_SIZE
            calc_meta_list = [None] * WORLD_SIZE

            for r in range(WORLD_SIZE):
                mock_group = MockProcessGroup(r, WORLD_SIZE)
                solver = DynamicAttnSolver(
                    # algorithm=NCQDynamicAttnAlgorithm(),
                    algorithm=BinaryGreedyParallelDynamicAttnAlgorithm(),
                    cp_group=mock_group,
                    dispatch_meta_q=metas_q[r],
                    dispatch_meta_k=metas_k[r],
                    num_heads_q=DATA_CONFIG.heads_q,
                    num_heads_kv=DATA_CONFIG.heads_kv,
                )

                solver.solve(
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_mask_type=attn_mask_type,
                    flatten_head_groups=magi_attention.is_flatten_head_groups_enable(),
                )

                comm_meta_list[r] = solver.make_comm_meta()
                calc_meta_list[r] = solver.make_calc_meta()

                if r == 0 and mask_idx == (mask_nums - 1):
                    save_path = os.path.join(
                        PROJECT_ROOT, f"buckets_{mask_type.value}.png"
                    )
                    print(f"    Visualizing result to {save_path}...")
                    solver.output_solve_result(visualize=True, save_path=save_path)

            if magi_attention.is_flatten_head_groups_enable():
                num_heads_group = DATA_CONFIG.heads_kv
                num_heads_q = DATA_CONFIG.heads_q // num_heads_group
                num_heads_kv = 1
            else:
                num_heads_group = 1
                num_heads_q = DATA_CONFIG.heads_q
                num_heads_kv = DATA_CONFIG.heads_kv
            # Calculate communication cost
            metric_data = MetricData(
                comm_meta_list=comm_meta_list,
                calc_meta_list=calc_meta_list,
                q_heads=num_heads_q,
                kv_heads=num_heads_kv,
                head_dim=DATA_CONFIG.hidden_size,
                pass_type=BENCH_CONFIG.workload[0],
                fwd_cast_dtype=DATA_CONFIG.dtype,
                fwd_reduce_dtype=DATA_CONFIG.dtype,
                bwd_cast_dtype=DATA_CONFIG.dtype,
                bwd_reduce_dtype=DATA_CONFIG.dtype,
            )

            comm_set_fwd = MetricDataCalculator.calculate(
                metrics_type=MetricsType.COMM_BYTES,
                metric_data=metric_data,
            )

            # Print statistics (format synchronized from _make_attn_meta.py)
            print(
                f"\n[Communication Cost] Forward Pass (head_dim={DATA_CONFIG.hidden_size}, "
                f"num_heads_q={DATA_CONFIG.heads_q}, num_heads_kv={DATA_CONFIG.heads_kv})"
            )
            print(f"World size: {WORLD_SIZE}")
            print("-" * 80)

            total_send_bytes = 0
            total_recv_bytes = 0
            total_send_to_self_bytes = 0

            debug_print_per_rank = mask_idx >= 20

            for rank_idx, comm_bytes_per_rank in enumerate(
                comm_set_fwd.comm_bytes_list
            ):
                rank_send_total = 0
                rank_recv_total = 0
                rank_send_to_self_total = 0

                if debug_print_per_rank:
                    print(f"Rank {rank_idx}:")
                for stage_idx, comm_bytes_tuple in enumerate(comm_bytes_per_rank):
                    # Handle both tuple (send_bytes, recv_bytes) and tuple (send_bytes, recv_bytes, send_to_self_bytes)
                    if len(comm_bytes_tuple) == 3:
                        send_bytes, recv_bytes, send_to_self_bytes = comm_bytes_tuple
                    else:
                        send_bytes, recv_bytes = comm_bytes_tuple
                        send_to_self_bytes = 0

                    rank_send_total += send_bytes
                    rank_recv_total += recv_bytes
                    rank_send_to_self_total += send_to_self_bytes
                    send_gb = send_bytes / (1024**3)
                    recv_gb = recv_bytes / (1024**3)
                    send_to_self_gb = send_to_self_bytes / (1024**3)
                    if debug_print_per_rank:
                        print(
                            f"  Stage {stage_idx}: Send={send_gb:.4f} GB ({send_bytes:,} bytes), "
                            f"Recv={recv_gb:.4f} GB ({recv_bytes:,} bytes), "
                            f"SendToSelf={send_to_self_gb:.4f} GB ({send_to_self_bytes:,} bytes)"
                        )

                total_send_bytes += rank_send_total
                total_recv_bytes += rank_recv_total
                total_send_to_self_bytes += rank_send_to_self_total
                rank_send_gb = rank_send_total / (1024**3)
                rank_recv_gb = rank_recv_total / (1024**3)
                rank_send_to_self_gb = rank_send_to_self_total / (1024**3)
                if debug_print_per_rank:
                    print(
                        f"  Rank {rank_idx} Total: Send={rank_send_gb:.4f} GB, Recv={rank_recv_gb:.4f} GB, "
                        f"SendToSelf={rank_send_to_self_gb:.4f} GB"
                    )
                    print()

            total_send_gb = total_send_bytes / (1024**3)
            total_recv_gb = total_recv_bytes / (1024**3)
            total_send_to_self_gb = total_send_to_self_bytes / (1024**3)
            print(
                f"Total Communication: Send={total_send_gb:.4f} GB, Recv={total_recv_gb:.4f} GB, "
                f"SendToSelf={total_send_to_self_gb:.4f} GB"
            )
            print("=" * 80)

            # Theoretical calculation of USP communication cost
            ulysses_size = min(8, DATA_CONFIG.heads_kv)
            ring_size = WORLD_SIZE // ulysses_size

            seq_per_rank = TOTAL_SEQLEN // WORLD_SIZE
            head_dim = DATA_CONFIG.hidden_size
            dtype_bytes = DATA_CONFIG.dtype.itemsize

            if BENCH_CONFIG.workload[0] == "fwd":
                # all 2 all for q and o
                comm_bytes = (
                    seq_per_rank
                    * DATA_CONFIG.heads_q
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ulysses_size - 1)
                    / ulysses_size
                )
                # all 2 all for k and v
                comm_bytes += (
                    seq_per_rank
                    * DATA_CONFIG.heads_kv
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ulysses_size - 1)
                    / ulysses_size
                )
                # ring p2p for k and v
                comm_bytes += (
                    seq_per_rank
                    * DATA_CONFIG.heads_kv
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ring_size - 1)
                )
                auto_grad_comm_bytes = 0

            elif BENCH_CONFIG.workload[0] == "bwd":
                # all 2 all for dq and do
                comm_bytes = (
                    seq_per_rank
                    * DATA_CONFIG.heads_q
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ulysses_size - 1)
                    / ulysses_size
                )
                # all 2 all for dk and dv
                comm_bytes += (
                    seq_per_rank
                    * DATA_CONFIG.heads_kv
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ulysses_size - 1)
                    / ulysses_size
                )
                # ring p2p for k and v
                comm_bytes += (
                    seq_per_rank
                    * DATA_CONFIG.heads_kv
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ring_size - 1)
                )
                # ring p2p for dk and dv
                comm_bytes += (
                    seq_per_rank
                    * DATA_CONFIG.heads_kv
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ring_size - 1)
                )

                # all 2 all for q and o (actually not needed)
                auto_grad_comm_bytes = (
                    seq_per_rank
                    * DATA_CONFIG.heads_q
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ulysses_size - 1)
                    / ulysses_size
                )
                # all 2 all for k and v (actually not needed)
                auto_grad_comm_bytes += (
                    seq_per_rank
                    * DATA_CONFIG.heads_kv
                    * head_dim
                    * 2
                    * dtype_bytes
                    * (ulysses_size - 1)
                    / ulysses_size
                )

            print(
                f"Theoretical USP Communication: total "
                f"{WORLD_SIZE * (comm_bytes + auto_grad_comm_bytes) / (1024**3):.4f} GB "
                f"actual {WORLD_SIZE * (comm_bytes) / (1024**3):.4f} GB"
            )


if __name__ == "__main__":
    simulate_solver_and_measure_cost()
