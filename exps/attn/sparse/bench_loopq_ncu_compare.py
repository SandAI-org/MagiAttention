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

"""NCU comparison: BWD LoopQ Dense vs BlockSparse at topk=16K.

Purpose: Verify that L2 cache miss is THE bottleneck for BWD LoopQ sparse
TFLOPS drop, and NOT atomicAdd/register pressure/occupancy.

Collects key metrics:
  - lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum  (L2 hit)
  - lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum (L2 miss)
  - sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed (tensor util)
  - sm__warps_active.avg.pct_of_peak_sustained_elapsed (warp occupancy)

Runs two configurations:
  1. Dense BWD LoopQ (S=32K, topk=16K) — high TFLOPS baseline
  2. BlockSparse BWD LoopQ (S=32K, topk=16K, kbs=128) — low TFLOPS

Usage:
  # Step 1: Generate minimal Python scripts for NCU profiling
  python exps/attn/sparse/bench_loopq_ncu_compare.py --generate-scripts

  # Step 2: Run NCU on the generated scripts (manually, needs root/sudo)
  # See generated shell commands in outs/loopq_ncu/run_ncu.sh

  # Step 3: Parse and compare NCU output
  python exps/attn/sparse/bench_loopq_ncu_compare.py --parse-ncu
"""

import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_SCRIPT_DIR, "outs", "loopq_ncu")

DENSE_SCRIPT = """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
import torch
from magi_attention.functional.flex_flash_attn import flex_flash_attn_func

B, S, NHQ, NHK, HD = 1, 32768, 128, 1, 128
TOPK = 16384
torch.manual_seed(42)
q = torch.randn(B, S, NHQ, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(B, S, NHK, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(B, S, NHK, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
do = torch.randn(B, S, NHQ, HD, dtype=torch.bfloat16, device="cuda")

# Dense BWD LoopQ (no sparsity, equivalent to topk=S)
out = flex_flash_attn_func(q, k, v, swap_bwd_qk_loop=False)
out.backward(do)
torch.cuda.synchronize()
print("[NCU-DENSE] BWD LoopQ completed")
"""

SPARSE_SCRIPT = """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
import torch
from magi_attention.functional.flex_flash_attn import flex_flash_attn_func

B, S, NHQ, NHK, HD = 1, 32768, 128, 1, 128
TOPK = 16384
KBS = 128

torch.manual_seed(42)
q = torch.randn(B, S, NHQ, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(B, S, NHK, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(B, S, NHK, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
do = torch.randn(B, S, NHQ, HD, dtype=torch.bfloat16, device="cuda")

# BlockSparse: block_mask selects first TOPK/KBS blocks per Q-block
n_kblocks = S // KBS
topk_blocks = TOPK // KBS
block_mask = torch.zeros(B, NHK, S // KBS, n_kblocks, dtype=torch.bool, device="cuda")
# Each Q-block selects the first topk_blocks K-blocks (deterministic pattern)
block_mask[:, :, :, :topk_blocks] = True

out = flex_flash_attn_func(
    q, k, v,
    block_sparse_block_mask=block_mask,
    block_sparse_k_block_size=KBS,
    swap_bwd_qk_loop=False,  # LoopQ
)
out.backward(do)
torch.cuda.synchronize()
print("[NCU-SPARSE] BWD LoopQ BlockSparse completed")
"""


def generate_scripts(gpu_id):
    os.makedirs(_OUT_DIR, exist_ok=True)

    dense_path = os.path.join(_OUT_DIR, "ncu_dense_loopq.py")
    sparse_path = os.path.join(_OUT_DIR, "ncu_sparse_loopq.py")

    with open(dense_path, "w") as f:
        f.write(DENSE_SCRIPT.format(gpu_id=gpu_id))
    with open(sparse_path, "w") as f:
        f.write(SPARSE_SCRIPT.format(gpu_id=gpu_id))

    metrics = ",".join(
        [
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
            "dram__bytes_read.sum",
        ]
    )

    ncu_bin = "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu_bin):
        ncu_bin = "/usr/local/cuda-12.8/bin/ncu"

    shell_script = os.path.join(_OUT_DIR, "run_ncu.sh")
    with open(shell_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# NCU profiling for BWD LoopQ Dense vs BlockSparse\n")
        f.write(f"# Run with: bash {shell_script}\n\n")
        f.write("export CUDA_HOME=/usr/local/cuda-13.0\n\n")
        f.write("# Dense BWD LoopQ\n")
        f.write(
            f'{ncu_bin} --kernel-name "regex:device_kernel" --launch-skip 3 --launch-count 1 '
            f"--metrics {metrics} --csv "
            f"python {dense_path} > {_OUT_DIR}/ncu_dense.csv 2>&1\n\n"
        )
        f.write("# Sparse BWD LoopQ\n")
        f.write(
            f'{ncu_bin} --kernel-name "regex:device_kernel" --launch-skip 3 --launch-count 1 '
            f"--metrics {metrics} --csv "
            f"python {sparse_path} > {_OUT_DIR}/ncu_sparse.csv 2>&1\n\n"
        )
        f.write(
            f'echo "Done. Compare {_OUT_DIR}/ncu_dense.csv vs {_OUT_DIR}/ncu_sparse.csv"\n'
        )

    os.chmod(shell_script, 0o755)
    print("Generated scripts:")
    print(f"  Dense:  {dense_path}")
    print(f"  Sparse: {sparse_path}")
    print(f"  NCU runner: {shell_script}")
    print(f"\nRun: bash {shell_script}")


def parse_ncu():
    for label, fname in [("Dense", "ncu_dense.csv"), ("Sparse", "ncu_sparse.csv")]:
        path = os.path.join(_OUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  {label}: FILE NOT FOUND ({path})")
            continue
        print(f"\n=== {label} BWD LoopQ ===")
        with open(path) as f:
            content = f.read()
        # Print last 20 lines (CSV metrics are at the end)
        lines = content.strip().split("\n")
        for line in lines[-20:]:
            print(f"  {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-scripts", action="store_true")
    parser.add_argument("--parse-ncu", action="store_true")
    parser.add_argument("--gpu", type=int, default=7, help="GPU ID to use")
    args = parser.parse_args()

    if args.generate_scripts:
        generate_scripts(args.gpu)
    elif args.parse_ncu:
        parse_ncu()
    else:
        parser.error("Specify --generate-scripts or --parse-ncu")


if __name__ == "__main__":
    main()
