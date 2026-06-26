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

"""NCU verification for GQA L2 cache hypothesis at TFLOPS inflection points.

Scenario A — BWD LoopQ S=topk=8K: D1B (543T) vs Dense-nBatch (287T)
  Hypothesis: pack_gqa=True -> 128 GQA groups compete for L2 -> Q data L2 miss
  Key metric: L2 hit/miss ratio

Scenario B — BWD LoopQ S=32K topk=16K: D1B (557T) vs Dense-nBatch (334T)
  Hypothesis: same as A + inv_indices scatter worsens L2
  Key metric: L2 hit/miss ratio

Scenario C — FWD S=topk=4K: IA (615T) vs D1B (599T)
  Hypothesis: q_ranges/k_ranges global access overhead in Dense path
  Key metric: dram__bytes_read.sum (total global read volume)

Usage:
  python bench_loopq_ncu_compare.py --generate   # generate minimal scripts
  python bench_loopq_ncu_compare.py --run        # run NCU (needs root)
  python bench_loopq_ncu_compare.py --parse      # parse & compare results
"""

import argparse
import os
import subprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_SCRIPT_DIR, "outs", "ncu_inflection")

NHQ, NHK, HD, KBS = 128, 1, 128, 128

NCU_METRICS = ",".join(
    [
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
    ]
)

# ═══════════════════════════════════════════════════════════════
#  Script templates
# ═══════════════════════════════════════════════════════════════

TEMPLATE_D1B_BWD_LOOPQ = """\
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
import torch
from magi_attention.functional import flex_flash_attn_func

S, TOPK = {S}, {TOPK}
NHQ, NHK, HD = {NHQ}, {NHK}, {HD}

torch.manual_seed(42)
q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(TOPK, NHK, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(TOPK, NHK, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)

q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device="cuda")
k_ranges = torch.tensor([[0, TOPK]], dtype=torch.int32, device="cuda")
atm = torch.zeros(1, dtype=torch.int32, device="cuda")

out, _ = flex_flash_attn_func(
    q, k, v, q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,
    pack_gqa=False, swap_bwd_qk_loop=False,
)
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE] D1B BWD LoopQ S={S} TOPK={TOPK}")
"""

TEMPLATE_DENSE_NB_BWD_LOOPQ = """\
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
import torch
from magi_attention.functional import flex_flash_attn_func
from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

S, TOPK = {S}, {TOPK}
NHQ, NHK, HD, KBS = {NHQ}, {NHK}, {HD}, {KBS}

torch.manual_seed(42)
q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device="cuda", requires_grad=True)

n_total = S // KBS
n_topk = TOPK // KBS
if n_topk >= n_total:
    idx = torch.arange(n_total, dtype=torch.int32, device="cuda")
    idx = idx.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()
else:
    gen = torch.Generator().manual_seed(42)
    rand_vals = torch.rand(S, n_total, generator=gen)
    perms = rand_vals.argsort(dim=1)[:, :n_topk].sort(dim=1).values
    idx = perms.unsqueeze(1).expand(-1, NHK, -1).to(dtype=torch.int32, device="cuda").contiguous()

ia_3d = idx.permute(1, 0, 2).contiguous()
q_ranges, k_ranges = generate_ranges_from_topk_indices(
    ia_3d, block_m=1, block_n=KBS, num_k_blocks=n_total
)
atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device="cuda")

out, _ = flex_flash_attn_func(
    q, k, v, q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,
    block_sparse=True, auto_range_merge=True, pack_gqa=True, swap_bwd_qk_loop=False,
)
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE] Dense-nB BWD LoopQ S={S} TOPK={TOPK}")
"""

TEMPLATE_IA_FWD = """\
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
import torch
from magi_attention.functional import flex_flash_attn_func

S = {S}
NHQ, NHK, HD, KBS = {NHQ}, {NHK}, {HD}, {KBS}

torch.manual_seed(42)
q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device="cuda")
k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device="cuda")
v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device="cuda")

n_kblocks = S // KBS
idx = torch.arange(n_kblocks, dtype=torch.int32, device="cuda")
idx = idx.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()

out, _ = flex_flash_attn_func(
    q, k, v, index_sparse_indices=idx, k_block_size=KBS,
    index_sparse=True, pack_gqa=True,
)
torch.cuda.synchronize()
print("[DONE] IA FWD S={S}")
"""

TEMPLATE_D1B_FWD = """\
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
import torch
from magi_attention.functional import flex_flash_attn_func

S = {S}
NHQ, NHK, HD = {NHQ}, {NHK}, {HD}

torch.manual_seed(42)
q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device="cuda")
k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device="cuda")
v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device="cuda")

q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device="cuda")
k_ranges = torch.tensor([[0, S]], dtype=torch.int32, device="cuda")
atm = torch.zeros(1, dtype=torch.int32, device="cuda")

out, _ = flex_flash_attn_func(
    q, k, v, q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,
    pack_gqa=False,
)
torch.cuda.synchronize()
print("[DONE] D1B FWD S={S}")
"""


# ═══════════════════════════════════════════════════════════════
#  Generate
# ═══════════════════════════════════════════════════════════════
def generate_scripts():
    os.makedirs(_OUT_DIR, exist_ok=True)

    fmt = dict(NHQ=NHQ, NHK=NHK, HD=HD, KBS=KBS)

    scenarios = [
        ("A_d1b", TEMPLATE_D1B_BWD_LOOPQ, dict(S=8192, TOPK=8192, **fmt)),
        ("A_dense_nb", TEMPLATE_DENSE_NB_BWD_LOOPQ, dict(S=8192, TOPK=8192, **fmt)),
        ("B_d1b", TEMPLATE_D1B_BWD_LOOPQ, dict(S=32768, TOPK=16384, **fmt)),
        ("B_dense_nb", TEMPLATE_DENSE_NB_BWD_LOOPQ, dict(S=32768, TOPK=16384, **fmt)),
        ("C_ia_fwd", TEMPLATE_IA_FWD, dict(S=4096, **fmt)),
        ("C_d1b_fwd", TEMPLATE_D1B_FWD, dict(S=4096, **fmt)),
    ]

    script_paths = {}
    for name, template, params in scenarios:
        path = os.path.join(_OUT_DIR, f"ncu_{name}.py")
        with open(path, "w") as f:
            f.write(template.format(**params))
        script_paths[name] = path
        print(f"  Generated: {path}")

    # Shell runner
    ncu_bin = "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu_bin):
        ncu_bin = "ncu"

    shell_path = os.path.join(_OUT_DIR, "run_all_ncu.sh")
    with open(shell_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write("export CUDA_HOME=/usr/local/cuda-13.0\n\n")

        for name, sp in script_paths.items():
            csv_out = os.path.join(_OUT_DIR, f"ncu_{name}.csv")
            f.write(f"echo '=== {name} ==='\n")
            f.write(
                f'{ncu_bin} --kernel-name "regex:device_kernel" '
                f"--launch-skip 0 --launch-count 1 "
                f"--metrics {NCU_METRICS} --csv "
                f"python {sp} > {csv_out} 2>&1\n\n"
            )
        f.write('echo "ALL NCU DONE"\n')

    os.chmod(shell_path, 0o755)
    print(f"\n  Runner: {shell_path}")
    print(f"  Run:    bash {shell_path}")


# ═══════════════════════════════════════════════════════════════
#  Run NCU
# ═══════════════════════════════════════════════════════════════
def run_ncu():
    shell_path = os.path.join(_OUT_DIR, "run_all_ncu.sh")
    if not os.path.exists(shell_path):
        print(f"ERROR: {shell_path} not found. Run --generate first.")
        return
    print(f"Running: bash {shell_path}")
    subprocess.run(["bash", shell_path], check=False)


# ═══════════════════════════════════════════════════════════════
#  Parse results
# ═══════════════════════════════════════════════════════════════
def parse_results():
    scenarios = [
        ("A", "BWD LoopQ S=topk=8K", ["A_d1b", "A_dense_nb"]),
        ("B", "BWD LoopQ S=32K topk=16K", ["B_d1b", "B_dense_nb"]),
        ("C", "FWD S=topk=4K", ["C_d1b_fwd", "C_ia_fwd"]),
    ]

    for scenario_id, desc, names in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario {scenario_id}: {desc}")
        print("=" * 60)

        for name in names:
            csv_path = os.path.join(_OUT_DIR, f"ncu_{name}.csv")
            if not os.path.exists(csv_path):
                print(f"  {name}: NOT FOUND ({csv_path})")
                continue

            print(f"\n  --- {name} ---")
            with open(csv_path) as f:
                lines = f.read().strip().split("\n")

            # Find metric lines (skip NCU header noise)
            for line in lines:
                for metric_key in [
                    "lts__t_sectors_srcunit_tex_op_read_lookup_hit",
                    "lts__t_sectors_srcunit_tex_op_read_lookup_miss",
                    "sm__pipe_tensor_cycles_active",
                    "sm__warps_active",
                    "dram__bytes_read",
                    "dram__bytes_write",
                ]:
                    if metric_key in line:
                        print(f"    {line.strip()}")
                        break

        # L2 hit ratio calculation for scenarios A/B
        if scenario_id in ("A", "B"):
            print("\n  L2 hit ratio comparison:")
            for name in names:
                csv_path = os.path.join(_OUT_DIR, f"ncu_{name}.csv")
                if not os.path.exists(csv_path):
                    continue
                hit, miss = None, None
                with open(csv_path) as f:
                    for line in f:
                        if "lookup_hit" in line:
                            parts = line.split(",")
                            for p in parts:
                                try:
                                    hit = float(p.strip().replace('"', ""))
                                except ValueError:
                                    pass
                        if "lookup_miss" in line:
                            parts = line.split(",")
                            for p in parts:
                                try:
                                    miss = float(p.strip().replace('"', ""))
                                except ValueError:
                                    pass
                if hit is not None and miss is not None and (hit + miss) > 0:
                    ratio = hit / (hit + miss) * 100
                    print(f"    {name}: L2 hit = {ratio:.1f}%")
                else:
                    print(f"    {name}: could not parse L2 metrics")


# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="NCU verification for GQA L2 hypothesis"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate NCU profiling scripts"
    )
    parser.add_argument("--run", action="store_true", help="Execute NCU profiling")
    parser.add_argument("--parse", action="store_true", help="Parse NCU results")
    args = parser.parse_args()

    if not any([args.generate, args.run, args.parse]):
        parser.error("Specify --generate, --run, or --parse")

    if args.generate:
        generate_scripts()
    if args.run:
        run_ncu()
    if args.parse:
        parse_results()


if __name__ == "__main__":
    main()
