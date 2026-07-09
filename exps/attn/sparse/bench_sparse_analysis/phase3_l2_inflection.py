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

import os
import subprocess
import sys

from bench_sparse_analysis._common import HD, KBS, NHK, NHQ, _out_dir, _ts


# ═══════════════════════════════════════════════════════════════
#  Phase 3: l2-inflection (NCU at specific inflection points)
# ═══════════════════════════════════════════════════════════════
def _phase3_ncu():
    phase = "3-l2-inflection"
    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    ncu_bin = "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu_bin):
        ncu_bin = "ncu"

    metrics = ",".join(
        [
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
        ]
    )

    scripts_dir = os.path.join(out, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    fmt = dict(NHQ=NHQ, NHK=NHK, HD=HD, KBS=KBS)

    TEMPLATE_D1B_BWD = """\
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
import torch
from magi_attention.functional import flex_flash_attn_func
S, TOPK = {S}, {TOPK}
torch.manual_seed(42)
q = torch.randn(S, {NHQ}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(TOPK, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(TOPK, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device="cuda")
k_ranges = torch.tensor([[0, TOPK]], dtype=torch.int32, device="cuda")
atm = torch.zeros(1, dtype=torch.int32, device="cuda")
out, _ = flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges,
    attn_type_map=atm, pack_gqa=False, swap_bwd_qk_loop=False)
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE] D1B BWD LoopQ S={S} TOPK={TOPK}")
"""

    TEMPLATE_DENSE_NB_BWD = """\
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
import torch
from magi_attention.functional import flex_flash_attn_func
from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices
S, TOPK = {S}, {TOPK}
torch.manual_seed(42)
q = torch.randn(S, {NHQ}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(S, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(S, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
n_total, n_topk = S // {KBS}, TOPK // {KBS}
if n_topk >= n_total:
    idx = torch.arange(n_total, dtype=torch.int32, device="cuda")
    idx = idx.unsqueeze(0).unsqueeze(0).expand(S, {NHK}, -1).contiguous()
else:
    gen = torch.Generator().manual_seed(42)
    rand_vals = torch.rand(S, n_total, generator=gen)
    perms = rand_vals.argsort(dim=1)[:, :n_topk].sort(dim=1).values
    idx = perms.unsqueeze(1).expand(-1, {NHK}, -1).to(dtype=torch.int32, device="cuda").contiguous()
ia_3d = idx.permute(1, 0, 2).contiguous()
q_ranges, k_ranges = generate_ranges_from_topk_indices(ia_3d, block_m=1, block_n={KBS}, num_k_blocks=n_total)
atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device="cuda")
out, _ = flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges,
    attn_type_map=atm, block_sparse=True, range_merge=True, pack_gqa=True,
    sparse_k_block_size={KBS}, swap_bwd_qk_loop=False)
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE] Dense-MultiBatch BWD LoopQ S={S} TOPK={TOPK}")
"""

    scenarios = [
        (
            "A_d1b",
            TEMPLATE_D1B_BWD,
            dict(S=8192, TOPK=8192, **fmt),
            "BWD LoopQ S=topk=8K: Dense-SingleBatch",
        ),
        (
            "A_dense_nb",
            TEMPLATE_DENSE_NB_BWD,
            dict(S=8192, TOPK=8192, **fmt),
            "BWD LoopQ S=topk=8K: Dense-MultiBatch",
        ),
        (
            "B_d1b",
            TEMPLATE_D1B_BWD,
            dict(S=32768, TOPK=16384, **fmt),
            "BWD LoopQ S=32K topk=16K: Dense-SingleBatch",
        ),
        (
            "B_dense_nb",
            TEMPLATE_DENSE_NB_BWD,
            dict(S=32768, TOPK=16384, **fmt),
            "BWD LoopQ S=32K topk=16K: Dense-MultiBatch",
        ),
    ]

    for name, template, params, desc in scenarios:
        script_path = os.path.join(scripts_dir, f"ncu_{name}.py")
        with open(script_path, "w") as f:
            f.write(template.format(**params))

        csv_path = os.path.join(out, f"ncu_{name}.csv")
        cmd = [
            ncu_bin,
            "--kernel-name",
            "regex:device_kernel",
            "--launch-skip",
            "3",
            "--launch-count",
            "1",
            "--metrics",
            metrics,
            "--csv",
            sys.executable,
            script_path,
        ]
        print(f"  [{_ts()}] NCU {desc}...", end=" ", flush=True)
        with open(csv_path, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, timeout=600)
        print("done", flush=True)

    print(f"\n[{_ts()}] Phase 3 NCU results in {out}/ncu_*.csv")

    # Parse L2 hit ratios
    print("\n  L2 hit ratio summary:")
    for name, _, _, desc in scenarios:
        csv_path = os.path.join(out, f"ncu_{name}.csv")
        if not os.path.exists(csv_path):
            print(f"    {name}: NOT FOUND")
            continue
        hit, miss = 0, 0
        with open(csv_path) as f:
            for line in f:
                if "lookup_hit" in line:
                    for p in line.split(","):
                        try:
                            hit = float(p.strip().replace('"', ""))
                        except ValueError:
                            pass
                if "lookup_miss" in line:
                    for p in line.split(","):
                        try:
                            miss = float(p.strip().replace('"', ""))
                        except ValueError:
                            pass
        if hit + miss > 0:
            ratio = hit / (hit + miss) * 100
            print(f"    {name}: L2 hit = {ratio:.1f}%")
        else:
            print(f"    {name}: could not parse")
