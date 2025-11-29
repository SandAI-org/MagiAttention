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

from magi_attention.benchmarking import Mark, report_all_from_perf

perf_key = "flops"
mask_type = "full"
wds = ["fwd", "bwd"]
root = "outs/test"

for wd in wds:
    path = f"{root}/{wd}"

    Mark.draw_from_csv(
        csv_path=f"{path}/{perf_key}.csv",
        perf_key=perf_key,
        plot_name=f"attn-{wd} with {mask_type} mask",
        line_arg="attn_impl",
        save_path=path,
        ylabel="Throughout (TFLOPs/s)",
        x_int=True,
        x_log=False,
    )

report_all_from_perf(
    save_root=root,
)
