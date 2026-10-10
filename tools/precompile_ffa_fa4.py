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

import torch

from magi_attention.functional.fa4_utils import precompile_ffa_fa4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compile FFA_FA4 kernels for common cases."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(os.cpu_count() or 1, 64),
        help="the number of worker processes (compilation uses mostly the CPU)",
    )
    args = parser.parse_args()

    # Define the parameter space for grid search
    # to pre-compile ffa fa4 kernels for common cases
    dtypes = [torch.float16, torch.bfloat16]
    head_dims = [(64, 64), (128, 128), (192, 128), (256, 256)]
    qhead_per_kvhead = [1, 4]
    func_nums = [2 * i + 1 for i in range(16)]  # 1, 3, 5, .., 31

    # Pre-compile the kernels for all combinations
    # in the defined parameter space
    precompile_ffa_fa4(
        dtypes=dtypes,
        head_dims=head_dims,
        qhead_per_kvhead=qhead_per_kvhead,
        func_nums=func_nums,
        num_workers=args.num_workers,
    )
