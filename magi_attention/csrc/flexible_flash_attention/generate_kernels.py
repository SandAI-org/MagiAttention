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

# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

KERNEL_BATCH = namedtuple("KERNEL_BATCH", ["template", "filename"])

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

DTYPE_OUT_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
    "fp32": "float",
}

SM = [90]  # Sm kernels support up to
HEAD_DIMENSIONS = [64, 128, 192]
SOFTCAP = [False, True]
DISABLE_FWD_ATOMIC_REDUCTION = [False, True]
DISABLE_BWD_DKV_ATOMIC_REDUCTION = [False, True]

KERNEL_IMPL_TEMPLATE_FWD_SM90 = """#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<{ARCH}, {DTYPE}, {DTYPE_OUT}, {HEAD_DIM}, {SOFTCAP}, {DISABLE_FWD_ATOMIC_REDUCTION}>\
(Flash_fwd_params &params, cudaStream_t stream);
#endif
"""


KERNEL_IMPL_TEMPLATE_BWD_SM90 = """#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_bwd_<{ARCH}, {DTYPE}, {DTYPE_OUT}, {HEAD_DIM}, {SOFTCAP}, {DISABLE_BWD_DKV_ATOMIC_REDUCTION}>\
(Flash_bwd_params &params, cudaStream_t stream);
#endif
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    dtype_out: str
    head_dim: int
    softcap: bool
    direction: str
    disable_fwd_atomic_reduction: bool = False  # Only used for fwd kernels
    disable_bwd_dkv_atomic_reduction: bool = False  # Only used for bwd kernels

    @property
    def template(self) -> str:
        if self.direction == "fwd":
            if self.sm == 90:
                return KERNEL_IMPL_TEMPLATE_FWD_SM90.format(
                    ARCH=str(self.sm),
                    DTYPE=DTYPE_MAP[self.dtype],
                    DTYPE_OUT=DTYPE_OUT_MAP[self.dtype_out],
                    HEAD_DIM=self.head_dim,
                    SOFTCAP=str(self.softcap).lower(),
                    DISABLE_FWD_ATOMIC_REDUCTION=str(
                        self.disable_fwd_atomic_reduction
                    ).lower(),
                )
            else:
                raise NotImplementedError(
                    "Support for SM versions other than 90 is not implemented yet."
                )
        elif self.direction == "bwd":
            if self.sm == 90:
                return KERNEL_IMPL_TEMPLATE_BWD_SM90.format(
                    ARCH=str(self.sm),
                    DTYPE=DTYPE_MAP[self.dtype],
                    DTYPE_OUT=DTYPE_OUT_MAP[self.dtype_out],
                    HEAD_DIM=self.head_dim,
                    SOFTCAP=str(self.softcap).lower(),
                    DISABLE_BWD_DKV_ATOMIC_REDUCTION=str(
                        self.disable_bwd_dkv_atomic_reduction
                    ).lower(),
                )
            else:
                raise NotImplementedError(
                    "Support for SM versions other than 90 is not implemented yet."
                )
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

    @property
    def filename(self) -> str:
        return (
            f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_{self.dtype_out}"
            f"{'_softcap' if self.softcap else ''}"
            f"{'_disable_fwd_atomic_reduction' if self.disable_fwd_atomic_reduction else ''}"
            f"{'_disable_bwd_dkv_atomic_reduction' if self.disable_bwd_dkv_atomic_reduction else ''}"
            f"_sm{self.sm}.cu"
        )


def get_all_kernels() -> Generator[Kernel, None, None]:
    for (
        dtype,
        dtype_out,
        head_dim,
        softcap,
        disable_fwd_atomic_reduction,
        sm,
    ) in itertools.product(
        DTYPE_MAP.keys(),
        DTYPE_OUT_MAP.keys(),
        HEAD_DIMENSIONS,
        SOFTCAP,
        DISABLE_FWD_ATOMIC_REDUCTION,
        SM,
    ):
        yield Kernel(
            sm=sm,
            dtype=dtype,
            dtype_out=dtype_out,
            head_dim=head_dim,
            softcap=softcap,
            direction="fwd",
            disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
        )

    for (
        dtype,
        dtype_out,
        head_dim,
        softcap,
        disable_bwd_dkv_atomic_reduction,
        sm,
    ) in itertools.product(
        DTYPE_MAP.keys(),
        DTYPE_OUT_MAP.keys(),
        HEAD_DIMENSIONS,
        SOFTCAP,
        DISABLE_BWD_DKV_ATOMIC_REDUCTION,
        SM,
    ):
        yield Kernel(
            sm=sm,
            dtype=dtype,
            dtype_out=dtype_out,
            head_dim=head_dim,
            softcap=softcap,
            direction="bwd",
            disable_bwd_dkv_atomic_reduction=disable_bwd_dkv_atomic_reduction,
        )


def batch_fwd(kernels_all) -> Generator[KERNEL_BATCH, None, None]:
    for dtype, softcap, sm in itertools.product(DTYPE_MAP.keys(), SOFTCAP, SM):
        if sm < 90:
            continue

        kernels = [
            k
            for k in kernels_all
            if k.direction == "fwd"
            and k.dtype == dtype
            and k.softcap == softcap
            and k.sm == sm
        ]
        if len(kernels) > 0:
            filename = (
                f"flash_fwd_hdimall_{dtype}{'_softcap' if softcap else ''}_sm{sm}.cu"
            )
            template = "\n".join([f'#include "{k.filename}"' for k in kernels])
            yield KERNEL_BATCH(template, filename)


def batch_bwd(kernels_all) -> Generator[KERNEL_BATCH, None, None]:
    # Bwd
    for dtype, softcap, sm in itertools.product(DTYPE_MAP.keys(), SOFTCAP, SM):
        if sm < 90:
            continue

        kernels = [
            k
            for k in kernels_all
            if k.direction == "bwd"
            and k.dtype == dtype
            and k.softcap == softcap
            and k.sm == sm
        ]
        if len(kernels) > 0:
            filename = (
                f"flash_bwd_hdimall_{dtype}{'_softcap' if softcap else ''}_sm{sm}.cu"
            )
            template = "\n".join([f'#include "{k.filename}"' for k in kernels])
            yield KERNEL_BATCH(template, filename)


def write_kernel(kernel: Kernel | KERNEL_BATCH, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels_all = list(get_all_kernels())

    for kernel in kernels_all:
        write_kernel(kernel, output_dir)
    for kernel in batch_fwd(kernels_all):
        write_kernel(kernel, output_dir)
    for kernel in batch_bwd(kernels_all):
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        default="instantiations",
        required=False,
        help="Where to generate the kernels " " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
