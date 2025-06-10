# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

KERNEL_BATCH = namedtuple("Kernel", ["template", "filename"])

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

KERNEL_IMPL_TEMPLATE_FWD_SM90 = """#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<{ARCH}, {DTYPE}, {DTYPE_OUT}, {HEAD_DIM}, {SOFTCAP}, {DISABLE_FWD_ATOMIC_REDUCTION}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
"""


KERNEL_IMPL_TEMPLATE_BWD_SM90 = """#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_bwd_<{ARCH}, {DTYPE}, {DTYPE_OUT}, {HEAD_DIM}, {SOFTCAP}>(Flash_bwd_params &params, cudaStream_t stream);
#endif
"""

@dataclass
class Kernel:
    sm: int
    dtype: str
    dtype_out: str
    head_dim: int
    softcap: bool
    disable_fwd_atomic_reduction: bool
    direction: str

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
                    DISABLE_FWD_ATOMIC_REDUCTION=str(self.disable_fwd_atomic_reduction).lower(),
                )
            else:
                ...
        elif self.direction == "bwd":
            if self.sm == 90:
                return KERNEL_IMPL_TEMPLATE_BWD_SM90.format(
                    ARCH=str(self.sm),
                    DTYPE=DTYPE_MAP[self.dtype],
                    DTYPE_OUT=DTYPE_OUT_MAP[self.dtype_out],
                    HEAD_DIM=self.head_dim,
                    SOFTCAP=str(self.softcap).lower(),
                )
            else:
                ...

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_{self.dtype_out}{'_softcap' if self.softcap else ''}{'_disable_fwd_atomic_reduction' if self.disable_fwd_atomic_reduction else ''}_sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for dtype, dtype_out, head_dim, softcap, disable_fwd_atomic_reduction, sm in itertools.product(
        DTYPE_MAP.keys(), DTYPE_OUT_MAP.keys(), HEAD_DIMENSIONS, SOFTCAP, DISABLE_FWD_ATOMIC_REDUCTION, SM
    ):
        yield Kernel(
            sm=sm,
            dtype=dtype,
            dtype_out=dtype_out,
            head_dim=head_dim,
            softcap=softcap,
            disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
            direction="fwd",
        )
        
    for dtype, dtype_out, head_dim, softcap, sm in itertools.product(
        DTYPE_MAP.keys(), DTYPE_OUT_MAP.keys(), HEAD_DIMENSIONS, SOFTCAP, SM
    ):
        yield Kernel(
            sm=sm,
            dtype=dtype,
            dtype_out=dtype_out,
            head_dim=head_dim,
            softcap=softcap,
            disable_fwd_atomic_reduction=False,
            direction="bwd",
        )


def batch_fwd(kernels_all) -> List[KERNEL_BATCH]:
    for dtype, softcap, sm in itertools.product(
        DTYPE_MAP.keys(), SOFTCAP, SM
    ):
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
            filename = f"flash_fwd_hdimall_{dtype}{'_softcap' if softcap else ''}_sm{sm}.cu"
            template = "\n".join([f'#include "{k.filename}"' for k in kernels])
            yield KERNEL_BATCH(template, filename)


def batch_bwd(kernels_all) -> List[KERNEL_BATCH]:
    # Bwd
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SM):
        if sm < 90:
            continue

        kernels = [
            k
            for k in kernels_all
            if k.direction == "bwd"
            and k.dtype == dtype
            and k.head_dim == head_dim
            and k.sm == sm
        ]
        if len(kernels) > 0:
            filename = f"flash_bwd_hdim{head_dim}_{dtype}_softcapall_sm{sm}.cu"
            template = "\n".join([f'#include "{k.filename}"' for k in kernels])
            yield KERNEL_BATCH(template, filename)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
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
