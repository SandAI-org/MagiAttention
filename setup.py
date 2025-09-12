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

import glob
import itertools
import os
import shutil
import subprocess
import sys
import sysconfig
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from packaging.version import Version, parse
from setuptools import Extension, find_namespace_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Note: ninja build requires include_dirs to be absolute paths
project_root = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "magi_attention"
NVIDIA_TOOLCHAIN_VERSION = {"nvcc": "12.6.85", "ptxas": "12.8.93"}
exe_extension = sysconfig.get_config_var("EXE")
USER_HOME = os.getenv("MAGI_ATTENTION_HOME")

# For CI: allow forcing C++11 ABI to match NVCR images that use C++11 ABI
FORCE_CXX11_ABI = os.getenv("MAGI_ATTENTION_FORCE_CXX11_ABI", "0") == "1"

# Skip building CUDA extension modules
SKIP_CUDA_BUILD = os.getenv("MAGI_ATTENTION_SKIP_CUDA_BUILD", "0") == "1"

# We no longer build the flexible_flash_attention_cuda module
# instead, we only pre-build some common options with ref_block_size=None if PREBUILD_FFA is True
# and leave others built in jit mode
PREBUILD_FFA = os.getenv("MAGI_ATTENTION_PREBUILD_FFA", "1") == "1"
PREBUILD_FFA_JOBS = int(os.getenv("MAGI_ATTENTION_PREBUILD_FFA_JOBS", "256"))

# You can also set the flags below to skip building other ext modules
SKIP_FFA_UTILS_BUILD = os.getenv("MAGI_ATTENTION_SKIP_FFA_UTILS_BUILD", "0") == "1"
SKIP_MAGI_ATTN_EXT_BUILD = (
    os.getenv("MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD", "0") == "1"
)

os.environ["MAGI_ATTENTION_BUILD_VERBOSE"] = "1"


title_left_str = "\n\n# -------------------     "
title_right_str = "     ------------------- #\n\n"


def is_in_info_stage() -> bool:
    return "info" in sys.argv[1]


def is_in_wheel_stage() -> bool:
    return "wheel" in sys.argv[1]


def is_in_bdist_wheel_stage() -> bool:
    return "bdist_wheel" == sys.argv[1]


def maybe_make_magi_cuda_extension(name, sources, *args, **kwargs) -> Extension | None:
    name = f"{PACKAGE_NAME}.{name}"

    is_skipped = kwargs.pop("is_skipped", False)

    if is_in_wheel_stage():
        build_repr_str = kwargs.pop(
            "build_repr_str", f"{title_left_str}Building {name}{title_right_str}"
        )
        skip_build_repr_str = kwargs.pop(
            "skip_build_repr_str",
            f"{title_left_str}Skipping Building {name}{title_right_str}",
        )
        if is_skipped:
            print(skip_build_repr_str)
        else:
            print(build_repr_str)

    if is_skipped:
        return None

    return CUDAExtension(name, sources, *args, **kwargs)


def get_cuda_bare_metal_version(cuda_dir) -> tuple[str, Version]:
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # Warn instead of error: users may be downloading prebuilt wheels; nvcc not required in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def nvcc_threads_args() -> list[str]:
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return ["--threads", nvcc_threads]


def init_ext_modules() -> None:
    if is_in_info_stage():
        print(f"\n{torch.__version__=}\n")

    check_if_cuda_home_none(PACKAGE_NAME)

    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    assert bare_metal_version >= Version(
        "12.8"
    ), f"{PACKAGE_NAME} is only supported on CUDA 12.8 and above"

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True


def build_ffa_utils_ext_module(
    repo_dir: Path,
    csrc_dir: Path,
    common_dir: Path,
) -> CUDAExtension | None:
    utils_dir_abs = csrc_dir / "utils"
    utils_dir_rel = utils_dir_abs.relative_to(repo_dir)

    sources = [
        f"{utils_dir_rel}/bindings.cpp",
        f"{utils_dir_rel}/unique_consecutive_pairs.cu",
    ]
    include_dirs = [
        common_dir,
        utils_dir_abs,
    ]

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": nvcc_threads_args()
        + [
            "-O3",
            "-Xptxas",
            "-v",
            "-std=c++17",
            "--use_fast_math",
            "-lineinfo",
            "-DNDEBUG",
        ],
    }

    return maybe_make_magi_cuda_extension(
        name="flexible_flash_attention_utils_cuda",
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
        is_skipped=SKIP_FFA_UTILS_BUILD,
    )


def build_magi_attn_ext_module(
    repo_dir: Path,
    csrc_dir: Path,
    common_dir: Path,
) -> CUDAExtension | None:
    magi_attn_ext_dir_abs = csrc_dir / "extensions"

    # init sources
    cpp_files = glob.glob(str(magi_attn_ext_dir_abs / "*.cpp"))
    sources = [str(Path(f).relative_to(repo_dir)) for f in cpp_files]

    # init include dirs
    include_dirs = [common_dir, magi_attn_ext_dir_abs]

    # init extra compile args
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}

    return maybe_make_magi_cuda_extension(
        name="magi_attn_ext",
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
        is_skipped=SKIP_MAGI_ATTN_EXT_BUILD,
    )


# build ext modules
ext_modules = []
if not SKIP_CUDA_BUILD:
    # init before building any ext module
    init_ext_modules()

    # define some paths for the ext modules below
    repo_dir = Path(project_root)
    csrc_dir = repo_dir / PACKAGE_NAME / "csrc"
    common_dir = csrc_dir / "common"
    cutlass_dir = csrc_dir / "cutlass"

    # build magi attn ext module
    magi_attn_ext_module = build_magi_attn_ext_module(
        repo_dir=repo_dir,
        csrc_dir=csrc_dir,
        common_dir=common_dir,
    )
    if magi_attn_ext_module is not None:
        ext_modules.append(magi_attn_ext_module)

    # build utils ext module
    ffa_utils_ext_module = build_ffa_utils_ext_module(
        repo_dir=repo_dir,
        csrc_dir=csrc_dir,
        common_dir=common_dir,
    )
    if ffa_utils_ext_module is not None:
        ext_modules.append(ffa_utils_ext_module)
else:
    print(f"{title_left_str}Skipping CUDA build{title_right_str}")


# init cmdclass


class MagiAttnBuildExtension(BuildExtension):
    """
    A BuildExtension that switches its behavior based on the command.

    - For development installs (`pip install -e .`), it caches build artifacts
      in the local `./build` directory for faster re-compilation.

    - For building a distributable wheel (`python -m build --wheel`), it uses
      the default temporary directory behavior of PyTorch's BuildExtension to
      ensure robust and correct packaging.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def initialize_options(self) -> None:
        super().initialize_options()

        # Core logic: check if wheel build is running. 'bdist_wheel' is triggered by `python -m build`.
        if not is_in_bdist_wheel_stage():
            # If not building a wheel (i.e., dev install like `pip install -e .`), enable local caching
            print("Development mode detected: Caching build artifacts in build/")
            self.build_temp = os.path.join(project_root, "build", "temp")
            self.build_lib = os.path.join(project_root, "build", "lib")

            # Ensure directories exist
            os.makedirs(self.build_temp, exist_ok=True)
            os.makedirs(self.build_lib, exist_ok=True)
        else:
            # If building a wheel, rely on the default PyTorch behavior so .so files are correctly packaged
            print(
                "Wheel build mode detected: Using default temporary directories in /tmp/ for robust packaging."
            )

    def build_extensions(self) -> None:
        super().build_extensions()


cmdclass = {"bdist_wheel": _bdist_wheel, "build_ext": MagiAttnBuildExtension}


def prebuild_ffa_kernels() -> None:
    if not is_in_wheel_stage():
        return

    if not PREBUILD_FFA:
        print(f"{title_left_str}Skipping Prebuilding FFA JIT kernels{title_right_str}")
        return

    print(
        f"{title_left_str}Prebuilding FFA JIT kernels (ref_block_size=None){title_right_str}"
        "NOTE: this progress may take around 20~30 minute for the first time.\n"
    )

    # During build time, the package isn't installed yet. Fall back to source tree import.
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from magi_attention.common.jit import env as jit_env
        from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_spec
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Prebuild failed: cannot import {PACKAGE_NAME} during build. "
            "Ensure source tree is available. Error: "
        ) from e

    # determine the combinations of prebuild options
    directions = ["fwd", "bwd"]
    head_dims = [64, 128]
    compute_dtypes = [torch.bfloat16, torch.float16]
    out_dtypes = [torch.float32, torch.bfloat16, torch.float16]
    softcaps = [False, True]
    disable_atomic_opts = [False, True]

    combos = itertools.product(
        directions,
        head_dims,
        compute_dtypes,
        out_dtypes,
        softcaps,
        disable_atomic_opts,
    )

    # prebuild the kernels in parallel for the determined options
    def _build_one(args):
        direction, h, cdtype, odtype, sc, da = args
        spec, uri = get_ffa_jit_spec(
            arch=(9, 0),
            direction=direction,
            head_dim=h,
            compute_dtype=cdtype,
            output_dtype=odtype,
            softcap=sc,
            disable_atomic_reduction=da,
            ref_block_size=None,
        )
        spec.build(verbose=True)
        src_dir = (jit_env.MAGI_ATTENTION_JIT_DIR / uri).resolve()
        dst_dir = (jit_env.MAGI_ATTENTION_AOT_DIR / uri).resolve()
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            # if is_in_bdist_wheel_stage():
            #     shutil.copytree(src_dir, bdist_dir, dirs_exist_ok=True)
        return uri

    with ThreadPoolExecutor(max_workers=PREBUILD_FFA_JOBS) as ex:
        futs = {ex.submit(_build_one, c): c for c in combos}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                uri = fut.result()
                print(f"Prebuilt: {uri}")
            except Exception as e:
                print(f"Prebuild failed for {c}: {e}")


# first optionally prebuild FFA JIT kernels (ref_block_size=None)
prebuild_ffa_kernels()


# setup
setup(
    name=PACKAGE_NAME,
    packages=find_namespace_packages(
        exclude=(
            "build",
            "tests",
            "dist",
            "docs",
            "tools",
            "assets",
        )
    ),
    # package data is defined in pyproject.toml
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
