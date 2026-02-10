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

import importlib
import importlib.resources
import itertools
import math
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
exe_extension = sysconfig.get_config_var("EXE")
USER_HOME = os.getenv("MAGI_ATTENTION_HOME")

# For CUDA13.0: the cccl header path needs to be explicitly included
CUDA13_CCCL_PATH = "/usr/local/cuda-13.0/include/cccl/"

# For CI: allow forcing C++11 ABI to match NVCR images that use C++11 ABI
FORCE_CXX11_ABI = os.getenv("MAGI_ATTENTION_FORCE_CXX11_ABI", "0") == "1"

# Skip building CUDA extension modules
SKIP_CUDA_BUILD = os.getenv("MAGI_ATTENTION_SKIP_CUDA_BUILD", "0") == "1"

# NOTE: this flag now only works for `magi_attn_comm` to disable sm90 features
# to be compatible with other architectures such as sm80
# thus we won't put it into docs until all other extensions such as FFA supports architectures other than sm90
DISABLE_SM90_FEATURES = os.getenv("MAGI_ATTENTION_DISABLE_SM90_FEATURES", "0") == "1"

# NOTE: this flag now only works for `magi_attn_comm` to disable aggressive PTX instructions
# such as LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
# however, it is set default to `1` as `.L1::no_allocate` might not be safe to to load volatile data
# according to this issue: https://github.com/deepseek-ai/DeepEP/issues/136
# REVIEW: however, we well test it and find no correctness issue but a notable performance gain
# so in the future we might need to dig deeper into this
DISABLE_AGGRESSIVE_PTX_INSTRS = os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "1") == "1"

# We no longer build the flexible_flash_attention_cuda module
# instead, we only pre-build some common options with ref_block_size=None if PREBUILD_FFA is True
# and leave others built in jit mode
PREBUILD_FFA = os.getenv("MAGI_ATTENTION_PREBUILD_FFA", "1") == "1"

# Set this environment variable to control the number of parallel compilation jobs
# including pre-build FFA jobs and other ext modules jobs
# defaults to the ceiling of 90% of the available CPU cores
default_jobs = math.ceil(os.cpu_count() * 0.9)  # type: ignore[operator]
PREBUILD_FFA_JOBS = int(
    os.getenv("MAGI_ATTENTION_PREBUILD_FFA_JOBS", str(default_jobs))
)
os.environ["MAX_JOBS"] = os.getenv("MAX_JOBS", str(default_jobs))

# You can also set the flags below to skip building other ext modules
SKIP_FFA_UTILS_BUILD = os.getenv("MAGI_ATTENTION_SKIP_FFA_UTILS_BUILD", "0") == "1"
SKIP_MAGI_ATTN_EXT_BUILD = (
    os.getenv("MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD", "0") == "1"
)
SKIP_MAGI_ATTN_COMM_BUILD = (
    os.getenv("MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD", "0") == "1"
)

BUILD_COMPUTE_CAPABILITY = os.getenv("MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY", "")

# Defaults to enable verbose building magi_attention
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


def get_device_compute_capability(
    with_minor: bool = True, with_a: bool = False, default_cap: str | None = None
) -> str:
    """Get the compute capability of the current CUDA device.
    Example: '80', '90', '100', etc.

    Args:
        with_minor (bool): Whether to include the minor version in the output.
            Defaults to ``True``.
        with_a (bool): Whether to append 'a' suffix to the capability.
            Defaults to ``False``.
        default_cap (str | None): The default capability to return if CUDA is not available.
            Defaults to ``None`` to raise an error if CUDA is not available.

    Returns:
        str: The compute capability of the current CUDA device.
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if with_minor:  # include minor version, like 90, 100, 103
            capability = f"{major}{minor}"
        else:  # only major version with minor as 0, like 90, 100
            capability = f"{major}0"

        if with_a:  # include suffix 'a' like 90a, 100a
            capability += "a"
    else:
        if default_cap is not None:
            capability = default_cap
        else:
            raise RuntimeError("CUDA device is not available to get compute capability")

    return capability


# Copied from https://github.com/deepseek-ai/DeepEP/blob/main/setup.py
# Wheel specific: The wheels only include the soname of the host library (libnvshmem_host.so.X)
def get_nvshmem_host_lib_name():
    for path in importlib.resources.files("nvidia.nvshmem").iterdir():
        for file in path.rglob("libnvshmem_host.so.*"):
            return file.name
    raise ModuleNotFoundError("libnvshmem_host.so not found")


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
    if bare_metal_version < Version("12.8"):
        warnings.warn(
            f"We recommend installing {PACKAGE_NAME} on well-tested CUDA 12.8 and above."
        )

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True


def build_ffa_utils_ext_module(
    repo_dir: Path,
    csrc_dir: Path,
    common_dir: Path,
) -> Extension | None:
    utils_dir_abs = csrc_dir / "utils"
    utils_dir_rel = utils_dir_abs.relative_to(repo_dir)

    sources = [
        f"{utils_dir_rel}/bindings.cpp",
        f"{utils_dir_rel}/unique_consecutive_pairs.cu",
        f"{utils_dir_rel}/profile_utils.cu",
        f"{utils_dir_rel}/preprocess_sparse_load.cu",
        f"{utils_dir_rel}/sort_and_reorder_ranges.cu",
    ]
    include_dirs = [
        common_dir,
        utils_dir_abs,
        CUDA13_CCCL_PATH,
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
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        is_skipped=SKIP_FFA_UTILS_BUILD,
    )


def build_magi_attn_ext_module(
    csrc_dir: Path,
) -> None:
    """
    Manually triggers the CMake build process for the 'magi_attn_ext' shared library.

    Unlike standard setuptools extensions, this module uses CMake to manage complex
    C++ dependencies and build configurations

    Returns:
        None: This function returns None because it compiles the library manually
              via subprocess calls, rather than returning a setuptools.Extension
              object for setuptools to handle.
    """
    if not is_in_wheel_stage():
        return

    # Check Environment Skip Flag
    # Allows users to bypass this specific build step via environment variable,
    # useful for CI/CD or partial rebuilds.
    if SKIP_MAGI_ATTN_EXT_BUILD:
        return None

    # Path Configuration
    # Define the absolute path to the extension source and the build directory.
    # We use an "out-of-source" build strategy (creating a separate 'build' folder)
    # to keep the source tree clean.
    magi_attn_ext_dir_abs = csrc_dir / "extensions"
    build_dir = magi_attn_ext_dir_abs / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"{title_left_str}Building magi_attn_ext with CMake{title_right_str}")

    # CMake Configuration Step
    # We invoke 'cmake' to generate the build system (Makefiles).
    # Critical Flag: -DCMAKE_PREFIX_PATH
    # This tells CMake where to find the PyTorch C++ installation (LibTorch),
    # ensuring we link against the correct Torch libraries matching the Python environment.
    subprocess.check_call(
        [
            "cmake",
            str(magi_attn_ext_dir_abs),  # Explicitly point to the source directory
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ],
        cwd=build_dir,
    )

    # Compilation Step
    # We invoke 'make' to actually compile the C++ code using the generated Makefiles.
    subprocess.check_call(
        ["make", f"-j{os.environ.get('MAX_JOBS', '8')}"],
        cwd=build_dir,
    )

    # Return None to indicate to setuptools that it does not need to manage
    # this extension, as we have successfully built it manually above.
    return None


def build_magi_attn_comm_module(
    repo_dir: Path,
    csrc_dir: Path,
    common_dir: Path,
    extensions_dir: Path,
    cutlass_dir: Path,
) -> Extension | None:
    """
    Constructs the CUDA extension configuration for the 'magi_attn_comm' module.
    This module handles communication primitives (likely for distributed attention),
    leveraging NVSHMEM for efficient GPU-to-GPU data movement.
    """
    # NOTE: we've found the compilation fails with `sm103`
    # thus we only use the major version with minor as `0`,
    # i.e. only `sm80`, `sm90`, `sm100`, etc.
    capability = BUILD_COMPUTE_CAPABILITY
    if capability == "":
        capability = get_device_compute_capability(
            with_minor=False, with_a=False, default_cap="90"
        )

    # ---   for grpcoll submodule   --- #

    # NVSHMEM Detection Logic
    # NVSHMEM is a library that allows GPUs to communicate directly.
    # We attempt to locate it via environment variables or installed Python packages.
    disable_nvshmem = False
    nvshmem_dir = os.getenv("NVSHMEM_DIR", None)
    nvshmem_host_lib = "libnvshmem_host.so"

    if nvshmem_dir is None:
        try:
            # Attempt to find NVSHMEM within the installed 'nvidia.nvshmem' python package
            nvshmem_dir = importlib.util.find_spec(  # type: ignore[union-attr,index]
                "nvidia.nvshmem"
            ).submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name()
            import nvidia.nvshmem as nvshmem  # noqa: F401

            if is_in_info_stage():
                print(
                    f"`NVSHMEM_DIR` is not specified, thus found from system module: {nvshmem_dir}"
                )
        except (ModuleNotFoundError, AttributeError, IndexError):
            # If neither env var nor python package is found, disable NVSHMEM features
            if is_in_info_stage():
                warnings.warn(
                    "Since `NVSHMEM_DIR` is not specified, and the system nvshmem module is not installed, "
                    "then all relative features used in native group collective comm kernels are disabled\n"
                )
            disable_nvshmem = True
    else:
        if is_in_info_stage():
            print(f"Found specified `NVSHMEM_DIR`: {nvshmem_dir}")
        disable_nvshmem = False

    # Validation: Ensure the directory actually exists if we aren't disabling it
    if not disable_nvshmem:
        assert os.path.exists(
            nvshmem_dir  # type: ignore[arg-type]
        ), f"The specified NVSHMEM directory does not exist: {nvshmem_dir}"

    # Path Setup
    # Define absolute and relative paths for the communication source code (grpcoll)
    magi_attn_comm_dir_abs = csrc_dir / "comm"
    grpcoll_dir_abs = magi_attn_comm_dir_abs / "grpcoll"
    grpcoll_dir_rel = grpcoll_dir_abs.relative_to(repo_dir)

    # Generate instantiations
    inst_dir_abs = grpcoll_dir_abs / "instantiations"
    inst_dir_abs.mkdir(parents=True, exist_ok=True)

    gen_script = grpcoll_dir_abs / "generate_inst.py"
    if gen_script.exists():
        print(f"Running {gen_script} to generate instantiation files...")
        subprocess.check_call([sys.executable, str(gen_script)], cwd=repo_dir)

    # Source File Collection
    # Initialize list of source files required for compilation
    sources = [
        f"{grpcoll_dir_rel}/buffer.cpp",  # Host-side buffer management
        f"{grpcoll_dir_rel}/kernels/runtime.cu",  # CUDA runtime helpers
        f"{grpcoll_dir_rel}/kernels/layout.cu",  # Memory layout management
    ]

    # Add instantiation files automatically.
    # Large CUDA projects often split template instantiations into separate files
    # to parallelize compilation and reduce build time.
    inst_dir_rel = f"{grpcoll_dir_rel}/instantiations"
    inst_dir_abs = grpcoll_dir_abs / "instantiations"
    if inst_dir_abs.exists():
        # Intranode: Communication within the same node (e.g., NVLink)
        for file in inst_dir_abs.glob("intranode_*.cu"):
            sources.append(f"{inst_dir_rel}/{file.name}")
        # Internode: Communication across nodes (e.g., IB/RoCE via NVSHMEM)
        for file in inst_dir_abs.glob("internode_*.cu"):
            sources.append(f"{inst_dir_rel}/{file.name}")

    # Add specific kernel implementations
    sources.append(f"{grpcoll_dir_rel}/kernels/intranode_notify_kernel.cu")
    sources.append(f"{grpcoll_dir_rel}/kernels/internode_ll.cu")  # Low-latency kernels
    sources.append(f"{grpcoll_dir_rel}/kernels/internode_notify_kernel.cu")
    sources.append(f"{grpcoll_dir_rel}/kernels/internode_utils.cu")

    # Include Directories
    # Specify where the compiler looks for header files (.h/.cuh)
    # CUDA13_CCCL_PATH: C++ Core Compute Libraries (modern CUDA standard libs)
    include_dirs = [
        CUDA13_CCCL_PATH,
        common_dir,
        extensions_dir,
        cutlass_dir,
        grpcoll_dir_abs,
        grpcoll_dir_abs / "kernels",
    ]

    # Compiler Flags
    # Flags for the standard C++ compiler (gcc/g++)
    cxx_flags = [
        "-O3",  # Maximize optimization
        "-Wno-deprecated-declarations",  # Suppress warnings about deprecated code
        "-Wno-unused-variable",  # Suppress warnings about unused variables
        "-Wno-sign-compare",  # Suppress signed/unsigned comparison warnings
        "-Wno-reorder",  # Suppress member initialization order warnings
        "-Wno-attributes",
        # "-ftime-report",  # Uncomment for profiling compilation time
    ]

    # Flags for the NVIDIA CUDA Compiler (nvcc)
    nvcc_flags = [
        "-O3",
        "-Xptxas",  # Pass arguments to ptxas (PTX assembler)
        "-v",  # Verbose output
        "-Xcompiler",  # Pass arguments to the host compiler
        "-std=c++17",  # Use C++17 standard
        "-lineinfo",  # Generate line-number information for profiling
        "-gencode",
        # Explicitly specify for current device compute capability
        f"arch=compute_{capability},code=sm_{capability}",
        # "-Xcompiler",  # Uncomment for profiling compilation time
        # "-ftime-report",  # Uncomment for profiling compilation time
    ]

    # Initialize lists for linking configuration
    library_dirs = []
    nvcc_dlink = []  # Device link flags (critical for RDC - Relocatable Device Code)
    extra_link_args = []

    # Linking against sibling extension
    # If the base 'magi_attn_ext' library exists, link against it.
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    magi_attn_ext_lib = repo_dir / PACKAGE_NAME / f"magi_attn_ext{ext_suffix}"

    if magi_attn_ext_lib.exists():
        # 1. Add the directory containing the library to the linker search path
        lib_dir = str(magi_attn_ext_lib.parent)
        extra_link_args.append(f"-L{lib_dir}")

        # 2. Link against the specific filename instead of the absolute path.
        # Using '-l::filename' (or '-l:filename') ensures the linker records
        # only the filename in the 'DT_NEEDED' section of the ELF header,
        # avoiding hardcoded absolute paths from the build environment.
        extra_link_args.append(f"-l:{magi_attn_ext_lib.name}")

        # 3. Set RPATH to $ORIGIN so the loader looks for dependencies in
        # the same directory as the extension at runtime.
        # Use '\$ORIGIN' to prevent the shell or compiler from expanding it as a variable.
        extra_link_args.append("-Wl,-rpath,$ORIGIN")

    # NVSHMEM Configuration (Conditional)
    if disable_nvshmem:
        # Define macro to disable code paths relying on NVSHMEM
        cxx_flags.append("-DDISABLE_NVSHMEM")
        nvcc_flags.append("-DDISABLE_NVSHMEM")
    else:
        # Enable NVSHMEM: Add includes, library paths, and link flags
        include_dirs.extend([f"{nvshmem_dir}/include"])  # type: ignore[list-item]
        library_dirs.extend([f"{nvshmem_dir}/lib"])

        # -dlink and -lnvshmem_device are required for device-side linking
        nvcc_dlink.extend(["-dlink", f"-L{nvshmem_dir}/lib", "-lnvshmem_device"])

        # Host-side linking
        extra_link_args.extend(
            [
                f"-l:{nvshmem_host_lib}",  # Link host library
                "-l:libnvshmem_device.a",  # Link static device library
                f"-Wl,-rpath,{nvshmem_dir}/lib",  # Add runtime search path
            ]
        )

    # SM90 (Hopper) Feature Configuration
    if DISABLE_SM90_FEATURES:
        # If SM90 features (FP8, TMA, etc.) are disabled globally:
        cxx_flags.append("-DDISABLE_SM90_FEATURES")
        nvcc_flags.append("-DDISABLE_SM90_FEATURES")
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

        # Logic enforcement: If we can't use SM90 features, we likely can't use
        # the advanced internode kernels that depend on them, so NVSHMEM must be disabled.
        assert disable_nvshmem
    else:
        # CUDA 12 / SM90 Enabled settings
        # -rdc=true: Enable Relocatable Device Code. This is usually required for
        #            NVSHMEM or when calling device functions across translation units.
        nvcc_flags.extend(["-rdc=true", "--ptxas-options=--register-usage-level=10"])

        # Aggressive PTX instructions optimization
        # Some custom PTX assembly tricks (like specific LD/ST cache hints) might
        # not be supported or safe in all environments.
        if DISABLE_AGGRESSIVE_PTX_INSTRS:
            cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
            nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

    # Final Argument Assembly
    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }
    # Only add 'nvcc_dlink' if we actually have device link flags (i.e., NVSHMEM is on)
    if len(nvcc_dlink) > 0:
        extra_compile_args["nvcc_dlink"] = nvcc_dlink

    # Extension Creation
    # Calls a wrapper function to instantiate the actual setuptools Extension object.
    return maybe_make_magi_cuda_extension(
        name="magi_attn_comm",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        sources=sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        is_skipped=SKIP_MAGI_ATTN_COMM_BUILD,  # Check if build is explicitly skipped via env var
    )


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
    compute_output_dtype_tuples = [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ]
    disable_atomic_reductions = [False, True]
    deterministics = [False, True]

    combos = itertools.product(
        directions,
        head_dims,
        compute_output_dtype_tuples,
        disable_atomic_reductions,
        deterministics,
    )

    # prebuild the kernels in parallel for the determined options
    def _build_one(args):
        (
            direction,
            head_dim,
            compute_output_dtype_tuple,
            disable_atomic_reduction,
            deterministic,
        ) = args
        compute_dtype, output_dtype = compute_output_dtype_tuple
        spec, uri = get_ffa_jit_spec(
            arch=(9, 0),
            direction=direction,
            head_dim=head_dim,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
            softcap=False,
            disable_atomic_reduction=disable_atomic_reduction,
            deterministic=deterministic,
            # optional args below mainly for sparse attn
            ref_block_size=None,
            auto_range_merge=False,
            swap_ab=False,
            pack_gqa=False,
            qhead_per_khead=1,
            sparse_load=False,
            swap_bwd_qk_loop=False,
            profile_mode=False,
            return_max_logits=False,
        )
        spec.build()
        src_dir = (jit_env.MAGI_ATTENTION_JIT_DIR / uri).resolve()
        dst_dir = (jit_env.MAGI_ATTENTION_AOT_DIR / uri).resolve()
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        return uri

    with ThreadPoolExecutor(max_workers=PREBUILD_FFA_JOBS) as ex:
        futs = {ex.submit(_build_one, c): c for c in combos}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                uri = fut.result()
                print(f"Prebuilt: {uri}")
            except Exception as e:
                raise RuntimeError(f"Prebuild failed for {c}: {e}") from e


# build ext modules
ext_modules = []
if not SKIP_CUDA_BUILD:
    # define some paths for the ext modules below
    repo_dir = Path(project_root)
    csrc_dir = repo_dir / PACKAGE_NAME / "csrc"
    common_dir = csrc_dir / "common"
    extensions_dir = csrc_dir / "extensions"
    cutlass_dir = csrc_dir / "cutlass" / "include"

    # build magi attn ext module
    build_magi_attn_ext_module(
        csrc_dir=csrc_dir,
    )

    # optionally prebuild FFA JIT kernels (ref_block_size=None)
    prebuild_ffa_kernels()

    # init before building any ext module
    init_ext_modules()

    # build ffa utils ext module
    ffa_utils_ext_module = build_ffa_utils_ext_module(
        repo_dir=repo_dir,
        csrc_dir=csrc_dir,
        common_dir=common_dir,
    )
    if ffa_utils_ext_module is not None:
        ext_modules.append(ffa_utils_ext_module)

    # build magi attn comm module
    magi_attn_comm_module = build_magi_attn_comm_module(
        repo_dir=repo_dir,
        csrc_dir=csrc_dir,
        common_dir=common_dir,
        extensions_dir=extensions_dir,
        cutlass_dir=cutlass_dir,
    )
    if magi_attn_comm_module is not None:
        ext_modules.append(magi_attn_comm_module)
else:
    print(f"{title_left_str}Skipping CUDA build{title_right_str}")


# customize build extension
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


# init cmdclass
cmdclass = {"bdist_wheel": _bdist_wheel, "build_ext": MagiAttnBuildExtension}


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
            "scripts",
            "extensions",
            "examples",
        )
    ),
    # package data is defined in pyproject.toml
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
