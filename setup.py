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
import importlib
import importlib.resources
import itertools
import os
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from packaging.version import Version, parse
from setuptools import find_namespace_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Note: ninja build requires include_dirs to be absolute paths
this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "magi_attention"
NVIDIA_TOOLCHAIN_VERSION = {"nvcc": "12.6.85", "ptxas": "12.8.93"}
exe_extension = sysconfig.get_config_var("EXE")
USER_HOME = os.getenv("MAGI_ATTENTION_HOME")

# For CI: allow forcing C++11 ABI to match NVCR images that use C++11 ABI
FORCE_CXX11_ABI = os.getenv("MAGI_ATTENTION_FORCE_CXX11_ABI", "0") == "1"

# Skip building CUDA extension modules
SKIP_CUDA_BUILD = os.getenv("MAGI_ATTENTION_SKIP_CUDA_BUILD", "0") == "1"

# NOTE: this flag now only used for magi_attn_comm to disable sm90 features
# to be compatible with other architectures such as sm80
DISABLE_SM90_FEATURES = os.getenv("MAGI_ATTENTION_DISABLE_SM90_FEATURES", "0") == "1"

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
SKIP_MAGI_ATTN_COMM_BUILD = (
    os.getenv("MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD", "0") == "1"
)

os.environ["MAGI_ATTENTION_BUILD_VERBOSE"] = "1"


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


# Copied from https://github.com/triton-lang/triton/blob/main/python/setup.py
def get_magi_attention_cache_path() -> str:
    user_home = USER_HOME
    if not user_home:
        user_home = (
            os.getenv("HOME")
            or os.getenv("USERPROFILE")
            or os.getenv("HOMEPATH")
            or None
        )
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, f".{PACKAGE_NAME}")


# Copied from https://github.com/deepseek-ai/DeepEP/blob/main/setup.py
# Wheel specific: The wheels only include the soname of the host library (libnvshmem_host.so.X)
def get_nvshmem_host_lib_name():
    for path in importlib.resources.files("nvidia.nvshmem").iterdir():
        for file in path.rglob("libnvshmem_host.so.*"):
            return file.name
    raise ModuleNotFoundError("libnvshmem_host.so not found")


def open_url(url):
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
    )
    headers = {
        "User-Agent": user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forever.
    return urllib.request.urlopen(request, timeout=300)


def nvcc_threads_args() -> list[str]:
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return ["--threads", nvcc_threads]


def init_ext_modules() -> None:
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


def to_full_ext_module_name(ext_module_name: str) -> str:
    return f"{PACKAGE_NAME}.{ext_module_name}"


def build_ffa_utils_ext_module(
    repo_dir: Path,
    csrc_dir: Path,
    common_dir: Path,
) -> CUDAExtension | None:
    ext_module_name = "flexible_flash_attention_utils_cuda"

    if SKIP_FFA_UTILS_BUILD:
        return None

    print(
        "\n# -------------------     Building flexible_flash_attention_utils_cuda     ------------------- #\n"
    )

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

    return CUDAExtension(
        name=to_full_ext_module_name(ext_module_name),
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
    )


def build_magi_attn_ext_module(
    repo_dir: Path,
    csrc_dir: Path,
    common_dir: Path,
) -> CUDAExtension | None:
    ext_module_name = "magi_attn_ext"

    if SKIP_MAGI_ATTN_EXT_BUILD:
        return None

    print(
        "\n# -------------------     Building magi_attn_ext     ------------------- #\n"
    )

    magi_attn_ext_dir_abs = csrc_dir / "extensions"

    # init sources
    cpp_files = glob.glob(str(magi_attn_ext_dir_abs / "*.cpp"))
    sources = [str(Path(f).relative_to(repo_dir)) for f in cpp_files]

    # init include dirs
    include_dirs = [common_dir, magi_attn_ext_dir_abs]

    # init extra compile args
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}

    return CUDAExtension(
        name=to_full_ext_module_name(ext_module_name),
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
    )


def build_magi_attn_comm_module(
    repo_dir: Path,
    csrc_dir: Path,
    common_dir: Path,
) -> CUDAExtension | None:
    ext_module_name = "magi_attn_comm"

    if SKIP_MAGI_ATTN_COMM_BUILD:
        return None

    print(
        "\n# -------------------     Building magi_attn_comm     ------------------- #\n"
    )

    # ---   for grpcoll submodule   --- #

    # find nvshmem
    disable_nvshmem = False
    nvshmem_dir = os.getenv("NVSHMEM_DIR", None)
    nvshmem_host_lib = "libnvshmem_host.so"
    if nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec(  # type: ignore[union-attr,index]
                "nvidia.nvshmem"
            ).submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name()
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            warnings.warn(
                "`NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. "
                "All internode and low-latency features are disabled\n"
            )
            disable_nvshmem = True
    else:
        disable_nvshmem = False

    if not disable_nvshmem:
        assert os.path.exists(
            nvshmem_dir  # type: ignore[arg-type]
        ), f"The specified NVSHMEM directory does not exist: {nvshmem_dir}"

    magi_attn_comm_dir_abs = csrc_dir / "comm"
    grpcoll_dir_abs = magi_attn_comm_dir_abs / "grpcoll"
    grpcoll_dir_rel = grpcoll_dir_abs.relative_to(repo_dir)

    # init sources
    sources = [
        f"{grpcoll_dir_rel}/buffer.cpp",
        f"{grpcoll_dir_rel}/kernels/runtime.cu",
        f"{grpcoll_dir_rel}/kernels/layout.cu",
        f"{grpcoll_dir_rel}/kernels/intranode.cu",
    ]

    # init include dirs
    include_dirs = [common_dir, grpcoll_dir_abs]

    # init extra compile args
    cxx_flags = [
        "-O3",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-attributes",
    ]
    nvcc_flags = [
        "-O3",
        "-Xcompiler",
        "-O3",
        "-gencode",
        "arch=compute_90,code=sm_90",
    ]  # Explicitly specify sm_90

    # extend flags, dirs and args
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = []
    if disable_nvshmem:
        cxx_flags.append("-DDISABLE_NVSHMEM")
        nvcc_flags.append("-DDISABLE_NVSHMEM")
    else:
        sources.extend(
            [
                f"{grpcoll_dir_rel}/kernels/internode.cu",
                f"{grpcoll_dir_rel}/kernels/internode_ll.cu",
            ]
        )
        include_dirs.extend([f"{nvshmem_dir}/include"])  # type: ignore[list-item]
        library_dirs.extend([f"{nvshmem_dir}/lib"])
        nvcc_dlink.extend(["-dlink", f"-L{nvshmem_dir}/lib", "-lnvshmem_device"])
        extra_link_args.extend(
            [
                f"-l:{nvshmem_host_lib}",
                "-l:libnvshmem_device.a",
                f"-Wl,-rpath,{nvshmem_dir}/lib",
            ]
        )

    if DISABLE_SM90_FEATURES:
        # Prefer A100
        os.environ["TORCH_CUDA_ARCH_LIST"] = os.getenv("TORCH_CUDA_ARCH_LIST", "8.0")

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append("-DDISABLE_SM90_FEATURES")
        nvcc_flags.append("-DDISABLE_SM90_FEATURES")

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        # Prefer H800 series
        os.environ["TORCH_CUDA_ARCH_LIST"] = os.getenv("TORCH_CUDA_ARCH_LIST", "9.0")

        # CUDA 12 flags
        nvcc_flags.extend(["-rdc=true", "--ptxas-options=--register-usage-level=10"])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ["TORCH_CUDA_ARCH_LIST"].strip() != "9.0":
        assert int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", 1)) == 1
        os.environ["DISABLE_AGGRESSIVE_PTX_INSTRS"] = "1"

    # Disable aggressive PTX instructions
    if int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "1")):
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

    # Put them together
    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args["nvcc_dlink"] = nvcc_dlink

    return CUDAExtension(
        name=to_full_ext_module_name(ext_module_name),
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        sources=sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


# build ext modules
ext_modules = []
if not SKIP_CUDA_BUILD:
    # init before building any ext module
    init_ext_modules()

    # define some paths for the ext modules below
    repo_dir = Path(this_dir)
    csrc_dir = repo_dir / PACKAGE_NAME / "csrc"
    common_dir = csrc_dir / "common"
    cutlass_dir = csrc_dir / "cutlass"

    # build ffa utils ext module
    ffa_utils_ext_module = build_ffa_utils_ext_module(
        repo_dir=repo_dir,
        csrc_dir=csrc_dir,
        common_dir=common_dir,
    )
    if ffa_utils_ext_module is not None:
        ext_modules.append(ffa_utils_ext_module)

    # build magi attn ext module
    magi_attn_ext_module = build_magi_attn_ext_module(
        repo_dir=repo_dir,
        csrc_dir=csrc_dir,
        common_dir=common_dir,
    )
    if magi_attn_ext_module is not None:
        ext_modules.append(magi_attn_ext_module)

    # build magi attn ext module
    magi_attn_comm_module = build_magi_attn_comm_module(
        repo_dir=repo_dir,
        csrc_dir=csrc_dir,
        common_dir=common_dir,
    )
    if magi_attn_comm_module is not None:
        ext_modules.append(magi_attn_comm_module)


def prebuild_ffa_kernels() -> None:
    print(
        "\n# -------------------     Prebuilding FFA JIT kernels (ref_block_size=None)     ------------------- #\n"
    )

    # During build time, the package isn't installed yet. Fall back to source tree import.
    project_root = os.path.dirname(os.path.abspath(__file__))
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


class MagiAttnBuildExtension(BuildExtension):
    """
    A BuildExtension that switches its behavior based on the command.

    - For development installs (`pip install -e .`), it caches build artifacts
      in the local `./build` directory for faster re-compilation.

    - For building a distributable wheel (`python -m build --wheel`), it uses
      the default temporary directory behavior of PyTorch's BuildExtension to
      ensure robust and correct packaging.
    """

    def initialize_options(self):
        super().initialize_options()

        # Before core extensions are built,
        # optionally prebuild FFA JIT kernels (ref_block_size=None)
        if not SKIP_CUDA_BUILD and PREBUILD_FFA:
            prebuild_ffa_kernels()

        # Core logic: check if wheel build is running. 'bdist_wheel' is triggered by `python -m build`.
        if "bdist_wheel" not in sys.argv:
            # If not building a wheel (i.e., dev install like `pip install -e .`), enable local caching
            print("Development mode detected: Caching build artifacts in build/")
            project_root = os.path.dirname(os.path.abspath(__file__))
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

    def build_extensions(self):
        super().build_extensions()


# init cmdclass
cmdclass = {"bdist_wheel": _bdist_wheel, "build_ext": MagiAttnBuildExtension}


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
