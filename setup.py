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

import itertools
import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import warnings
from pathlib import Path

import torch
from packaging.version import Version, parse
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    COMMON_HIP_FLAGS,
    CUDA_HOME,
    IS_HIP_EXTENSION,
    IS_WINDOWS,
    SUBPROCESS_DECODE_ARGS,
    BuildExtension,
    CUDAExtension,
    _is_cuda_file,
    _join_cuda_home,
    _join_rocm_home,
    _maybe_write,
    get_cxx_compiler,
)
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "magi_attention"


# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files,
# without any cuda compilation
FORCE_BUILD = os.getenv("MAGI_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("MAGI_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("MAGI_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"


# TODO: remove flags to compile with sm80
# which we do not support for now
def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_cuda,
    **kwargs,  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append(f"cxx = {compiler}")
    if with_cuda or cuda_dlink_post_cflags:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home("bin", "hipcc")
        else:
            nvcc = _join_cuda_home("bin", "nvcc")
        if "PYTORCH_NVCC" in os.environ:
            nvcc_from_env = os.getenv(
                "PYTORCH_NVCC"
            )  # user can set nvcc compiler with ccache using the environment variable here
        else:
            nvcc_from_env = nvcc
        config.append(f"nvcc_from_env = {nvcc_from_env}")
        config.append(f"nvcc = {nvcc}")

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        cuda_post_cflags_sm80 = [
            s if s != "arch=compute_90a,code=sm_90a" else "arch=compute_80,code=sm_80"
            for s in cuda_post_cflags
        ]
        flags.append(f'cuda_post_cflags_sm80 = {" ".join(cuda_post_cflags_sm80)}')
        cuda_post_cflags_sm80_sm90 = cuda_post_cflags + [
            "-gencode",
            "arch=compute_80,code=sm_80",
        ]
        flags.append(
            f'cuda_post_cflags_sm80_sm90 = {" ".join(cuda_post_cflags_sm80_sm90)}'
        )
        cuda_post_cflags_sm100 = [
            s
            if s != "arch=compute_90a,code=sm_90a"
            else "arch=compute_100a,code=sm_100a"
            for s in cuda_post_cflags
        ]
        flags.append(f'cuda_post_cflags_sm100 = {" ".join(cuda_post_cflags_sm100)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    if IS_WINDOWS:
        compile_rule.append(
            "  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags"
        )
        compile_rule.append("  deps = msvc")
    else:
        compile_rule.append(
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        )
        compile_rule.append("  depfile = $out.d")
        compile_rule.append("  deps = gcc")

    if with_cuda:
        cuda_compile_rule = ["rule cuda_compile"]
        nvcc_gendeps = ""
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if (
            torch.version.cuda is not None
            and os.getenv("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "0") != "1"
        ):
            cuda_compile_rule.append("  depfile = $out.d")
            cuda_compile_rule.append("  deps = gcc")
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = (
                "--generate-dependencies-with-compile --dependency-output $out.d"
            )
        cuda_compile_rule_sm80 = (
            ["rule cuda_compile_sm80"]
            + cuda_compile_rule[1:]
            + [
                f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80"
            ]
        )
        cuda_compile_rule_sm80_sm90 = (
            ["rule cuda_compile_sm80_sm90"]
            + cuda_compile_rule[1:]
            + [
                f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80_sm90"
            ]
        )
        cuda_compile_rule_sm100 = (
            ["rule cuda_compile_sm100"]
            + cuda_compile_rule[1:]
            + [
                f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm100"
            ]
        )
        cuda_compile_rule.append(
            f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags"
        )

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith("_sm90.cu"):
                rule = "cuda_compile"
            elif source_file.endswith("_sm80.cu"):
                rule = "cuda_compile_sm80"
            elif source_file.endswith("_sm100.cu"):
                rule = "cuda_compile_sm100"
            else:
                rule = "cuda_compile_sm80_sm90"
        else:
            rule = "compile"
        if IS_WINDOWS:
            source_file = source_file.replace(":", "$:")
            object_file = object_file.replace(":", "$:")
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), "dlink.o")
        devlink_rule = ["rule cuda_devlink"]
        devlink_rule.append("  command = $nvcc $in -o $out $cuda_dlink_post_cflags")
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ["rule link"]
        if IS_WINDOWS:
            cl_paths = (
                subprocess.check_output(["where", "cl"])
                .decode(*SUBPROCESS_DECODE_ARGS)
                .split("\r\n")
            )
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(":", "$:")
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(
                f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out'
            )
        else:
            link_rule.append("  command = $cxx $in $ldflags -o $out")

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80_sm90)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm100)  # type: ignore[possibly-undefined]
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "-gencode arch=compute_sm90a,code=sm_90a" to files ending in '_sm90.cu',
# and pass "-gencode arch=compute_sm80,code=sm_80" to files ending in '_sm80.cu'
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
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
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


# Copied from https://github.com/triton-lang/triton/blob/main/python/setup.py
def get_magi_attention_cache_path():
    user_home = os.getenv("MAGI_ATTENTION_HOME")
    if not user_home:
        user_home = (
            os.getenv("HOME")
            or os.getenv("USERPROFILE")
            or os.getenv("HOMEPATH")
            or None
        )
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".magi_attention")


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


def download_and_copy(name, src_func, dst_path, version, url_func):
    magi_attention_cache_path = get_magi_attention_cache_path()
    base_dir = os.path.dirname(__file__)
    system = platform.system()
    arch = platform.machine()
    arch = {"arm64": "aarch64"}.get(arch, arch)
    supported = {"Linux": "linux", "Darwin": "linux"}
    url = url_func(supported[system], arch, version)
    src_path = src_func(supported[system], arch, version)
    tmp_path = os.path.join(
        magi_attention_cache_path, "nvidia", name
    )  # path to cache the download
    dst_path = os.path.join(
        base_dir, os.pardir, "third_party", "nvidia", "backend", dst_path
    )  # final binary path
    src_path = os.path.join(tmp_path, src_path)
    download = not os.path.exists(src_path)
    if download:
        print(f"downloading and extracting {url} ...")
        file = tarfile.open(fileobj=open_url(url), mode="r|*")
        file.extractall(path=tmp_path)
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    print(f"copy {src_path} to {dst_path} ...")
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy(src_path, dst_path)


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return ["--threads", nvcc_threads]


NVIDIA_TOOLCHAIN_VERSION = {"nvcc": "12.6.85", "ptxas": "12.8.93"}

exe_extension = sysconfig.get_config_var("EXE")


cmdclass = {}  # type: ignore[var-annotated]
ext_modules = []


if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none(PACKAGE_NAME)
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError("magi_attention is only supported on CUDA 12.3 and above")

    # ptxas 12.8 gives the best perf currently
    # We want to use the nvcc front end from 12.6 however, since if we use nvcc 12.8
    # Cutlass 3.8 will expect the new data types in cuda.h from CTK 12.8, which we don't have.
    if bare_metal_version != Version("12.8"):
        download_and_copy(
            name="nvcc",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin",
            dst_path="bin",
            version=NVIDIA_TOOLCHAIN_VERSION["nvcc"],
            url_func=lambda system, arch, version: f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",  # noqa: E501
        )
        download_and_copy(
            name="ptxas",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin/ptxas",
            dst_path="bin",
            version=NVIDIA_TOOLCHAIN_VERSION["ptxas"],
            url_func=lambda system, arch, version: f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",  # noqa: E501
        )
        download_and_copy(
            name="ptxas",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/nvvm/bin",
            dst_path="nvvm/bin",
            version=NVIDIA_TOOLCHAIN_VERSION["ptxas"],
            url_func=lambda system, arch, version: f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",  # noqa: E501
        )
        base_dir = os.path.dirname(__file__)
        ctk_path_new = os.path.abspath(
            os.path.join(base_dir, os.pardir, "third_party", "nvidia", "backend", "bin")
        )
        nvcc_path_new = os.path.join(ctk_path_new, f"nvcc{exe_extension}")
        # Need to append to path otherwise nvcc can't find cicc in nvvm/bin/cicc
        # nvcc 12.8 seems to hard-code looking for cicc in ../nvvm/bin/cicc
        os.environ["PATH"] = ctk_path_new + os.pathsep + os.environ["PATH"]
        os.environ["PYTORCH_NVCC"] = nvcc_path_new
        # Make nvcc executable, sometimes after the copy it loses its permissions
        os.chmod(nvcc_path_new, os.stat(nvcc_path_new).st_mode | stat.S_IEXEC)

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90a,code=sm_90a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    repo_dir = Path(this_dir)

    cutlass_dir = repo_dir / "magi_attention" / "csrc" / "cutlass"

    ffa_dir_abs = repo_dir / "magi_attention" / "csrc" / "flexible_flash_attention"
    ffa_dir_rel = ffa_dir_abs.relative_to(repo_dir)

    common_dir = repo_dir / "magi_attention" / "csrc" / "common"

    # custom flags
    DISABLE_HDIM64 = False
    DISABLE_HDIM96 = True
    DISABLE_HDIM128 = False
    DISABLE_HDIM192 = True
    DISABLE_HDIM256 = True
    DISABLE_FP16 = False
    DISABLE_BACKWARD = False
    DISABLE_SOFTCAP = False
    DISABLE_CLUSTER = False

    feature_args = (
        []
        + (["-DFLASHATTENTION_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DFLASHATTENTION_DISABLE_SOFTCAP"] if DISABLE_SOFTCAP else [])
        + (["-DFLASHATTENTION_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DFLASHATTENTION_DISABLE_CLUSTER"] if DISABLE_CLUSTER else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM96"] if DISABLE_HDIM96 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM192"] if DISABLE_HDIM192 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
    )

    DTYPE = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    HEAD_DIMENSIONS = ["all"]
    SOFTCAP = [""] + (["_softcap"] if not DISABLE_SOFTCAP else [])
    sources_fwd_sm90 = [
        f"{ffa_dir_rel}/instantiations/flash_fwd_hdim{hdim}_{dtype}{softcap}_sm90.cu"
        for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS, DTYPE, SOFTCAP)
    ]
    sources_bwd_sm90 = [
        f"{ffa_dir_rel}/instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm90.cu"
        for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS, DTYPE, SOFTCAP)
    ]
    if DISABLE_BACKWARD:
        sources_bwd_sm90 = []
    sources = [f"{ffa_dir_rel}/flash_api.cpp"] + sources_fwd_sm90 + sources_bwd_sm90
    sources += [f"{ffa_dir_rel}/fast_zero_fill.cu"]
    sources += [f"{ffa_dir_rel}/unique_consecutive_pairs.cu"]
    nvcc_flags = [
        "-O3",
        "-Xptxas",
        "-v",
        "-std=c++17",
        "--ftemplate-backtrace-limit=0",  # To debug template code
        "--use_fast_math",
        # "--keep",
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage", # printing out number of registers
        "--resource-usage",  # printing out number of registers
        # f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
        "-lineinfo",  # TODO: disable this for release to reduce binary size
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
    ]
    if get_platform() == "win_amd64":
        nvcc_flags.extend(
            [
                "-D_USE_MATH_DEFINES",  # for M_LN2
                "-Xcompiler=/Zc:__cplusplus",  # sets __cplusplus correctly, CUTLASS_CONSTEXPR_IF_CXX17 needed for cutlass::gcd
            ]
        )
    include_dirs = [
        common_dir,
        ffa_dir_abs,
        cutlass_dir / "include",
    ]

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"] + feature_args,
        "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
    }

    ext_modules.append(
        CUDAExtension(
            name="flexible_flash_attention_cuda",
            sources=sources,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
        )
    )

package_data = {
    "magi_attention": ["*.pyi", "**/*.pyi"],
}


setup(
    name="magi_attention",
    packages=find_packages(
        exclude=(
            "build",
            "tests",
            "dist",
            "docs",
            "tools",
            "assets",
        )
    ),
    package_data=package_data,
    include_package_data=True,
    py_modules=["magi_attention"],
    description="A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
