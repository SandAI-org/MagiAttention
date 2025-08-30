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

# Copyright (c) 2024 by FlashInfer team.
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

import dataclasses
import importlib.machinery
import logging
import os
from pathlib import Path
from typing import List, Optional

from filelock import FileLock
from torch.utils.cpp_extension import _import_module_from_library

from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

os.makedirs(jit_env.MAGI_ATTENTION_WORKSPACE_DIR, exist_ok=True)
os.makedirs(jit_env.MAGI_ATTENTION_CSRC_DIR, exist_ok=True)


class MagiAttentionJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = jit_env.MAGI_ATTENTION_WORKSPACE_DIR / "magi_attention_jit.log"
        if not os.path.exists(log_path):
            # create an empty file
            with open(log_path, "w") as f:  # noqa: F841
                pass
        self.addHandler(logging.FileHandler(log_path))
        # Configure log format
        self.handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.handlers[1].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def info(self, msg):
        super().info("magi.jit: " + msg)


logger = MagiAttentionJITLogger("magi.jit")


def clear_cache_dir():
    if os.path.exists(jit_env.MAGI_ATTENTION_JIT_DIR):
        import shutil

        shutil.rmtree(jit_env.MAGI_ATTENTION_JIT_DIR)


@dataclasses.dataclass
class JitSpec:
    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[str]]
    is_class: bool = False
    needs_device_linking: bool = False

    def __repr__(self):
        def _fmt_list(values, indent: str = "  ", max_items: int = 8) -> str:
            if values is None:
                return "None"
            if isinstance(values, list) and len(values) == 0:
                return "[]"
            items = [str(v) for v in values]
            display_items = items[:max_items]
            if len(items) > max_items:
                display_items.append(f"... (+{len(items) - max_items} more)")
            joined = (",\n" + indent + "  ").join(display_items)
            return "[\n" + indent + "  " + joined + "\n" + indent + "]"

        return (
            "JitSpec(\n"
            f"  name={self.name!r},\n"
            f"  sources={_fmt_list(self.sources, indent='  ')},\n"
            f"  extra_cflags={_fmt_list(self.extra_cflags, indent='  ')},\n"
            f"  extra_cuda_cflags={_fmt_list(self.extra_cuda_cflags, indent='  ')},\n"
            f"  extra_ldflags={_fmt_list(self.extra_ldflags, indent='  ')},\n"
            f"  extra_include_dirs={_fmt_list(self.extra_include_dirs, indent='  ')},\n"
            f"  is_class={self.is_class},\n"
            f"  needs_device_linking={self.needs_device_linking}\n"
            ")"
        )

    @property
    def ninja_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_JIT_DIR / self.name / "build.ninja"

    @property
    def jit_library_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_JIT_DIR / self.name

    @property
    def aot_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_AOT_DIR / self.name

    def write_ninja(self) -> None:
        ninja_path = self.ninja_path
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
            needs_device_linking=self.needs_device_linking,
        )
        write_if_different(ninja_path, content)

    def build(self, verbose: bool) -> None:
        tmpdir = get_tmpdir()
        with FileLock(tmpdir / f"{self.name}.lock", thread_local=False):
            self.write_ninja()
            run_ninja(
                jit_env.MAGI_ATTENTION_JIT_DIR / f"{self.name}",
                self.ninja_path,
                verbose,
            )

    def build_and_load(self):
        verbose = os.environ.get("MAGI_ATTENTION_BUILD_VERBOSE", "0") == "1"
        mod_name = self.name

        def _artifact_exists(lib_dir: Path, module_name: str) -> bool:
            for sfx in importlib.machinery.EXTENSION_SUFFIXES:
                if (lib_dir / f"{module_name}{sfx}").exists():
                    return True
            return False

        if self.aot_path.exists() and _artifact_exists(self.aot_path, mod_name):
            lib_dir = self.aot_path
        else:
            self.build(verbose)
            lib_dir = self.jit_library_path

        return _import_module_from_library(
            mod_name, str(lib_dir), is_python_module=True
        )


def gen_jit_spec(
    name: str,
    sources: List[str],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    needs_device_linking: bool = False,
) -> JitSpec:
    debug = os.environ.get("MAGI_ATTENTION_BUILD_DEBUG", "0") == "1"

    cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
        f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
    ]
    if debug:
        cuda_cflags += [
            "-g",
            "--keep",
            "-lineinfo",
            "--ftemplate-backtrace-limit=0"
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
            "--resource-usage",  # printing out number of registers
            "-DCUTLASS_DEBUG_TRACE_LEVEL=2",
        ]
    else:
        # non debug mode
        cuda_cflags += ["-DNDEBUG"]

    if extra_cflags is not None:
        cflags += extra_cflags
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags
    sources = [Path(x) for x in sources]

    spec = JitSpec(
        name=name,
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_dirs=extra_include_paths,
        needs_device_linking=needs_device_linking,
    )

    return spec


def get_tmpdir() -> Path:
    # TODO(lequn): Try /dev/shm first. This should help Lock on NFS.
    tmpdir = jit_env.MAGI_ATTENTION_JIT_DIR / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir
