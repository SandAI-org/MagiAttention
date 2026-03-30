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
import warnings
from pathlib import Path
from typing import Callable, List, Optional

from filelock import FileLock
from torch.utils.cpp_extension import _import_module_from_library

from magi_attention.env.build import is_build_debug, is_build_verbose, is_force_jit_build, nvcc_threads

from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

os.makedirs(jit_env.MAGI_ATTENTION_WORKSPACE_DIR, exist_ok=True)
os.makedirs(jit_env.MAGI_ATTENTION_CSRC_DIR, exist_ok=True)

force_jit = is_force_jit_build()


logger = logging.getLogger(__name__)

# JIT-specific file handler: always append to a persistent log file so that
# JIT build history can be inspected even when stderr has scrolled away.
_jit_log_path = jit_env.MAGI_ATTENTION_WORKSPACE_DIR / "magi_attention_jit.log"
_jit_file_handler = logging.FileHandler(_jit_log_path)
_jit_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(_jit_file_handler)


def clear_cache_dir():
    if os.path.exists(jit_env.MAGI_ATTENTION_JIT_DIR):
        import shutil

        logger.info("Clearing JIT cache directory: %s", jit_env.MAGI_ATTENTION_JIT_DIR)
        shutil.rmtree(jit_env.MAGI_ATTENTION_JIT_DIR)


@dataclasses.dataclass
class JitSpec:
    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[str]]
    extra_objects_cb: Optional[Callable[[], list[str]]]
    extra_objects: Optional[list[str]] = None
    needs_device_linking: bool = False

    def __repr__(self) -> str:  # pragma: no cover
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
            f"  needs_device_linking={self.needs_device_linking}\n"
            ")"
        )

    @property
    def ninja_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_JIT_DIR / self.name / "build.ninja"

    @property
    def workspace_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_JIT_DIR / self.name

    @property
    def aot_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_AOT_DIR / self.name

    def write_ninja_if_different(self) -> bool:
        ninja_path = self.ninja_path
        logger.info("Writing ninja build file for '%s' at %s", self.name, ninja_path)
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
            extra_objects=self.extra_objects,
            needs_device_linking=self.needs_device_linking,
        )
        is_write = write_if_different(ninja_path, content)

        if is_write:
            logger.info("Ninja build file for '%s' was updated (content changed)", self.name)
        else:
            logger.info("Ninja build file for '%s' is up-to-date (no changes)", self.name)

        if "common" in self.name and is_write:
            warnings.warn(
                f"{self.name} build.ninja file has been changed, and it is build for "
                f"common files. Please check whether this behavior is correct."
            )

        return is_write

    def build(self) -> None:
        verbose = is_build_verbose()
        tmpdir = get_tmpdir()

        logger.info("Building JIT module '%s' (verbose=%s)", self.name, verbose)

        if self.extra_objects_cb is not None:
            logger.info("Resolving extra objects via callback for '%s'", self.name)
            self.extra_objects = self.extra_objects_cb()
            logger.info("Resolved %d extra object(s) for '%s'", len(self.extra_objects), self.name)

        with FileLock(tmpdir / f"{self.name}.lock", thread_local=False):
            logger.info("Acquired build lock for '%s'", self.name)
            self.write_ninja_if_different()
            run_ninja(
                jit_env.MAGI_ATTENTION_JIT_DIR / f"{self.name}",
                self.ninja_path,
                verbose,
            )
            logger.info("Ninja build completed for '%s'", self.name)

    def build_and_load(self) -> None:
        mod_name = self.name

        def _artifact_exists(lib_dir: Path, module_name: str) -> bool:
            for sfx in importlib.machinery.EXTENSION_SUFFIXES:
                if (lib_dir / f"{module_name}{sfx}").exists():
                    return True
            return False

        if (
            not force_jit
            and self.aot_path.exists()
            and _artifact_exists(self.aot_path, mod_name)
        ):
            lib_dir = self.aot_path
            logger.info("Loading AOT artifact for '%s' from %s", mod_name, lib_dir)
        else:
            logger.info("No AOT artifact for '%s', triggering JIT build", mod_name)
            self.build()
            lib_dir = self.workspace_path
            logger.info("JIT build done for '%s', loading from %s", mod_name, lib_dir)

        return _import_module_from_library(
            mod_name, str(lib_dir), is_python_module=True
        )

    def build_and_get_objects(self) -> List[str]:
        logger.info("Building and collecting object files for '%s'", self.name)
        self.build()
        objects = []

        object_file_path = self.workspace_path

        for common_obj in object_file_path.glob("*.o"):
            objects.append(str(common_obj.resolve()))

        logger.info("Collected %d object file(s) for '%s'", len(objects), self.name)
        return objects


def gen_jit_spec(
    name: str,
    sources: List[str],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    extra_objects_cb: Optional[Callable[[], list[str]]] = None,
    needs_device_linking: bool = False,
) -> JitSpec:
    debug = is_build_debug()
    verbose = is_build_verbose()

    logger.info(
        "Generating JitSpec '%s' (debug=%s, verbose=%s, sources=%d, device_link=%s)",
        name, debug, verbose, len(sources), needs_device_linking,
    )

    cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
        f"--split-compile={nvcc_threads()}",  # split-compile is faster
    ]
    if verbose or debug:
        cuda_cflags += [
            "-lineinfo",
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
        ]
    if debug:
        cuda_cflags += [
            "-g",
            "--keep",
            "--ftemplate-backtrace-limit=0",
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
        extra_objects_cb=extra_objects_cb,
        needs_device_linking=needs_device_linking,
    )

    logger.info("JitSpec '%s' created successfully", name)
    return spec


def get_tmpdir() -> Path:
    # TODO(lequn): Try /dev/shm first. This should help Lock on NFS.
    tmpdir = jit_env.MAGI_ATTENTION_JIT_DIR / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir
