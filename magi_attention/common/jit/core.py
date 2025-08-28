import dataclasses
import logging
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Union
import hashlib
import json
import importlib.machinery

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import _import_module_from_library
from filelock import FileLock

from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

os.makedirs(jit_env.MAGI_ATTENTION_WORKSPACE_DIR, exist_ok=True)
os.makedirs(jit_env.MAGI_ATTENTION_CSRC_DIR, exist_ok=True)


class FlashInferJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = jit_env.MAGI_ATTENTION_WORKSPACE_DIR / "magi_jit.log"
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


logger = FlashInferJITLogger("magi.jit")


def check_cuda_arch():
    # CUDA arch check (currently for FP8 readiness)
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")


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
    extra_include_dirs: Optional[List[Path]]
    is_class: bool = False
    needs_device_linking: bool = False

    @property
    def ninja_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_JIT_DIR / self.name / "build.ninja"

    @property
    def _meta_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_JIT_DIR / self.name / "version_meta.json"

    @property
    def jit_library_path(self) -> Path:
        # 固定产物名，不使用版本后缀
        return jit_env.MAGI_ATTENTION_JIT_DIR / self.name

    def get_library_path(self) -> Path:
        if self.aot_path.exists():
            return self.aot_path
        return self.jit_library_path

    @property
    def aot_path(self) -> Path:
        return jit_env.MAGI_ATTENTION_AOT_DIR / self.name

    def _compute_signature(self) -> str:
        def _file_sha256(p: Path) -> str:
            try:
                with open(p, "rb") as f:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                    return h.hexdigest()
            except Exception:
                try:
                    return f"mtime:{int(p.stat().st_mtime)}"
                except Exception:
                    return "missing"

        sig = {
            "sources": [{"path": str(p), "sha256": _file_sha256(p)} for p in self.sources],
            "extra_cflags": self.extra_cflags,
            "extra_cuda_cflags": self.extra_cuda_cflags,
            "extra_ldflags": self.extra_ldflags,
            "extra_include_dirs": [str(p) for p in (self.extra_include_dirs or [])],
        }
        return hashlib.sha256(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()

    def _update_signature(self) -> None:
        # Compute signature and track changes to decide if rebuild is necessary
        self.ninja_path.parent.mkdir(parents=True, exist_ok=True)
        new_hash = self._compute_signature()
        version = 0
        old_hash = None
        if self._meta_path.exists():
            try:
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                version = int(meta.get("version", 0))
                old_hash = meta.get("sig_hash")
            except Exception:
                pass
        sig_changed = new_hash != old_hash

        print(f"{old_hash=} {new_hash=} {sig_changed=}")
        if sig_changed:
            version += 1
            self._meta_path.write_text(json.dumps({"version": version, "sig_hash": new_hash}), encoding="utf-8")
        setattr(self, "__sig_changed", sig_changed)

    def write_ninja(self) -> None:
        ninja_path = self.ninja_path
        self._update_signature()
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
            run_ninja(jit_env.MAGI_ATTENTION_JIT_DIR, self.ninja_path, verbose)

    def build_and_load(self):
        force_build = os.environ.get("MAGI_ATTENTION_FORCE_BUILD", "0") == "1"
        verbose = os.environ.get("MAGI_ATTENTION_BUILD_VERBOSE", "0") == "1"
        mod_name = self.name

        def _artifact_exists(lib_dir: Path, module_name: str) -> bool:
            for sfx in importlib.machinery.EXTENSION_SUFFIXES:
                if (lib_dir / f"{module_name}{sfx}").exists():
                    return True
            return False

        if force_build:
            self.write_ninja()
            lib_dir = self.jit_library_path
            self.build(verbose)
        elif self.aot_path.exists() and _artifact_exists(self.aot_path, mod_name):
            lib_dir = self.aot_path
        else:
            # Write ninja and decide whether to rebuild based on signature
            self.write_ninja()
            lib_dir = self.jit_library_path
            sig_changed = getattr(self, "__sig_changed", True)
            if sig_changed or not _artifact_exists(lib_dir, mod_name):
                self.build(verbose)

        return _import_module_from_library(mod_name, str(lib_dir), is_python_module=True)

def gen_jit_spec(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
    needs_device_linking: bool = False,
) -> JitSpec:
    check_cuda_arch()
    debug = os.environ.get("MAGI_ATTENTION_BUILD_DEBUG", "0") == "1"

    cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90", # For PDL
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", # Necessary for the WGMMA shapes that we use
        f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
    ]
    if debug:
        cuda_cflags += [
            "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",
            "--keep",
            "--ftemplate-backtrace-limit=0",
            "-Xptxas",
            "-v",
            "-g",
            "-lineinfo",
            "--ptxas-options=-v",
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=2",
            "--resource-usage",  # printing out number of registers
        ]
    else:
        # non debug mode
        cuda_cflags += ["-DNDEBUG"]

    if extra_cflags is not None:
        cflags += extra_cflags
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags
    if extra_include_paths is not None:
        extra_include_paths = [Path(x) for x in extra_include_paths]
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
    spec.write_ninja()
    return spec


def get_tmpdir() -> Path:
    # TODO(lequn): Try /dev/shm first. This should help Lock on NFS.
    tmpdir = jit_env.MAGI_ATTENTION_JIT_DIR / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir
