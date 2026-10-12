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

import hashlib
import itertools
import logging
import math
import multiprocessing
import os
import pickle
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import cutlass.cute as cute
import torch
from tqdm import tqdm

from magi_attention.common import AttnRanges
from magi_attention.env import ffa as ffa_env
from magi_attention.meta.collection.calc_meta import FA4AttnArg

logger = logging.getLogger(__name__)

try:
    from flash_attn_cute.interface import (
        _bwd_postprocess_convert,
        _bwd_preprocess,
        _flash_attn_bwd,
        _flash_attn_fwd,
    )
except ImportError:
    is_fa4_installed = False
else:
    is_fa4_installed = True


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
FFA_FA4_CACHE_DIR = os.environ.get(
    ffa_env.FA4_CACHE_DIR,
    os.path.join(parent_dir, "lib", "ffa_fa4_cache"),
)
KERNEL_SYMBOL_NAME = "cached_kernel_func"
_KERNEL_FUNC_NAME_FILE = "func_name.txt"
_KERNEL_KEY_FILE = "compiled_key.pkl"
_KERNEL_OBJ_FILE = "kernel_obj.o"
_KERNEL_LIB_FILE = "kernel_lib.so"


COMPILED_META_DICT = {
    "fwd": {"cache_dict": _flash_attn_fwd.compile_cache},
    "bwd": {"cache_dict": _flash_attn_bwd.compile_cache},
    "bwd_pre": {"cache_dict": _bwd_preprocess.compile_cache},
    "bwd_post": {"cache_dict": _bwd_postprocess_convert.compile_cache},
}


def load_precompiled_ffa_fa4():
    assert (
        is_fa4_installed
    ), "FlashAttn4 is not installed, cannot load pre-compiled kernels"

    logger.info(f"Loading pre-compiled FFA_FA4 kernels from {FFA_FA4_CACHE_DIR} ...")

    has_kernel_loaded = False
    for compiled_cache_name, compiled_meta in COMPILED_META_DICT.items():
        dir_path = os.path.join(FFA_FA4_CACHE_DIR, compiled_cache_name)
        if not os.path.exists(dir_path):
            logger.info(f"\t=> {compiled_cache_name}: 0 kernels loaded")
            continue

        cache_dict = compiled_meta["cache_dict"]

        for kernel_folder in os.listdir(dir_path):
            folder = os.path.join(dir_path, kernel_folder)
            key_path = os.path.join(folder, _KERNEL_KEY_FILE)
            so_path = os.path.join(folder, _KERNEL_LIB_FILE)

            if os.path.exists(key_path) and os.path.exists(so_path):
                with open(key_path, "rb") as f:
                    key = pickle.load(f)

                func_name_path = os.path.join(folder, _KERNEL_FUNC_NAME_FILE)
                if os.path.exists(func_name_path):
                    with open(func_name_path, "r") as f:
                        func_name = f.read().strip()
                else:
                    func_name = KERNEL_SYMBOL_NAME

                mod = cute.runtime.load_module(so_path, enable_tvm_ffi=True)
                # flash_attn's interface calls cached kernels positionally and
                # the stream is an implicit TVM FFI env argument, so the raw
                # exported function is stored as-is, the same way flash_attn's
                # own JITPersistentCache does.
                cache_dict[key] = getattr(mod, func_name)

        num_loaded = len(getattr(cache_dict, "cache", cache_dict))
        logger.info(f"\t=> {compiled_cache_name}: {num_loaded} kernels loaded")
        has_kernel_loaded = has_kernel_loaded or num_loaded > 0

    if not has_kernel_loaded:
        logger.info("No pre-compiled FFA_FA4 kernels to load.")
    else:
        logger.info("Pre-compiled FFA_FA4 kernels loaded successfully.")


# Sequence lengths affect the compile keys: fwd uses q_stage, and bwd
# records whether Q and K each fit in one block.
# Use short and long mock inputs to cover common values of these keys.
# The K length is the unit below multiplied by func_num.
# Different lengths can reuse a kernel if their compile keys match.
# Calls with keys outside this grid trigger JIT compilation.
_PRECOMPILE_SEQLEN_Q_LENGTHS = (1024, 128)
_PRECOMPILE_SEQLEN_K_UNITS = (256, 32)

FA4PrecompileConfig = tuple[torch.dtype, int, int, int, int]
"""(dtype, head_dim, head_dim_v, qhead_per_kvhead, func_num)"""


def _compile_ffa_fa4_config(config: FA4PrecompileConfig, device: int) -> None:
    """Run the FFA_FA4 fwd and bwd on mock inputs to JIT-compile the kernels.

    The mock FA4AttnArg has the same head dims, GQA ratio and default tile
    sizes as the FA4AttnArg that ``CalcMeta`` makes for a real call. For the
    sequence lengths, see the comment at ``_PRECOMPILE_SEQLEN_Q_LENGTHS``.
    """
    from magi_attention.functional.fa4 import fa4_bwd, fa4_fwd

    dtype, head_dim, head_dim_v, qhead_per_kvhead, func_num = config
    softmax_scale = 1.0 / math.sqrt(head_dim)
    for seq_q, seq_k_unit in itertools.product(
        _PRECOMPILE_SEQLEN_Q_LENGTHS, _PRECOMPILE_SEQLEN_K_UNITS
    ):
        seq_k = seq_k_unit * func_num
        if func_num == 1:
            k_ranges = AttnRanges.from_ranges([(0, seq_k)])
        else:
            k_ranges = AttnRanges.from_ranges(
                [(i * seq_k_unit, (i + 1) * seq_k_unit) for i in range(1, func_num, 2)]
            )
        attn_arg = FA4AttnArg(
            q_ranges=AttnRanges.from_ranges([(0, seq_q)] * len(k_ranges)),
            k_ranges=k_ranges,
            attn_type_map=[0] * len(k_ranges),
            seqlen_q=seq_q,
            seqlen_k=seq_k,
            headdim=head_dim,
            headdim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
        )
        assert (
            attn_arg.n_func == func_num
        ), f"Mismatch in function number for attn_arg, expected {func_num}, got {attn_arg.n_func}"

        nhkv = 1
        nhq = nhkv * qhead_per_kvhead
        q = torch.empty((seq_q, nhq, head_dim), dtype=dtype, device=device)
        k = torch.empty((seq_k, nhkv, head_dim), dtype=dtype, device=device)
        v = torch.empty((seq_k, nhkv, head_dim_v), dtype=dtype, device=device)
        o = torch.empty((seq_q, nhq, head_dim_v), dtype=dtype, device=device)
        do = torch.empty_like(o)
        lse = torch.empty((seq_q, nhq), dtype=torch.float32, device=device)

        fa4_fwd(
            q=q,
            k=k,
            v=v,
            sink=None,
            attn_arg=attn_arg,
            softmax_scale=softmax_scale,
            softcap=0.0,
        )
        fa4_bwd(
            do=do,
            q=q,
            k=k,
            v=v,
            sink=None,
            o=o,
            lse=lse,
            attn_arg=attn_arg,
            softmax_scale=softmax_scale,
            softcap=0.0,
        )


def _export_compiled_ffa_fa4(runtime_libs: list[str]) -> None:
    """Export each kernel in the in-memory compile caches to its kernel dir.

    Many workers can export the same kernel at the same time. For example, many
    grid points use the same bwd_pre and bwd_post kernels. All workers write
    the same content for one kernel. Thus the result does not depend on the
    worker that writes last.

    Each file goes to a unique temporary name in the kernel dir first.
    ``os.replace`` then moves it to its final name. A reader never sees a part
    of a file.

    ``load_precompiled_ffa_fa4`` reads a kernel dir only if the key file and
    the lib file exist. It also reads the func name file. Thus the key file
    moves to its final name last.

    The loader reads only the final file names. Temporary files from a crashed
    worker have no effect on the loader.

    A rerun writes all files of its kernels again and replaces the old files.
    """
    for compiled_cache_name, compiled_meta in COMPILED_META_DICT.items():
        # flash_attn's compile caches are JITCache wrappers around a plain
        # dict; export walks the dict (the wrapper has no len/items).
        compiled_cache = compiled_meta["cache_dict"]
        compiled_cache = getattr(compiled_cache, "cache", compiled_cache)
        this_cached_dir = os.path.join(FFA_FA4_CACHE_DIR, compiled_cache_name)

        for compiled_key, kernel in compiled_cache.items():
            hash = int(
                hashlib.sha256(
                    f"{compiled_cache_name}_{compiled_key}".encode("utf-8")
                ).hexdigest(),
                16,
            )
            kernel_cached_dir = os.path.join(this_cached_dir, str(hash))
            os.makedirs(kernel_cached_dir, exist_ok=True)

            # The temporary names keep the file extension, because gcc selects
            # the input type from the extension. The order is the move order.
            tmp_prefix = f".{uuid.uuid4().hex}."
            file_names = (
                _KERNEL_FUNC_NAME_FILE,
                _KERNEL_OBJ_FILE,
                _KERNEL_LIB_FILE,
                _KERNEL_KEY_FILE,
            )
            tmp_paths = {
                name: os.path.join(kernel_cached_dir, tmp_prefix + name)
                for name in file_names
            }
            func_name = f"kernel_{hash}"
            try:
                with open(tmp_paths[_KERNEL_FUNC_NAME_FILE], "w") as f:
                    f.write(func_name)
                obj_path = tmp_paths[_KERNEL_OBJ_FILE]
                kernel.export_to_c(obj_path, function_name=func_name)
                so_path = tmp_paths[_KERNEL_LIB_FILE]
                cmd = ["gcc", "-shared", "-fPIC", "-o", so_path, obj_path]
                subprocess.run([*cmd, *runtime_libs], check=True)
                with open(tmp_paths[_KERNEL_KEY_FILE], "wb") as f:
                    pickle.dump(compiled_key, f)
                for name in file_names:
                    os.replace(tmp_paths[name], os.path.join(kernel_cached_dir, name))
            finally:
                for tmp_path in tmp_paths.values():
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            logger.info(f"\t=> Exported: {kernel_cached_dir}")


def _precompile_ffa_fa4_worker(configs: list[FA4PrecompileConfig], device: int) -> None:
    """Compile and export one shard of the configs on one device."""
    torch.cuda.set_device(device)
    runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
    # The import of magi_attention.functional loads the kernels in the cache
    # dir into the in-memory caches. Clear the in-memory caches. Then this
    # worker compiles and exports these kernels again, and a rerun replaces
    # old kernels.
    for compiled_meta in COMPILED_META_DICT.values():
        compiled_meta["cache_dict"].clear()
    for config in configs:
        _compile_ffa_fa4_config(config, device)
    _export_compiled_ffa_fa4(runtime_libs)


def precompile_ffa_fa4(
    dtypes: list[torch.dtype],
    head_dims: list[tuple[int, int]],
    qhead_per_kvhead: list[int],
    func_nums: list[int],
    num_workers: int = 1,
) -> None:
    """Pre-compile FFA_FA4 kernels for a grid of cases into ``FFA_FA4_CACHE_DIR``.

    Args:
        dtypes: The input dtypes.
        head_dims: The ``(head_dim, head_dim_v)`` pairs, for example ``(192, 128)``.
        qhead_per_kvhead: The GQA ratios.
        func_nums: The numbers of arbitrary-mask functions. Use odd numbers
            (see ``FA4AttnArg``).
        num_workers: The number of worker processes. Compilation uses mostly
            the CPU. The workers use the visible CUDA devices in turn, and the
            devices only run the mock launches. If the value is ``1``, this
            process compiles all cases.

    A rerun replaces the kernels of the grid in the cache dir. The kernels
    outside the grid stay in the cache dir.
    """
    assert is_fa4_installed, "FlashAttn4 is not installed, cannot pre-compile kernels"
    assert num_workers >= 1, f"num_workers must be positive, got {num_workers}"

    configs: list[FA4PrecompileConfig] = [
        (dtype, head_dim, head_dim_v, nhg, func_num)
        for dtype, (head_dim, head_dim_v), nhg, func_num in itertools.product(
            dtypes, head_dims, qhead_per_kvhead, func_nums
        )
    ]
    if not configs:
        logger.info("No FFA_FA4 kernels to pre-compile.")
        return

    num_workers = min(num_workers, len(configs))
    if num_workers == 1:
        _precompile_ffa_fa4_worker(configs, torch.cuda.current_device())
    else:
        num_devices = torch.cuda.device_count()
        assert num_devices > 0, "pre-compiling FFA_FA4 kernels needs a CUDA device"
        # Each worker takes every num_workers-th config. Thus the configs with
        # large head dims, which compile slowly, go to different workers.
        shards = [configs[i::num_workers] for i in range(num_workers)]
        # Use spawn. A forked child process cannot initialize CUDA again.
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(_precompile_ffa_fa4_worker, shard, i % num_devices)
                for i, shard in enumerate(shards)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Pre-compiling FFA_FA4 kernels",
                dynamic_ncols=True,
                unit="worker",
            ):
                future.result()

    logger.info(
        f"FFA_FA4 kernels pre-compiled successfully: {len(configs)} cases "
        f"exported to {FFA_FA4_CACHE_DIR}."
    )
