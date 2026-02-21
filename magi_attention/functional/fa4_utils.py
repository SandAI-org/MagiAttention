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

import os
import hashlib
import itertools
import math
import os
import pickle
import subprocess

import cutlass.cute as cute
import torch
from tqdm import tqdm
from tvm_ffi.utils import kwargs_wrapper

from magi_attention.common import AttnRanges
from magi_attention.meta.collection.calc_meta import FA4AttnArg

is_fa4_installed = False
try:
    from flash_attn_cute.interface import _flash_attn_bwd, _flash_attn_fwd

    is_fa4_installed = True
except ImportError:
    pass


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
FFA_FA4_CACHE_DIR = os.path.join(parent_dir, "lib", "ffa_fa4_cache")
FFA_FA4_CACHE_DIR = os.environ.get("MAGI_ATTENTION_FFA_FA4_CACHE_DIR", FFA_FA4_CACHE_DIR)
KERNEL_SYMBOL_NAME = "cached_kernel_func"


COMPILED_META_DICT = {
    "fwd": {
        "cache_dict": _flash_attn_fwd.compile_cache,
        "arg_names": [
            "q",
            "k",
            "v",
            "out",
            "lse",
            "softmax_scale",
            "current_stream",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "seqused_q",
            "seqused_k",
            "page_table",
            "window_size_left",
            "window_size_right",
            "learnable_sink",
            "blocksparse_tensors",
            "aux_tensors",
        ],
    },
    "bwd": {
        "cache_dict": _flash_attn_bwd.compile_cache,
        "arg_names": [
            "q",
            "k",
            "v",
            "dout",
            "lse_log2",
            "dpsum",
            "dq_accum",
            "dk_or_accum",
            "dv_or_accum",
            "softmax_scale",
            "current_stream",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "seqused_q",
            "seqused_k",
            "blocksparse_tensors",
            "aux_tensors",
            "mdQ_semaphore",
            "mdK_semaphore",
            "mdV_semaphore",
        ],
    },
    "bwd_pre": {
        "cache_dict": _flash_attn_bwd.compile_cache_pre,
        "arg_names": [
            "out",
            "dout",
            "dpsum",
            "lse",
            "lse_log2",
            "dq_accum",
            "cu_seqlens_q",
            "seqused_q",
            "current_stream",
        ],
    },
    "bwd_post": {
        "cache_dict": _flash_attn_bwd.compile_cache_post,
        "arg_names": [
            "accum",
            "out",
            "softmax_scale",
            "cu_seqlens",
            "seqused",
            "current_stream",
        ],
    },
}


def load_precompiled_ffa_fa4():
    assert (
        is_fa4_installed
    ), "FlashAttn4 is not installed, cannot load pre-compiled kernels"

    print(
        f"Loading pre-compiled FFA_FA4 kernels from {FFA_FA4_CACHE_DIR} ...",
        flush=True,
    )

    has_kernel_loaded = False
    for compiled_cache_name, compiled_meta in COMPILED_META_DICT.items():
        dir_path = os.path.join(FFA_FA4_CACHE_DIR, compiled_cache_name)
        if not os.path.exists(dir_path):
            print(f"\t=> {compiled_cache_name}: 0 kernels loaded", flush=True)
            continue

        cache_dict = compiled_meta["cache_dict"]
        arg_names = compiled_meta["arg_names"]

        for kernel_folder in os.listdir(dir_path):
            folder = os.path.join(dir_path, kernel_folder)
            key_path = os.path.join(folder, "compiled_key.pkl")
            so_path = os.path.join(folder, "kernel_lib.so")

            if os.path.exists(key_path) and os.path.exists(so_path):
                with open(key_path, "rb") as f:
                    key = pickle.load(f)

                # Load the .so file as a module using cute.runtime
                mod = cute.runtime.load_module(so_path)
                raw_func = getattr(mod, KERNEL_SYMBOL_NAME)

                # Wrap the raw function with kwargs wrapper to match the expected signature
                wrapped = kwargs_wrapper.make_kwargs_wrapper(
                    raw_func, arg_names=arg_names
                )

                cache_dict[key] = wrapped

        print(
            f"\t=> {compiled_cache_name}: {len(cache_dict)} kernels loaded",
            flush=True,
        )
        has_kernel_loaded = has_kernel_loaded or len(cache_dict) > 0

    if not has_kernel_loaded:
        print("No pre-compiled FFA_FA4 kernels to load.", flush=True)
    else:
        print("Pre-compiled FFA_FA4 kernels loaded successfully.", flush=True)


def precompile_ffa_fa4(
    dtypes: list[torch.dtype],
    head_dims: list[int],
    qhead_per_kvhead: list[int],
    func_nums: list[int],
):
    assert is_fa4_installed, "FlashAttn4 is not installed, cannot pre-compile kernels"

    from magi_attention.functional.fa4 import fa4_bwd, fa4_fwd

    # 0. Set up device, runtime libraries and the cache directory
    device = torch.cuda.current_device()
    runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
    os.makedirs(FFA_FA4_CACHE_DIR, exist_ok=True)

    # 1. Clear existing caches to ensure fresh compilation
    for compiled_meta in COMPILED_META_DICT.values():
        compiled_cache = compiled_meta["cache_dict"]
        compiled_cache.clear()

    # 2. Iterate through combinations
    configs = itertools.product(
        dtypes,
        head_dims,
        qhead_per_kvhead,
        func_nums,
    )
    num_configs = len(dtypes) * len(head_dims) * len(qhead_per_kvhead) * len(func_nums)

    # 3. For each combination,
    # create mock inputs and call the FFA_FA4 functions to trigger compilation
    seq_q, seq_k_unit = 16, 4
    dtype_str_dict = {torch.bfloat16: "bf16", torch.float16: "fp16"}
    pbar = tqdm(
        configs,
        total=num_configs,
        desc="Pre-compiling FFA_FA4 kernels",
        dynamic_ncols=True,
        unit="kernel",
    )
    for dtype, hdim, qhead_per_kvhead, func_num in pbar:
        dtype_str = dtype_str_dict[dtype]
        pbar.set_description(
            f"\t=> Compiling case: [dtype={dtype_str}]x[hdim={hdim}]x[nhg={qhead_per_kvhead}]x[nfunc={func_num}]"
        )

        # 3-1. Mock FA4AttnArg for compilation
        seq_k = seq_k_unit * func_num
        if func_num == 1:
            k_ranges = AttnRanges.from_ranges([(0, seq_k)])
        else:
            k_ranges = AttnRanges.from_ranges(
                [(i * seq_k_unit, (i + 1) * seq_k_unit) for i in range(1, func_num, 2)]
            )
        q_ranges = AttnRanges.from_ranges([(0, seq_q)] * len(k_ranges))
        attn_type_map = [0] * len(k_ranges)
        attn_arg = FA4AttnArg(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            seqlen_q=seq_q,
            seqlen_k=seq_k,
        )
        assert (
            attn_arg.n_func == func_num
        ), f"Mismatch in function number for attn_arg, expected {func_num}, got {attn_arg.n_func}"

        # 3-2. Create dummy tensors for compilation signature
        # Shapes don't need to be huge, just valid for the kernel logic
        nhkv = 1
        nhq = nhkv * qhead_per_kvhead
        softmax_scale = 1.0 / math.sqrt(hdim)
        q = torch.empty((seq_q, nhq, hdim), dtype=dtype, device=device)
        k = torch.empty((seq_k, nhkv, hdim), dtype=dtype, device=device)
        v = torch.empty((seq_k, nhkv, hdim), dtype=dtype, device=device)
        o, do = torch.empty_like(q), torch.empty_like(q)
        lse = torch.empty((seq_q, nhq), dtype=torch.float32, device=device)

        # 3-3. Call the FFA_FA4 forward function to trigger compilation
        fa4_fwd(
            q=q,
            k=k,
            v=v,
            sink=None,
            attn_arg=attn_arg,
            softmax_scale=softmax_scale,
            softcap=0.0,
        )

        # 3-4. Call the FFA_FA4 backward function to trigger compilation
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
    print("", flush=True)

    # 4. Export to keys, .o/.so files for each compiled kernel in each compiled cache
    for compiled_cache_name, compiled_meta in COMPILED_META_DICT.items():
        compiled_cache = compiled_meta["cache_dict"]
        print(
            f"Export compiled FFA_FA4 kernels for {compiled_cache_name}: {len(compiled_cache)}",
            flush=True,
        )
        this_cached_dir = os.path.join(FFA_FA4_CACHE_DIR, compiled_cache_name)
        os.makedirs(this_cached_dir, exist_ok=True)

        for compiled_key, kernel in compiled_cache.items():
            hash = int(
                hashlib.sha256(
                    f"{compiled_cache_name}_{compiled_key}".encode("utf-8")
                ).hexdigest(),
                16,
            )
            kernel_cached_dir = os.path.join(this_cached_dir, str(hash))
            os.makedirs(kernel_cached_dir, exist_ok=True)

            # 4-1. Export compiled key
            key_path = os.path.join(kernel_cached_dir, "compiled_key.pkl")
            with open(key_path, "wb") as f:
                pickle.dump(compiled_key, f)

            # 4-2. Export .o file
            obj_path = os.path.join(kernel_cached_dir, "kernel_obj.o")
            kernel.export_to_c(obj_path, function_name=KERNEL_SYMBOL_NAME)

            # 4-3. Export .so file
            so_path = os.path.join(kernel_cached_dir, "kernel_lib.so")
            cmd = ["gcc", "-shared", "-fPIC", "-o", so_path, obj_path, *runtime_libs]
            subprocess.run(cmd, check=True)

            print(f"\t=> Exported: {so_path}")
        print("", flush=True)

    if num_configs == 0:
        print("No FFA_FA4 kernels to pre-compile.", flush=True)
    else:
        print("FFA_FA4 kernels pre-compiled successfully.", flush=True)
