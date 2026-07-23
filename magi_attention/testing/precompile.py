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

"""Standard per-test-class FFA kernel precompile interface.

Any test class that exercises FFA JIT kernels beyond the library's default
prebuild set declares them via a uniform classmethod::

    class TestFoo(...):
        @classmethod
        def precompile_kernel_specs(cls) -> dict[str, JitSpec]:
            specs: dict[str, JitSpec] = {}
            add_ffa_spec(specs, direction="fwd", index_sparse=True, ...)
            return specs

The collector below imports the registered test modules, gathers the specs
of every class that *defines* (not merely inherits) this classmethod,
dedupes them by kernel URI, and builds them in parallel.

It is consumed by:
  * ``setup.py`` CI prebuild (``MAGI_ATTENTION_PREBUILD_LEVEL=ci`` or
    auto-detected CI environment), so ``pip install`` prebuilds exactly the
    kernels the test suite needs;
  * manual runs: ``python -m magi_attention.testing.precompile`` (optionally
    ``--list`` to only print the collected URIs).

Keeping the spec enumeration next to the test class (ideally derived from
the same constants fed to ``@parameterize``) prevents the CI prebuild list
from silently de-syncing from the actual test parameter space.
"""

import argparse
import importlib
import os
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from magi_attention.common.jit.core import JitSpec

__all__ = [
    "PRECOMPILE_CLASSMETHOD_NAME",
    "TEST_MODULES_WITH_KERNEL_SPECS",
    "add_ffa_spec",
    "collect_test_kernel_specs",
    "build_kernel_specs",
]

# Name of the uniform classmethod every kernel-consuming test class defines.
PRECOMPILE_CLASSMETHOD_NAME = "precompile_kernel_specs"

# Test modules scanned by the collector. A module is listed here iff at
# least one of its test classes defines `precompile_kernel_specs`.
TEST_MODULES_WITH_KERNEL_SPECS = [
    "tests.test_attn.test_flex_flash_attn",
    "tests.test_attn.test_block_sparse",
    "tests.test_attn.test_index_sparse",
    "tests.test_pipeline",
]

_DEFAULT_ARCH = (9, 0)


def add_ffa_spec(
    specs: dict[str, "JitSpec"],
    *,
    direction: str,
    head_dim: int = 128,
    compute_dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype | None = None,
    ref_block_size: tuple[int, int] | None = None,
    disable_atomic: bool = False,
    disable_dq_atomic: bool = False,
    deterministic: bool = False,
    range_merge: bool = False,
    swap_ab: bool = False,
    pack_gqa: bool = False,
    cat_gqa: bool = False,
    pack_gqa_factor: int = 1,
    block_sparse: bool = False,
    index_sparse: bool = False,
    bwd_inner_loop_k: bool = False,
    sparse_k_block_size: int = 1,
    return_max_logits: bool = False,
    bwd_dq_bf16: bool = False,
    bwd_dkv_bf16: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """Resolve one FFA kernel spec (with test-friendly defaults) and record it
    into ``specs`` keyed by URI.

    ``env`` temporarily sets environment variables during spec resolution —
    needed for compile-time variants driven by env vars (e.g. sparse
    inner-load/store modes).
    """
    from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_spec

    # Mirror runtime auto-set in flex_flash_attn_func:
    # BlockSparse always requires RangeMerge (C++ static_assert).
    if block_sparse:
        range_merge = True

    # flex_flash_attn_func forces ref_block_size=(128,128) for sparse FWD paths;
    # BWD never passes ref_block_size (tile is inferred from head_dim alone).
    if direction == "fwd" and (index_sparse or block_sparse) and ref_block_size is None:
        ref_block_size = (128, 128)

    if direction == "fwd":
        out_dtype = output_dtype if output_dtype is not None else compute_dtype
        dq_dtype = None
        dkv_dtype = None
    else:
        out_dtype = None
        # Outer direct-store path uses native dtype (matches flex_flash_attn.py backward):
        #   dQ outer (InnerLoopK + disable_dq_atomic) → compute_dtype
        #   dKV outer (InnerLoopQ + disable_atomic) → compute_dtype
        _dq_native = bwd_dq_bf16 or (disable_dq_atomic and bwd_inner_loop_k)
        _dkv_native = bwd_dkv_bf16 or (disable_atomic and not bwd_inner_loop_k)
        dq_dtype = compute_dtype if _dq_native else torch.float32
        dkv_dtype = compute_dtype if _dkv_native else torch.float32

    saved_env: dict[str, str | None] = {}
    if env:
        for key, val in env.items():
            saved_env[key] = os.environ.get(key)
            os.environ[key] = val
    try:
        spec, uri = get_ffa_jit_spec(
            arch=_DEFAULT_ARCH,
            direction=direction,  # type: ignore[arg-type]
            head_dim=head_dim,
            compute_dtype=compute_dtype,
            output_dtype=out_dtype,
            softcap=False,
            disable_atomic_reduction=disable_atomic,
            disable_dq_atomic_reduction=disable_dq_atomic,
            deterministic=deterministic,
            ref_block_size=ref_block_size,
            range_merge=range_merge,
            swap_ab=swap_ab,
            pack_gqa=pack_gqa,
            cat_gqa=cat_gqa,
            pack_gqa_factor=pack_gqa_factor,
            block_sparse=block_sparse,
            index_sparse=index_sparse,
            bwd_inner_loop_k=bwd_inner_loop_k,
            return_max_logits=return_max_logits,
            dq_dtype=dq_dtype,
            dkv_dtype=dkv_dtype,
            sparse_k_block_size=sparse_k_block_size,
        )
    finally:
        for key, old in saved_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    specs[uri] = spec


def collect_test_kernel_specs(
    module_names: Iterable[str] | None = None,
    repo_root: str | Path | None = None,
    log: Callable[[str], Any] = print,
) -> dict[str, "JitSpec"]:
    """Import test modules, gather kernel specs from every test class that
    defines ``precompile_kernel_specs``, and return the URI-deduped union.
    """
    if module_names is None:
        module_names = TEST_MODULES_WITH_KERNEL_SPECS

    if repo_root is None:
        # magi_attention/testing/precompile.py -> repo root two levels up
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = str(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    all_specs: dict[str, "JitSpec"] = {}
    for mod_name in module_names:
        module = importlib.import_module(mod_name)
        for attr_name in dir(module):
            klass = getattr(module, attr_name)
            if not isinstance(klass, type):
                continue
            # only classes that define the interface themselves — skip
            # subclasses inheriting it (e.g. TestPipelineWithWorldSizeN)
            # to avoid redundant re-collection of identical specs
            if PRECOMPILE_CLASSMETHOD_NAME not in vars(klass):
                continue
            class_specs = getattr(klass, PRECOMPILE_CLASSMETHOD_NAME)()
            n_new = len(set(class_specs) - set(all_specs))
            log(
                f"[precompile] {mod_name}.{attr_name}: "
                f"{len(class_specs)} kernels ({n_new} new)"
            )
            all_specs.update(class_specs)

    log(f"[precompile] collected {len(all_specs)} unique kernels total")
    return all_specs


def build_kernel_specs(
    specs: dict[str, "JitSpec"],
    jobs: int = 8,
    copy_to_aot: bool = False,
    raise_on_error: bool = False,
    log: Callable[[str], Any] = print,
) -> tuple[int, int]:
    """Build all specs in parallel with progress logging.

    Returns ``(n_ok, n_failed)``. Failures are logged as warnings unless
    ``raise_on_error`` is set (some enumerated combos may be legitimately
    unsupported, e.g. exceeding SM90 smem limits).
    """

    def _build_one(uri: str, spec: "JitSpec") -> str:
        spec.build()
        if copy_to_aot:
            import shutil

            from magi_attention.common.jit import env as jit_env

            src_dir = (jit_env.MAGI_ATTENTION_JIT_DIR / uri).resolve()
            dst_dir = (jit_env.MAGI_ATTENTION_AOT_DIR / uri).resolve()
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        return uri

    n_total = len(specs)
    n_done = 0
    n_failed = 0
    t_start = time.time()
    log(f"[precompile] building {n_total} kernels with {jobs} parallel jobs")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_build_one, uri, spec): uri for uri, spec in specs.items()}
        for fut in as_completed(futs):
            uri = futs[fut]
            n_done += 1
            elapsed = time.time() - t_start
            try:
                fut.result()
                log(f"[precompile {n_done}/{n_total}] ({elapsed:.0f}s) OK: {uri}")
            except Exception as e:
                n_failed += 1
                if raise_on_error:
                    raise RuntimeError(f"precompile failed for {uri}: {e}") from e
                log(
                    f"[precompile {n_done}/{n_total}] ({elapsed:.0f}s) "
                    f"WARNING: failed {uri}: {e}"
                )
    log(
        f"[precompile] finished: {n_total - n_failed}/{n_total} ok, "
        f"{n_failed} failed, total {time.time() - t_start:.0f}s"
    )
    return n_total - n_failed, n_failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and prebuild FFA kernels declared by test classes"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="only print collected kernel URIs, do not build",
    )
    parser.add_argument("--jobs", type=int, default=8, help="parallel build jobs")
    parser.add_argument(
        "--module",
        action="append",
        default=None,
        help="restrict to specific test module(s); default: all registered",
    )
    args = parser.parse_args()

    specs = collect_test_kernel_specs(module_names=args.module)
    if args.list:
        for uri in sorted(specs):
            print(uri)
        return
    _, n_failed = build_kernel_specs(specs, jobs=args.jobs)
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
