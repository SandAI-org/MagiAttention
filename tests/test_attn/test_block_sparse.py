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

import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.utils.sparse_utils import (
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask_triton,
    get_sdpa_mask_from_block_sparse_mask,
)
from tests.test_attn.sparse_test_utils import (
    SparsePackLayout,
    compare_sdpa_bwd_all,
    compare_sdpa_fwd,
    inner_loop_env,
    pack_kv_for_ffa,
    pack_q_for_ffa,
    sdpa_ref_bwd_grads,
    sdpa_ref_output,
    unpack_ffa_output,
)

# ═══════════════════════════════════════════════════════════
# TestBlockSparseSweep — Classic CI sweep
# ═══════════════════════════════════════════════════════════


class TestBlockSparseSweep(DistTestBase):
    """BlockSparse Classic sweep — CI gate.

    Fixed compile params: NHQ=128, NHK=1(MQA), D=128, q_block=1, kbs=128, PackGQA=True.
    Varies runtime: seqlen × sparsity × swap_bwd_qk_loop(LoopK/LoopQ).
    """

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 600

    # parameter space shared by @parameterize and precompile_kernel_specs
    _PARAM_SPACE: dict[str, list] = dict(
        q_seqlen=[512, 1000, 16384],
        kv_seqlen=[512, 1000, 16384],
        sparsity=[0.2],
        swap_bwd_qk_loop=[False, True],
    )

    @classmethod
    def precompile_kernel_specs(cls):
        """Standard precompile interface — see magi_attention/testing/precompile.py.

        All classic-sweep combos share the same compile-time params
        (MQA128 + kbs=128); only the BWD loop direction changes kernels.
        The dense reference adds the non-sparse ARM+PackGQA128 kernels.
        """
        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}
        # dense reference: block_sparse=False, ARM, PackGQA128, swap=True
        add_ffa_spec(
            specs,
            direction="fwd",
            pack_gqa=True,
            pack_gqa_factor=128,
            range_merge=True,
        )
        add_ffa_spec(
            specs,
            direction="bwd",
            pack_gqa=True,
            pack_gqa_factor=128,
            range_merge=True,
            bwd_inner_loop_k=True,
        )
        # sparse FWD (auto-flags: disable_fwd_atomic, ref forced (128,128))
        add_ffa_spec(
            specs,
            direction="fwd",
            ref_block_size=(128, 128),
            disable_atomic=True,
            pack_gqa=True,
            pack_gqa_factor=128,
            block_sparse=True,
            range_merge=True,
            sparse_k_block_size=128,
        )
        for swap in cls._PARAM_SPACE["swap_bwd_qk_loop"]:
            add_ffa_spec(
                specs,
                direction="bwd",
                disable_atomic=not swap,  # LoopQ + PackGQA → dkv-atomic disabled
                disable_dq_atomic=swap,  # LoopK → dq-atomic disabled
                pack_gqa=True,
                pack_gqa_factor=128,
                block_sparse=True,
                range_merge=True,
                bwd_inner_loop_k=swap,
                sparse_k_block_size=128,
                bwd_dq_bf16=swap,
            )
        return specs

    @with_run_in_mp
    @parameterize("q_seqlen", _PARAM_SPACE["q_seqlen"])
    @parameterize("kv_seqlen", _PARAM_SPACE["kv_seqlen"])
    @parameterize("sparsity", _PARAM_SPACE["sparsity"])
    @parameterize("swap_bwd_qk_loop", _PARAM_SPACE["swap_bwd_qk_loop"])
    def test_block_sparse_mqa_sweep(
        self, q_seqlen, kv_seqlen, sparsity, swap_bwd_qk_loop
    ):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        device = self.device
        nhq, nhk, head_dim = 128, 1, 128
        dtype = torch.bfloat16
        kbs = 128

        n_q_blocks = q_seqlen
        n_k_blocks = kv_seqlen // kbs
        n_attend = max(1, int(n_k_blocks * (1.0 - sparsity)))

        sel = torch.rand(n_q_blocks, n_k_blocks, device=device).argsort(dim=1)[
            :, :n_attend
        ]
        block_mask = torch.zeros(
            1, nhk, n_q_blocks, n_k_blocks, dtype=torch.bool, device=device
        )
        block_mask[0, 0].scatter_(1, sel, True)
        q_ranges, k_ranges = generate_ranges_from_block_mask_triton(block_mask, 1, kbs)
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=device)

        q0 = torch.randn(q_seqlen, nhq, head_dim, device=device, dtype=dtype)
        k0 = torch.randn(kv_seqlen, nhk, head_dim, device=device, dtype=dtype)
        v0 = torch.randn(kv_seqlen, nhk, head_dim, device=device, dtype=dtype)
        do = torch.randn(q_seqlen, nhq, head_dim, device=device, dtype=dtype)

        def run(block_sparse, swap):
            q = q0.clone().requires_grad_(True)
            k = k0.clone().requires_grad_(True)
            v = v0.clone().requires_grad_(True)
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                block_sparse=block_sparse,
                range_merge=True,
                pack_gqa=True,
                swap_bwd_qk_loop=swap,
            )
            out.backward(do)
            return out.detach(), q.grad, k.grad, v.grad

        ref = run(block_sparse=False, swap=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        got = run(block_sparse=True, swap=swap_bwd_qk_loop)
        loop_name = "loopk" if swap_bwd_qk_loop else "loopq"
        tol = 2e-2
        for name, a, b in zip(("out", "dq", "dk", "dv"), got, ref):
            err = (
                (a.float() - b.float()).abs().max()
                / b.float().abs().max().clamp_min(1e-6)
            ).item()
            assert (
                err < tol
            ), f"sweep[Sq={q_seqlen},Skv={kv_seqlen},sp={sparsity},{loop_name}] {name} max_rel_err={err:.3e} >= {tol}"


# ═══════════════════════════════════════════════════════════
# TestBlockSparseComprehensiveSweep — comprehensive coverage (CI)
# ═══════════════════════════════════════════════════════════


class TestBlockSparseComprehensiveSweep(DistTestBase):
    """BlockSparse Comprehensive sweep — CI.

    Cross-product of GQA config × block size × inner env variants × loop order.
    Split by (head_dim, loop_order) into four @with_run_in_mp methods so each
    subprocess loads at most ~120 unique kernel .so files, staying well under
    the PTHREAD_KEYS_MAX=1024 TSS key limit.
    """

    _PARAM_SPACE: dict[str, list] = dict(
        nhq_nhk=[(128, 1), (4, 1), (128, 2), (32, 4), (4, 4), (32, 32)],
        head_dim=[64, 128],
        q_size=[1, 8, 128],
        k_size=[1, 128],
        sparsity_ratio=[0.5],
        swap_bwd_qk_loop=[True, False],
        inner_dir=["true", "false"],
        inner_load_mode=["tma", "cpasync"],
        inner_store_mode=["tma", "tma1d", "atomicadd"],
    )

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    @classmethod
    def precompile_kernel_specs(cls):
        """Standard precompile interface — see magi_attention/testing/precompile.py.

        FWD + BWD (LoopK + LoopQ) for each (pack_f × hd × k_size × inner modes) combo.
        q_size and sparsity_ratio are runtime-only (don't affect compilation).
        """
        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}
        seen_pack_f: set = set()
        kBlockN = 128
        kBlockM = 128
        for nhq, nhk in cls._PARAM_SPACE["nhq_nhk"]:
            pack_f = nhq // nhk
            if pack_f in seen_pack_f:
                continue
            seen_pack_f.add(pack_f)
            for hd in cls._PARAM_SPACE["head_dim"]:
                for k_size in cls._PARAM_SPACE["k_size"]:
                    common = dict(
                        head_dim=hd,
                        pack_gqa=True,
                        pack_gqa_factor=pack_f,
                        block_sparse=True,
                        range_merge=True,
                        sparse_k_block_size=k_size,
                    )
                    for inner_dir in cls._PARAM_SPACE["inner_dir"]:
                        for inner_load in cls._PARAM_SPACE["inner_load_mode"]:
                            if inner_load == "tma" and k_size < kBlockN:
                                continue
                            env_fwd = {
                                "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN": inner_dir,
                                "MAGI_ATTENTION_FFA_INNER_LOAD_MODE": inner_load,
                            }
                            add_ffa_spec(
                                specs,
                                direction="fwd",
                                env=env_fwd,
                                disable_atomic=True,
                                **common,
                            )
                            for inner_store in cls._PARAM_SPACE["inner_store_mode"]:
                                env_bwd = {
                                    "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN": inner_dir,
                                    "MAGI_ATTENTION_FFA_INNER_LOAD_MODE": inner_load,
                                    "MAGI_ATTENTION_FFA_INNER_STORE_MODE": inner_store,
                                }
                                for swap in cls._PARAM_SPACE["swap_bwd_qk_loop"]:
                                    if (
                                        not swap
                                        and inner_load == "tma"
                                        and pack_f < kBlockM
                                    ):
                                        continue
                                    add_ffa_spec(
                                        specs,
                                        direction="bwd",
                                        env=env_bwd,
                                        disable_atomic=not swap,
                                        disable_dq_atomic=swap,
                                        bwd_inner_loop_k=swap,
                                        bwd_dq_bf16=swap,
                                        **common,
                                    )
        return specs

    def _run_comprehensive_case(
        self,
        nhq_nhk,
        head_dim,
        q_size,
        k_size,
        sparsity_ratio,
        swap_bwd_qk_loop,
        inner_dir,
        inner_load_mode,
        inner_store_mode,
    ):
        nhq, nhk = nhq_nhk
        hd = head_dim

        kBlockN = 128
        kBlockM = 128
        if inner_load_mode == "tma" and k_size < kBlockN:
            return
        pack_gqa_factor = nhq // nhk
        if (
            not swap_bwd_qk_loop
            and inner_load_mode == "tma"
            and pack_gqa_factor < kBlockM
        ):
            return

        loop_tag = "loopk" if swap_bwd_qk_loop else "loopq"
        torch.manual_seed(42)
        seqlen = 2048
        dtype = torch.bfloat16

        num_q_blocks = seqlen // q_size
        num_kv_blocks = seqlen // k_size
        block_mask, _ = generate_block_sparse_pattern(
            num_q_heads=nhq,
            num_kv_heads=nhk,
            num_q_blocks=num_q_blocks,
            num_kv_blocks=num_kv_blocks,
            sparsity=sparsity_ratio,
            mode="per_kv_head",
            sparse_format="block_mask",
            device="cuda",
        )

        q = torch.randn(1, seqlen, nhq, hd, dtype=dtype, device=self.device)
        k = torch.randn(1, seqlen, nhk, hd, dtype=dtype, device=self.device)
        v = torch.randn(1, seqlen, nhk, hd, dtype=dtype, device=self.device)
        do = torch.randn_like(q)

        test_case = (
            f"[comprehensive][nhq={nhq},nhk={nhk},hd={hd},"
            f"q={q_size},k={k_size}]"
            f"[sp={sparsity_ratio}]"
            f"[dir={inner_dir},load={inner_load_mode},"
            f"store={inner_store_mode},{loop_tag}]"
        )

        q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
            block_mask, q_size, k_size
        )
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

        q_ffa = pack_q_for_ffa(q, nhk, SparsePackLayout.HEAD_MAJOR, requires_grad=True)
        k_ffa, v_ffa = pack_kv_for_ffa(
            k, v, SparsePackLayout.HEAD_MAJOR, requires_grad=True
        )

        inner_env = {
            "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN": inner_dir,
            "MAGI_ATTENTION_FFA_INNER_LOAD_MODE": inner_load_mode,
            "MAGI_ATTENTION_FFA_INNER_STORE_MODE": inner_store_mode,
        }
        do_packed = rearrange(
            do,
            "b s (h1 h2) d -> (b h1 s) h2 d",
            h1=nhk,
        )
        with inner_loop_env(inner_env):
            o_ffa, _ = flex_flash_attn_func(
                q_ffa,
                k_ffa,
                v_ffa,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                max_seqlen_q=q_size,
                attn_type_map=attn_type_map,
                range_merge=True,
                pack_gqa=True,
                block_sparse=True,
                swap_bwd_qk_loop=swap_bwd_qk_loop,
                ref_block_size=(64, 128),
            )
            o_ffa.backward(do_packed)
        torch.cuda.synchronize()

        o_unpacked = unpack_ffa_output(
            o_ffa,
            B=1,
            S=seqlen,
            NHK=nhk,
            layout=SparsePackLayout.HEAD_MAJOR,
        )

        sdpa_mask = get_sdpa_mask_from_block_sparse_mask(
            block_mask,
            seqlen,
            seqlen,
            q_size,
            k_size,
            nhk,
        )
        o_ref = sdpa_ref_output(q, k, v, sdpa_mask, B=1, NHQ=nhq, NHK=nhk)
        compare_sdpa_fwd(o_unpacked, o_ref, test_case=test_case)

        sdpa_dq, sdpa_dk, sdpa_dv = sdpa_ref_bwd_grads(
            do,
            q,
            k,
            v,
            sdpa_mask,
            NHQ=nhq,
            NHK=nhk,
        )
        dq_ref_packed = rearrange(
            sdpa_dq,
            "b (h1 h2) s d -> (b h1 s) h2 d",
            h1=nhk,
        )
        dk_ref_packed = rearrange(sdpa_dk, "b h s d -> (b h s) 1 d")
        dv_ref_packed = rearrange(sdpa_dv, "b h s d -> (b h s) 1 d")
        compare_sdpa_bwd_all(
            q_ffa.grad,
            k_ffa.grad,
            v_ffa.grad,
            dq_ref_packed,
            dk_ref_packed,
            dv_ref_packed,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("nhq_nhk", _PARAM_SPACE["nhq_nhk"])
    @parameterize("q_size", _PARAM_SPACE["q_size"])
    @parameterize("k_size", _PARAM_SPACE["k_size"])
    @parameterize("sparsity_ratio", _PARAM_SPACE["sparsity_ratio"])
    @parameterize("inner_dir", _PARAM_SPACE["inner_dir"])
    @parameterize("inner_load_mode", _PARAM_SPACE["inner_load_mode"])
    @parameterize("inner_store_mode", _PARAM_SPACE["inner_store_mode"])
    def test_block_sparse_comprehensive_sweep_hd64(
        self,
        nhq_nhk,
        q_size,
        k_size,
        sparsity_ratio,
        inner_dir,
        inner_load_mode,
        inner_store_mode,
    ):
        self._run_comprehensive_case(
            nhq_nhk,
            64,
            q_size,
            k_size,
            sparsity_ratio,
            True,
            inner_dir,
            inner_load_mode,
            inner_store_mode,
        )

    @with_run_in_mp
    @parameterize("nhq_nhk", _PARAM_SPACE["nhq_nhk"])
    @parameterize("q_size", _PARAM_SPACE["q_size"])
    @parameterize("k_size", _PARAM_SPACE["k_size"])
    @parameterize("sparsity_ratio", _PARAM_SPACE["sparsity_ratio"])
    @parameterize("inner_dir", _PARAM_SPACE["inner_dir"])
    @parameterize("inner_load_mode", _PARAM_SPACE["inner_load_mode"])
    @parameterize("inner_store_mode", _PARAM_SPACE["inner_store_mode"])
    def test_block_sparse_comprehensive_sweep_hd128(
        self,
        nhq_nhk,
        q_size,
        k_size,
        sparsity_ratio,
        inner_dir,
        inner_load_mode,
        inner_store_mode,
    ):
        self._run_comprehensive_case(
            nhq_nhk,
            128,
            q_size,
            k_size,
            sparsity_ratio,
            True,
            inner_dir,
            inner_load_mode,
            inner_store_mode,
        )

    @with_run_in_mp
    @parameterize("nhq_nhk", _PARAM_SPACE["nhq_nhk"])
    @parameterize("q_size", _PARAM_SPACE["q_size"])
    @parameterize("k_size", _PARAM_SPACE["k_size"])
    @parameterize("sparsity_ratio", _PARAM_SPACE["sparsity_ratio"])
    @parameterize("inner_dir", _PARAM_SPACE["inner_dir"])
    @parameterize("inner_load_mode", _PARAM_SPACE["inner_load_mode"])
    @parameterize("inner_store_mode", _PARAM_SPACE["inner_store_mode"])
    def test_block_sparse_comprehensive_sweep_loopq_hd64(
        self,
        nhq_nhk,
        q_size,
        k_size,
        sparsity_ratio,
        inner_dir,
        inner_load_mode,
        inner_store_mode,
    ):
        self._run_comprehensive_case(
            nhq_nhk,
            64,
            q_size,
            k_size,
            sparsity_ratio,
            False,
            inner_dir,
            inner_load_mode,
            inner_store_mode,
        )

    @with_run_in_mp
    @parameterize("nhq_nhk", _PARAM_SPACE["nhq_nhk"])
    @parameterize("q_size", _PARAM_SPACE["q_size"])
    @parameterize("k_size", _PARAM_SPACE["k_size"])
    @parameterize("sparsity_ratio", _PARAM_SPACE["sparsity_ratio"])
    @parameterize("inner_dir", _PARAM_SPACE["inner_dir"])
    @parameterize("inner_load_mode", _PARAM_SPACE["inner_load_mode"])
    @parameterize("inner_store_mode", _PARAM_SPACE["inner_store_mode"])
    def test_block_sparse_comprehensive_sweep_loopq_hd128(
        self,
        nhq_nhk,
        q_size,
        k_size,
        sparsity_ratio,
        inner_dir,
        inner_load_mode,
        inner_store_mode,
    ):
        self._run_comprehensive_case(
            nhq_nhk,
            128,
            q_size,
            k_size,
            sparsity_ratio,
            False,
            inner_dir,
            inner_load_mode,
            inner_store_mode,
        )


if __name__ == "__main__":
    run_tests()
