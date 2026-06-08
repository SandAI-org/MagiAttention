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

"""Lightweight regression tests for flex_flash_attn.

Each test runs in a single process for fast iteration and easy
sanitizer/debugger attachment.
"""

import time
import unittest
from typing import Any

import torch

from magi_attention.common import AttnRanges
from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.utils.sparse_utils import (
    build_index_attn_indices,
    get_sdpa_mask_from_index_attn_indices,
)


class TestSimpleAttn(unittest.TestCase):
    """Lightweight single-process regression tests.
    Uses TestFlexFlashAttn.assert_close_to_torch_ref via a helper instance.
    """

    @property
    def device(self):
        return torch.cuda.current_device()

    def assert_close_to_torch_ref(self, **kwargs):
        from tests.test_attn.test_flex_flash_attn import TestFlexFlashAttn

        TestFlexFlashAttn.assert_close_to_torch_ref(self, **kwargs)

    # ─── IndexAttn FWD+BWD ───

    INDEX_ATTN_CONFIGS = [
        {
            "name": "mqa128_pack_gqa",
            "B": 1,
            "S": 256,
            "NHQ": 128,
            "NHK": 1,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa_32_4_pack_gqa",
            "B": 1,
            "S": 256,
            "NHQ": 32,
            "NHK": 4,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "mha_aligned",
            "B": 1,
            "S": 256,
            "NHQ": 4,
            "NHK": 4,
            "D": 64,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "gqa_4_2_aligned",
            "B": 1,
            "S": 256,
            "NHQ": 4,
            "NHK": 2,
            "D": 64,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "mha_unaligned_seqlen",
            "B": 1,
            "S": 200,
            "NHQ": 4,
            "NHK": 4,
            "D": 64,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "gqa_8_2_small",
            "B": 2,
            "S": 256,
            "NHQ": 8,
            "NHK": 2,
            "D": 64,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "gqa_4_2_pack_gqa_d128",
            "B": 1,
            "S": 256,
            "NHQ": 4,
            "NHK": 2,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa_8_2_pack_gqa_d128",
            "B": 1,
            "S": 256,
            "NHQ": 8,
            "NHK": 2,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa_4_1_pack_gqa_d128",
            "B": 1,
            "S": 256,
            "NHQ": 4,
            "NHK": 1,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa_8_4_pack_gqa_d128",
            "B": 2,
            "S": 256,
            "NHQ": 8,
            "NHK": 4,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
    ]

    @parameterize("cfg", INDEX_ATTN_CONFIGS)
    def test_index_attn_simple(self, cfg: dict[str, Any]):
        """IndexAttn FWD+BWD correctness against SDPA reference.

        The view trick flattens K from (B,S,NHK,D) to (B*S*NHK, 1, D), so
        the kernel sees NHK_eff=1. Indices must be built in this flat token
        space (NHK=1, S_flat=S*NHK) with logical positions.
        """
        from magi_attention.utils import set_random_seed

        set_random_seed(42)
        B, S, NHQ, NHK, D, topk = (
            cfg["B"],
            cfg["S"],
            cfg["NHQ"],
            cfg["NHK"],
            cfg["D"],
            cfg["topk"],
        )
        pack_gqa = cfg["pack_gqa"]
        device = self.device

        gqa = NHQ // NHK
        S_flat = S * NHK
        NHQ_eff = gqa

        indices = build_index_attn_indices(B, 1, S_flat, S_flat, topk, topk, device)

        q_raw = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=device)
        k_raw = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=device)
        v_raw = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=device)

        q_ffa = (
            q_raw.reshape(B, S, NHK, gqa, D)
            .permute(0, 1, 2, 3, 4)
            .reshape(B * S * NHK, gqa, D)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        k_ffa = k_raw.reshape(B * S * NHK, 1, D).detach().clone().requires_grad_(True)
        v_ffa = v_raw.reshape(B * S * NHK, 1, D).detach().clone().requires_grad_(True)

        o_sparse, _ = flex_flash_attn_func(
            q_ffa,
            k_ffa,
            v_ffa,
            index_attn_indices=indices,
            q_block_size=1,
            k_block_size=1,
            pack_gqa=pack_gqa,
        )

        mask = get_sdpa_mask_from_index_attn_indices(
            indices, B, NHQ_eff, 1, S_flat, S_flat, device
        )

        # FWD verification (compare in flat token space)
        for b in range(B):
            sl = slice(b * S_flat, (b + 1) * S_flat)
            q_b = q_ffa[sl].detach().reshape(1, S_flat, NHQ_eff, D).transpose(1, 2)
            k_b = k_ffa[sl].detach().reshape(1, S_flat, 1, D).transpose(1, 2)
            v_b = v_ffa[sl].detach().reshape(1, S_flat, 1, D).transpose(1, 2)
            if NHQ_eff > 1:
                k_b = k_b.expand(1, NHQ_eff, S_flat, D)
                v_b = v_b.expand(1, NHQ_eff, S_flat, D)

            with torch.no_grad():
                o_ref = torch.nn.functional.scaled_dot_product_attention(
                    q_b, k_b, v_b, attn_mask=mask[b].unsqueeze(0)
                )
            o_ref = o_ref.squeeze(0).transpose(0, 1)

            max_diff = (o_sparse[sl].float() - o_ref.float()).abs().max().item()
            assert max_diff < 0.01, (
                f"[test_index_attn][{cfg['name']}] "
                f"FWD batch {b}: max_diff={max_diff:.6f} >= 0.01"
            )

        # BWD verification
        do = torch.randn_like(o_sparse)
        o_sparse.backward(do)
        dq_ffa = q_ffa.grad.clone()

        for b in range(B):
            sl = slice(b * S_flat, (b + 1) * S_flat)
            q_b = (
                q_ffa[sl]
                .detach()
                .clone()
                .reshape(1, S_flat, NHQ_eff, D)
                .transpose(1, 2)
                .requires_grad_(True)
            )
            k_b = (
                k_ffa[sl]
                .detach()
                .clone()
                .reshape(1, S_flat, 1, D)
                .transpose(1, 2)
                .requires_grad_(True)
            )
            v_b = (
                v_ffa[sl]
                .detach()
                .clone()
                .reshape(1, S_flat, 1, D)
                .transpose(1, 2)
                .requires_grad_(True)
            )
            k_exp = k_b.expand(1, NHQ_eff, S_flat, D) if NHQ_eff > 1 else k_b
            v_exp = v_b.expand(1, NHQ_eff, S_flat, D) if NHQ_eff > 1 else v_b

            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_b, k_exp, v_exp, attn_mask=mask[b].unsqueeze(0)
            )
            do_b = do[sl].reshape(1, S_flat, NHQ_eff, D).transpose(1, 2)
            o_ref.backward(do_b)

            dq_ref_b = q_b.grad.squeeze(0).transpose(0, 1)
            max_dq_diff = (dq_ffa[sl].float() - dq_ref_b.float()).abs().max().item()
            assert max_dq_diff < 0.05, (
                f"[test_index_attn][{cfg['name']}] "
                f"BWD batch {b}: dQ max_diff={max_dq_diff:.6f} >= 0.05"
            )

    # ─── Dense FWD+BWD ───

    DENSE_FWD_BWD_CONFIGS = [
        {
            "name": "Full+MHA+aligned",
            "S_q": 512,
            "S_k": 512,
            "NHQ": 8,
            "NHK": 8,
            "attn_type": 0,
        },
        {
            "name": "Causal+MHA+unaligned",
            "S_q": 300,
            "S_k": 300,
            "NHQ": 8,
            "NHK": 8,
            "attn_type": 1,
        },
        {
            "name": "Full+GQA+aligned",
            "S_q": 256,
            "S_k": 256,
            "NHQ": 8,
            "NHK": 2,
            "attn_type": 0,
        },
        {
            "name": "Causal+GQA+unaligned",
            "S_q": 370,
            "S_k": 370,
            "NHQ": 8,
            "NHK": 2,
            "attn_type": 1,
        },
        {
            "name": "InvCausal+MHA+unaligned",
            "S_q": 300,
            "S_k": 300,
            "NHQ": 8,
            "NHK": 8,
            "attn_type": 2,
        },
    ]

    @parameterize("dense_cfg", DENSE_FWD_BWD_CONFIGS)
    def test_dense_fwd_bwd_simple(self, dense_cfg):
        """Quick regression test for Dense FWD+BWD paths.

        Covers: Full/Causal/InvCausal × MHA/GQA × aligned/unaligned seqlen.
        """
        device = self.device
        torch.manual_seed(42)

        head_dim = 128
        dtype = torch.bfloat16

        cfg_name = dense_cfg["name"]
        S_q, S_k = dense_cfg["S_q"], dense_cfg["S_k"]
        NHQ, NHK = dense_cfg["NHQ"], dense_cfg["NHK"]
        attn_type_val = dense_cfg["attn_type"]

        q_ranges_list = [[0, S_q]]
        k_ranges_list = [[0, S_k]]
        attn_type_map_list = [attn_type_val]
        attn_type_map_tensor = torch.tensor(
            attn_type_map_list, dtype=torch.int32, device=device
        )

        q = torch.randn(
            S_q, NHQ, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        k = torch.randn(
            S_k, NHK, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        v = torch.randn(
            S_k, NHK, head_dim, dtype=dtype, device=device, requires_grad=True
        )
        do = torch.randn(S_q, NHQ, head_dim, dtype=dtype, device=device)

        q_ranges_tensor = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
        k_ranges_tensor = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)

        o, meta = flex_flash_attn_func(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map_tensor,
        )
        lse = meta.lse
        o.backward(do)

        test_case = f"[test_dense_fwd_bwd_simple][{cfg_name}]"
        self.assert_close_to_torch_ref(
            q_ranges=AttnRanges.from_ranges(q_ranges_list),
            k_ranges=AttnRanges.from_ranges(k_ranges_list),
            attn_type_map=attn_type_map_list,
            total_seqlen_q=S_q,
            total_seqlen_k=S_k,
            total_q=q,
            total_k=k,
            total_v=v,
            total_sink=None,
            total_out=o,
            total_lse=lse,
            grad_total_q=q.grad,
            grad_total_k=k.grad,
            grad_total_v=v.grad,
            grad_total_sink=None,
            grad_total_out=do,
            dtype=dtype,
            sink_layout="sh",
            test_case=test_case,
        )

    # ─── FWD+BWD RangeMerge ───

    RANGEMERGE_CONFIGS = [
        {
            "name": "LoopK+RM+Causal",
            "swap": True,
            "merge": True,
            "attn_type": 1,
            "k_ranges_key": "unaligned",
        },
        {
            "name": "LoopQ+RM+Full",
            "swap": False,
            "merge": True,
            "attn_type": 0,
            "k_ranges_key": "unaligned",
        },
        {
            "name": "LoopK+Dense+Causal",
            "swap": True,
            "merge": False,
            "attn_type": 1,
            "k_ranges_key": "unaligned",
        },
        {
            "name": "LoopK+RM+aligned",
            "swap": True,
            "merge": True,
            "attn_type": 0,
            "k_ranges_key": "aligned",
        },
    ]

    @parameterize("rm_cfg", RANGEMERGE_CONFIGS)
    def test_rangemerge(self, rm_cfg):
        """Test FWD+BWD RangeMerge with BlockMeta for LoopK and LoopQ paths."""
        device = self.device
        torch.manual_seed(42)

        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 128
        dtype = torch.bfloat16

        q_ranges_list = [[0, 256], [256, 512], [512, 768], [768, 1024]]
        k_ranges_map = {
            "unaligned": [[0, 170], [0, 170], [170, 384], [170, 384]],
            "aligned": [[0, 128], [0, 128], [128, 384], [128, 384]],
        }

        total_q = max(r[1] for r in q_ranges_list)
        cur_k_ranges_list = k_ranges_map[rm_cfg["k_ranges_key"]]
        cur_total_k = max(r[1] for r in cur_k_ranges_list)

        q = torch.randn(
            total_q,
            num_heads_q,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            cur_total_k,
            num_heads_kv,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            cur_total_k,
            num_heads_kv,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        do = torch.randn(total_q, num_heads_q, head_dim, dtype=dtype, device=device)

        q_ranges_tensor = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
        k_ranges_tensor = torch.tensor(
            cur_k_ranges_list, dtype=torch.int32, device=device
        )

        num_batches = len(q_ranges_list)
        attn_type_val = rm_cfg["attn_type"]
        attn_type_map_list = [attn_type_val] * num_batches
        attn_type_map_tensor = torch.full(
            (num_batches,), attn_type_val, dtype=torch.int32, device=device
        )

        o, meta = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map_tensor,
            auto_range_merge=rm_cfg["merge"],
            swap_bwd_qk_loop=rm_cfg["swap"],
        )
        lse = meta.lse
        o.backward(do)

        cfg_name = rm_cfg["name"]
        test_case = f"[test_rangemerge][{cfg_name}]"
        self.assert_close_to_torch_ref(
            q_ranges=AttnRanges.from_ranges(q_ranges_list),
            k_ranges=AttnRanges.from_ranges(cur_k_ranges_list),
            attn_type_map=attn_type_map_list,
            total_seqlen_q=total_q,
            total_seqlen_k=cur_total_k,
            total_q=q,
            total_k=k,
            total_v=v,
            total_sink=None,
            total_out=o,
            total_lse=lse,
            grad_total_q=q.grad,
            grad_total_k=k.grad,
            grad_total_v=v.grad,
            grad_total_sink=None,
            grad_total_out=do,
            dtype=dtype,
            sink_layout="sh",
            test_case=test_case,
        )

    # ─── FWD+BWD Deterministic ───

    DETERMINISTIC_CONFIGS = [
        {"name": "LoopQ+Deterministic", "swap": False, "merge": False},
        # NOTE: Deterministic+Merge is excluded because auto_range_merge+deterministic
        # is not yet supported (assert in flex_flash_attn.py).
    ]

    @parameterize("det_cfg", DETERMINISTIC_CONFIGS)
    def test_deterministic(self, det_cfg):
        """Verify bit-exact reproducibility: two runs with identical inputs
        must produce identical O, dQ, dK, dV."""
        device = self.device
        torch.manual_seed(42)

        num_heads_q = 8
        num_heads_kv = 2
        head_dim = 128
        dtype = torch.bfloat16

        q_ranges_list = [[0, 256], [256, 512], [512, 768], [768, 1024]]
        k_ranges_list = [[0, 170], [0, 170], [170, 384], [170, 384]]

        total_q = max(r[1] for r in q_ranges_list)
        total_k = max(r[1] for r in k_ranges_list)

        q_ranges_tensor = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
        k_ranges_tensor = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)
        num_batches = len(q_ranges_list)
        attn_type_map_tensor = torch.zeros(
            num_batches, dtype=torch.int32, device=device
        )

        swap = det_cfg["swap"]
        merge = det_cfg["merge"]
        cfg_name = det_cfg["name"]

        results = []
        for _ in range(2):
            q = torch.randn(
                total_q,
                num_heads_q,
                head_dim,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            k = torch.randn(
                total_k,
                num_heads_kv,
                head_dim,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            v = torch.randn(
                total_k,
                num_heads_kv,
                head_dim,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            do = torch.randn(total_q, num_heads_q, head_dim, dtype=dtype, device=device)

            if len(results) == 0:
                q_data = q.data.clone()
                k_data = k.data.clone()
                v_data = v.data.clone()
                do_data = do.clone()
            else:
                q.data.copy_(q_data)
                k.data.copy_(k_data)
                v.data.copy_(v_data)
                do = do_data.clone()

            o, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=q_ranges_tensor,
                k_ranges=k_ranges_tensor,
                attn_type_map=attn_type_map_tensor,
                auto_range_merge=merge,
                deterministic=True,
                swap_bwd_qk_loop=swap,
            )
            o.backward(do)
            results.append(
                (o.detach().clone(), q.grad.clone(), k.grad.clone(), v.grad.clone())
            )

        o1, dq1, dk1, dv1 = results[0]
        o2, dq2, dk2, dv2 = results[1]

        assert torch.equal(o1, o2), f"[{cfg_name}] FWD output not deterministic"
        assert torch.equal(dq1, dq2), f"[{cfg_name}] dQ not deterministic"
        assert torch.equal(dk1, dk2), f"[{cfg_name}] dK not deterministic"
        assert torch.equal(dv1, dv2), f"[{cfg_name}] dV not deterministic"

    # ─── Block-Sparse FWD (very simple) ───

    VERY_SIMPLE_BLOCK_SPARSE_CONFIGS = [
        {
            "name": "swap_ab_q128k128",
            "q_size": 128,
            "k_size": 128,
            "swap_ab": True,
            "sparse_load": False,
            "ref_block_size": (64, 64),
        },
        {
            "name": "sparse_load_q64k64",
            "q_size": 64,
            "k_size": 64,
            "swap_ab": False,
            "sparse_load": True,
            "ref_block_size": (64, 128),
        },
        {
            "name": "sparse_load_q128k1",
            "q_size": 128,
            "k_size": 1,
            "swap_ab": False,
            "sparse_load": True,
            "ref_block_size": (128, 128),
        },
    ]

    @parameterize("cfg", VERY_SIMPLE_BLOCK_SPARSE_CONFIGS)
    def test_very_simple_block_sparse(self, cfg):
        """Lightweight block-sparse FWD test (GQA NHQ=16, NHK=4)."""
        torch.manual_seed(42)
        device = self.device

        seqlen = 2048
        dtype = torch.bfloat16
        num_heads_q = 16
        num_heads_kv = 4
        head_dim = 128

        q_block_size = cfg["q_size"]
        k_block_size = cfg["k_size"]
        swap_ab = cfg["swap_ab"]
        sparse_load = cfg["sparse_load"]
        ref_block_size = cfg["ref_block_size"]
        block_size = (q_block_size, k_block_size)
        max_seqlen_q = q_block_size

        from tests.test_attn.test_block_sparse_attn import TestBlockSparseAttn

        helper = TestBlockSparseAttn.__new__(TestBlockSparseAttn)

        (
            block_mask,
            block_sizes,
            block_row_sz,
            block_col_sz,
        ) = helper._generate_sparse_pattern(
            test_type="uniform",
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            seqlen=seqlen,
            sparsity_ratio=0.5,
            sparsity_granularity="per_kv_head",
            sparse_format="block_mask",
            block_size=block_size,
        )

        q = torch.randn(
            1,
            seqlen,
            num_heads_q,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            1,
            seqlen,
            num_heads_kv,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            1,
            seqlen,
            num_heads_kv,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        do = torch.randn_like(q)

        test_case = f"[very_simple_block_sparse][{cfg['name']}]"
        print(f"\n>>> {test_case} START", flush=True)
        t0 = time.time()
        helper.assert_close_to_torch_ref(
            dtype=dtype,
            q=q,
            k=k,
            v=v,
            grad_output=do,
            seqlen=seqlen,
            block_size=block_sizes,
            block_mask=block_mask,
            head_wise="per_kv_head",
            sparse_format="block_mask",
            nhq=num_heads_q,
            nhk=num_heads_kv,
            pack_gqa=True,
            deterministic=False,
            test_accumulation_inplace=False,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            sparse_load=sparse_load,
            swap_bwd_qk_loop=sparse_load,
            test_case=test_case,
            sparsity_ratio=0.5,
            uniform=True,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            max_seqlen_q=max_seqlen_q,
        )
        print(f">>> {test_case} PASSED  ({time.time() - t0:.1f}s)", flush=True)

    # ─── SparseLoad + SwapAB coverage ───

    SPARSE_LOAD_SWAPAB_CONFIGS = [
        {
            "name": "qBlockM128_q32k64",
            "q_size": 32,
            "k_size": 64,
            "swap_ab": False,
            "ref_block_size": (128, 128),
        },
        {
            "name": "qBlockM64_q16k64",
            "q_size": 16,
            "k_size": 64,
            "swap_ab": False,
            "ref_block_size": (64, 128),
        },
        {
            "name": "qBlockM128_q32k128",
            "q_size": 32,
            "k_size": 128,
            "swap_ab": False,
            "ref_block_size": (128, 128),
        },
        {
            "name": "qBlockM64_q16k128",
            "q_size": 16,
            "k_size": 128,
            "swap_ab": False,
            "ref_block_size": (64, 128),
        },
        # SwapAB=True → kBlockN=64 → NumRowsPerGroup=4 (tests advance_producer with smaller group)
        {
            "name": "swapab_q32k64",
            "q_size": 32,
            "k_size": 64,
            "swap_ab": True,
            "ref_block_size": (32, 64),
        },
        {
            "name": "swapab_q16k64",
            "q_size": 16,
            "k_size": 64,
            "swap_ab": True,
            "ref_block_size": (16, 64),
        },
    ]

    @parameterize("cfg", SPARSE_LOAD_SWAPAB_CONFIGS)
    def test_sparse_load_swapab(self, cfg):
        """SparseLoad + SwapAB coverage (GQA NHQ=16, NHK=4, group=4)."""
        torch.manual_seed(42)
        device = self.device

        seqlen = 2048
        dtype = torch.bfloat16
        head_dim = 128
        num_heads_q = 16
        num_heads_kv = 4

        q_block_size = cfg["q_size"]
        k_block_size = cfg["k_size"]
        swap_ab = cfg["swap_ab"]
        ref_block_size = cfg["ref_block_size"]
        block_size = (q_block_size, k_block_size)
        max_seqlen_q = q_block_size

        from tests.test_attn.test_block_sparse_attn import TestBlockSparseAttn

        helper = TestBlockSparseAttn.__new__(TestBlockSparseAttn)

        (
            block_mask,
            block_sizes,
            block_row_sz,
            block_col_sz,
        ) = helper._generate_sparse_pattern(
            test_type="uniform",
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            seqlen=seqlen,
            sparsity_ratio=0.5,
            sparsity_granularity="per_kv_head",
            sparse_format="block_mask",
            block_size=block_size,
        )

        q = torch.randn(
            1,
            seqlen,
            num_heads_q,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            1,
            seqlen,
            num_heads_kv,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            1,
            seqlen,
            num_heads_kv,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        do = torch.randn_like(q)

        group_size = num_heads_q // num_heads_kv
        qBlockM = group_size * q_block_size
        test_case = f"[sparse_load_swapab][{cfg['name']},qBlockM={qBlockM}]"
        print(f"\n>>> {test_case} START", flush=True)
        t0 = time.time()
        helper.assert_close_to_torch_ref(
            dtype=dtype,
            q=q,
            k=k,
            v=v,
            grad_output=do,
            seqlen=seqlen,
            block_size=block_sizes,
            block_mask=block_mask,
            head_wise="per_kv_head",
            sparse_format="block_mask",
            nhq=num_heads_q,
            nhk=num_heads_kv,
            pack_gqa=False,
            deterministic=False,
            test_accumulation_inplace=False,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            sparse_load=True,
            swap_bwd_qk_loop=True,
            test_case=test_case,
            sparsity_ratio=0.5,
            uniform=True,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            max_seqlen_q=max_seqlen_q,
        )
        print(f">>> {test_case} PASSED  ({time.time() - t0:.1f}s)", flush=True)

    # ─── Mask build performance: vectorized vs python-loop ───

    def test_sdpa_mask_build_vectorized(self):
        """Verify vectorized get_sdpa_mask_from_block_sparse_mask matches a
        naive python-loop reference and is significantly faster."""
        from magi_attention.utils.sparse_utils import (
            deprecated_slow_get_sdpa_mask_from_block_sparse_mask,
            generate_block_sparse_pattern,
            get_sdpa_mask_from_block_sparse_mask,
        )

        device = self.device
        seqlen, q_bs, k_bs = 512, 64, 64
        nhq, nhk = 8, 2
        nqb, nkb = seqlen // q_bs, seqlen // k_bs

        block_mask, _ = generate_block_sparse_pattern(
            num_q_heads=nhq,
            num_kv_heads=nhk,
            num_q_blocks=nqb,
            num_kv_blocks=nkb,
            sparsity=0.5,
            mode="per_kv_head",
            sparse_format="block_mask",
            device=device,
        )

        # --- reference: deprecated slow python-loop implementation ---
        torch.cuda.synchronize()
        t0 = time.time()

        mask_ref = deprecated_slow_get_sdpa_mask_from_block_sparse_mask(
            block_mask, seqlen, seqlen, q_bs, k_bs, nhq
        )

        torch.cuda.synchronize()
        t_loop = time.time() - t0

        # --- current: vectorized implementation ---
        torch.cuda.synchronize()
        t1 = time.time()

        mask_vec = get_sdpa_mask_from_block_sparse_mask(
            block_mask, seqlen, seqlen, q_bs, k_bs, nhq
        )

        torch.cuda.synchronize()
        t_vec = time.time() - t1

        print(
            f"\n  mask build (seqlen={seqlen}, q_bs={q_bs}, k_bs={k_bs}):"
            f"  loop={t_loop:.3f}s  vec={t_vec:.4f}s  speedup={t_loop / max(t_vec, 1e-9):.0f}x",
            flush=True,
        )

        assert torch.equal(mask_ref, mask_vec), "vectorized mask != loop mask"

    # ─── IntraWGOverlap=false unit test ───

    def test_intra_wg_overlap_off(self):
        """Verify Dense FWD+BWD passes with IntraWGOverlap=false (non-overlapped V load)."""
        import os

        from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_mod

        device = self.device
        torch.manual_seed(42)

        os.environ["MAGI_ATTENTION_FFA_INTRA_WG_OVERLAP"] = "false"
        if hasattr(get_ffa_jit_mod, "cache_clear"):
            get_ffa_jit_mod.cache_clear()
        try:
            S_q, S_k, NHQ, NHK, head_dim = 256, 256, 4, 4, 128
            dtype = torch.bfloat16
            q = torch.randn(
                S_q, NHQ, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            k = torch.randn(
                S_k, NHK, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            v = torch.randn(
                S_k, NHK, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            do = torch.randn(S_q, NHQ, head_dim, dtype=dtype, device=device)

            q_ranges = torch.tensor([[0, S_q]], dtype=torch.int32, device=device)
            k_ranges = torch.tensor([[0, S_k]], dtype=torch.int32, device=device)
            attn_type_map = torch.tensor([0], dtype=torch.int32, device=device)

            o, meta = flex_flash_attn_func(
                q=q,
                k=k,
                v=v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
            )
            o.backward(do)

            self.assert_close_to_torch_ref(
                q_ranges=AttnRanges.from_ranges([[0, S_q]]),
                k_ranges=AttnRanges.from_ranges([[0, S_k]]),
                attn_type_map=[0],
                total_seqlen_q=S_q,
                total_seqlen_k=S_k,
                total_q=q,
                total_k=k,
                total_v=v,
                total_sink=None,
                total_out=o,
                total_lse=meta.lse,
                grad_total_q=q.grad,
                grad_total_k=k.grad,
                grad_total_v=v.grad,
                grad_total_sink=None,
                grad_total_out=do,
                dtype=dtype,
                sink_layout="sh",
                test_case="[test_intra_wg_overlap_off]",
            )
        finally:
            del os.environ["MAGI_ATTENTION_FFA_INTRA_WG_OVERLAP"]
            if hasattr(get_ffa_jit_mod, "cache_clear"):
                get_ffa_jit_mod.cache_clear()

    # ─── InnerDir MinToMax unit test ───

    def test_inner_dir_min_to_max(self):
        """Verify Dense + IndexAttn FWD+BWD with InnerDir=MinToMax.

        Dense: causal 256 seqlen, verifies reversed traversal order.
        IndexAttn: topk=100 with max_topk=128 (28 padding tokens in the last
        block), verifies padding_mask is applied to the correct block when
        the sparse iteration direction is flipped.
        """
        import os

        from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_mod

        device = self.device
        torch.manual_seed(42)

        os.environ["MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"] = "false"
        if hasattr(get_ffa_jit_mod, "cache_clear"):
            get_ffa_jit_mod.cache_clear()
        try:
            # ── Part 1: Dense causal ──
            S_q, S_k, NHQ, NHK, head_dim = 256, 256, 4, 4, 128
            dtype = torch.bfloat16
            q = torch.randn(
                S_q, NHQ, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            k = torch.randn(
                S_k, NHK, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            v = torch.randn(
                S_k, NHK, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            do = torch.randn(S_q, NHQ, head_dim, dtype=dtype, device=device)

            q_ranges = torch.tensor([[0, S_q]], dtype=torch.int32, device=device)
            k_ranges = torch.tensor([[0, S_k]], dtype=torch.int32, device=device)
            attn_type_map = torch.tensor([1], dtype=torch.int32, device=device)

            o, meta = flex_flash_attn_func(
                q=q,
                k=k,
                v=v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
            )
            o.backward(do)

            self.assert_close_to_torch_ref(
                q_ranges=AttnRanges.from_ranges([[0, S_q]]),
                k_ranges=AttnRanges.from_ranges([[0, S_k]]),
                attn_type_map=[1],
                total_seqlen_q=S_q,
                total_seqlen_k=S_k,
                total_q=q,
                total_k=k,
                total_v=v,
                total_sink=None,
                total_out=o,
                total_lse=meta.lse,
                grad_total_q=q.grad,
                grad_total_k=k.grad,
                grad_total_v=v.grad,
                grad_total_sink=None,
                grad_total_out=do,
                dtype=dtype,
                sink_layout="sh",
                test_case="[test_inner_dir_min_to_max/dense_causal]",
            )

            # ── Part 2: IndexAttn with non-aligned topk (padding block) ──
            B, S, NHQ_ia, NHK_ia, D = 1, 256, 32, 4, 128
            actual_topk = 100
            max_topk = 128
            gqa_ia = NHQ_ia // NHK_ia
            S_flat = S * NHK_ia
            NHQ_eff = gqa_ia

            indices = build_index_attn_indices(
                B, 1, S_flat, S_flat, actual_topk, max_topk, device
            )

            q_raw = torch.randn(B, S, NHQ_ia, D, dtype=dtype, device=device)
            k_raw = torch.randn(B, S, NHK_ia, D, dtype=dtype, device=device)
            v_raw = torch.randn(B, S, NHK_ia, D, dtype=dtype, device=device)

            q_ffa = (
                q_raw.reshape(B, S, NHK_ia, gqa_ia, D)
                .permute(0, 1, 2, 3, 4)
                .reshape(B * S * NHK_ia, gqa_ia, D)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            k_ffa = (
                k_raw.reshape(B * S * NHK_ia, 1, D)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            v_ffa = (
                v_raw.reshape(B * S * NHK_ia, 1, D)
                .detach()
                .clone()
                .requires_grad_(True)
            )

            o_sparse, _ = flex_flash_attn_func(
                q_ffa,
                k_ffa,
                v_ffa,
                index_attn_indices=indices,
                q_block_size=1,
                k_block_size=1,
                pack_gqa=True,
            )

            ref_mask = get_sdpa_mask_from_index_attn_indices(
                indices, B, NHQ_eff, 1, S_flat, S_flat, device
            )

            for b_i in range(B):
                sl = slice(b_i * S_flat, (b_i + 1) * S_flat)
                q_b = q_ffa[sl].detach().reshape(1, S_flat, NHQ_eff, D).transpose(1, 2)
                k_b = k_ffa[sl].detach().reshape(1, S_flat, 1, D).transpose(1, 2)
                v_b = v_ffa[sl].detach().reshape(1, S_flat, 1, D).transpose(1, 2)
                if NHQ_eff > 1:
                    k_b = k_b.expand(1, NHQ_eff, S_flat, D)
                    v_b = v_b.expand(1, NHQ_eff, S_flat, D)
                with torch.no_grad():
                    o_ref = torch.nn.functional.scaled_dot_product_attention(
                        q_b, k_b, v_b, attn_mask=ref_mask[b_i].unsqueeze(0)
                    )
                o_ref = o_ref.squeeze(0).transpose(0, 1)
                max_diff = (o_sparse[sl].float() - o_ref.float()).abs().max().item()
                assert max_diff < 0.01, (
                    f"[test_inner_dir_min_to_max/index_attn] "
                    f"FWD batch {b_i}: max_diff={max_diff:.6f} >= 0.01"
                )

            # BWD check
            do_sparse = torch.randn_like(o_sparse)
            o_sparse.backward(do_sparse)
            assert (
                q_ffa.grad is not None
            ), "[test_inner_dir_min_to_max/index_attn] BWD: q_ffa.grad is None"

        finally:
            del os.environ["MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"]
            if hasattr(get_ffa_jit_mod, "cache_clear"):
                get_ffa_jit_mod.cache_clear()

    # ─── UseMaskDispatch=false unit test ───

    def test_use_mask_dispatch_off(self):
        """Verify Dense FWD+BWD passes with UseMaskDispatch=false (original mask loop)."""
        import os

        from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_mod

        device = self.device
        torch.manual_seed(42)

        os.environ["MAGI_ATTENTION_FFA_USE_MASK_DISPATCH"] = "false"
        if hasattr(get_ffa_jit_mod, "cache_clear"):
            get_ffa_jit_mod.cache_clear()
        try:
            S_q, S_k, NHQ, NHK, head_dim = 256, 256, 4, 4, 128
            dtype = torch.bfloat16
            q = torch.randn(
                S_q, NHQ, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            k = torch.randn(
                S_k, NHK, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            v = torch.randn(
                S_k, NHK, head_dim, dtype=dtype, device=device, requires_grad=True
            )
            do = torch.randn(S_q, NHQ, head_dim, dtype=dtype, device=device)

            q_ranges = torch.tensor([[0, S_q]], dtype=torch.int32, device=device)
            k_ranges = torch.tensor([[0, S_k]], dtype=torch.int32, device=device)
            attn_type_map = torch.tensor([1], dtype=torch.int32, device=device)

            o, meta = flex_flash_attn_func(
                q=q,
                k=k,
                v=v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
            )
            o.backward(do)

            self.assert_close_to_torch_ref(
                q_ranges=AttnRanges.from_ranges([[0, S_q]]),
                k_ranges=AttnRanges.from_ranges([[0, S_k]]),
                attn_type_map=[1],
                total_seqlen_q=S_q,
                total_seqlen_k=S_k,
                total_q=q,
                total_k=k,
                total_v=v,
                total_sink=None,
                total_out=o,
                total_lse=meta.lse,
                grad_total_q=q.grad,
                grad_total_k=k.grad,
                grad_total_v=v.grad,
                grad_total_sink=None,
                grad_total_out=do,
                dtype=dtype,
                sink_layout="sh",
                test_case="[test_use_mask_dispatch_off]",
            )
        finally:
            del os.environ["MAGI_ATTENTION_FFA_USE_MASK_DISPATCH"]
            if hasattr(get_ffa_jit_mod, "cache_clear"):
                get_ffa_jit_mod.cache_clear()


if __name__ == "__main__":
    unittest.main()
