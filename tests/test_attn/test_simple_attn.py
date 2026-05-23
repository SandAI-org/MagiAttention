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

    # ─── index_attn_indices direct path (forward only) ───

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
    ]

    @parameterize("sparse_config", INDEX_ATTN_CONFIGS)
    def test_index_attn_indices_simple(self, sparse_config: dict[str, Any]):
        """Lightweight index_attn_indices test (forward only)."""
        from einops import rearrange as einops_rearrange

        from magi_attention.utils import set_random_seed

        set_random_seed(42)
        B = sparse_config["B"]
        S = sparse_config["S"]
        NHQ = sparse_config["NHQ"]
        NHK = sparse_config["NHK"]
        D = sparse_config["D"]
        topk = sparse_config["topk"]
        pack_gqa = sparse_config["pack_gqa"]

        device = self.device
        total_q = B * S

        indices = torch.full((total_q, NHK, topk), -1, dtype=torch.int32, device=device)
        for b in range(B):
            for qi in range(S):
                row = b * S + qi
                for h in range(NHK):
                    perm = torch.randperm(S, device=device)[:topk].sort().values
                    global_ids = (b * S + perm) * NHK + h
                    indices[row, h, :topk] = global_ids.int()

        q = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=device)

        q_ffa = einops_rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=NHK)
        k_ffa = einops_rearrange(k, "b s h d -> (b s h) 1 d")
        v_ffa = einops_rearrange(v, "b s h d -> (b s h) 1 d")

        with torch.no_grad():
            o_sparse, _ = flex_flash_attn_func(
                q_ffa.clone(),
                k_ffa.clone(),
                v_ffa.clone(),
                index_attn_indices=indices,
                q_block_size=1,
                k_block_size=1,
                pack_gqa=pack_gqa,
            )

        o_reshaped = einops_rearrange(
            o_sparse, "(b s h1) h2 d -> b s (h1 h2) d", b=B, h1=NHK, s=S
        )

        gqa = NHQ // NHK
        mask = torch.zeros(B, NHQ, S, S, dtype=torch.bool, device=device)
        for b in range(B):
            for qi in range(S):
                row = b * S + qi
                for h_kv in range(NHK):
                    global_ids = indices[row, h_kv, :]
                    valid = global_ids[global_ids >= 0].long()
                    local_kv = valid // NHK - b * S
                    for h_q_off in range(gqa):
                        h_q = h_kv * gqa + h_q_off
                        mask[b, h_q, qi, local_kv] = True

        for b in range(B):
            q_sdpa = einops_rearrange(q[b], "s h d -> 1 h s d")
            k_sdpa = einops_rearrange(k[b], "s h d -> 1 h s d")
            v_sdpa = einops_rearrange(v[b], "s h d -> 1 h s d")
            if gqa > 1:
                k_sdpa = k_sdpa.repeat_interleave(gqa, dim=1)
                v_sdpa = v_sdpa.repeat_interleave(gqa, dim=1)

            with torch.no_grad():
                o_ref = torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, attn_mask=mask[b].unsqueeze(0)
                )
            o_ref = einops_rearrange(o_ref, "1 h s d -> s h d")

            max_diff = (o_reshaped[b].float() - o_ref.float()).abs().max().item()
            assert max_diff < 0.01, (
                f"[test_index_attn_indices_simple][{sparse_config['name']}] "
                f"batch {b}: max_diff={max_diff:.6f} >= 0.01"
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
            pack_gqa=True,
            deterministic=False,
            test_accumulation_inplace=False,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            sparse_load=True,
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


if __name__ == "__main__":
    unittest.main()
