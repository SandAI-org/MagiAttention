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

import random
import time
import unittest
from datetime import datetime
from typing import Any

import pytest
import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnSinkLayout
from magi_attention.functional import flex_flash_attn_func
from magi_attention.functional.flex_flash_attn import (
    _flex_flash_attn_backward,
    _flex_flash_attn_forward,
    merge_ranges,
)
from magi_attention.functional.utils import correct_attn_out_lse
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.flag_generator import FlagCombGenerator
from magi_attention.testing.precision import (
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.utils import is_list_value_any, make_attn_mask_from_ffa_args


class TestFlexFlashAttn(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # all valid ref_block_config
        # Store as instance variable so we can access it later by index
        # NOTE: this may cause excessive compilation time.
        self.valid_ref_block_configs = [
            # general
            {
                "swap_ab": False,
                "ref_block_size": None,
                "pack_gqa": False,
                "block_sparse": False,
            },
            # pack_gqa
            {
                "swap_ab": False,
                "ref_block_size": (128, 128),
                "pack_gqa": True,
                "block_sparse": False,
            },
            # block_sparse
            {
                "swap_ab": False,
                "ref_block_size": (128, 128),
                "pack_gqa": False,
                "block_sparse": True,
            },
            {
                "swap_ab": False,
                "ref_block_size": (64, 128),
                "pack_gqa": False,
                "block_sparse": True,
            },
            # block_sparse & pack_gqa
            {
                "swap_ab": False,
                "ref_block_size": (64, 128),
                "pack_gqa": True,
                "block_sparse": True,
            },
            # block_sparse & swap_ab
            {
                "swap_ab": True,
                "ref_block_size": (16, 64),
                "pack_gqa": False,
                "block_sparse": True,
            },
            # swap_ab
            {
                "swap_ab": True,
                "ref_block_size": (8, 64),
                "pack_gqa": False,
                "block_sparse": False,
            },
            {
                "swap_ab": True,
                "ref_block_size": (16, 64),
                "pack_gqa": False,
                "block_sparse": False,
            },
            {
                "swap_ab": True,
                "ref_block_size": (32, 64),
                "pack_gqa": False,
                "block_sparse": False,
            },
            # swap_ab & pack_gqa
            {
                "swap_ab": True,
                "ref_block_size": (64, 64),
                "pack_gqa": True,
                "block_sparse": False,
            },
            # swap_ab & pack_gqa & block_sparse
            {
                "swap_ab": True,
                "ref_block_size": (64, 64),
                "pack_gqa": True,
                "block_sparse": True,
            },
        ]

        ref_block_config_indices = list(range(len(self.valid_ref_block_configs)))

        self.flag_generator = FlagCombGenerator(
            flags=[
                "test_accumulation_inplace",
                "deterministic",
                "auto_range_merge",
                "random_attn_type_map",
                "swap_bwd_qk_loop",
                "ref_block_config_idx",
                "max_seqlen_q",
                "return_max_logits",
                "cat_gqa",
            ],
            options={
                "ref_block_config_idx": ref_block_config_indices,
            },
            defaults={
                "ref_block_config_idx": 0,
            },
            groups=[("auto_range_merge", "swap_bwd_qk_loop")],
            strategy="heuristic",
        )
        self.flag_iterator = iter(self.flag_generator)

    # -- Kernel precompile for CI --
    _REF_BLOCK_CONFIGS: list[dict] = [
        {
            "swap_ab": False,
            "ref_block_size": None,
            "pack_gqa": False,
            "block_sparse": False,
        },
        {
            "swap_ab": False,
            "ref_block_size": (128, 128),
            "pack_gqa": True,
            "block_sparse": False,
        },
        {
            "swap_ab": False,
            "ref_block_size": (128, 128),
            "pack_gqa": False,
            "block_sparse": True,
        },
        {
            "swap_ab": False,
            "ref_block_size": (64, 128),
            "pack_gqa": False,
            "block_sparse": True,
        },
        {
            "swap_ab": False,
            "ref_block_size": (64, 128),
            "pack_gqa": True,
            "block_sparse": True,
        },
        {
            "swap_ab": True,
            "ref_block_size": (16, 64),
            "pack_gqa": False,
            "block_sparse": True,
        },
        {
            "swap_ab": True,
            "ref_block_size": (8, 64),
            "pack_gqa": False,
            "block_sparse": False,
        },
        {
            "swap_ab": True,
            "ref_block_size": (16, 64),
            "pack_gqa": False,
            "block_sparse": False,
        },
        {
            "swap_ab": True,
            "ref_block_size": (32, 64),
            "pack_gqa": False,
            "block_sparse": False,
        },
        {
            "swap_ab": True,
            "ref_block_size": (64, 64),
            "pack_gqa": True,
            "block_sparse": False,
        },
        {
            "swap_ab": True,
            "ref_block_size": (64, 64),
            "pack_gqa": True,
            "block_sparse": True,
        },
    ]
    _HEAD_DIMS = [64, 128]
    _PACK_GQA_FACTORS = [1, 2, 8]

    @classmethod
    def precompile_kernel_specs(cls) -> dict:
        from itertools import product

        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}
        _DTYPES = [torch.bfloat16, torch.float16]
        _PGF = cls._PACK_GQA_FACTORS  # [1, 2, 8]

        def _pgf(hd: int) -> int:
            return 2 if hd == 64 else 8

        # ═══════════════════════════════════════════════════════════════════
        # Section 1: Base tile configs (from _REF_BLOCK_CONFIGS)
        # ═══════════════════════════════════════════════════════════════════
        for hd, dt, cfg in product(cls._HEAD_DIMS, _DTYPES, cls._REF_BLOCK_CONFIGS):
            swap_ab = cfg["swap_ab"]
            tile = cfg["ref_block_size"]
            pack_gqa = cfg["pack_gqa"]
            block_sparse = cfg["block_sparse"]
            pgf_list = _PGF if pack_gqa else [1]

            for pgf in pgf_list:
                for rml in [False, True]:
                    add_ffa_spec(
                        specs,
                        direction="fwd",
                        head_dim=hd,
                        compute_dtype=dt,
                        output_dtype=torch.float32,
                        ref_block_size=tile,
                        swap_ab=swap_ab,
                        pack_gqa=pack_gqa,
                        pack_gqa_factor=pgf,
                        block_sparse=block_sparse,
                        return_max_logits=rml,
                    )
                if not swap_ab:
                    add_ffa_spec(
                        specs,
                        direction="bwd",
                        head_dim=hd,
                        compute_dtype=dt,
                        disable_atomic=block_sparse,
                        pack_gqa=pack_gqa,
                        pack_gqa_factor=pgf,
                        block_sparse=block_sparse,
                    )
                    if not block_sparse:
                        for dda in [True, False]:
                            add_ffa_spec(
                                specs,
                                direction="bwd",
                                head_dim=hd,
                                compute_dtype=dt,
                                bwd_inner_loop_k=True,
                                disable_dq_atomic=dda,
                                pack_gqa=pack_gqa,
                                pack_gqa_factor=pgf,
                            )

        # ═══════════════════════════════════════════════════════════════════
        # Section 2: Feature combos (det, rm, rml) on dense kernels
        # ═══════════════════════════════════════════════════════════════════
        _FEATURE_COMBOS: list[dict] = [
            {"deterministic": True},
            {"return_max_logits": True},
            {"deterministic": True, "return_max_logits": True},
            {"range_merge": True},
            {"deterministic": True, "range_merge": True},
        ]
        for hd, dt, feat in product(cls._HEAD_DIMS, _DTYPES, _FEATURE_COMBOS):
            pgf = _pgf(hd)
            det = feat.get("deterministic", False)
            rm = feat.get("range_merge", False)
            rml = feat.get("return_max_logits", False)

            # FWD dense (base + disable_atomic + pack_gqa)
            for da in [False, True]:
                add_ffa_spec(
                    specs,
                    direction="fwd",
                    head_dim=hd,
                    compute_dtype=dt,
                    output_dtype=torch.float32,
                    disable_atomic=da,
                    deterministic=det,
                    range_merge=rm,
                    return_max_logits=rml,
                )
            for p in [1, pgf]:
                add_ffa_spec(
                    specs,
                    direction="fwd",
                    head_dim=hd,
                    compute_dtype=dt,
                    output_dtype=torch.float32,
                    pack_gqa=True,
                    pack_gqa_factor=p,
                    deterministic=det,
                    range_merge=rm,
                    return_max_logits=rml,
                )

            # BWD dense
            add_ffa_spec(
                specs,
                direction="bwd",
                head_dim=hd,
                compute_dtype=dt,
                deterministic=det,
                range_merge=rm,
            )
            for p in [1, pgf]:
                add_ffa_spec(
                    specs,
                    direction="bwd",
                    head_dim=hd,
                    compute_dtype=dt,
                    pack_gqa=p > 1,
                    pack_gqa_factor=p,
                    deterministic=det,
                    range_merge=rm,
                )
                if p == 1 and (det or rm):
                    add_ffa_spec(
                        specs,
                        direction="bwd",
                        head_dim=hd,
                        compute_dtype=dt,
                        pack_gqa=True,
                        pack_gqa_factor=1,
                        deterministic=det,
                        range_merge=rm,
                    )
            # BWD cat_gqa
            add_ffa_spec(
                specs,
                direction="bwd",
                head_dim=hd,
                compute_dtype=dt,
                cat_gqa=True,
                pack_gqa_factor=pgf,
                deterministic=det,
                range_merge=rm,
            )
            # BWD LoopK (only with range_merge)
            if rm:
                for p in [1, pgf]:
                    add_ffa_spec(
                        specs,
                        direction="bwd",
                        head_dim=hd,
                        compute_dtype=dt,
                        bwd_inner_loop_k=True,
                        disable_dq_atomic=True,
                        pack_gqa=p > 1,
                        pack_gqa_factor=p,
                        range_merge=True,
                    )
                    if p == 1:
                        for dda in [True, False]:
                            add_ffa_spec(
                                specs,
                                direction="bwd",
                                head_dim=hd,
                                compute_dtype=dt,
                                bwd_inner_loop_k=True,
                                disable_dq_atomic=dda,
                                pack_gqa=True,
                                pack_gqa_factor=1,
                                range_merge=True,
                            )

        # ═══════════════════════════════════════════════════════════════════
        # Section 3: FWD swap_ab + feature flags
        # ═══════════════════════════════════════════════════════════════════
        _SWAP_TILES = [(8, 64), (16, 64), (32, 64)]
        _SWAP_FEATURES: list[dict] = [
            {"deterministic": True},
            {"range_merge": True},
            {"deterministic": True, "range_merge": True},
            {"return_max_logits": True},
            {"deterministic": True, "return_max_logits": True},
            {"range_merge": True, "return_max_logits": True},
            {"deterministic": True, "range_merge": True, "return_max_logits": True},
        ]
        for hd, dt in product(cls._HEAD_DIMS, _DTYPES):
            pgf = _pgf(hd)
            for tile, feat in product(_SWAP_TILES, _SWAP_FEATURES):
                add_ffa_spec(
                    specs,
                    direction="fwd",
                    head_dim=hd,
                    compute_dtype=dt,
                    output_dtype=torch.float32,
                    swap_ab=True,
                    ref_block_size=tile,
                    deterministic=feat.get("deterministic", False),
                    range_merge=feat.get("range_merge", False),
                    return_max_logits=feat.get("return_max_logits", False),
                )
            # swap_ab + pack_gqa (tile (64,64))
            for p in [1, pgf]:
                for rml in [False, True]:
                    add_ffa_spec(
                        specs,
                        direction="fwd",
                        head_dim=hd,
                        compute_dtype=dt,
                        output_dtype=torch.float32,
                        swap_ab=True,
                        ref_block_size=(64, 64),
                        pack_gqa=True,
                        pack_gqa_factor=p,
                        deterministic=True,
                        return_max_logits=rml,
                    )
            # swap_ab + packgqa + deterministic + range_merge
            for p in [1, pgf]:
                add_ffa_spec(
                    specs,
                    direction="fwd",
                    head_dim=hd,
                    compute_dtype=dt,
                    output_dtype=torch.float32,
                    swap_ab=True,
                    ref_block_size=(64, 64),
                    pack_gqa=True,
                    pack_gqa_factor=p,
                    deterministic=True,
                    range_merge=True,
                )

        # ═══════════════════════════════════════════════════════════════════
        # Section 4: Block sparse BWD with range_merge
        # ═══════════════════════════════════════════════════════════════════
        _BS_BWD_CONFIGS: list[tuple[tuple[int, int], bool, int]] = [
            ((128, 128), False, 1),
            ((64, 128), True, 4),
        ]
        for hd, dt in product([128], _DTYPES):
            for rbs, bs_pack, bs_pgf in _BS_BWD_CONFIGS:
                add_ffa_spec(
                    specs,
                    direction="bwd",
                    head_dim=hd,
                    compute_dtype=dt,
                    disable_atomic=True,
                    block_sparse=True,
                    range_merge=True,
                    pack_gqa=bs_pack,
                    pack_gqa_factor=bs_pgf,
                    sparse_k_block_size=rbs[1] // 2,
                )
                if bs_pack:
                    add_ffa_spec(
                        specs,
                        direction="bwd",
                        head_dim=hd,
                        compute_dtype=dt,
                        disable_atomic=True,
                        block_sparse=True,
                        range_merge=True,
                        pack_gqa=True,
                        pack_gqa_factor=bs_pgf,
                    )

        # ═══════════════════════════════════════════════════════════════════
        # Section 5: FWD dense packgqa + range_merge + return_max_logits
        # ═══════════════════════════════════════════════════════════════════
        for hd, dt in product(cls._HEAD_DIMS, _DTYPES):
            pgf = _pgf(hd)
            for p in [1, pgf if hd == 64 else 1]:
                for det in [True, False]:
                    add_ffa_spec(
                        specs,
                        direction="fwd",
                        head_dim=hd,
                        compute_dtype=dt,
                        output_dtype=torch.float32,
                        pack_gqa=True,
                        pack_gqa_factor=p,
                        deterministic=det,
                        range_merge=True,
                        return_max_logits=True,
                    )
            add_ffa_spec(
                specs,
                direction="fwd",
                head_dim=hd,
                compute_dtype=dt,
                output_dtype=torch.float32,
                pack_gqa=True,
                pack_gqa_factor=1,
                range_merge=True,
                return_max_logits=True,
            )
            if hd == 64:
                add_ffa_spec(
                    specs,
                    direction="fwd",
                    head_dim=hd,
                    compute_dtype=dt,
                    output_dtype=torch.float32,
                    ref_block_size=(128, 128),
                    pack_gqa=True,
                    pack_gqa_factor=1,
                    range_merge=True,
                    return_max_logits=True,
                )

        # ═══════════════════════════════════════════════════════════════════
        # Section 6: Special cases
        # ═══════════════════════════════════════════════════════════════════
        # Index sparse BWD LoopK
        add_ffa_spec(
            specs,
            direction="bwd",
            head_dim=64,
            compute_dtype=torch.bfloat16,
            index_sparse=True,
            bwd_inner_loop_k=True,
            disable_dq_atomic=True,
            bwd_dq_bf16=True,
        )

        return specs

    @property
    def seed(self):
        return 40

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 8

    @property
    def timeout(self) -> int:
        return 4000

    def generate_non_overlapping_qk_pairs(
        self,
        total_seqlen_q: int,
        total_seqlen_k: int,
        num_pairs: int,
        min_len_q: int = 16,
        max_len_q: int = 128,
        min_len_k: int = 16,
        max_len_k: int = 128,
        max_consecutive_failures: int = 500,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Generates non-overlapping q_ranges and k_ranges on a potentially non-square attention area.

        The function attempts to generate a specified number of attention pairs, defined by `num_pairs`.
        It has two termination conditions:
        1. Success: When the number of generated pairs reaches `num_pairs`.
        2. Saturation: When it fails to find a new, non-overlapping (q,k) pair for a number of
        consecutive attempts (controlled by `max_consecutive_failures`), it terminates even if
        `num_pairs` has not been reached.

        Core Constraint (No Area Overlap):
        Imagine each (q_range, k_range) pair as a rectangle on a 2D plane. This function ensures
        that no two such generated rectangles have overlapping areas.

        Args:
            total_seqlen_q (int): The total sequence length of the Query vector.
            total_seqlen_k (int): The total sequence length of the Key vector.
            num_pairs (int): The target number of (q_range, k_range) pairs to generate.
            min_len_q (int): The minimum length for a q_range.
            max_len_q (int): The maximum length for a q_range.
            min_len_k (int): The minimum length for a k_range.
            max_len_k (int): The maximum length for a k_range.
            max_consecutive_failures (int):
                The upper limit for consecutive failed attempts, used to determine if the space is saturated.

        Returns:
            tuple[list[list[int]], list[list[int]]]:
            A tuple containing two lists: q_ranges and k_ranges.
        """
        q_ranges: list = []
        k_ranges: list = []

        consecutive_failures = 0

        def _create_one_random_q_range() -> list[int]:
            """Generates a random range [start, end] for the Query axis."""
            effective_max = min(max_len_q, total_seqlen_q)
            if min_len_q > effective_max:
                raise ValueError(
                    f"min_len_q ({min_len_q}) cannot be greater than its effective max ({effective_max})"
                )

            length = random.randint(min_len_q, effective_max)
            start = random.randint(0, total_seqlen_q - length)
            return [start, start + length]

        def _create_one_random_k_range() -> list[int]:
            """Generates a random range [start, end] for the Key axis."""
            effective_max = min(max_len_k, total_seqlen_k)
            if min_len_k > effective_max:
                raise ValueError(
                    f"min_len_k ({min_len_k}) cannot be greater than its effective max ({effective_max})"
                )

            length = random.randint(min_len_k, effective_max)
            start = random.randint(0, total_seqlen_k - length)
            return [start, start + length]

        # Main loop, continues until one of the termination conditions is met:
        # the target number of pairs is reached, or the consecutive failure threshold is exceeded.
        while (
            len(q_ranges) < num_pairs
            and consecutive_failures < max_consecutive_failures
        ):
            candidate_q_range = _create_one_random_q_range()
            candidate_k_range = _create_one_random_k_range()

            is_valid_placement = True

            for i in range(len(q_ranges)):
                q_axis_overlaps = (
                    candidate_q_range[0] < q_ranges[i][1]
                    and candidate_q_range[1] > q_ranges[i][0]
                )
                k_axis_overlaps = (
                    candidate_k_range[0] < k_ranges[i][1]
                    and candidate_k_range[1] > k_ranges[i][0]
                )

                if q_axis_overlaps and k_axis_overlaps:
                    is_valid_placement = False
                    break

            if is_valid_placement:
                q_ranges.append(candidate_q_range)
                k_ranges.append(candidate_k_range)
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            # Provide an informative printout at the end of the function.
            """
            if len(q_ranges) == num_pairs:
                print(
                    f"Successfully reached the target, generating {num_pairs} non-overlapping qk-ranges."
                )
            else:
                print(
                    f"Terminated after {max_consecutive_failures} consecutive failures. "
                    f"Target was {num_pairs} pairs, actually generated {len(q_ranges)} pairs."
                )
            """
        return q_ranges, k_ranges

    def check_deterministic(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor | None,
        sink_layout: AttnSinkLayout,
        do: torch.Tensor,
        q_ranges_tensor: torch.Tensor,
        k_ranges_tensor: torch.Tensor,
        attn_type_map_tensor: torch.Tensor,
        range_merge: bool,
        block_sparse: bool,
        o_ref: torch.Tensor,
        lse_ref: torch.Tensor,
        dq_ref: torch.Tensor,
        dk_ref: torch.Tensor,
        dv_ref: torch.Tensor,
        dsink_ref: torch.Tensor | None,
        swap_ab: bool,
        ref_block_size: tuple[int, int] | None,
        pack_gqa: bool,
        cat_gqa: bool,
        test_case: str,
    ) -> list[str]:
        """Check deterministic behavior
        in which we will compare the output and gradients with a second run
        and if any of them is not equal, we will collect the error messages to return
        """
        err_msg_list: list[str] = []

        q = q.clone().detach().requires_grad_(True)
        k = k.clone().detach().requires_grad_(True)
        v = v.clone().detach().requires_grad_(True)
        sink = sink.clone().detach().requires_grad_(True) if sink is not None else None
        do = do.clone()

        o, meta = flex_flash_attn_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=None,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map_tensor,
            sink=sink,
            sink_layout=sink_layout,
            range_merge=range_merge,
            deterministic=True,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            pack_gqa=pack_gqa,
            cat_gqa=cat_gqa,
            block_sparse=block_sparse,
        )
        lse = meta.lse
        o.backward(do)

        try:
            assert torch.equal(
                o, o_ref
            ), f"For {test_case=}: forward output not deterministic"

            assert torch.equal(
                lse, lse_ref
            ), f"For {test_case=}: forward lse not deterministic"

            assert torch.equal(
                q.grad, dq_ref
            ), f"For {test_case=}: backward dq not deterministic"

            assert torch.equal(
                k.grad, dk_ref
            ), f"For {test_case=}: backward dk not deterministic"

            assert torch.equal(
                v.grad, dv_ref
            ), f"For {test_case=}: backward dv not deterministic"

            if sink is not None:
                assert torch.equal(
                    sink.grad, dsink_ref
                ), f"For {test_case=}: backward dsink not deterministic"
        except Exception as e:
            err_msg_list.append(str(e))

        return err_msg_list

    def check_flex_flash_attn_accumulation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        q_ranges_tensor: torch.Tensor,
        k_ranges_tensor: torch.Tensor,
        attn_type_map_tensor: torch.Tensor,
        range_merge: bool,
        deterministic: bool,
        pack_gqa: bool,
        cat_gqa: bool,
        test_case: str,
        max_seqlen_q: int | None = None,
        swap_bwd_qk_loop: bool = False,
    ):
        t, h, d = q.shape
        o_acc = torch.randn_like(q, dtype=torch.float32)
        lse_acc = torch.randn([t, h], device=q.device, dtype=torch.float32)
        max_logits_acc = torch.randn([h], device=q.device, dtype=torch.float32)

        softmax_scale = 1.0 / (d**0.5)

        if range_merge:
            (
                merge_q_ranges,
                fwd_q_ranges,
                fwd_k_ranges,
                fwd_attn_type_map,
                fwd_qk_map,
                fwd_unique_count,
            ) = merge_ranges(q_ranges_tensor, k_ranges_tensor, attn_type_map_tensor)
            if swap_bwd_qk_loop:
                (
                    merge_k_ranges,
                    bwd_q_ranges,
                    bwd_k_ranges,
                    bwd_attn_type_map,
                    bwd_kq_map,
                    bwd_unique_count,
                ) = merge_ranges(
                    q_ranges_tensor, k_ranges_tensor, attn_type_map=attn_type_map_tensor
                )
            else:
                (
                    merge_k_ranges,
                    bwd_k_ranges,
                    bwd_q_ranges,
                    bwd_attn_type_map,
                    bwd_kq_map,
                    bwd_unique_count,
                ) = merge_ranges(
                    k_ranges_tensor, q_ranges_tensor, attn_type_map=attn_type_map_tensor
                )
        else:
            fwd_q_ranges = q_ranges_tensor
            fwd_k_ranges = k_ranges_tensor
            bwd_q_ranges = q_ranges_tensor
            bwd_k_ranges = k_ranges_tensor
            fwd_attn_type_map = attn_type_map_tensor
            bwd_attn_type_map = attn_type_map_tensor
            merge_q_ranges = None
            merge_k_ranges = None
            fwd_qk_map = None
            bwd_kq_map = None
            fwd_unique_count = None
            bwd_unique_count = None

        o, meta = _flex_flash_attn_forward(
            q=q,
            k=k,
            v=v,
            sink=None,
            sink_layout="sh",
            out=None,
            lse=None,
            q_ranges=fwd_q_ranges,
            k_ranges=fwd_k_ranges,
            attn_type_map=fwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=0.0,
            out_type=torch.float32,
            disable_fwd_atomic_reduction=False,
            deterministic=deterministic,
            sm_margin=0,
            # optional args below mainly for sparse attn
            ref_block_size=None,
            max_outer_range_width=max_seqlen_q,
            range_merge=range_merge,
            merge_q_ranges=merge_q_ranges,
            fwd_qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            swap_ab=False,
            pack_gqa=pack_gqa,
            block_sparse=False,
            return_max_logits=True,
            max_logits=None,
        )
        lse = meta.lse
        max_logits = meta.max_logits

        o_ref, lse_ref = correct_attn_out_lse(
            out1=o,
            lse1=lse,
            out2=o_acc,
            lse2=lse_acc,
        )

        # per-head max logits over score matrix
        max_logits_ref = torch.maximum(max_logits, max_logits_acc)

        # NOTE: The auto accumulation call must follow the non-auto accumulation call,
        # as the latter modifies the input tensors, and the former relies on these modified tensors.
        o_auto_acc, meta_auto_acc = _flex_flash_attn_forward(
            q=q,
            k=k,
            v=v,
            sink=None,
            sink_layout="sh",
            out=o_acc,
            lse=lse_acc,
            q_ranges=fwd_q_ranges,
            k_ranges=fwd_k_ranges,
            attn_type_map=fwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=0.0,
            out_type=None,
            disable_fwd_atomic_reduction=False,
            deterministic=deterministic,
            sm_margin=0,
            # optional args below mainly for sparse attn
            ref_block_size=None,
            max_outer_range_width=max_seqlen_q,
            range_merge=range_merge,
            merge_q_ranges=merge_q_ranges,
            fwd_qk_map=fwd_qk_map,
            fwd_unique_count=fwd_unique_count,
            swap_ab=False,
            pack_gqa=pack_gqa,
            block_sparse=False,
            return_max_logits=True,
            max_logits=max_logits_acc,
        )
        lse_auto_acc = meta_auto_acc.lse
        max_logits_auto_acc = meta_auto_acc.max_logits

        assert_close(
            o_auto_acc,
            o_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => o",
            print_rank=-1,
        )
        assert_close(
            lse_auto_acc,
            lse_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => lse",
            print_rank=-1,
        )
        assert_close(
            max_logits_auto_acc,
            max_logits_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => max_logits",
            print_rank=-1,
        )

        dq_acc = torch.randn_like(q, dtype=torch.float32)
        dk_acc = torch.randn_like(k, dtype=torch.float32)
        dv_acc = torch.randn_like(v, dtype=torch.float32)

        dq_ref, dk_ref, dv_ref, _ = _flex_flash_attn_backward(
            do,
            q,
            k,
            v,
            None,  # sink
            "sh",  # sink_layout
            o_ref.to(q.dtype),
            lse_ref,
            None,  # dq
            None,  # dk
            None,  # dv
            None,  # dsink
            bwd_q_ranges,
            bwd_k_ranges,
            bwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,  # TODO: test when it's `True`
            disable_bwd_dq_atomic_reduction=False,
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
            range_merge=range_merge,
            merge_k_ranges=merge_k_ranges,
            bwd_kq_map=bwd_kq_map,
            bwd_unique_count=bwd_unique_count,
            bwd_inner_loop_k=swap_bwd_qk_loop,
            pack_gqa=pack_gqa,
            cat_gqa=cat_gqa,
        )

        dq_ref += dq_acc
        dk_ref += dk_acc
        dv_ref += dv_acc

        dq_acc, dk_acc, dv_acc, _ = _flex_flash_attn_backward(
            do,
            q,
            k,
            v,
            None,  # sink
            "sh",  # sink_layout
            o_ref.to(q.dtype),
            lse_ref,
            dq_acc,  # dq
            dk_acc,  # dk
            dv_acc,  # dv
            None,  # dsink
            bwd_q_ranges,
            bwd_k_ranges,
            bwd_attn_type_map,
            softmax_scale=softmax_scale,
            softcap=0.0,
            disable_bwd_dkv_atomic_reduction=False,  # TODO: test when it's `True`
            disable_bwd_dq_atomic_reduction=False,
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=deterministic,
            sm_margin=0,
            range_merge=range_merge,
            merge_k_ranges=merge_k_ranges,
            bwd_kq_map=bwd_kq_map,
            bwd_unique_count=bwd_unique_count,
            bwd_inner_loop_k=swap_bwd_qk_loop,
            pack_gqa=pack_gqa,
            cat_gqa=cat_gqa,
        )

        assert_close(
            dq_acc,
            dq_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dq",
            print_rank=-1,
        )
        assert_close(
            dk_acc,
            dk_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dk",
            print_rank=-1,
        )
        assert_close(
            dv_acc,
            dv_ref,
            atol=1e-5,
            rtol=1e-4,
            mismatch_threshold=0.005,
            test_case=f"{test_case} => dv",
            print_rank=-1,
        )

    def assert_close_to_torch_ref(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_type_map: list[int],
        total_seqlen_q: int,
        total_seqlen_k: int,
        total_q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_sink: torch.Tensor | None,
        total_out: torch.Tensor,
        total_lse: torch.Tensor,
        grad_total_q: torch.Tensor,
        grad_total_k: torch.Tensor,
        grad_total_v: torch.Tensor,
        grad_total_sink: torch.Tensor | None,
        grad_total_out: torch.Tensor,
        dtype: torch.dtype,
        sink_layout: AttnSinkLayout,
        test_case: str = "",
        err_msg_list: list[str] = [],
        err_ratio_dict: dict[str, float] = {},
        max_seqlen_q: int | None = None,
        total_max_logits: torch.Tensor | None = None,
        return_max_logits: bool = False,
    ) -> None:
        # -----   customize tolerance / threshold  ---- #

        if total_sink is not None:
            assert isinstance(total_sink, torch.Tensor)
            has_sink = True
        else:
            has_sink = False

        o_atol = EPSILON
        o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        o_norm_rtol_ratio = err_ratio_dict.get("o_norm_rtol_ratio", NORM_RTOL_RATIO)
        o_min_norm_rtol = err_ratio_dict.get("o_min_norm_rtol", 0.0)
        o_mismatch_thres_ratio = err_ratio_dict.get(
            "o_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        o_min_mismatch_thres = err_ratio_dict.get("o_min_mismatch_thres", 0.0)
        o_max_mismatch_thres = err_ratio_dict.get(
            "o_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        lse_atol = EPSILON
        lse_rtol = 0.001
        lse_norm_rtol_ratio = err_ratio_dict.get("lse_norm_rtol_ratio", NORM_RTOL_RATIO)
        lse_min_norm_rtol = err_ratio_dict.get("lse_min_norm_rtol", 0.0)
        lse_mismatch_thres_ratio = err_ratio_dict.get(
            "lse_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        lse_min_mismatch_thres = err_ratio_dict.get("lse_min_mismatch_thres", 0.0)
        lse_max_mismatch_thres = err_ratio_dict.get(
            "lse_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dq_atol = EPSILON
        dq_rtol = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)
        dq_norm_rtol_ratio = err_ratio_dict.get("dq_norm_rtol_ratio", NORM_RTOL_RATIO)
        dq_min_norm_rtol = err_ratio_dict.get("dq_min_norm_rtol", 0.0)
        dq_mismatch_thres_ratio = err_ratio_dict.get(
            "dq_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dq_min_mismatch_thres = err_ratio_dict.get("dq_min_mismatch_thres", 0.0)
        dq_max_mismatch_thres = err_ratio_dict.get(
            "dq_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dk_atol = EPSILON
        dk_rtol = {torch.bfloat16: 0.15, torch.float16: 0.08}.get(dtype, 0.08)
        dk_norm_rtol_ratio = err_ratio_dict.get("dk_norm_rtol_ratio", NORM_RTOL_RATIO)
        dk_min_norm_rtol = err_ratio_dict.get("dk_min_norm_rtol", 0.0)
        dk_mismatch_thres_ratio = err_ratio_dict.get(
            "dk_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dk_min_mismatch_thres = err_ratio_dict.get("dk_min_mismatch_thres", 0.0)
        dk_max_mismatch_thres = err_ratio_dict.get(
            "dk_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dv_atol = EPSILON
        dv_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        dv_norm_rtol_ratio = err_ratio_dict.get("dv_norm_rtol_ratio", NORM_RTOL_RATIO)
        dv_min_norm_rtol = err_ratio_dict.get("dv_min_norm_rtol", 0.0)
        dv_mismatch_thres_ratio = err_ratio_dict.get(
            "dv_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dv_min_mismatch_thres = err_ratio_dict.get("dv_min_mismatch_thres", 0.0)
        dv_max_mismatch_thres = err_ratio_dict.get(
            "dv_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dsink_atol = err_ratio_dict.get("dsink_atol", EPSILON)
        dsink_rtol = err_ratio_dict.get("dsink_rtol", 0.05)
        dsink_norm_rtol_ratio = err_ratio_dict.get(
            "dsink_norm_rtol_ratio", NORM_RTOL_RATIO
        )
        dsink_min_norm_rtol = err_ratio_dict.get("dsink_min_norm_rtol", 0.0)
        dsink_mismatch_thres_ratio = err_ratio_dict.get(
            "dsink_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dsink_min_mismatch_thres = err_ratio_dict.get("dsink_min_mismatch_thres", 0.0)
        dsink_max_mismatch_thres = err_ratio_dict.get(
            "dsink_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        max_logits_atol = err_ratio_dict.get("max_logits_atol", EPSILON)
        max_logits_rtol = err_ratio_dict.get("max_logits_rtol", 0.001)

        # -----   build attn mask   ---- #

        mask = make_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=self.device,
        )

        # -----   ref1. torch ref with high precision (fp64)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None
        if has_sink:
            total_sink.grad = None

        total_out_ref_high_precision, total_meta_ref_high_precision = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            sink=total_sink,
            mask=mask,
            layout="thd",
            sink_layout=sink_layout,
            high_precision=True,
            backend="torch" if has_sink else "sdpa",
            return_lse=True,
            return_max_logits=return_max_logits,
        )
        total_lse_ref_high_precision = total_meta_ref_high_precision.lse
        assert total_lse_ref_high_precision is not None
        total_out_ref_high_precision.backward(grad_total_out)
        (
            grad_total_q_ref_high_precision,
            grad_total_k_ref_high_precision,
            grad_total_v_ref_high_precision,
            grad_total_sink_ref_high_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
            total_sink.grad if has_sink else None,
        )

        # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None
        if has_sink:
            total_sink.grad = None

        total_out_ref_low_precision, total_meta_ref_low_precision = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            sink=total_sink,
            mask=mask,
            layout="thd",
            sink_layout=sink_layout,
            backend="torch" if has_sink else "sdpa",
            high_precision=False,
            return_lse=True,
            return_max_logits=return_max_logits,
        )
        total_lse_ref_low_precision = total_meta_ref_low_precision.lse
        assert total_lse_ref_low_precision is not None

        total_out_ref_low_precision.backward(grad_total_out)
        (
            grad_total_q_ref_low_precision,
            grad_total_k_ref_low_precision,
            grad_total_v_ref_low_precision,
            grad_total_sink_ref_low_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
            total_sink.grad if has_sink else None,
        )

        # -----   assert close for fwd out   ---- #

        # fa style with Linf norm
        out_norm = calc_inf_norm(total_out, total_out_ref_high_precision)
        out_ref_norm = calc_inf_norm(
            total_out_ref_low_precision, total_out_ref_high_precision
        )
        try:
            self.assertLessEqual(
                out_norm,
                max(o_min_norm_rtol, o_norm_rtol_ratio * out_ref_norm),
                msg=(
                    f"For {test_case=}: {out_norm=} should be no greater than "
                    f"max({o_min_norm_rtol}, {o_norm_rtol_ratio} x {out_ref_norm=})",
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        o_thres = extract_mismatch_threshold(
            actual=total_out_ref_low_precision,
            expected=total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=o_mismatch_thres_ratio,
            min_mismatch_thres=o_min_mismatch_thres,
            max_mismatch_thres=o_max_mismatch_thres,
        )
        try:
            assert_close(
                total_out,
                total_out_ref_high_precision,
                atol=o_atol,
                rtol=o_rtol,
                mismatch_threshold=o_thres,
                test_case=f"{test_case} => o",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for fwd lse   ---- #

        # fa style with Linf norm
        lse_norm = calc_inf_norm(total_lse, total_lse_ref_high_precision)
        lse_ref_norm = calc_inf_norm(
            total_lse_ref_low_precision, total_lse_ref_high_precision
        )

        try:
            self.assertLessEqual(
                lse_norm,
                max(lse_min_norm_rtol, lse_norm_rtol_ratio * lse_ref_norm),
                msg=(
                    f"For {test_case=}: {lse_norm=} should be no greater than "
                    f"max({lse_min_norm_rtol}, {lse_norm_rtol_ratio} x {lse_ref_norm=})"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        lse_thres = extract_mismatch_threshold(
            actual=total_lse_ref_low_precision,
            expected=total_lse_ref_high_precision,
            atol=lse_atol,
            rtol=lse_rtol,
            mismatch_thres_ratio=lse_mismatch_thres_ratio,
            min_mismatch_thres=lse_min_mismatch_thres,
            max_mismatch_thres=lse_max_mismatch_thres,
        )
        try:
            assert_close(
                total_lse,
                total_lse_ref_high_precision,
                atol=lse_atol,
                rtol=lse_rtol,
                mismatch_threshold=lse_thres,
                test_case=f"{test_case} => lse",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for fwd max_logits (when return_max_logits)   ---- #

        if return_max_logits and total_max_logits is not None:
            max_logits_ref = total_meta_ref_high_precision.max_logits
            assert max_logits_ref is not None, "ref max_logits should be computed"
            try:
                assert_close(
                    total_max_logits,
                    max_logits_ref,
                    atol=max_logits_atol,
                    rtol=max_logits_rtol,
                    mismatch_threshold=0.005,
                    test_case=f"{test_case} => max_logits",
                    print_rank=-1,
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   assert close for bwd dq   ---- #

        # fa style with Linf norm
        dq_norm = calc_inf_norm(grad_total_q, grad_total_q_ref_high_precision)
        dq_ref_norm = calc_inf_norm(
            grad_total_q_ref_low_precision, grad_total_q_ref_high_precision
        )
        try:
            self.assertLessEqual(
                dq_norm,
                max(dq_min_norm_rtol, dq_norm_rtol_ratio * dq_ref_norm),
                msg=(
                    f"For {test_case=}: {dq_norm=} should be no greater than "
                    f"max({dq_min_norm_rtol}, {dq_norm_rtol_ratio} x {dq_ref_norm=})"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dq_thres = extract_mismatch_threshold(
            actual=grad_total_q_ref_low_precision,
            expected=grad_total_q_ref_high_precision,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_thres_ratio=dq_mismatch_thres_ratio,
            min_mismatch_thres=dq_min_mismatch_thres,
            max_mismatch_thres=dq_max_mismatch_thres,
        )
        try:
            assert_close(
                grad_total_q,
                grad_total_q_ref_high_precision,
                atol=dq_atol,
                rtol=dq_rtol,
                mismatch_threshold=dq_thres,
                test_case=f"{test_case} => dq",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dk   ---- #

        # fa style with Linf norm
        dk_norm = calc_inf_norm(grad_total_k, grad_total_k_ref_high_precision)
        dk_ref_norm = calc_inf_norm(
            grad_total_k_ref_low_precision, grad_total_k_ref_high_precision
        )
        try:
            self.assertLessEqual(
                dk_norm,
                max(dk_min_norm_rtol, dk_norm_rtol_ratio * dk_ref_norm),
                msg=(
                    f"For {test_case=}: {dk_norm=} should be no greater than "
                    f"max({dk_min_norm_rtol}, {dk_norm_rtol_ratio} x {dk_ref_norm=})"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dk_thres = extract_mismatch_threshold(
            actual=grad_total_k_ref_low_precision,
            expected=grad_total_k_ref_high_precision,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_thres_ratio=dk_mismatch_thres_ratio,
            min_mismatch_thres=dk_min_mismatch_thres,
            max_mismatch_thres=dk_max_mismatch_thres,
        )
        try:
            assert_close(
                grad_total_k,
                grad_total_k_ref_high_precision,
                atol=dk_atol,
                rtol=dk_rtol,
                mismatch_threshold=dk_thres,
                test_case=f"{test_case} => dk",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dv   ---- #

        # fa style with Linf norm
        dv_norm = calc_inf_norm(grad_total_v, grad_total_v_ref_high_precision)
        dv_ref_norm = calc_inf_norm(
            grad_total_v_ref_low_precision, grad_total_v_ref_high_precision
        )
        try:
            self.assertLessEqual(
                dv_norm,
                max(dv_min_norm_rtol, dv_norm_rtol_ratio * dv_ref_norm),
                msg=(
                    f"For {test_case=}: {dv_norm=} should be no greater than "
                    f"max({dv_min_norm_rtol}, {dv_norm_rtol_ratio} x {dv_ref_norm=})"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        dv_thres = extract_mismatch_threshold(
            actual=grad_total_v_ref_low_precision,
            expected=grad_total_v_ref_high_precision,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_thres_ratio=dv_mismatch_thres_ratio,
            min_mismatch_thres=dv_min_mismatch_thres,
            max_mismatch_thres=dv_max_mismatch_thres,
        )
        try:
            assert_close(
                grad_total_v,
                grad_total_v_ref_high_precision,
                atol=dv_atol,
                rtol=dv_rtol,
                mismatch_threshold=dv_thres,
                test_case=f"{test_case} => dv",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for bwd dsink   ---- #

        if has_sink:
            # fa style with Linf norm
            dsink_norm = calc_inf_norm(
                grad_total_sink, grad_total_sink_ref_high_precision
            )
            dsink_ref_norm = calc_inf_norm(
                grad_total_sink_ref_low_precision, grad_total_sink_ref_high_precision
            )
            try:
                self.assertLessEqual(
                    dsink_norm,
                    max(dsink_min_norm_rtol, dsink_norm_rtol_ratio * dsink_ref_norm),
                    msg=(
                        f"For {test_case=}: {dsink_norm=} should be no greater than "
                        f"max({dsink_min_norm_rtol}, {dsink_norm_rtol_ratio} x {dsink_ref_norm=})"
                    ),
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            dsink_thres = extract_mismatch_threshold(
                actual=grad_total_sink_ref_low_precision,
                expected=grad_total_sink_ref_high_precision,
                atol=dsink_atol,
                rtol=dsink_rtol,
                mismatch_thres_ratio=dsink_mismatch_thres_ratio,
                min_mismatch_thres=dsink_min_mismatch_thres,
                max_mismatch_thres=dsink_max_mismatch_thres,
            )
            try:
                assert_close(
                    grad_total_sink,
                    grad_total_sink_ref_high_precision,
                    atol=dsink_atol,
                    rtol=dsink_rtol,
                    mismatch_threshold=dsink_thres,
                    test_case=f"{test_case} => dsink",
                    print_rank=-1,
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   raise error if any error occurs   ---- #

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

    def run_test_case(
        self,
        seqlen_q: int,
        seqlen_kv: int,
        seqlen_sink: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        dtype: torch.dtype,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_type_map: list[int],
        range_merge: bool,
        deterministic: bool,
        test_accumulation_inplace: bool,
        block_sparse: bool,
        sink_layout: AttnSinkLayout,
        swap_ab: bool,
        ref_block_size: tuple[int, int] | None,
        pack_gqa: bool,
        test_case: str,
        swap_bwd_qk_loop: bool = False,
        err_ratio_dict: dict[str, float] = {},
        max_seqlen_q: int | None = None,
        return_max_logits: bool = False,
        cat_gqa: bool = False,
    ) -> None:
        if block_sparse:  # sparse load supports only range_merge and full attn_type
            if not range_merge or test_accumulation_inplace:
                return
            for attn_type in attn_type_map:
                if attn_type != 0:
                    return
            # sparse load only applies to swapped backward QK loop
            if not swap_bwd_qk_loop:
                return

        # FIXME: for square bi-causal mask, i.e. when only the main diagonal is valid
        # ffa bwd kernel encounters with some precision issue with dq/dk,
        # thus we skip here and will fix it asap
        if is_list_value_any(attn_type_map, 3):
            return

        # construct data
        q = torch.randn(
            (seqlen_q, num_heads_q, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        k = torch.randn(
            (seqlen_kv, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        v = torch.randn(
            (seqlen_kv, num_heads_kv, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        do = torch.randn_like(q)
        has_sink = seqlen_sink > 0
        if has_sink:
            match sink_layout:
                case "sh":
                    sink = torch.randn(
                        (seqlen_sink, num_heads_q),
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=True,
                    )
                case "ssh":
                    sink = torch.randn(
                        (seqlen_q, seqlen_sink, num_heads_q),
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=True,
                    )
                case "shd":
                    raise NotImplementedError(
                        f"sink_layout {sink_layout} is not supported yet"
                    )
                case _:
                    raise ValueError(f"Invalid sink_layout {sink_layout}")
        else:
            sink = None

        # construct meta args
        q_ranges_tensor = q_ranges.to_tensor(device=self.device)
        k_ranges_tensor = k_ranges.to_tensor(device=self.device)
        attn_type_map_tensor = torch.tensor(
            attn_type_map, dtype=torch.int32, device=self.device
        )

        if test_accumulation_inplace:
            # If test_accumulation_inplace is True, we will test the accumulation and return
            self.check_flex_flash_attn_accumulation(
                q=q,
                k=k,
                v=v,
                do=do,
                q_ranges_tensor=q_ranges_tensor,
                k_ranges_tensor=k_ranges_tensor,
                attn_type_map_tensor=attn_type_map_tensor,
                range_merge=range_merge,
                deterministic=deterministic,
                pack_gqa=pack_gqa,
                cat_gqa=cat_gqa,
                test_case=test_case,
                max_seqlen_q=max_seqlen_q,
                swap_bwd_qk_loop=swap_bwd_qk_loop,
            )
            return

        # run ffa forward
        # TMA descriptor reuse: if block_sparse + TMA, stale descriptors from prior
        # allocations can cause garbage output. Isolate with empty_cache().
        if block_sparse:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        o, meta = flex_flash_attn_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=max_seqlen_q,
            q_ranges=q_ranges_tensor,
            k_ranges=k_ranges_tensor,
            attn_type_map=attn_type_map_tensor,
            sink=sink,
            sink_layout=sink_layout,
            range_merge=range_merge,
            deterministic=deterministic,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            pack_gqa=pack_gqa,
            cat_gqa=cat_gqa,
            block_sparse=block_sparse,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
            return_max_logits=return_max_logits,
        )
        lse = meta.lse
        max_logits = meta.max_logits if return_max_logits else None

        # run ffa backward
        o.backward(do)

        err_msg_list = []

        # if deterministic is True, check deterministic behavior and return error messages
        if deterministic:
            err_msg_list = self.check_deterministic(
                q=q,
                k=k,
                v=v,
                sink=sink,
                sink_layout=sink_layout,
                do=do,
                q_ranges_tensor=q_ranges_tensor,
                k_ranges_tensor=k_ranges_tensor,
                attn_type_map_tensor=attn_type_map_tensor,
                range_merge=range_merge,
                block_sparse=block_sparse,
                o_ref=o,
                lse_ref=lse,
                dq_ref=q.grad,
                dk_ref=k.grad,
                dv_ref=v.grad,
                dsink_ref=sink.grad if has_sink else None,
                swap_ab=swap_ab,
                ref_block_size=ref_block_size,
                pack_gqa=pack_gqa,
                cat_gqa=cat_gqa,
                test_case=test_case,
            )

        # compare with reference
        self.assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=seqlen_q,
            total_seqlen_k=seqlen_kv,
            total_q=q,
            total_k=k,
            total_v=v,
            total_sink=sink,
            total_out=o,
            total_lse=lse,
            grad_total_q=q.grad,
            grad_total_k=k.grad,
            grad_total_v=v.grad,
            grad_total_sink=sink.grad if has_sink else None,
            grad_total_out=do,
            dtype=dtype,
            sink_layout=sink_layout,
            test_case=test_case,
            err_msg_list=err_msg_list,
            err_ratio_dict=err_ratio_dict,
            max_seqlen_q=max_seqlen_q,
            total_max_logits=max_logits,
            return_max_logits=return_max_logits,
        )

    MODEL_CONFIGS = [
        {
            "name": "mha_nh8_hd128",
            "num_heads_q": 8,
            "num_heads_kv": 8,
            "head_dim": 128,
        },
        {
            "name": "gqa_nhq32_nhkv4_hd128",
            "num_heads_q": 32,
            "num_heads_kv": 4,
            "head_dim": 128,
        },
        {
            "name": "mha_nh1_hd64",
            "num_heads_q": 1,
            "num_heads_kv": 1,
            "head_dim": 64,
        },
        {
            "name": "gqa_nhq4_nhkv2_hd64",
            "num_heads_q": 4,
            "num_heads_kv": 2,
            "head_dim": 64,
        },
    ]

    @with_run_in_mp
    @parameterize(
        "attn_mask_config",
        [
            {
                "name": "full_4k",
                "seqlen": 4096,
                "seqlen_sink": 1,
                "sink_layout": "ssh",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 4096],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 4096],
                    ]
                ),
                "attn_type_map": [0],
            },
            {
                "name": "varlen_full_1k",
                "seqlen": 1024,
                "seqlen_sink": 2,
                "sink_layout": "sh",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 366],
                        [366, 391],
                        [391, 471],
                        [471, 835],
                        [835, 984],
                        [984, 1005],
                        [1005, 1017],
                        [1017, 1020],
                        [1020, 1023],
                        [1023, 1024],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 366],
                        [366, 391],
                        [391, 471],
                        [471, 835],
                        [835, 984],
                        [984, 1005],
                        [1005, 1017],
                        [1017, 1020],
                        [1020, 1023],
                        [1023, 1024],
                    ]
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "varlen_full_4k",
                "seqlen": 4096,
                "seqlen_sink": 4,
                "sink_layout": "ssh",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                        [2048, 4096],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                        [2048, 4096],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "block_causal_2k",
                "seqlen": 2048,
                "seqlen_sink": 6,
                "sink_layout": "sh",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 512],
                        [0, 1024],
                        [0, 1280],
                        [0, 1536],
                        [0, 1792],
                        [0, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "varlen_block_causal_2k",
                "seqlen": 2048,
                "seqlen_sink": 8,
                "sink_layout": "ssh",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 512],
                        [0, 1024],
                        [1024, 1280],
                        [1024, 1536],
                        [1024, 1792],
                        [1024, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "sparse_attn_2k",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 256],
                        [0, 256],
                        [256, 512],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [512, 768],
                        [1011, 1123],
                        [0, 512],
                        [777, 888],
                        [0, 1024],
                        [1024, 1280],
                        [0, 128],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],
                        [1024, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "varlen_block_causal_2k_with_disjoint_ranges",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 512],
                        [0, 1024],
                        [1024, 1280],
                        [1024, 1536],
                        [1024, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0],
            },
            {
                "name": "sparse_attn_2k_with_disjoint_ranges",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 256],
                        [0, 256],
                        [256, 512],
                        [256, 512],
                        [1024, 1280],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [512, 768],
                        [1011, 1123],
                        [0, 512],
                        [777, 888],
                        [1024, 1280],
                        [0, 128],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],
                        [1024, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "name": "deterministic_sample",
                "seqlen": 2500,
                "q_ranges": AttnRanges.from_ranges(
                    [[i * 50, (i + 1) * 50] for i in range(50) for j in range(50)]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [[i * 50, (i + 1) * 50] for i in range(50)] * 50
                ),
                "attn_type_map": [0, 1] * 1250,
            },
            {
                "name": "sparse_attn_2k_with_same_k_ranges",
                "seqlen": 2048,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 256],
                        [0, 256],
                        [256, 512],
                        [256, 512],
                        [1024, 1280],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [512, 768],
                        [1011, 1123],
                        [0, 256],
                        [777, 888],
                        [1024, 1536],
                        [0, 128],
                        [555, 556],
                        [777, 982],
                        [1024, 1536],
                        [1689, 1898],
                        [1024, 1792],
                        [1024, 2048],
                    ],
                ),
                "attn_type_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        ],
    )
    @parameterize("model_config", MODEL_CONFIGS)
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    def test_ffa_simple(
        self,
        attn_mask_config: dict[str, Any],
        model_config: dict[str, Any],
        dtype: torch.dtype,
    ):
        # -----    switch env flags by FlagCombGenerator   ---- #

        flag_comb = next(self.flag_iterator)
        flag_comb_test_case = FlagCombGenerator.to_test_case(flag_comb)

        # extract config
        seqlen: int = attn_mask_config["seqlen"]
        seqlen_sink: int = attn_mask_config.get("seqlen_sink", 0)
        sink_layout: AttnSinkLayout = attn_mask_config.get("sink_layout", "sh")
        num_heads_q: int = model_config["num_heads_q"]
        q_ranges: AttnRanges = attn_mask_config["q_ranges"]
        k_ranges: AttnRanges = attn_mask_config["k_ranges"]
        attn_type_map: list[int] = attn_mask_config["attn_type_map"]
        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        assert len(q_ranges) == len(k_ranges) == len(attn_type_map), (
            "q_ranges, k_ranges and attn_type_map should have the same length"
            f", but got {len(q_ranges)=}, {len(k_ranges)=}, {len(attn_type_map)=}"
        )

        # extract flags
        test_accumulation_inplace = bool(
            flag_comb.get("test_accumulation_inplace", False)
        )
        deterministic = bool(flag_comb.get("deterministic", False))
        range_merge = bool(flag_comb.get("range_merge", False))
        random_attn_type_map = bool(flag_comb.get("random_attn_type_map", False))
        swap_bwd_qk_loop = bool(flag_comb.get("swap_bwd_qk_loop", False))
        enable_max_seqlen_q = bool(flag_comb.get("max_seqlen_q", False))
        # NOTE: we use ref_block_config_idx to extract ref_block_config since it is a non-hashable dict
        ref_block_config_idx = flag_comb.get("ref_block_config_idx", 0)
        ref_block_config = self.valid_ref_block_configs[ref_block_config_idx]
        swap_ab = ref_block_config["swap_ab"]
        ref_block_size = ref_block_config["ref_block_size"]
        pack_gqa = ref_block_config["pack_gqa"]
        block_sparse = ref_block_config["block_sparse"]
        return_max_logits = bool(flag_comb.get("return_max_logits", False))
        cat_gqa = bool(flag_comb.get("cat_gqa", False))

        # -----    skip invalid flag combinations   ---- #

        # TODO: Avoid skipping many flag combinations; instead, regenerate combinations with
        #       constraints to exclude invalid cases while covering more valid ones.
        if swap_bwd_qk_loop:
            # TODO: support deterministic mode with swap_bwd_qk_loop
            if deterministic:
                return

        if cat_gqa:
            # NOTE: pack_gqa and cat_gqa cannot be both True
            if pack_gqa:
                return

            # NOTE: swap_bwd_qk_loop is not implemented for CatGQA
            if swap_bwd_qk_loop:
                return

        if random_attn_type_map:
            # we now support attn type idx in {0, 1, 2, 3}
            attn_type_map = torch.randint(0, 4, (len(attn_type_map),)).tolist()

        # Calculate max_seqlen_q from q_ranges (maximum length of any q range)
        max_seqlen_q = (
            q_ranges.max_seqlen
            if enable_max_seqlen_q and not q_ranges.is_empty()
            else None
        )

        test_case = (
            f"[RANK {self.rank}][test_ffa_simple]"
            f"[{attn_mask_config['name']}]"
            f"[{model_config['name']}]"
            f"[dtype={dtype}]"
            f"[swap_ab={swap_ab}]"
            f"[ref_block_size={ref_block_size}]"
            f"[pack_gqa={pack_gqa}]"
            f"[block_sparse={block_sparse}]"
            f"[has_sink={seqlen_sink > 0}]"
            f"[sink_layout={sink_layout}] x "
            f"{flag_comb_test_case}"
        )
        print(f"[{datetime.now()}] START {test_case}", flush=True)

        self.run_test_case(
            seqlen_q=seqlen,
            seqlen_kv=seqlen,
            seqlen_sink=seqlen_sink,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            dtype=dtype,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            range_merge=range_merge,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            block_sparse=block_sparse,
            sink_layout=sink_layout,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            pack_gqa=pack_gqa,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
            max_seqlen_q=max_seqlen_q,
            test_case=test_case,
            return_max_logits=return_max_logits,
            cat_gqa=cat_gqa,
            err_ratio_dict={
                "dq_min_mismatch_thres": 5e-3,
                # FIXME: dsink ratios are fragile right now, need to be improved later
                "dsink_mismatch_thres_ratio": MISMATCH_THRES_RATIO * 1.5,
                "dsink_min_mismatch_thres": max(1 / (seqlen_sink * num_heads_q), 8e-2)
                if seqlen_sink > 0 and sink_layout == "sh"
                else 8e-2,
                "dsink_min_norm_rtol": 0.015,
                "dsink_norm_rtol_ratio": NORM_RTOL_RATIO * 2,
                "dsink_atol": 2e-4 if sink_layout == "sh" else EPSILON,
                "dsink_rtol": 0.1,
            },
        )
        print(f"[{datetime.now()}] FINISH {test_case}", flush=True)

    @with_run_in_mp
    @parameterize("model_config", MODEL_CONFIGS)
    @parameterize(
        "generate_config",
        [
            {
                "name": "dense_test_q_1024_k_1024",
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "min_len_q": 16,
                "max_len_q": 64,
                "min_len_k": 16,
                "max_len_k": 64,
            },
            {
                "name": "dense_test_q_2048_k_2048",
                "total_seqlen_q": 2048,
                "total_seqlen_k": 2048,
                "min_len_q": 32,
                "max_len_q": 128,
                "min_len_k": 32,
                "max_len_k": 128,
            },
            {
                "name": "sparse_test_q_2048_k_4096",
                "total_seqlen_q": 2048,
                "total_seqlen_k": 4096,
                "min_len_q": 16,
                "max_len_q": 256,
                "min_len_k": 64,
                "max_len_k": 512,
            },
            {
                "name": "sparse_test_q_4096_k_2048",
                "total_seqlen_q": 4096,
                "total_seqlen_k": 2048,
                "min_len_q": 64,
                "max_len_q": 512,
                "min_len_k": 16,
                "max_len_k": 256,
            },
            # FIXME: ut failed for the following 2 test cases, maybe due to very small qk ranges.
            # {
            #    "name": "strange_test_1",
            #    "total_seqlen_q": 513,
            #    "total_seqlen_k": 123,
            #    "min_len_q": 1,
            #    "max_len_q": 513,
            #    "min_len_k": 1,
            #    "max_len_k": 123,
            # },
            # {
            #    "name": "strange_test_2",
            #    "total_seqlen_q": 1023,
            #    "total_seqlen_k": 2077,
            #    "min_len_q": 1,
            #    "max_len_q": 1023,
            #    "min_len_k": 1,
            #    "max_len_k": 2077,
            # },
        ],
    )
    @parameterize(
        "num_pairs", [10, 100, 1000]
    )  # the max num of qk range pairs to generate
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize(
        "attn_type", [0, 1, 2, 3, 4]
    )  # 0 - 3 means attn type are all 0/1/2/3, 4 means random attn type.
    def test_ffa_random(
        self,
        model_config: dict[str, Any],
        generate_config: dict[str, Any],
        num_pairs: int,
        dtype: torch.dtype,
        attn_type: int,
    ):
        """in this test, we generate q,k range randomly and as complicate as possible"""
        # -----    switch env flags by FlagCombGenerator   ---- #
        flag_comb = next(self.flag_iterator)
        flag_comb_test_case = FlagCombGenerator.to_test_case(flag_comb)

        # extract config
        total_seqlen_q: int = generate_config["total_seqlen_q"]
        total_seqlen_k: int = generate_config["total_seqlen_k"]
        min_len_q: int = generate_config["min_len_q"]
        min_len_k: int = generate_config["min_len_k"]
        max_len_q: int = generate_config["max_len_q"]
        max_len_k: int = generate_config["min_len_k"]

        q_list, k_list = self.generate_non_overlapping_qk_pairs(
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            num_pairs=num_pairs,
            min_len_q=min_len_q,
            max_len_q=max_len_q,
            min_len_k=min_len_k,
            max_len_k=max_len_k,
            max_consecutive_failures=200,
        )
        q_ranges: AttnRanges = AttnRanges.from_ranges(q_list)
        k_ranges: AttnRanges = AttnRanges.from_ranges(k_list)
        attn_type_map = [attn_type] * q_ranges.size
        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        assert len(q_ranges) == len(k_ranges) == len(attn_type_map), (
            "q_ranges, k_ranges and attn_type_map should have the same length"
            f", but got {len(q_ranges)=}, {len(k_ranges)=}, {len(attn_type_map)=}"
        )

        # extract flags
        test_accumulation_inplace = bool(
            flag_comb.get("test_accumulation_inplace", False)
        )
        deterministic = bool(flag_comb.get("deterministic", False))
        range_merge = bool(flag_comb.get("range_merge", False))
        swap_bwd_qk_loop = bool(flag_comb.get("swap_bwd_qk_loop", False))
        enable_max_seqlen_q = bool(flag_comb.get("max_seqlen_q", False))
        # NOTE: we use ref_block_config_idx to extract ref_block_config since it is a non-hashable dict
        ref_block_config_idx = flag_comb.get("ref_block_config_idx", 0)
        ref_block_config = self.valid_ref_block_configs[ref_block_config_idx]
        swap_ab = ref_block_config["swap_ab"]
        ref_block_size = ref_block_config["ref_block_size"]
        pack_gqa = ref_block_config["pack_gqa"]
        block_sparse = ref_block_config["block_sparse"]
        return_max_logits = bool(flag_comb.get("return_max_logits", False))
        cat_gqa = bool(flag_comb.get("cat_gqa", False))

        # random attn type
        if attn_type == 4:
            # we now support attn type idx in {0, 1, 2, 3}
            attn_type_map = torch.randint(0, 4, (len(attn_type_map),)).tolist()

        # -----    skip invalid flag combinations   ---- #

        if swap_bwd_qk_loop:
            # TODO: support deterministic mode with swap_bwd_qk_loop
            if deterministic:
                return

        if block_sparse:
            # Sparse kernel requires NHQ==NHK or view-trick (pack_gqa with
            # single KV head per batch slice).  test_ffa_random passes raw
            # multi-KV-head data without view-trick, so skip GQA combos.
            if num_heads_q != num_heads_kv:
                return

        if cat_gqa:
            # NOTE: pack_gqa and cat_gqa cannot be both True
            if pack_gqa:
                return

            # NOTE: swap_bwd_qk_loop is not implemented for CatGQA
            if swap_bwd_qk_loop:
                return

        # Calculate max_seqlen_q from q_ranges (maximum length of any q range)
        max_seqlen_q = (
            q_ranges.max_seqlen
            if enable_max_seqlen_q and not q_ranges.is_empty()
            else None
        )

        test_case = (
            f"[RANK {self.rank}][test_ffa_random]"
            f"[{model_config['name']}]"
            f"[{generate_config['name']}]"
            f"[num_pairs={num_pairs}]"
            f"[dtype={dtype}]"
            f"[attn_type_map=[{attn_type}] x {q_ranges.size}]"
            f"[swap_ab={swap_ab}]"
            f"[ref_block_size={ref_block_size}]"
            f"[pack_gqa={pack_gqa}] x "
            f"[block_sparse={block_sparse}]"
            f"{flag_comb_test_case}"
        )
        print(f"[{datetime.now()}] START {test_case}", flush=True)

        self.run_test_case(
            seqlen_q=total_seqlen_q,
            seqlen_kv=total_seqlen_k,
            seqlen_sink=0,  # pass testing attn sink for now
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            dtype=dtype,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            range_merge=range_merge,
            deterministic=deterministic,
            test_accumulation_inplace=test_accumulation_inplace,
            block_sparse=block_sparse,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            pack_gqa=pack_gqa,
            cat_gqa=cat_gqa,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
            test_case=test_case,
            sink_layout="sh",
            max_seqlen_q=max_seqlen_q,
            return_max_logits=return_max_logits,
            err_ratio_dict={
                "dq_mismatch_thres_ratio": MISMATCH_THRES_RATIO * 1.5,
                "dq_min_mismatch_thres": 0.025,
                "dk_mismatch_thres_ratio": MISMATCH_THRES_RATIO * 1.5,
                "dk_min_mismatch_thres": 0.025,
                "dv_norm_rtol_ratio": NORM_RTOL_RATIO * 1.5,
            },
        )
        print(f"[{datetime.now()}] FINISH {test_case}", flush=True)

    @parameterize("sink_layout", ["sh", "ssh"])  # ["sh", "ssh", "shd"])
    def test_ffa_compiled(self, sink_layout: AttnSinkLayout):
        s, s_sink = 2048, 4
        hq, hk, d = 6, 3, 128

        q = torch.randn(s, hq, d, dtype=torch.bfloat16, device=self.device)
        k = torch.randn(s, hk, d, dtype=torch.bfloat16, device=self.device)
        v = torch.randn_like(k)
        do = torch.randn_like(q)
        match sink_layout:
            case "sh":
                sink = torch.randn(s_sink, hq, dtype=torch.float32, device=self.device)
            case "ssh":
                sink = torch.randn(
                    s, s_sink, hq, dtype=torch.float32, device=self.device
                )
            case "shd":
                raise NotImplementedError(
                    f"sink_layout {sink_layout} is not supported yet"
                )
            case _:
                raise ValueError(f"Invalid sink_layout {sink_layout}")

        [x.requires_grad_(True) for x in (q, k, v, sink)]

        q_ranges = AttnRanges.from_ranges([[0, s // 2], [s // 2, s]])
        k_ranges = AttnRanges.from_ranges([[0, s // 2], [s // 2, s]])
        attn_type_map = [0, 1]

        compiled_ffa_func = torch.compile(fullgraph=True)(flex_flash_attn_func)

        o, meta = compiled_ffa_func(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges.to_tensor(self.device),
            k_ranges=k_ranges.to_tensor(self.device),
            attn_type_map=torch.tensor(
                attn_type_map, dtype=torch.int32, device=self.device
            ),
            sink=sink,
            sink_layout=sink_layout,
            # FIXME: compiling does not support range_merge
            # due to custom unique_consecutive_pairs kernel with dynamic output shape
            range_merge=False,
            block_sparse=False,
        )
        lse = meta.lse
        o.backward(do)
        dq, dk, dv, dsink = q.grad, k.grad, v.grad, sink.grad

        self.assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=s,
            total_seqlen_k=s,
            total_q=q,
            total_k=k,
            total_v=v,
            total_sink=sink,
            total_out=o,
            total_lse=lse,
            grad_total_q=dq,
            grad_total_k=dk,
            grad_total_v=dv,
            grad_total_sink=dsink,
            grad_total_out=do,
            dtype=torch.bfloat16,
            sink_layout=sink_layout,
            test_case=("[test_ffa_compiled]" f"[sink_layout={sink_layout}]"),
            max_seqlen_q=None,
        )


class TestFlexFlashAttnSimple(unittest.TestCase):
    """Lightweight single-process regression tests for flex_flash_attn.

    Extracted from test_simple_attn.py — covers env-var-toggled code paths,
    mask build verification, and IndexSparse with non-standard inner direction.
    """

    @property
    def device(self):
        return torch.cuda.current_device()

    def assert_close_to_torch_ref(self, **kwargs):
        TestFlexFlashAttn.assert_close_to_torch_ref(self, **kwargs)

    def _check_block_sparse_vs_dense_ref(
        self,
        *,
        S: int,
        n_attend: int,
        k_block: int,
        swap_bwd_qk_loop_cases: tuple[bool, ...],
        test_case: str,
        tol: float = 2e-2,
    ):
        """Comparison helper: block_sparse variants vs the dense-TMA ffa reference."""
        from magi_attention.utils.sparse_utils import (
            generate_ranges_from_block_mask_triton,
        )

        device = self.device
        nhq, nhk, head_dim = 128, 1, 128
        dtype = torch.bfloat16
        torch.manual_seed(42)

        n_q_blocks, n_k_blocks = S, S // k_block
        sel = torch.rand(n_q_blocks, n_k_blocks, device=device).argsort(dim=1)[
            :, : min(n_attend, n_k_blocks)
        ]
        block_mask = torch.zeros(
            1, nhk, n_q_blocks, n_k_blocks, dtype=torch.bool, device=device
        )
        block_mask[0, 0].scatter_(1, sel, True)
        q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
            block_mask, 1, k_block
        )
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=device)

        q0 = torch.randn(S, nhq, head_dim, device=device, dtype=dtype)
        k0 = torch.randn(S, nhk, head_dim, device=device, dtype=dtype)
        v0 = torch.randn(S, nhk, head_dim, device=device, dtype=dtype)
        do = torch.randn(S, nhq, head_dim, device=device, dtype=dtype)

        def run(block_sparse: bool, swap_bwd_qk_loop: bool):
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
                swap_bwd_qk_loop=swap_bwd_qk_loop,
            )
            out.backward(do)
            return out.detach(), q.grad, k.grad, v.grad

        ref = run(block_sparse=False, swap_bwd_qk_loop=True)
        for swap in swap_bwd_qk_loop_cases:
            got = run(block_sparse=True, swap_bwd_qk_loop=swap)
            loop_name = "loopk" if swap else "loopq"
            for name, a, b in zip(("out", "dq", "dk", "dv"), got, ref):
                err = (
                    (a.float() - b.float()).abs().max()
                    / b.float().abs().max().clamp_min(1e-6)
                ).item()
                assert (
                    err < tol
                ), f"{test_case}[{loop_name}] {name} max_rel_err={err:.3e} >= {tol}"

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

    # ─── Tier-1: env-var-toggled code paths ───

    @pytest.mark.slow
    def test_consumer_dkv_store(self):
        """Tier-1: consumer-side scatter dX store
        (MAGI_ATTENTION_FFA_INNER_STORE_IN_PRODUCER=false) for both
        LoopK (dKV from consumer WGs) and LoopQ (dQ from consumer WGs)."""
        import os

        from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_mod

        test_case = "[consumer_dkv_store]"
        print(f"\n>>> {test_case} START", flush=True)
        t0 = time.time()
        os.environ["MAGI_ATTENTION_FFA_INNER_STORE_IN_PRODUCER"] = "false"
        if hasattr(get_ffa_jit_mod, "cache_clear"):
            get_ffa_jit_mod.cache_clear()
        try:
            self._check_block_sparse_vs_dense_ref(
                S=2048,
                n_attend=8,
                k_block=128,
                swap_bwd_qk_loop_cases=(True, False),
                test_case=test_case,
            )
        finally:
            del os.environ["MAGI_ATTENTION_FFA_INNER_STORE_IN_PRODUCER"]
            if hasattr(get_ffa_jit_mod, "cache_clear"):
                get_ffa_jit_mod.cache_clear()
        print(f">>> {test_case} PASSED  ({time.time() - t0:.1f}s)", flush=True)

    @pytest.mark.slow
    def test_scalar_dx_store(self):
        """Tier-1: scalar atomicAdd dX store fallback
        (MAGI_ATTENTION_FFA_INNER_STORE_MODE=atomicadd, i.e.
        kInnerStoreMode==InnerStoreMode::AtomicAdd) for both LoopK and LoopQ."""
        import os

        from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_mod

        test_case = "[scalar_dx_store]"
        print(f"\n>>> {test_case} START", flush=True)
        t0 = time.time()
        os.environ["MAGI_ATTENTION_FFA_INNER_STORE_MODE"] = "atomicadd"
        if hasattr(get_ffa_jit_mod, "cache_clear"):
            get_ffa_jit_mod.cache_clear()
        try:
            self._check_block_sparse_vs_dense_ref(
                S=2048,
                n_attend=8,
                k_block=128,
                swap_bwd_qk_loop_cases=(True, False),
                test_case=test_case,
            )
        finally:
            del os.environ["MAGI_ATTENTION_FFA_INNER_STORE_MODE"]
            if hasattr(get_ffa_jit_mod, "cache_clear"):
                get_ffa_jit_mod.cache_clear()
        print(f">>> {test_case} PASSED  ({time.time() - t0:.1f}s)", flush=True)

    # ─── IntraWGOverlap=false unit test ───

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_inner_dir_min_to_max(self):
        """Verify Dense + IndexSparse FWD+BWD with InnerDir=MinToMax.

        Dense: causal 256 seqlen, verifies reversed traversal order.
        IndexSparse: topk=100 with max_topk=128 (28 padding tokens in the last
        block), verifies padding_mask is applied to the correct block when
        the sparse iteration direction is flipped.
        """
        import os

        from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_mod
        from magi_attention.utils.sparse_utils import (
            build_index_sparse_indices,
            get_sdpa_mask_from_index_sparse_indices,
        )

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

            # ── Part 2: IndexSparse with non-aligned topk (padding block) ──
            B, S, NHQ_ia, NHK_ia, D = 1, 256, 32, 4, 128
            actual_topk = 100
            max_topk = 128
            gqa_ia = NHQ_ia // NHK_ia
            S_flat = S * NHK_ia
            NHQ_eff = gqa_ia

            indices = build_index_sparse_indices(
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
                index_sparse_indices=indices,
                q_block_size=1,
                sparse_k_block_size=1,
                pack_gqa=True,
            )

            ref_mask = get_sdpa_mask_from_index_sparse_indices(
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
                    f"[test_inner_dir_min_to_max/index_sparse] "
                    f"FWD batch {b_i}: max_diff={max_diff:.6f} >= 0.01"
                )

            # BWD check
            do_sparse = torch.randn_like(o_sparse)
            o_sparse.backward(do_sparse)
            assert (
                q_ffa.grad is not None
            ), "[test_inner_dir_min_to_max/index_sparse] BWD: q_ffa.grad is None"

        finally:
            del os.environ["MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"]
            if hasattr(get_ffa_jit_mod, "cache_clear"):
                get_ffa_jit_mod.cache_clear()

    # ─── UseMaskDispatch=false unit test ───

    @pytest.mark.slow
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
    run_tests()
