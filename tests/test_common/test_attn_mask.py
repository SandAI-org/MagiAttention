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

from functools import partial

import numpy as np
import pytest
import torch


@pytest.fixture(params=["python"])
def backend(request):
    """AttnMask is Python-only, so only test with Python backend."""
    yield request.param


class TestAttnMask:
    def test_mask_factory_constructors(self, backend):
        from magi_attention.common import AttnMask, AttnMaskType, AttnRanges

        # --------     factory constructor1: from_ranges  --------- #

        q_ranges = AttnRanges.from_ranges(
            [
                (0, 3),
                (3, 5),
                (5, 8),
                (8, 9),
                (9, 14),
                (14, 16),
            ]
        )
        total_seqlen_q = q_ranges.end

        k_ranges = AttnRanges.from_ranges(
            [
                (0, 4),
                (2, 4),
                (3, 7),
                (4, 12),
                (6, 9),
                (1, 13),
            ]
        )
        total_seqlen_k = q_ranges.end  # for self-attn: use the end of sq

        attn_mask_type = [
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
        ]

        attn_mask = AttnMask.from_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
        )

        ref_mask_flag_list = [
            # col 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 0
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 1
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 2
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 3
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 4
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 5
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 6
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 7
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # row 8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 10
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 11
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # row 12
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # row 13
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # row 14
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # row 15
        ]
        ref_mask_flag_tensor = torch.tensor(
            ref_mask_flag_list,
            dtype=torch.int32,
            device=AttnMask.device,
        )

        assert np.equal(
            attn_mask.mask_flag_array,
            np.array(ref_mask_flag_list),
        ).all()

        assert torch.equal(
            attn_mask.mask_tensor[..., AttnMask.mask_flag_dim_idx],
            ref_mask_flag_tensor,
        )

        # --------     factory constructor1: from_mask  --------- #

        attn_mask2 = AttnMask.from_mask(
            mask=ref_mask_flag_tensor,
        )

        assert attn_mask2.q_ranges == q_ranges
        assert attn_mask2.k_ranges == k_ranges
        assert attn_mask2.attn_mask_type == attn_mask_type

        # --------     __init__ is forbidden to users  --------- #

        with pytest.raises(RuntimeError):
            AttnMask(
                mask_tensor=ref_mask_flag_tensor.unsqueeze(0),
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
            )

    def test_make_sub_mask_with_calc_sub_area(self, backend):
        from magi_attention.common import AttnMask, AttnMaskType, AttnRange, AttnRanges

        # --------------      init sample meta      -------------- #

        q_ranges = AttnRanges.from_ranges(
            [
                (0, 6),
                (6, 9),
                (9, 12),
                (12, 16),
            ]
        )
        total_seqlen_q = q_ranges.end

        k_ranges = AttnRanges.from_ranges(
            [
                (0, 4),
                (4, 12),
                (12, 15),
                (1, 13),
            ]
        )
        total_seqlen_k = q_ranges.end  # for self-attn: use the end of sq

        attn_mask_type = [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
        ]

        # --------------      init attn mask       -------------- #

        attn_mask = AttnMask.from_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
        )
        assert attn_mask.area == 82

        # --------------      sub mask 1       -------------- #

        sub_q_range = AttnRange.from_range((4, 13))
        sub_k_range = AttnRange.from_range((1, 13))

        sub_area1 = attn_mask.calc_sub_area(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_area1 == 41

        sub_attn_mask1 = attn_mask.make_sub_mask(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_attn_mask1.area == sub_area1

        assert sub_attn_mask1.q_ranges == AttnRanges.from_ranges(
            [[0, 2], [2, 5], [5, 8], [8, 9]]
        )
        assert sub_attn_mask1.k_ranges == AttnRanges.from_ranges(
            [[0, 3], [3, 11], [11, 12], [0, 9]]
        )
        assert sub_attn_mask1.attn_mask_type == [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.FULL,
            AttnMaskType.FULL,
        ]

        # --------------      sub mask 2       -------------- #

        sub_q_range = AttnRange.from_range((0, 14))
        sub_k_range = AttnRange.from_range((0, 7))

        sub_area2 = attn_mask.calc_sub_area(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_area2 == 31

        sub_attn_mask2 = attn_mask.make_sub_mask(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_attn_mask2.area == sub_area2

        assert sub_attn_mask2.q_ranges == AttnRanges.from_ranges(
            [[0, 6], [6, 9], [9, 12], [12, 14]]
        )
        assert sub_attn_mask2.k_ranges == AttnRanges.from_ranges(
            [[0, 4], [4, 7], [9, 9], [1, 7]]
        )
        assert sub_attn_mask2.attn_mask_type == [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
        ]

        # --------------      sub mask 3       -------------- #

        sub_q_range = AttnRange.from_range((5, 16))
        sub_k_range = AttnRange.from_range((3, 11))

        sub_area3 = attn_mask.calc_sub_area(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_area3 == 53

        sub_attn_mask3 = attn_mask.make_sub_mask(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_attn_mask3.area == sub_area3

        assert sub_attn_mask3.q_ranges == AttnRanges.from_ranges(
            [[0, 1], [1, 4], [4, 7], [7, 9], [9, 11]]
        )
        assert sub_attn_mask3.k_ranges == AttnRanges.from_ranges(
            [[0, 1], [1, 8], [4, 4], [0, 8], [0, 8]]
        )
        assert sub_attn_mask3.attn_mask_type == [
            AttnMaskType.FULL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
        ]

        # --------------      sub mask 4       -------------- #

        sub_q_range = AttnRange.from_range((4, 12))
        sub_k_range = AttnRange.from_range((4, 12))

        sub_area4 = attn_mask.calc_sub_area(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_area4 == 24

        sub_attn_mask4 = attn_mask.make_sub_mask(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_attn_mask4.area == sub_area4

        assert sub_attn_mask4.q_ranges == AttnRanges.from_ranges(
            [[0, 2], [2, 5], [5, 8]]
        )
        assert sub_attn_mask4.k_ranges == AttnRanges.from_ranges(
            [[0, 0], [0, 8], [5, 5]]
        )
        assert sub_attn_mask4.attn_mask_type == [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
        ]

    def test_mask_type_bool_func(self, backend):
        from magi_attention.common import AttnMask, AttnMaskType, AttnRanges

        q_ranges = AttnRanges.from_ranges(
            [
                [0, 2],
                [2, 8],
                [8, 24],
                [24, 48],
                [48, 64],
                [64, 100],
            ]
        )
        total_seqlen_q = q_ranges.end
        total_seqlen_k = q_ranges.end  # for self-attn: use the end of sq

        make_attn_mask = partial(
            AttnMask.from_ranges,
            q_ranges=q_ranges,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
        )

        # --------    pure full mask  --------- #

        k_ranges = AttnRanges.from_ranges(
            [
                [0, 100],
                [0, 100],
                [0, 100],
                [0, 100],
                [0, 100],
                [0, 100],
            ]
        )

        attn_mask_type = [AttnMaskType.FULL] * len(q_ranges)

        pure_full_mask = make_attn_mask(
            k_ranges=k_ranges, attn_mask_type=attn_mask_type
        )

        assert pure_full_mask.is_square()
        assert pure_full_mask.is_pure_full()

        # --------    pure causal mask  --------- #

        k_ranges = AttnRanges.from_ranges(
            [
                [0, 2],
                [0, 8],
                [0, 24],
                [0, 48],
                [0, 64],
                [0, 100],
            ]
        )

        attn_mask_type = [AttnMaskType.CAUSAL] * len(q_ranges)

        pure_full_mask = make_attn_mask(
            k_ranges=k_ranges, attn_mask_type=attn_mask_type
        )

        assert pure_full_mask.is_pure_causal()

        # --------    varlen full mask  --------- #

        k_ranges1 = AttnRanges.from_ranges(
            [
                [0, 3],
                [3, 9],
                [9, 23],
                [23, 49],
                [49, 63],
                [63, 100],
            ]
        )
        k_ranges2 = AttnRanges.from_ranges(
            [
                [0, 3],
                [2, 9],  # overlapped with the previous sample
                [9, 23],
                [23, 49],
                [49, 63],
                [63, 100],
            ]
        )
        k_ranges3 = AttnRanges.from_ranges(
            [
                [0, 3],
                [5, 9],  # disjoint with the previous sample
                [9, 23],
                [23, 49],
                [49, 63],
                [63, 100],
            ]
        )
        k_ranges4 = AttnRanges.from_ranges(
            [
                [0, 3],
                [3, 9],
                [9, 23],
                [23, 49],
                [49, 63],
                [63, 99],  # not reach the end of the mask
            ]
        )

        attn_mask_type = [AttnMaskType.FULL] * len(q_ranges)

        varlen_full_mask1 = make_attn_mask(
            k_ranges=k_ranges1, attn_mask_type=attn_mask_type
        )
        varlen_full_mask2 = make_attn_mask(
            k_ranges=k_ranges2, attn_mask_type=attn_mask_type
        )
        varlen_full_mask3 = make_attn_mask(
            k_ranges=k_ranges3, attn_mask_type=attn_mask_type
        )
        varlen_full_mask4 = make_attn_mask(
            k_ranges=k_ranges4, attn_mask_type=attn_mask_type
        )

        assert varlen_full_mask1.is_varlen_full()
        assert not varlen_full_mask2.is_varlen_full()
        assert not varlen_full_mask3.is_varlen_full()
        assert not varlen_full_mask4.is_varlen_full()

        # --------    varlen causal mask  --------- #

        assert not varlen_full_mask1.is_varlen_causal()
        assert not varlen_full_mask2.is_varlen_causal()
        assert not varlen_full_mask3.is_varlen_causal()
        assert not varlen_full_mask4.is_varlen_causal()

        attn_mask_type = [AttnMaskType.CAUSAL] * len(q_ranges)

        varlen_causal_mask1 = make_attn_mask(
            k_ranges=k_ranges1, attn_mask_type=attn_mask_type
        )
        varlen_causal_mask2 = make_attn_mask(
            k_ranges=k_ranges2, attn_mask_type=attn_mask_type
        )
        varlen_causal_mask3 = make_attn_mask(
            k_ranges=k_ranges3, attn_mask_type=attn_mask_type
        )
        varlen_causal_mask4 = make_attn_mask(
            k_ranges=k_ranges4, attn_mask_type=attn_mask_type
        )

        assert varlen_causal_mask1.is_varlen_causal()
        assert not varlen_causal_mask2.is_varlen_causal()
        assert not varlen_causal_mask3.is_varlen_causal()
        assert not varlen_causal_mask4.is_varlen_causal()

    def test_make_causal_mask(self, backend):
        from magi_attention.common import AttnMask

        # ---------    square causal mask     --------- #

        ref_mask = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.int32,
            device="cpu",
        )

        mask_tl = AttnMask.make_causal_mask(
            seqlen_q=4,
            seqlen_k=4,
            align="top-left",
            dtype=torch.int32,
            device="cpu",
        )
        mask_br = AttnMask.make_causal_mask(
            seqlen_q=4,
            seqlen_k=4,
            align="bottom-right",
            dtype=torch.int32,
            device="cpu",
        )
        assert torch.equal(mask_tl, ref_mask)
        assert torch.equal(mask_br, ref_mask)

        # ---------    "tall" causal mask when sq > sk     --------- #

        ref_mask_tl = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.int32,
            device="cpu",
        )

        mask_tl = AttnMask.make_causal_mask(
            seqlen_q=7,
            seqlen_k=4,
            align="top-left",
            dtype=torch.int32,
            device="cpu",
        )

        assert torch.equal(mask_tl, ref_mask_tl)

        ref_mask_br = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.int32,
            device="cpu",
        )

        mask_br = AttnMask.make_causal_mask(
            seqlen_q=7,
            seqlen_k=4,
            align="bottom-right",
            dtype=torch.int32,
            device="cpu",
        )

        assert torch.equal(mask_br, ref_mask_br)

        # ---------    "fat" causal mask when sq < sk     --------- #

        ref_mask_tl = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.int32,
            device="cpu",
        )

        mask_tl = AttnMask.make_causal_mask(
            seqlen_q=4,
            seqlen_k=7,
            align="top-left",
            dtype=torch.int32,
            device="cpu",
        )

        assert torch.equal(mask_tl, ref_mask_tl)

        ref_mask_br = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0],  # num(1) = sk - sq + 1
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.int32,
            device="cpu",
        )

        mask_br = AttnMask.make_causal_mask(
            seqlen_q=4,
            seqlen_k=7,
            align="bottom-right",
            dtype=torch.int32,
            device="cpu",
        )

        assert torch.equal(mask_br, ref_mask_br)

    def test_from_ranges_invalid_mask_type(self, backend):
        from magi_attention.common import AttnMask, AttnRanges

        q_ranges = AttnRanges.from_ranges([(0, 5)])
        k_ranges = AttnRanges.from_ranges([(0, 5)])
        with pytest.raises(ValueError, match="Invalid mask type"):
            with AttnMask.can_instantiate_ctx():
                AttnMask.from_ranges(
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_mask_type=["invalid_type"],
                )

    def test_is_empty(self, backend):
        from magi_attention.common import AttnMask, AttnMaskType, AttnRanges

        q_ranges = AttnRanges.from_ranges([(0, 5)])
        k_ranges = AttnRanges.from_ranges([(0, 5)])
        with AttnMask.can_instantiate_ctx():
            mask = AttnMask.from_ranges(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=[AttnMaskType.FULL],
            )
        assert not mask.is_empty()

    def test_eq_with_non_mask(self, backend):
        from magi_attention.common import AttnMask, AttnMaskType, AttnRanges

        q_ranges = AttnRanges.from_ranges([(0, 3)])
        k_ranges = AttnRanges.from_ranges([(0, 3)])
        with AttnMask.can_instantiate_ctx():
            mask = AttnMask.from_ranges(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=[AttnMaskType.FULL],
            )
        assert mask != "not a mask"
        assert mask != 42

    def test_make_causal_mask_invalid_align(self, backend):
        from magi_attention.common import AttnMask

        with pytest.raises(ValueError, match="Invalid alignment"):
            AttnMask.make_causal_mask(5, 5, align="center")
