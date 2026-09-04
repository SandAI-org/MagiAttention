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

"""Host-side rejection matrix for the CuTeDSL q/k-range validators.

``validate_range_feature_support`` only inspects flags and
``isinstance(mask_types, Tensor)``, so it runs on CPU; ``validate_true_ranges``
requires CUDA range tensors.

Run:
    pytest tests/test_kernel/cutedsl/test_ffa_range_validation.py -v
"""

import pytest
import torch

from magi_attention.kernel.cutedsl.ffa_utils import (
    MT_MAP,
    validate_range_feature_support,
    validate_true_ranges,
)

PER_RANGE = torch.zeros(2, dtype=torch.int32)
RUNTIME_MASK_FEATURES = (
    "has_mask_mod",
    "has_block_sparse",
    "has_score_mod",
    "has_softcap",
)


def _validate(**overrides):
    kwargs = dict(
        major_arch=10,
        has_ranges=True,
        mask_types=MT_MAP.full,
        range_merge=False,
        range_merge_unique_writer=True,
        has_mask_mod=False,
        has_block_sparse=False,
        has_score_mod=False,
        has_softcap=False,
    )
    kwargs.update(overrides)
    validate_range_feature_support(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"has_ranges": False, "major_arch": 9},
        {"has_ranges": False, "deterministic": True},
        {"range_merge": True, "major_arch": 11},
        {"mask_types": PER_RANGE, "major_arch": 11},
        {"range_merge": True, "bwd_head_dim": 128},
        {"bwd_head_dim": 192},
        *({f: True} for f in RUNTIME_MASK_FEATURES),
        *({f: True, "major_arch": 9} for f in RUNTIME_MASK_FEATURES),
    ],
)
def test_accepted(overrides):
    _validate(**overrides)


@pytest.mark.parametrize(
    "overrides, exc",
    [
        ({"has_ranges": False, "mask_types": PER_RANGE}, NotImplementedError),
        ({"deterministic": True}, NotImplementedError),
        ({"range_merge": True, "range_merge_unique_writer": False}, ValueError),
        ({"range_merge": True, "major_arch": 9}, NotImplementedError),
        ({"mask_types": PER_RANGE, "major_arch": 9}, NotImplementedError),
        ({"range_merge": True, "bwd_head_dim": 192}, NotImplementedError),
        *(
            ({"range_merge": True, f: True}, NotImplementedError)
            for f in RUNTIME_MASK_FEATURES
        ),
        *(
            ({"mask_types": PER_RANGE, f: True}, NotImplementedError)
            for f in RUNTIME_MASK_FEATURES
        ),
    ],
)
def test_rejected(overrides, exc):
    with pytest.raises(exc):
        _validate(**overrides)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA range tensors")
def test_validate_true_ranges():
    ranges = torch.tensor([[0, 4], [4, 8]], dtype=torch.int32, device="cuda")
    assert validate_true_ranges(None, None) is False
    assert validate_true_ranges(ranges, ranges) is True
    with pytest.raises(ValueError):
        validate_true_ranges(ranges, None)
    with pytest.raises(ValueError):
        validate_true_ranges(None, ranges)
    with pytest.raises(ValueError):
        validate_true_ranges(
            ranges,
            ranges,
            mask_types=torch.zeros(3, dtype=torch.int32, device="cuda"),
        )
