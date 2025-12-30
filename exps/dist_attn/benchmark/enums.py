# Copyright (c) 2025 SandAI. All Rights Reserved.
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

from enum import Enum


class FlashMaskType(Enum):
    FULL = "full"
    CAUSAL = "causal"
    CAUSAL_DOCUMENT = "causal_document"
    FULL_DOCUMENT = "full_document"
    SHARE_QUESTION = "share_question"
    CAUSAL_BLOCKWISE = "causal_blockwise"
    PREFIX_LM_DOCUMENT = "prefix_lm_document"
    PREFIX_LM_CAUSAL = "prefix_lm_causal"
    QK_SPARSE = "qk_sparse"
    HASH_SPARSE = "hash_sparse"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_CAUSAL = "sliding_window_causal"
    GLOBAL_SLIDING_WINDOW = "global_sliding_window"
    BLOCK_CAUSAL_DOCUMENT = "block_causal_document"


class MetricsType(Enum):
    RANGES_INFORMATION_ENTROPY = "ranges_information_entropy"
    AREA_INFORMATION_ENTROPY = "areas_information_entropy"
    REMOTE_NORMALIZED_VALUE = "remote_normalized_value"
    MAX_AREA_DIVIDED_BY_TOTAL_AREA = "max_area_divided_by_total_area"
    MAX_AREA_DIVIDED_BY_AVERAGE_AREA = "max_area_divided_by_average_area"
    AREA_GINI_IMPURITY = "area_gini_impurity"
    RANGES_GIMI_IMPURITY = "ranges_gimi_impurity"
    COST_MODEL = "cost_model"
    COMPUTATION_AMOUNT = "computation_amount"
    COMM_BYTES = "comm_bytes"
