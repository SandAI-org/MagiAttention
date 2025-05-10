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

import numpy as np

from magi_attention.common import AttnRange
from magi_attention.common.enum import AttnMaskType


def add_range_to_array(
    array: np.ndarray,
    q_range: AttnRange,
    k_range: AttnRange,
    masktype: AttnMaskType = AttnMaskType.FULL,
    check: bool = False,
):
    # get start and end of range
    x_start, x_end = q_range.start, q_range.end
    y_start, y_end = k_range.start, k_range.end

    if check:
        # check whether the current slice has been filled
        assert np.all(array[x_start:x_end, y_start:y_end] == 0), (
            f"Part of the area has been added," f"when {q_range=} and {k_range=}"
        )

    # fill the area according to the type of the mask.
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            if masktype == AttnMaskType.FULL:
                array[i][j] = 1
            elif masktype == AttnMaskType.CAUSAL:
                b = y_end - x_end
                fx = i + b
                if j <= fx:
                    array[i][j] = 1
                else:
                    array[i][j] = 0
            # HACK do not support INV_CAUSAL and BI_CAUSAL now
            # elif masktype == AttnMaskType.INV_CAUSAL:
            #     b = y_start - x_start
            #     fx = i + b
            #     if j >= fx:
            #         array[i][j] = 1
            #     else:
            #         array[i][j] = 0
            # elif masktype == AttnMaskType.BI_CAUSAL:
            #     causal_b = y_end - x_end
            #     f_causal = i + causal_b

            #     inv_causal_b = y_start - x_start
            #     f_inv_causal = i + inv_causal_b
            #     if j <= f_causal and j >= f_inv_causal:
            #         array[i][j] = 1
            #     else:
            #         array[i][j] = 0

    return array
