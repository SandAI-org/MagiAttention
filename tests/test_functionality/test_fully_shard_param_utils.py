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

from typing import Literal, cast

import torch
from torch import nn
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs

from magi_fsdp import (
    MagiFSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
    magi_fsdp_switch_params,
    magi_fsdp_use_params,
)
from magi_fsdp.testing import (
    assert_close,
    parameterize,
)
from magi_fsdp.testing.common_fsdp import (
    FSDPTest,
    MultiDtypeTransformer,
)


class TestFullyShardSwitchWeight(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    @staticmethod
    def get_model_with_mixed_precision() -> MagiFSDPModule:
        # Initialize model arguments
        model_args = ModelArgs(
            n_layers=5,
            n_heads=4,
            dim=128,
            vocab_size=1024,
            max_seq_len=1024,
            dropout_p=0,
            use_attn_mask=False,
            weight_tying=False,
            checkpoint_activations=False,
        )

        # Initialize model with mixed precision configuration
        model = MultiDtypeTransformer(model_args, {})
        fully_shard(
            model.layers[0],
            mp_policy=MixedPrecisionPolicy(main_param_dtype=torch.float64),
        )
        fully_shard(
            model.layers[1],
            mp_policy=MixedPrecisionPolicy(ema_param_dtype=torch.float64),
        )
        fully_shard(
            model.layers[2],
            mp_policy=MixedPrecisionPolicy(
                main_param_dtype=torch.float64, ema_param_dtype=torch.float64
            ),
        )
        fully_shard(
            model.layers[3],
            mp_policy=MixedPrecisionPolicy(ema_param_dtype=torch.float64),
        )
        fully_shard(
            model.layers[4],
            mp_policy=MixedPrecisionPolicy(main_param_dtype=torch.float64),
        )
        fully_shard(model)

        return model

    @parameterize("param_type", ["main", "ema"])
    def test_magi_fsdp_switch_params(self, param_type: Literal["main", "ema"]) -> None:
        model = self.get_model_with_mixed_precision()

        with magi_fsdp_switch_params(model, param_type=param_type):
            param_ids = list(map(id, cast(nn.Module, model).parameters()))

        for magi_fsdp_param in model.magi_fsdp_parameters():
            ref_param = (
                magi_fsdp_param.sharded_main_param
                if param_type == "main"
                else magi_fsdp_param.sharded_ema_param
            )
            if ref_param is not None:
                self.assertEqual(ref_param.dtype, torch.float64)
                self.assertIn(id(ref_param), param_ids)

    @parameterize("param_type", ["main", "ema"])
    def test_magi_fsdp_use_params(self, param_type: Literal["main", "ema"]) -> None:
        model = self.get_model_with_mixed_precision()
        magi_fsdp_use_params(model, param_type=param_type)

        for magi_fsdp_param in model.magi_fsdp_parameters():
            ref_param = (
                magi_fsdp_param.sharded_main_param
                if param_type == "main"
                else magi_fsdp_param.sharded_ema_param
            )
            if ref_param is not None:
                assert_close(magi_fsdp_param.sharded_param.to(torch.float64), ref_param)


if __name__ == "__main__":
    run_tests()
