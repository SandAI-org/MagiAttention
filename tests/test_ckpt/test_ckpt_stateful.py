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
    EMAParamStatefulWrapper,
    MagiFSDPModule,
    MainParamStatefulWrapper,
    MixedPrecisionPolicy,
    fully_shard,
    magi_fsdp_switch_params,
)
from magi_fsdp.testing import parameterize
from magi_fsdp.testing.common_fsdp import (
    FSDPTest,
    MultiDtypeTransformer,
)


class TestFullyStateful(FSDPTest):
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
    def test_magi_fsdp_stateful_simple(
        self, param_type: Literal["main", "ema"]
    ) -> None:
        model1 = self.get_model_with_mixed_precision()
        model2 = self.get_model_with_mixed_precision()

        if param_type == "main":
            model1 = MainParamStatefulWrapper(model1)
            model2 = MainParamStatefulWrapper(model2)
        else:
            model1 = EMAParamStatefulWrapper(model1)
            model2 = EMAParamStatefulWrapper(model2)

        assert isinstance(model1, (MainParamStatefulWrapper, EMAParamStatefulWrapper))
        assert isinstance(model2, (MainParamStatefulWrapper, EMAParamStatefulWrapper))

        model2.load_state_dict(model1.state_dict())

        with (
            magi_fsdp_switch_params(model1.model, param_type=param_type),
            magi_fsdp_switch_params(model2.model, param_type=param_type),
        ):
            for p1, p2 in zip(
                cast(nn.Module, model1.model).parameters(),
                cast(nn.Module, model2.model).parameters(),
            ):
                self.assertTrue(torch.equal(p1, p2))

    # TODO(littsk): Add more tests for complex scenarios


if __name__ == "__main__":
    run_tests()
