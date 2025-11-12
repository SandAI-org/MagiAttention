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

import functools

import torch
from torch.distributed.tensor import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs

from magi_fsdp import OptimPolicy, fully_shard
from magi_fsdp._fsdp_module import switch_named_main_params_on_modules
from magi_fsdp.testing import parameterize
from magi_fsdp.testing.common_fsdp import FSDPTest, MultiDtypeTransformer


class TestInitOptimMainParams(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    @parameterize(
        "multi_dtype",
        [
            {
                "TransformerBlock": {
                    0: torch.float32,
                    1: torch.bfloat16,
                    2: torch.float32,
                    3: torch.bfloat16,
                    4: torch.float32,
                },
                "LayerNorm": torch.float32,
            },
        ],
    )
    def test_init_optimizer_main_params(
        self,
        multi_dtype: dict[str, dict[int, torch.dtype]],
    ):
        torch.manual_seed(42)
        vocab_size = 1024
        model_args = ModelArgs(
            n_layers=5,
            n_heads=4,
            dim=128,
            vocab_size=vocab_size,
            max_seq_len=1024,
            dropout_p=0,
            use_attn_mask=False,
            weight_tying=False,
            checkpoint_activations=False,
        )
        model = MultiDtypeTransformer(model_args, multi_dtype)
        mesh = init_device_mesh("cuda", (self.world_size,))
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            optim_policy=OptimPolicy(enable_ema_param=False, enable_main_param=True),
        )
        fully_shard_fn([model.layers[0], model.layers[1]])
        fully_shard_fn([model.layers[2], model.layers[3], model.layers[4]])
        fully_shard_fn(model)
        model.create_main_params()
        main_params = list(model.main_parameters())
        with switch_named_main_params_on_modules(model, enable_main_param=True):
            orig_params = list(model.parameters())
            self.assertEqual(main_params, orig_params)
