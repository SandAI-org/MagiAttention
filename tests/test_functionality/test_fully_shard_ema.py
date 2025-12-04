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

import copy

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests

from magi_fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    magi_fsdp_switch_params,
    magi_fsdp_use_params,
)
from magi_fsdp.testing import parameterize
from magi_fsdp.testing.common_fsdp import MLP, FSDPTest


class TestFullyShardStateDictMultiProcess(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    @parameterize("decay", [0.9, 0.99])
    @parameterize("recurse", [True, False])
    def test_ema(self, decay: float, recurse: bool) -> None:
        """
        Tests the Exponential Moving Average (EMA) parameter update functionality,
        including support for nested FSDP models.

        Args:
            decay (float): The decay factor for the EMA calculation.
            recurse (bool): Whether to apply EMA operations recursively to all
                MagiFSDP submodules or just the top-level module.
        """
        # 0. Initialize the mixed precision policy to enable EMA parameters.
        mp_policy = MixedPrecisionPolicy(ema_param_dtype=torch.float32)

        # 1. Initialize two models. `old_model` represents the initial EMA state,
        #    and `new_model` represents the latest model weights to be averaged in.
        #    Both are simple MLPs and are moved to the current CUDA device.
        old_model: torch.nn.Module = MLP(3, device=torch.cuda.current_device())
        new_model: torch.nn.Module = MLP(3, device=torch.cuda.current_device())

        # 2. Synchronize the initial parameters of both models across all processes
        #    by averaging them. This ensures all ranks start with the same weights.
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
            dist.all_reduce(old_param.data, op=dist.ReduceOp.AVG)
            dist.all_reduce(new_param.data, op=dist.ReduceOp.AVG)

        # 3. Create a reference model to compute the expected EMA result manually.
        #    The formula for EMA is: ema_new = decay * ema_old + (1 - decay) * model_new.
        #    The logic here depends on the `recurse` flag.
        ref_model = copy.deepcopy(old_model)
        if recurse:
            # If recurring, apply EMA to all parameters in the model.
            for ref_param, new_param in zip(
                ref_model.parameters(), new_model.parameters()
            ):
                ref_param.data = decay * ref_param.data + (1 - decay) * new_param.data
        else:
            # If not recurring, the EMA operation should only affect the parameters
            # of the top-level FSDP module (`out_proj`), not the nested one (`in_proj`).
            # The parameters of the nested FSDP module are expected to be overwritten by `new_model`'s weights.
            for ref_param, new_param in zip(
                ref_model.in_proj.parameters(), new_model.in_proj.parameters()
            ):
                ref_param.data.copy_(new_param.data)
            for ref_param, new_param in zip(
                ref_model.out_proj.parameters(), new_model.out_proj.parameters()
            ):
                ref_param.data = decay * ref_param.data + (1 - decay) * new_param.data

        # 4. Wrap the models with `fully_shard` to create a nested FSDP structure.
        fully_shard(old_model.in_proj, mp_policy=mp_policy)
        fsdp_model = fully_shard(old_model, mp_policy=mp_policy)
        fully_shard(new_model.in_proj, mp_policy=mp_policy)
        fsdp_new_model = fully_shard(new_model, mp_policy=mp_policy)

        # 5. Load the `new_model` parameters into the `fsdp_model` to simulate
        #    receiving new weights before the EMA update.
        for param, new_param in zip(
            fsdp_model.parameters(), fsdp_new_model.parameters()
        ):
            param.data.copy_(new_param.data)

        # 6. Perform the EMA update.
        fsdp_model.update_ema_params(decay=decay, async_op=False, recurse=recurse)

        # 7. Compare the parameters of the FSDP model (now holding the EMA values)
        #    with the manually computed reference model parameters.
        with magi_fsdp_switch_params(
            fsdp_model, param_type="ema", recurse=recurse, raise_if_missing=True
        ):
            for param, ref_param in zip(
                fsdp_model.parameters(), ref_model.parameters()
            ):
                param = param.full_tensor()
                torch.testing.assert_close(param, ref_param, atol=1e-10, rtol=1e-5)

        #  8. Verify that the active parameters of `fsdp_model` have been restored
        #    and are now identical to the `fsdp_new_model` parameters.
        for param, new_param in zip(
            fsdp_model.parameters(), fsdp_new_model.parameters()
        ):
            param = param.full_tensor()
            new_param = (
                new_param.full_tensor()
            )  # Gather the new_param as well since it's from an FSDP model.
            torch.testing.assert_close(param, new_param, atol=1e-10, rtol=1e-5)

        # 9. Finally, verify that using `magi_fsdp_use_params` to switch to EMA
        #    parameters works correctly.
        magi_fsdp_use_params(
            fsdp_model, recurse=recurse, param_type="ema", raise_if_missing=True
        )
        for param, ref_param in zip(fsdp_model.parameters(), ref_model.parameters()):
            param = param.full_tensor()
            torch.testing.assert_close(param, ref_param, atol=1e-10, rtol=1e-5)


if __name__ == "__main__":
    # Entry point to run the tests.
    run_tests()
