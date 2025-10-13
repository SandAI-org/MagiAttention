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

from magi_fsdp import fully_shard
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
        #    The `in_proj` submodule is wrapped first, then the entire model.
        fully_shard(old_model.in_proj)
        fsdp_model = fully_shard(old_model)
        fully_shard(new_model.in_proj)
        fsdp_new_model = fully_shard(new_model)

        # 5. Initialize the internal storage for EMA parameters within the FSDP model.
        #    `recurse` determines if storage is created for nested FSDP modules.
        fsdp_model.create_ema_params(recurse=recurse)

        # 6. To set up the EMA update, we first load the `new_model` weights into
        #    the main parameters of `fsdp_model`. The `update_ema_params` function
        #    will then use these main parameters to update its internal EMA storage.
        for param, new_param in zip(
            fsdp_model.parameters(), fsdp_new_model.parameters()
        ):
            param.data.copy_(new_param.data)

        # 7. Perform the EMA update. `recurse` controls whether this is applied to submodules.
        #    After the update, swap the active model parameters with the updated EMA parameters.
        fsdp_model.update_ema_params(decay=decay, async_op=False, recurse=recurse)
        fsdp_model.swap_ema_params(recurse=recurse)

        # 8. Compare the parameters of the FSDP model (now holding the EMA values)
        #    with the manually computed reference model parameters.
        #    `param.full_tensor()` gathers the sharded parameter across all ranks.
        for param, ref_param in zip(fsdp_model.parameters(), ref_model.parameters()):
            param = param.full_tensor()
            torch.testing.assert_close(param, ref_param, atol=1e-10, rtol=1e-5)

        # 9. Test swapping back and then using the `use_ema_params` method.
        #    First, swap back, controlled by the `recurse` flag.
        fsdp_model.swap_ema_params(recurse=recurse)

        #    Verify that the active parameters of `fsdp_model` have been restored
        #    and are now identical to the `fsdp_new_model` parameters.
        for param, new_param in zip(
            fsdp_model.parameters(), fsdp_new_model.parameters()
        ):
            param = param.full_tensor()
            new_param = (
                new_param.full_tensor()
            )  # Gather the new_param as well since it's from an FSDP model.
            torch.testing.assert_close(param, new_param, atol=1e-10, rtol=1e-5)

        #    Now, call `use_ema_params`. This copies the EMA values into the active parameters.
        #    The scope of this operation is also controlled by `recurse`.
        fsdp_model.use_ema_params(recurse=recurse)
        #    Finally, verify that the active parameters once again match the reference model.
        for param, ref_param in zip(fsdp_model.parameters(), ref_model.parameters()):
            param = param.full_tensor()
            torch.testing.assert_close(param, ref_param, atol=1e-10, rtol=1e-5)


if __name__ == "__main__":
    # Entry point to run the tests.
    run_tests()
