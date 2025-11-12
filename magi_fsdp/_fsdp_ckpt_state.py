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

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from magi_fsdp._fsdp_module import switch_named_main_params_on_modules

MAIN_PARAM = "main_param"
EMA_PARAM = "ema_param"


class OptimizerStatefulWrapper(Stateful):
    """
    Wrapper for managing optimizer states in MagiFSDP.

    When main parameters are enabled, a context manager temporarily swaps
    module attributes (see `switch_named_main_params_on_modules`) so that DCP
    can correctly replace FQNs for optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        enable_main_param: bool = False,
    ) -> None:
        self.model = model
        self.optim = optim
        self.enable_main_param = enable_main_param

    def state_dict(self) -> Dict[str, Any]:
        with switch_named_main_params_on_modules(
            self.model, enable_main_param=self.enable_main_param
        ):
            return get_optimizer_state_dict(self.model, self.optim)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        with switch_named_main_params_on_modules(
            self.model, enable_main_param=self.enable_main_param
        ):
            set_optimizer_state_dict(
                self.model,
                self.optim,
                state_dict,
                options=StateDictOptions(strict=False),
            )


class MainParamStatefulWrapper(Stateful):
    """
    Wrapper for managing main parameters in MagiFSDP.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> Dict[str, Any]:
        return {
            MAIN_PARAM: {
                name: param for name, param in self.model.named_main_parameters()
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


class EMAParamStatefulWrapper(Stateful):
    """
    Wrapper for managing EMA parameters in MagiFSDP.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> Dict[str, Any]:
        return {
            EMA_PARAM: {
                name: param for name, param in self.model.named_ema_parameters()
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
