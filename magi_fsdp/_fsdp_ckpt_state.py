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

from torch.distributed.checkpoint.stateful import Stateful
from typeguard import check_type

from magi_fsdp._fsdp_module import MagiFSDPModule

MAIN_PARAM = "main_param"
EMA_PARAM = "ema_param"


# TODO(littsk): Use a single wrapper
class MainParamStatefulWrapper(Stateful):
    """
    Wrapper for managing main parameters in MagiFSDP.
    """

    def __init__(self, model: MagiFSDPModule) -> None:
        check_type(model, MagiFSDPModule)
        self.model = model

    def state_dict(self) -> Dict[str, Any]:
        # TODO(littsk): Use FQN here to be consistent with DCP
        return {
            MAIN_PARAM: {
                name: param for name, param in self.model.named_main_parameters()
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        curr_state_dict = self.state_dict()

        for name, param in state_dict.get(MAIN_PARAM, {}).items():
            curr_param = curr_state_dict[MAIN_PARAM][name]
            curr_param.data.copy_(param.data)


class EMAParamStatefulWrapper(Stateful):
    """
    Wrapper for managing EMA parameters in MagiFSDP.
    """

    def __init__(self, model: MagiFSDPModule) -> None:
        check_type(model, MagiFSDPModule)
        self.model = model

    def state_dict(self) -> Dict[str, Any]:
        # TODO(littsk): Use FQN here to be consistent with DCP
        return {
            EMA_PARAM: {
                name: param for name, param in self.model.named_ema_parameters()
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        curr_state_dict = self.state_dict()

        for name, param in state_dict.get(EMA_PARAM, {}).items():
            curr_param = curr_state_dict[EMA_PARAM][name]
            curr_param.data.copy_(param.data)
