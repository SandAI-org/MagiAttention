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

# Copyright (c) Meta Platforms, Inc. and affiliates

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """
    This configures MagiFSDP's mixed precision. Unlike autocast, this applies mixed
    precision at the module level, not op level, which means low-precision
    activations are saved for backward and high-to-low-precision casts are
    incurred only at module boundaries.

    MagiFSDP works well with module-level mixed precision since it keeps the
    high-precision sharded parameters in memory anyway. In other words, MagiFSDP
    does not require any extra memory to keep a high-precision copy of the
    parameters for the optimizer step.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for
            the unsharded parameter and hence the dtype for forward/backward
            computation and the parameter all-gather. If this is ``None``, then
            the unsharded parameter uses the original dtype. The optimizer step
            uses the sharded parameter in the original dtype. (Default:
            ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then the reduction
            uses the compute dtype. This can be used to run gradient reduction
            in full precision while using low precision for compute. If also
            gradient reduction is disabled via :meth:`set_requires_gradient_sync`,
            then MagiFSDP will accumulate gradients using ``reduce_dtype``.
            (Default: ``None``)
        output_dtype (Optional[torch.dtype]): This specifies the dtype for
            casting floating-point forward outputs. This can be used to
            help implement cases where different modules have different mixed
            precision policies. (Default: ``None``)
        cast_forward_inputs (bool): This specifies whether MagiFSDP should cast the
            forward's floating-point input tensors to ``param_dtype`` or not.
        main_param_dtype (Optional[torch.dtype]): The data type used for main
            parameters if enabled. This can be used to maintain main parameters
            in higher precision (e.g., ``torch.float32``) while training in
            mixed precision. If ``None`` and main parameters are enabled, the main
            parameter dtype will be set to ``torch.float32``. (Default: ``None``)
    """

    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True
    main_param_dtype: Optional[torch.dtype] = None
    ema_param_dtype: Optional[torch.dtype] = None


@dataclass
class OffloadPolicy:
    """
    This base class represents the policy of no offloading and is only used as
    the default value for the ``offload_policy`` arg.
    """


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    """
    This offload policy offloads activation, parameters, gradients, and optimizer
    states to CPU. Sharded parameters are copied host-to-device before all-gather.
    The all-gathered parameters are freed according to ``reshard_after_forward``.
    Sharded gradients are copied device-to-host in backward, and the optimizer
    step runs on CPU with CPU optimizer states. Activations are copied device-to-host
    after forward, and are copied host-to-device in after backward.

    Attributes:
        pin_memory (bool): Whether to pin activation, sharded parameter and gradient
            memory. Pinning memory allows both more efficient H2D/D2H copies
            and for the copies to overlap with compute. However, the pinned
            memory cannot be used by other processes. Set this to ``False`` if
            you have insufficient CPU memory. (Default: ``True``)
        offload_param (bool): Whether to offload param, grad and optimizer states to CPU.
            (Default: ``False``)
        offload_activation (bool): Whether to offload activation to CPU.
            (Default: ``True``)
        foreach_offload (bool): Offload all activations of a module at once, or transfer
            each activation individually. (Default: ``True``)
    """

    pin_memory: bool = True

    # whether to offload param, grad and optimizer
    offload_param: bool = False

    # whether to offload activation
    offload_activation: bool = True
    foreach_offload: bool = False
