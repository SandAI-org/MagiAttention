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

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Optional, Union, overload

import torch.nn as nn
from torch.distributed._composable import contract
from torch.distributed.utils import _get_root_modules

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_common import MagiFSDPMeshInfo, MagiHSDPMeshInfo
from ._fsdp_init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from ._fsdp_module import MagiFSDPModule
from ._fsdp_param_group import MagiFSDPParamGroup
from ._fsdp_state import MagiFSDPState

if TYPE_CHECKING:
    from torch.distributed.tensor import DeviceMesh, Shard

__all__ = [
    "fully_shard",
    "register_fsdp_forward_method",
]


cls_to_fsdp_cls: dict[type, type] = {}


@overload
def fully_shard(
    module: list[nn.Module],
    *,
    mesh: Optional[DeviceMesh] = ...,
    reshard_after_forward: Union[bool, int] = ...,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: Optional[set[nn.Parameter]] = ...,
) -> list[MagiFSDPModule]:
    ...


@overload
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = ...,
    reshard_after_forward: Union[bool, int] = ...,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    offload_policy: OffloadPolicy = ...,
    ignored_params: Optional[set[nn.Parameter]] = ...,
) -> MagiFSDPModule:
    ...


# The decorator adds a state object to `module` that can be accessed via
# `fully_shard.state(module)`. The state object and module are 1:1.
# [1] Python runtime decorator does not play well with static type checking
# so suppressing some type checks to support type overloads
# such that caller can still get correct return types based on input type
@contract(state_cls=MagiFSDPState)  # type: ignore[misc] # see [1]
def fully_shard(
    module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: Optional[set[nn.Parameter]] = None,
):
    """
    Apply fully sharded data parallelism (MagiFSDP) to ``module``, where MagiFSDP
    shards module parameters, gradients, and optimizer states across data
    parallel workers to save memory at the cost of communication.

    At initialization, MagiFSDP shards the module's parameters across the data
    parallel workers given by ``mesh``. Before forward, MagiFSDP all-gathers the
    sharded parameters across the data-parallel workers to get the unsharded
    parameters for forward computation. If ``reshard_after_forward`` is
    ``True``, then MagiFSDP frees the unsharded parameters after forward and
    re-all-gathers them in backward before gradient computation. After gradient
    computation, MagiFSDP frees the unsharded parameters and reduce-scatters the
    unsharded gradients across data-parallel workers.

    This implementation represents the sharded parameters as :class:`DTensor` s
    sharded on dim-0, while the unsharded parameters will be like the original
    parameters on ``module`` (e.g. :class:`torch.Tensor` if originally
    :class:`torch.Tensor`). A module
    `forward pre-hook
    <https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook>`_
    on ``module`` all-gathers the parameters, and a module
    `forward hook
    <https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook>`_
    on ``module`` frees them (if needed). Similar backward hooks all-gather
    parameters and later free parameters and reduce-scatter gradients.

    Since grouping multiple tensors together for one collective is critical for
    communication efficiency, this implementation makes this grouping first
    class. Calling :meth:`fully_shard` on ``module`` constructs one group that
    includes the parameters in ``module.parameters()`` except those already
    assigned to a group from an earlier call on a submodule. This means that
    :meth:`fully_shard` should be called bottom-up on your model. Each group's
    parameters are all-gathered in one collective, and its gradients are
    reduce-scattered in one collective. Partitioning the model into multiple
    groups ("layer by layer") allows for peak memory savings and communication/computation
    overlap. Users generally should *not* call :meth:`fully_shard` only on the
    topmost root module.

    Args:
        module (Union[nn.Module, List[nn.Module]): The module or modules to
            shard with MagiFSDP and group together for communication.
        mesh (Optional[DeviceMesh]): This data parallel mesh defines the
            sharding and device. If 1D, then parameters are fully sharded
            across the 1D mesh (FSDP) with ``(Shard(0),)`` placement. If 2D,
            then parameters are sharded across the 1st dim and replicated
            across the 0th dim (HSDP) with ``(Replicate(), Shard(0))``
            placement. The mesh's device type gives the device type used for
            communication; if a CUDA or CUDA-like device type, then we use the
            current device.
        reshard_after_forward (Union[bool, int]): This controls the parameter
            behavior after forward and can trade off memory and communication:

            - If ``True``, then this reshards parameters after forward and
              re-all-gathers in backward.
            - If ``False``, then this keeps the unsharded parameters in memory
              after forward and avoids the all-gather in backward.
            - If an ``int``, then this represents the world size to reshard to
              after forward. It should be a non-trivial divisor of the ``mesh``
              shard dim size (i.e. excluding 1 and the dim size itself). A
              choice may be the intra-node size (e.g. ``torch.cuda.device_count()``).
              This allows the all-gather in backward to be over a smaller world
              size at the cost of higher memory usage than setting to ``True``.
            - The root MagiFSDP state has its value specially set to ``False`` as a
              heuristic since its parameters would typically be immediately
              all-gathered for backward.
            - After forward, the parameters registered to the module depend on
              to this: The registered parameters are the sharded parameters if
              ``True``; unsharded parameters if ``False``; and the paramters
              resharded to the smaller mesh otherwise. To modify the parameters
              between forward and backward, the registered parameters must be
              the sharded parameters. For ``False`` or an ``int``, this can be
              done by manually resharding via :meth:`reshard`.
        shard_placement_fn (Optional[Callable[[nn.Parameter], Optional[Shard]]]):
            This callable can be used to override the sharding placement for a
            parameter to shard a parameter on a dimension other than dim-0. If
            this callable returns a :class:`Shard` placement (not ``None``),
            then MagiFSDP will shard according to that placement (e.g. ``Shard(1)``).
            If sharding on a nonzero dim, we currently require even sharding,
            i.e. the tensor dim size on that dim must be divisible by the MagiFSDP
            shard mesh size.
        mp_policy (MixedPrecisionPolicy): This controls the mixed precision
            policy, which offers parameter/reduction mixed precision for this
            module. See :class:`MixedPrecisionPolicy` for details.
        offload_policy (OffloadPolicy): This controls the offloading policy,
            which offers parameter/gradient/optimizer state offloading. See
            :class:`OffloadPolicy` and its subclasses for details.
        ignored_params: Optional(Set[nn.Parameter]): The set of parameters to be
            ignored by MagiFSDP. They will not be sharded, nor moved to the device
            during init, nor have their gradients reduced in backward.

    Returns:
        MagiFSDPModule: The module with MagiFSDP applied (in-place).
    """
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = MagiFSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        if mesh.mesh_dim_names is None:
            raise AssertionError(
                "Please init the 2D mesh for HSDP with mesh_dim_names specified"
            )
        mesh_info = MagiHSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined] # see [1]
    state.init(modules, device, mp_policy)

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = MagiFSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place MagiFSDP leftmost for highest priority in the method resolution order
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"MagiFSDP{cls.__name__}", (MagiFSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return arg_module


def _unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "MagiFSDP does not support deepcopy. Please use state dict for serialization."
    )


def register_fsdp_forward_method(module: nn.Module, method_name: str) -> None:
    """
    Registers a method on ``module`` to be considered a forward method for
    MagiFSDP.

    MagiFSDP all-gathers parameters pre-forward and optionally frees parameters
    post-forward (depending on ``reshard_after_forward``). MagiFSDP only knows to
    do this for :meth:`nn.Module.forward` by default. This function patches a
    user-specified method to run the pre/post-forward hooks before/after the
    method, respectively. If ``module`` is not an :class:`MagiFSDPModule`, then
    this is a no-op.

    Args:
        module (nn.Module): Module to register the forward method on.
        method_name (str): Name of the forward method.
    """
    if not isinstance(module, MagiFSDPModule):
        # Make no-op to allow including both when using/not using MagiFSDP
        return
    if not hasattr(module, method_name):
        raise ValueError(f"{type(module)} does not have a method {method_name}")
    orig_method = getattr(module, method_name)

    @functools.wraps(orig_method)
    def wrapped_method(self, *args, **kwargs):
        fsdp_state = self._get_fsdp_state()
        args, kwargs = fsdp_state._pre_forward(self, args, kwargs)
        out = orig_method(*args, **kwargs)
        return fsdp_state._post_forward(self, args, out)

    # Use `__get__` to make `wrapped_method` an instance method
    setattr(
        module,
        method_name,
        wrapped_method.__get__(module, type(module)),  # type:ignore[attr-defined]
    )
