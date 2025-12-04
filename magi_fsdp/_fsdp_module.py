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

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, Optional, cast

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_map
from typeguard import check_type

from ._fsdp_init import _get_post_forward_mesh_info
from ._fsdp_param import (
    MagiFSDPParam,
    ShardedEmaParam,
    ShardedMainParam,
)
from ._fsdp_param_group import MagiFSDPParamGroup
from ._fsdp_state import MagiFSDPState, _get_module_fsdp_state

_SWITCH_LOCK_ATTR = "_magi_fsdp_switch_active"


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "MagiFSDPModule",
    "UnshardHandle",
]


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, MagiFSDPModule):
            raise ValueError(f"Expects MagiFSDPModule but got {type(module)}: {module}")


class MagiFSDPModule:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the MagiFSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `MagiFSDP<...>` class
        # and index 1 is the `MagiFSDPModule` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)

        return self

    def reshard(self) -> None:
        """
        Reshards the module's parameters, freeing the unsharded parameters if
        they are allocated and registering the sharded parameters to the
        module. This method is *not* recursive.
        """
        state = self._get_fsdp_state()
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.reshard()

    def unshard(self, async_op: bool = False) -> Optional[UnshardHandle]:
        """
        Unshards the module's parameters by allocating memory and all-gathering
        the parameters. This method is *not* recursive. The unshard follows the
        :class:`MixedPrecisionPolicy`, so it will all-gather following
        ``param_dtype`` if set.

        Args:
            async_op (bool): If ``True``, then returns a :class:`UnshardHandle`
                that has a :meth:`wait` method to wait on the unshard op. If
                ``False``, then returns ``None`` and waits on the handle inside
                this function.

        .. note:: If ``async_op=True``, then MagiFSDP will wait on the pending
            unshard in the module's pre-forward for the user. The user only
            needs to call :meth:`wait` explicitly if the wait should happen
            before pre-forward.
        """
        state = self._get_fsdp_state()
        fsdp_param_group = state._fsdp_param_group
        if fsdp_param_group is not None:
            fsdp_param_group.lazy_init()
            fsdp_param_group.unshard(async_op=async_op)
        handle = _UnshardHandleImpl(fsdp_param_group)
        if async_op:
            return handle
        handle.wait()
        return None

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        Sets whether the next backward is the last one. On the last backward,
        MagiFSDP waits on pending gradient reduction and clears internal data
        data structures for backward prefetching. This can be useful for
        microbatching.
        """
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation *without communication*. For HSDP, this controls
        both reduce-scatter and all-reduce together. This is the equivalence of
        `no_sync` in FSDP1.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
            recurse (bool): Whether to set for all MagiFSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reduce_grads = requires_gradient_sync
                    fsdp_param_group.all_reduce_grads = requires_gradient_sync

    def set_requires_update_param(
        self, requires_update_param: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should copy main sharded parameters to sharded parameters
        when main parameters are enabled. This can be useful for
        gradient accumulation and resume training.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.copy_main_param = requires_update_param

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should all-reduce gradients. This can be used to
        implement gradient accumulation with only reduce-scatter but not
        all-reduce for HSDP.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.all_reduce_grads = requires_all_reduce

    def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after forward. This can be
        used to change the ``reshard_after_forward`` MagiFSDP arg at runtime. For
        example, this can be used to set the MagiFSDP root module's value to
        ``True`` (since it is otherwise specially set to ``False``), or it can
        set an MagiFSDP module's value to ``False`` for running evals and set back
        to ``True`` for training.

        Args:
            reshard_after_forward (bool): Whether to reshard parameters after
                forward.
            recurse (bool): Whether to set for all MagiFSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.post_forward_mesh_info = (
                        _get_post_forward_mesh_info(
                            reshard_after_forward, fsdp_param_group.mesh_info
                        )
                    )

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after backward. This can
        be used during gradient accumulation to trade off higher memory for
        reduced communication since the unsharded parameters do not need to be
        re-all-gathered before the next forward.

        Args:
            reshard_after_backward (bool): Whether to reshard parameters after
                backward.
            recurse (bool): Whether to set for all MagiFSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reshard_after_backward = reshard_after_backward

    def set_offload_activation_after_forward(
        self, offload_activation_after_forward: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should offload activation after forward. This can
        be used during activation offload to set the ``offload_activation_after_forward``
        of last module. The activations of the last module should not be offloaded.

        Args:
            offload_activation_after_forward (bool): Whether to offload activation after
                forward. True for offload and False for not offload.
            recurse (bool): Whether to set for all MagiFSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                state._cpu_offload_manager.offload_activation_after_forward = (
                    offload_activation_after_forward
                )

    def set_modules_to_forward_prefetch(self, modules: list[MagiFSDPModule]) -> None:
        """
        Sets the MagiFSDP modules for which this MagiFSDP module should explicitly
        prefetch all-gathers in forward. The prefetching runs after this
        module's all-gather copy-out.

        Passing a singleton list containing the next MagiFSDP module gives the same
        all-gather overlap behavior as the default overlap behavior, except the
        prefetched all-gather is issued earlier from the CPU. Passing a list
        with at least length two is required for more aggressive overlap and
        will use more reserved memory.

        Args:
            modules (List[MagiFSDPModule]): MagiFSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_forward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_modules_to_backward_prefetch(self, modules: list[MagiFSDPModule]) -> None:
        """
        Sets the MagiFSDP modules for which this MagiFSDP module should explicitly
        prefetch all-gathers in backward. This overrides the default backward
        pretching implementation that prefetches the next MagiFSDP module based on
        the reverse post-forward order.

        Passing a singleton list containing the previous MagiFSDP module gives the
        same all-gather overlap behavior as the default overlap behavior.
        Passing a list with at least length two is required for more aggressive
        overlap and will use more reserved memory.

        Args:
            modules (List[MagiFSDPModule]): MagiFSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_backward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_all_reduce_hook(
        self,
        hook: Callable[[torch.Tensor], None],
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """
        Args:
            hook (Callable[[torch.Tensor], None]): User-defined all-reduce hook
                with expected signature ``hook(reduce_output: torch.Tensor) -> None``
                where ``reduce_output`` is the reduce-scatter output if only
                using MagiFSDP or the all-reduce output if using native HSDP.
            stream (Optional[torch.cuda.Stream]): Stream to run the all-reduce
                hook in. This should only be set if not using native HSDP. If
                using native HSDP, the hook will run in the internally defined
                all-reduce stream used by the native HSDP all-reduce.
        """
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group._all_reduce_hook = hook
            if stream is not None:
                if fsdp_param_group._is_hsdp:
                    raise ValueError("stream cannot be set when using native HSDP")
                fsdp_param_group._all_reduce_hook_stream = stream

    def set_post_optim_event(self, event: torch.Event) -> None:
        """
        Sets a post-optimizer-step event for the root MagiFSDP module to wait the
        all-gather streams on.

        By default, the root MagiFSDP module waits the all-gather streams on the
        current stream to ensure that the optimizer step has finished before
        all-gathering. However, this may introduce false dependencies if
        there is unrelated computation after the optimizer step. This API
        allows the user to provide their own event to wait on. After the root
        waits on the event, the event is discarded, so this API should be
        called with a new event each iteration.

        Args:
            event (torch.Event): Event recorded after the optimizer step
                to wait all-gather streams on.
        """
        self._get_fsdp_state()._state_ctx.post_optim_event = event

    def set_reduce_scatter_divide_factor(self, factor: float) -> None:
        """
        Sets a custom divide factor for the reduce-scatter. This becomes a
        custom reduce op using NCCL's PreMulSum, which allows multiplying by
        the factor before reduction.

        Args:
            factor (float): Custom divide factor.
        """
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            mul_factor = 1.0 / float(factor)
            reduce_op = torch.distributed._make_nccl_premul_sum(mul_factor)
            fsdp_param_group.reduce_scatter_reduce_op = reduce_op

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        """
        Sets whether the MagiFSDP module's parameters need to be unsharded in
        backward. This can be used in expert cases when the user knows that all
        parameters in this MagiFSDP module's parameter group are not needed for
        backward computation (e.g. embedding).
        """
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group.unshard_in_backward = unshard_in_backward

    def _set_unshard_async_op(self, async_op: bool):
        """
        Sets whether to use ``async_op=True`` or ``False`` for the pre-forward
        and pre-backward unshard op. This defaults to ``False`` but can be set
        to ``True`` with this method.

        Setting this to ``True`` allows the all-gather allocations to happen in
        the default stream, avoiding inter-stream memory fragmentation.
        However, you must use explicit prefetching (e.g. via :meth:`unshard`)
        in forward to still get overlap, and the pre-all-gather ops like dtype
        casting and copy-in will not overlap with compute.
        """
        self_module = cast(nn.Module, self)
        for module in self_module.modules():
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.unshard_async_op = async_op

    def _get_fsdp_state(self) -> MagiFSDPState:
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is None:
            raise AssertionError(f"No MagiFSDP state found on {self}")
        return state

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        # Reshard to ensure that sharded parameters are registered
        self.reshard()
        ret = cast(nn.Module, super())._apply(*args, **kwargs)  # type: ignore[misc]
        state = self._get_fsdp_state()
        if not (fsdp_param_group := state._fsdp_param_group):
            return ret

        # We pass recurse=False because nn.Module._apply recursively calls _apply on submodules.
        # Example hierarchy:
        # MagiFSDPModule(block)
        #   ├── MagiFSDPModule(layer1)
        #   └── nn.Module(layer2)
        # block._apply() will invoke layer1._apply() and layer2._apply().
        # If recurse=True, magi_fsdp_switch_params would be triggered twice for layer1
        # (once by block, once by itself), which is incorrect.
        with magi_fsdp_switch_params(self, param_type="main", recurse=False):
            # Apply to main params. Buffers will be applied again, but this is acceptable.
            cast(nn.Module, super())._apply(*args, **kwargs)

        with magi_fsdp_switch_params(self, param_type="ema", recurse=False):
            # Apply to EMA params. Buffers will be applied again, but this is acceptable.
            cast(nn.Module, super())._apply(*args, **kwargs)

        # TODO(littsk): Remove this padding logic once DTensor pads the local tensor:
        # https://github.com/pytorch/pytorch/issues/113045
        with torch.no_grad():
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.reset_sharded_param()

                # If main/EMA params exist, copy the data from sharded_param to them
                # so that their data is consistent after _apply
                fsdp_param.reset_sharded_main_param()
                fsdp_param.reset_sharded_ema_param()

        return ret

    # Parameter-related methods
    def reset_parameters(self) -> None:
        if not hasattr(super(), "reset_parameters"):
            return

        super().reset_parameters()  # type: ignore[misc]

        # TODO(littsk): Eliminate redundant copying caused by recurse
        mfdp_params: list[MagiFSDPParam] = list(self.magi_fsdp_parameters(recurse=True))
        tree_map(lambda mfdp_param: mfdp_param.reset_sharded_main_param(), mfdp_params)
        tree_map(lambda mfdp_param: mfdp_param.reset_sharded_ema_param(), mfdp_params)

    def magi_fsdp_parameters(self, recurse: bool = True) -> Generator[MagiFSDPParam]:
        """
        Similar to nn.Module.parameters(), but only returns MagiFSDPParam type parameters.
        NOTE: The returned results are deduplicated and ordered.
        """
        if recurse:
            modules = list(cast(nn.Module, self).modules())
        else:
            modules = [self]

        cast(nn.Module, self).parameters()

        # Get the fsdp_state of each module, None if it doesn't exist
        states = tree_map(
            lambda module: _get_module_fsdp_state(module),
            list(modules),
        )

        # Filter out None states, regular nn.Module won't have fsdp_state
        states: list[MagiFSDPState] = [state for state in states if state is not None]

        param_groups: list[MagiFSDPParamGroup] = tree_map(
            lambda state: cast(MagiFSDPState, state)._fsdp_param_group,
            states,
        )

        # Filter out None param groups, MagiFSDPState may not have param group
        param_groups = [pg for pg in param_groups if pg is not None]

        params, _ = tree_flatten(
            [params_group.fsdp_params for params_group in param_groups]
        )

        assert all(
            isinstance(param, MagiFSDPParam) for param in params
        ), "Expected all parameters to be MagiFSDPParam."

        # Deduplicate and preserve order
        # FIXME: Is this deduplication step really necessary?
        seen_sharded_params = set()
        unique_params = []
        for param in cast(list[MagiFSDPParam], params):
            if param.sharded_param not in seen_sharded_params:
                seen_sharded_params.add(param.sharded_param)
                unique_params.append(param)

        for param in unique_params:
            yield param

    def update_ema_params(
        self, decay: float, async_op: bool = False, recurse: bool = True
    ) -> None:
        """
        Update the EMA params with the current model parameters.

        Args:
            decay (float): The decay factor for the EMA calculation.
                formula for EMA is: ema_param = decay * ema_param + (1 - decay) * param
            async_op (bool): If ``True``, then the EMA update is done
                asynchronously. Otherwise, it is done synchronously.
            recurse (bool): Whether to update EMA params for all MagiFSDP
                submodules or just the passed-in module.
        """

        # TODO: support async_op
        assert not async_op, "async_op is not supported yet"

        mfdp_params = list(self.magi_fsdp_parameters(recurse=recurse))

        tree_map(
            lambda mfdp_param: cast(MagiFSDPParam, mfdp_param).update_ema_param(
                decay=decay
            ),
            mfdp_params,
        )

    def named_main_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Generator[tuple[str, ShardedMainParam]]:
        with magi_fsdp_switch_params(self, param_type="main"):
            yield from cast(nn.Module, self).named_parameters(
                prefix, recurse, remove_duplicate
            )

    def main_parameters(self, recurse: bool = True) -> Generator[ShardedMainParam]:
        with magi_fsdp_switch_params(self, param_type="main"):
            yield from cast(nn.Module, self).parameters(recurse)

    def named_ema_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Generator[tuple[str, ShardedEmaParam]]:
        with magi_fsdp_switch_params(self, param_type="ema"):
            yield from cast(nn.Module, self).named_parameters(
                prefix, recurse, remove_duplicate
            )

    def ema_parameters(self, recurse: bool = True) -> Generator[ShardedEmaParam]:
        with magi_fsdp_switch_params(self, param_type="ema"):
            yield from cast(nn.Module, self).parameters(recurse)


class UnshardHandle:
    """
    A handle to wait on a :meth:`MagiFSDPModule.unshard` op.
    """

    def wait(self) -> None:
        """
        Waits on the unshard op. This ensures that the current stream can use
        the unsharded parameters, which are now registered to the module.
        """
        return


class _UnshardHandleImpl(UnshardHandle):
    def __init__(self, fsdp_param_group: Optional[MagiFSDPParamGroup]):
        self._fsdp_param_group = fsdp_param_group

    def wait(self):
        if self._fsdp_param_group is not None:
            self._fsdp_param_group.wait_for_unshard()
            # Avoid keeping a reference
            self._fsdp_param_group = None


def magi_fsdp_use_params(
    module: MagiFSDPModule | nn.Module,
    recurse: bool = True,
    param_type: Literal["main", "ema"] = "main",
    raise_if_missing: bool = False,
) -> MagiFSDPModule | nn.Module:
    """
    Replace parameters in a MagiFSDPModule with main parameters or EMA parameters

    Args:
        module (MagiFSDPModule | nn.Module): The MagiFSDPModule to operate on, or a regular nn.Module.
            If a regular nn.Module is passed in, no replacement will be performed.
        recurse (bool): Whether to recursively replace parameters in submodules
        param_type (Literal["main", "ema"]): The type of parameters to replace with,
            "main" for main parameters, "ema" for EMA parameters
        raise_if_missing (bool): If True and the target parameter type doesn't exist
            for some parameter, raise an exception

    Returns:
        module (MagiFSDPModule | nn.Module): The input module with parameters replaced.
    """
    # Guard Clauses
    if not isinstance(module, MagiFSDPModule):
        return module

    check_type(param_type, Literal["main", "ema"])

    def use_param(param: MagiFSDPParam):
        check_type(param, MagiFSDPParam)

        tgt_param = (
            param.sharded_main_param
            if param_type == "main"
            else param.sharded_ema_param
        )

        if tgt_param is None and raise_if_missing:
            raise RuntimeError(f"{param_type} param is not created for param: {param}")
        elif tgt_param is not None:
            param.sharded_param.data.copy_(tgt_param.data)

    mfdp_params = list(module.magi_fsdp_parameters(recurse=recurse))

    tree_map(use_param, mfdp_params)

    return module


@contextmanager
def magi_fsdp_switch_params(
    module: MagiFSDPModule | nn.Module,
    param_type: Literal["main", "ema"],
    recurse: bool = True,
    raise_if_missing: bool = False,
    retain_if_missing: bool = False,
):
    """
    Context manager to switch MagiFSDPModule parameters to main parameters or EMA parameters

    Args:
        module (MagiFSDPModule | nn.Module): The MagiFSDPModule to operate on, or a regular nn.Module.
            If a regular nn.Module is passed in, no switching will be performed.
        param_type (Literal["main", "ema"]): The type of parameters to switch to,
            "main" for main parameters, "ema" for EMA parameters
        recurse (bool): Whether to recursively switch parameters in submodules
        raise_if_missing (bool): If True and the target parameter type doesn't exist
            for some parameter, raise an exception
        retain_if_missing (bool): If True and the target parameter type doesn't exist
            for some parameter, keep the original parameter unchanged
    """
    # Guard Clauses
    assert (
        not raise_if_missing or not retain_if_missing
    ), "Only one of raise_if_missing or retain_if_missing can be True."

    check_type(param_type, Literal["main", "ema"])

    if not isinstance(module, MagiFSDPModule):
        # Not a MagiFSDPModule, nothing to switch
        yield
        return

    # Identify the scope of modules to lock.
    # If recurse is True, we must prevent nested usage on any submodule.
    if recurse:
        modules_to_lock = list(cast(nn.Module, module).modules())
    else:
        modules_to_lock = [module]

    # Check if any module in the scope is already being switched
    for m in modules_to_lock:
        if getattr(m, _SWITCH_LOCK_ATTR, False):
            raise RuntimeError(
                f"Nested usage of 'magi_fsdp_switch_params' detected on module '{type(m).__name__}'."
            )

    # Apply lock to all affected modules
    for m in modules_to_lock:
        object.__setattr__(m, _SWITCH_LOCK_ATTR, True)

    try:

        def switch_param(param: MagiFSDPParam):
            check_type(param, MagiFSDPParam)

            tgt_param = (
                param.sharded_main_param
                if param_type == "main"
                else param.sharded_ema_param
            )

            if tgt_param is None and raise_if_missing:
                raise RuntimeError(
                    f"{param_type} param is not created for param: {param}"
                )

            org_param = param._swap_param_on_modules(tgt_param)

            if tgt_param is None and retain_if_missing:
                # If the target param is missing, we retain the original param in place.
                param._setattr_on_modules(org_param)

            return org_param

        mfdp_params = list(module.magi_fsdp_parameters(recurse=recurse))

        org_params = tree_map(switch_param, mfdp_params)

        yield
    finally:
        try:
            # Restore parameters in the finally block to ensure execution
            # even if an exception occurs during the yield
            def restore_param(param: MagiFSDPParam, org_param: nn.Parameter):
                check_type(param, MagiFSDPParam)
                check_type(org_param, nn.Parameter)
                param._setattr_on_modules(org_param)

            tree_map(restore_param, mfdp_params, org_params)

        finally:
            # Always release locks, regardless of whether the switch happened or failed
            for m in modules_to_lock:
                if hasattr(m, _SWITCH_LOCK_ATTR):
                    object.__delattr__(m, _SWITCH_LOCK_ATTR)
