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

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Optional, Set, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from ._fsdp_api import CkptLoadPolicy, CkptSavePolicy, OptimPolicy
from ._fsdp_init import _get_post_forward_mesh_info
from ._fsdp_param import ParamPackInfo
from ._fsdp_param_group import MagiFSDPParamGroup
from ._fsdp_state import MagiFSDPState, _get_module_fsdp_state

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

    def set_is_first_forward(
        self, is_first_forward: bool, recurse: bool = True
    ) -> None:
        """
        Sets whether the next forward is the first one. On the first forward,
        when main parameters are enabled, MagiFSDP will perform a copy from
        sharded main parameters to sharded parameters. This can be useful for
        gradient accumulation and resume training.
        """
        fsdp_param_groups = self.collect_fsdp_param_groups(recurse)
        for param_gruop in fsdp_param_groups:
            param_gruop.is_first_forward = is_first_forward  # type: ignore[attr-defined]

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
        ret = super()._apply(*args, **kwargs)  # type: ignore[misc]
        state = self._get_fsdp_state()
        if not (fsdp_param_group := state._fsdp_param_group):
            return ret
        # TODO: Remove this padding logic once DTensor pads the local tensor:
        # https://github.com/pytorch/pytorch/issues/113045
        with torch.no_grad():
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.reset_sharded_param()
        return ret

    def create_main_params(self) -> None:
        """
        Create Main params for the model parameters manually to defered init.
        """
        optim_policy = self.optim_policy
        if optim_policy and optim_policy.enable_main_param:
            fsdp_param_groups = self.collect_fsdp_param_groups(True)
            for param_gruop in fsdp_param_groups:
                param_gruop.create_main_params()
        else:
            raise RuntimeError("optim_policy is missing or main_param is not enabled.")

    # EMA-related methods
    def create_ema_params(
        self, cpu_offload: bool = False, recurse: bool = True
    ) -> None:
        """
        Create EMA params for the model parameters.

        Args:
            cpu_offload (bool): Whether to offload the EMA params to CPU.
            recurse (bool): Whether to create EMA params for all MagiFSDP
                submodules or just the passed-in module.
        """
        assert not cpu_offload, "cpu_offload is not supported yet"
        # TODO: support cpu_offload

        self_module = cast(nn.Module, self)
        if recurse:
            modules = list(self_module.modules())
        else:
            modules = [self_module]
        states = set()
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if (
                    state not in states
                    and (fsdp_param_group := state._fsdp_param_group) is not None
                ):
                    fsdp_param_group.create_ema_params()
                    states.add(state)

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
        assert not async_op, "async_op is not supported yet"
        # TODO: support async_op
        self_module = cast(nn.Module, self)
        if recurse:
            modules = list(self_module.modules())
        else:
            modules = [self_module]
        states = set()
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if (
                    state not in states
                    and (fsdp_param_group := state._fsdp_param_group) is not None
                ):
                    fsdp_param_group.update_ema_params(decay=decay)
                    states.add(state)

    def use_ema_params(self, recurse: bool = True) -> None:
        """
        Use the EMA params for the model parameters.

        Args:
            recurse (bool): Whether to use EMA params for all MagiFSDP
                submodules or just the passed-in module.
        """
        self_module = cast(nn.Module, self)
        if recurse:
            modules = list(self_module.modules())
        else:
            modules = [self_module]
        states = set()
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if (
                    state not in states
                    and (fsdp_param_group := state._fsdp_param_group) is not None
                ):
                    fsdp_param_group.use_ema_params()
                    states.add(state)

    def swap_ema_params(self, recurse: bool = True) -> None:
        """
        Swap the EMA params with the current model parameters.

        Args:
            recurse (bool): Whether to swap EMA params for all MagiFSDP
                submodules or just the passed-in module.
        """
        self_module = cast(nn.Module, self)
        if recurse:
            modules = list(self_module.modules())
        else:
            modules = [self_module]
        states = set()
        for module in modules:
            if isinstance(module, MagiFSDPModule):
                state = module._get_fsdp_state()
                if (
                    state not in states
                    and (fsdp_param_group := state._fsdp_param_group) is not None
                ):
                    fsdp_param_group.swap_ema_params()
                    states.add(state)

    def collect_fsdp_param_groups(
        self, recurse: bool = True
    ) -> Set[MagiFSDPParamGroup]:
        """
        Collect unique MagiFSDPParamGroup instances from modules that have FSDP state.

        Args:
            recurse (bool): Whether to collect all param groups for all MagiFSDP
                submodules or just the passed-in module.
        """
        all_param_groups: set[MagiFSDPParamGroup] = set()
        self_module = cast(nn.Module, self)
        if recurse:
            modules = list(self_module.modules())
        else:
            modules = [self_module]
        for module in modules:
            if (state := _get_module_fsdp_state(module)) is None:
                continue
            if (param_group := state._fsdp_param_group) is not None:
                all_param_groups.add(param_group)
        return all_param_groups

    def collect_fsdp_param_pack_infos(
        self, recurse: bool = True
    ) -> dict[DTensor, ParamPackInfo]:
        """
        Collect ParamPackInfo for all parameters.

        Args:
            recurse (bool): Whether to collect ParamPackInfo for all FSDP
                submodules or just the passed-in module.

        Returns:
            dict[DTensor, ParamPackInfo]: Mapping from each sharded parameter to its ParamPackInfo.
        """
        fsdp_param_groups = self.collect_fsdp_param_groups(recurse)
        param_pack_infos: dict[DTensor, ParamPackInfo] = {}
        for param_gruop in fsdp_param_groups:
            param_pack_infos.update(param_gruop.param_pack_infos)
        return param_pack_infos

    def manualy_copy_from_main_params(self, recurse: bool = True) -> None:
        """
        Manually perform a copy from sharded main parameters to sharded parameters.
        This can be useful when saving a checkpoint at the last iteration.

        Args:
            recurse (bool): Whether to perform copy for all FSDP
                submodules or just the passed-in module.
        """
        fsdp_param_groups = self.collect_fsdp_param_groups(recurse)
        for param_gruop in fsdp_param_groups:
            param_gruop.copy_from_main_params()

    def named_main_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
        skip_none: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """
        Similar to `nn.Module.named_parameters`, but yielding the sharded main parameters
        managed by FSDP instead of the original module parameters.

        Args:
            skip_none (bool): Whether to skip none main parameters.
        """
        param_pack_infos = self.collect_fsdp_param_pack_infos(recurse)
        for name, sharded_param in self.named_parameters(prefix, recurse, remove_duplicate):  # type: ignore[attr-defined]
            if sharded_param in param_pack_infos:
                if (
                    not skip_none
                    or param_pack_infos[sharded_param].sharded_main_param is not None
                ):
                    yield name, param_pack_infos[sharded_param].sharded_main_param

    def main_parameters(
        self, recurse: bool = True, skip_none: bool = True
    ) -> Iterable[DTensor]:
        """
        Return an iterator over MagiFSDPModule main parameters.

        Args:
            recurse (bool): Whether to yield main parameters of this module
                and all submodules or just the passed-in module.
            skip_none (bool): Whether to skip none main parameters.

        Yields:
            DTensor: module main parameter
        """
        for _, main_param in self.named_main_parameters(
            recurse=recurse, skip_none=skip_none
        ):
            yield main_param

    def named_ema_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
        skip_none: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """
        Similar to `nn.Module.named_parameters`, but yielding the sharded EMA parameters
        managed by FSDP instead of the original module parameters.

        Args:
            skip_none (bool): Whether to skip none EMA parameters.
        """
        param_pack_infos = self.collect_fsdp_param_pack_infos(recurse)
        for name, sharded_param in self.named_parameters(prefix, recurse, remove_duplicate):  # type: ignore[attr-defined]
            if sharded_param in param_pack_infos:
                if (
                    not skip_none
                    or param_pack_infos[sharded_param].sharded_ema_param is not None
                ):
                    yield name, param_pack_infos[sharded_param].sharded_ema_param

    def ema_parameters(
        self, recurse: bool = True, skip_none: bool = True
    ) -> Iterable[DTensor]:
        """
        Return an iterator over MagiFSDPModule EMA parameters.

        Args:
            recurse (bool): Whether to yield EMA parameters of this module
                and all submodules or just the passed-in module.
            skip_none (bool): Whether to skip none EMA parameters.

        Yields:
            DTensor: module EMA parameter
        """
        for _, ema_param in self.named_ema_parameters(
            recurse=recurse, skip_none=skip_none
        ):
            yield ema_param

    @property
    def save_policy(self) -> CkptSavePolicy:
        """
        Save policy of the MagiFSDPModule.
        """
        if hasattr(self, "_save_policy"):
            return self._save_policy
        else:
            warnings.warn(
                "MagiFSDPModule has no save_policy. Defaulting not to save all optim parameters.",
                UserWarning,
            )
            return CkptSavePolicy(to_save_main_params=False, to_save_ema_params=False)

    @property
    def load_policy(self) -> CkptLoadPolicy:
        """
        Load policy of the MagiFSDPModule.
        """
        if hasattr(self, "_load_policy"):
            return self._load_policy
        else:
            warnings.warn(
                "MagiFSDPModule has no load_policy. Defaulting not to load all optim parameters.",
                UserWarning,
            )
            return CkptLoadPolicy(to_load_main_params=False, to_load_ema_params=False)

    @property
    def optim_policy(self) -> OptimPolicy:
        """
        Optim policy of the MagiFSDPModule.
        """
        if hasattr(self, "_optim_policy"):
            return self._optim_policy
        else:
            warnings.warn(
                "MagiFSDPModule has no optim_policy. Defaulting not to enable all optim parameters.",
                UserWarning,
            )
            return OptimPolicy(enable_main_param=False, enable_ema_param=False)


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


@contextmanager
def switch_named_main_params_on_modules(
    model: MagiFSDPModule, recurse: bool = True, enable_main_param: bool = False
):
    """
    Context manager that temporarily replaces module parameters attribute with their
    corresponding sharded main parameters for named parameter access. This is primarily
    used so that DCP can correctly remap optimizer parameter IDs to model parameter FQNs.
    """
    if enable_main_param:
        assert isinstance(model, MagiFSDPModule)
        param_to_orig_attr = {}
        fsdp_param_groups = model.collect_fsdp_param_groups(recurse)
        for param_group in fsdp_param_groups:
            for fsdp_param in param_group.fsdp_params:
                if fsdp_param.sharded_main_param is not None:
                    param_to_orig_attr[fsdp_param] = getattr(
                        fsdp_param._module_info.module,
                        fsdp_param._module_info.param_name,
                    )
                    fsdp_param._setattr_on_modules(fsdp_param.sharded_main_param)
    yield

    if enable_main_param:
        for fsdp_param, orig_param in param_to_orig_attr.items():
            fsdp_param._setattr_on_modules(orig_param)
        param_to_orig_attr.clear()
        del param_to_orig_attr
