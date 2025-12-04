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

import contextlib
import copy
import functools
from collections import defaultdict
from functools import lru_cache
from typing import Any, Literal, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed.tensor import init_device_mesh
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import get_cycles_per_ms
from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs

from magi_fsdp import (
    CPUOffloadPolicy,
    MagiFSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
    magi_fsdp_switch_params,
)
from magi_fsdp._fsdp_module import MagiFSDPModule as FSDPModule
from magi_fsdp.testing import (
    assert_close,
    parameterize,
    switch_envvar_decorator,
)
from magi_fsdp.testing.common_fsdp import (
    FeedForward,
    FSDPTest,
    MultiDtypeTransformer,
    compiled_fsdp_test,
    patch_all_gather,
    patch_reduce_scatter,
)

c10d_ops = torch.ops.c10d
funcol = torch.ops.c10d_functional


# Define Some Helper Functions
def get_precision_meta_info():
    # Only consider relative error, not absolute error, so set a very small absolute error tolerance
    EPSILON = 1e-8
    atol = EPSILON
    rtol = 0.05
    # Allow some outliers to not match, so set mismatch threshold
    mismatch_threshold: float = 0.01

    return atol, rtol, mismatch_threshold


def get_model_args():
    return ModelArgs(
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


def get_delayed_collective_context(
    delay_in_ms: int, delay_before_all_gather: bool, delay_before_reduce_scatter: bool
):
    orig_all_gather = dist.all_gather_into_tensor
    orig_reduce_scatter = dist.reduce_scatter_tensor

    def delayed_all_gather(*args, **kwargs):
        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
        return orig_all_gather(*args, **kwargs)

    def delayed_reduce_scatter(*args, **kwargs):
        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
        return orig_reduce_scatter(*args, **kwargs)

    patch_all_gather_ctx = (
        patch_all_gather(delayed_all_gather)
        if delay_before_all_gather
        else contextlib.nullcontext()
    )
    patch_reduce_scatter_ctx = (
        patch_reduce_scatter(delayed_reduce_scatter)
        if delay_before_reduce_scatter
        else contextlib.nullcontext()
    )

    return patch_all_gather_ctx, patch_reduce_scatter_ctx


def set_grad_sync_flag(
    module: nn.Module,
    reduce_scatter_only,
    is_last_microbatch: bool,
    recurse: bool = True,
):
    if reduce_scatter_only:
        module.set_requires_all_reduce(is_last_microbatch, recurse=recurse)
    else:
        module.set_requires_gradient_sync(is_last_microbatch, recurse=recurse)


class TestFullyShardMultiDtypeTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    @switch_envvar_decorator("MAGI_FSDP_MULTI_DTYPE_REDUCE")
    @compiled_fsdp_test(compile_compute_on_module=MultiDtypeTransformer)
    @parameterize("reshard_after_forward", [True, False])
    @parameterize("offload_policy", [OffloadPolicy()])
    @parameterize("device_type", ["cuda"])
    @parameterize(
        "delay_flags",
        [
            {
                "delay_after_forward": False,
                "delay_before_all_gather": False,
                "delay_before_reduce_scatter": False,
                "delay_before_optim": False,
            },
            {
                "delay_after_forward": True,
                "delay_before_all_gather": False,
                "delay_before_reduce_scatter": False,
                "delay_before_optim": False,
            },
            {
                "delay_after_forward": False,
                "delay_before_all_gather": True,
                "delay_before_reduce_scatter": False,
                "delay_before_optim": False,
            },
            {
                "delay_after_forward": False,
                "delay_before_all_gather": True,
                "delay_before_reduce_scatter": True,
                "delay_before_optim": False,
            },
        ],
    )
    @parameterize("unshard_async_op", [False])
    @parameterize(
        "multi_dtype",
        [
            {
                "TransformerBlock": {
                    0: torch.float32,
                    1: torch.float64,
                    2: torch.float32,
                    3: torch.float64,
                    4: torch.float32,
                },
                "LayerNorm": torch.float32,
            },
        ],
    )
    def test_train_parity_multi_dtype(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        device_type: str,
        delay_flags: dict[str, bool],
        unshard_async_op: bool,
        multi_dtype: dict[str, Any],
    ):
        """
        Tests train parity against DDP when using multiple dtypes groups for
        communication (for communication and computation overlap plus memory
        reduction).
        """
        # Guard Clause
        assert device_type in ("cuda", "cpu"), f"{device_type}"

        # Set random seed
        torch.manual_seed(42 + self.rank)

        # Get precision meta info
        atol, rtol, mismatch_threshold = get_precision_meta_info()

        # Set delayed configuration
        delay_after_forward = delay_flags["delay_after_forward"]
        delay_before_all_gather = delay_flags["delay_before_all_gather"]
        delay_before_reduce_scatter = delay_flags["delay_before_reduce_scatter"]
        delay_before_optim = delay_flags["delay_before_optim"]
        delay_in_ms = 100
        patch_all_gather_ctx, patch_reduce_scatter_ctx = get_delayed_collective_context(
            delay_in_ms=delay_in_ms,
            delay_before_all_gather=delay_before_all_gather,
            delay_before_reduce_scatter=delay_before_reduce_scatter,
        )

        # Get model args
        model_args = get_model_args()

        # Initialize model
        with torch.random.fork_rng():
            torch.random.manual_seed(42)
            model = MultiDtypeTransformer(model_args, multi_dtype)

        # Initialize reference model
        ref_model = copy.deepcopy(model)
        # Set up DDP for reference model
        if device_type == "cuda":
            replicate(ref_model.cuda(), device_ids=[self.rank])
        else:
            gloo_pg = dist.new_group(backend="gloo")
            replicate(ref_model, process_group=gloo_pg)

        # Set up FSDP for model
        mesh = init_device_mesh("cuda", (self.world_size,))
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )
        fully_shard_fn([model.layers[0], model.layers[1]])
        fully_shard_fn([model.layers[2], model.layers[3], model.layers[4]])
        fully_shard_fn(model)
        if unshard_async_op:
            model._set_unshard_async_op(unshard_async_op)

        # Initialize optimizer
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # Initialize reference optimizer
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        with patch_all_gather_ctx, patch_reduce_scatter_ctx:
            for iter_idx in range(10):
                # Set test case string
                test_case = (
                    f"[{reshard_after_forward=}][{offload_policy=}][{reshard_after_forward=}][{delay_after_forward=}]"
                    f"[{delay_before_all_gather=}][{delay_before_reduce_scatter=}][{delay_before_optim=}]"
                    f"[{unshard_async_op=}][{iter_idx=}]"
                )

                inp = torch.randint(
                    0, model_args.vocab_size, (3, 1024), device=device_type
                )

                out = model(inp)
                if delay_after_forward:
                    torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                ref_out = ref_model(inp)
                loss = out.mean()
                ref_loss = ref_out.mean()
                loss.backward()
                ref_loss.backward()

                if delay_before_optim:
                    torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                optim.step()
                ref_optim.step()
                optim.zero_grad()
                ref_optim.zero_grad()

                assert_close(
                    out,
                    ref_out,
                    atol=atol,
                    rtol=rtol,
                    mismatch_threshold=mismatch_threshold,
                    test_case=f"{test_case} => o",
                )

        torch.cuda.empty_cache()

    @skip_if_lt_x_gpu(2)
    @switch_envvar_decorator("MAGI_FSDP_MULTI_DTYPE_REDUCE")
    @compiled_fsdp_test(compile_compute_on_module=MultiDtypeTransformer)
    @parameterize("reshard_after_forward", [True])
    @parameterize("offload_policy", [OffloadPolicy()])
    @parameterize("device_type", ["cuda"])
    @parameterize(
        "delay_flags",
        [
            {
                "delay_after_forward": False,
                "delay_before_all_gather": False,
                "delay_before_reduce_scatter": False,
                "delay_before_optim": False,
            },
            {
                "delay_after_forward": False,
                "delay_before_all_gather": True,
                "delay_before_reduce_scatter": True,
                "delay_before_optim": False,
            },
        ],
    )
    @parameterize("unshard_async_op", [False])
    @parameterize(
        "multi_dtype",
        [
            {
                "TransformerBlock": {
                    0: torch.float32,
                    1: torch.float64,
                    2: torch.float64,
                    3: torch.float64,
                    4: torch.float32,
                },
                "LayerNorm": torch.float32,
            }
        ],
    )
    @parameterize("num_microbatches", [1, 3])
    def test_train_parity_fsdp_main_parameters_multi_dtype(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        device_type: str,
        delay_flags: dict[str, bool],
        unshard_async_op: bool,
        multi_dtype: dict[str, Any],
        num_microbatches: int,
    ):
        """
        Tests train parity against DDP when using multiple dtypes groups with main parameters
        for communication (for communication and computation overlap plus memory
        reduction).
        """
        # Guard Clause
        assert device_type in ("cuda", "cpu"), f"{device_type}"

        # Set random seed
        torch.manual_seed(42 + self.rank)

        # Get precision meta info
        atol, rtol, mismatch_threshold = get_precision_meta_info()

        # Set delayed configuration
        delay_after_forward = delay_flags["delay_after_forward"]
        delay_before_all_gather = delay_flags["delay_before_all_gather"]
        delay_before_reduce_scatter = delay_flags["delay_before_reduce_scatter"]
        delay_before_optim = delay_flags["delay_before_optim"]
        delay_in_ms = 100
        patch_all_gather_ctx, patch_reduce_scatter_ctx = get_delayed_collective_context(
            delay_in_ms=delay_in_ms,
            delay_before_all_gather=delay_before_all_gather,
            delay_before_reduce_scatter=delay_before_reduce_scatter,
        )

        # Get model args
        model_args = get_model_args()

        # Initialize model
        with torch.random.fork_rng():
            torch.random.manual_seed(42)
            model = MultiDtypeTransformer(model_args, multi_dtype)

        # Initialize reference model
        ref_model = copy.deepcopy(model)
        # Set up DDP for reference model
        if device_type == "cuda":
            replicate(ref_model.cuda(), device_ids=[self.rank])
        else:
            gloo_pg = dist.new_group(backend="gloo")
            replicate(ref_model, process_group=gloo_pg)

        # Set up FSDP for model
        mp_policy = MixedPrecisionPolicy(main_param_dtype=torch.float64)
        mesh = init_device_mesh("cuda", (self.world_size,))
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
            mp_policy=mp_policy,
        )
        fully_shard_fn([model.layers[0], model.layers[1]])
        fully_shard_fn([model.layers[2], model.layers[3], model.layers[4]])
        fully_shard_fn(model)
        if unshard_async_op:
            model._set_unshard_async_op(unshard_async_op)

        # Initialize optimizer
        with magi_fsdp_switch_params(model, param_type="main", retain_if_missing=True):
            optim = torch.optim.Adam(cast(nn.Module, model).parameters(), lr=1e-2)
        # Initialize reference optimizer
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        with patch_all_gather_ctx, patch_reduce_scatter_ctx:
            for iter_idx in range(10):
                # Set test case string
                test_case = (
                    f"[{reshard_after_forward=}][{offload_policy=}][{reshard_after_forward=}][{delay_after_forward=}]"
                    f"[{delay_before_all_gather=}][{delay_before_reduce_scatter=}][{delay_before_optim=}]"
                    f"[{unshard_async_op=}][{iter_idx=}]"
                )

                inps = [
                    torch.randint(
                        0, model_args.vocab_size, (3, 1024), device=device_type
                    )
                    for _ in range(num_microbatches)
                ]

                for inp_idx, inp in enumerate(inps):
                    is_last_microbatch = inp_idx == num_microbatches - 1
                    model.set_requires_gradient_sync(is_last_microbatch)
                    model.set_is_last_backward(is_last_microbatch)
                    is_first_microbatch = inp_idx == 0
                    model.set_requires_update_param(is_first_microbatch)

                    out = model(inp)
                    if delay_after_forward:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    ref_out = ref_model(inp)
                    loss = out.mean()
                    ref_loss = ref_out.mean()
                    loss.backward()
                    ref_loss.backward()

                    assert_close(
                        out,
                        ref_out,
                        atol=atol,
                        rtol=rtol,
                        mismatch_threshold=mismatch_threshold,
                        test_case=f"{test_case}[{inp_idx=}] => o",
                    )

                if delay_before_optim:
                    torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                optim.step()
                ref_optim.step()
                optim.zero_grad()
                ref_optim.zero_grad()

        torch.cuda.empty_cache()


class TestFullyShardMultiDtypeGradientAccumulation(FSDPTest):
    fsdp_mesh = None
    hsdp_mesh = None

    @property
    def world_size(self) -> int:
        return 4

    @property
    @lru_cache()
    def hybrid_mesh(self):
        return init_device_mesh(
            "cuda",
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )

    @property
    @lru_cache()
    def flat_mesh(self):
        return init_device_mesh("cuda", (self.world_size,))

    @staticmethod
    def set_fine_grained_communication_flags(
        model: nn.Sequential,
        is_first_microbatch: bool,
        is_last_microbatch: bool,
        # Config parameters exposed explicitly
        mode: Literal["all", "root_only", "some_layers"],
        reduce_scatter_only: bool,
        reshard_after_backward: bool,
        some_layer_start_idx: int,
        some_layer_end_idx: int,
    ):
        """
        Configures gradient synchronization and resharding flags for the backward pass.
        """

        def _apply_flags(target_module: MagiFSDPModule, recurse: bool):
            # 1. Handle Gradient Synchronization
            if reduce_scatter_only:
                target_module.set_requires_all_reduce(
                    is_last_microbatch, recurse=recurse
                )
            else:
                target_module.set_requires_gradient_sync(
                    is_last_microbatch, recurse=recurse
                )

            # 2. Handle Resharding (Parameter persistence)
            if not reshard_after_backward:
                target_module.set_reshard_after_backward(
                    is_last_microbatch, recurse=recurse
                )

            # 3. Handle param update from main param
            target_module.set_requires_update_param(
                is_first_microbatch, recurse=recurse
            )

        # Determine targets and recursion strategy
        targets: MagiFSDPModule | nn.Sequential
        should_recurse: bool

        match mode:
            case "root_only":
                targets = [model]
                should_recurse = False

            case "some_layers":
                # Determine slice range (Magic number 3 assumes specific block structure)
                targets = model[some_layer_start_idx:some_layer_end_idx]
                should_recurse = True

            case _:  # Default: "all"
                targets = [model]
                should_recurse = True

        for target in targets:
            _apply_flags(target, recurse=should_recurse)

    @staticmethod
    def get_test_model(
        num_layers, lin_dim, multi_dtype: dict[str, Any]
    ) -> nn.Sequential:
        modules = [nn.Linear(lin_dim, lin_dim)]
        for layer_idx in range(num_layers):
            modules.append(
                FeedForward(
                    lin_dim,
                    4 * lin_dim,
                    0.0,
                    dtype=multi_dtype["FeedForward"].get(layer_idx, torch.float32),  # type: ignore[index]
                )
            )
        model = nn.Sequential(*modules)
        return model

    @skip_if_lt_x_gpu(4)
    @switch_envvar_decorator("MAGI_FSDP_MULTI_DTYPE_REDUCE")
    @parameterize("hybrid_mesh", [False, True])
    @parameterize("reshard_after_forward", [True, False, 2])
    @parameterize(
        "mode",
        [
            "all",  # "all": disable reduce-scatter for all modules
            "root_only",  # "root_only": disable reduce-scatter for root's linear only
            "some_layers",  # "some_layers": disable reduce-scatter for some MultiDtypeTransformerBlocks
        ],
    )
    @parameterize("reshard_after_backward", [False, True])
    @parameterize(
        "offload_policy",
        [
            OffloadPolicy(),
            CPUOffloadPolicy(offload_param=True),
        ],
    )
    # For HSDP only:
    # `True`: reduce-scatter only (no all-reduce) each microbatch
    # until the last microbatch
    # `False`: neither reduce-scatter nor all-reduce each
    # microbatch until the last microbatch
    @parameterize("reduce_scatter_only", [False, True])  # for HSDP
    @parameterize(
        "multi_dtype",
        [
            {
                "FeedForward": {
                    0: torch.float32,
                    1: torch.bfloat16,
                    2: torch.float32,
                    3: torch.bfloat16,
                    4: torch.bfloat16,
                    5: torch.float32,
                }
            },
        ],
    )
    def test_gradient_accumulation_multi_dtype(
        self,
        hybrid_mesh: bool,
        reshard_after_forward: Union[bool, int],
        mode: Literal["all", "root_only", "some_layers"],
        reshard_after_backward: bool,
        offload_policy: OffloadPolicy,
        reduce_scatter_only: bool,  # for HSDP
        multi_dtype: dict[str, Any],
    ):
        """
        Tests gradient accumulation with/without gradient reduction and
        with/without resharding after backward.
        """

        device_mesh = self.hybrid_mesh if hybrid_mesh else self.flat_mesh

        # Guard Clause
        if isinstance(offload_policy, CPUOffloadPolicy) and not reshard_after_forward:
            # If enable CPU offload, must reshard after forward
            return

        if device_mesh.ndim != 2 and reduce_scatter_only:
            # reduce_scatter_only is only for HSDP (2D mesh)
            return

        if not reshard_after_backward and reshard_after_forward is not False:
            # reshard after forward but not after backward is not supported
            return

        # Get precision meta info
        atol, rtol, mismatch_threshold = get_precision_meta_info()

        # Set random seed
        torch.manual_seed(42 + self.rank)

        # Set model hyperparameters
        batch_size, lin_dim, num_layers, num_microbatches = (2, 32, 6, 3)

        # Initialize model
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = self.get_test_model(num_layers, lin_dim, multi_dtype)

        # Initialize reference model
        # NOTE(littsk): ref_model does not use DDP to avoid affecting subsequent communication count statistics
        ref_model = copy.deepcopy(model)
        # Set up DDP for reference model
        replicate(ref_model.cuda(), device_ids=[self.rank])

        # Set up FSDP for model
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=device_mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )
        fully_shard_fn([model[1], model[2], model[3]])
        fully_shard_fn([model[4], model[5], model[6]])
        # root gets the 1st linear
        fully_shard_fn(model)

        # Initialize optimizer
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        for iter_idx in range(5):
            comm_count_list = []

            optim.zero_grad(set_to_none=(iter_idx % 2))
            ref_optim.zero_grad(set_to_none=(iter_idx % 2))
            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1

                self.set_fine_grained_communication_flags(
                    model,
                    True,
                    is_last_microbatch,
                    mode,
                    reduce_scatter_only,
                    reshard_after_backward,
                    1,
                    4,
                )

                inp = torch.randn(batch_size, lin_dim, device="cuda")

                with CommDebugMode() as comm_mode:
                    out = model(inp)
                    loss = out.mean()
                    loss.backward()

                comm_count_list.append(comm_mode.get_comm_counts())

                ref_out = ref_model(inp)
                ref_loss = ref_out.mean()
                ref_loss.backward()

                test_case = (
                    f"[{hybrid_mesh=}][{mode=}][{reshard_after_forward=}][{reshard_after_backward=}]"
                    f"[{offload_policy=}][{reduce_scatter_only=}][{iter_idx=}][{microbatch_idx=}]"
                )

                assert_close(
                    out,
                    ref_out,
                    atol=atol,
                    rtol=rtol,
                    mismatch_threshold=mismatch_threshold,
                    test_case=f"{test_case} => o",
                )

            comm_counts: dict[str, int] = defaultdict(int)

            for comm_count_dict in comm_count_list:
                for collective, count in comm_count_dict.items():
                    comm_counts[collective] += count

            all_gather_count = comm_counts[c10d_ops._allgather_base_]
            reduce_scatter_count = comm_counts[c10d_ops._reduce_scatter_base_]
            all_reduce_count = comm_counts[c10d_ops.allreduce_]

            # Expect two reduce-scatter per FeedForward plus one for the root's linear
            # on the last microbatch
            expected_reduce_scatter_count = 2 * (num_layers // 3) + 1
            if mode == "some_layers":
                # Expect additional reduce-scatters for non-disabled FeedForwards and
                # the root's linear
                expected_reduce_scatter_count += (2 * ((num_layers // 3) - 1) + 1) * (
                    num_microbatches - 1
                )
            elif mode == "root_only":
                # Expect additional reduce-scatters for all FeedForwards
                expected_reduce_scatter_count += (
                    2 * (num_layers // 3) * (num_microbatches - 1)
                )
            expected_all_reduce_count = (
                expected_reduce_scatter_count if device_mesh.ndim == 2 else 0
            )

            if reduce_scatter_only:
                # Specially for HSDP if only reduce-scattering but not
                # all-reducing until the last microbatch, expect one
                # reduce-scatter per MLP plus for the root per microbatch
                expected_reduce_scatter_count = (
                    2 * (num_layers // 3) + 1
                ) * num_microbatches

            self.assertEqual(reduce_scatter_count, expected_reduce_scatter_count)
            self.assertEqual(all_reduce_count, expected_all_reduce_count)

            # Expect one all-gather per FeedForward plus one for the root's linear in
            # the first microbatch's forward
            expected_all_gather_count = num_layers // 3 + 1
            if reshard_after_forward is not False:  # `True` or `2`
                # Add the number of MLPs without the +1 for the backward
                # all-gathers since the root does not reshard after forward
                expected_all_gather_count += num_layers // 3
                # Multiply by the number of microbatches since these
                # all-gathers run every microbatch
                expected_all_gather_count *= num_microbatches
            elif reshard_after_backward:  # `reshard_after_forward=False`
                expected_all_gather_count *= num_microbatches
            elif mode == "all":  # `reshard_after_forward/backward=False`
                # Only reshard parameters after the last microbatch's backward,
                # so there should not be any more all-gathers
                pass
            elif mode == "root_only":  # `reshard_after_forward/backward=False`
                # The MLPs should still contribute all-gathers in each
                # microbatch forward
                expected_all_gather_count += (num_layers // 3) * (num_microbatches - 1)
            elif mode == "some_layers":  # `reshard_after_forward/backward=False`
                # The non-disabled MLPs and root should still contribute all-gathers
                # in each microbatch forward
                expected_all_gather_count += (((num_layers // 3) - 1) + 1) * (
                    num_microbatches - 1
                )
            self.assertEqual(all_gather_count, expected_all_gather_count)

            optim.step()
            ref_optim.step()

    @skip_if_lt_x_gpu(4)
    @switch_envvar_decorator("MAGI_FSDP_MULTI_DTYPE_REDUCE")
    @parameterize("hybrid_mesh", [False, True])
    @parameterize("reshard_after_forward", [True, False, 2])
    @parameterize(
        "mode",
        [
            "all",  # "all": disable reduce-scatter for all modules
            "root_only",  # "root_only": disable reduce-scatter for root's linear only
            "some_layers",  # "some_layers": disable reduce-scatter for some MultiDtypeTransformerBlocks
        ],
    )
    @parameterize("reshard_after_backward", [False, True])
    # For HSDP only:
    # `True`: reduce-scatter only (no all-reduce) each microbatch
    # until the last microbatch
    # `False`: neither reduce-scatter nor all-reduce each
    # microbatch until the last microbatch
    @parameterize("reduce_scatter_only", [False, True])  # for HSDP
    @parameterize(
        "multi_dtype",
        [
            {
                "FeedForward": {
                    0: torch.float64,
                    1: torch.float64,
                    2: torch.float32,
                    3: torch.float64,
                    4: torch.float64,
                    5: torch.float32,
                }
            },
        ],
    )
    def test_gradient_accumulation_fsdp_main_param_multi_dtype(
        self,
        hybrid_mesh: bool,
        reshard_after_forward: Union[bool, int],
        mode: Literal["all", "root_only", "some_layers"],
        reshard_after_backward: bool,
        reduce_scatter_only: bool,  # for HSDP
        multi_dtype: dict[str, Any],
    ):
        """
        Tests gradient accumulation with/without gradient reduction and
        with/without resharding after backward.
        """
        # Guard Clause
        if not reshard_after_backward and reshard_after_forward is not False:
            # reshard after forward but not after backward is not supported
            return

        # Set random seed
        torch.manual_seed(42 + self.rank)

        # Get precision meta info
        atol, rtol, mismatch_threshold = get_precision_meta_info()

        # Initialize mesh
        mesh = self.hybrid_mesh if hybrid_mesh else self.flat_mesh

        # Initialize model hyperparameters
        batch_size, lin_dim, num_layers, num_microbatches = (2, 32, 6, 3)

        # Initialize model
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = self.get_test_model(num_layers, lin_dim, multi_dtype)

        # Initialize reference model
        ref_model = copy.deepcopy(model)
        # Set up DDP for reference model
        replicate(ref_model.cuda(), device_ids=[self.rank])

        # Set up FSDP for model
        mp_policy = MixedPrecisionPolicy(main_param_dtype=torch.float64)
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
        fully_shard_fn([model[1], model[2], model[3]])
        fully_shard_fn([model[4], model[5], model[6]])
        fully_shard_fn(model)  # root gets the 1st linear

        with magi_fsdp_switch_params(model, param_type="main", retain_if_missing=True):
            optim = torch.optim.Adam(cast(nn.Module, model).parameters(), lr=1e-2)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        for iter_idx in range(5):
            comm_count_list = []

            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                is_first_microbatch = microbatch_idx == 0
                self.set_fine_grained_communication_flags(
                    model,
                    is_first_microbatch,
                    is_last_microbatch,
                    mode,
                    reduce_scatter_only,
                    reshard_after_backward,
                    1,
                    4,
                )

                inp = torch.randn(batch_size, lin_dim, device="cuda")

                with CommDebugMode() as comm_mode:
                    out = model(inp)
                    loss = out.mean()
                    loss.backward()

                comm_count_list.append(comm_mode.get_comm_counts())

                ref_out = ref_model(inp)
                ref_loss = ref_out.mean()
                ref_loss.backward()

                test_case = (
                    f"[{hybrid_mesh=}][{mode=}][{reshard_after_forward=}][{reshard_after_backward=}]"
                    f"[{reduce_scatter_only=}][{iter_idx=}][{microbatch_idx=}]"
                )

                # -----   assert close for fwd out   ---- #
                assert_close(
                    out,
                    ref_out,
                    atol=atol,
                    rtol=rtol,
                    mismatch_threshold=mismatch_threshold,
                    test_case=f"{test_case} => o",
                )

            comm_counts: dict[str, int] = defaultdict(int)
            for comm_count_dict in comm_count_list:
                for collective, count in comm_count_dict.items():
                    comm_counts[collective] += count

            all_gather_count = comm_counts[c10d_ops._allgather_base_]
            reduce_scatter_count = comm_counts[c10d_ops._reduce_scatter_base_]
            all_reduce_count = comm_counts[c10d_ops.allreduce_]

            # Expect two reduce-scatter per FeedForward plus one for the root's linear
            # on the last microbatch
            expected_reduce_scatter_count = 2 * (num_layers // 3) + 1
            if mode == "some_layers":
                # Expect additional reduce-scatters for non-disabled FeedForwards and
                # the root's linear
                expected_reduce_scatter_count += (2 * ((num_layers // 3) - 1) + 1) * (
                    num_microbatches - 1
                )
            elif mode == "root_only":
                # Expect additional reduce-scatters for all FeedForwards
                expected_reduce_scatter_count += (
                    2 * (num_layers // 3) * (num_microbatches - 1)
                )
            expected_all_reduce_count = (
                expected_reduce_scatter_count if mesh.ndim == 2 else 0
            )
            if reduce_scatter_only:
                # Specially for HSDP if only reduce-scattering but not
                # all-reducing until the last microbatch, expect one
                # reduce-scatter per MLP plus for the root per microbatch
                expected_reduce_scatter_count = (
                    2 * (num_layers // 3) + 1
                ) * num_microbatches
            self.assertEqual(reduce_scatter_count, expected_reduce_scatter_count)
            self.assertEqual(all_reduce_count, expected_all_reduce_count)

            # Expect one all-gather per FeedForward plus one for the root's linear in
            # the first microbatch's forward
            expected_all_gather_count = num_layers // 3 + 1
            if reshard_after_forward is not False:  # `True` or `2`
                # Add the number of MLPs without the +1 for the backward
                # all-gathers since the root does not reshard after forward
                expected_all_gather_count += num_layers // 3
                # Multiply by the number of microbatches since these
                # all-gathers run every microbatch
                expected_all_gather_count *= num_microbatches
            elif reshard_after_backward:  # `reshard_after_forward=False`
                expected_all_gather_count *= num_microbatches
            elif mode == "all":  # `reshard_after_forward/backward=False`
                # Only reshard parameters after the last microbatch's backward,
                # so there should not be any more all-gathers
                pass
            elif mode == "root_only":  # `reshard_after_forward/backward=False`
                # The MLPs should still contribute all-gathers in each
                # microbatch forward
                expected_all_gather_count += (num_layers // 3) * (num_microbatches - 1)
            elif mode == "some_layers":  # `reshard_after_forward/backward=False`
                # The non-disabled MLPs and root should still contribute all-gathers
                # in each microbatch forward
                expected_all_gather_count += (((num_layers // 3) - 1) + 1) * (
                    num_microbatches - 1
                )
            self.assertEqual(all_gather_count, expected_all_gather_count)

            optim.step()
            ref_optim.step()
            optim.zero_grad()
            ref_optim.zero_grad()

    @skip_if_lt_x_gpu(4)
    @switch_envvar_decorator("MAGI_FSDP_MULTI_DTYPE_REDUCE")
    @parameterize("use_explicit_unshard", [False, True])
    @parameterize("reshard_after_backward", [False, True])
    @parameterize(
        "multi_dtype",
        [
            {
                "TransformerBlock": {
                    0: torch.float32,
                    1: torch.float32,
                    2: torch.float32,
                    3: torch.float32,
                    4: torch.float32,
                }
            },
            {
                "TransformerBlock": {
                    0: torch.float32,
                    1: torch.float64,
                    2: torch.float32,
                    3: torch.float64,
                    4: torch.float32,
                },
                "LayerNorm": torch.float32,
            },
        ],
    )
    def test_1f1b_microbatching_multi_dtype(
        self,
        use_explicit_unshard: bool,
        reshard_after_backward: bool,
        multi_dtype: dict[str, Any],
    ):
        # Get precision meta information
        atol, rtol, mismatch_thresh_ratio = get_precision_meta_info()

        # Set random seed
        torch.manual_seed(42 + self.rank)

        # Initialize model
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model_args = get_model_args()
            model = MultiDtypeTransformer(model_args, multi_dtype)

        # Initialize reference model
        ref_model = copy.deepcopy(model)
        # Set up DDP for reference model
        replicate(ref_model.cuda(), device_ids=[self.rank])

        # Set up FSDP for model
        fully_shard([model.layers[0], model.layers[1]], reshard_after_forward=False)
        fully_shard(
            [model.layers[2], model.layers[3], model.layers[4]],
            reshard_after_forward=False,
        )
        fully_shard(model, reshard_after_forward=False)

        # Initialize optimizer
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)

        num_microbatches = 3
        local_batch_size = 2

        # Before pipelining, we may prefer to issue all all-gathers ahead of
        # time to increase overlap opportunity at no difference in parameter
        # memory usage since we do not reshard after forward
        if use_explicit_unshard:
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.unshard(async_op=True)

        # Emulate the 1f1b pipeline schedule and only reduce gradients on the
        # last microbatch
        for iter_idx in range(3):
            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                model.set_requires_gradient_sync(is_last_microbatch)
                model.set_is_last_backward(is_last_microbatch)
                if not reshard_after_backward:
                    model.set_reshard_after_backward(is_last_microbatch)

                inp = torch.randint(
                    0, model_args.vocab_size, (local_batch_size, 1024), device="cuda"
                )

                out = model(inp)
                loss = out.mean()
                loss.backward()
                ref_out = ref_model(inp)
                ref_loss = ref_out.mean()
                ref_loss.backward()

                test_case = f"[{use_explicit_unshard=}][{reshard_after_backward=}][{iter_idx=}][{microbatch_idx=}]"

                # -----   assert close for fwd out   ---- #
                assert_close(
                    out,
                    ref_out,
                    atol=atol,
                    rtol=rtol,
                    mismatch_threshold=mismatch_thresh_ratio,
                    test_case=f"{test_case} => o",
                )

            optim.step()
            ref_optim.step()
            optim.zero_grad()
            ref_optim.zero_grad()
