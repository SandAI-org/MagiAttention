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
from typing import Any, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed.tensor import init_device_mesh
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import get_cycles_per_ms
from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs

import magi_fsdp
from magi_fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from magi_fsdp._fsdp_module import MagiFSDPModule as FSDPModule
from magi_fsdp.testing import (
    extract_mismatch_threshold,
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
                    1: torch.float32,
                    2: torch.float32,
                    3: torch.float32,
                    4: torch.float32,
                }
            },
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
            {
                "TransformerBlock": {
                    0: torch.float32,
                    1: torch.bfloat16,
                    2: torch.bfloat16,
                    3: torch.bfloat16,
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
        EPSILON = 1e-8

        o_atol = EPSILON
        o_rtol = 0.05

        mismatch_thres_ratio: float = 2.0

        delay_after_forward = delay_flags["delay_after_forward"]
        delay_before_all_gather = delay_flags["delay_before_all_gather"]
        delay_before_reduce_scatter = delay_flags["delay_before_reduce_scatter"]
        delay_before_optim = delay_flags["delay_before_optim"]

        assert device_type in ("cuda", "cpu"), f"{device_type}"
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
        ref_model = copy.deepcopy(model)
        if device_type == "cuda":
            replicate(ref_model.cuda(), device_ids=[self.rank])
        else:
            gloo_pg = dist.new_group(backend="gloo")
            replicate(ref_model, process_group=gloo_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        mesh = init_device_mesh("cuda", (self.world_size,))
        mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
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
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        delay_in_ms = 100
        orig_all_gather = dist.all_gather_into_tensor
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def delayed_all_gather(*args, **kwargs):
            torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
            return orig_all_gather(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(42 + self.rank + 1)
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
        with patch_all_gather_ctx, patch_reduce_scatter_ctx:
            for iter_idx in range(10):
                inp = torch.randint(0, vocab_size, (3, 1024), device=device_type)

                losses: list[torch.Tensor] = []
                outs: list[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                    outs.append(_model(inp))
                    losses.append(outs[-1].sum())
                    if _model is model and delay_after_forward:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    losses[-1].backward()
                    if _model is model and delay_before_optim:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    _optim.step()
                max_out_diff = torch.abs(outs[0] - outs[1]).max()
                loss_diff = torch.abs(losses[0] - losses[1]).max()
                print(
                    f"Step {iter_idx=}, max forward diff: {max_out_diff}, loss diff: {loss_diff}"
                )

                test_case = (
                    f"[{reshard_after_forward=}][{offload_policy=}][{reshard_after_forward=}][{delay_after_forward=}]"
                    f"[{delay_before_all_gather=}][{delay_before_reduce_scatter=}][{delay_before_optim=}]"
                    f"[{unshard_async_op=}][{iter_idx=}]"
                )
                # -----   assert close for fwd out   ---- #

                # torch style with atol + rtol + mismatch threshold
                o_thres = extract_mismatch_threshold(
                    actual=outs[1],
                    expected=outs[0],
                    atol=o_atol,
                    rtol=o_rtol,
                    mismatch_thres_ratio=mismatch_thres_ratio,
                )

                magi_fsdp.testing.assert_close(
                    outs[1],
                    outs[0],
                    atol=o_atol,
                    rtol=o_rtol,
                    mismatch_threshold=o_thres,
                    test_case=f"{test_case} => o",
                )

        torch.cuda.empty_cache()

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
                    1: torch.float32,
                    2: torch.float32,
                    3: torch.float32,
                    4: torch.float32,
                }
            },
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
            {
                "TransformerBlock": {
                    0: torch.float32,
                    1: torch.float16,
                    2: torch.float16,
                    3: torch.float16,
                    4: torch.float32,
                },
                "LayerNorm": torch.float32,
            },
        ],
    )
    def test_train_parity_main_parameters_multi_dtype(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        device_type: str,
        delay_flags: dict[str, bool],
        unshard_async_op: bool,
        multi_dtype: dict[str, Any],
    ):
        """
        Tests train parity against DDP when using multiple dtypes groups with main parameters
        for communication (for communication and computation overlap plus memory
        reduction).
        """
        EPSILON = 1e-8

        o_atol = EPSILON
        o_rtol = 0.05

        # NOTE: an experimental value from magi_fsdp testing
        mismatch_thres_ratio: float = 2.0

        delay_after_forward = delay_flags["delay_after_forward"]
        delay_before_all_gather = delay_flags["delay_before_all_gather"]
        delay_before_reduce_scatter = delay_flags["delay_before_reduce_scatter"]
        delay_before_optim = delay_flags["delay_before_optim"]

        assert device_type in ("cuda", "cpu"), f"{device_type}"
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
        ref_model = copy.deepcopy(model)
        if device_type == "cuda":
            replicate(ref_model.cuda(), device_ids=[self.rank])
        else:
            gloo_pg = dist.new_group(backend="gloo")
            replicate(ref_model, process_group=gloo_pg)

        # ref model main parameters
        ref_main_params = [
            p.detach().clone().float().requires_grad_(True)
            for p in ref_model.parameters()
        ]
        ref_optim = torch.optim.Adam(ref_main_params, lr=1e-2)
        mesh = init_device_mesh("cuda", (self.world_size,))
        mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
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

        # fsdp model main parameters
        main_params = [
            p.detach().clone().float().requires_grad_(True) for p in model.parameters()
        ]
        optim = torch.optim.Adam(main_params, lr=1e-2)

        delay_in_ms = 100
        orig_all_gather = dist.all_gather_into_tensor
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def delayed_all_gather(*args, **kwargs):
            torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
            return orig_all_gather(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(42 + self.rank + 1)
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
        with patch_all_gather_ctx, patch_reduce_scatter_ctx:
            for iter_idx in range(10):
                inp = torch.randint(0, vocab_size, (3, 1024), device=device_type)

                losses: list[torch.Tensor] = []
                outs: list[torch.Tensor] = []
                for _model, _optim, _main_params in (
                    (ref_model, ref_optim, ref_main_params),
                    (model, optim, main_params),
                ):
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                    outs.append(_model(inp))
                    losses.append(outs[-1].sum())
                    if _model is model and delay_after_forward:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    losses[-1].backward()
                    # cast grad to fp32
                    for model_p, master_p in zip(_model.parameters(), _main_params):
                        if model_p.grad is not None:
                            if master_p.grad is None:
                                master_p.grad = model_p.grad.detach().float().clone()
                            else:
                                master_p.grad.copy_(model_p.grad.detach().float())
                    if _model is model and delay_before_optim:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    _optim.step()
                    # copy main params to model params
                    for model_p, master_p in zip(_model.parameters(), _main_params):
                        model_p.data.copy_(master_p.data.to(dtype=model_p.dtype))
                max_out_diff = torch.abs(outs[0] - outs[1]).max()
                loss_diff = torch.abs(losses[0] - losses[1]).max()
                print(
                    f"Step {iter_idx=}, max forward diff: {max_out_diff}, loss diff: {loss_diff}"
                )

                test_case = (
                    f"[{reshard_after_forward=}][{offload_policy=}][{reshard_after_forward=}][{delay_after_forward=}]"
                    f"[{delay_before_all_gather=}][{delay_before_reduce_scatter=}][{delay_before_optim=}]"
                    f"[{unshard_async_op=}][{iter_idx=}]"
                )
                # -----   assert close for fwd out   ---- #

                # torch style with atol + rtol + mismatch threshold
                o_thres = extract_mismatch_threshold(
                    actual=outs[1],
                    expected=outs[0],
                    atol=o_atol,
                    rtol=o_rtol,
                    mismatch_thres_ratio=mismatch_thres_ratio,
                )

                magi_fsdp.testing.assert_close(
                    outs[1],
                    outs[0],
                    atol=o_atol,
                    rtol=o_rtol,
                    mismatch_threshold=o_thres,
                    test_case=f"{test_case} => o",
                )
        torch.cuda.empty_cache()


class TestFullyShardMultiDtypeGradientAccumulation(FSDPTest):
    fsdp_mesh = None
    hsdp_mesh = None

    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
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
    @parameterize("offload_policy", [OffloadPolicy(), CPUOffloadPolicy()])
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
        mode: str,
        reshard_after_backward: bool,
        offload_policy: OffloadPolicy,
        reduce_scatter_only: bool,  # for HSDP
        multi_dtype: dict[str, Any],
    ):
        """
        Tests gradient accumulation with/without gradient reduction and
        with/without resharding after backward.
        """

        EPSILON = 1e-8

        o_atol = EPSILON
        o_rtol = 0.01

        # NOTE: an experimental value from magi_fsdp testing
        mismatch_thres_ratio: float = 2.0

        if not hybrid_mesh:
            if TestFullyShardMultiDtypeGradientAccumulation.fsdp_mesh is None:
                TestFullyShardMultiDtypeGradientAccumulation.fsdp_mesh = (
                    init_device_mesh("cuda", (self.world_size,))
                )
            mesh = TestFullyShardMultiDtypeGradientAccumulation.fsdp_mesh
        else:
            if self.world_size == 4:  # test HSDP too if enough GPUs
                if TestFullyShardMultiDtypeGradientAccumulation.hsdp_mesh is None:
                    TestFullyShardMultiDtypeGradientAccumulation.hsdp_mesh = (
                        init_device_mesh(
                            "cuda",
                            (2, 2),
                            mesh_dim_names=("dp_replicate", "dp_shard"),
                        )
                    )
                mesh = TestFullyShardMultiDtypeGradientAccumulation.hsdp_mesh
            else:
                if TestFullyShardMultiDtypeGradientAccumulation.fsdp_mesh is None:
                    TestFullyShardMultiDtypeGradientAccumulation.fsdp_mesh = (
                        init_device_mesh("cuda", (self.world_size,))
                    )
                mesh = TestFullyShardMultiDtypeGradientAccumulation.fsdp_mesh

        if (
            (
                not reshard_after_backward
                and (reshard_after_forward is not False or mode == "some_layers")
            )
            or (
                isinstance(offload_policy, CPUOffloadPolicy)
                and reshard_after_forward is not True
            )
            or (mesh.ndim != 2 and reduce_scatter_only)
        ):
            return  # skip since not common or applicable

        torch.manual_seed(42)
        batch_size, lin_dim, num_layers, num_microbatches = (2, 32, 6, 3)
        if mode == "some_layers":
            num_layers_to_disable_reduce_scatter = 1
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
        ref_model = copy.deepcopy(model).cuda()
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )
        fully_shard_fn([model[1], model[2], model[3]])
        fully_shard_fn([model[4], model[5], model[6]])
        fully_shard_fn(model)  # root gets the 1st linear
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        def set_grad_sync_flag(
            module: nn.Module, is_last_microbatch: bool, recurse: bool = True
        ):
            if reduce_scatter_only:
                module.set_requires_all_reduce(is_last_microbatch, recurse=recurse)
            else:
                module.set_requires_gradient_sync(is_last_microbatch, recurse=recurse)

        def set_backward_flags(_model: nn.Module, is_last_microbatch: bool):
            if mode == "all":
                set_grad_sync_flag(_model, is_last_microbatch)
                if not reshard_after_backward:
                    _model.set_reshard_after_backward(is_last_microbatch)
            elif mode == "some_layers":
                for layer in model[1 : 1 + 3 * num_layers_to_disable_reduce_scatter]:
                    set_grad_sync_flag(layer, is_last_microbatch)
                    if not reshard_after_backward:
                        layer.set_reshard_after_backward(is_last_microbatch)
            elif mode == "root_only":
                set_grad_sync_flag(model, is_last_microbatch, recurse=False)
                if not reshard_after_backward:
                    model.set_reshard_after_backward(is_last_microbatch, recurse=False)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(5):
            comm_count_list = []

            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                set_backward_flags(model, is_last_microbatch)
                inp = torch.randn(batch_size, lin_dim, device="cuda")
                outs: list[torch.Tensor] = []
                losses: list[torch.Tensor] = []
                for _model in (ref_model, model):
                    with CommDebugMode() as comm_mode:
                        outs.append(_model(inp))
                        losses.append(outs[-1].sum())
                        losses[-1].backward()
                    comm_count_list.append(comm_mode.get_comm_counts())
                out_diff = torch.abs(outs[0] - outs[1]).max()
                loss_diff = torch.abs(losses[0] - losses[1]).max()
                print(
                    f"Step {iter_idx=}, {microbatch_idx=}, out diff: {out_diff}, loss diff: {loss_diff}"
                )

                test_case = (
                    f"[{hybrid_mesh=}][{mode=}][{reshard_after_forward=}][{reshard_after_backward=}]"
                    f"[{offload_policy=}][{reduce_scatter_only=}][{iter_idx=}][{microbatch_idx=}]"
                )
                # -----   assert close for fwd out   ---- #

                # torch style with atol + rtol + mismatch threshold
                o_thres = extract_mismatch_threshold(
                    actual=outs[1],
                    expected=outs[0],
                    atol=o_atol,
                    rtol=o_rtol,
                    mismatch_thres_ratio=mismatch_thres_ratio,
                )

                magi_fsdp.testing.assert_close(
                    outs[1],
                    outs[0],
                    atol=o_atol,
                    rtol=o_rtol,
                    mismatch_threshold=o_thres,
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
                expected_reduce_scatter_count += (
                    2 * ((num_layers // 3) - num_layers_to_disable_reduce_scatter) + 1
                ) * (num_microbatches - 1)
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
            self.assertEqual(all_gather_count, expected_all_gather_count)

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            # check_sharded_parity(self, ref_model, model)
            for _optim in (optim, ref_optim):
                _optim.step()
                # When `set_to_none=False`, we are exercising mixing
                # gradient accumulation with and without communication
                _optim.zero_grad(set_to_none=(iter_idx % 2))

    @skip_if_lt_x_gpu(2)
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
                    1: torch.bfloat16,
                    2: torch.float32,
                    3: torch.bfloat16,
                    4: torch.float32,
                },
                "LayerNorm": torch.float32,
            },
            {
                "TransformerBlock": {
                    0: torch.float32,
                    1: torch.bfloat16,
                    2: torch.bfloat16,
                    3: torch.bfloat16,
                    4: torch.float32,
                },
                "LayerNorm": torch.float32,
            },
            {
                "TransformerBlock": {
                    0: torch.bfloat16,
                    1: torch.bfloat16,
                    2: torch.bfloat16,
                    3: torch.bfloat16,
                    4: torch.float32,
                }
            },
        ],
    )
    def test_1f1b_microbatching_multi_dtype(
        self,
        use_explicit_unshard: bool,
        reshard_after_backward: bool,
        multi_dtype: dict[str, Any],
    ):
        EPSILON = 1e-8

        o_atol = EPSILON
        o_rtol = 0.05

        # NOTE: an experimental value from magi_fsdp testing
        mismatch_thres_ratio: float = 2.0

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

        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        fully_shard([model.layers[0], model.layers[1]], reshard_after_forward=False)
        fully_shard(
            [model.layers[2], model.layers[3], model.layers[4]],
            reshard_after_forward=False,
        )
        fully_shard(model, reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

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
        torch.manual_seed(42 + self.rank + 1)
        for iter_step in range(3):
            outs: list[torch.Tensor] = []
            ref_outs: list[torch.Tensor] = []
            losses: list[torch.Tensor] = []
            ref_losses: list[torch.Tensor] = []
            inps = [
                torch.randint(
                    0, model_args.vocab_size, (local_batch_size, 1024), device="cuda"
                )
                for _ in range(num_microbatches)
            ]
            optim.zero_grad()
            ref_optim.zero_grad()
            for inp_idx, inp in enumerate(inps):
                is_last_microbatch = inp_idx == num_microbatches - 1
                model.set_requires_gradient_sync(is_last_microbatch)
                model.set_is_last_backward(is_last_microbatch)
                if not reshard_after_backward:
                    model.set_reshard_after_backward(is_last_microbatch)
                outs.append(model(inp))
                losses.append(outs[-1].sum())
                losses[-1].backward()
                ref_outs.append(ref_model(inp))
                ref_losses.append(ref_outs[-1].sum())
                ref_losses[-1].backward()

                max_out_diff = torch.abs(outs[-1] - ref_outs[-1]).max()
                loss_diff = torch.abs(losses[-1] - ref_losses[-1]).max()
                print(
                    f"Step {iter_step=}, {inp_idx=}, max forward diff: {max_out_diff}, loss diff: {loss_diff}"
                )

                test_case = f"[{use_explicit_unshard=}][{reshard_after_backward=}][{iter_step=}][{inp_idx=}]"

                # -----   assert close for fwd out   ---- #

                # torch style with atol + rtol + mismatch threshold
                o_thres = extract_mismatch_threshold(
                    actual=outs[-1],
                    expected=ref_outs[-1],
                    atol=o_atol,
                    rtol=o_rtol,
                    mismatch_thres_ratio=mismatch_thres_ratio,
                )

                magi_fsdp.testing.assert_close(
                    outs[-1],
                    ref_outs[-1],
                    atol=o_atol,
                    rtol=o_rtol,
                    mismatch_threshold=o_thres,
                    test_case=f"{test_case} => o",
                )

            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

            optim.step()
            ref_optim.step()

        # TODO
        # check_sharded_parity(self, ref_model, model)
