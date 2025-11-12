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
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    TransformerBlock,
)

import magi_fsdp
from magi_fsdp import fully_shard
from magi_fsdp._fsdp_api import CPUOffloadPolicy
from magi_fsdp._fully_shard import register_fsdp_forward_method
from magi_fsdp.testing import extract_mismatch_threshold, parameterize
from magi_fsdp.testing.common_fsdp import FSDPTest


class VisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14)

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.patch_proj(imgs).flatten(2).transpose(1, 2)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(imgs).sum(dim=1)


class VisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit, self.projector = VisionTransformer(), nn.Linear(1024, 256)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # Run `vit.forward_features`, which is not `forward`!
        patch_embeddings = self.vit.forward_features(imgs)
        return self.projector(patch_embeddings)


class Transformer(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
        layer_nums: int = 5,
    ):
        super().__init__()
        self.layer_nums = layer_nums
        self.model_args = model_args
        self.transformer_block = nn.ModuleList(
            [TransformerBlock(self.model_args) for _ in range(layer_nums)]
        )

    def forward(self, x: torch.Tensor):
        for block in self.transformer_block:
            x = block(x)
        return x


# TODO: For activation offloading, FSDP only synchronizes between the default stream and the offload stream.
# TODO: If the model uses activations on other streams, it must handle synchronization with the default stream manually.
# TODO: Add a unit test to cover this scenario.
class TestFullyShardCPUOffload(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 4)

    @skip_if_lt_x_gpu(4)
    @parameterize("foreach_offload", [True, False])
    @parameterize("pin_memory", [True, False])
    def test_register_fsdp_forward_method(
        self,
        foreach_offload: bool,
        pin_memory: bool,
    ):
        """Based on https://github.com/pytorch/pytorch/issues/109385"""

        EPSILON = 1e-8
        grad_atol = EPSILON
        grad_rtol = 0.05

        mismatch_thres_ratio: float = 2.0

        testcase = f"{foreach_offload=} x {pin_memory=}"
        torch.manual_seed(42)
        model = VisionModel()
        ref_model = copy.deepcopy(model).cuda()
        fully_shard(
            model.vit,
            offload_policy=CPUOffloadPolicy(
                foreach_offload=foreach_offload, pin_memory=pin_memory
            ),
        )
        fully_shard(
            model.projector,
            offload_policy=CPUOffloadPolicy(
                foreach_offload=foreach_offload, pin_memory=pin_memory
            ),
        )
        fully_shard(
            model,
            offload_policy=CPUOffloadPolicy(
                foreach_offload=foreach_offload, pin_memory=pin_memory
            ),
        )
        register_fsdp_forward_method(model.vit, "forward_features")

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn(4, 3, 224, 224, device="cuda")
        for _ in range(3):
            ref_loss = ref_model(inp).sum()
            loss = model(inp).sum()
            self.assertEqual(ref_loss, loss)
            ref_loss.backward()
            loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            self.check_sharded_parity(
                ref_model, model, testcase, grad_atol, grad_rtol, mismatch_thres_ratio
            )

    @skip_if_lt_x_gpu(4)
    @parameterize(
        "layer_nums",
        [2, 4, 6],
    )
    @parameterize(
        "dim",
        [128, 256, 512],
    )
    @parameterize("seqlen", [2048, 4096])
    @parameterize("foreach_offload", [True, False])
    @parameterize("pin_memory", [True, False])
    @parameterize("random_offload", [False])
    def test_fsdp_activation_cpu_offload_policy(
        self,
        layer_nums: int,
        dim: int,
        seqlen: int,
        foreach_offload: bool,
        pin_memory: bool,
        random_offload: bool,
    ):
        """Based on https://github.com/pytorch/pytorch/issues/109385"""

        EPSILON = 1e-8

        o_atol = EPSILON
        o_rtol = 0.05

        grad_atol = EPSILON
        grad_rtol = 0.05

        mismatch_thres_ratio: float = 2.0

        model_args = ModelArgs(
            n_layers=layer_nums,
            n_heads=32,
            dim=dim,
            vocab_size=4096,
            max_seq_len=seqlen,
            dropout_p=0,
            use_attn_mask=False,
            weight_tying=False,
            checkpoint_activations=False,
        )

        testcase = (
            f"{layer_nums=} x {dim=} x {seqlen=} x {foreach_offload=} x {pin_memory=}"
        )

        torch.manual_seed(42)
        model = Transformer(model_args=model_args, layer_nums=layer_nums)
        ref_model = copy.deepcopy(model).cuda()
        for i in range(model.layer_nums):
            if i != model.layer_nums - 1:
                fully_shard(
                    model.transformer_block[i],
                    offload_policy=CPUOffloadPolicy(
                        foreach_offload=foreach_offload, pin_memory=pin_memory
                    ),
                )
            else:
                fully_shard(
                    model.transformer_block[i],
                    offload_policy=CPUOffloadPolicy(
                        foreach_offload=foreach_offload, pin_memory=pin_memory
                    ),
                )
                model.transformer_block[i].set_offload_activation_after_forward(False)
        fully_shard(
            model,
            offload_policy=CPUOffloadPolicy(
                foreach_offload=foreach_offload, pin_memory=pin_memory
            ),
        )

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn(seqlen, 4, dim, device="cuda")

        for _ in range(3):
            ref_out = ref_model(inp)
            out = model(inp)
            ref_loss = ref_out.sum()
            loss = out.sum()
            self.assertEqual(ref_loss, loss)
            ref_loss.backward()
            loss.backward()

            # average param grad of ref model
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            # check grad
            self.check_sharded_parity(
                ref_model, model, testcase, grad_atol, grad_rtol, mismatch_thres_ratio
            )
            self.assert_equal_with_tol(
                ref_out, out, o_atol, o_rtol, mismatch_thres_ratio, testcase, "o"
            )

    def assert_equal_with_tol(
        self,
        ref_tensor: torch.Tensor,
        tensor: torch.Tensor,
        atol: float,
        rtol: float,
        mismatch_thres_ratio: float,
        testcase: str,
        type: str,
    ):
        # torch style with atol + rtol + mismatch threshold
        grad_thres = extract_mismatch_threshold(
            actual=ref_tensor,
            expected=tensor,
            atol=atol,
            rtol=rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )

        magi_fsdp.testing.assert_close(
            ref_tensor,
            tensor,
            atol=atol,
            rtol=rtol,
            mismatch_threshold=grad_thres,
            test_case=f"{testcase} => {type}",
        )

    def check_sharded_parity(
        cls,  # unit test class
        replicated_module: nn.Module,
        sharded_module: nn.Module,
        testcase: str,
        atol: float,
        rtol: float,
        mismatch_thres_ratio: float,
        prefixes_to_ignore: tuple[str, ...] = (),
    ):
        for (replicated_name, replicated_param), (sharded_name, sharded_param) in zip(
            replicated_module.named_parameters(), sharded_module.named_parameters()
        ):
            clean_sharded_name = sharded_name
            for prefix in prefixes_to_ignore:
                clean_sharded_name = clean_sharded_name.replace(prefix, "")
            cls.assertEqual(replicated_name, clean_sharded_name)
            cls.assertIsInstance(sharded_param, DTensor)
            assert isinstance(sharded_param, DTensor)  # mypy
            mesh, placements = sharded_param.device_mesh, sharded_param.placements
            if tuple(placements) == (Shard(0), Shard(0)):
                raise AssertionError(
                    "FSDP's (Shard(0), Shard(0)) layout differs from distribute_tensor(), "
                    "so we cannot check for equality using it"
                )
            sharded_ref_param = distribute_tensor(replicated_param, mesh, placements)
            cls.assertEqual(sharded_param.to_local(), sharded_ref_param.to_local())
            if replicated_param.grad is None:
                cls.assertIsNone(sharded_param.grad)
                continue
            cls.assertIsNotNone(sharded_param.grad)
            sharded_ref_grad = distribute_tensor(
                replicated_param.grad, mesh, placements
            )
            cls.assertIsInstance(sharded_param.grad, DTensor)
            assert isinstance(sharded_param.grad, DTensor)  # mypy

            cls.assert_equal_with_tol(
                sharded_param.grad.to_local(),
                sharded_ref_grad.to_local(),
                atol,
                rtol,
                mismatch_thres_ratio,
                testcase,
                "shared_param_grad",
            )
