# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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

from dataclasses import dataclass

import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import MT_MAP, flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import (
    GQA_HD128,
    KERNEL_PATHS,
    MHA_HD64,
    MHA_HD128,
    MQA_HD128,
    AttnTensors,
    CuteDSLDistTestBase,
    HeadConfig,
    KernelPath,
    assert_close_to_torch_ref,
    make_qkv,
    ranges_from_lengths,
    seed_test_case,
    to_ranges_tensor,
    use_kernel_path,
)
from magi_attention.testing.dist_common import with_run_in_mp


@dataclass(frozen=True)
class DenseConfig:
    name: str
    batch_size: int
    seqlen_q: int
    seqlen_k: int
    heads: HeadConfig
    dtype: torch.dtype
    mask_type: int


DENSE_CONFIGS = [
    DenseConfig("mha_self", 2, 1024, 1024, MHA_HD64, torch.bfloat16, MT_MAP.full),
    DenseConfig("mha_cross", 2, 203, 123, MHA_HD128, torch.bfloat16, MT_MAP.causal),
    DenseConfig("gqa_cross", 2, 203, 123, GQA_HD128, torch.float16, MT_MAP.causal),
    DenseConfig("mqa_self", 2, 256, 256, MQA_HD128, torch.float16, MT_MAP.full),
]


@dataclass(frozen=True)
class VarlenConfig:
    name: str
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    heads: HeadConfig
    dtype: torch.dtype
    mask_type: int


VARLEN_CONFIGS = [
    VarlenConfig(
        "single_q_tile",
        (128,) * 4,
        (128,) * 4,
        MHA_HD64,
        torch.bfloat16,
        MT_MAP.full,
    ),
    VarlenConfig(
        "gqa_uneven_cross",
        (128, 384, 64, 448),
        (256, 128, 512, 192),
        GQA_HD128,
        torch.bfloat16,
        MT_MAP.causal,
    ),
    VarlenConfig(
        "mqa_self",
        (512, 1024),
        (512, 1024),
        MQA_HD128,
        torch.float16,
        MT_MAP.full,
    ),
]


def _to_thd(t: torch.Tensor) -> torch.Tensor:
    return rearrange(t, "b s h d -> (b s) h d")


class TestFlexFlashAttn(CuteDSLDistTestBase):
    @with_run_in_mp
    @parameterize("seq_config", DENSE_CONFIGS)
    @parameterize("kernel_path", KERNEL_PATHS)
    def test_dense_fwd_bwd(
        self,
        seq_config: DenseConfig,
        kernel_path: KernelPath,
    ):
        """Dense (batch, seqlen, heads, dim) inputs."""
        heads, dtype, mask_type = (
            seq_config.heads,
            seq_config.dtype,
            seq_config.mask_type,
        )
        test_case = (
            f"[test_dense_fwd_bwd][{seq_config.name}][{heads.name}]"
            f"[{mask_type=}][{dtype=}][{kernel_path.value}]"
        )
        seed_test_case(test_case)
        batch_size = seq_config.batch_size
        q, k, v = make_qkv(
            (batch_size, seq_config.seqlen_q),
            (batch_size, seq_config.seqlen_k),
            heads,
            dtype,
            self.device,
        )

        with use_kernel_path(kernel_path):
            out, _ = flex_flash_attn_func(q, k, v, mask_types=mask_type)
            do = torch.randn_like(out)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)

        # One relation per batch element over the flattened tokens.
        assert_close_to_torch_ref(
            q=_to_thd(q.detach()),
            k=_to_thd(k.detach()),
            v=_to_thd(v.detach()),
            do=_to_thd(do),
            actual=AttnTensors(
                out=_to_thd(out), dq=_to_thd(dq), dk=_to_thd(dk), dv=_to_thd(dv)
            ),
            q_ranges=ranges_from_lengths([seq_config.seqlen_q] * batch_size),
            k_ranges=ranges_from_lengths([seq_config.seqlen_k] * batch_size),
            attn_type_map=[mask_type] * batch_size,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("seq_config", VARLEN_CONFIGS)
    @parameterize("kernel_path", KERNEL_PATHS)
    def test_varlen_fwd_bwd(
        self,
        seq_config: VarlenConfig,
        kernel_path: KernelPath,
    ):
        """Consecutive, disjoint q/k ranges partition packed tokens by sequence."""
        heads, dtype, mask_type = (
            seq_config.heads,
            seq_config.dtype,
            seq_config.mask_type,
        )
        test_case = (
            f"[test_varlen_fwd_bwd][{seq_config.name}][{heads.name}]"
            f"[{mask_type=}][{dtype=}][{kernel_path.value}]"
        )
        seed_test_case(test_case)
        q_ranges = ranges_from_lengths(seq_config.q_lengths)
        k_ranges = ranges_from_lengths(seq_config.k_lengths)
        q, k, v = make_qkv(
            (sum(seq_config.q_lengths),),
            (sum(seq_config.k_lengths),),
            heads,
            dtype,
            self.device,
        )

        with use_kernel_path(kernel_path):
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=to_ranges_tensor(q_ranges, self.device),
                k_ranges=to_ranges_tensor(k_ranges, self.device),
                mask_types=mask_type,
                max_seqlen_q=max(seq_config.q_lengths),
                max_seqlen_k=max(seq_config.k_lengths),
            )
            do = torch.randn_like(out)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)

        assert_close_to_torch_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=[mask_type] * len(q_ranges),
            test_case=test_case,
        )


if __name__ == "__main__":
    run_tests()
