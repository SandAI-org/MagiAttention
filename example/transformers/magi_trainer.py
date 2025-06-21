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

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import functools
import inspect
import logging
import os
from typing import Any, Union

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, Trainer
from transformers.training_args import OptimizerNames
from transformers.utils import check_min_version, is_accelerate_available
from transformers.utils.versions import require_version
from typing_extensions import override

from magi_attention.api import (
    get_most_recent_key,
    get_position_ids,
    magi_attn_varlen_dispatch,
    undispatch,
)
from magi_attention.api.functools import (
    compute_pad_size,
    full_attention_to_varlen_attention,
    squash_batch_dim,
)
from magi_attention.config import DistAttnConfig

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.51.0")

require_version(
    "datasets>=2.14.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
if is_accelerate_available():
    from accelerate.utils import DistributedType

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class MagiAccelerator(Accelerator):
    @override
    def _prepare_device_mesh(self):
        """
        Prepare the device mesh for distributed training. The dataloader will determine how to load data based on the
        device mesh.
        """
        cp_size = int(os.environ.get("cp_size", 1))

        if self.state.torch_tp_plugin:
            return self.state.torch_tp_plugin.torch_device_mesh
        elif self.distributed_type == DistributedType.DEEPSPEED and hasattr(
            self.state, "ds_device_mesh"
        ):
            return self.state.ds_device_mesh
        elif cp_size > 1:
            device_mesh = torch.arange(0, torch.distributed.get_world_size()).reshape(
                torch.distributed.get_world_size() // cp_size,  # dp_size
                cp_size,
            )

            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=device_mesh,
                mesh_dim_names=(
                    "dp",
                    "tp",
                ),  # hack tp with cp here, set dp-tp 2-dim parallel
            )

            return device_mesh

        return None


class MagiTrainer(Trainer):
    @override
    def create_accelerator_and_postprocess(self):
        # We explicitly don't rely on the `Accelerator` to do gradient accumulation
        grad_acc_kwargs = {}
        if (
            is_accelerate_available("0.28.0")
            and self.args.accelerator_config.gradient_accumulation_kwargs is not None
        ):
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        # check if num_steps is attempted to be passed in gradient_accumulation_kwargs
        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                # raise because we do not know which setting is intended.
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` "
                    "is greater than 1 in the passed `TrainingArguments`"
                    "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments`"
                    " `gradient_accumulation_steps`."
                )
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]

        accelerator_config = self.args.accelerator_config.to_dict()

        if is_accelerate_available("0.28.0"):
            # Extract dataloader config params from accelerator config
            dataloader_params = [
                "split_batches",
                "dispatch_batches",
                "even_batches",
                "use_seedable_sampler",
            ]
            dataloader_config = DataLoaderConfiguration(
                **{param: accelerator_config.pop(param) for param in dataloader_params}
            )
            if is_accelerate_available("1.1.0"):
                dataloader_config.data_seed = self.args.data_seed

        non_blocking = accelerator_config.pop("non_blocking")
        if not is_accelerate_available("0.30.0"):
            if non_blocking:
                raise ImportError(
                    "`non_blocking` is only supported in accelerate v0.30.0 and above. "
                    "Please upgrade accelerate to use this feature."
                )
        else:
            if non_blocking and not self.args.dataloader_pin_memory:
                logger.warning(
                    "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, "
                    "it's recommended to enable both."
                )
            dataloader_config.non_blocking = non_blocking
        # this would have been updated above, no need for it anymore
        accelerator_config.pop("gradient_accumulation_kwargs")

        args = {
            "deepspeed_plugin": self.args.deepspeed_plugin,
        }
        if is_accelerate_available("0.28.0"):
            args["dataloader_config"] = dataloader_config
        else:
            args.update(accelerator_config)
        # tp is initialized at Accelerator init phase so
        # args should be prepared here
        # ignore tp here.
        """
        if self.args.tp_size > 1:
            self.is_tp_enabled = True
            if version.parse(accelerate_version) > version.parse("1.3.0"):
                args["torch_tp_plugin"] = TorchTensorParallelPlugin(
                    tp_size=self.args.tp_size
                )
            else:
                raise ValueError("Requires accelerate>1.3.0 to use Tensor Parallelism.")
        """
        # create accelerator object
        self.accelerator = MagiAccelerator(**args)
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        if (
            "use_gather_object"
            in inspect.signature(self.gather_function).parameters.keys()
        ):
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self.is_fsdp_enabled = (
            getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        )
        self.is_tp_enabled = (
            getattr(self.accelerator.state, "torch_tp_plugin", None) is not None
        )
        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            for param in ["limit_all_gathers", "activation_checkpointing"]:
                setattr(
                    fsdp_plugin,
                    param,
                    self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)),
                )
            if (
                fsdp_plugin.activation_checkpointing
                and self.args.gradient_checkpointing
            ):
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

        if (
            self.is_deepspeed_enabled
            and getattr(self.args, "hf_deepspeed_config", None) is None
        ):
            self.propagate_args_to_deepspeed()

        # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(
                f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`."
            )

        # `auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3

        if (
            self.is_deepspeed_enabled
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            and self.args.auto_find_batch_size
        ):
            raise ValueError(
                "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3."
                "Please consider using Zero-2, Zero-1, or FSDP"
            )
        if (
            self.args.save_only_model
            and self.is_fsdp_enabled
            and "SHARDED_STATE_DICT"
            in str(self.accelerator.state.fsdp_plugin.state_dict_type)
        ):
            raise ValueError(
                "save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'"
            )

    def _prepare_magi_data(self, inputs, head_dim):
        seqlen = inputs.size(1)
        batch_size = inputs.size(0)
        local_input = squash_batch_dim(inputs)
        cp_size = int(os.environ.get("cp_size", 1))
        pad_size, _ = compute_pad_size(local_input.size(0), cp_size, head_dim)
        cu_seqlens_q, cu_seqlens_k = full_attention_to_varlen_attention(
            batch_size, seqlen
        )
        local_input = local_input.unsqueeze(0)

        return local_input, cu_seqlens_q, cu_seqlens_k, pad_size

    def _build_cp_group(self):
        # cp_group do not change during training step.
        if hasattr(self, "cp_group"):
            return self.cp_group

        cp_size = int(os.environ.get("cp_size", 1))
        device_mesh = torch.arange(0, torch.distributed.get_world_size()).reshape(
            torch.distributed.get_world_size() // cp_size,  # dp_size
            cp_size,
        )

        device_mesh = DeviceMesh(
            device_type="cuda",
            mesh=device_mesh,
            mesh_dim_names=("dp", "cp"),  # set dp-cp 2-dim parallel
        )

        cp_group = device_mesh.get_group("cp")
        self.cp_group = cp_group

        return cp_group

    def _prepare_magi_attention(
        self, inputs, cu_seqlens_q, cu_seqlens_k, pad_size, head_dim
    ):
        # ---   magi_attn_flex_dispatch   --- #
        dist_attn_config = DistAttnConfig()
        cp_group = self._build_cp_group()

        inputs = squash_batch_dim(inputs)

        x_padded, dist_attn_runtime_key = magi_attn_varlen_dispatch(
            inputs,
            cu_seqlens_q,
            cu_seqlens_k,
            head_dim=head_dim,
            pad_size=pad_size,
            cp_group=cp_group,
            causal=True,
            dist_attn_config=dist_attn_config,
        )
        x_padded = x_padded.unsqueeze(0)

        return x_padded, dist_attn_runtime_key

    @override
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)
        logits = outputs.logits

        magi_attn_key = get_most_recent_key()
        if magi_attn_key is not None:
            logits = squash_batch_dim(logits)

            logits = undispatch(logits, magi_attn_key)
            logits = logits.unsqueeze(0)

        loss = self.model.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.model.config.vocab_size,
            **loss_kwargs,
        )

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    # TODO: should override here, but mypy fail.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        local_input, cu_seqlens_q, cu_seqlens_k, pad_size = self._prepare_magi_data(
            inputs["input_ids"], self.model.config.head_dim
        )

        local_input, magi_attn_key = self._prepare_magi_attention(
            local_input,
            cu_seqlens_q,
            cu_seqlens_k,
            pad_size,
            self.model.config.head_dim,
        )

        position_ids = get_position_ids(magi_attn_key).unsqueeze(0)

        inputs["position_ids"] = position_ids
        inputs["input_ids"] = local_input

        return inputs

    # TODO: should override here, but mypy fail.
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Finally we need to normalize the loss for reporting
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        cp_size = int(os.environ.get("cp_size", 1))
        backward_loss = loss * cp_size
        self.accelerator.backward(backward_loss, **kwargs)

        return loss.detach()
