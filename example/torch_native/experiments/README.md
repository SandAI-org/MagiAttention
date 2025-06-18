# Using MagiAttention with HuggingFace Transformers
We provide you an example to train llama-3-1b model with magiattention utilizing the HuggingFace Transformers ecosystem. We also do experiments to compare the loss of training with/without magiattention to show the correctness of MagiAttention.

## Install Transformers and Accelerate
```shell
pip install transformers==4.51.3
pip install accelerate==1.6.0
pip install datasets==3.5.1
pip install tiktoken==0.9.0
pip install blobfile
pip install evaluate
```

## Prepare model and datasets
We load from [Llama-3-1b](https://huggingface.co/meta-llama/Llama-3.2-1B) model meta provided and continue pretraining it with [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext) datasets.

You can download the same model from modelscope [here](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-1B/).


## Prepare trainer
Transformers provide an example of training language model [here](https://github.com/huggingface/transformers/tree/v4.51.3/examples/pytorch/language-modeling), you can utilize [run_clm.py](https://github.com/huggingface/transformers/blob/v4.51.3/examples/pytorch/language-modeling/run_clm.py) to train casual language model like gpt-2 and llama. You can specify a distributed training strategy, such as DDP or FSDP, and accelerate will automatically handle the underlying logic related to your datasets and the distributed setup.

However, MagiAttention is a context parallel strategy that isn't natively supported by the transformers or accelerate libraries. Therefore, integrating it requires overriding the `transformers.Trainer` and `accelerate.Accelerator` classes.


### Override transformers

Specifically, we need to override `_prepare_inputs` to prepare data and position ids for MagiAttention:
```diff
@override
def _prepare_inputs():
    ...
    if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
+   local_input, cu_seqlens_q, cu_seqlens_k, pad_size = self._prepare_magi_data(
+               inputs["input_ids"], self.model.config.head_dim
+           )

+   local_input, magi_attn_key = self._prepare_magi_attention(
+       local_input,
+       cu_seqlens_q,
+       cu_seqlens_k,
+       pad_size,
+       self.model.config.head_dim,
+   )

+   position_ids = get_position_ids(magi_attn_key).unsqueeze(0)

+   inputs["position_ids"] = position_ids
+   inputs["input_ids"] = local_input

    return inputs
```


We also need to override `compute_loss` because we need to undispatch logits and compute loss:
```diff
def compute_loss():
    ...
    outputs = model(**inputs)
+   logits = outputs.logits

+   magi_attn_key = get_magi_attention_key()
+   if magi_attn_key is not None:
+       logits = squash_batch_dim(logits)

+       logits = undispatch(logits, magi_attn_key)
+       logits = logits.unsqueeze(0)

+   loss = self.model.loss_function(
+       logits=logits,
+       labels=labels,
+       vocab_size=self.model.config.vocab_size,
+       **loss_kwargs,
+   )
    ...

    return (loss, outputs) if return_outputs else loss
```
Override `training_step`: We need to multiply the loss by cp_size before the backward pass because the gradients are divided by an extra factor of cp_size during the all_reduce averaging process.
```diff
def training_step():
    if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
        loss = loss / self.args.gradient_accumulation_steps
+   import os

+   cp_size = int(os.environ.get("cp_size", 1))
+   backward_loss = loss * cp_size
+   self.accelerator.backward(backward_loss, **kwargs)
-   self.accelerator.backward(loss, **kwargs)

    return loss.detach()
```
Override `create_accelerator_and_postprocess` because we want to use the accelerator we overrided:
```diff
def create_accelerator_and_postprocess():
...
-   self.accelerator = Accelerator(**args)
+   self.accelerator = MagiAccelerator(**args)
...
```

### Override accelerate
Override `_prepare_device_mesh` to prepare correct data for cp + dp sceneria.
```python
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
```


### register Magi_Attention implementation
What's more, MagiAttention provides a new type of attention implenmentation(flexible flash attention), so we need to register it for use:
``` python
def Magi_Attention_forward(
     module: nn.Module,
     query: torch.Tensor,  # (b, num_heads, seq_len, hidden_dim)
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    magi_attn_key = get_magi_attention_key()

    dtype = query.dtype
    q, k, v = [
        rearrange(e, "b nh s hd -> (b s) nh hd").to(
            torch.bfloat16
        )  # ffa only supports fp16/bf16 for now
        for e in (query, key, value)
    ]

    o = calc_attn(q, k, v, magi_attn_key)[0]
    o = rearrange(o, "(1 s) nh hd -> 1 s (nh hd)").to(dtype)  # assume batch_size is 1

    return o, None

# register Magi_Attention as attn_backend globally.
ALL_ATTENTION_FUNCTIONS.register("Magi_Attention", Magi_Attention_forward)
```
And change model's attention inplementation:
```diff
...
elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
...
+config._attn_implementation = "Magi_Attention"
...
```

We don't need to make any modifications to modeling_llama.py



## Experiments
### Training Environment
| **Env**                 | **version**                                                                                |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
|  docker             |  ngc25.02-py3  |
|  MagiAttention      |  Tags: v1.0.2
|  transformers       |  Tags: 4.51.3
|  accelerate         |  Tags: 1.6.0

### Training Settings

| **Configuration**                 | **Value**                                                                                |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| **Dataset**                   | [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)                        |
| **Model**                | [LLaMA-3-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)                                                                                     |
| **Number of Layers**          | 16                                                                                           |
| **Hidden Size**               | 2048                                                                                         |
| **Number of Attention Heads** | 32                                                                                           |
| **Group Query Attention**     | Enabled                                                                                      |
| **Number of Query Groups**    | 8                                                                                            |
| **Sequence Length**           | 8192                                                                                         |
| **Parallel Size**     | CP1/2/4/8 (MagiAttention) vs no cp(torch native) with a global batch size of 8        |
| **Training Iterations**       | 2000                                                                                      |


### Results
You can refer to run_magi_clm.sh and run_origin_clm.sh for our experiment commands.


MagiAttention aligns well with torch native training:
![Results](./results.png)
