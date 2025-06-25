# Using MagiAttention with HuggingFace Transformers

We provide an example for training a Llama-3 1B model with MagiAttention (using a DP+CP approach) based on the Hugging Face [transformers](https://github.com/huggingface/transformers) library.

To verify its correctness, we include experiments that compare the training loss of the model with MagiAttention against a standard baseline

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

We load from [Llama-3-1b](https://huggingface.co/meta-llama/Llama-3.2-1B) model and continue training it with [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext) datasets.

You can also download the model from [modelscope](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-1B/).


## Prepare Trainer

The HuggingFace Transformers library provides the [run_clm.py](https://github.com/huggingface/transformers/blob/v4.51.3/examples/pytorch/language-modeling/run_clm.py) script for training causal language models (CLMs) like gpt and llama. This script integrates seamlessly with accelerate, allowing you to specify a distributed training strategy (such as DDP or FSDP) while accelerate automatically handles the underlying data distribution and process setup. As a reference, we have included `run_origin_clm.py` and `run_origin_clm.sh` in this directory, which mirror the official implementation for training causal language models.

However, MagiAttention is a context parallel strategy that isn't natively supported by the transformers and accelerate libraries. Therefore, integrating it requires customizing the `transformers.Trainer` and `accelerate.Accelerator` classes. We provide `run_magi_clm.py` and `run_magi_clm.sh` in this dir to train llama-3-1b model with MagiAttention.

### Customize accelerate Accelerator

The following code are all available at `magi_trainer.py`.

We need to prepare a custom Accelerator called MagiAccelerator inheriting from the `accelerate.Accelerator` class:
```python
class MagiAccelerator(Accelerator):
    ...
```

Override `_prepare_device_mesh` to prepare correct data for dp + cp sceneria:

**NOTE:** We are implementing Context Parallelism (CP) by treating it as Tensor Parallelism (TP) at the data loading stage. This allows us to leverage accelerate's built-in data loader for DP+TP scenarios, which provides the exact data distribution we need: ranks within the same TP group receive identical data, while different TP groups receive sharded data.
```diff
def _prepare_device_mesh(self):
    """
    Prepare the device mesh for distributed training. The dataloader will determine how to load data based on thedevice mesh.
    """
+   cp_size = int(os.environ.get("CP_SIZE", 1))

    if self.state.torch_tp_plugin:
        return self.state.torch_tp_plugin.torch_device_mesh
    elif self.distributed_type == DistributedType.DEEPSPEED and hasattr(
        self.state, "ds_device_mesh"
    ):
        return self.state.ds_device_mesh
+   elif cp_size > 1:
+       device_mesh = torch.arange(0, torch.distributed.get_world_size()).reshape(
+           torch.distributed.get_world_size() // cp_size,  # dp_size
+           cp_size,
+       )
+
+       device_mesh = DeviceMesh(
+           device_type="cuda",
+           mesh=device_mesh,
+           mesh_dim_names=(
+               "dp",
+               "tp",
+           ),  # hack tp as cp here, set dp-tp 2-dim parallel
+       )
+
+       return device_mesh

    return None
```

### Customize Transformers Trainer

The following code are all available at `magi_trainer.py`.

We need to prepare a custom Trainer called `MagiTrainer` inheriting from the `transformers.Trainer` class:
```python
class MagiTrainer(Trainer):
    ...
```

Override `_prepare_inputs` to prepare data and position_ids for MagiAttention:
```diff
@override
def _prepare_inputs():
    ...
+   local_input, cu_seqlens_q, cu_seqlens_k, pad_size = self._prepare_magi_data(
+       inputs["input_ids"], self.model.config.head_dim
+   )
+
+   local_input, magi_attn_key = self._prepare_magi_attention(
+       local_input,
+       cu_seqlens_q,
+       cu_seqlens_k,
+       pad_size,
+       self.model.config.head_dim,
+   )
+   position_ids = get_position_ids(magi_attn_key).unsqueeze(0)
+
+   inputs["position_ids"] = position_ids
+   inputs["input_ids"] = local_input

    return inputs

# dispatch data and prepare key
+ def _prepare_magi_attention(
+    self, inputs, cu_seqlens_q, cu_seqlens_k, pad_size, head_dim
+ ):
+    # ---   magi_attn_flex_dispatch   --- #
+    dist_attn_config = DistAttnConfig()
+    cp_group = self._build_cp_group()
+    inputs = squash_batch_dim(inputs)
+
+    x_padded, dist_attn_runtime_key = magi_attn_varlen_dispatch(
+        inputs,
+        cu_seqlens_q,
+        cu_seqlens_k,
+        head_dim=head_dim,
+        pad_size=pad_size,
+        cp_group=cp_group,
+        causal=True,
+        dist_attn_config=dist_attn_config,
+    )
+    x_padded = x_padded.unsqueeze(0)
+
+    return x_padded, dist_attn_runtime_key
```

Override `compute_loss` because we need to undispatch logits first:
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

Override `training_step`:

We must scale the loss by cp_size before the backward pass, as the dp/fsdp all_reduce averaging process divides the gradients by an additional factor of cp_size(accelerate does not recognize CP. As a result, it treats a combined CP+DP strategy as standard DP.).
```diff
def training_step():
    ...
+   cp_size = int(os.environ.get("CP_SIZE", 1))
+   backward_loss = loss * cp_size
+   self.accelerator.backward(backward_loss, **kwargs)
-   self.accelerator.backward(loss, **kwargs)

    return loss.detach()
```
Override `create_accelerator_and_postprocess` because we want to use our customize accelerator.
```diff
def create_accelerator_and_postprocess():
...
-   self.accelerator = Accelerator(**args)
+   self.accelerator = MagiAccelerator(**args)
...
```

Last but not least, we need to create and use `MagiTrainer` in `run_magi_clm.py`:
```diff
+ from magi_trainer import MagiTrainer

+ trainer = MagiTrainer(...)
- trainer = Trainer(...)

...

trainer.train()
```

### Register Magi_Attention implementation
The following code are all avaliable at Magi_attention.py.

What's more, MagiAttention provides a new type of attention implenmentation(flexible flash_attention), so we need to register it for use:
``` python
def magi_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # (b, num_heads, seq_len, head_dim)
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    magi_attn_key = get_most_recent_key()

    dtype = query.dtype
    q, k, v = [
        rearrange(e, "1 nh s hd -> (1 s) nh hd").to(
            torch.bfloat16
        )  # ffa only supports fp16/bf16 for now
        for e in (query, key, value)
    ]

    o = calc_attn(q, k, v, magi_attn_key)[0]
    o = rearrange(o, "(1 s) nh hd -> 1 s (nh hd)").to(dtype)  # assume batch_size is 1

    return o, None

# register Magi_Attention as attn_backend globally.
ALL_ATTENTION_FUNCTIONS.register("Magi_Attention", magi_attention_forward)
```
Use `Magi_Attention` as model's attention inplementation in `run_magi_clm.py`:
```diff
...
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, **config_kwargs
    )
...

+ config._attn_implementation = "Magi_Attention"

...
```

> [!NOTE]
> We don't need to make any modifications to modeling_llama.py


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
| **Parallel Size**     | CP1/4/8 (MagiAttention) vs no cp(torch native) with a global batch size of 8        |
| **Training Steps**       | 3000                                                                                      |


### Results

MagiAttention aligns well with torch native training:
![Result](./result.png)
