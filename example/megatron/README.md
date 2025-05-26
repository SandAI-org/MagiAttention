## train llama-3-1b with magiattention
We fork a repo of megatron and provide an example for you to train llama-3-1b with magiattention with different cp size. You can refer to https://github.com/SandAI-org/Megatron-LM-MagiAttention/pull/1 for more details.

### Prepare checkpoints
You can refer to the shell in dir magiattention_example/checkpoints/ and
run prepare_llama-3.2-1b_checkpoint.sh to download checkpoint from modelscope and convert checkpoint from huggingface format to megatron format.

tokenizer.model is necessary for training from scratch.

### Prepare data
We use openwebtext(https://huggingface.co/datasets/Skylion007/openwebtext) dataset.

You can run prepare_data.sh in ./magiattention_example/data to download data from huggingface and preprocess data.

### Intergrate with magiattention
We intergrate magiattention with transformer_engine and local transformer inplementation.
Main changes:
- modify pretrain_gpt.py for magi_attention.
    - mainly change dispatch_along_cp_rank, prepare_data and prepare_magi_attention function.
    - main changes:
```python
    def get_batch(data_iterator):
        """Generate a batch."""
        batch = get_batch_on_this_tp_rank(data_iterator)
-       batch = get_batch_on_this_cp_rank(batch)
+       batch = dispatch_along_cp_rank(batch)

        return batch.values()

+   def dispatch_along_cp_rank(batch: Dict[str, Any]):
        ...
+       tokens, labels, cu_seqlens_q, cu_seqlens_k, pad_size = prepare_data(tokens, labels)
+       input, dist_attn_runtime_key = prepare_magi_attention(
               tokens, cu_seqlens_q, cu_seqlens_k, pad_size, mpu.get_context_parallel_group())
        ...
        return batch

    def forward_step(data_iterator, model: GPTModel):
        ...
        with stimer(bdata=True):
-            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
                data_iterator)
+           tokens, labels, loss_mask, attention_mask, position_ids, key = get_batch(
                data_iterator)

        ...
-       output_tensor = model(tokens, position_ids, attention_mask,
                               labels=labels)
+       output_tensor = model(tokens, position_ids, attention_mask,
                            labels=labels, magi_attention_key=key)

        return output_tensor, partial(loss_func, loss_mask)

```

- add magi_attention.py for magiattention implementation.
- replace core_attention with magi_attention in gpt_layer_specs.py for local transformer_impl.
- replace forward function of TEDoctProductAttention with magi_attention forward function for Te transformer_engine transformer_impl:
```python
    class TEDotProductAttention(te.pytorch.DotProductAttention):
        def __init__():
            ...
+           self.magi_attention = MagiAttention(config=config,
                                            layer_number=layer_number,
                                            attn_mask_type=attn_mask_type,
                                            attention_type=attention_type,
                                            attention_dropout=attention_dropout,
                                            softmax_scale=softmax_scale)
            ...

        def forward():
+           core_attn_out = self.magi_attention.forward(query=query, key=key, value=value, attention_mask=attention_mask, attention_bias=attention_bias, packed_seq_params=None, magi_attention_key=magi_attention_key)
+           return core_attn_out
```
- pass magi_attention_key through gpt model forward pass.
- replace get_pos_emb_on_this_cp_rank with get_pos_emb_on_this_cp_rank_magi for rope:
```python
+ def get_pos_emb_on_this_cp_rank_magi(pos_emb: Tensor, magi_attention_key) -> Tensor:
     from magi_attention.api import get_position_ids

     cp_idx = get_position_ids(magi_attention_key)
     pos_emb = pos_emb[cp_idx]

     return pos_emb
```

shells:
- train_llama_1b_from_scratch.sh: You can run this shell to train llama-1b model from scratch.
- resume_from_checkpoint.sh: You can run this shell to train llama-1b model from megatron checkpoint.
- run_llama_from_checkpoint.sh: You can run this shell to train llama-1b model from checkpoint converted from hugging face format.

### Experiments
You can run ./magiattention_example/train_llama_1b_from_scratch.sh to train llama-1b from scratch with magiattention.
training_settings:
- model-size: llama-1b
    - num-layers: 16
    - hidden-size: 2048
    - num-attention-heads: 32
    - group-query-attention
    - num-query-groups: 8
- seqlen: 8192
- context_parallel_size: cp1/2/4/8(magiattention vs te ring attention) with global batch size 16.
- train_iters: 100000

Results:
MagiAttention aligns well with te ring attention.
 ![alt text](./results.png)

Feel free to open any issue in this repo if you have any question!
