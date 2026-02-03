## integrate MagiAttention with FSDP

We provide a toy example in this direcotry to show you how to integrate MagiAttention with FSDP to train a Llama-3 1B model on randomly generated input data.

### modeling llama
We provide an native inplementation of llama model in `modeling_llama.py`.

To integrate with MagiAttention, we need to pass magi_attn key through the model forward pass:
```diff
class LlamaModel(nn.Module):
    ...
    def forward(
        self,
        v: torch.LongTensor,
+       magi_attention_runtime_key: DistAttnRuntimeKey | None = None,
    ) -> torch.Tensor:
    ...
```
The key will be used in the attention computation func:
```python
def magi_attention_func(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    magi_attention_runtime_key: DistAttnRuntimeKey,
) -> torch.Tensor:
    dtype = q.dtype

    q, k, v = [
        rearrange(e, "1 nh s hd -> (1 s) nh hd").to(
            torch.float16
        )  # ffa only supports fp16/bf16 for now
        for e in (q, k, v)
    ]

    o = calc_attn(q, k, v, magi_attention_runtime_key)[0]
    o = rearrange(o, "(1 s) nh hd -> s (nh hd)").to(dtype)

    return o
```

Llama model configurations are managed in `configuration_llama.py`, which includes a default setting for llama-1b.

### main training loop
The main training loop is inplemented in `main.py`. You can modify the related training setting in `llama_pretrain_config.py`.

```python
# ---   initialize distributed env   --- #
init_env(backend="nccl")

# ---   build device mesh  --- #
device_mesh = build_mesh()

# ---   set seed   --- #
torch.manual_seed(SEED)

# ---   build llama model  --- #
model = build_llama_model()   # build model from modeling llama.

# --   apply parallisim(fsdp + magi_attention)   --- #
parallize_model(model, device_mesh)

# ---   build optimizer and lr_scheduler   --- #
optimizer, lr_scheduler = build_optimizer(model, train_config["optimizer_config"])

# ---   main training loop   --- #
train(model, optimizer, lr_scheduler, device_mesh, train_config["train_iters"])
```


**parallize_model:** we can integrate magi_attention with fsdp2 in a very simple way.
```python
def apply_fsdp(model, device_mesh):
    """
    apply fsdp2 for llama model.
    """
    for module in model.modules():
        if isinstance(module, LlamaDecoderLayer):
            fully_shard(module, mesh=device_mesh)
    fully_shard(model, mesh=device_mesh)

    return model


def parallize_model(model, device_mesh):
    # pass dp_cp mesh to fsdp, fsdp will handle the gradient sync of both dp and cp.
    apply_fsdp(model, device_mesh["dp_cp"])
```


**train func:**
```python
def train(model, optimizer, lr_scheduler, device_mesh, train_iter):
    """main training loop"""
    model.train()

    for iter in range(train_iter):
        input, label, cu_seqlens_q, cu_seqlens_k, pad_size = prepare_data(
            device_mesh, train_iter
        )

        dist_attn_runtime_key = None

        if (
            parallel_config["context_parallel_size"] > 1
            and parallel_config["context_parallel_backend"] == "magi_attention"
        ):
            # dispatched input and prepare magi_attn key.
            input, dist_attn_runtime_key = prepare_magi_attention(
                input=input,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                num_heads_q=model.config.num_attention_heads,
                num_heads_kv=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
                pad_size=pad_size,
                cp_group=device_mesh.get_group("cp"),
            )

        output = model(input, dist_attn_runtime_key)
        loss = loss_func(output, label, device_mesh, dist_attn_runtime_key)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

**Prepare input data:** Generate input data randomly and prepare for magiattention(squash batch dim and compute pad size).
```python
def prepare_data(device_mesh, train_iter):
    # set different seed for each iter to ensure different random data.
    torch.manual_seed(SEED + train_iter)

    # ---   prepare and shard input data and label   --- #
    global_input = torch.randint(
        size=(batch_size, seqlen),
        high=vocab_size,
        device=torch.cuda.current_device(),
    )

    global_label = torch.randint_like(
        global_input,
        high=vocab_size,
    )

    local_input = _shard_along_batch_dim_among_dp(global_input, device_mesh)
    local_label = _shard_along_batch_dim_among_dp(global_label, device_mesh)

    # ---   prepare data for magi_attention   --- #
    # magi_attention do not support input data with batch dim.
    local_input = squash_batch_dim(local_input)
    cp_size = parallel_config["context_parallel_size"]
    head_dim = LlamaConfig().head_dim

    # pad seqlen of input data for better performance.
    pad_size = compute_pad_size(local_input.size(0), cp_size, CHUNK_SIZE)
    cu_seqlens_q, cu_seqlens_k = infer_varlen_mask_from_batch(
        batch_size // dp_size, seqlen
    )

    local_label = squash_batch_dim(local_label)

    return local_input, local_label, cu_seqlens_q, cu_seqlens_k, pad_size
```

**Prepare magi_attn_key:** Dispatch input data along cp dim and get dist_attn_runtime_key.
```python
def prepare_magi_attention(
    input: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    pad_size: int,
    cp_group: dist.ProcessGroup,
):
    # ---   magi_attn_varlen_dispatch   --- #

    dist_attn_config = DistAttnConfig()

    dist_attn_runtime_key = magi_attn_varlen_key(
        cu_seqlens_q,
        cu_seqlens_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=CHUNK_SIZE,
        cp_group_or_mesh=cp_group,
        causal=LlamaConfig().is_causal,
        dist_attn_config=dist_attn_config,
    )

    x_padded = dispatch(input, key=dist_attn_runtime_key)

    return x_padded, dist_attn_runtime_key
```

**Running commands:**
```shell
export GPUS_PER_NODE=${GPUS_PER_NODE:-8} # just set this to 1 to enable the non-dist training
export NNODES=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-16989}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS main.py"
$TORCHRUN_CMD
```
