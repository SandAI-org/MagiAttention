# QuickStart

```{contents}
:local: true
```

## Basic Usage Snippets

### How to Use Flex-Flash-Attention

:::{note}
The following snippet shows how to call `flex_flash_attn_func` for the non-distributed (i.e., single-GPU) scenario with a minimal workflow: configuration, tensor initialization, and calculate attention for both forward and backward passes.
:::

```python
import torch
from magi_attention.api import flex_flash_attn_func

# --- Define attention config --- #

total_seqlen = 2048    # 2k tokens
seqlen_sink = 4        # 4 sink tokens
num_heads_q = 8        # number of attention (query) heads
num_heads_kv = 2       # number of key/value heads (GQA)
head_dim = 128         # dimension of each attention head
dtype = torch.bfloat16 # attention activation / computation dtype (while the reduction dtype is always fp32 for ffa right now)
device = "cuda"
has_sink = True        # whether to apply attention sink

# --- Initialize q,k,v,do tensors --- #

q = torch.randn(total_seqlen, num_heads_q, head_dim, dtype=dtype, device=device, requires_grad=True)
k = torch.randn(total_seqlen, num_heads_kv, head_dim, dtype=dtype, device=device, requires_grad=True)
v = torch.randn(total_seqlen, num_heads_kv, head_dim, dtype=dtype, device=device, requires_grad=True)
do = torch.randn_like(q)

# --- Initialize optional sink tensor --- #

sink = torch.randn(seqlen_sink, num_heads_q, dtype=torch.float32, device=device, requires_grad=True) if has_sink else None

# --- Initialize FFA meta args for customized attention mask --- #

# the following customized attention mask looks like (`*` for unmasked, `0` for masked):
#     - - - - - - - - -> (k)
#   | * * * * 0 0 0 0
#   | * * * * 0 0 0 0
#   | * * * * 0 0 0 0
#   | * * * * 0 0 0 0
#   | * * * * * 0 0 0
#   | * * * * * * 0 0
#   | * * * * * * * 0
#   | * * * * * * * *
#   V
#  (q)
q_ranges_tensor = torch.tensor([[0, 1024], [1024, 2048]], dtype=torch.int32, device=device)
k_ranges_tensor = torch.tensor([[0, 1024], [0, 2048]], dtype=torch.int32, device=device)
attn_type_map_tensor = torch.tensor([0, 1], dtype=torch.int32, device=device) # full mask for 1st slice, causal mask for 2nd

# --- Attention computation --- #

out, meta = flex_flash_attn_func(
    q=q,
    k=k,
    v=v,
    q_ranges=q_ranges_tensor,
    k_ranges=k_ranges_tensor,
    attn_type_map=attn_type_map_tensor,
    sink=sink, # Defaults to None to not apply attention sink
    softmax_scale=None, # Defaults to 1/sqrt(head_dim)
    softcap=0, # Defaults to 0
)
lse = meta.lse

out.backward(do)

dq, dk, dv = q.grad, k.grad, v.grad
dsink = sink.grad if has_sink else None
```

### How to Use MagiAttention

:::{note}
The snippet below illustrates a full distributed workflow: environment setup (*via torchrun*), dist-attn runtime key creation, dispatch/undispatch, distributed attention calculation for both forward and backward passes, as well as an example of handling gradient reduction of attention sink.
:::

```python
# run this python script with the command like:
# torchrun --standalone --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 ${SCRIPT_PATH}
import torch
import torch.nn as nn
import torch.distributed as dist

import magi_attention
from magi_attention.api import (
    magi_attn_flex_key, dispatch, calc_attn, undispatch, # interface functions
    compute_pad_size, # helper functions
)
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.utils import setup_dist_env, clearup_dist_env

# --- Set up distributed environment --- #

rank, local_rank, world_size, num_nodes, num_local_ranks, world_group, device, seed = setup_dist_env()

# --- Define attention config --- #

total_seqlen = 32 * 1024   # 32k tokens, if we dispatch it to 8 GPUs, then each GPU holds 4k tokens
seqlen_sink = 4            # 4 sink tokens
num_heads_q = 48           # number of attention (query) heads
num_heads_kv = 8           # number of key/value heads (GQA)
head_dim = 128             # dimension of each attention head
chunk_size = 512           # chunk size to chunk the input tensor x along the seqlen dim for dispatch to control the granularity of computation load-balance.
dtype = torch.bfloat16     # attention activation / computation dtype (while the reduction dtype for partial attention outputs is always fp32 for magi_attention right now)
has_sink = True            # whether to apply attention sink

# --- Initialize token embedding tensor --- #

embed_dim = 4096
x = torch.randn(total_seqlen, embed_dim, device=device, dtype=dtype, requires_grad=True)

# --- Initialize MagiAttention meta configs for customized attention mask --- #

# the following customized attention mask is known as `block-causal` mask where `block_size` = 4096 (4k),
# which looks like (`*` for unmasked, `0` for masked):
#     - - - - - - - - -> (k)
#   | * * 0 0 0 0 0 0
#   | * * 0 0 0 0 0 0
#   | * * * * 0 0 0 0
#   | * * * * 0 0 0 0
#   | * * * * * * 0 0
#   | * * * * * * 0 0
#   | * * * * * * * *
#   | * * * * * * * *
#   V
#  (q)
q_ranges = AttnRanges.from_ranges(
    [
        [0, 4096], # 0~4k
        [4096, 8192], # 4k~8k
        [8192, 12288], # 8k~12k
        [12288, 16384], # 12k~16k
        [16384, 20480], # 16k~20k
        [20480, 24576], # 20k~24k
        [24576, 28672], # 24k~28k
        [28672, 32768], # 28k~32k
    ]
)
k_ranges = AttnRanges.from_ranges(
    [
        [0, 4096], # 0~4k
        [0, 8192], # 0~8k
        [0, 12288], # 0~12k
        [0, 16384], # 0~16k
        [0, 20480], # 0~20k
        [0, 24576], # 0~24k
        [0, 28672], # 0~28k
        [0, 32768], # 0~32k
    ]
)
attn_mask_type = [AttnMaskType.FULL] * len(q_ranges)
total_seqlen_q = total_seqlen_k = total_seqlen
pad_size = compute_pad_size( # pad embeds along seqlen dim for better performance
    total_seqlen_q=total_seqlen_q,
    cp_size=world_size, # assuming we only have 1-dim context parallelism (cp)
    chunk_size=chunk_size,
)

# --- Dispatch token embedding tensor along seqlen dim to multiple ranks --- #

# NOTE:
# 1. the dispatched local token embedding may be shuffled along seqlen dim,
#    so it's safe for token-wise operations such as matmul, layer-norm, etc
#    while for sample-wise operations like RoPE, you might need to be more careful
# 2. the `magi_attn_runtime_key` holds some inner meta data,
#    as a required argument for many APIs of ``magi_attention``,
#    which users don't have to bother with
magi_attn_runtime_key = magi_attn_flex_key(
    q_ranges=q_ranges,
    k_ranges=k_ranges,
    attn_mask_type=attn_mask_type,
    total_seqlen_q=total_seqlen_q,
    total_seqlen_k=total_seqlen_k,
    num_heads_q=num_heads_q,
    num_heads_kv=num_heads_kv,
    head_dim=head_dim,
    pad_size=pad_size,
    chunk_size=chunk_size,
    cp_group_or_mesh=world_group, # assuming we only have 1-dim context parallelism (cp)
)
local_x = dispatch(x, key=magi_attn_runtime_key)

# --- Simulate QKV projection --- #

q_proj = nn.Linear(embed_dim, num_heads_q * head_dim, dtype=dtype, device=device)
k_proj = nn.Linear(embed_dim, num_heads_kv * head_dim, dtype=dtype, device=device)
v_proj = nn.Linear(embed_dim, num_heads_kv * head_dim, dtype=dtype, device=device)

local_q = q_proj(local_x).view(-1, num_heads_q, head_dim)
local_k = k_proj(local_x).view(-1, num_heads_kv, head_dim)
local_v = v_proj(local_x).view(-1, num_heads_kv, head_dim)

# --- Simulate attention sink parameter --- #

global_sink = nn.Parameter(torch.randn(seqlen_sink, num_heads_q, dtype=torch.float32, device=device)) if has_sink else None

# --- Distributed attention computation --- #

local_out, meta = calc_attn(
    q=local_q,
    k=local_k,
    v=local_v,
    key=magi_attn_runtime_key,
    sink=global_sink, # Defaults to None to not apply attention sink
)
local_lse = meta.lse

# --- Undispatch the output tensor along seqlen dim from multiple ranks and unpad --- #

# NOTE: the undispatch API may not be used until the moment you need the seqlen dimension to be compelete and ordered,
# e.g. for either aforementioned sample-wise operations, or loss computation
total_out = undispatch(
    x=local_out,
    key=magi_attn_runtime_key,
)

# --- Simulate loss computation --- #

loss = total_out.sum()

# --- Simulate backward pass --- #

loss.backward()

dx = x.grad
dq_proj, dk_proj, dv_proj = q_proj.weight.grad, k_proj.weight.grad, v_proj.weight.grad

if has_sink:
    dsink = global_sink.grad
    # NOTE: since usually the training framework such as Megatron-LM, FSDP
    # will handle the reduction of parameters' gradients across the whole dp x cp group
    # so by default, MagiAttention will skip the reduction of sink's gradients
    # unless the users specify the environment variable `MAGI_ATTENTION_DSINK_ALL_REDUCE_OP` (see our docs for more details)
    if (op:=magi_attention.comm.dsink_all_reduce_op()) != "none":
        match op:
            case "sum":
                dist.all_reduce(dsink, op=dist.ReduceOp.SUM, group=world_group)
            case "avg":
                dist.all_reduce(dsink, op=dist.ReduceOp.AVG, group=world_group)
            case _:
                raise ValueError(f"Unknown all_reduce_op: {op}")

# --- Clear up distributed environment --- #

clearup_dist_env()
```

## Examples

:::{warning}
The following examples are a work in progress. More complete and detailed examples will be provided in a future update.
:::

### Integration with FSDP2

An example demonstrating how to integrate MagiAttention with FSDP2 is provided in [examples/torch_native](https://github.com/SandAI-org/MagiAttention/tree/main/examples/torch_native). Use the included `run.sh` to execute the example.

This example constructs a LLaMA-1B model and demonstrates applying FSDP2 with MagiAttention as the parallelism strategy.

- `examples/torch_native/modeling_llama.py`: Model construction and MagiAttention integration.
- `examples/torch_native/main.py`: Training loop and execution script.


### Integration with Megatron-LM
    
We maintain a repository [Megatron-LM-MagiAttention](https://github.com/SandAI-org/Megatron-LM-MagiAttention/tree/magi_attention) that demonstrates integrating MagiAttention with Megatron-LM. The repository is forked from [Megatron-LM v0.11.0](https://github.com/NVIDIA/Megatron-LM/tree/v0.11.0) and provides a working example for training LLaMA-1B with Megatron-LM + MagiAttention. We also include experiment logs for a LLaMA-3-1B training run to validate convergence.

For more information, consult [examples/megatron](https://github.com/SandAI-org/MagiAttention/tree/main/examples/megatron).


### Integration with Transformers

An integration example for transformers is available in [examples/transformers](https://github.com/SandAI-org/MagiAttention/tree/main/examples/transformers). 

We used this integration to run a continued-training experiment on LLaMA-3-1B and validated MagiAttention convergence.