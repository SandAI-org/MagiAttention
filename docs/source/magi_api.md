# MagiAttention API

```{eval-rst}
.. py:module:: magi_attention.api
```

```{contents}
:local: true
```

## Flexible Flash Attention

To support computing irregular-shaped masks, we implemented a `flexible_flash_attention` kernel, which can be invoked through the following interface.

```{eval-rst}
.. currentmodule:: magi_attention.functional.flex_flash_attn
```

```{eval-rst}
.. autofunction:: flex_flash_attn_func
```

## How to Use MagiAttention

The typical process for calling MagiAttention is: initialize the required parameters → use `compute_pad_size` to get the pad size → call the dispatch function → pass x through projection to obtain qkv → perform attention calculation → undispatch. An example call is shown below.

<details>
<summary>Basic Usage For Varlen Api</summary>

```python
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from magi_attention.api import (
    AttnOverlapMode,
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
    UniformOverlapAlg,
    calc_attn,
    compute_pad_size,
    full_attention_to_varlen_attention,
    magi_attn_varlen_dispatch,
    squash_batch_dim,
    undispatch,
)

# ---  prepare data and args for magi_attention --- #
# init params
embed_dim = 1024
dtype = torch.bfloat16
cp_size = 2
head_dim = 128
chunk_size = 512
q_heads = 48
kv_heads = 8
batch_size = 5
seqlen = 25

dist_attn_config = DistAttnConfig(
    dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
    overlap_config=OverlapConfig(
        enable=True,
        mode=AttnOverlapMode.STATIC,
        degree=2,
        min_chunk_size=512,
        max_num_chunks=64,
        alg=UniformOverlapAlg(
            random_costs=True,
            random_seed=42,
        ),
    ),
    high_bandwith_domain_size=1,
)

# init distributed environment if necessary
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == cp_size
dist.init_process_group(
    backend="nccl",
    world_size=world_size,
    rank=rank,
    timeout=timedelta(minutes=30),
)
local_rank = rank % 8
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

# init cp_group
cp_group = dist.new_group(list(range(cp_size)), backend="nccl")
cp_mesh = None
# if you want to use hierarchical_comm
# first export MAGI_ATTENTION_HIERARCHICAL_COMM = 1 and CUDA_DEVICE_MAX_CONNECTIONS = 8
# second set cp_group = None and init cp_mesh with init_hierarchical_mesh function

# create input data with shape (bs, seqlen, h)
x_with_batch = torch.randn(
    batch_size, seqlen, embed_dim, device=device, dtype=dtype, requires_grad=True
)

# squash the batch dim, magi_attention do not support input data with batch dim.
x = squash_batch_dim(x_with_batch)  # ((b, seqlen), h)

# get cu_seqlens_q,k after squashing.
cu_seqlens_q, cu_seqlens_k = full_attention_to_varlen_attention(batch_size, seqlen)
total_seqlen_q: int = batch_size * seqlen
total_seqlen_k: int = batch_size * seqlen

# pad input seqlen for better performance
pad_size = compute_pad_size(total_seqlen_q, cp_size, head_dim, chunk_size)

# ---   magi_attention dispatch   --- #

# dispatch global input tensor to each rank and get the runtime_key
(
    local_x,
    magi_attn_runtime_key,
) = magi_attn_varlen_dispatch(  # local_x with shape ((total_seq + pad_size) / cp_size), h)
    x,
    cu_seqlens_q,
    cu_seqlens_k,
    head_dim=head_dim,
    pad_size=pad_size,
    chunk_size=chunk_size,
    cp_group=cp_group,
    cp_mesh=cp_mesh,
    causal=False,
    dist_attn_config=dist_attn_config,
)

# ---  magi_attention calculation and undispatch  --- #
# do q k v projection, here's just an example
q_proj = torch.nn.Linear(embed_dim, q_heads * head_dim, dtype=dtype, device=device)
k_proj = torch.nn.Linear(embed_dim, kv_heads * head_dim, dtype=dtype, device=device)
v_proj = torch.nn.Linear(embed_dim, kv_heads * head_dim, dtype=dtype, device=device)

local_q, local_k, local_v = (
    q_proj(local_x).view(-1, q_heads, head_dim),
    k_proj(local_x).view(-1, kv_heads, head_dim),
    v_proj(local_x).view(-1, kv_heads, head_dim),
)  # q, k, v with shape ((bs * seqlen + pad_size) / cp_size, nh, hd)

# Do local attention computation with runtime key
local_out, _ = calc_attn(
    local_q, local_k, local_v, magi_attn_runtime_key
)  # local out with shape ((bs * seqlen + pad_size) / cp_size, nh, hd)

# Gather local attention results to global result with runtime key
total_out = undispatch(
    local_out, magi_attn_runtime_key
)  # total out with shape (bs * seqlen, nh, hd)
```

</details>

<details>
<summary>Basic Usage For Flexible Api</summary>

```python
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from magi_attention.api import (
    AttnMaskType,
    AttnOverlapMode,
    AttnRanges,
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
    UniformOverlapAlg,
    calc_attn,
    compute_pad_size,
    magi_attn_flex_dispatch,
    undispatch,
)

# init params
embed_dim = 1024
dtype = torch.bfloat16
cp_size = 2
head_dim = 128
total_seqlen_q = 960
total_seqlen_k = 960
chunk_size = 512
q_heads = 48
kv_heads = 8

dist_attn_config = DistAttnConfig(
    dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
    overlap_config=OverlapConfig(
        enable=True,
        mode=AttnOverlapMode.STATIC,
        degree=2,
        min_chunk_size=512,
        max_num_chunks=64,
        alg=UniformOverlapAlg(
            random_costs=True,
            random_seed=42,
        ),
    ),
    high_bandwith_domain_size=1,
)

# init distributed environment if necessary
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == cp_size
dist.init_process_group(
    backend="nccl",
    world_size=world_size,
    rank=rank,
    timeout=timedelta(minutes=30),
)
local_rank = rank % 8
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

# init cp_group
cp_group = dist.new_group(list(range(cp_size)), backend="nccl")
cp_mesh = None
# if you want to use hierarchical_comm
# first export MAGI_ATTENTION_HIERARCHICAL_COMM = 1 and CUDA_DEVICE_MAX_CONNECTIONS = 8
# second set cp_group = None and init cp_mesh with init_hierarchical_mesh function

# init x input
x = torch.randn(
    total_seqlen_q, embed_dim, device=device, dtype=dtype, requires_grad=True
)

# init mask shape
q_ranges = AttnRanges.from_ranges(
    [
        [0, 128],
        [128, 256],
        [256, 384],
        [384, 512],
        [512, 640],
        [640, 768],
        [768, 960],
    ]
)

k_ranges = AttnRanges.from_ranges(
    [
        [0, 128],
        [0, 256],
        [0, 384],
        [0, 512],
        [512, 640],
        [512, 768],
        [768, 960],
    ]
)

# you can also init attn_mask_type with list[str]
# such as  attn_mask_type = ["full"] * 7
attn_mask_type = [AttnMaskType.FULL] * 7

# calc pad_size
pad_size = compute_pad_size(total_seqlen_q, cp_size, head_dim, chunk_size)

(
    local_x,
    magi_attn_runtime_key,
) = magi_attn_flex_dispatch(  # local_x with shape (total_seqlen_q + pad_size) / cp_size, h)
    x,
    q_ranges=q_ranges,
    k_ranges=k_ranges,
    attn_mask_type=attn_mask_type,
    total_seqlen_q=total_seqlen_q,
    total_seqlen_k=total_seqlen_k,
    head_dim=head_dim,
    pad_size=pad_size,
    chunk_size=chunk_size,
    cp_group=cp_group,
    cp_mesh=cp_mesh,
    dist_attn_config=dist_attn_config,
    is_same_source=True,
    is_q_permutable=True,
    is_k_permutable=True,
)

# ---  magi_attention calculation and undispatch  --- #
# do q k v projection, here's just an example
q_proj = torch.nn.Linear(embed_dim, q_heads * head_dim, dtype=dtype, device=device)
k_proj = torch.nn.Linear(embed_dim, kv_heads * head_dim, dtype=dtype, device=device)
v_proj = torch.nn.Linear(embed_dim, kv_heads * head_dim, dtype=dtype, device=device)

local_q, local_k, local_v = (
    q_proj(local_x).view(-1, q_heads, head_dim),
    k_proj(local_x).view(-1, kv_heads, head_dim),
    v_proj(local_x).view(-1, kv_heads, head_dim),
)  # q, k, v with shape (s, nh, hd)

# Do local attention computation with runtime key
local_out, _ = calc_attn(
    local_q, local_k, local_v, magi_attn_runtime_key
)  # local out with shape (s, nh, hd)

# Gather local attention results and unpad to global result with runtime key
total_out = undispatch(
    local_out, magi_attn_runtime_key
)  # total out with shape (totoal_seqlen_q, nh, hd)
```

</details>

## Compute Pad Size

During the use of MagiAttention, we divide the `total_seqlen` into multiple chunks of size `chunk_size` and evenly distribute them across multiple GPUs. To ensure that `total_seqlen` is divisible by `chunk_size` and that each GPU receives the same number of chunks, we need to pad the original input. You can call `compute_pad_size` to calculate the required padding length, and use this value as a parameter in subsequent functions.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: compute_pad_size
```

## Dispatch

### Dispatch for varlen masks

If you're using a mask defined by `cu_seqlens`, such as a varlen full or varlen causal mask, we've designed a similar interface inspired by FlashAttention's API, making it easy for you to get started quickly. In the function named `magi_attn_varlen_dispatch`, you can obtain the dispatched `x` and `key`.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: magi_attn_varlen_dispatch
```

The logic of the `magi_attn_varlen_dispatch` function mainly consists of two parts: it first calls `magi_attn_varlen_key` to compute a key value, and then uses this key to dispatch the input x. The description of `magi_attn_varlen_key` is as follows.

```{eval-rst}
.. autofunction:: magi_attn_varlen_key
```

### Dispatch for flexible masks

If the masks you're using are not limited to varlen full or varlen causal, but also include sliding window masks or other more diverse types, we recommend using the following API. By calling `magi_attn_flex_dispatch`, you can obtain the dispatched x and key.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: magi_attn_flex_dispatch
```

Similar to the logic of `magi_attn_varlen_dispatch`, `magi_attn_flex_dispatch` first calls `magi_attn_flex_key` to obtain a key, and then uses this key to dispatch x. The description of `magi_attn_flex_key` is as follows.

```{eval-rst}
.. autofunction:: magi_attn_flex_key
```

## Calculate Attention

After dispatch and projection, you should obtain the query, key, and value needed for computation. Using the key obtained from the dispatch function mentioned above, you can perform the computation by calling `calc_attn`, which returns the results out and lse. The description of calc_attn is as follows.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: calc_attn
```

## Undispatch

After the attention computation, communication is needed to gather the results back to all GPUs. We provide an API to perform the undispatch process.

```{eval-rst}
.. currentmodule:: magi_attention.api.magi_attn_interface
```

```{eval-rst}
.. autofunction:: undispatch
```

## Utility Functions

To initialize `attn_mask_type`, you can use either the `AttnMaskType` enum type or its corresponding string representation.

```{eval-rst}
.. currentmodule:: magi_attention.api
```

```{eval-rst}
.. autoclass:: AttnMaskType
    :members:
    :exclude-members: __.*
```

In the dispatch function, a parameter of type `DistAttnConfig` is required. You can configure it according to the following instructions.

```{eval-rst}
.. currentmodule:: magi_attention.api
```

```{eval-rst}
.. autoclass:: DistAttnConfig
```

In the dispatch function, you can enable the hierarchical mode by setting `cp_group` to `None` and providing a `DeviceMesh` type parameter instead. We offer the `init_hierarchical_mesh` function to help you easily initialize the `cp_mesh`.

```{eval-rst}
.. currentmodule:: magi_attention.api.functools
```

```{eval-rst}
.. autofunction:: init_hierarchical_mesh
```
