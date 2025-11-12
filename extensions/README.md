# MagiAttention Extensions


## FlashAttention with Attention Sink

### Unitest

```bash
pytest extensions/tests/test_fa_interface_with_sink.py
```

### Basic Usage for FlashAttention 3

#### Basic Usage for fa3_func_with_sink

```python
import torch
from extensions.fa3_interface_with_sink import fa3_func_with_sink

b = 2
sq, sk, s_sink = 2048, 2048, 2
nhq, nhk, hd = 8, 4, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

q = torch.randn((b, sq, nhq, hd), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((b, sk, nhk, hd), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((b, sk, nhk, hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nhq), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn_like(q)

out, lse = fa3_func_with_sink(
    q=q,
    k=k,
    v=v,
    sink=sink,
    causal=causal,
    return_attn_probs=True,
)
out.backward(do)

dq, dk, dv, dsink = q.grad, k.grad, v.grad, sink.grad
```

#### Basic Usage for fa3_varlen_func_with_sink

```python
import torch
from extensions.fa3_interface_with_sink import fa3_varlen_func_with_sink

sq, sk, s_sink = 2048, 2048, 2
nhq, nhk, hd = 8, 4, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

q = torch.randn((sq, nhq, hd), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((sk, nhk, hd), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((sk, nhk, hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nhq), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn_like(q)

cu_seqlens_q = torch.tensor([0, sq // 2, sq], dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor([0, sk // 2, sk], dtype=torch.int32, device=device)
max_seqlen_q = sq // 2
max_seqlen_k = sk // 2

out, lse = fa3_varlen_func_with_sink(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    sink=sink,
    causal=causal,
    return_attn_probs=True,
)

out.backward(do)

dq, dk, dv, dsink = q.grad, k.grad, v.grad, sink.grad
```

#### Basic Usage for fa3_qkvpacked_func_with_sink

```python
import torch
from extensions.fa3_interface_with_sink import fa3_qkvpacked_func_with_sink

b = 2
s, s_sink = 2048, 2
nhq, nhk, hd = 8, 4, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

qkv = torch.randn((b, s, (nhq + nhk*2), hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nhq), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn((b, s, nhq, hd), dtype=dtype, device=device, requires_grad=True)

out, lse = fa3_qkvpacked_func_with_sink(
    qkv=qkv,
    sink=sink,
    causal=causal,
    num_heads_q=nhq,
    return_attn_probs=True,
)
out.backward(do)

dqkv, dsink = qkv.grad, sink.grad
```


### Basic Usage for FlashAttention 2

#### Basic Usage for fa2_func_with_sink

```python
import torch
from extensions.fa2_interface_with_sink import fa2_func_with_sink

b = 2
sq, sk, s_sink = 2048, 2048, 2
nhq, nhk, hd = 8, 4, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

q = torch.randn((b, sq, nhq, hd), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((b, sk, nhk, hd), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((b, sk, nhk, hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nhq), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn_like(q)

out = fa2_func_with_sink(
    q=q,
    k=k,
    v=v,
    sink=sink,
    causal=causal,
    return_attn_probs=False,
)
out.backward(do)

dq, dk, dv, dsink = q.grad, k.grad, v.grad, sink.grad
```

#### Basic Usage for fa2_varlen_func_with_sink

```python
import torch
from extensions.fa2_interface_with_sink import fa2_varlen_func_with_sink

sq, sk, s_sink = 2048, 2048, 2
nhq, nhk, hd = 8, 4, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

q = torch.randn((sq, nhq, hd), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((sk, nhk, hd), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((sk, nhk, hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nhq), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn_like(q)

cu_seqlens_q = torch.tensor([0, sq // 2, sq], dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor([0, sk // 2, sk], dtype=torch.int32, device=device)
max_seqlen_q = sq // 2
max_seqlen_k = sk // 2

out = fa2_varlen_func_with_sink(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    sink=sink,
    causal=causal,
    return_attn_probs=False,
)

out.backward(do)

dq, dk, dv, dsink = q.grad, k.grad, v.grad, sink.grad
```

#### Basic Usage for fa2_qkvpacked_func_with_sink

```python
import torch
from extensions.fa2_interface_with_sink import fa2_qkvpacked_func_with_sink

b = 2
s, s_sink = 2048, 2
nh, hd = 8, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

qkv = torch.randn((b, s, 3, nh, hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nh), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn((b, s, nh, hd), dtype=dtype, device=device, requires_grad=True)

out = fa2_qkvpacked_func_with_sink(
    qkv=qkv,
    sink=sink,
    causal=causal,
    return_attn_probs=False,
)
out.backward(do)

dqkv, dsink = qkv.grad, sink.grad
```

#### Basic Usage for fa2_kvpacked_func_with_sink

```python
import torch
from extensions.fa2_interface_with_sink import fa2_kvpacked_func_with_sink

b = 2
sq, sk, s_sink = 2048, 2048, 2
nhq, nhk, hd = 8, 4, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

q = torch.randn((b, sq, nhq, hd), dtype=dtype, device=device, requires_grad=True)
kv = torch.randn((b, sk, 2, nhk, hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nhq), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn_like(q)

out = fa2_kvpacked_func_with_sink(
    q=q,
    kv=kv,
    sink=sink,
    causal=causal,
    return_attn_probs=False,
)
out.backward(do)

dq, dkv, dsink = q.grad, kv.grad, sink.grad
```

#### Basic Usage for fa2_varlen_qkvpacked_func_with_sink

```python
import torch
from extensions.fa2_interface_with_sink import fa2_varlen_qkvpacked_func_with_sink

s, s_sink = 2048, 2
nh, hd = 8, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

qkv = torch.randn((s, 3, nh, hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nh), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn((s, nh, hd), dtype=dtype, device=device, requires_grad=True)

cu_seqlens = torch.tensor([0, s // 2, s], dtype=torch.int32, device=device)
max_seqlen = s // 2

out = fa2_varlen_qkvpacked_func_with_sink(
    qkv=qkv,
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
    sink=sink,
    causal=causal,
    return_attn_probs=False,
)

out.backward(do)

dqkv, dsink = qkv.grad, sink.grad
```

#### Basic Usage for fa2_varlen_kvpacked_func_with_sink

```python
import torch
from extensions.fa2_interface_with_sink import fa2_varlen_kvpacked_func_with_sink

sq, sk, s_sink = 2048, 2048, 2
nhq, nhk, hd = 8, 4, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

q = torch.randn((sq, nhq, hd), dtype=dtype, device=device, requires_grad=True)
kv = torch.randn((sk, 2, nhk, hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nhq), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn_like(q)

cu_seqlens_q = torch.tensor([0, sq // 2, sq], dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor([0, sk // 2, sk], dtype=torch.int32, device=device)
max_seqlen_q = sq // 2
max_seqlen_k = sk // 2


out = fa2_varlen_kvpacked_func_with_sink(
    q=q,
    kv=kv,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    sink=sink,
    causal=causal,
    return_attn_probs=False,
)

out.backward(do)

dq, dkv, dsink = q.grad, kv.grad, sink.grad
```
