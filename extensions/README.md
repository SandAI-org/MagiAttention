# MagiAttention Extensions


## FlashAttention with Attention Sink

### Unitest

```bash
pytest extensions/tests/test_fa_interface_with_sink.py
```

### Basic Usage

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
sq, sk, s_sink = 2048, 2048, 2
nhq, nhk, hd = 8, 4, 128
dtype = torch.bfloat16
device = torch.cuda.current_device()
causal = True

qkv = torch.randn((b, sq, (nhq + nhk*2), hd), dtype=dtype, device=device, requires_grad=True)
sink = torch.randn((s_sink, nhq), dtype=torch.float32, device=device, requires_grad=True)
do = torch.randn((b, sq, nhq, hd), dtype=dtype, device=device, requires_grad=True)

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
