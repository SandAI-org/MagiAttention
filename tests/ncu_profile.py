# NCU profiling script for fa4 vs ffa_fa4
import os
import sys
import torch

# 选择要 profile 的实现: "fa4" 或 "ffa_fa4"
IMPL = os.environ.get("PROFILE_IMPL", "fa4")

from baselines.attn_impl import fa4_varlen_func, ffa_fa4_func
from baselines.utils import generate_seqlens, seqlens2cu_seqlens, seqlens2curanges
import random

random.seed(42)

# ========== 配置参数 ==========
seqlen = 8 * 1024  # 8k
doc_len = 2048  # 每个 doc 2k
nhq = 48
nhk = 4
hd = 128
dtype = torch.bfloat16
device = torch.cuda.current_device()

# ========== 生成数据 ==========
varlen_seqlen_distribution = {(doc_len, doc_len + 1): 1.0}
seqlens = generate_seqlens(varlen_seqlen_distribution, seqlen)
cu_seqlens = seqlens2cu_seqlens(seqlens)
cu_ranges = seqlens2curanges(seqlens)

print(f"Profiling: {IMPL}")
print(f"seqlen={seqlen}, doc_len={doc_len}, num_docs={len(seqlens)}")
print(f"nhq={nhq}, nhk={nhk}, hd={hd}")

# 准备 tensor
q = torch.randn(seqlen, nhq, hd, device=device, dtype=dtype)
k = torch.randn(seqlen, nhk, hd, device=device, dtype=dtype)
v = torch.randn(seqlen, nhk, hd, device=device, dtype=dtype)

cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
q_ranges = torch.tensor(cu_ranges, dtype=torch.int32, device=device)
k_ranges = torch.tensor(cu_ranges, dtype=torch.int32, device=device)
attn_type_map = torch.ones(len(cu_ranges), dtype=torch.int32, device=device)  # causal

# ========== Warmup ==========
print("Warmup...")
if IMPL == "fa4":
    for _ in range(3):
        _ = fa4_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True)
        torch.cuda.synchronize()
elif IMPL == "ffa_fa4":
    # 第一次调用创建 cached FA4AttnArg处理好mask转换
    _ = ffa_fa4_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges, 
                     attn_type_map=attn_type_map, reuse_attn_arg=False)
    torch.cuda.synchronize()
    for _ in range(2):
        _ = ffa_fa4_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges,
                         attn_type_map=attn_type_map, reuse_attn_arg=True)
        torch.cuda.synchronize()

# ========== Profile 目标 kernel ==========
print(f"Running {IMPL} kernel for profiling...")
torch.cuda.synchronize()

# 使用 CUDA profiler range 标记
torch.cuda.cudart().cudaProfilerStart()

if IMPL == "fa4":
    out = fa4_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True)
elif IMPL == "ffa_fa4":
    out = ffa_fa4_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges,
                       attn_type_map=attn_type_map, reuse_attn_arg=True)

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

print(f"Done! Output shape: {out[0].shape}")

