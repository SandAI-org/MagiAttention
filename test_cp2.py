#!/usr/bin/env python3
"""单独复现 illegal memory access 的 test case"""
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
# 设置环境变量（根据出错 case 的 flag_comb）
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
os.environ["MAGI_ATTENTION_DETERMINISTIC_MODE"] = "0"   # False
os.environ["MAGI_ATTENTION_HIERARCHICAL_COMM"] = "0"    # False
os.environ["MAGI_ATTENTION_QO_COMM"] = "0"              # False
os.environ["MAGI_ATTENTION_NATIVE_GRPCOLL"] = "0"       # False
os.environ["MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE"] = "0"   # False
os.environ["MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE"] = "1"  # True
import magi_attention
from magi_attention import init_dist_attn_runtime_mgr
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import DistAttnConfig, DispatchConfig, OverlapConfig, MinHeapDispatchAlg
def main():
    # 初始化分布式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # 创建 process group
    nccl_group = dist.new_group(list(range(world_size)), backend="nccl")
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(1, world_size), mesh_dim_names=("inter", "intra"))
    # ===== 配置参数（对应出错的 case）=====
    # full_attn_14k + INVCAUSAL mask
    q_ranges = AttnRanges.from_ranges([[0, 14336]])
    k_ranges = AttnRanges.from_ranges([[0, 14336]])
    attn_mask_type = [AttnMaskType.INVCAUSAL]  # attn_type_mapping=[2]
    total_seqlen_q = 14336
    total_seqlen_k = 14336
    chunk_size = 512
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 128
    dtype = torch.float16
    softmax_scale = None
    softcap = 0.0
    # ===== 创建 dist attn config =====
    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        overlap_config=OverlapConfig(enable=False),
    )
    print(f"[RANK {rank}] Initializing dist_attn_runtime_mgr...", flush=True)
    # ===== 初始化 runtime manager =====
    dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        chunk_size=chunk_size,
        cp_group=nccl_group,
        is_same_source=True,
        is_q_permutable=True,
        is_k_permutable=True,
        dist_attn_config=dist_attn_config,
        cp_mesh=device_mesh,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
    )
    print(f"[RANK {rank}] Generating tensors...", flush=True)
    # ===== 生成输入 tensor =====
    total_q = torch.randn(total_seqlen_q, num_heads_q, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    total_k = torch.randn(total_seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    total_v = torch.randn(total_seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    dist.all_reduce(total_q.data, group=nccl_group)
    dist.all_reduce(total_k.data, group=nccl_group)
    dist.all_reduce(total_v.data, group=nccl_group)
    # ===== Dispatch =====
    print(f"[RANK {rank}] Dispatching...", flush=True)
    local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
    local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
    local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)
    # ===== Forward =====
    print(f"[RANK {rank}] Running forward...", flush=True)
    local_out, local_lse = dist_attn_runtime_mgr.calc_attn(
        q=local_q,
        k=local_k,
        v=local_v,
        sink=None,
        softmax_scale=softmax_scale,
        softcap=softcap,
    )
    print(f"[RANK {rank}] Forward done!", flush=True)
    # ===== Undispatch =====
    total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)
    # ===== Backward =====
    print(f"[RANK {rank}] Start Backward", flush=True)
    grad_total_out = torch.randn_like(total_out).detach()
    dist.all_reduce(grad_total_out.data, group=nccl_group)
    torch.cuda.synchronize()
    print(f"[RANK {rank}] Call Backward", flush=True)
    total_out.backward(grad_total_out)
    print(f"[RANK {rank}] Backward done!", flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    main()
