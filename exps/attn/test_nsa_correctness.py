# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import sys
from typing import Tuple

import torch
from einops import rearrange

# Add baselines to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines"))

from baselines.attn_impl import ffa_func, sdpa_func
from baselines.utils import seed_everything
from baselines.nsa_ref.ops import compressed_attention, linear_compress
from baselines.nsa_ref.ops.topk_sparse_attention import _topk_sparse_attention_fwd

from magi_attention.utils.sparse_utils import generate_ranges_from_topk_index, get_sdpa_mask_from_topk_index, generate_ranges_from_topk_index_token_major

def create_cu_seqlens(seqlen: int) -> torch.Tensor:
    """Create cumulative sequence lengths tensor for batch processing."""
    return torch.arange(0, 2 * seqlen, seqlen, dtype=torch.int32)


class NSACorrectnessTest:
    """Test correctness of NSA implementation by comparing FFA and NSA reference outputs."""
    
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        seed_everything()
    
    def setup_test_data(
        self, 
        seqlen: int, 
        num_q_heads: int, 
        num_k_heads: int, 
        head_dim: int,
        kernel_size: int = 32,
        kernel_stride: int = 16
    ) -> Tuple[torch.Tensor, ...]:
        """Generate test data for NSA correctness test."""
        
        # Generate random q, k, v tensors
        q = torch.randn(seqlen, num_q_heads, head_dim, device=self.device, dtype=self.dtype)
        k = torch.randn(seqlen, num_k_heads, head_dim, device=self.device, dtype=self.dtype)
        v = torch.randn(seqlen, num_k_heads, head_dim, device=self.device, dtype=self.dtype)
        
        # Generate NSA-specific compression parameters
        compress_key = torch.randn(num_k_heads, head_dim * kernel_size, head_dim, 
                                 device=self.device, dtype=self.dtype)
        compress_value = torch.randn(num_k_heads, head_dim * kernel_size, head_dim, 
                                   device=self.device, dtype=self.dtype)
        intra_block_pe = torch.randn(num_k_heads, kernel_size, head_dim, 
                                   device=self.device, dtype=self.dtype)
        
        # Create cumulative sequence lengths
        cu_seqlens = create_cu_seqlens(seqlen).to(self.device)
        
        return q, k, v, compress_key, compress_value, intra_block_pe, cu_seqlens
    
    def compute_topk_indices(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        compress_key: torch.Tensor,
        compress_value: torch.Tensor,
        intra_block_pe: torch.Tensor,
        cu_seqlens: torch.Tensor,
        kernel_size: int,
        kernel_stride: int,
        block_size: int,
        sparsity_ratio: float,
        head_dim: int
    ) -> torch.Tensor:
        """Compute topk indices using compressed attention."""
        
        # Compute compressed representations
        compressed_k, compressed_cu_seqlens = linear_compress(
            k, compress_key, cu_seqlens, kernel_size, kernel_stride, intra_block_pe
        )
        compressed_v, _ = linear_compress(
            v, compress_value, cu_seqlens, kernel_size, kernel_stride, None
        )
        
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        seqlen = cu_seqlens[1].item()
        
        # Calculate topk value
        num_k_blocks = seqlen // block_size
        topk = int(sparsity_ratio * num_k_blocks)
        
        # Compute attention scores and get topk indices
        _, topk_idx = compressed_attention(
            q=q,
            k=compressed_k,
            v=compressed_v,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=compressed_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=compressed_seqlens.max().item(),
            sm_scale=1.0 / math.sqrt(head_dim),
            init_blocks=1,
            local_blocks=2,
            parallel_topk_compute=False,
        )
        num_k_block = (topk_idx != -1).sum().item()
        print(f"Total K Block: {num_k_block}")
        
        return topk_idx
    
    def compute_ffa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_idx: torch.Tensor,
        block_size: int,
        num_group: int,
        head_dim: int
    ) -> torch.Tensor:
        """Compute attention using FFA implementation."""
        
        # Reshape tensors for FFA format
        q_ffa = rearrange(q, "s h d -> (s h) 1 d")
        k_ffa = rearrange(k, "s h d -> (h s) 1 d")
        v_ffa = rearrange(v, "s h d -> (h s) 1 d")
        
        # Generate ranges from topk indices
        seqlen = q.shape[0]
        q_ranges, k_ranges = generate_ranges_from_topk_index_token_major(
            topk_idx, num_group, block_size, seqlen
        )
        
        # Create attention type map (all zeros for standard attention)
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=self.device)
        
        # Compute attention
        output, *rest = ffa_func(
            q_ffa,
            k_ffa,
            v_ffa,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            max_seqlen_q=num_group,
            max_seqlen_k=block_size,
            softmax_scale=1.0 / math.sqrt(head_dim),
            auto_range_merge=True,
        )
        print(output.shape)
        
        # Reshape back to original format
        output = rearrange(output, "(s h) 1 d -> s h d", h=q.shape[1])
        return output
    
    def compute_sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_idx: torch.Tensor,
        block_size: int,
        num_group: int,
        head_dim: int
    ):
        """Compute attention using standard dense attention with masking."""
        softmax_scale = 1 / math.sqrt(head_dim)
        seqlen = q.shape[0]
        # Generate SDPA mask from topk indices
        sdpa_mask = get_sdpa_mask_from_topk_index(
            topk_idx, num_group=num_group, block_n=block_size, seqlen_k=seqlen
        )
        q_sdpa = rearrange(q, "s h d -> h s d").unsqueeze(0)
        k_sdpa = rearrange(k, "s h d -> h s d").unsqueeze(0)
        v_sdpa = rearrange(v, "s h d -> h s d").unsqueeze(0)

        output = sdpa_func(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=sdpa_mask,
            is_causal=False,
            scale=softmax_scale,
            enable_gqa=True,
        )        
        output = rearrange(output, "b h s d -> b s h d").squeeze(0)
        return output
    
    def compute_nsa_ref_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        topk_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        block_size: int,
        head_dim: int
    ) -> torch.Tensor:
        """Compute attention using NSA reference implementation."""
        
        seqlen = cu_seqlens[1].item()
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        output, _ = _topk_sparse_attention_fwd(
            q, k, v, topk_idx, block_size,
            cu_seqlens, cu_seqlens, seqlen, seqlen, sm_scale
        )
        
        return output
    
    def compare_outputs(
        self, 
        output_ffa: torch.Tensor, 
        output_nsa: torch.Tensor, 
        rtol: float = 1e-3, 
        atol: float = 1e-3
    ) -> dict:
        """Compare outputs from FFA and NSA implementations."""
        
        # Compute differences
        abs_diff = torch.abs(output_ffa - output_nsa)
        rel_diff = abs_diff / (torch.abs(output_nsa) + 1e-8)
        
        # Statistics
        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        mean_rel_diff = rel_diff.mean().item()
        
        # Check if outputs are close
        outputs_close = torch.allclose(output_ffa, output_nsa, rtol=rtol, atol=atol)
        
        return {
            "outputs_close": outputs_close,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mean_abs_diff": mean_abs_diff,
            "mean_rel_diff": mean_rel_diff,
            "rtol_used": rtol,
            "atol_used": atol
        }
    
    def run_single_test(
        self,
        seqlen: int = 1024,
        num_q_heads: int = 32,
        num_k_heads: int = 4,
        head_dim: int = 128,
        block_size: int = 64,
        sparsity_ratio: float = 0.2,
        kernel_size: int = 32,
        kernel_stride: int = 16,
        rtol: float = 1e-3,
        atol: float = 1e-3
    ) -> dict:
        """Run a single correctness test."""
        
        print(f"\n{'='*60}")
        print(f"Running NSA Correctness Test")
        print(f"seqlen: {seqlen}, heads: {num_q_heads}:{num_k_heads}, head_dim: {head_dim}")
        print(f"block_size: {block_size}, sparsity: {sparsity_ratio}")
        print(f"{'='*60}")
        
        # Setup test data
        q, k, v, compress_key, compress_value, intra_block_pe, cu_seqlens = self.setup_test_data(
            seqlen, num_q_heads, num_k_heads, head_dim, kernel_size, kernel_stride
        )
        
        # Compute topk indices
        print("Computing topk indices...")
        topk_idx = self.compute_topk_indices(
            q, k, v, compress_key, compress_value, intra_block_pe, cu_seqlens,
            kernel_size, kernel_stride, block_size, sparsity_ratio, head_dim
        )
        
        print(f"Topk indices shape: {topk_idx.shape}")
        print(f"Topk range: [{topk_idx.min().item()}, {topk_idx.max().item()}]")
        
        # Compute FFA attention
        print("Computing FFA attention...")
        num_group = num_q_heads // num_k_heads
        output_ffa = self.compute_ffa_attention(
            q, k, v, topk_idx, block_size, num_group, head_dim
        )
        # import pdb; pdb.set_trace()
        # print(output_ffa)
        
        # Compute NSA reference attention
        print("Computing NSA reference attention...")
        # output_nsa = self.compute_nsa_ref_attention(
        #     q, k, v, topk_idx, cu_seqlens, block_size, head_dim
        # )
        # print(output_nsa)
        
        # print(f"FFA output shape: {output_ffa.shape}")
        # print(f"NSA output shape: {output_nsa.shape}")

        output_sdpa = self.compute_sdpa_attention(
            q, k, v, topk_idx, block_size, num_group, head_dim
        )
        # print(output_sdpa)
        # breakpoint()
        
        # Compare outputs
        print("Comparing outputs...")
        comparison = self.compare_outputs(output_ffa, output_sdpa, rtol, atol)
        
        # Print results
        print(f"\nComparison Results:")
        print(f"Outputs close: {comparison['outputs_close']}")
        print(f"Max absolute diff: {comparison['max_abs_diff']:.6e}")
        print(f"Max relative diff: {comparison['max_rel_diff']:.6e}")
        print(f"Mean absolute diff: {comparison['mean_abs_diff']:.6e}")
        print(f"Mean relative diff: {comparison['mean_rel_diff']:.6e}")
        print(f"Tolerances used - rtol: {rtol}, atol: {atol}")
        
        return comparison
    
    def run_comprehensive_test(self) -> dict:
        """Run comprehensive correctness tests with different configurations."""
        
        test_configs = [
            # Small tests
            {"seqlen": 1024, "num_q_heads": 8, "num_k_heads": 8, "head_dim": 128, 
             "block_size": 64, "sparsity_ratio": 0.2},
            
            # Medium tests  
            {"seqlen": 4096, "num_q_heads": 32, "num_k_heads": 8, "head_dim": 128, 
             "block_size": 64, "sparsity_ratio": 0.15},
             
            # Different sparsity ratios
            {"seqlen": 1024, "num_q_heads": 16, "num_k_heads": 4, "head_dim": 128, 
             "block_size": 64, "sparsity_ratio": 0.3},
        ]
        
        results = []
        passed_tests = 0
        
        for i, config in enumerate(test_configs):
            print(f"\n{'#'*80}")
            print(f"Test {i+1}/{len(test_configs)}")
            print(f"{'#'*80}")
            
            try:
                result = self.run_single_test(**config)
                result["config"] = config
                result["test_passed"] = result["outputs_close"]
                results.append(result)
                
                if result["test_passed"]:
                    passed_tests += 1
                    print("✅ TEST PASSED")
                else:
                    print("❌ TEST FAILED")
                    
            except Exception as e:
                print(f"❌ TEST ERROR: {str(e)}")
                result = {"config": config, "test_passed": False, "error": str(e)}
                results.append(result)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {len(test_configs)}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {len(test_configs) - passed_tests}")
        print(f"Success rate: {100 * passed_tests / len(test_configs):.1f}%")
        
        return {
            "total_tests": len(test_configs),
            "passed_tests": passed_tests,
            "success_rate": 100 * passed_tests / len(test_configs),
            "detailed_results": results
        }


def main():
    """Main function to run NSA correctness tests."""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    # Initialize test suite
    test_suite = NSACorrectnessTest()
    
    # Run comprehensive tests
    results = test_suite.run_comprehensive_test()
    
    # Exit with appropriate code
    if results["success_rate"] == 100.0:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed ({results['success_rate']:.1f}% success rate)")
        sys.exit(1)


if __name__ == "__main__":
    main()