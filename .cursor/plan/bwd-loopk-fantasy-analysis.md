# BWD LoopK Fantasy Analysis: 339→534 TFLOPS Gap Root Cause

**分支**: `perf/bwd-loopk-fantasy-analysis` (基于 `feat/bwd-loopq-sparse-load`)
**场景**: S=topk=32K, nhq=128, nhk=1, hd=128, PackGQA, dense full mask, H100-80GB

---

## 完整实验数据 (2026-07-03 04:05)

| # | Config | TFLOPS | ms | Δms saved | gap% |
|---|---|---|---|---|---|
| 1 | **LoopK: baseline** | **339** | **518.7** | — | **0%** |
| 2 | **LoopQ: baseline** | **534** | **329.3** | — | **100%** |
| 3 | LoopK: light V load | 338 | 520.0 | -1.3 | -1% |
| 4 | LoopK: no dV store | 401 | 438.7 | 80.0 | 42% |
| 5 | LoopK: no dK store | 345 | 509.5 | 9.2 | 5% |
| 6 | LoopK: no dV MMA | 390 | 451.0 | 67.7 | 36% |
| 7 | LoopK: **no dV MMA+store** | **483** | **364.2** | **154.5** | **82%** |
| 8 | LoopK: no dK+dV store | 418 | 421.2 | 97.5 | 52% |
| 9 | LoopK: **no dV path + no dK store** | **509** | **345.6** | **173.1** | **91%** |
| 10 | LoopK: **skip all (V+dV+dK)** | **508** | **346.5** | **172.2** | **91%** |
| 11 | LoopK: lseU=1 | 338 | 519.9 | -1.2 | -1% |
| 12 | LoopK: M64N64 | 196 | 895.1 | — | — |
| 13 | LoopK: skip all + lseU1 | 513 | 342.7 | 176.0 | 93% |
| 14 | LoopK: **skip all + M64N128** | **404** | **435.5** | 83.2 | **44%** |
| 15 | LoopK: skip all + dS=2 stgV1 | 510 | 345.0 | 173.7 | 92% |
| 16 | LoopK: skip all + M64N128 lseU1 | 391 | 449.7 | 69.0 | 36% |

LoopQ 所有 skip 配置均为 533-537T（±3ms noise），证明 LoopQ 完全隐藏了 dV/dK 路径开销。

---

## 关键发现

### 1. dV pipeline 是主要瓶颈 (82% of gap)

`skip dV MMA + skip dV store` 合计 = **483T** (154.5ms saved, 82% of gap)

但单独 skip：
- skip dV MMA alone = 390T (67.7ms, 36%)
- skip dV store alone = 401T (80.0ms, 42%)
- 36% + 42% = 78% ≠ 82%

说明 dV MMA 和 dV store 有 **4% 交互效应**：两者在 pipeline 中串行（MMA → R2S → barrier → store → barrier），单独跳过一个只部分释放了 pipeline 压力。

### 2. V load 不是瓶颈 (0%)

skip V load = 0ms saving。V 的 TMA load 完全被 pipeline 隐藏。

### 3. dK store 小开销 (5%)

skip dK store alone = 9ms saving。但配合 dV 路径移除时几乎无贡献（#9 vs #10: 345.6 vs 346.5 ≈ noise）。

### 4. M64N128 对 LoopK 是**反优化** (−100T)

**关键发现**: skip all + M64N128 (404T) 比 skip all + M128N64 (508T) **慢 100T**！

原因：LoopK 的 outer loop 是 K tiles，inner loop 是 Q tiles。M128N64 的 M=128 意味着 Q 方向覆盖更多 tokens per inner tile，减少 inner 迭代次数。M64N128 反过来：M=64 使 Q 方向更细，增加 inner 迭代。

LoopQ 的 M64N128 高效是因为 LoopQ 的 outer=K (N 方向 128 更好), inner=Q (M=64 ok 因为 dV/dK 在 register 累加)。

**结论**: LoopK 的最优 tile 就是 M128N64，改 M64N128 反而更差。"结构性约束"不是瓶颈，而是正确的设计。

### 5. dS=2 和 lseU=1 影响极小 (~1-2%)

skip all + dS=2 (510T) ≈ skip all baseline (508T)
skip all + lseU=1 (513T) ≈ skip all baseline (508T)

---

## 根因总结

**LoopK vs LoopQ 的 189ms gap 分解**:

| Root Cause | ms saved | % of gap | 详细 |
|---|---|---|---|
| **dV pipeline (MMA+R2S+store)** | **154.5** | **82%** | LoopK 每个 inner iter 做 dV WGMMA → R2S copy → barrier → TMA reduce-add → barrier 完整 pipeline。LoopQ 只做 WGMMA (register 累加, zero_init=false)，pipeline 开销为零 |
| **dK store** | **9** | **5%** | LoopK 每个 inner iter 做 dK TMA reduce-add/atomicAdd。LoopQ dK 在 register 累加 |
| **dK+dV 交互** | **9** | **5%** | dK 和 dV 共用 union SMEM，串行写回 |
| **residual (barrier overhead)** | **17** | **9%** | 即使跳过所有 MMA/store/load，LoopK 的 barrier 乒乓架构仍有 ~17ms 固定开销 |
| **合计** | **189.4** | **100%** | |

**LoopK 的核心架构瓶颈**: dKV 在 SMEM 中累加 + 每个 inner iter 通过 producer warp atomicAdd 写回 global memory。这导致：
1. 每次 inner iter 都有 consumer→producer→consumer 的 barrier 同步开销
2. atomicAdd/TMA reduce-add 的 global memory traffic 线性增长
3. SMEM 占用大（dkacc+dvacc），限制了 tile size 和 staging 选择

**LoopQ 为什么不受影响**: dV/dK 在 register 中 `zero_init=false` 累加，inner loop 结束后一次性写出，WGMMA 被 producer load latency 完全隐藏。

---

## 下一步 (2026-07-03 04:05)

### P7: 精细 pipeline 分解（需要新 skip flag）

目标：分离 dV pipeline 的各环节开销：
1. `skip_dv_r2s`: 跳过 consumer R2S copy + barrier 但保留 WGMMA → 验证 WGMMA 是否真的被隐藏
2. `skip_dv_barrier`: 只跳过 barrier 同步 → 量化 barrier 开销

预期：如果 skip_dv_r2s ≈ skip_dv_mma+store (154ms)，则 barrier+R2S 是主导因素。

### P8: NCU profiling (需要 MPS-free 窗口)

收集 baseline vs skip_all vs LoopQ 的 RED traffic / tensor pipe util / barrier stall ratio。
