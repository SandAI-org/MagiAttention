# BWD InnerLoopK Debug Analysis: 339→534 TFLOPS Gap Root Cause

**分支**: `perf/bwd-loopk-debug-analysis` (基于 `feat/bwd-loopq-sparse-load`)
**场景**: S=topk=32K, nhq=128, nhk=1, hd=128, PackGQA, dense full mask, H100-80GB

**术语约定**:
- **InnerLoopK** (SwapBwdQKLoop=true): outer loop 遍历 Q tiles, inner loop 遍历 K tiles
- **InnerLoopQ** (SwapBwdQKLoop=false): outer loop 遍历 K tiles, inner loop 遍历 Q tiles
- 代码中 `mma_with_loop_k` / `mma_with_loop_q` 需要重命名为 `mma_with_inner_loop_k` / `mma_with_inner_loop_q`（P-rename）

---

## 精确对称性分析 (2026-07-03 12:15)

### Inner loop 每次迭代的操作对比

| 操作 | InnerLoopQ (per Q tile) | InnerLoopK (per K tile) | 对称？ |
|---|---|---|---|
| **Inner load 1** | Q (kBlockN × hd) | K (kBlockN × hd) | ✓ Q↔K |
| **Inner load 2** | dO (kBlockN × hd) | V (kBlockN × hd) | ✓ dO↔V |
| **MMA: S=QK^T** | ✓ | ✓ | ✓ |
| **MMA: P=softmax(S)** | ✓ | ✓ | ✓ |
| **MMA: dS** | ✓ | ✓ | ✓ |
| **MMA: dV=P^T×dO** | ✓ (reg accum, zero_init=false) | ✓ (zero_init=true → SMEM) | ✓ MMA 本身相同 |
| **MMA: dK=dS^T×Q** | ✓ (reg accum, zero_init=false) | ✓ (zero_init=true → SMEM) | ✓ MMA 本身相同 |
| **MMA: dQ=dS×K** | ✓ (zero_init=true → atomicAdd) | ✓ (reg accum) | ✓ 对称交换 |
| **Inner store: dQ** | register → `atomicAdd` global (直接) | register 累加，loop 结束一次性写 | ✗ |
| **Inner store: dK** | register 累加，loop 结束一次性写 | register → R2S → SMEM → barrier → producer TMA → barrier | ✗ **关键不对称** |
| **Inner store: dV** | register 累加，loop 结束一次性写 | register → R2S → SMEM → barrier → producer TMA → barrier | ✗ **额外不对称** |

### 关键不对称

**1. Inner store 数量不同**:
- InnerLoopQ: **1 个** inner atomicAdd (dQ)
- InnerLoopK: **2 个** inner SMEM pipeline (dK + dV)

**2. Inner store 机制不同** (这是性能差异的核心!):
- InnerLoopQ 的 dQ: register → **直接** `atomicAdd` global（L3117-3127，无 SMEM、无 barrier）
- InnerLoopK 的 dK/dV: register → R2S → SMEM → barrier → producer TMA reduce-add → barrier
  - 每个 inner iter 需要 **两轮完整 barrier 同步**（dV + dK）
  - Consumer 被 barrier 阻塞等待 producer store warp 完成 TMA

**3. 真正要消除的差异**:
- **dV 的 inner store**: InnerLoopQ 不需要 dV inner store（register 累加），InnerLoopK 必须每 iter 写回 → 这是额外的
- **dK 的 store 机制**: InnerLoopQ 的 dQ 用直接 `atomicAdd`，InnerLoopK 的 dK 用 SMEM pipeline → 机制不对称

### `DkvaccBypassSmem=1`: 真正的对称路径

代码中已有 `DkvaccBypassSmem` 模板参数（L84），启用后：
- InnerLoopK 的 dK/dV 改用 register → **直接** `atomicAdd` global（L3735-3762）
- 跟 InnerLoopQ 的 dQ 路径**完全对称**
- 消除了 SMEM pipeline 的 R2S + barrier 同步开销
- 释放了 smem_dkacc + smem_dvacc 的 SMEM 空间
- 环境变量: `MAGI_ATTENTION_FFA_BWD_DKVACC_BYPASS=1`

---

## 之前分析的错误纠正

❌ **"dV MMA 占 36% gap"** — InnerLoopQ 也做 dV WGMMA 但零开销。问题不是 MMA 本身。
❌ **"LoopK outer=K, inner=Q"** — 完全反了。InnerLoopK = inner loop K, outer loop Q。
❌ **"结构性约束不可逾越"** — bypass=1 释放 dKV SMEM 后，M64N128 和 dS=2 可能可行。
❌ **gap 分解中不应包含 dV MMA** — 因为这个 MMA 在 InnerLoopQ 中也存在且无开销。

---

## Phase 4 已有数据 (2026-07-02~03)

| # | Config | TFLOPS | ms |
|---|---|---|---|
| 1 | InnerLoopK: baseline (SMEM pipeline) | 339 | 518.7 |
| 2 | InnerLoopQ: baseline (reg accum) | 534 | 329.3 |
| 3 | InnerLoopK: skip dV store (TMA only) | 401 | 438.7 |
| 4 | InnerLoopK: skip dK store | 345 | 509.5 |
| 5 | InnerLoopK: skip dV MMA | 390 | 451.0 |
| 6 | InnerLoopK: skip dV MMA+store | 483 | 364.2 |
| 7 | InnerLoopK: skip all | 508 | 346.5 |
| 8 | InnerLoopK: skip all + M64N128 | 404 | 435.5 |
| 9 | InnerLoopK: skip all + dS=2 | 510 | 345.0 |
| 10 | InnerLoopK: skip all + lseU1 | 513 | 342.7 |

注意：#3-#6 的 skip 只跳过了部分操作（TMA write / WGMMA），R2S + barrier 同步仍在运行，所以不是完整的"路径消除"。

---

## 修正后计划 (2026-07-03 12:15)

> 以下 P5/P6/bypass 计划已被 P5-v2 取代（DkvaccBypassSmem=1 有 misaligned address bug，bypass 系列暂停）。

---

## Phase 4 已有数据分析 (2026-07-03 14:00)

### 已有数据（bench_phase4_final.log）

| # | Config | TFLOPS | ms | 备注 |
|---|---|---|---|---|
| 1 | InnerLoopK baseline | 334 | 526 | |
| 2 | InnerLoopQ baseline | 526 | 335 | |
| 3 | InnerLoopK: light V load | 333 | 529 | V load 不是瓶颈（L2 cached 跟 scatter 一样快） |
| 4 | InnerLoopK: no dV store | 396 | 444 | +62T = dV TMA store 贡献 |
| 5 | InnerLoopK: no dK store | 345 | 511 | +11T = dK TMA store 贡献小 |
| 6 | InnerLoopK: no dK+dV store | 413 | 426 | +79T |
| 7 | InnerLoopK: no dV MMA | 391 | 450 | **不公平**：LoopQ 也做 dV MMA |
| 8 | InnerLoopK: no dV MMA+store | 479 | 367 | **不公平** |
| 9 | InnerLoopK: skip all | 504 | 349 | **不公平**（含 skip dV MMA） |

### 关键观察

1. `no dV store` (+62T) vs `no dK store` (+11T): dV store 开销远大于 dK store
   - 原因：dV 和 dK 共用 SMEM (union)，dV 先写 → 必须等 producer TMA 完成 → 才能写 dK
   - dV store 的等待时间 = producer TMA reduce-add latency
2. `light V load` (+0T): V load 本身不是瓶颈（可能被 pipeline 完全隐藏）
3. `skip all` 含 `SKIP_DV_MMA`，**不公平**！InnerLoopQ 也做 dV MMA
4. **缺少关键实验**：`SKIP_V_LOAD + SKIP_DV_WRITEBACK`（消除 V 和 dV 的额外开销，但保留 dV MMA）

### 正确的对称性分析

**InnerLoopQ 每次 inner iteration**:
- Load: Q, dO
- MMA: S=Q@K^T, P=softmax(S), dS, dV=P^T@dO (reg accum), dK=dS^T@Q (reg accum), **dQ=dS@K → atomicAdd**
- Store: dQ (1 个 inner store，直接 atomicAdd)

**InnerLoopK 每次 inner iteration**:
- Load: K, **V** ← EXTRA（InnerLoopQ 的 V 在 outer loop）
- MMA: S=Q@K^T, P=softmax(S), dS, **dV=P^T@dO → R2S+barrier+TMA**, **dK=dS^T@Q → R2S+barrier+TMA**, dQ (reg accum)
- Store: **dK** + **dV** (2 个 inner store，R2S+barrier+TMA reduce)

**InnerLoopK 相对 InnerLoopQ 的额外开销**:
1. **V load** (inner loop): InnerLoopQ 没有 V inner load
2. **dV writeback pipeline** (R2S+barrier+TMA): InnerLoopQ 的 dV 在 register 累加，不 per-iter store
3. **dK store 机制差异**: atomicAdd (InnerLoopQ dQ) vs R2S+barrier+TMA (InnerLoopK dK)

---

## P5-v2: 正确的对称实验 (2026-07-03 14:00) [ACTIVE]

### 实验设计

目标：逐步消除 InnerLoopK 的额外开销，观察 TFLOPS 是否逼近 InnerLoopQ 的 526T。

| # | Config | 消除的额外开销 | 保留 | 预期 |
|---|---|---|---|---|
| A | `SKIP_DV_WRITEBACK=1` | dV writeback (R2S+barrier+TMA) | V load, dK store, 全部 MMA | ~400T? |
| B | `SKIP_V_LOAD=1 + SKIP_DV_WRITEBACK=1` | V load + dV writeback | dK store, 全部 MMA | **关键**：若≈526T → gap = V+dV |
| C | `SKIP_V_LOAD=1 + SKIP_DV_WRITEBACK=1 + SKIP_DK_STORE=1` | V load + dV writeback + dK store | 只剩 MMA | 上界 |

- Config B 是**核心对称实验**：InnerLoopK 仅做 K load + dK store + 全部 MMA，对称于 InnerLoopQ 的 Q load + dQ store + 全部 MMA
- Config B vs Config A → 量化 V load 的真实开销（可能被 pipeline 隐藏）
- Config C vs Config B → 量化 dK store 的开销（R2S+barrier+TMA 机制差异）

### 后续（取决于 B 的结果）

- 若 B ≈ 526T：gap 完全由 V load + dV writeback 解释。优化方向 = 减少 dV writeback 开销
- 若 B < 526T：残余 gap = dK store 的 R2S+barrier+TMA vs dQ 的 atomicAdd 差异 → 需进一步分析
- 若 B > 526T（不太可能）：InnerLoopK 在去除额外开销后更快 → 需要检查实验设计

### P-rename (低优先级): 代码重命名

`mma_with_loop_k` → `mma_with_inner_loop_k`，`mma_with_loop_q` → `mma_with_inner_loop_q`，以及相关 load/store 函数名。避免术语歧义。

---

## 更新: 2026-07-03 19:38 — SVW barrier fix + 完整 P5-v2 benchmark

### SVW (PerfPerfDebugSkipDvWriteback) Barrier Race Fix

**Bug**: `PerfDebugSkipDvWriteback` 在 SparseLoad 非 scatter 模式下 kernel hang（IndexSparse scatter 模式不受影响）。

**根因**: 非 scatter 模式的 `store_body` 使用 `iterate_range` 循环，每个 inner block 调 `store_dV()` + `store_dK()`。
Warp 1 执行 `store_dV`（PerfDebugSkipDvWriteback 时只做 `arrive(dKEmpty)` 秒退），warp 2 执行 `store_dK`（`sync(dKFull)` 阻塞等 consumer）。
Warp 1 与 warp 2 **独立执行**，warp 1 因为无阻塞点而疯狂领先 → 在 `dKEmpty` barrier 上堆积过量 arrive → 5×32=160 触发 barrier spurious fire → consumer barrier 相位错乱 → deadlock。

**修复** (`mainloop_bwd_sm90_tma_gmma_ws.hpp`):
1. **Producer `store_dV`**: 移除 `PerfDebugSkipDvWriteback` 的 early return，保留 `sync(dVFull)` 阻塞（防止 warp 1 racing），跳过 TMA store
2. **Consumer dV R2S**: `sync(dVEmpty)` + `arrive(dVFull)` 始终执行（跳过中间 R2S copy），保持 barrier 握手

### 完整 benchmark 结果 (S=topk=32768, GPU0 独占)

| # | Config | TFLOPS | ms | gap frac | 备注 |
|---|--------|--------|------|---------|------|
| 1 | InnerLoopK: baseline | 334 | 526.3 | 0% | |
| 2 | InnerLoopQ: baseline | 530 | 331.8 | 100% | |
| 3 | LoopK: light V load | 335 | 525.0 | +0.6% | V load 被 pipeline 完全隐藏 |
| 4 | LoopQ: light V load | 525 | 334.9 | +98% | |
| 5 | LoopK: no dV store | 396 | 444.3 | +42% | dV TMA store = 42% gap |
| 6 | LoopQ: no dV store | 529 | 332.4 | ~100% | LoopQ dV store 零开销 |
| 7 | LoopK: no dK store | 345 | 509.8 | +8.5% | dK TMA store = 8.5% gap |
| 8 | LoopQ: no dK store | 532 | 330.6 | ~100% | |
| 9 | LoopK: no dV MMA | 386 | 455.8 | +36% | ⚠️ 不公平比较 |
| 10 | LoopQ: no dV MMA | 261 | 673.4 | -76% | ⚠️ LoopQ 严重退化 |
| 11 | LoopK: no dK+dV store | 414 | 425.3 | +52% | |
| 12 | LoopQ: no dK+dV store | 518 | 339.8 | +96% | |
| 13 | LoopK: skip all (含 dV MMA) | 508 | 346.1 | +93% | ⚠️ 不公平 |
| 14 | LoopQ: skip all (含 dV MMA) | 260 | 675.6 | -77% | ⚠️ LoopQ 严重退化 |
| 15 | LoopK: lseU=1 | 335 | 525.2 | +0.5% | |
| 16 | LoopK: M64N64 | 196 | 896.1 | -190% | 严重退化 |
| 17 | LoopK: skip all + lseU1 | 511 | 344.0 | +94% | |
| 18 | LoopK: skip all + M64N128 | 402 | 437.5 | +46% | |
| 19 | LoopK: skip all + dS=2 stgV1 | 507 | 346.8 | +92% | |
| 20 | LoopK: skip all + M64N128 lseU1 | 390 | 451.2 | +39% | |
| 21 | LoopK: no dV MMA+store | 478 | 367.9 | +82% | |
| 22 | LoopK: no dV path + no dK store | 363 | 484.1 | +22% | |
| **23** | **InnerLoopK: no dV writeback (SVW)** | **540** | **325.5** | **+103%** | ✅ **超过 LoopQ!** 之前 hang |
| **24** | **InnerLoopK: symmetric (SVL+SVW)** | **542** | **324.4** | **+104%** | ✅ V load+dV writeback 消除 |
| **25** | **InnerLoopK: symmetric + no dK store** | **572** | **307.3** | **+113%** | ✅ 最快配置 |

### 关键结论

1. **SVW fix 成功**: 之前 hang 的 3 个配置全部跑通 (#23/#24/#25)
2. **InnerLoopK: no dV writeback (540T) > LoopQ baseline (530T)**: 仅消除 dV writeback pipeline 就超过了 LoopQ！
   - dV writeback pipeline (R2S+barrier+TMA) = 整个 gap 的 **103%**
   - 说明 InnerLoopK 在消除 dV writeback 后底层效率可能**微优于** LoopQ
3. **V load 无影响 (#3 vs #24)**: SVL 仅提升 2T，被 pipeline 完全隐藏
4. **dK store 机制差异 (#24→#25)**: +30T (+6% gap)，R2S+barrier+TMA vs atomicAdd 的差异
5. **LoopQ skip dV MMA 退化 (#10/#14)**: LoopQ 移除 dV MMA 后严重退化到 261T — 说明 dV MMA 在 LoopQ 中对 pipeline balance 至关重要，不能简单去掉

### 优化方向总结

| 优化点 | 潜在 TFLOPS 提升 | gap 占比 | 方向 |
|--------|----------------|---------|------|
| **dV writeback pipeline** | +206T (334→540) | ~103% | SMEM pipeline 改造/bypass |
| dK store 机制 | +30T (540→572) | ~6% | atomicAdd 替代 / bypass |
| V load | ~0T | ~0% | 已被隐藏，无需优化 |

**核心优化目标**: 消除或大幅减少 dV writeback 的 R2S+barrier+TMA pipeline 开销。

---

## 深度分析: 为什么 dV writeback 占 103% 而 dK store 仅占 8.5%？ (2026-07-03 19:56)

### Consumer 每次 inner iteration 的精确执行顺序

```
MMA3: dV = P^T @ dO        (wg_wait=-1, 不等待, dV WGMMA 异步 issue)
MMA4: dK = dS^T @ Q        (wg_wait=1 → dV MMA 完成 → dV 寄存器结果 ready)
─── dV R2S ───
  sync(dVEmpty)             ← consumer 等 SMEM 空闲 (等 producer 上一轮 dK TMA 完成)
  R2S copy: reg→SMEM
  arrive(dVFull)            ← 通知 producer: dV 数据在 SMEM 里了
─── dQ MMA ───
MMA5: dQ = dS @ K           (wg_wait=1 → dK MMA 完成 → dK 寄存器结果 ready)
                             (只是 issue WGMMA 指令, 瞬间完成)
─── dK R2S ───
  sync(dKEmpty)             ← consumer 等 SMEM 空闲 (等 producer dV TMA 完成!)
  R2S copy: reg→SMEM
  arrive(dKFull)            ← 通知 producer: dK 数据在 SMEM 里了
─── next iteration ───
  pipeline read → softmax → dS → MMA3(dV) → MMA4(dK)
  sync(dVEmpty)             ← consumer 等 SMEM 空闲 (等 producer dK TMA 完成)
```

### Barrier 链 (union 模式, dV 和 dK 共用同一块 SMEM)

```
consumer arrive(dVFull)
  ↓ producer warp 1: sync(dVFull) → dV TMA reduce-add → arrive(dKEmpty)
consumer sync(dKEmpty)  ← 在此等待 dV TMA!
  ↓
consumer arrive(dKFull)
  ↓ producer warp 2: sync(dKFull) → dK TMA reduce-add → arrive(dVEmpty)
consumer sync(dVEmpty)  ← 在此等待 dK TMA! (但前面有大量 MMA 可以 overlap)
```

### 根本原因: Pipeline Overlap 的极度不对称

**dV TMA 延迟 = 100% 在 critical path 上 (零 overlap)**:
- consumer arrive(dVFull) → 下一步是 MMA5(dQ) **issue** (只是发射 WGMMA 指令，纳秒级)
- 然后 **立即** sync(dKEmpty) → 必须等 producer 完成 dV TMA
- **consumer 在 dV TMA 期间完全无事可做**

**dK TMA 延迟 = 完全被隐藏 (100% overlap)**:
- consumer arrive(dKFull) → 进入下一迭代
- 下一迭代做: pipeline read → softmax → dS → MMA3(dV) → MMA4(dK) (**4+ 个 WGMMA**)
- 然后才 sync(dVEmpty) → 此时 dK TMA 早已完成
- **consumer 有 100+μs 的计算完全覆盖 dK TMA**

```
时间线:
Consumer: ─dV_R2S─[arrive dVFull]─issue_dQ_MMA─[BLOCKED on dKEmpty]──────────dK_R2S─[arrive dKFull]─MMA1─MMA2─MMA3─MMA4─...
Producer:                         [sync dVFull]───dV_TMA───[arrive dKEmpty]                         [sync dKFull]─dK_TMA─[arrive dVEmpty]
                                                ^^^^^^^^^^^^                                                      ^^^^^^^^^^^^^^
                                                关键瓶颈: 零overlap                                                完全被 4个MMA 隐藏
```

### 实验数据与分析完全一致

| 实验 | 做了什么 | TFLOPS | ms 节省 | 解释 |
|------|---------|--------|---------|------|
| SKIP_DV_STORE | 仅跳过 dV TMA write (R2S+barrier 仍在) | 396 (+62T) | -82ms | 移除了 critical path 上的 TMA latency |
| SKIP_DK_STORE | 仅跳过 dK TMA write (R2S+barrier 仍在) | 345 (+11T) | -16ms | 移除了已被 overlap 的 TMA，效果微小 |
| SKIP_DV_WRITEBACK | 消除整个 dV pipeline (R2S+barrier+TMA) | 540 (+206T) | -201ms | 移除了 critical path 上全部死等待 |

### "让 dV 先 store" 有用吗？— 不行，只是交换了瓶颈

如果交换 R2S 顺序为 dK first → dV second：
- dK TMA 变成 critical path（consumer dK R2S 后立即 dV R2S，无 overlap）
- dV TMA 变成 hidden（consumer 有 4+ MMA 覆盖）
- **总延迟不变**，只是 dK 和 dV 的角色交换

### 已有的 producer 专属 warp 已经是最优方案吗？

当前实现**已经**有 producer 专属 warp (WG0 warp1=dV, warp2=dK)。问题不在于"谁来做 TMA"，而在于 **union SMEM 强制串行化**：dV 和 dK 共用同一块 SMEM，consumer 必须等 dV TMA 读完 SMEM 后才能写 dK R2S。即使 TMA 在 producer warp 上异步执行，consumer 仍然被 barrier 阻塞。

---

## 优化方向 (按可行性排序) (2026-07-03 19:56)

### O1: Ununion SMEM — 消除 dV/dK 串行化 [首选]

**原理**: dV 和 dK 使用独立 SMEM buffer → dV TMA 和 dK R2S 可以并行 → consumer 不再被 dV TMA 阻塞

```
当前 (union):   dV_R2S → [wait dV TMA] → dK_R2S → [wait dK TMA, hidden]
ununion:        dV_R2S → dK_R2S         (producer 两个 TMA 独立 pipeline, 都被 MMA 隐藏)
```

**SMEM 预算**: M128N64 baseline = 198KB, ununion 需要 +32KB (dkacc+dvacc 各 16KB) = 230KB > 228KB H100 limit

**可行条件**: 释放 ≥2KB SMEM。方案：
- `stgV=1` (减少 V pipeline 一个 stage): 节省 ~16KB → 230-16=214KB ✓
- `SKIP_V_LOAD` debug: 节省 ~32KB → 直接可行
- `lseU=1`: 复用 dkacc 的前 1KB 放 LSE → 节省 ~1KB (不够)

**预期收益**: 消除 dV TMA 的 critical path 阻塞 (~82ms)，可能达到 ~420-460T

### O2: DkvaccBypassSmem — 全 atomicAdd 消除 SMEM pipeline [最高收益]

**原理**: dK/dV 直接从 register atomicAdd 到 GMEM。无 R2S、无 barrier、无 producer TMA。与 InnerLoopQ 的 dQ 路径**完全对称**。

**已知问题**: misaligned address bug（之前尝试时遇到）。需定位和修复。

**预期收益**: 完全消除 SMEM pipeline 开销 → 理论上达到 InnerLoopQ 水平 (~530T)

### O3: 将 dK R2S 提前到 MMA5(dQ) 之前 — 改善 overlap [低改动]

**原理**: 交换 dV 和 dK 的 R2S 顺序，但同时**将 dK R2S 移到 MMA5(dQ) issue 之后**：

```
当前:    MMA4(dK) → dV_R2S → MMA5(dQ)_issue → dK_R2S → [next iter]
优化:    MMA4(dK) → dK_R2S → MMA5(dQ)_issue → dV_R2S → [next iter MMA1-MMA4]
```

这样 dV TMA 被 next-iter 的 4+ MMA 隐藏，dK TMA 被 MMA5 发射后的... 等等，dK TMA 依然没有 overlap。

**结论**: 无论怎么重排，union SMEM 下总有一个 TMA 在 critical path 上。**必须 ununion 才能根本解决**。

### O4: 减少 per-iter TMA 次数 — 跨 iteration 累加 [架构改动]

**原理**: 不在每个 inner iter 都做 TMA reduce-add，而是在 SMEM 中多个 iter 累加后一次性 TMA。需要扩大 SMEM accumulator 为 fp32 多 stage。

**复杂度**: 高，需要重新设计 SMEM layout + 累加逻辑。但 dKV union SMEM 本身已经是 fp32 accumulator。

### 优先级

1. **O1 (ununion + stgV=1)**: 改动小、SMEM 刚好够、收益明确。先实验。
2. **O2 (bypass)**: 收益最大但有 bug 要修。O1 不够时再推进。
3. **O3**: 效果有限（union 下无法根本解决），但代码改动最小，可作为辅助。
4. **O4**: 复杂度高，作为后续长期方案。

### 下一步执行

1. **实验 O1**: 设置 `MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC=1` + `MAGI_ATTENTION_FFA_BWD_STAGES_V=1`
   - 验证 SMEM ≤ 228KB
   - Precompile + bench S=32K → 对比 baseline 334T
2. **若 O1 SMEM 超限**: 同时加 `SKIP_V_LOAD=1` 释放 SMEM → 验证 ununion 的纯收益
3. **分析 O2 bypass bug**: 定位 misaligned address 原因

---

## O1 实验结果 (2026-07-03 20:36)

### SMEM 验证
- `ununion_only (stgV=2)`: 230KB > 228KB → **CUDA error: invalid argument** (SMEM 超限 2KB)
- `ununion+stgV1`: **214KB** → fits ✓ (vs baseline 198KB, +16KB)
- 寄存器: 168 (与 baseline 相同，无退化)

### Benchmark 数据 (S=32K, NHQ=128, NHK=1, PackGQA)

| Config                       | TFLOPS | ms    | Δ vs LoopK | gap% |
|------------------------------|--------|-------|------------|------|
| **LoopK baseline**           | 334.3  | 526.3 | —          | —    |
| LoopK ununion+stgV1          | 368.6  | 477.2 | +34.3T     | 18%  |
| LoopK ununion+stgV1+SVL      | 376.7  | 467.0 | +42.4T     | 23%  |
| LoopK SVW (no dV writeback)  | 534.8  | 328.9 | +200.5T    | 106% |
| LoopK ununion+stgV1+SVW      | 533.3  | 329.9 | +199.0T    | 106% |
| LoopK ununion+stgV1+SVL+SVW  | 535.9  | 328.3 | +201.6T    | 107% |
| **LoopQ baseline**           | 522.7  | 336.5 | ref        | —    |

gap% = Δ / (LoopQ - LoopK) = Δ / 188.4T

### 关键发现

1. **LoopK+SVW (534.8T) > LoopQ (522.7T)**: 当 dV writeback 完全移除后 LoopK **反超** LoopQ 12T。
   → **100% gap 来自 dV writeback pipeline**
2. **Ununion 单独收益有限**: +34.3T (18% of gap)。虽然 ununion 解除了 dV/dK SMEM 串行化，
   但 dV 的 R2S + TMA store + barrier sync 总延迟仍在 critical path 上。
3. **SVW 激活后 ununion 无额外收益**: SVW+ununion (533.3) ≈ SVW (534.8) → 当 dV writeback 不存在时，
   union/ununion 无差别。
4. **代价**: ununion 需要 stgV=1 → SMEM 214KB (vs 198KB)，V pipeline 深度从 2 降为 1。

### 结论

O1 (ununion) 是一个 **incremental** 优化 (+34T, 10%)，可以落地但不能根本解决问题。
根本瓶颈是 dV writeback pipeline 本身（R2S + barrier + TMA store）。

### 下一步: O2 (DkvaccBypassSmem)

需要调查 misaligned address bug，如果 bypass 能工作：
- 完全消除 dV SMEM pipeline (无 R2S, 无 barrier, 无 TMA store)
- register → atomicAdd(GMEM) 直接写回，与 LoopQ 的 dQ 路径对称
- 预期: 接近 LoopK+SVW (534T) 的性能

---

## O2 实验结果: DkvaccBypassSmem (2026-07-03 21:00)

### Bug 修复: misaligned address

**Root cause**: 非 scatter 路径使用 `recast<float4>` 做 atomicAdd，
但 `r2s_thr_copy_dKVaccum.partition_D(gdKaccum)` 的 GMEM 分区布局与 WGMMA accumulator
的 thread-value 布局不匹配 → 连续 4 个逻辑元素在 GMEM 中非连续 → float4 指针未对齐。

**Fix**: 将非 scatter 路径改为 scalar atomicAdd (`atomicAdd(float*, float)`)。
**验证**: S=256/1024 numerically correct (dK rel diff < 0.1%, dV exact match)。

### Benchmark: Bypass 性能 (S=32K)

| Config             | TFLOPS | ms      | 对比     |
|--------------------|--------|---------|----------|
| LoopK baseline     | 333.4  | 527.6   | —        |
| **LoopK bypass**   | **161.3** | **1090.9** | **-51.6%** |
| LoopK SVW (ceiling)| 534.8  | 328.9   | +60%     |

**结论: Bypass 对 dense 模式完全不可行。**

- TMA reduce-add: 每 row 1 次操作 (128 floats bulk)
- scalar atomicAdd: 每 row 128 次操作 (1 float each)
- float4 atomicAdd: CUDA 无原生 float4 atomic → 仍然 4× scalar = 32 ops/row，与 scalar 等效
- AtomicAdd 总开销 762ms >> SMEM pipeline 199ms (3.8×)

**Bypass 仅适用于 scatter (IndexSparse) 路径**，dense 路径应继续使用 SMEM + TMA pipeline。

---

## dS_stage=2 可行性检查 (2026-07-03 21:05)

| Config                    | SMEM     | 状态      |
|---------------------------|----------|-----------|
| baseline (stgV=2, stds=1) | 198KB    | ✓ fits    |
| dSstg2_only               | 230KB    | ✗ > 228KB |
| ununion+stgV1+dSstg2      | **246KB** | ✗ > 228KB |
| ununion+stgV1+dSstg2+SVL  | ~230KB   | ✗ > 228KB |

**结论: dS_stage=2 对 InnerLoopK 不可行** — 额外 32KB SMEM 超出 H100 228KB 限制。

LoopQ 能用 dSstg2 (199KB) 是因为 LoopQ 不需要 dKV SMEM accumulator（用 atomicAdd），
所以有 ~30KB 额外空间。LoopK 的 dV+dK SMEM accumulators 占用了这些空间。

---

## 综合结论 (2026-07-03 21:10)

### LoopK-LoopQ 性能 gap 完整分析

| 优化方向                        | TFLOPS | Δ     | gap%  | 可行性      |
|---------------------------------|--------|-------|-------|-------------|
| LoopK baseline                  | 334    | —     | —     | ✓ current   |
| **O1: ununion+stgV1**           | **368** | **+34** | **18%** | **✓ 可落地** |
| O2: bypass (scalar atomicAdd)   | 161    | -173  | —     | ✗ 太慢      |
| dS_stage=2                      | —      | —     | —     | ✗ SMEM > 228KB |
| LoopK + SVW (理论上限)           | 535    | +201  | 107%  | debug     |
| **LoopQ baseline**              | **523** | ref   | —     | ✓ current   |

### 根本原因

LoopK 的 188T gap (334→523) 100% 来自 **dV writeback pipeline** (R2S + barrier + TMA store)。
这是 LoopK 固有的架构特征：
- LoopK inner loop 遍历 Q，每轮产生 partial dV/dK 需要 reduce-add 到 GMEM
- 需要 SMEM accumulator + producer/consumer barrier 协议 + TMA reduce-add
- dV 和 dK 的 SMEM buffer union → 串行化 → dV TMA 100% 在 critical path 上

LoopQ 没有这个问题：
- LoopQ inner loop 产生的 partial dQ 直接 atomicAdd 到 GMEM（无 SMEM pipeline）
- dK/dV 在 outer loop 累加，只在 outer loop 结束时写一次

### 可落地优化

1. **O1 (ununion+stgV1)**: +34T (+10%)，SMEM 214KB → fits 228KB。
   - 消除 dV/dK SMEM 串行化 → dV TMA 和 dK R2S 可以部分并行
   - 代价：V pipeline 深度从 2 降为 1（stgV=1）
   - **建议：作为默认优化合入**

2. **进一步优化方向**：
   - 在 ununion 基础上做 producer/consumer 重排序（利用独立 SMEM buffer 的并行性）
   - 探索减少 dV TMA 频率（跨 iteration 累加，但需要 fp32 SMEM 扩大）
   - 调整 kBlockM/kBlockN tile 大小（如 M64N128 → 需要解决 260KB SMEM 问题）

### 下一步

1. 将 O1 结果整合到 `bench_sparse_analysis.py` 的 phase 4 实验中
2. 将 O2 bypass misalignment fix 提交（正确性修复，不影响性能）

---

## P5-v4: dV/dK 对称性验证 (2026-07-03 22:26)

### 目标
验证 ununion 后 dV 和 dK writeback pipeline 开销是否对称。

### 新增实现
- `PerfDebugSkipDkWriteback` — 对称于 `PerfDebugSkipDvWriteback`，跳过 dK 的 R2S copy + softmax_scale + fence，保留 barrier 握手
- 环境变量: `MAGI_ATTENTION_FFA_BWD_SKIP_DK_WRITEBACK`
- 代码: `mainloop_bwd_sm90_tma_gmma_ws.hpp`, `flash_bwd_launch_template.h`, `bwd_inst_template.jinja`, `_flex_flash_attn_jit.py`

### 基准数据 (S=topk=32K, nhq=128, nhk=1, hd=128, bf16, H100)

| Config | TFLOPS | ms | vs O1 |
|--------|--------|----|-------|
| O1 baseline (ununion+stgV1) | 358 | 490.9 | — |
| O1 + SVL (skip V load) | 362 | 485.7 | +4T |
| **O1 + SVW (skip dV writeback)** | **502** | **350.3** | **+144T** |
| **O1 + SKW (skip dK writeback)** | **339** | **518.8** | **−19T** |
| O1 + SVS (skip dV store) | 365 | 481.6 | +7T |
| O1 + SKS (skip dK store) | 384 | 458.1 | +26T |
| **O1 + SVW + SKW (skip both wb)** | **602** | **292.0** | **+244T** |
| LoopQ baseline (参照) | 499 | 352.6 | — |

### 核心发现：强烈非对称

1. **dV writeback 是绝对瓶颈**: 跳过 dV writeback → +144T（从 358→502T，直接追平 LoopQ）
2. **dK writeback 不在 critical path**: 跳过 dK writeback → **-19T** 性能反降！
   - 原因：dK 的 R2S 和 TMA store 被 dV writeback pipeline 完全遮盖（overlap）
   - 跳过 dK R2S 后，producer store_dK 收到 dKFull 但 SMEM 无有效数据 → TMA store 写入 garbage → 没有节省时间反而可能打乱 pipeline 节奏
3. **Store-only 级别反转**: SVS(dV store skip)=+7T，SKS(dK store skip)=+26T
   - dV TMA store 在 critical path 上，跳过只省了 TMA 本身（小部分），但 R2S+barrier 仍在
   - dK TMA store 本来被 overlap，跳过后释放了 TMA unit → 让 dV TMA 更快启动 → 额外收益
4. **同时跳过 SVW+SKW → 602T > LoopQ baseline 499T**
   - 说明 dV+dK writeback 合计 = 198.9ms = 40.5% of kernel time
   - 超过 LoopQ 是因为 LoopK 的 dQ atomicAdd 比 LoopQ 的 dQ TMA 更轻量

### 结论

dV/dK writeback pipeline 完全非对称：
- **dV writeback**: critical path 上，串行化瓶颈 → 优化价值极高
- **dK writeback**: 被 dV pipeline 遮盖 → 优化无直接收益（但释放资源间接有益）

优化优先级：
1. ✅ O1 (ununion): 已验证 +38T（部分解除 dV/dK 串行化）
2. **高优**: 缩短 dV R2S 延迟（减少 R2S copy 数量或使用更快的 copy 指令）
3. **高优**: 减少 dV barrier 等待（调整 producer/consumer 重叠策略）
4. **低优**: dK 侧优化（目前完全被 overlap，收益有限）

### 图表
- `exps/attn/sparse/outs/sparse_analysis/4-loopk-debug/loopk_optimization_symmetry.png`
- 已集成到 `bench_sparse_analysis.py` 的 `_phase4_opt_plot()` 函数

---

## P5-v4b: SKW producer bug fix + 修正数据 (2026-07-03 23:28)

### Bug 发现

第一版 SKW 结果异常（339T < O1 baseline 368T，反降 -19T），根因：

**producer `store_dK` 缺少 `PerfDebugSkipDkWriteback` guard**
- `store_dV` line 2466: `if constexpr (!PerfDebugSkipDvStore && !PerfDebugSkipDvWriteback)` — 检查两个 flag
- `store_dK` line 2510: `if constexpr (!PerfDebugSkipDkStore)` — **只检查 SkipDkStore，漏掉了 SkipDkWriteback**

结果：consumer 跳过 R2S（SMEM 无有效数据），但 producer 仍执行 TMA store（写 garbage 到 GMEM） + `tma_store_wait<0>` → TMA 开销不变 + 额外 TMA 冲突。

**修复**: `if constexpr (!PerfDebugSkipDkStore && !PerfDebugSkipDkWriteback)`

### 修正后数据 (3 次平均)

| Config | TFLOPS | ms | vs O1 |
|--------|--------|----|-------|
| O1 baseline | 368 | 478.3 | — |
| O1 + SVW (skip dV writeback) | **535** | **328.6** | **+167T** |
| O1 + SKW (skip dK writeback) | **472** | **372.2** | **+104T** |
| O1 + SVS (skip dV store) | 408 | 431.7 | +40T |
| O1 + SKS (skip dK store) | 406 | 433.7 | +38T |

### 核心结论

1. **Store 级别完全对称**: SVS ≈ SKS (408 vs 406, delta=0.5%) → dV/dK TMA store 本身开销相同
2. **Writeback 级别有结构性差异**: SVW > SKW (535 vs 472, delta=12%) → dV R2S+barrier 比 dK 贵 ~63T

### 为什么 dV writeback 比 dK writeback 更贵？

**根本原因: consumer 流水线中 dV R2S 在前，dK R2S 在后**

Consumer 内循环时序:
```
MMA(S,softmax,dP,dS) → dV_R2S(等dVEmpty) → MMA4(dK) → MMA5(dQ) → dK_R2S(等dKEmpty) → 下一轮MMA...
```

Producer 时序:
```
sync(dVFull) → TMA_dV → arrive(dVEmpty) → sync(dKFull) → TMA_dK → arrive(dKEmpty) → 下一轮...
```

- **dV R2S** 是每轮迭代的第一个 writeback → 等待 dVEmpty(N-1) = 上轮 producer 完成 TMA_dV
  - 可用 overlap 时间: 仅有上轮 dK writeback 后到本轮 dV writeback 前的 MMA 时间
- **dK R2S** 是每轮迭代的最后一个 writeback → 等待 dKEmpty(N-1) = 上轮 producer 完成 TMA_dK
  - 可用 overlap 时间: dV R2S + MMA4(dK) + MMA5(dQ) → 更多 MMA 可以遮盖 TMA 延迟

**结论: dK 有更多 consumer MMA 来 overlap 其 TMA 延迟，因此实际阻塞更少。**

### dV store 能否像 dK 一样被遮盖？

关键约束: **dV TMA 读 smem_dvacc 期间，consumer 不能写新数据到同一 buffer**。

- 当前 smem_dvacc 没有 double-buffer → dVEmpty 必须在 TMA 完成后才能 arrive
- 如果 double-buffer smem_dvacc（2× 空间）→ TMA(N) 和 R2S(N+1) 可以并行
  - 但 smem_dvacc = kBlockN×kHeadDim×4B = 64×128×4 = 32KB
  - double-buffer = +32KB → ununion+stgV1 从 214KB → 246KB > 228KB limit → **不可行**

其他可能:
- `tma_store_wait<1>` pipeline（NOTE(058 P2a-2) 试过，scatter 模式零收益，dense 模式待验证）
- 减少 R2S 数据量（更小 tile 或更少 MMA 输出）
- 跨迭代 dV 累加（多轮累积后统一 R2S+TMA，减少频率）

### 后续优化方向 (优先级排序)

1. **P1 高优: 验证 dense 模式 TMA pipeline overlap**
   - 在 store_dV 中用 `tma_store_wait<1>` + 在 store_dK 开头 `tma_store_wait<0>` + arrive(dVEmpty)
   - NOTE(058 P2a-2) 仅在 scatter 模式测试过，dense 模式可能有不同结果

2. **P2 中优: 减少 dV R2S 频率**
   - 跨 2 轮 inner 迭代累加 dV（fp32 register 容量允许）
   - 每 2 轮做一次 R2S+TMA → 减半 dV writeback 频率
   - 但需要 2× dV 寄存器（可能触发 spill）

3. **P3 中优: producer/consumer barrier 重排序**
   - 调换 store_dV 和 store_dK 的执行顺序：先 store_dK 再 store_dV
   - 让 dV TMA 发生在 producer 最后 → 更多 consumer MMA 时间来 overlap
   - 风险: dK 的 overlap 时间减少，可能总时间不变

4. **P4 低优: smem_dvacc double-buffer**
   - 需要 +32KB SMEM → 超出 228KB 限制
   - 仅在配合 stgV=1 + smaller tile 时可能可行

---

## P5-v5: DeferDvR2S 实验结果 (2026-07-04 02:05)

### 实验: 将 consumer dV R2S 从 MMA3→MMA4 之间移到 MMA5 之后

**动机**: dV R2S 在 MMA3 和 MMA4 之间执行，需要等 dVEmpty barrier。将其推迟到 MMA4+MMA5
之后，给 producer 的 TMA_dV 更多 consumer MMA overlap 时间，理论上可减少 dVEmpty barrier stall。

**实现**: 新增 `MAGI_ATTENTION_FFA_BWD_DEFER_DV_R2S` debug flag，将 consumer dV R2S（sync + R2S + arrive）
从 MMA3→MMA4 之间移到 MMA5→dK_R2S 之间。

### BWD-only 数据 (GPU1, S=topk=32K, PackGQA128, IndexSparse kbs128)

| Config | BWD TFLOPS | BWD ms |
|--------|-----------|--------|
| O1 baseline | **356** | 494.2 |
| O1 + DeferDvR2S | **355** | 495.8 |

**结论: DeferDvR2S 性能中性 (delta = -0.3%)，R2S 位置不影响 BWD 吞吐。**

### 分析

1. **dVEmpty stall 在稳态下接近零**: 从 arrive(dVFull(N-1)) 到 sync(dVEmpty(N-1))
   之间有一整个 consumer 迭代的 MMA 时间（MMA4+MMA5+dK_R2S+下一轮 MMA 链），
   远超 TMA_dV 延迟 → 无论 R2S 在哪，consumer 到达 sync(dVEmpty) 时 producer 已完成。

2. **MMA3 latency hiding 不是关键因素**: 理论上 dV R2S 在 MMA3→MMA4 之间可以 overlap
   MMA3 执行。但 MMA4 的 wg_wait<1> 本身就等 MMA3 完成，移除 R2S 后 MMA4
   的等待时间增加，但不影响总时间（pipeline 深度不变）。

3. **SVW 167T 的来源重新理解**: SVW 跳过 dV writeback 的 167T 收益主要来自:
   - R2S 拷贝本身 (32KB reg→smem, ~1-2μs per iter × 262144 total iters)
   - fence + barrier arrive/sync 开销
   - **而非** dVEmpty stall（stall 接近零）

### 修正后的优化方向

| 优先级 | 方向 | 预期收益 | 可行性 |
|--------|------|---------|--------|
| ~~P1~~ | ~~DeferDvR2S~~ | 0T | ❌ 已验证无效 |
| **P2** | 减少 R2S 频率 (cross-iter accum) | 中 | ❌ 不可行：inner 每轮写不同 K block 的 dV/dK (实现后验证：数值错误，已 revert) |
| ~~P3~~ | 调换 store_dV/dK 顺序 | 0T | ❌ dense 模式 warp 已并行 |
| ~~P4~~ | smem_dvacc double-buffer | 中 | ❌ SMEM 超限 |
| **NEW-1** | 减少 R2S 数据量 | 低-中 | 需 fp16 dV 累积器或更小 tile |
| **NEW-2** | 合并 dV+dK 成单次 R2S | 中 | 需重构 SMEM layout |
| **NEW-3** | M=64 N=128 tile (空出 SMEM) | 中 | SMEM 不够：M64N128 需 260KB > 228KB |

### 核心结论

LoopK vs LoopQ 的 TFLOPS 差距根源是 **每轮 inner 迭代的 writeback 次数差异**:
- LoopK: 2 次 writeback / 迭代 (dV + dK → 各自的 K block)
- LoopQ: 1 次 writeback / 迭代 (dQ → 当前 Q block)

每次 writeback 的固定开销 = R2S copy + fence + barrier sync/arrive ≈ 1-2μs。
累积: 262144 迭代 × 1μs × 额外 1 次 writeback ≈ 262ms → 对应 SVW 的 167T gap (实测 164ms)。

**根本解法**: 减少 LoopK 每轮 inner 迭代的 writeback 次数（从 2 降到 1），
或降低每次 writeback 的固定开销。当前的 pipeline 重排序/overlap 优化已证明无效。

---

## DeferDvR2S 集成到 phase4 bench (2026-07-04 12:10)

### bench_sparse_analysis.py 更新

- 新增 `loopk_ununion_stgv1_ddv` config: O1 + `MAGI_ATTENTION_FFA_BWD_DEFER_DV_R2S=1`
- 新增 `MAGI_ATTENTION_FFA_BWD_DEFER_DV_R2S` 到 `_DEBUG_ENV_KEYS`
- phase4 bench 结果: **377 TFLOPS, 466.4ms, gap frac = +41.9%**

与 O1 baseline (368T) 相比仅 +9T（2.4%），确认 R2S 位置调整对性能无实质影响，
与之前 BWD-only 测试 (356 vs 355T) 结论一致。

---

## dKV Pool 机制分析 (2026-07-04 12:10)

### 机制
`MAGI_ATTENTION_FFA_BWD_DKV_POOL_COUNT` 控制，每个 CTA 写独立 dK/dV slice
（`blockIdx.x % pool_count * total_k` 偏移），kernel 后 `.sum(0)` 合并。
目标：减少 TMA reduce-add 的 L2 cache line contention。

### 已有 benchmark 证据 (.tmp/067-m-dkv-pool/, 2026-06-15)

**Dense MQA (nhq=128, nhk=1)**:
| pool | S=8192 | S=16384 |
|------|--------|---------|
| 1 | **240.1T** | **237.3T** |
| 2 | 243.9T | 237.3T |
| 4 | 240.1T | 230.7T (-3%) |
| 8 | 185.7T (-23%) | 126.3T (-47%) |

**IndexAttn MQA (topk=2048)**:
| pool | S=8192 | S=16384 |
|------|--------|---------|
| 1 | **159.9T** | **162.4T** |
| 4 | 159.7T | 113.6T (-30%) |

### 结论：**Pool 无用，可安全删除**
1. Pool 从未带来正向收益：`pool=1` 始终最优或持平
2. Pool 越大越差：post-kernel `.sum(0)` reduction 开销 + 额外内存分配
3. 理论上不应该有效：瓶颈在 consumer 侧 R2S+barrier，不在 producer TMA 的 L2 contention
4. 代码复杂性：每处 TMA/scatter 地址都要加 `pool_offset`，增加了维护成本
5. **建议：在 MagiAttention-2 等其他分支可以放心删除此逻辑**

---

## 综合进展总结 (2026-07-04 12:10)

### 已验证无效的优化方向

| 方向 | 结果 | 原因 |
|------|------|------|
| DeferDvR2S (R2S 位置调整) | +9T (2.4%) | dVEmpty stall 在稳态下接近零 |
| dKV Pool (L2 contention 分散) | 0T 或负面 | TMA L2 不是瓶颈 + reduction 开销 |
| Bypass atomicAdd (O2) | -173T (-52%) | scalar atomicAdd 远慢于 TMA |
| Pipeline 重排序 (dV/dK R2S 交换) | 0T | union 下总有一个 TMA 在 critical path |
| smem_dvacc double-buffer | N/A | SMEM 超限 (+32KB > 228KB) |
| dS_stage=2 | N/A | SMEM 超限 (+32KB) |
| M64N128 tile | N/A | SMEM 超限 (260KB > 228KB) |

### 已验证有效的优化

| 方向 | 收益 | 可落地 |
|------|------|--------|
| O1: Ununion+stgV1 | +34T (10%) | ✓ 可作为默认优化 |

### 根本瓶颈（不变）

LoopK 每轮 inner 迭代需要 2 次 writeback (dV + dK)，
每次 writeback 的固定开销 (R2S + fence + barrier) ≈ 1-2μs，
累积 262144 迭代 × 1μs × 额外 1 次 = ~262ms → 对应 SVW 的 167T gap。

### 剩余可探索方向

1. **跨迭代 dV 累加** — 多轮 inner iter 累加 dV 后统一 R2S+TMA → 减半 writeback 频率。
   需要 2× dV 寄存器，可能触发 spill。需验证寄存器是否足够。
2. ~~fp16 dV SMEM 累积器~~ — 精度损失不可接受，放弃。
3. **Producer/consumer 重叠优化** — 利用 ununion 后独立 SMEM buffer 的并行性，
   让 producer 在 dV TMA 未完成时就开始 dK TMA（流水线化 producer）。

---

## Tile Size 和 Outer Accumulation 对称性分析 (2026-07-04 12:30)

### Tile Size 对比 (hd=128, bf16)

**代码**: `tile_size.h` L100-104

| | InnerLoopK (SwapQK=true) | InnerLoopQ (SwapQK=false) |
|---|---|---|
| **kBlockM** | **128** | **64** |
| **kBlockN** | **64** | **128** |
| M 的语义 | outer (Q tiles) | outer (K tiles) |
| N 的语义 | inner (K tiles) | inner (Q tiles) |

**关键: M×N 互换 — InnerLoopK=M128N64, InnerLoopQ=M64N128。总 tile = 8192 tokens×hd。**

### SMEM Layout 差异

**LoopQ (`TensorStorageLoopQ`)** — `flash_bwd_kernel_sm90.h` L683-694:
- smem_k: kStages × kBlockN(128) × hd(128) × 2B = 2 × 128×128×2 = **64KB**
- smem_v: kStages × kBlockN(128) × hd(128) × 2B = 2 × 128×128×2 = **64KB**
- smem_q: kStages × kBlockM(64) × hd(128) × 2B = 2 × 64×128×2 = **32KB** (inner pipeline)
- smem_do: kStages_dO × kBlockM(64) × hd(128) × 2B = 2 × 64×128×2 = **32KB**
- smem_dqacc: kBlockM(64) × hd(128) × 4B = **32KB** (inner dQ 的 SMEM accumulator)
- **无 dKV SMEM accumulator** — dK/dV 在 register 累加
- Total ≈ 64+64+32+32+32 + smem_p/ds/lse = **~230KB** → 实际编译通过说明有优化

**LoopK (`TensorStorageLoopK`)** — L715-746:
- smem_k: kStages × kBlockN(64) × hd(128) × 2B = 2 × 64×128×2 = **32KB** (inner pipeline)
- smem_v: kStages_V × kBlockN(64) × hd(128) × 2B = 2 × 64×128×2 = **32KB** (inner pipeline)
- smem_q: kBlockM(128) × hd(128) × 2B = 128×128×2 = **32KB** (outer, 1 stage)
- smem_do: kBlockM(128) × hd(128) × 2B = 128×128×2 = **32KB** (outer, 1 stage)
- dkvacc_storage (union): kBlockN(64) × hd(128) × 4B = **32KB** (smem_dkacc/dvacc 共用)
- **无 dQ SMEM accumulator** — dQ 直接 register → atomicAdd
- Total ≈ 32+32+32+32+32 + smem_p/ds/lse = **~198KB**

### Outer dX Accumulation（代码证据）

**InnerLoopQ — outer dX = dK/dV: REGISTER 累加**
- `flash_bwd_kernel_sm90.h` L403-407:
  ```cpp
  Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, ...); // register fragment
  Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, ...); // register fragment
  clear(tdKrdK); clear(tdVrdV);
  ```
- 传入 `mma_with_loop_q(&tdKrdK, &tdVrdV)` → inner loop 中 `zero_init=false` 逐 iter 累加
- outer loop 结束后一次性 `epilogue.store_dkv()` 写到 GMEM

**InnerLoopK — outer dX = dQ: REGISTER → 即时 atomicAdd**
- `flash_bwd_kernel_sm90.h` L644-645:
  ```cpp
  Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, ...); // register fragment
  clear(tdQrdQ);
  ```
- 传入 `mma_with_loop_k(&tdQrdQ)` → 但 inner loop 中 dQ 是 `zero_init=true`（L3741 等）
  每轮 iter 独立产生 partial dQ，**立即 atomicAdd 到 GMEM**（L3147）
- 外层 `tdQrdQ` 也累加了所有 inner iter 的 dQ，outer loop 结束后 `epilogue.store_dq()` 写出

**InnerLoopK — inner dX = dK/dV: REGISTER → 每 iter R2S → SMEM → TMA**
- `mainloop_bwd_sm90_tma_gmma_ws.hpp` L3721/L3735:
  ```cpp
  Tensor tdVrdV = partition_fragment_C(...); // 每次 inner iter 重新声明
  flash::gemm<zero_init=true, ...>(..., tdVrdV); // 每 iter 零初始化
  ```
- 每轮 iter 产生 partial dV/dK → R2S copy 到 smem_dvacc/smem_dkacc → producer TMA reduce-add
- **这就是瓶颈：每 iter 2 次 R2S + barrier + TMA**

### 核心不对称总结

| | InnerLoopQ | InnerLoopK |
|---|---|---|
| Outer dX | dK/dV → **register** 累加 → outer 结束一次写 | dQ → register → **每 iter atomicAdd** |
| Inner dX | dQ → **register → SMEM → TMA** (每 iter 1 次) | dK + dV → **register → SMEM → TMA** (每 iter **2** 次) |
| SMEM accum | smem_dqacc (32KB) | smem_dkacc + smem_dvacc (32KB union / 64KB ununion) |
| Inner writeback 次数 | 1 次/iter (dQ) | 2 次/iter (dK + dV) |

**LoopQ 不需要 dKV SMEM accumulator 是因为 dK/dV 在 register 中跨 inner 迭代累加，只在 outer loop 结束时写一次。**

**LoopK 需要 dKV SMEM accumulator 是因为 dK/dV 的目标 K block 每轮 inner iter 都变（inner 遍历不同 K blocks），无法 register 跨 iter 累加。**

### Tile Size 调整分析 (M128N64 → M64N128)

如果 InnerLoopK 改为 M64N128（swap M/N）：

| SMEM Buffer | M128N64 (当前) | M64N128 (调整后) | 变化 |
|---|---|---|---|
| smem_k (inner, kStages) | 2×64×128×2=32KB | 2×128×128×2=**64KB** | +32KB |
| smem_v (inner, kStages_V) | 2×64×128×2=32KB | 2×128×128×2=**64KB** | +32KB |
| smem_q (outer, 1 stage) | 128×128×2=32KB | 64×128×2=**16KB** | -16KB |
| smem_do (outer, 1 stage) | 128×128×2=32KB | 64×128×2=**16KB** | -16KB |
| smem_dkacc (accum) | 64×128×4=32KB | 128×128×4=**64KB** | +32KB |
| smem_dvacc (union) | (共用 dkacc) | (共用 dkacc) | +32KB |
| Total | ~198KB | ~198-32+64+32=**262KB** | **+64KB** |

**M64N128 = 262KB >> 228KB → 严重超限！**

原因：inner tile 从 64→128，smem_k/v 的 pipeline buffer 翻倍（+64KB），smem_dkacc 也翻倍（+32KB），
而 outer tile 缩小只节省 32KB（Q+dO 各 -16KB）。净增 +64KB。

**即使 stgV=1**: 262 - 32 = 230KB > 228KB → 仍然超限。
**即使 stgV=1 + ununion→union**: 不变，union 在当前也是 32KB。

**结论：M64N128 对 InnerLoopK 在 hd=128 下不可行，SMEM 差距太大。**

### 为什么 InnerLoopQ 能用 M64N128？

InnerLoopQ 不需要 dKV SMEM accumulator（省 32-64KB），
smem_dqacc 只有 32KB（kBlockM(64)×hd(128)×4B）。
所以总 SMEM 刚好 ~230KB（实际有进一步优化），能塞进 228KB。

### 下一步优化方向（更新）

1. **[高优] 跨迭代 dV 累加** — 将每 iter 的 dV R2S+TMA 改为每 2 iter 累加一次
   - dV MMA 改为 `zero_init=false`，2 轮累加后一次 R2S+TMA
   - writeback 频率从 2 次/iter 降为 1.5 次/iter（dK 每轮 + dV 每 2 轮）
   - 需要 dV 在 register 中跨 iter 累加 → 寄存器压力增加
   - **关键约束**：每轮 inner iter 写的是不同 K block 的 dV，只有相邻 2 轮写同一 K block 时才能累加
   - 实际上 inner loop 是逐 K block 遍历，**不同 iter 写不同 K block** → **不可行！**

2. **[中优] Producer TMA 流水线化** — 利用 ununion 后的独立 buffer：
   - 当前 producer store_dV 和 store_dK 是串行的（union 时必须串行）
   - ununion 后两个 warp 可以并行 TMA（但实际已经是 warp1=dV, warp2=dK 并行的）
   - 真正的瓶颈是 consumer 侧的 R2S copy + fence + barrier arrive，这些在 producer 开始前就已经消耗了时间

3. ~~Producer TMA 流水线化~~ — 代码注释 NOTE(058 P2a-2) 已明确记录：
   producer store warps 的 TMA wait 不在 critical path 上，重叠 dV/dK TMA 零收益。
   Dense 模式 warp 1(dV) 和 warp 2(dK) 本身就是并行的。

---

## P5-v6: Stage 替代方案对比 (2026-07-04 14:42)

### 动机

Ununion 需要 +32KB SMEM (230KB > 228KB)，之前用 stgV=1 省 16KB 腾出空间。
但同样可以用 stgK=1 (K pipeline 2→1) 省 16KB，或者 stgK=1+stgV=1 省 32KB。
哪个 stage 减少的代价更小？

### SMEM 预算

| Config | K stg | V stg | ununion | 省/增 | Total SMEM |
|--------|-------|-------|---------|-------|------------|
| baseline | 2 | 2 | no | — | 198KB |
| stgV=1 | 2 | **1** | no | -16KB | 182KB |
| stgK=1 | **1** | 2 | no | -16KB | 182KB |
| ununion+stgV1 (O1) | 2 | **1** | **yes** | -16+32 | 214KB ✓ |
| ununion+stgK1 | **1** | 2 | **yes** | -16+32 | 214KB ✓ |
| ununion+stgK1V1 | **1** | **1** | **yes** | -32+32 | 198KB ✓ |

### Benchmark 数据 (S=topk=32K, GPU1)

| Config | TFLOPS | ms | Δ vs baseline |
|--------|--------|----|---------------|
| LoopK baseline | 321 | 548 | — |
| **stgV1 only (no ununion)** | **336** | 524 | **+15T (+5%)** |
| **stgK1 only (no ununion)** | **241** | 731 | **-80T (-25%)** |
| ununion+stgV1 (O1) | **368** | 478 | +47T (+15%) |
| **ununion+stgK1** | **125** | 1404 | **-196T (-61%)** |
| **ununion+stgK1+stgV1** | **193** | 910 | **-128T (-40%)** |
| ununion+stgK1+SVW | 370 | 475 | +49T |
| ununion+stgK1V1+SVW | 370 | 476 | +49T |
| LoopQ baseline | 499 | 352 | ref |

### 关键结论

1. **K pipeline stage 极度关键**: stgK=1 alone 造成 -25% 退化 (321→241T)。
   K load 是每轮 inner iteration 的第一个依赖（S=Q@K^T 需要 K），双缓冲对隐藏 K load latency 不可或缺。

2. **V pipeline stage 几乎无影响**: stgV=1 alone 仅 +15T (+5%)，甚至微增。
   V load 完全被 pipeline 隐藏（与 SVL=+0T 一致），减少 V stage 不伤性能。

3. **ununion+stgK1 灾难性**: 125T (-61%)。ununion 的 +32KB 加上 K 单缓冲 → K load 延迟无法隐藏，
   成为新的瓶颈。即使 SVW (跳过 dV writeback) 也只恢复到 370T，远低于 stgV1+SVW 的 535T。

4. **结论: O1 (ununion+stgV1) 是唯一可行的 stage 方案。stgK=1 和 stgdS 均不可替代 stgV=1。**

---

## 最终结论 (2026-07-04 17:15, 更新)

### Fresh benchmark 数据刷新 + stgV1 性能中性结论确认

**stgV1 结论修正 (NCU + repeat benchmark 双重验证)**:
- 5 次重复测试：stgV=1 (262.9T) vs stgV=2 (265.2T)，差异 -0.87%，标准差 ~1%
- NCU 验证：寄存器相同 (168)，无 spill (local_ld/st=0)，指令差异仅 0.71%
- 结论：stgV=1 **性能中性** (此前 +15T 为测量噪声)
- stgV=1 真正价值：节省 16KB SMEM，使 ununion 可在 228KB 限制内可行

### 所有优化方向均已穷尽

| 方向 | 结果 | 状态 |
|------|------|------|
| **O1: Ununion+stgV1** | **+38T (+19% gap)** | **✅ 唯一可落地优化** |
| Ununion+stgK1 | -196T (-61%) | ❌ K pipeline 不可缩减 |
| Ununion+stgK1+stgV1 | -128T (-40%) | ❌ K pipeline 不可缩减 |
| stgK1 only | -80T (-25%) | ❌ K pipeline 不可缩减 |
| stgV1 only (无 ununion) | **-1T (噪声)** | ✓ 性能中性，节省 16KB SMEM |
| DeferDvR2S (R2S 位置调整) | +9T (2.4%) | ❌ 无效 |
| dKV Pool (L2 分散) | 0T 或负面 | ❌ 无效，可删除 |
| O2: Bypass atomicAdd | -173T (-52%) | ❌ 太慢 |
| Producer TMA 流水线化 | 0T (NOTE 058 P2a-2) | ❌ 已验证无效 |
| Pipeline 重排序 (dV/dK R2S 交换) | 0T | ❌ union 下无解 |
| smem_dvacc double-buffer | N/A | ❌ SMEM 超限 |
| dS_stage=2 | N/A | ❌ SMEM 超限 |
| M64N128 tile | N/A | ❌ SMEM 超限 (+64KB) |
| 跨迭代 dV 累加 | N/A | ❌ 不同 iter 写不同 K block |
| fp16 dV 累积器 | N/A | ❌ 精度不可接受 |

### O1 增益拆解 (fresh data)

- LoopK baseline: 336T | LoopQ baseline: 537T | Gap: 202T (38%)
- O1 (ununion+stgV1): +38T (+19% of gap)
  - stgV1 贡献: +-1T (V pipeline 2→1, 中性)
  - ununion 贡献: +39T (dKV accumulator 分离)
- SVW ceiling (no dV wb): +199T (99% of gap) — debug only

### 根本原因（不可更改的架构约束）

**InnerLoopK 每轮 inner iteration 有 2 次 writeback (dV + dK)**，而 **InnerLoopQ 只有 1 次 (dQ)**。

- InnerLoopK inner 遍历 K blocks → 每轮 partial dV/dK 属于不同 K block → 必须立即 R2S+TMA reduce-add
- InnerLoopQ inner 遍历 Q blocks → dK/dV 属于同一 K block → register 跨 iter 累加，outer loop 结束一次写

每次 writeback 固有开销 = R2S copy(32KB) + fence + barrier sync/arrive ≈ 1-2μs/iter。
dV writeback (R2S+barrier+TMA) = 161T of gap。

### 代码重构 (2026-07-04 17:14)

`bench_sparse_analysis.py` (2706 行) 拆分为 package:
```
exps/attn/sparse/bench_sparse_analysis/
  __init__.py          — 包描述
  __main__.py          — CLI 入口
  _common.py           — 共享常量、GPU选择、计时、tensor工具
  phase0_method_parity.py  — Phase 0: 5 methods at S=topk
  phase1_topk_sweep.py     — Phase 1: topk 扫描
  phase2_kbs_compare.py    — Phase 2: kbs=1 vs kbs=128
  phase3_l2_inflection.py  — Phase 3: NCU inflection points
  phase4_loopk_debug.py    — Phase 4: LoopK debug skip 实验
```
原 `bench_sparse_analysis.py` 保留为 thin shim，两种调用方式均兼容:
- `python bench_sparse_analysis.py --plot 4-loopk-debug`
- `python -m bench_sparse_analysis --plot 4-loopk-debug`

### 图表输出

- `loopk_debug_skip.png` — 对称 cost 对比 + gap attribution
- `loopk_optimization_symmetry.png` — dV vs dK writeback/store 对称性
- `loopk_optimization_summary.png` — 综合优化 landscape + gap 分解瀑布图

---

## 补充分析 (2026-07-04 17:37)

### H100 SM 资源上限 (实测)

| 资源 | 值 | 来源 |
|------|------|------|
| SMEM/SM | 228 KB (233,472 bytes) | `torch.cuda.get_device_properties` |
| SMEM/block (opt-in) | 227 KB (232,448 bytes) | 同上 |
| Regs/SM | 65,536 (32-bit) | 同上 |
| Regs/thread (当前kernel) | 168 | NCU `launch__registers_per_thread` |
| Max threads/SM | 2,048 | 同上 |

168 regs × 384 threads = 64,512 → 98.4% 占用 → 1 block/SM。

### dK 需要 softmax_scale 而 dV 不需要

dV = softmax(S·α)^T @ dO — scale 已包含在 P 中。
dK = dS^T @ Q — Q 不含 scale，需在 writeback 时补乘。
代码证据：全文 `taccdKrdK(dki) *= params.softmax_scale` 有 4 处；`taccdVrdV.*softmax` 搜索结果为 0。

### InnerDxStoreInProducer

- 默认 = true（jinja `default('true')`）
- `=false` 时 consumer 可走 contiguous atomicAdd（reg→GMEM，无需 smem_dkacc/dvacc）
- 但注释明确 "trips an nvcc ICE" — 当前 dense 场景未启用
- 可省 SMEM: smem_dkacc(32KB) + smem_dvacc(32KB) = 64KB
- 状态：**blocked by nvcc ICE**，如修复可作为 SMEM 节省方案

### Register spill 验证

NCU 结果：stgV2/stgV1/O1 等所有显示配置均为 `local_ld=0, local_st=0, regs=168`。
Debug skip 是 `if constexpr` 编译期裁剪，不改变 register 分配。

### InnerDxStoreInProducer=false 测试 (2026-07-04 17:41)

- CUDA 13 编译：**static_assert 失败** (line 3860)，非 nvcc ICE
- 根因：`recast<float4>` 后 dV register tensor 与 GMEM tensor 大小不匹配
- dense LoopK consumer atomicAdd 路径**代码不完整**，需修复 tensor partition
- 状态：❌ blocked by incomplete code path

### Producer warp 架构 (2026-07-04 17:50)

**Dense LoopK (当前场景)**：producer WG = 3 warps
- warp0: loader (TMA load K/V/Q/dO)
- warp1: dV storer (TMA reduce-add dV → GMEM)
- warp2: dK storer (TMA reduce-add dK → GMEM)

dV 和 dK 由**独立 warp 串行**执行 `store_dV()` → `store_dK()`。
warp1 只负责 dV，warp2 只负责 dK，互不干扰。
但两者在 `store_body()` 中是**顺序调用**的（warp1 先做完 dV，warp2 再做 dK）。

NOTE(058 P2a-2)：曾尝试 dV/dK TMA 流水线化（defer dV wait → issue dK first），
**零增益**（159/161 TF 不变）— store warp 的 wait 不在 critical path 上。

### 删除 debug_skip 对称图 (旧 phase4_plot)

`loopk_debug_skip.png` 对称比较方法有误导性（LoopQ 跳过 dV store 反而变慢 347ms），
已被 `loopk_optimization_summary.png` 和 `loopk_optimization_symmetry.png` 替代。
TODO: 移除 `_phase4_plot()` 或标记为 deprecated。

### Rename Debug* → PerfDebug* (2026-07-04 19:58)

统一将 performance debugging 开关从 `Debug*` 前缀改为 `PerfDebug*` 前缀，
更准确地表达"性能调试"的含义，区别于一般的 debug 标志。

**影响范围审计** (仅限主干代码，不含 .tmp/):

| 文件 | 匹配数 | 说明 |
|------|--------|------|
| `mainloop_bwd_sm90_tma_gmma_ws.hpp` | 33 (template+constexpr) + 26 (usage) | `DebugSkip*` → `PerfDebugSkip*`, `DebugDeferDvR2S` → `PerfDebugDeferDvR2S` |
| `flash_bwd_launch_template.h` | 28 | template param pass-through |
| `bwd_inst_template.jinja` | 8 | JIT codegen `kSkipVLoad` 等 (jinja var 不变，C++ constexpr 改名) |
| `_flex_flash_attn_jit.py` | 14 | 环境变量读取 + URI 生成 |
| `phase4_loopk_debug.py` | 61 | bench 环境变量 key + 注释 |
| `bwd-loopk-fantasy-analysis.md` | 14 | plan 文档 |

**改名映射**:
- C++ constexpr: `DebugSkipVLoad` → `PerfDebugSkipVLoad` (同理其他 6 个)
- C++ template param: `SkipVLoad_` 不变 (仅内部别名改)
- Jinja var: `skip_v_load` 不变 (jinja → C++ 映射不变)
- 环境变量: `MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD` 不变 (用户接口不变)
- Python JIT: 内部变量名不变 (只改注释)
- Bench: `_DEBUG_ENV_KEYS` / `_DEBUG_CONFIGS` 不变 (只改注释)

**原则：只改 C++ constexpr 别名 (19 处 PerfDebug 引用)，不改 template param / env var / jinja var / Python 接口**。
最小化对主干代码的入侵。

**执行完成** (2026-07-04 20:00):

实际改动统计：
| 文件 | 改动 | 说明 |
|------|------|------|
| `mainloop_bwd_sm90_tma_gmma_ws.hpp` | 7 constexpr 定义 + 24 usage + 1 comment | `Debug*` → `PerfDebug*` |
| `phase4_loopk_debug.py` | 2 comments | "debug skip" → "perf-debug skip" |
| `__init__.py` (bench_sparse_analysis) | 1 comment | 同上 |
| `bwd-loopk-fantasy-analysis.md` | 14 | plan 文档历史引用 |

**不变的部分** (主干代码零入侵):
- 环境变量名: `MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD` 等 (不变)
- C++ template param: `SkipVLoad_` 等 (不变)
- Jinja: `skip_v_load` 等 (不变)
- Python `_flex_flash_attn_jit.py`: env var key (不变)
- Bench `_DEBUG_ENV_KEYS` / `_DEBUG_CONFIGS` dict keys (不变)

### Bench 重测 + NCU Spill 验证 (2026-07-05 00:22)

PerfDebug 重命名后全量重测 40 configs，数据与之前一致（±5T 噪声）。

**核心数据** (S=topk=32K, nhq=128, nhk=1, hd=128, bf16, H100):

| 实验 | TFLOPS | ms | gap% | 说明 |
|------|--------|-----|------|------|
| LoopK baseline | 341 | 516.3 | 0% | M128N64, stg=2, stgV=2 |
| **LoopQ baseline** | **534** | **329.4** | **100%** | M64N128 |
| stgV1 only | 338 | 521.3 | -2.7% | V stage 性能中性，省 16KB SMEM |
| **O1 (ununion+stgV1)** | **378** | **464.9** | **+27.5%** | 可落地优化 |
| O1+SVL | 384 | 457.9 | +31.3% | V load 无影响 |
| O1+DDV | 376 | 468.3 | +25.7% | defer R2S 无效 |
| O1+SVS (skip dV store) | 410 | 428.8 | +46.8% | dV TMA = 36ms |
| O1+SKS (skip dK store) | 407 | 432.2 | +45.0% | dK TMA = 33ms (对称) |
| O1+SKW (skip dK wb) | 488 | 360.6 | +83.3% | dK R2S+TMA = 104ms |
| **O1+SVW (skip dV wb)** | **536** | **328.3** | **+100.6%** | dV R2S+TMA = 137ms |
| **O1+SVW+SKW** | **710** | **247.8** | **+143.7%** | 无 writeback 天花板 |
| stgK1 only | 246 | 713.8 | -105.7% | K pipeline 灾难性下降 |
| bypass atomicAdd | 156 | 1128.3 | -327.6% | scalar atomicAdd 太慢 |

**NCU Spill 验证** (2026-07-05 23:12):

| Config | Regs/Thread | Local LD | Local ST | Spill |
|--------|-------------|----------|----------|-------|
| LoopK baseline | 168 | 0 | 0 | NO |
| LoopQ baseline | 168 | 256 | 256 | YES (轻微) |
| O1 (ununion+stgV1) | 168 | 0 | 0 | NO |
| O1+SVW | 168 | 0 | 0 | NO |
| O1+SKW | 168 | 0 | 0 | NO |
| O1+SVS | 168 | 0 | 0 | NO |
| O1+SKS | 168 | 0 | 0 | NO |
| O1+SVW+SKW | 168 | 0 | 0 | NO |
| O1+DDV | 168 | 0 | 0 | NO |

所有 InnerLoopK 变体无 spill。LoopQ 反而有轻微 spill (256 sectors)，
说明 LoopK 的性能差距不是寄存器问题。

**结论**:
- Gap 根因 = dV writeback pipeline（R2S 32KB + fence + barrier + TMA reduce-add），每 inner iteration 0.5ms × 256 iterations = 137ms
- O1+SVW (536T) ≈ LoopQ (534T) → **去掉 dV writeback 后 InnerLoopK 追平 InnerLoopQ**
- O1 是唯一可落地优化 (+37T, +11%)，通过分离 dKV SMEM buffer 部分并行化 writeback
- dV/dK store 对称 (SVS ≈ SKS)，writeback 不对称 (SVW > SKW) — dV 在 critical path 更早期

**图表清理** (已完成):
- 删除过时的 `loopk_fantasy.png` 和 `loopk_gap_waterfall.png`
- 保留 `loopk_optimization_summary.png` (landscape + waterfall) 和 `loopk_optimization_symmetry.png` (dV/dK 对称)

### 下一步探索方向 (优先级排序)

1. **P1: 修复 consumer atomicAdd 路径** — 修复 `recast<float4>` partition 不匹配，
   使 `InnerDxStoreInProducer=false` 可编译。可省 smem_dkacc+dvacc (64KB)。
   优先级中等：修复工作量 ~1天，且需验证 atomicAdd 性能是否可接受。

2. ~~P2: NCU 批量验证~~ **已完成** — 所有 O1 变体无 spill。

3. ~~P3: 图表清理~~ **已完成** — 保留 summary + symmetry 两张图。

4. ~~**P4: 跨迭代 dV 累加探索**~~ **❌ 已实现并 revert** — 数值错误。
   InnerLoopK 每次 inner iteration 处理不同 K-block position 的 dV，不能累加到同一个
   register fragment。实现后 benchmark 显示 TFLOPS 提升（因为少做了 writeback），但
   结果数值错误。详见下方 "P4 Cross-iteration dV Accumulation — REVERTED" 章节。

---

## P4 Cross-iteration dV Accumulation — REVERTED (2026-07-06)

### CI Fix (kept)
- `.pre-commit-config.yaml`: merge duplicate `exclude` keys for `chinese_checker`,
  add `^\.cursor/` exclusion to allow Chinese in plan/rules files.
- `phase4_loopk_debug.py`: translate Chinese comments to English.

### P4 Status: ❌ REVERTED — Fundamentally incorrect for InnerLoopK

**Root cause**: In InnerLoopK, each inner iteration processes a **different K-block
position**. The `dV` computed in iteration `i` belongs to K-block `n_block_i`, while
`dV` in iteration `i+1` belongs to K-block `n_block_{i+1}`. Accumulating these in a
single register fragment (`tdVrdV_persist`) mixes gradients for different memory
locations, producing numerically incorrect results.

The earlier plan had flagged this as "不可行!" (not feasible!) in the initial analysis,
but the flag was overlooked during implementation.

**Why the benchmark showed improvement**: The implementation performed fewer writebacks
(50% reduction), so wall-clock time decreased. However, the accumulated values were
wrong — each flush wrote the sum of `dV` from 2 different K-block positions into just
one of them.

**What was reverted (2026-07-06 12:00)**:
- `mainloop_bwd_sm90_tma_gmma_ws.hpp`: removed `DvCrossIterAccum_` template param,
  `PerfDebugDvCrossIterAccum`, `kDvFlushInterval`, `tdVrdV_persist`, flush gating logic,
  and residual flush. Restored per-iteration `tdVrdV` allocation with `zero_init=true`.
- `_flex_flash_attn_jit.py`: removed `MAGI_ATTENTION_FFA_BWD_DV_CROSS_ITER_ACCUM` env var.
- `bwd_inst_template.jinja`: removed `kBwdDvCrossIterAccum` constexpr and template arg.
- `flash_bwd_launch_template.h`: removed `DvCrossIterAccum` from both `run_flash_bwd`
  and `run_mha_bwd_` templates.
- `phase4_loopk_debug.py`: removed `_CROSS_ITER_CONFIGS`.

**Note**: The PerfPerfDebug→PerfDebug rename (from the same commit) is a valid fix
and was kept.

### Lessons learned
- Cross-iteration register accumulation is only valid when consecutive iterations
  write to the **same** output position (e.g., InnerLoopQ where all inner iterations
  contribute to the same Q-block's `dQ`).
- For InnerLoopK, the alternative "double-buffer SMEM" approach (allocating a second
  `dV` SMEM buffer to overlap R2S/TMA with the next MMA) remains the theoretically
  correct approach, but requires +32KB SMEM which exceeds the 228KB H100 limit
  under current tile/stage settings.

---

## O1 Landing + InnerDxStoreInProducer 分析 (2026-07-06 14:15)

### T1: O1 落地 — ununion+stgV1 作为 LoopK 默认 [P0]

**目标**: InnerLoopK BWD 默认启用 ununion+stgV1 (+38T, +11%)，提供 PerfDebug 开关恢复旧行为。

**实现方案**: 仿照 `bwd_lse_union` 对 LoopQ 的默认处理 (L501-510)：
1. `_flex_flash_attn_jit.py`: LoopK 时自动注入 `bwd_ununion_dkvacc=1` + `bwd_stages_v=1`（除非 env 显式覆盖）
2. 新增 `MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2=1` — 一键恢复 union+stgV2
3. C++/jinja/template 层无需改动（defaults 仍为 0，由 JIT 注入）
4. 仅影响 LoopK (SwapBwdQKLoop=true)，LoopQ 不受影响

**验证**: 远程 precompile → 跑 `test_block_sparse.py` + `test_index_sparse.py` 正确性

### T2: InnerDxStoreInProducer=false 可行性分析 [不可行，存档]

**机制**: consumer MMA WG 直接 atomicAdd dK/dV 到 GMEM → 省 smem_dkacc+dvacc (64KB)，
消除 producer store warps。

**结论: 对 dense LoopK 不可行**：
- 代码路径不完整: `recast<float4>` partition 不匹配导致 static_assert 失败
- 即使修复: consumer atomicAdd 对 dense = O2 bypass 的等价路径 → -51.6% (161T)
- atomicAdd 128 floats/row vs TMA 1 次 bulk reduce-add/row → 128× 更多 L2 transactions
- SMEM 节省 64KB 虽大，但无法抵消 atomicAdd 的 3.8× 性能惩罚
- 仅对 IndexSparse scatter 路径有潜在价值（scatter 本身就是 per-token atomicAdd）

### T3: dKV Pool 删除 [DEFERRED — 待合并时处理]

feat/bwd-loopq-sparse-load 可能已清理。合并时确认，如未删则一并清理。

### 执行顺序

1. **T1 实现** → JIT 默认值 + PerfDebug 开关 + commit
2. **T1 验证** → 远程 precompile + correctness test
3. **T1 benchmark** → 验证 O1 默认 TFLOPS 与之前一致

---

## Merge feat/bwd-loopq-sparse-load (2026-07-06 16:20)

### 动机
feat/ 分支有大量新进展 (121 commits): 测试重组、crash fixes、rename、dKV pool 删除、
OuterUseAtomicReduction、TMA 1D bulk load 等。本分支最终要合回 feat/，先拉入保持同步。

### 合并结果
- **Merge commit**: `c7196214`
- **冲突 (4个, 全部解决)**:
  1. `.pre-commit-config.yaml` — 保留 `.cursor/` 排除
  2. `bench_sparse_analysis.py` — 接受我方删除 (已有 package 版本)
  3. `bwd_inst_template.jinja` — 取 feat/ 新增 (`kOuterUseAtomicReduction`, `kDisableDqAtomic`, 删除 dKV pool) + 保留我方 perf debug flags
  4. `_flex_flash_attn_jit.py` — 自动解决 (注释措辞)

### feat/ 带来的关键变更
| 分类 | 变更 |
|------|------|
| **架构** | DisableOuterAtomicReduction → OuterUseAtomicReduction; disable_bwd_dq_atomic_reduction |
| **dKV pool** | 已删除 — 用 sparse outer dx store+empty 模式替代 |
| **KV head** | IndexSparse KV head 3D 表达 (unflatten) |
| **测试** | 3-tier BlockSparse + IndexSparse test suites; crash fixes |
| **TMA** | TMA 1D bulk load for BWD scatter paths |
| **Naming** | SparseLoad→BlockSparse, IndexAttn→IndexSparse, IsLoopQ→IsInnerLoopQ |
| **Infra** | clang-format-20, .cursor/ in .gitignore |

### 后续任务方向 (优先级排序)

**已确认无效方向 (不再探索)**:
- ~~P4 跨迭代 dV 累加~~ — 数值错误 (inner 每轮不同 K-block)
- ~~smem_dvacc double-buffer~~ — SMEM 超 228KB 限制
- ~~InnerDxStoreInProducer=false~~ — atomicAdd = -51.6%
- ~~DeferDvR2S~~ — 已验证无效

**可继续探索的优化方向**:
1. **Pipeline overlap 改进**: 探索 dK writeback 与下一轮 MMA overlap 的可能性
   (dK 和 dV 是独立位置，barrier 独立 → 有空间)
2. **Tile shape 调优**: M=64/N=128 vs M=128/N=64 → SMEM/register trade-off
3. **InnerLoopK specific**: dV R2S 延迟方案 — 不是跨迭代累加，而是利用
   SMEM 双 stage 让 R2S 和 MMA4(dK) 真正 overlap (需 feat/ 新 SMEM budget)
4. **Benchmark with feat/ new features**: O1 (ununion+stgV1) + feat/'s TMA 1D
   load 对 TFLOPS 的综合影响

**当前需要验证**:
- [x] CI 通过 (merge 后的综合 lint) — ✅ 2026-07-06 12:42
- [ ] 合并后 correctness: `test_block_sparse.py` + `test_index_sparse.py`

---

## Post-merge Benchmark Results (2026-07-06 18:05)

Phase 4 full-power re-run after merge. GPU: H100, S=topk=32K, nhq=128, nhk=1, hd=128, bf16, PackGQA.
**Note**: Baseline now auto-applies O1 (ununion+stgV1) via JIT default.

### Core numbers

| Config | TFLOPS | ms | Δ vs baseline |
|--------|--------|----|---------------|
| **LoopK baseline (=O1)** | **371.9** | 473 | — |
| LoopQ baseline | 527.3 | 334 | +155.4T (gap=100%) |
| O1+SVW (no dV writeback) | 535.9 | 328 | +164.0T |
| O1+SKW (no dK writeback) | 487.9 | 361 | +116.0T |
| O1+SVW+SKW (no wb at all) | 709.9 | 248 | +338.0T |
| Skip dV writeback only | 546.1 | 322 | +174.2T |
| Symmetric (no V, no dV wb) | 547.2 | 321 | +175.3T |
| Light V load | 383.2 | 459 | +11.3T |
| No dV store | 410.1 | 429 | +38.2T |
| No dK store | 408.3 | 431 | +36.4T |
| No dV MMA | 472.8 | 372 | +100.9T |
| No dV MMA+store | 486.9 | 361 | +115.0T |
| Skip all | 516.3 | 341 | +144.4T |
| Bypass atomicAdd | 155.9 | 1128 | -216.0T ❌ |
| StgK1 only | 246.5 | 713 | -125.4T ❌ |
| StgV1 only | 337.5 | 521 | -34.4T |

### Gap decomposition (LoopK baseline=O1 → LoopQ)

Gap = 527.3 - 371.9 = **155.4 T**

| Factor | TFLOPS recovered | % of gap |
|--------|-----------------|----------|
| dV writeback pipeline (SVW) | 164.0T | **105%** (> gap!) |
| dK writeback pipeline (SKW) | 116.0T | 75% |
| V inner load (SVL) | 11.3T | 7% |
| dV MMA (no compute) | 100.9T | 65% |

**Key findings**:
1. dV writeback alone (**164T**) exceeds the entire LoopK-LoopQ gap (155T).
   This means LoopQ has its own overhead (dQ atomicAdd) that partially offsets.
2. dK writeback is also expensive (116T) — but LoopK baseline already uses
   ununion (independent dV/dK buffers), so both pipelines run in parallel to some extent.
3. When BOTH writebacks are removed (709.9T), LoopK actually exceeds LoopQ
   by 182.6T — confirming LoopK's core MMA path is fundamentally faster.
4. The "symmetric" config (no V load + no dV writeback) = 547.2T ≈ LoopQ (527.3T),
   confirming these two factors fully explain the gap.
5. O1 (ununion+stgV1) is now the DEFAULT. Previous runs without it showed ~339T baseline;
   current 371.9T confirms the +38T improvement is baked in.

### Remaining optimization space

The dV writeback pipeline (R2S + barrier + TMA reduce-add) is the dominant bottleneck.
Given P4 (cross-iter accum) is invalid, remaining options:
- **Pipeline overlap**: Can dV R2S overlap with MMA4(dK)? Currently barrier-serialized.
- **Reduce R2S data**: fp16 dV accumulator → halve transfer, but accuracy loss risk.
- **Better MMA/pipeline scheduling**: Defer dV R2S was already tested (DeferDvR2S) → no gain.
- **Architectural**: M64N128 tile reduces writeback iterations but needs 282KB SMEM.
