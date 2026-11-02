# FFABwdSm100Index（DSA 反向）性能记录

本文记录 2026-11 这一轮针对 DSA 场景的优化：测了什么、合入了什么、试过但没用的方案和原因，以及后续方向。目标是"不论序列长度都尽量变快"，所以所有结论都按序列长度分区间给出。

## 1. 测量方法

- 场景：sm90 契约下的 bench 数据（`exps/attn/cutedsl/run_index_sparse_sm100_benchmark.py` 的造数方式）。每个 q block 用 `randint` 生成排好序的 topk 个 K id，`nhk = 1`，一个 q block 的 token × q head 正好拼成 128 行 kernel Q tile。
  - `mqa128`：128 个 q head，qbs = 1。
  - `gqa8`：8 个 q head，qbs = 16。
  - `gqa16`：16 个 q head，qbs = 8。
- 只计主 kernel：把 `compile_cache` 包一层，用 cuda event 夹住主 kernel 的那次调用，不含 pre/postprocess。
- 先冷却 3 s，再连续跑 N 次。"冷启动"取前 3 次平均，"持续"取后一半的中位数。S = 32K 时连续运行会撞 1100 W 功耗墙，持续值比冷启动慢 6–8%；长序列是访存瓶颈，两者基本一样。
- A/B：把 HEAD 版本拷成 `_head_bwd_sm100_index.py`（JIT 缓存名改成 `bwd_sm100_index_head`），在同一张卡上新旧交替跑。
- 机器上有别人的任务：开跑前先看 `nvidia-smi`。有一轮 GPU 4/7 被占满（240 W、100% 利用率），HEAD 自己在 S=131K 下从 71.6 ms 变成 175 ms，那一轮数据全部作废。
- 插桩：用内联 PTX 读 `%clock64`，在每个 pipeline 等待点累计周期，最后原子加到全局缓冲（`make_instr.py`）。ncu/nsys 在这台机器上挂在 cuInit，插桩是目前唯一能看到"谁在等谁"的手段。注意插桩本身会改变时序：S=524288/topk=4096 下，插桩版里异步 V 比 HEAD 快 3%，不插桩时反而慢 3%。所以插桩只用来看分布，计时一律用不插桩的版本。

脚本都在 `/home/niubility2/lijin/ncu_reports/ffa_bwd_sm100_index_dsa/`：

| 脚本 | 用途 |
|---|---|
| `dsa_bench.py` | 主 kernel 计时。`SCEN S TOPK ITERS [module] [--kspan=N] [--unsorted] [--gva=0/1]` |
| `ab.sh` / `sweep.sh` | 多模块、多规模批量 A/B |
| `check_vs_head.py` | 和 HEAD 拷贝逐用例对比：dQ 逐位、dK/dV 相对差、NaN。`FFA_GVA=0/1` 强制 gather 方式 |
| `make_ablate3.py` / `abl.sh` | 消融开关：nogather、nobulk、nostage、nosoftmax |
| `make_ablate_boundary.py` | Q tile 边界消融：qfix（Q/dO 始终命中 L2）、nodqepi（跳过 dQ epilogue） |
| `make_instr.py` / `instr_run.py` | `%clock64` 插桩及各角色等待分布 |
| `make_exp_cache.py` / `pol.sh` | L2 cache hint 实验 |
| `make_knobs.py` / `knob.sh` | `FFA_RCHUNK` / `FFA_KVSTAGE` 实验 |
| `variants/` | 试过的 load_KV 变体源码 |

## 2. 基线特征（HEAD，MQA128，冷启动）

| S | topk | ms | TF |
|---|---|---|---|
| 32768 | 2048 | 12.78 | 860 |
| 32768 | 4096 | 25.3 | 868 |
| 65536 | 4096 | 59.8 | 735 |
| 131072 | 2048 | 71.6 | 614 |
| 131072 | 4096 | 131 | 670 |
| 262144 | 4096 | 288 | 611 |
| 524288 | 2048 | 383 | 460 |
| 524288 | 4096 | 656 | 536 |

S 增大时性能下降，拐点在工作集超过 L2 的位置（B300 L2 为 126.5 MB）。工作集按每个 K 行 1.5 KB 算：bf16 的 K、V 各 256 B，fp32 的 dK、dV 累加器各 512 B。S=32K 约 48 MB，S=64K 约 96 MB，S=128K 约 192 MB。

上界实验（`--kspan`）：Q 仍是 524288 个 token，但所有索引都落在前 32768 个 K 行内。topk=2048 从 460 TF 涨到 808 TF（1.76×），topk=4096 从 533 TF 涨到 784 TF（1.47×）。也就是说，长序列丢掉的性能全部来自 L2 装不下随机访问的工作集。

消融（topk=2048，冷启动 ms）：

| S | 基线 | nobulk | nogather + nobulk |
|---|---|---|---|
| 32768 | 12.78 | 10.21 | 9.67 |
| 131072 | 71.6 | 49.5 | 38.5 |
| 524288 | 384 | 228 | 153 |

计算侧的下限在各个 S 下都是约 1140–1150 TF，与 S 无关。nogather 单独开在 32K 下反而变慢（smem 是垃圾数据），不能作为依据。

## 3. 三个区间的瓶颈（插桩结论）

**短序列（工作集 ≤ L2/2，例如 S ≤ 32K）：卡在 dK/dV 写回的 L2 归约。**
- reduce warp 有 54% 的时间卡在发出 `cp.reduce.async.bulk` 上，也就是 TMA 被 L2 反压。
- 写回速率约 5.3–5.4 TB/s，而之前用 microbench 测的 L2 归约上限是 6.2 TB/s，已经到上限的 85–90%。
- HEAD 下 MMA warp 也有 18% 在等 K、10% 在等 V。可是 gather 一变快，等待就转移到 dV 累加器的释放上（326 → 676 周期/K tile），总时间不变。

**中间（L2/2 < 工作集 ≲ 3–4× L2，S≈64K–256K）：卡在 gather 的 DRAM 延迟。**
- HEAD 的 load_KV 每个 K tile 串行等两次（先等 K，再等 V），每次只有 16 KB 在路上。

**长序列（工作集 ≈ 6× L2，S=512K）：DRAM 带宽饱和。**
- HEAD 下 MMA warp 有 41% 在等 K、25% 在等 V；reduce warp 反而 55–60% 空闲。
- 但加深预取没用（见 5.6），说明是带宽而不是延迟。
- 按字节估算：随机 gather 加累加器读改写，每个 (Q tile, K 行) 约 2.5 KB，S=512K/topk=2048 时约 7 TB/s，已经接近 HBM 上限。

## 4. 合入的优化：V 行异步落地（`gather_V_async`，按工作集自动开启）

### 改动

- HEAD 的 load_KV 每个 K tile：gather K → `wait_group(0)` → fence → arrive → gather V → `wait_group(0)` → fence → arrive。
- 开启 `gather_V_async` 后，K 保持原样；V 改为每个线程发完 cp.async 就执行 `cp.async.mbarrier.arrive.noinc`，然后直接去等下一个 sK stage。
  - sV 的 full barrier 改为 128 个线程到达。
  - MMA warp 在每次等到 sV full 之后执行 `fence.proxy.async`：cp.async 属于 generic proxy，UMMA 走 async proxy，而发射 cp.async 的线程从不等待，所以 fence 只能放在消费端。
- 下一个 K tile 的 `wait_group(0)` 会顺带把上一个 tile 的 V 也等掉。因为 V 早发了一个 K 的时间，通常已经落地。
- K 必须保持同步：S 需要的下一个 K，就在 dQ 释放 sK 之后马上要用，只有一个 tile 的提前量。实测 K 用异步到达在 32K 下慢 6%（见 5.1）。

### 自动开关

规则在 `gather_v_async_default`：记 `total_k × nhk × head_dim × (2×元素字节 + 2×4)` 为工作集（K/V 行加 fp32 dK/dV 累加器行），工作集在 (L2/2, 4×L2] 区间内才开启。

- 低于 L2/2：gather 命中 L2，瓶颈在 L2 归约。gather 并发度一高就会和归约争 L2，强制开启时 32K 下慢 0.8–1.7%。
- (L2/2, 4×L2]：gather 会 miss L2，瓶颈在 DRAM 延迟，异步 V 能把 V 的延迟藏在等 sK 的时间里。
- 超过 4×L2：随机 gather 和累加器读改写把 DRAM 带宽占满。强制开启时 S=512K/topk=4096 慢 2–3%（多次复测稳定），topk=1024 快 4%，所以统一关闭，保证不回退。
- B300 上一个 kv head、head_dim 128 时，大约 S ∈ (43K, 345K] 会开启。
- host 入口 `gather_v_async: Optional[bool] = None`，None 表示自动。

### 结果（主 kernel TFLOPS，越大越好）

FLOPs 按 bench 的公式 `10 × S × topk × nhq × 128` 计算。"冷"为冷却 3 s 后前 3 次平均，"持续"为连续运行后半段的中位数（S=32K 时会撞功耗墙）。

| 场景 | HEAD 冷 | 优化后 冷 | 变化 | HEAD 持续 | 优化后 持续 | 变化 |
|---|---|---|---|---|---|---|
| MQA128 32K / 1024 | 823 | 823 | 持平 | 781 | 782 | 持平 |
| MQA128 32K / 2048 | 861 | 861 | 持平 | 806 | 806 | 持平 |
| MQA128 32K / 4096 | 863 | 869 | +0.7% | 795 | 796 | 持平 |
| GQA8 32K / 4096 | 815 | 815 | 持平 | 818 | 818 | 持平 |
| MQA128 65K / 2048 | 695 | 808 | +16.3% | 679 | 774 | +14.0% |
| MQA128 65K / 4096 | 743 | 827 | +11.3% | 718 | 778 | +8.4% |
| MQA128 131K / 2048 | 613 | 694 | +13.2% | 607 | 680 | +12.0% |
| MQA128 131K / 4096 | 686 | 773 | +12.7% | 670 | 732 | +9.3% |
| GQA8 131K / 4096 | 690 | 807 | +17.0% | 691 | 807 | +16.8% |
| GQA16 131K / 4096 | 697 | 818 | +17.4% | 671 | 745 | +11.0% |
| MQA128 262K / 2048 | 533 | 538 | +0.9% | 530 | 536 | +1.1% |
| MQA128 262K / 4096 | 603 | 664 | +10.1% | 598 | 655 | +9.5% |
| MQA128 524K / 1024 | 414 | 414 | 持平 | 414 | 414 | 持平 |
| MQA128 524K / 2048 | 460 | 460 | 持平 | 460 | 460 | 持平 |
| MQA128 524K / 4096 | 537 | 536 | 持平 | 536 | 535 | 持平 |
| GQA8 524K / 4096 | 541 | 540 | 持平 | 540 | 538 | 持平 |

"持平"表示差异在 ±0.3% 以内，和 HEAD 重复测量的波动一样大。S=32K 和 S=512K 走的是和 HEAD 相同的同步路径。

### 正确性

- dQ 与 HEAD 逐位一致，两条路径都是。
- dK/dV 与 HEAD 的相对差在 bf16 一个 ulp 量级。HEAD 对 HEAD 自己跑两次的差异也在同一量级，来自 fp32 归约顺序。
- `test_index_sparse_bwd_sm100.py` 新增两个测试：
  - 强制两条路径各跑三种配置：MQA128、多 segment 的 MQA4、带 -1 padding 且长度不是 64 倍数的索引行，全部和 SDPA 参考对比。
  - 单测自动规则的上下阈值。

## 5. 试过但没有合入的方案

### 5.1 K 和 V 都用异步到达

| S | HEAD | 异步 K+V |
|---|---|---|
| 32K | 12.78 | 13.57（−6%） |
| 131K | 71.5 | 62.7（+12%） |
| 524K | 382.5 | 385.8（持平） |

去掉 MMA 端的 fence 也一样慢（13.62），所以慢不是 fence 造成的，而是 K 的异步到达完成得比"等待 + fence + arrive"晚。K 只有一个 tile 的提前量，对这点延迟很敏感。

### 5.2 每个 tile 先发 V 再发 K

- K 也异步时：32K 13.94 ms，比先 K 更差。
- K 同步、V 异步时：32K 13.33 ms，524K/4096 745 ms（−14%）。

推测原因：K 的 `wait_group(0)` 会把刚发出去的 V 一起等掉，而且 V 更早发出会更早和归约争带宽。

### 5.3 K、V 拆给不同 warp（warp 0–1 搬 K，warp 2–3 搬 V）

| 方案 | 32K | 131K |
|---|---|---|
| V 异步 | 13.19（−3.2%） | 62.9 |
| V 同步 | 12.94（−1.2%） | 63.6 |

两种都不如同一组 warp 上"K 同步、V 异步"（12.87 / 63.2）。规律是：gather 并发度越高，L2 区间越慢。

### 5.4 L2 cache hint（S=131K/topk=2048）

| 配置 | ms |
|---|---|
| 不加 | 62.9 |
| gather evict_first | 66.0（−5%） |
| gather evict_first + reduce evict_last | 64.1 |
| reduce evict_last | 63.0 |

K/V 行会被 topk 个 Q tile 反复读到，L2 默认替换已经在帮忙，强制 evict_first 是负收益。

实现时的坑：给 cp.async 加 `L2::cache_hint` 要走内联 PTX，而 swizzled smem 指针的 `.toint()` 拿到的是未 swizzle 的地址，必须手动做一次 `a ^ (((a >> 7) & 7) << 4)`（SW128，buffer 按 1024 B 对齐），否则结果全错。

### 5.5 Q tile 边界开销

用 topk = 256/512/2048 拟合，每个 Q tile 有约 3.1 µs 固定开销。占比：topk=2048 约 5%，topk=1024 约 10%，topk=512 约 18%。

消融（S=32K/topk=512）：

| 配置 | ms | 变化 |
|---|---|---|
| 基线 | 3.629 | — |
| Q/dO 始终命中 L2 | 3.605 | −0.7% |
| 跳过 dQ epilogue | 3.454 | −4.8% |
| 两者都去掉 | 3.385 | −6.7% |

结论：
- 预取下一个 tile 的 Q/dO 到 L2 不值得做。
- 把 dQ epilogue 改成每个线程直接 st.global 自己那一行（去掉 smem、TMA store 和 named barrier）反而更慢：topk=512 慢 5%，topk=2048 慢 3%。原因是一个 lane 一行，每条 store 拆成 32 个事务，一个 tile 要 2048 个 16 B 事务，LSU 吞吐远不如 TMA 一次搬 32 KB。
- 剩下约 2 µs 是流水排空和重填。要和上一个 tile 重叠，需要 Q/dO 双缓冲（再多 64 KB smem），现在 smem 已满。

### 5.6 kv_stage 3 + reduce_chunk 16（暂存区让出 32 KB 给 K/V 第三个 stage）

| S/topk | HEAD | 调整后 |
|---|---|---|
| 32K/2048 | 12.77 | 13.89（−9%） |
| 131K/2048 | 72.0 | 71.4（无变化） |
| 524K/4096 | 656 | 653（无变化） |

短序列下 reduce 是瓶颈，暂存块变小更慢；长序列加深预取没有收益，说明长序列是 DRAM 带宽瓶颈而不是延迟。

### 5.7 load_KV 软件流水预读下一个 tile 的行号

第一版把预读放在 K 的 `fence.proxy.async` 之前，32K 慢 5.8%：proxy fence 会等本线程之前所有的访存完成，行号读取于是落到了 sK 提交路径上。挪到 V 发出之后，结果和 HEAD 持平（±0.5%），说明行号延迟本来就被藏住了，没有合入。

### 5.8 其他

- 索引不排序：32K 下慢 2.5%（847 vs 868 TF）。现在 kernel 不依赖排序。
- 全局相位对齐的 K 扫描（所有 CTA 同时扫 K 空间的同一段）：只做了分析，没有实现。每个 wave 内同一个 K 行只会被摸到 148·topk/S 次，S=131K、topk=2048 时约 2.3 次。对齐后变成 LRU 最不擅长的循环扫描，可能比随机顺序的命中率还低。只有在 148·topk/S 远大于 1 时才可能有收益。

## 6. 后续方向（按预期收益排序）

1. **长序列：按 K 区间多趟处理，换取 L2 局部性。** 上界见第 2 节（524K 下 1.47–1.76×）。做法：把 K 空间切成能装进 L2 的段（每段约 40–60K 行），每一趟所有 Q tile 只处理落在当前段内的索引。
   - 需要每行每段的起止位置：索引已排序时可以直接 `torch.searchsorted`；未排序就要先排序，S=512K 时代价可能达到 kernel 时间的 5–10%。
   - dQ 要改成 fp32 跨趟累加，并加一个 postprocess。
   - 最大的障碍是每次访问一个 Q tile 都有约 3 µs 固定开销，而 topk=2048、切 8 段时每次只处理约 4 个 K tile，开销占 30–40%。所以要先把 Q tile 切换做成近乎零成本（Q/dO 双缓冲，或者重新规划 smem），这个方向才成立。
2. **短序列：只有减少 dK/dV 写回字节才可能再快。** 现在已经到 L2 归约上限的 85–90%。DSA bench 里每个索引行只对应一个 Q tile，没有可以合并的写回；qbs×G 是 128 的整数倍以上时，可以让两个 Q tile 在 TMEM 里先合并 dK/dV 再写回（即之前 history 里的方向 B）。
3. **小 topk 的 Q tile 固定开销**：dQ epilogue 约 0.8 µs，流水重填约 2 µs。都需要额外 smem 才能和上一个 tile 重叠。

## 7. 复现

```bash
cd /home/niubility2/lijin/ncu_reports/ffa_bwd_sm100_index_dsa
# 先生成 HEAD 拷贝作为 A/B 基线
git -C /home/niubility2/lijin/MagiAttention show HEAD:magi_attention/kernel/cutedsl/ffa_bwd_sm100_index.py \
  | sed 's/get_jit_cache("bwd_sm100_index")/get_jit_cache("bwd_sm100_index_head")/' \
  > /home/niubility2/lijin/MagiAttention/magi_attention/kernel/cutedsl/_head_bwd_sm100_index.py
./ab.sh 3 "_head_bwd_sm100_index ffa_bwd_sm100_index" "mqa128 131072 2048 6"
CUDA_VISIBLE_DEVICES=3 FFA_GVA=1 python check_vs_head.py
# 用完删掉 cutedsl/ 下的 _head_*.py、_ablate_*.py、_ins_*.py、_kn_*.py、_exp_*.py
```

`cutedsl/` 下多出来的 `.py` 文件会改变 JIT 磁盘缓存的源码指纹（如果开启了磁盘缓存），用完要删。
