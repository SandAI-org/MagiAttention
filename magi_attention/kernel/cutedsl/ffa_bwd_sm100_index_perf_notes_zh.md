# FFABwdSm100Index（DSA 反向）性能记录

本文记录 2026-11 针对 DSA 场景的两轮优化：测了什么、合入了什么、试过但没用的方案和原因，以及后续方向。目标是"不论序列长度都尽量变快"，所以所有结论都按序列长度分区间给出。

- 第一轮（提交 4f452746）：V 行异步落地 `gather_V_async`，主要提升 S≈64K–256K（第 4 节）。
- 第二轮：按 K 区间多趟处理 `num_k_passes`，主要提升 S≥196K（第 5 节）。

## 1. 测量方法

- 场景：sm90 契约下的 bench 数据（`exps/attn/cutedsl/run_index_sparse_sm100_benchmark.py` 的造数方式）。每个 q block 用 `randint` 生成排好序的 topk 个 K id，`nhk = 1`，一个 q block 的 token × q head 正好拼成 128 行 kernel Q tile。
  - `mqa128`：128 个 q head，qbs = 1。
  - `gqa8`：8 个 q head，qbs = 16。
  - `gqa16`：16 个 q head，qbs = 8。
- 只计主 kernel：把 `compile_cache` 包一层，用 cuda event 夹住主 kernel 的调用，多趟时把一次反向的各趟加起来，不含 pre/postprocess。另外报"端到端"：整个 `ffa_bwd_sm100_index` 调用，含索引分组、preprocess、postprocess。多趟模式的 host 侧开销更大，所以第二轮以端到端为准。
- 先冷却 3 s，再连续跑 N 次。"冷启动"取前 3 次平均，"持续"取后一半的中位数。S = 32K 时连续运行会撞 1100 W 功耗墙，持续值比冷启动慢 6–8%；长序列是访存瓶颈，两者基本一样。
- A/B：把 HEAD 版本拷成 `_head_bwd_sm100_index.py`（JIT 缓存名改成 `bwd_sm100_index_head`），在同一张卡上新旧交替跑。
- 机器上有别人的任务：开跑前先看 `nvidia-smi`。有一轮 GPU 4/7 被占满（240 W、100% 利用率），HEAD 自己在 S=131K 下从 71.6 ms 变成 175 ms，那一轮数据全部作废。
- 插桩：用内联 PTX 读 `%clock64`，在每个 pipeline 等待点累计周期，最后原子加到全局缓冲（`make_instr.py`）。插桩本身会改变时序：S=524288/topk=4096 下，插桩版里异步 V 比 HEAD 快 3%，不插桩时反而慢 3%。所以插桩只用来看分布，计时一律用不插桩的版本。
- ncu：这台机器 GPU 3 的 NVLink 有问题，nvperf 初始化扫描全卡时会挂住。用 `ncu-on-gpus.sh <物理卡号> <ncu 参数> python ...` 在私有 mount namespace 里只暴露选定的卡（不要选 3）。驱动脚本是 `prof_driver.py`：先跑一次反向触发编译，再跑一次给 ncu 采；多趟时用 `--launch-skip` 选到具体某一趟。

脚本都在 `/home/niubility2/lijin/ncu_reports/ffa_bwd_sm100_index_dsa/`：

| 脚本 | 用途 |
|---|---|
| `dsa_bench.py` | 主 kernel 和端到端计时。`SCEN S TOPK ITERS [module] [--kspan=N] [--unsorted] [--gva=0/1] [--nkp=P]` |
| `ab.sh` / `sweep.sh` | 多模块、多规模批量 A/B |
| `check_vs_head.py` | 和 HEAD 拷贝逐用例对比：dQ 逐位（多趟时为相对差）、dK/dV 相对差、NaN。`FFA_GVA=0/1` 强制 gather 方式，`FFA_NKP=P` 强制趟数 |
| `ncu-on-gpus.sh` / `prof_driver.py` | ncu 采集（见上） |
| `keymetrics.py` | 多份 ncu 报告的关键吞吐指标并排对比 |
| `role_stalls.py` | 从 ncu 的 SASS 导出（`--page source --csv --print-source sass`）按角色统计 stall |
| `check_partition.py` | 索引分组 kernel 与 stable sort 参考对比并计时 |
| `dq_accum_precision.py` | 模拟 dQ 跨趟用 bf16 累加时的误差 |
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

## 3. 三个区间的瓶颈（插桩 + ncu 结论）

**短序列（工作集 ≤ L2/2，例如 S ≤ 32K）：卡在 SM 往 L2 的出口，即 dK/dV 的 fp32 归约写。**
- ncu（S=32K/topk=2048）：L1TEX→crossbar 请求通道忙 83%，TMA 归约写出 5.35 TB/s（写字节峰值的 59%）；L2 原子单元输入只忙 46%，L2 整体 64%，DRAM 不到 4%，tensor pipe 37%。
- 第一轮插桩看到 reduce warp 54% 的时间卡在发 `cp.reduce.async.bulk` 上，当时归因到"L2 原子归约反压"，ncu 显示实际是每个 SM 自己的出口被占满。DSMEM 和寄存器 `red` 走的是同一个出口，所以这个区间离上限只剩约 10%，只有减少写回字节才能再快。
- HEAD 下 MMA warp 也有 18% 在等 K、10% 在等 V。可是 gather 一变快，等待就转移到 dV 累加器的释放上（326 → 676 周期/K tile），总时间不变。

**中间（L2/2 < 工作集 ≲ 3–4× L2，S≈64K–256K）：卡在 gather 的 DRAM 延迟。**
- HEAD 的 load_KV 每个 K tile 串行等两次（先等 K，再等 V），每次只有 16 KB 在路上。

**长序列（工作集 ≈ 6× L2，S=512K）：DRAM 带宽饱和。**
- HEAD 下 MMA warp 有 41% 在等 K、25% 在等 V；reduce warp 反而 55–60% 空闲。
- 但加深预取没用（见 6.6），说明是带宽而不是延迟。ncu（单趟 S=512K/topk=2048）：DRAM 读 3.38 + 写 2.18 TB/s，L2 命中率 16%，tensor pipe 20%。

**有效 L2 容量远小于 126.5 MB。** ncu 测"理想多趟"（Q 是 524288 个 token，索引只落在 65536 行、工作集 96 MB 以内）时，L2 命中率也只有 46%；上界实验里 K 范围 32768 行（48 MB）能到 808 TF，65536 行（96 MB）只有 687 TF。对这种被全部 SM 随机访问的数据，实际可用的 L2 大约 60 MB。可能是 B300 的 L2 分在两个 die 上，同一行在两边各缓存一份。这一点决定了第 5 节多趟方案的天花板。

## 4. 合入的优化：V 行异步落地（`gather_V_async`，按工作集自动开启）

### 改动

- HEAD 的 load_KV 每个 K tile：gather K → `wait_group(0)` → fence → arrive → gather V → `wait_group(0)` → fence → arrive。
- 开启 `gather_V_async` 后，K 保持原样；V 改为每个线程发完 cp.async 就执行 `cp.async.mbarrier.arrive.noinc`，然后直接去等下一个 sK stage。
  - sV 的 full barrier 改为 128 个线程到达。
  - MMA warp 在每次等到 sV full 之后执行 `fence.proxy.async`：cp.async 属于 generic proxy，UMMA 走 async proxy，而发射 cp.async 的线程从不等待，所以 fence 只能放在消费端。
- 下一个 K tile 的 `wait_group(0)` 会顺带把上一个 tile 的 V 也等掉。因为 V 早发了一个 K 的时间，通常已经落地。
- K 必须保持同步：S 需要的下一个 K，就在 dQ 释放 sK 之后马上要用，只有一个 tile 的提前量。实测 K 用异步到达在 32K 下慢 6%（见 6.1）。

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

## 5. 第二轮合入：按 K 区间多趟处理（`num_k_passes`，按工作集和 topk 自动开启）

### 思路

长序列的工作集远大于 L2，随机 gather 和 dK/dV 累加器的读改写都打到 DRAM。改成把 K token 切成 P 段（段长约 0.8×L2），kernel 跑 P 趟，第 c 趟所有 Q tile 只处理落在第 c 段的索引，这样一趟内被访问的 K/V 行和累加器行大部分留在 L2。代价是每个 Q tile 被访问 P 次（每次重新加载 Q/dO、流水重填），dQ 要跨趟累加。

### 改动

- **索引分组**：`magi_attention/utils/sparse_utils.py::partition_index_sparse_rows_by_k_chunk`，一个 Triton kernel 对每行做按段的计数排序，段内保持原顺序（结果确定），-1 padding 仍在尾部，同时输出每行每段的起点和长度。
  - 用 4 个 16 位字段打包进一个 int64，一次扫描同时给 4 段排名次；每行的起点和长度先攒成向量、只写一次（标量 `tl.store` 会被块内所有线程各执行一遍）。
  - 耗时：S=524K/topk=2048、8 段约 8–10 ms，topk=4096 约 20 ms。对未排序输入，`torch.sort` 要 28/62 ms。
- **整 tile 对齐**：第 c 趟在第 c 段最后一个整 K tile 处结束，剩下不足 64 个的 id 并入下一趟（这些 id 在下一趟会 miss L2），这样除最后一趟外每趟都是整 tile。没有这一步时，每趟平均多半个不满的 K tile。
- **kernel**（编译期开关 `k_chunk_pass`、`k_chunk_first`）：
  - 索引行从 `mTopkStart[qb, h]` 开始读（load_KV 和 reduce 两处，`_segment_ids`）。
  - dQ 以 fp32 分两半（每半 (128, 64)，正好放进 32 KB 的 sP/sdS 区域，R2S 复用 S 的分区）TMA 写到 fp32 累加器：第一趟 store（没有索引的 tile 写 0，因此累加器不用清零），之后各趟 reduce-add。
  - 多趟模式下 Q/dO 的 TMA 加载和 dQ 的 TMA store/reduce 都带 L2 evict_first，避免这两股只用一次的流把当前段挤出 L2。
- **host**：fp32 dQ 累加器用 token-major 的 `(rows, nh, hd)` 布局（head-major 摊平的 `(nh, rows*hd)` 在 S=524K MQA128 下超过 Int32），最后用 `dq.copy_` 转成 bf16。每趟按该段的大小重新决定 `gather_v_async`（一段约 0.76×L2，所以都会开启）。
- **额外显存**：fp32 dQ 累加器（S=524K MQA128 时 34 GB），以及一份分组后的索引（与输入索引同大小）。

### 自动规则 `num_k_passes_default`

工作集按 `total_k × nhk × head_dim × (2×元素字节 + 2×4)` 计算：

- 工作集 ≤ 2×L2：单趟。S=131K 时 2 趟只能毛赚 3–7%，扣掉多趟开销就没了。
- 否则趟数 P = ⌈工作集 / (0.8×L2)⌉，上限 16。B300、单 kv head、hd128 时，S=196K/262K/393K/524K 分别为 3/4/6/8 趟，段长都是 65536 行。
- 但如果 topk < 256×P（每次访问不足 4 个 K tile），退回单趟。例如 S=524K/topk=1024 强制 8 趟慢 21%，2 趟和 4 趟也分别慢 7% 和 9%。
- host 入口 `num_k_passes: Optional[int] = None`，None 表示自动；显式传入时按传入值切分。

### 结果（TFLOPS，冷启动；对比上一轮提交 4f452746）

| 场景 | 趟数 | HEAD 主 kernel | 新主 kernel | 变化 | HEAD 端到端 | 新端到端 | 变化 |
|---|---|---|---|---|---|---|---|
| MQA128 32K / 1024、2048、4096 | 1 | 819 / 857 / 866 | 819 / 857 / 855 | 持平 | 730 / 803 / 825 | 730 / 805 / 818 | 持平 |
| GQA8 32K / 4096 | 1 | 822 | 821 | 持平 | — | — | — |
| MQA128 65K / 2048 | 1 | 800 | 811 | 持平 | 761 | 768 | 持平 |
| MQA128 131K / 2048、4096 | 1 | 697 / 765 | 700 / 763 | 持平 | 668 / 740 | 671 / 738 | 持平 |
| GQA8 131K / 4096 | 1 | 811 | 811 | 持平 | 733 | 740 | 持平 |
| MQA128 196K / 2048 | 3 | 600 | 699 | +16.5% | 580 | 649 | +11.9% |
| MQA128 262K / 1024 | 4 | 465 | 515 | +10.8% | 447 | 472 | +5.6% |
| MQA128 262K / 2048 | 4 | 528 | 648 | +22.7% | 513 | 602 | +17.3% |
| MQA128 262K / 4096 | 4 | 683 | 718 | +5.1% | 664 | 677 | +2.0% |
| MQA128 393K / 2048 | 6 | 482 | 578 | +19.9% | 469 | 537 | +14.5% |
| MQA128 524K / 1024 | 1 | 414 | 414 | 持平 | 400 | 400 | 持平 |
| MQA128 524K / 2048 | 8 | 459 | 507 | +10.5% | 448 | 476 | +6.3% |
| MQA128 524K / 4096 | 8 | 537 | 634 | +18.1% | 526 | 597 | +13.5% |
| GQA8 524K / 4096 | 8 | 541 | 667 | +23.3% | 522 | 619 | +18.6% |

"持平"指差异在 ±1% 以内（32K/4096 的冷启动在 HEAD 自己的重复测量里也在 855–869 TF 之间波动）。单趟场景走的是和 HEAD 相同的代码路径。GQA8 32K 的端到端只有约 2 ms，主要是 host 调度噪声，没有列出。

### 正确性

- 单趟：dQ 与 HEAD 逐位一致。
- 多趟：dQ 与 HEAD 的相对差约 1e-3（bf16 一个 ulp 量级，来自 fp32 累加顺序），dK/dV 与以前同一量级。
- 新增测试：
  - `tests/test_utils/test_sparse_utils.py`：分组函数与 stable sort 参考逐元素一致，覆盖整行、带 -1 尾巴、段数不是 4 的倍数、单段。
  - `test_index_sparse_bwd_sm100.py`：
    - 强制 2/3 趟跑四种配置（MQA128、多 segment 的 MQA4、两个 batch 的 -1 padding 行、两个 kv head 分开），和 SDPA 参考对比。
    - 单测自动规则（2×L2 阈值、趟数、256 id 下限、16 趟上限、kv head 累加）。

## 6. 试过但没有合入的方案

### 6.1 K 和 V 都用异步到达

| S | HEAD | 异步 K+V |
|---|---|---|
| 32K | 12.78 | 13.57（−6%） |
| 131K | 71.5 | 62.7（+12%） |
| 524K | 382.5 | 385.8（持平） |

去掉 MMA 端的 fence 也一样慢（13.62），所以慢不是 fence 造成的，而是 K 的异步到达完成得比"等待 + fence + arrive"晚。K 只有一个 tile 的提前量，对这点延迟很敏感。

### 6.2 每个 tile 先发 V 再发 K

- K 也异步时：32K 13.94 ms，比先 K 更差。
- K 同步、V 异步时：32K 13.33 ms，524K/4096 745 ms（−14%）。

推测原因：K 的 `wait_group(0)` 会把刚发出去的 V 一起等掉，而且 V 更早发出会更早和归约争带宽。

### 6.3 K、V 拆给不同 warp（warp 0–1 搬 K，warp 2–3 搬 V）

| 方案 | 32K | 131K |
|---|---|---|
| V 异步 | 13.19（−3.2%） | 62.9 |
| V 同步 | 12.94（−1.2%） | 63.6 |

两种都不如同一组 warp 上"K 同步、V 异步"（12.87 / 63.2）。规律是：gather 并发度越高，L2 区间越慢。

### 6.4 L2 cache hint（S=131K/topk=2048）

| 配置 | ms |
|---|---|
| 不加 | 62.9 |
| gather evict_first | 66.0（−5%） |
| gather evict_first + reduce evict_last | 64.1 |
| reduce evict_last | 63.0 |

K/V 行会被 topk 个 Q tile 反复读到，L2 默认替换已经在帮忙，强制 evict_first 是负收益。

实现时的坑：给 cp.async 加 `L2::cache_hint` 要走内联 PTX，而 swizzled smem 指针的 `.toint()` 拿到的是未 swizzle 的地址，必须手动做一次 `a ^ (((a >> 7) & 7) << 4)`（SW128，buffer 按 1024 B 对齐），否则结果全错。

### 6.5 Q tile 边界开销

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

### 6.6 kv_stage 3 + reduce_chunk 16（暂存区让出 32 KB 给 K/V 第三个 stage）

| S/topk | HEAD | 调整后 |
|---|---|---|
| 32K/2048 | 12.77 | 13.89（−9%） |
| 131K/2048 | 72.0 | 71.4（无变化） |
| 524K/4096 | 656 | 653（无变化） |

短序列下 reduce 是瓶颈，暂存块变小更慢；长序列加深预取没有收益，说明长序列是 DRAM 带宽瓶颈而不是延迟。

### 6.7 load_KV 软件流水预读下一个 tile 的行号

第一版把预读放在 K 的 `fence.proxy.async` 之前，32K 慢 5.8%：proxy fence 会等本线程之前所有的访存完成，行号读取于是落到了 sK 提交路径上。挪到 V 发出之后，结果和 HEAD 持平（±0.5%），说明行号延迟本来就被藏住了，没有合入。

### 6.8 其他

- 索引不排序：32K 下慢 2.5%（847 vs 868 TF）。现在 kernel 不依赖排序。
- 全局相位对齐的 K 扫描（所有 CTA 同时扫 K 空间的同一段）：只做了分析，没有实现。每个 wave 内同一个 K 行只会被摸到 148·topk/S 次，S=131K、topk=2048 时约 2.3 次。对齐后变成 LRU 最不擅长的循环扫描，可能比随机顺序的命中率还低。只有在 148·topk/S 远大于 1 时才可能有收益。

### 6.9 第二轮：多趟方案中试过的变体

- **dQ 跨趟用 bf16 累加**（直接对 bf16 输出做 TMA reduce-add）：字节减半，还能省掉累加器和最后的转换，但模拟显示 8 趟时 dQ 的平均误差是单趟的 2.6 倍，最大误差 4 倍，属于数值回退，放弃。
- **dQ 在 SM 里读改写**（TMA load 累加器 → 相加 → store，避开 L2 原子单元）：消融显示，同样字节量的普通 fp32 TMA store 也要 345 ms（归约 368，完全不写回 300，S=524K/topk=2048），读改写只会更慢，没做。dQ 写回的主要代价是这股流在 L2 里的写分配会挤掉当前段；加 evict_first 后归约降到 348.5 ms，和普通 store 持平。
- **去掉 dQ 归约后对 TMA 读完 smem 的等待**：只快 0.5%，说明代价在字节量而不在延迟。
- **趟数**（最终代码，主 kernel 冷启动）：S=524K/topk=2048 时 6/8/10 趟为 507/507/487 TF（早期版本里 4 趟和 12 趟都更差）；S=262K/topk=4096 时 2/4/8 趟为 670/720/677 TF。更多趟能让一段更接近真正放进 L2（实测有效容量约 60 MB），但每次访问的固定开销会压过收益。
- **分组 kernel 提速**：
  - 单程序单行版要 11.2 ms（S=524K/topk=2048）；
  - 改成持久化循环没有变化，说明 CTA 调度不是瓶颈；
  - 改成 4 段一次的 int64 扫描为 10 ms；
  - 每行的起点和长度改为向量一次写出，降到约 8 ms；
  - 段长改 2 的幂、用移位代替除法，只快 15%，没合入；
  - int32 两段一次的扫描在 topk=4096 下反而更慢（31 ms）。
  - 拆开单测时，读写、扫描、分散写合计约 4–5 ms，剩余部分还没定位。

## 7. 后续方向（按预期收益排序）

1. **多趟模式的每次访问固定开销**（约 3 µs/次：Q/dO 重载、流水重填、dQ 写回）。每段只有约 4–8 个 K tile，这是多趟收益的主要上限。需要额外 smem 做 Q/dO 双缓冲，才能和上一个 tile 重叠，或者重新规划 smem。
2. **多趟模式的 host 侧开销**：S=524K/topk=2048 下端到端比主 kernel 多约 23 ms（HEAD 约 10 ms），其中分组约 8–10 ms、dQ 的 fp32→bf16 转换约 7 ms。分组 kernel 理论上 2 ms 以内；转换可以并进最后一趟（读累加器 + 加本趟部分和 + 写 bf16）。
3. **短序列：只有减少 dK/dV 写回字节才可能再快。** ncu 显示 SM 出口已占满 83%。DSA bench 里每个索引行只对应一个 Q tile，没有可以合并的写回；qbs×G 是 128 的整数倍以上时，可以让两个 Q tile 在 TMEM 里先合并 dK/dV 再写回（即之前 history 里的方向 B）。
4. **小 topk 的 Q tile 固定开销**：dQ epilogue 约 0.8 µs，流水重填约 2 µs，都需要额外 smem 才能和上一个 tile 重叠。

## 8. 复现

```bash
cd /home/niubility2/lijin/ncu_reports/ffa_bwd_sm100_index_dsa
# 先生成 HEAD 拷贝作为 A/B 基线
git -C /home/niubility2/lijin/MagiAttention show HEAD:magi_attention/kernel/cutedsl/ffa_bwd_sm100_index.py \
  | sed 's/get_jit_cache("bwd_sm100_index")/get_jit_cache("bwd_sm100_index_head")/' \
  > /home/niubility2/lijin/MagiAttention/magi_attention/kernel/cutedsl/_head_bwd_sm100_index.py
./ab.sh 1 "_head_bwd_sm100_index ffa_bwd_sm100_index" "mqa128 131072 2048 6" "mqa128 524288 2048 2"
CUDA_VISIBLE_DEVICES=1 FFA_GVA=1 python check_vs_head.py
CUDA_VISIBLE_DEVICES=1 FFA_NKP=3 python check_vs_head.py
# ncu（不要用 GPU 3）：多趟 8 段，采第二次反向的第 2 趟
./ncu-on-gpus.sh 2 -k regex:FFABwdSm100Index --launch-skip 9 --launch-count 1 --set full \
  -o ncu/pass1 -f python prof_driver.py mqa128 524288 2048 8
python keymetrics.py ncu/pass1.ncu-rep
# 用完删掉 cutedsl/ 下的 _head_*.py、_ablate_*.py、_ins_*.py、_kn_*.py、_exp_*.py
```

`cutedsl/` 下多出来的 `.py` 文件会改变 JIT 磁盘缓存的源码指纹（如果开启了磁盘缓存），用完要删。
