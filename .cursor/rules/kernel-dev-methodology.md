---
description: "CUDA kernel development, testing, and debugging methodology for MagiAttention"
globs: "**/*"
alwaysApply: true
---

重点！环境：所有命令必须 source /root/.magi_env（已配好 CUDA_HOME=cuda-13.0）。不带 env 跑会导致：1) SM90 deterministic hang（sm=0）；2) wgmma 串行导致 bench 不准。选 GPU 时不要固定用卡 0（经常有人占用），用 nvidia-smi 找 gpu util 0% 且显存占用最小的卡（bench 要求空闲卡，利用率 0% 且显存尽量无占用）。

重点！预编译：批量测试或 benchmark 前一定要预先并行编译 kernel（尤其清理 cache 之后）。写多个简单 test case 覆盖所有待测 config，用 CPU 并行一口气编完（并行度=待测 config 总数），之后再执行测试。参考：.tmp/062-tma2d-relax/precompile_all.py 或 .tmp/precompile_128x128.py。precompile 的目标是让每个 test case 在 1min 内完成，超时基本就是 kernel hang 而非编译慢（通过 .so 生成时间判断）。注意事项：1) precompile 后跑测试时不要开 MAGI_ATTENTION_FORCE_JIT_BUILD=1（否则又重复编译）；2) cpp 代码改动后及时删旧 cache 再 precompile；3) 并行编译时可能争抢 128hd_common 共享库构建产生 race condition，解法是先单独编一个让 common 就绪，再全部并行（若 cache 已有 common 则无此问题）；4) 检查是否有残余挂起的 compile 进程导致 .so 损坏；5) splitcompile 和 O3 等编译配置不要动，改了可能混入异常 cache。

重点！benchmark：验证 dense perf 是否回退一定要用 exps/attn/run_benchmark.py（调整里面测试内容，一般只测 ffa，dense 只测 2048/4096/8192/16384，full mask 不是 causal！），不要自己写 bench 文件。其他场景也应用 exps/attn/ 下已有代码：IndexAttn 用 run_index_attn_comparison_benchmark.py，SparseLoad 用 run_sparse_load_benchmark.py，causal partition 参考 bench_causal_partition.py。基线对比历史数据：PR #320（3 次运行取平均值，忽略 seqlen=1024）。benchmark 必须用 cuda event 计时（已有基础设施都用了，精确不易出错）。如果经典场景低于 10-50 TFLOPS 或低于预期 2-3x，多半是 bench 写法有问题或 lru_cache 未命中正确 kernel（检查 env var 是否在 cache key 外被读取）。

Sparse vs Dense 对比场景：选 topk=kvseqlen=32k，qhead=128 + PackGQA + kblocksize=128，此时 tload/store/mma/tile 流程一致，差异即为 sparse 框架额外开销，再用 ncu 对比找优化点。IndexAttn 场景：nhq=128, nhk=1, PackGQA, topk=2048。SparseLoad 场景：nhq=128, nhk=1, hd=128, qblk=1, kblk=128, effective_kv=2048, PackGQA。

ncu 关键指标：l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st（local spill）、launch__registers_per_thread（寄存器数）、sass__inst_executed_per_opcode（指令分类差异）。性能分析依据：ncu 对比两版本指标和指令分布，判断是数据搬运慢还是计算慢，有依据地调整 producer/consumer 配比。cuobjdump 做静态分析仅供参考（看 reg 分布），不如 ncu 的 local_op_ld/st 指标扎实。寄存器平衡：producer 减少的数量须与 mma 增加的数量对上（加权平均 168 的问题、非 8 整倍数问题），否则会导致 hang。

kernel 问题调试方法论（hang / 数值不对 / illegal memory）：少静态分析，多跑多打印拿事实。手段：1) printf 观察 warpN 卡在哪步；2) compute-sanitizer 定位 illegal mem；3) ncu 看效率和寄存器；4) cuobjdump 看 SASS。快速定位：在有问题的编译结果上多测几组不同 shape/config，对比缩小范围。

遇到 kernel hang：判断编译卡住还是执行卡住——检查 .so 是否已生成及时间戳。未生成=编译前卡住；很早生成=执行时 hang。常见原因：producer/consumer barrier 数量不匹配。precompile 后简单 case 超 1min 未完成即判定 kernel hang。

遇到 CI test 超时：1) 本地单卡执行报错的 test_xxx（world_size 改 1，去 @run_with_mp）；2) 每 case 前后打印 [case] start/finish + datetime，每 2min 检查日志定位 hang case；3) 写 repro_xxx.py 小场景复现；4) 修复后确认 repro 通过再跑完整 test 无回归。关键：先在未修改代码上复现（证明 bug 存在），再修复后验证（证明修复有效）。

排查清单：清 JIT cache（{WORKSPACE_BASE}/.cache/magi_attention/）；检查 AOT 残留（magi_attention/lib/ 下 stale .so）；回退 main 对比；开 compute-sanitizer；缩小到最小复现场景。

核心测试位置（不要自己写，用已有的）：Dense 正确性 tests/test_attn/test_flex_flash_attn.py 中 TestFlexFlashAttn 类；Sparse 正确性 tests/test_attn/test_block_sparse.py 中 TestBlockSparseSimple 类和 tests/test_attn/test_index_sparse.py 中 TestIndexSparseSimple 类；TFLOPS 效率测试 exps/attn/sparse/bench_sparse_analysis.py；Dense baseline benchmark 用 exps/attn/dense/run_benchmark.py。
