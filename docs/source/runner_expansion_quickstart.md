# Runner 扩容快速开始 / Runner Expansion Quick Start

这是一个快速参考指南，帮助您快速开始扩展 GitHub Actions runner 容量。

This is a quick reference guide to help you quickly start expanding GitHub Actions runner capacity.

## 快速决策树 / Quick Decision Tree

### 我应该选择哪种方案？ / Which option should I choose?

```
需要 GPU 测试吗？ / Need GPU tests?
├─ 是 / Yes → 使用自托管 Runner / Use Self-Hosted Runners (方案 2 / Option 2)
│
└─ 否 / No → CI/CD 运行时间长吗？ / Is CI/CD slow?
    ├─ 是 / Yes → 预算充足吗？ / Good budget?
    │   ├─ 是 / Yes → 使用更大的 GitHub 托管 Runner / Use Larger GitHub-Hosted Runners (方案 1 / Option 1)
    │   └─ 否 / No → 使用并行执行策略 / Use Parallel Execution (方案 3 / Option 3)
    │
    └─ 否 / No → 优化现有配置 / Optimize existing config
        └─ 添加缓存 / Add caching
        └─ 条件执行 / Conditional execution
```

## 5 分钟快速开始 / 5-Minute Quick Start

### 方案 1: 使用更大的 Runner (最简单) / Option 1: Use Larger Runner (Easiest)

**1 步完成：** / **Complete in 1 step:**

```bash
# 编辑 .github/workflows/lint.yaml
# Edit .github/workflows/lint.yaml
# 将 runs-on: ubuntu-latest 改为 / Change runs-on: ubuntu-latest to
runs-on: ubuntu-latest-4-cores
```

✅ **完成！** / **Done!**

### 方案 2: 设置自托管 GPU Runner / Option 2: Setup Self-Hosted GPU Runner

**3 步完成：** / **Complete in 3 steps:**

```bash
# 步骤 1: 下载并配置 runner / Step 1: Download and configure runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.320.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.320.0/actions-runner-linux-x64-2.320.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.320.0.tar.gz
./config.sh --url https://github.com/SandAI-org/MagiAttention --token YOUR_TOKEN

# 步骤 2: 安装为服务 / Step 2: Install as service
sudo ./svc.sh install
sudo ./svc.sh start

# 步骤 3: 复制示例工作流 / Step 3: Copy example workflow
cp .github/workflows/examples/gpu-tests-self-hosted.yaml.example \
   .github/workflows/gpu-tests.yaml
```

✅ **完成！** / **Done!**

### 方案 3: 使用混合策略 / Option 3: Use Hybrid Strategy

**直接使用示例：** / **Use example directly:**

```bash
# 复制混合策略示例 / Copy hybrid strategy example
cp .github/workflows/examples/hybrid-strategy.yaml.example \
   .github/workflows/ci-hybrid.yaml

# 根据需要编辑 / Edit as needed
vim .github/workflows/ci-hybrid.yaml
```

✅ **完成！** / **Done!**

## 常见场景快速方案 / Quick Solutions for Common Scenarios

### 场景 1: Lint 很慢 / Scenario 1: Lint is Slow

```yaml
# 简单修改 / Simple change
runs-on: ubuntu-latest-4-cores  # 加速 4x / 4x faster
```

### 场景 2: 文档构建很慢 / Scenario 2: Documentation Build is Slow

```bash
# 使用优化的 Sphinx 工作流 / Use optimized Sphinx workflow
cp .github/workflows/examples/sphinx-optimized.yaml.example \
   .github/workflows/sphinx.yaml
```

包含：/ Includes:
- ✅ 更大的 runner / Larger runner
- ✅ 依赖缓存 / Dependency caching
- ✅ 构建缓存 / Build caching

### 场景 3: 需要测试 GPU 功能 / Scenario 3: Need to Test GPU Features

```yaml
# 使用自托管 GPU runner / Use self-hosted GPU runner
jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/ -v -m gpu
```

### 场景 4: 有多种类型的任务 / Scenario 4: Multiple Types of Jobs

```bash
# 使用混合策略 / Use hybrid strategy
cp .github/workflows/examples/hybrid-strategy.yaml.example \
   .github/workflows/ci.yaml
```

包含：/ Includes:
- ✅ 轻量级任务用标准 runner / Standard runner for lightweight tasks
- ✅ 重量级任务用大 runner / Large runner for heavy tasks
- ✅ GPU 任务用自托管 runner / Self-hosted runner for GPU tasks

## 成本估算 / Cost Estimation

### GitHub 托管 Runner 费用 / GitHub-Hosted Runner Costs

| Runner 类型 / Type | 倍率 / Multiplier | 示例：运行 1 小时 / Example: 1 hour |
|-------------------|------------------|------------------------------|
| ubuntu-latest | 1x | 基准成本 / Base cost |
| ubuntu-latest-4-cores | ~4x | 约 4 倍成本 / ~4x cost |
| ubuntu-latest-8-cores | ~8x | 约 8 倍成本 / ~8x cost |
| ubuntu-latest-16-cores | ~16x | 约 16 倍成本 / ~16x cost |

> **提示 / Tip**: 虽然大 runner 成本更高，但通常能更快完成，总体可能更便宜
> 
> Although larger runners cost more per minute, they often complete faster, potentially reducing overall cost

### 自托管 Runner / Self-Hosted Runners

| 项目 / Item | 成本 / Cost |
|------------|-----------|
| 硬件 / Hardware | 一次性投资 / One-time investment |
| 维护 / Maintenance | 持续成本 / Ongoing cost |
| GitHub 使用费 / GitHub usage | 免费 / Free |

> **提示 / Tip**: 长期来看，自托管通常更便宜
> 
> Self-hosted is usually cheaper in the long run

## 下一步 / Next Steps

📚 **详细文档 / Detailed Documentation**: [GitHub Actions Runner Guide](./github_actions_runners.md)

📁 **示例工作流 / Example Workflows**: [Workflow Examples](../../.github/workflows/examples/)

🔧 **贡献指南 / Contributing Guide**: [CONTRIBUTING.md](../../CONTRIBUTING.md)

## 故障排除 / Troubleshooting

### 问题: Runner 显示离线 / Issue: Runner Shows Offline

```bash
# 检查 runner 状态 / Check runner status
sudo ./svc.sh status

# 重启 runner / Restart runner
sudo ./svc.sh stop
sudo ./svc.sh start
```

### 问题: 找不到 Self-Hosted Runner / Issue: Can't Find Self-Hosted Runner

1. 检查 runner 标签 / Check runner labels
2. 确保工作流中的标签匹配 / Ensure workflow labels match
3. 查看 Settings > Actions > Runners

### 问题: 任务卡住 / Issue: Jobs Get Stuck

添加超时：/ Add timeout:

```yaml
jobs:
  my-job:
    timeout-minutes: 30  # 防止卡住 / Prevent stuck jobs
```

## 获取帮助 / Get Help

- 📖 查看完整文档 / See full documentation: [github_actions_runners.md](./github_actions_runners.md)
- 💬 提出问题 / Ask questions: [GitHub Issues](https://github.com/SandAI-org/MagiAttention/issues)
- 📧 联系团队 / Contact team: See [README.md](../../README.md#acknowledgement)
