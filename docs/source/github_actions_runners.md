# GitHub Actions Runner 扩容指南 / Runner Capacity Expansion Guide

本文档介绍如何为 MagiAttention 项目扩展 GitHub Actions runner 的容量。

This document explains how to expand GitHub Actions runner capacity for the MagiAttention project.

## 当前配置 / Current Configuration

当前项目使用 GitHub 托管的 `ubuntu-latest` runner 运行 CI/CD 工作流：

Currently, the project uses GitHub-hosted `ubuntu-latest` runners for CI/CD workflows:

- **lint.yaml**: 使用标准 `ubuntu-latest` runner 进行代码格式检查
- **sphinx.yaml**: 使用 `ubuntu-latest` runner 配合 NVIDIA PyTorch Docker 容器构建文档

## 扩容方案 / Expansion Options

### 方案 1: 使用更大的 GitHub 托管 Runner / Option 1: Use Larger GitHub-Hosted Runners

GitHub 提供了更大规格的托管 runner，适合需要更多 CPU、内存或存储的任务。

GitHub provides larger hosted runners with more CPU, memory, and storage.

#### 可用规格 / Available Sizes:

```yaml
runs-on: ubuntu-latest-4-cores  # 4 核心 / 4 cores, 16 GB RAM
runs-on: ubuntu-latest-8-cores  # 8 核心 / 8 cores, 32 GB RAM
runs-on: ubuntu-latest-16-cores # 16 核心 / 16 cores, 64 GB RAM
```

#### 示例修改 / Example Modification:

修改 `.github/workflows/lint.yaml`:

```yaml
jobs:
  lint:
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    runs-on: ubuntu-latest-4-cores  # 升级到 4 核心 runner
    steps:
      # ... rest of the configuration
```

**优点 / Advantages:**
- 设置简单，只需修改一行配置 / Simple setup, only one line change
- GitHub 负责维护和更新 / GitHub handles maintenance and updates
- 无需额外基础设施 / No additional infrastructure needed

**缺点 / Disadvantages:**
- 需要 GitHub Team 或 Enterprise 计划 / Requires GitHub Team or Enterprise plan
- 费用较高 / Higher cost
- 仍受 GitHub 资源限制 / Still subject to GitHub's resource limits

### 方案 2: 使用自托管 Runner / Option 2: Use Self-Hosted Runners

自托管 runner 让您可以完全控制运行环境，适合有特殊硬件需求（如 GPU）的场景。

Self-hosted runners give you full control over the execution environment, ideal for special hardware requirements (e.g., GPUs).

#### 设置步骤 / Setup Steps:

1. **在服务器上安装 runner / Install runner on your server:**

```bash
# 下载 runner / Download runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.320.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.320.0/actions-runner-linux-x64-2.320.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.320.0.tar.gz

# 配置 runner / Configure runner
./config.sh --url https://github.com/SandAI-org/MagiAttention --token YOUR_TOKEN

# 启动 runner / Start runner
./run.sh
```

2. **将 runner 注册为服务（推荐）/ Register runner as a service (recommended):**

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

3. **为 runner 添加标签 / Add labels to runner:**

在配置时添加标签，如 `gpu`, `high-memory`, `cuda` 等。

Add labels during configuration, such as `gpu`, `high-memory`, `cuda`, etc.

4. **修改工作流使用自托管 runner / Modify workflow to use self-hosted runner:**

```yaml
jobs:
  lint:
    runs-on: [self-hosted, linux, x64]  # 使用自托管 runner
    # 或使用特定标签 / Or use specific labels
    # runs-on: [self-hosted, gpu, cuda]
    steps:
      # ... rest of the configuration
```

#### GPU Runner 配置示例 / GPU Runner Configuration Example:

对于需要 GPU 的测试（如 MagiAttention 的核心功能测试）：

For tests requiring GPUs (like MagiAttention's core functionality tests):

```yaml
jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - uses: actions/checkout@v4
      - name: Run GPU tests
        run: |
          pytest tests/ -v -m gpu
```

**优点 / Advantages:**
- 完全控制硬件和软件环境 / Full control over hardware and software
- 可使用 GPU、大内存等特殊硬件 / Can use GPUs, large memory, etc.
- 长期来看成本更低 / Lower cost in the long run
- 可缓存依赖和构建产物 / Can cache dependencies and build artifacts

**缺点 / Disadvantages:**
- 需要自己维护基础设施 / Requires infrastructure maintenance
- 需要处理安全问题 / Need to handle security concerns
- 初始设置较复杂 / More complex initial setup

### 方案 3: 使用并行执行策略 / Option 3: Use Parallel Execution Strategies

通过矩阵策略并行运行多个任务，充分利用 runner 资源。

Use matrix strategies to run multiple jobs in parallel, maximizing runner utilization.

#### 当前配置 / Current Configuration:

```yaml
jobs:
  lint:
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    runs-on: ubuntu-latest
```

这已经在并行运行 3 个 Python 版本的测试。

This already runs tests for 3 Python versions in parallel.

#### 扩展示例 / Extended Example:

```yaml
jobs:
  lint:
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        os: [ubuntu-latest, ubuntu-latest-4-cores]
      fail-fast: false  # 即使某个任务失败也继续其他任务
      max-parallel: 6   # 最大并行任务数
    runs-on: ${{ matrix.os }}
    steps:
      # ... rest of the configuration
```

**优点 / Advantages:**
- 加快 CI/CD 流程 / Faster CI/CD pipeline
- 更好的资源利用率 / Better resource utilization
- 易于配置 / Easy to configure

**缺点 / Disadvantages:**
- 可能达到并发限制 / May hit concurrency limits
- 增加总计算时间（但减少等待时间）/ Increases total compute time (but reduces wait time)

### 方案 4: 混合策略 / Option 4: Hybrid Strategy

结合多种方案，针对不同任务使用不同类型的 runner。

Combine multiple approaches, using different runner types for different tasks.

#### 示例配置 / Example Configuration:

```yaml
jobs:
  # 轻量级任务使用标准 runner / Use standard runner for lightweight tasks
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run linters
        run: pre-commit run --all-files

  # 文档构建使用中等规格 runner / Use medium runner for doc builds
  docs:
    runs-on: ubuntu-latest-4-cores
    container:
      image: nvcr.io/nvidia/pytorch:25.05-py3
    steps:
      - uses: actions/checkout@v4
      - name: Build docs
        run: |
          cd docs
          make clean
          sphinx-build -b html source/ html/

  # GPU 测试使用自托管 GPU runner / Use self-hosted GPU runner for GPU tests
  gpu-tests:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - uses: actions/checkout@v4
      - name: Run GPU tests
        run: pytest tests/ -v -m gpu
```

## 最佳实践 / Best Practices

### 1. 资源优化 / Resource Optimization

```yaml
# 对于简单任务使用较小的 runner / Use smaller runners for simple tasks
jobs:
  quick-lint:
    runs-on: ubuntu-latest
    
  # 对于耗时任务使用较大的 runner / Use larger runners for time-consuming tasks
  build-and-test:
    runs-on: ubuntu-latest-8-cores
```

### 2. 缓存策略 / Caching Strategy

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: actions/cache@v4
    with:
      path: |
        ~/.cache/pip
        ~/.cache/pre-commit
      key: ${{ runner.os }}-${{ hashFiles('**/requirements*.txt') }}
```

### 3. 条件执行 / Conditional Execution

```yaml
jobs:
  expensive-test:
    # 仅在主分支或特定标签时运行 / Only run on main branch or specific labels
    if: github.ref == 'refs/heads/main' || contains(github.event.pull_request.labels.*.name, 'run-gpu-tests')
    runs-on: [self-hosted, gpu]
```

### 4. 超时设置 / Timeout Settings

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30  # 防止任务卡住 / Prevent stuck jobs
```

### 5. 安全考虑 / Security Considerations

对于自托管 runner：

For self-hosted runners:

- 使用专用服务器，不要在开发机器上运行 / Use dedicated servers, don't run on development machines
- 定期更新 runner 软件 / Regularly update runner software
- 为公共仓库使用隔离的环境（如容器或 VM）/ Use isolated environments (containers or VMs) for public repositories
- 限制 runner 的网络访问 / Limit network access for runners
- 使用 GitHub Secrets 存储敏感信息 / Use GitHub Secrets for sensitive information

## 监控和维护 / Monitoring and Maintenance

### 查看 Runner 状态 / Check Runner Status

在 GitHub 仓库中导航到：
Navigate in GitHub repository to:

```
Settings > Actions > Runners
```

### 监控指标 / Monitoring Metrics

关注以下指标：
Monitor the following metrics:

- **队列时间 / Queue Time**: 任务等待 runner 的时间
- **执行时间 / Execution Time**: 任务实际运行时间
- **成功率 / Success Rate**: 任务成功完成的百分比
- **资源利用率 / Resource Utilization**: CPU、内存、磁盘使用情况

### 日志和调试 / Logging and Debugging

启用详细日志：
Enable verbose logging:

```yaml
jobs:
  debug:
    runs-on: ubuntu-latest
    steps:
      - name: Enable debug logging
        run: |
          echo "ACTIONS_STEP_DEBUG=true" >> $GITHUB_ENV
          echo "ACTIONS_RUNNER_DEBUG=true" >> $GITHUB_ENV
```

## 推荐配置 / Recommended Configuration

针对 MagiAttention 项目的推荐配置：

Recommended configuration for the MagiAttention project:

1. **Lint 工作流 / Lint Workflow**: 保持使用 `ubuntu-latest`（已足够）
2. **文档构建 / Documentation Build**: 升级到 `ubuntu-latest-4-cores` 以加速 Sphinx 构建
3. **GPU 测试 / GPU Tests**: 添加自托管 GPU runner（对于需要 Hopper GPU 的测试）
4. **发布构建 / Release Builds**: 使用 `ubuntu-latest-8-cores` 或自托管 runner

## 参考资料 / References

- [GitHub Actions 官方文档](https://docs.github.com/en/actions)
- [关于自托管 runner](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners)
- [GitHub-hosted runner 规格](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners)
- [Runner 安全最佳实践](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
