# GitHub Actions Workflow Examples

This directory contains example workflow configurations demonstrating various runner expansion strategies for the MagiAttention project.

## Available Examples

### 1. `lint-with-larger-runner.yaml.example`

**Purpose**: Demonstrates how to use larger GitHub-hosted runners for faster lint execution.

**Key Changes**:
- Uses `ubuntu-latest-4-cores` instead of `ubuntu-latest`
- Suitable for projects that need faster CI/CD pipeline

**When to Use**:
- Lint tasks are taking too long
- You have GitHub Team or Enterprise plan
- Want to reduce developer wait time

### 2. `gpu-tests-self-hosted.yaml.example`

**Purpose**: Shows how to set up self-hosted GPU runners for testing MagiAttention's GPU-dependent features.

**Key Features**:
- Uses self-hosted runners with GPU and CUDA tags
- Includes conditional execution (only on main branch or with specific label)
- Includes cleanup steps for GPU resources
- Has timeout protection

**When to Use**:
- Need to run tests that require Hopper GPUs
- Want to test actual hardware performance
- GitHub-hosted runners don't support required GPU hardware

### 3. `sphinx-optimized.yaml.example`

**Purpose**: Optimized documentation build with caching and larger runner.

**Key Features**:
- Uses `ubuntu-latest-4-cores` for faster builds
- Implements caching for pip dependencies and Sphinx builds
- Reduces build time significantly

**When to Use**:
- Documentation builds are slow
- Want to reduce redundant dependency downloads
- Need faster iteration on documentation changes

### 4. `hybrid-strategy.yaml.example`

**Purpose**: Demonstrates a comprehensive hybrid approach using different runners for different jobs.

**Key Features**:
- Quick lint on standard runner for fast feedback
- Full lint matrix on larger runner
- CPU tests on standard runner
- GPU tests on self-hosted runner (conditional)
- Documentation build on larger runner
- Uses job dependencies for optimal workflow

**When to Use**:
- Want to optimize cost and performance
- Different jobs have different resource requirements
- Need a production-ready CI/CD setup

## How to Use These Examples

1. **Choose an example** that fits your needs
2. **Copy the example file** and remove the `.example` extension:
   ```bash
   cp .github/workflows/examples/lint-with-larger-runner.yaml.example .github/workflows/lint-with-larger-runner.yaml
   ```
3. **Modify as needed** for your specific requirements
4. **Commit and push** to activate the workflow

## Prerequisites

### For Larger GitHub-Hosted Runners
- GitHub Team or Enterprise plan
- No additional setup required

### For Self-Hosted Runners
1. Set up a server with required hardware (GPU, CUDA, etc.)
2. Install the GitHub Actions runner:
   ```bash
   # Download and extract runner
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-linux-x64-2.320.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.320.0/actions-runner-linux-x64-2.320.0.tar.gz
   tar xzf ./actions-runner-linux-x64-2.320.0.tar.gz
   
   # Configure runner
   ./config.sh --url https://github.com/SandAI-org/MagiAttention --token YOUR_TOKEN
   
   # Install as service
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```
3. Add appropriate labels (e.g., `gpu`, `cuda`, `linux`, `x64`)
4. Verify runner appears in repository Settings > Actions > Runners

## Additional Resources

For comprehensive information about runner expansion strategies, see:
- [GitHub Actions Runner Guide](../../../docs/source/github_actions_runners.md)
- [Contributing Guide](../../../CONTRIBUTING.md)

## Tips

- **Start small**: Begin with one example and gradually expand
- **Monitor costs**: Track runner usage to manage expenses
- **Security**: Use dedicated servers for self-hosted runners, never development machines
- **Labels**: Use descriptive labels for self-hosted runners to target them precisely
- **Timeouts**: Always set job timeouts to prevent stuck jobs
- **Caching**: Implement caching for dependencies to speed up builds
