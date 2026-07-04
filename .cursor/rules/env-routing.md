---
description: "Multi-workspace environment routing and governance"
globs: "**/*"
alwaysApply: true
---

多个环境：三台机器 h100-6、h100-5、h100-1（完整主机名 Sensecore-bma-h100-6 等，已在 /root/.ssh/config 配好 ProxyJump 跳板，私钥 /home/niubility2/cenzhiyao/tmp-2.txt），每台各有五个容器 cenn、cenn-1、cenn-2、cenn-3、cenn-4（GPU 均为 H100-80GB×8，base image 为 athena:nanoflow，故意不挂宿主机 cuda12.8，容器内统一使用 /usr/local/cuda-13.0）。所有容器通过 -v /mnt/afs/:/home/niubility2 挂载共享文件系统，因此 /home/niubility2/cenzhiyao/ 下的代码和文件各容器路径一致可见；但 /root/.cache/ 是容器本地独立不共享的。每个容器内有 /root/.magi_env 文件配置了 CUDA_HOME、MAGI_ATTENTION_WORKSPACE_BASE、MAGI_COMPILE_CACHE_ROOT_DIR 三个环境变量，非交互执行命令时必须 source /root/.magi_env。

代码目录采用五副本制：athena、athena-1、athena-2、athena-3、athena-4 和 MagiAttention、MagiAttention-1、MagiAttention-2、MagiAttention-3、MagiAttention-4（均在 /home/niubility2/cenzhiyao/ 下）。MagiCompiler 是 athena 的 git submodule（位于 athena/pkgs/MagiCompiler），clone athena 后需 git submodule update --init。重点：代码后缀编号与容器编号严格一一对应（MagiAttention-1 → cenn-1，MagiAttention-2 → cenn-2，MagiAttention-3 → cenn-3，MagiAttention-4 → cenn-4，无后缀 → cenn），容器编号跟随代码编号而非机器编号，与机器无关。未指定机器时默认 h100-6。执行命令的统一模板：ssh Sensecore-bma-h100-6 "docker exec cenn-1 bash -c 'source /root/.magi_env && cd /home/niubility2/cenzhiyao/MagiAttention-1 && {command}'"。

环境变量治理（均已写入各容器 /root/.magi_env，source 一次全部生效）：CUDA_HOME=/usr/local/cuda-13.0（必须！不设置会导致 SM90 deterministic hang + wgmma 串行性能不准）；MAGI_ATTENTION_WORKSPACE_BASE 控制 MagiAttention JIT cache 位置（写入 {MagiAttention-x}/.cache/magi_attention/90/）；MAGI_COMPILE_CACHE_ROOT_DIR 控制 MagiCompiler cache 位置（写入 {athena-x}/.cache/magi_compiler/），其来源是 pydantic SettingsConfigDict env_prefix="MAGI_COMPILE_" 映射 cache_root_dir 字段；GIT_SSH_COMMAND 配置 git push/pull 的 SSH 私钥（注意与 ssh config 跳板用的 tmp-2.txt 是不同 key）。cache 均跟随代码目录而非 ~/.cache，已在 .gitignore 中，方便观察和隔离。

pip install：每个容器需安装对应副本的两个包——MAGI_ATTENTION_SKIP_CUDA_BUILD=1 pip install -e /home/niubility2/cenzhiyao/MagiAttention{-x} --no-build-isolation（跳过 C 扩展编译，运行时 JIT on demand）；pip install -e /home/niubility2/cenzhiyao/athena{-x}/pkgs/MagiCompiler。

CI 机器 h100-1：athena CI runner 在 athena_ci_new 容器，MagiCompiler CI 在 magi_compiler_ci 容器，还有一个裸机上的 MagiAttention runner（不归我管不要动）。h100-1 上的 cenn/cenn-1/cenn-2/cenn-3/cenn-4 是个人开发用，非必要不使用以免影响 CI 正常流程（除非是立刻跑编译好的小 kernel bench）。

文件传输：跨机器中转用 /home/niubility2/cenzhiyao/tmp/NNN-description/ 目录（共享 AFS）；rsync 到本地 Mac 桌面 /Users/cenn/Desktop/xxx-datetime/（注意 zsh 下远程通配符需转义或引号包裹）；参考命令：rsync -avP Sensecore-bma-h100-6:"/home/niubility2/cenzhiyao/tmp/043-xxx/" ~/Desktop/xxx-20260703/。快速 clone 用 git-cache 加速：/home/niubility2/cenzhiyao/ci/git-cache/ 下有 bare repo（world-sim-dev--athena.git、SandAI-org--MagiAttention.git、SandAI-org--MagiCompiler.git），clone 时加 --reference 参数。

代码提交：一旦完成计划文件中的某一大步骤，考虑提交推送一版阶段性成果。流程：pre-commit run --all-files 自动修复 lint（lint 一律用 pre-commit 自动修复，不要手动改）→ git add → git commit → git push（push 凭证已在 .magi_env 中配好，source 后直接 push 即可）。MagiAttention 远端 https://github.com/SandAI-org/MagiAttention，athena 远端 https://github.com/world-sim-dev/athena。

Docker 创建新容器模板：docker run -d -it --privileged --gpus all --network host --ipc host -v /mnt/afs/:/home/niubility2 -v /mnt/afs-gaoxiao:/home/production -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker --name cenn-{N} registry.cn-sh-01.sensecore.cn/sandai-ccr/athena:nanoflow bash（h100-1 上 local tag 为 athena:nanoflow 无需 registry 前缀）。
