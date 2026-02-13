# Installation

:::{warning}
MagiAttention currently supports only <u>Hopper</u> and <u>Blackwell</u>. 
We are actively working to support more GPU architectures in upcoming releases.
:::

```{contents}
:local: true
```

## Setup Environment

### Activate an NGC-PyTorch Container

:::{tip}
We recomment you to use the standard [NGC-PyTorch Docker Releases]((https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)) for consistency of basic dependencies such as `Python`, `CUDA`, `PyTorch`, etc.
:::

* docker run command:

    ```bash
    # choose one compatible version
    MAJOR_VERSION=25
    MINOR_VERSION=10 # choose from {05, 06, 08, 09, 10}

    # specify your own names and paths
    CONTAINER_NAME=...
    HOST_MNT_ROOT=...
    CONTAINER_MNT_ROOT=...

    docker run --name ${CONTAINER_NAME} -v ${HOST_MNT_ROOT}:${CONTAINER_MNT_ROOT} -it -d --privileged --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${MAJOR_VERSION}.${MINOR_VERSION}-py3 /bin/bash
    ```

* docker exec command:

    ```bash
    docker exec -it ${CONTAINER_NAME} /bin/bash
    ```

### Pull Source Code

* git commands:

    ```bash
    git clone https://github.com/SandAI-org/MagiAttention.git

    cd MagiAttention

    git submodule update --init --recursive
    ```

### Enable IBGDA (optional)

:::{note}
If you would like to try using our [native group-collective kernels](https://SandAI-org.github.io/MagiAttention/docs/main/blog/native_grpcoll.html) when `cp_size > 8` as the communication backend, i.e. a process group involving both intranode (*connected through `NVLink`*) and internode (*visible through `RDMA`*) peers, you're required to enable `IBGDA` on your bare-metal host machine. 
:::

:::{warning}
This step needs to be performed on the **BARE-METAL HOST OPERATING SYSTEM**, **NOT** inside a Docker or other containerized environment, as containers do not manage the host kernel.
:::

* bash script:

    ```bash
    bash scripts/enable_ibgda_on_host.sh
    ```


## Setup Dependencies

### Install Required Packages

* pip install command:

    ```bash
    pip install -r requirements.txt
    ```

### Install flash_attn_cute (optional)

:::{note}
If you would like to try MagiAttention on Blackwell, for now you're required to install `flash_attn_cute` package to enable [FFA_FA backend](https://SandAI-org.github.io/MagiAttention/docs/main/blog/blackwell_ffa_fa4.html) as a temporary workaround.
:::

* bash script:

    ```bash
    bash scripts/install_flash_attn_cute.sh
    ```


## Install MagiAttention

### Install MagiAttention From Source

:::{warning}
This progress may take around 20~30 minutes and occupies `90%` of CPU resources for the first time.
:::

:::{note}
We have several [environment variables](https://SandAI-org.github.io/MagiAttention/docs/main/user_guide/env_variables.html#for-build) to fine-grained control the installation progress, especially for CUDA extension modules building.
:::

* pip install command for Hopper:

    ```bash
    pip install --no-build-isolation .
    ```

* pip install command for Blackwell:

    ```bash
    export MAGI_ATTENTION_PREBUILD_FFA=0
    pip install --no-build-isolation .
    
    export MAGI_ATTENTION_FA4_BACKEND=1 # always set it when using MagiAttention on Blackwell
    ```