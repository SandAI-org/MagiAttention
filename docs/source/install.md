# Installation

```{contents}
:local: true
```

## Step1: Activate an NGC pytorch docker container

* release note: [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-05.html#rel-25-05)
* docker image version: nvcr.io/nvidia/pytorch:25.05-py3
* docker run command:

    ```bash
    docker run --name {container_name} -v {host_mnt_root}:{container_mnt_root} -it -d --privileged --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:25.05-py3 /bin/bash
    ```

* docker exec command:

    ```bash
    docker exec -it {container_name} /bin/bash
    ```

## Step2: Install required packages

* command:

    ```bash
    pip install -r requirements.txt
    ```


## Step3: Install MagiAttention from source

* command:

    ```bash
    git clone https://github.com/SandAI-org/MagiAttention.git

    cd MagiAttention

    git submodule update --init --recursive

    pip install --no-build-isolation .
    ```
