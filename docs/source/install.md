# Installation

```{contents}
:local: true
```

## Step1: Activate an NGC pytorch docker container

* NGC pytorch docker release note: [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
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
