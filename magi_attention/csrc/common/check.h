#pragma once

#include <assert.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                        \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)


#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


#define CHECK_NCCL(cmd) do {                             \
    ncclResult_t res = cmd;                             \
    if (res != ncclSuccess) {                           \
        printf(                                         \
            "Failed, NCCL Error: %s:%d '%s'\n",         \
            __FILE__, __LINE__, ncclGetErrorString(res) \
        );                                              \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while (0)
