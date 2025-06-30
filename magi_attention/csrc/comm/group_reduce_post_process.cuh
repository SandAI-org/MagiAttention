
#include "group_collective.cuh"
#include "repeat_reduce_launch_template.h"


#define GROUP_REDUCE_POST_PROCESS_NUM_SMS 32 /* use the maximum number of SMs of nccl comm kernel */
#define GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE 1024 /* use the maximum block size for any SM */

/** TODO: finish the cute kernel to handle the oob and become persistent */
// #define GROUP_REDUCE_POST_PROCESS_WITH_CUTE

namespace torch::cuda::nccl {

    __device__ size_t binary_search_split(
        const int64_t* d_cu_split_size_list,
        size_t start,
        size_t end,
        size_t idx,
        size_t stride
    ) {
        size_t low = start, high = end;
        while (low < high) { // [low, high)
            size_t mid = low + (high - low) / 2;
            if (idx < d_cu_split_size_list[mid] * stride) {
                high = mid; // [low, mid)
            } else {
                low = mid + 1; // [mid + 1, high)
            }
        }
        return low - 1; // low == high
    }

    /** pure-cuda naive repeat-interleaved reduce
     * NOTE: this version uses uses each block to process a single row
     * but the highest HBM throughput (no limit to grid size) only reachs ~30%
    */
    template <typename scalar_t>
    __global__ void repeat_reduce_cuda_kernel(
        scalar_t* recv_buffer,
        const scalar_t* repeated_recv_buffer,
        const int64_t* d_split_size_list,
        const int64_t* d_num_repeats_list,
        const int64_t* d_cu_split_size_list,
        const int64_t* d_repeated_cu_split_size_list,
        size_t seqlen,
        size_t num_splits,
        size_t stride0
    ) {
        extern __shared__ size_t shared_split_info[];

        size_t bid = blockIdx.x, tid_in_block = threadIdx.x;
        size_t num_blocks_per_grid = gridDim.x, num_threads_per_block = blockDim.x;

        size_t split_idx = 0;
        for (auto row_idx = bid; row_idx < seqlen; row_idx += num_blocks_per_grid) {
            if (!tid_in_block) {
                // only the thread 0 in this block searchs for split idx that the current idx belongs
                split_idx = binary_search_split(
                    d_cu_split_size_list,
                    split_idx,
                    num_splits,
                    row_idx,
                    1
                );
                // thread 0 gets the info about this split and writes it to shared memory
                shared_split_info[0] = row_idx * stride0; // row_start
                shared_split_info[1] = d_cu_split_size_list[split_idx] * stride0; // recv_split_start
                shared_split_info[2] = d_repeated_cu_split_size_list[split_idx] * stride0; // repeated_recv_split_start
                shared_split_info[3] = d_split_size_list[split_idx] * stride0; // recv_split_size
                shared_split_info[4] = d_num_repeats_list[split_idx]; // num_repeats
            } __syncthreads(); // all threads in this block wait for the same split info to be ready

            // get the info about this split
            auto row_start = shared_split_info[0];
            auto recv_split_start = shared_split_info[1];
            auto repeated_recv_split_start = shared_split_info[2];
            auto recv_split_size = shared_split_info[3];
            auto num_repeats = shared_split_info[4];
            auto recv_split_offset_to_idx = row_start - recv_split_start;

            // get the row start ptr of recv_buffer
            scalar_t* recv_data_ptr = (recv_buffer + row_start);

            // get the row start ptr of first partial split of repeated_recv_buffer
            const scalar_t* repeated_recv_data_ptr = (repeated_recv_buffer + repeated_recv_split_start + recv_split_offset_to_idx);

            // for-loop this row
            for (auto col_idx = tid_in_block; col_idx < stride0; col_idx += num_threads_per_block) {
                // get the ptr of current col
                auto recv_data_ptr_this_col = (recv_data_ptr + col_idx);

                // load the original data of current col to be reduced to
                scalar_t recv_reduce_data = *recv_data_ptr_this_col;

                // get the corr. ptr of first partial data
                auto repeated_recv_data_ptr_this_col = (repeated_recv_data_ptr + col_idx);

                // load and reduce each corr. partial data
                #pragma unroll (8)
                for (size_t r = 0; r < num_repeats; ++r) {
                    recv_reduce_data += __ldg(repeated_recv_data_ptr_this_col + r * recv_split_size);
                }
             
                // write the reduced data back to recv_buffer
                *recv_data_ptr_this_col = recv_reduce_data;
            }
        }
    }

    /** post-process reduce kernel from repeated_recv_buffer to recv_buffer
     * NOTE: 
     * 1. we don't want the post-process kernel occupies too many SMs
     *      thus we set a small fixed grid size for pure-cuda kernel,
     *      or adopt the persistent kernel with the sm limit if using cute
     * 2. the post-process kernel is supposed to run on the same stream 
     *      as the group reduce kernel, i.e. nccl stream
     */
    void run_group_reduce_post_process(GroupReducePostProcessArgs& args) {
        #ifdef GROUP_REDUCE_POST_PROCESS_WITH_CUTE
        repeat_reduce_cute_kernel<cutlass::bfloat16_t, 128>(args);
        #else
        int blockSize = GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE; int gridSize = args.seqlen;
        gridSize = std::min(gridSize, GROUP_REDUCE_POST_PROCESS_NUM_SMS);

        dim3 gridDims(gridSize);
        dim3 blockDims(blockSize);
        size_t sharedMemSize = 5 * sizeof(size_t); // share meta info about split range of one row

        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, /* add float16/bfloat16 to dispatch types */
            args.type,
            "repeat_reduce_cuda_kernel",
            [&] {
            repeat_reduce_cuda_kernel<scalar_t> /* auto-deduced `scalar_t` by the macro */
                <<<gridDims, blockDims, sharedMemSize, args.stream>>>(
                    static_cast<scalar_t*>(args.recv_buffer),
                    static_cast<const scalar_t*>(args.repeated_recv_buffer),

                    args.d_split_size_list,
                    args.d_num_repeats_list,
                    args.d_cu_split_size_list,
                    args.d_repeated_cu_split_size_list,

                    args.seqlen,
                    args.num_splits,
                    args.stride0
                );
            }
        );
        #endif
    }
}