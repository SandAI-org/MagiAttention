#ifndef USE_NCCL
#define USE_NCCL
#endif

#include "group_collective.cuh"
#include "group_reduce_post_process.cuh"


ncclDataType_t to_nccl_data_type(c10::ScalarType type) {
    switch (type) {
        case at::kFloat:
            return ncclDataType_t::ncclFloat;
        case at::kHalf:
            return ncclDataType_t::ncclHalf;
        case at::kDouble:
            return ncclDataType_t::ncclDouble;
        case at::kLong:
            return ncclDataType_t::ncclInt64;
        case at::kInt:
            return ncclDataType_t::ncclInt;
        case at::kChar:
            return ncclDataType_t::ncclChar;
        // NOLINTNEXTLINE(*-narrowing-conversions, bugprone-branch-clone)
        case at::kByte:
            return ncclDataType_t::ncclUint8;
        case at::kBool:
            return ncclDataType_t::ncclUint8;
    #if defined(USE_ROCM)
        case at::kFloat8_e4m3fnuz:
            return ncclDataType_t::ncclUint8;
        case at::kFloat8_e5m2fnuz:
            return ncclDataType_t::ncclUint8;
    #else
        case at::kFloat8_e4m3fn:
            return ncclDataType_t::ncclUint8;
        case at::kFloat8_e5m2:
            return ncclDataType_t::ncclUint8;
    #endif
    #if HAS_NCCL_BF16_DATATYPE
        case at::kBFloat16:
            return ncclDataType_t::ncclBfloat16;
    #endif
        default:
            TORCH_CHECK(false, "Unconvertible NCCL type ", type);
    }
}

ncclComm_t to_nccl_comm(torch::cuda::nccl::ncclComm_t var) {
    return reinterpret_cast<ncclComm_t>(var);
}


namespace torch::cuda::nccl {
    
    GroupReduceMetaInfo compute_group_reduce_meta_info(
        const c10::IntArrayRef recv_buffer_shape,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<std::vector<int64_t>>& src_indices_list,
        const int repeat_dim
    ) {
        size_t seqlen = recv_buffer_shape[repeat_dim];
        size_t num_splits = output_split_size_list.size();
        std::vector<int64_t> num_repeats_list; num_repeats_list.reserve(num_splits);
        std::vector<int64_t> cu_split_size_list; cu_split_size_list.reserve(num_splits + 1); cu_split_size_list.push_back(0);
        std::vector<int64_t> repeated_cu_split_size_list; repeated_cu_split_size_list.reserve(num_splits + 1); repeated_cu_split_size_list.push_back(0);
        std::vector<int64_t> repeated_recv_buffer_shape(recv_buffer_shape.begin(), recv_buffer_shape.end());

        int64_t repeated_seqlen = 0; int64_t max_split_size = 0;
        for (int64_t output_split_idx = 0; output_split_idx < num_splits; ++output_split_idx) {
            auto num_repeats = src_indices_list[output_split_idx].size();
            auto split_size = output_split_size_list[output_split_idx];
            auto repeat_split_size = split_size * num_repeats;
            max_split_size = std::max(max_split_size, split_size);

            num_repeats_list.push_back(num_repeats);
            cu_split_size_list.push_back(cu_split_size_list[output_split_idx] + split_size);
            repeated_cu_split_size_list.push_back(repeated_cu_split_size_list[output_split_idx] + repeat_split_size);
            repeated_seqlen += repeat_split_size;
        }
        repeated_recv_buffer_shape[repeat_dim] = repeated_seqlen;

        /** NOTE: do not wrap it to c10::ArrayRef:
         *  return c10::makeArrayRef(repeated_recv_buffer_shape);
         * since c10::ArrayRef only holds the reference which is local to this function
         * thus might resulting in dangling reference
         */
        return GroupReduceMetaInfo(
            seqlen,
            num_splits,
            repeated_seqlen,
            max_split_size,
            std::move(num_repeats_list),
            std::move(cu_split_size_list),
            std::move(repeated_cu_split_size_list),
            std::move(repeated_recv_buffer_shape)
        );
    }

    // group cast collective
    void group_cast_nccl_kernel(
        void* send_buffer,
        void* recv_buffer,
        const std::vector<int64_t>& input_split_size_list,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<std::vector<int64_t>>& dst_indices_list,
        const std::vector<int64_t>& src_index_list,
        size_t stride0,
        size_t element_size,
        c10::ScalarType type,
        ncclComm_t comm,
        at::cuda::CUDAStream& stream
    ) {
        auto num_input_splits = input_split_size_list.size();
        auto num_output_splits = output_split_size_list.size();

        auto nccl_data_type = to_nccl_data_type(type);
        auto nccl_comm = to_nccl_comm(comm);

        int64_t input_offset = 0, output_offset = 0;
        CHECK_NCCL(ncclGroupStart());
        for (size_t input_split_idx = 0; input_split_idx < num_input_splits; ++input_split_idx) {
            auto input_size = input_split_size_list[input_split_idx] * stride0;
            for (auto dst_rank : dst_indices_list[input_split_idx]) {
                CHECK_NCCL(ncclSend(
                    (const void*) (send_buffer + input_offset * element_size),
                    input_size,
                    nccl_data_type,
                    dst_rank,
                    nccl_comm,
                    stream
                ));
            }
            input_offset += input_size;
        }
        for (size_t output_split_idx = 0; output_split_idx < num_output_splits; ++output_split_idx) {
            auto src_rank = src_index_list[output_split_idx];
            auto output_size = output_split_size_list[output_split_idx] * stride0;
            CHECK_NCCL(ncclRecv(
                (void *) (recv_buffer + output_offset * element_size),
                output_size,
                nccl_data_type,
                src_rank,
                nccl_comm,
                stream
            ));
            output_offset += output_size;
        }
        CHECK_NCCL(ncclGroupEnd());
    }

    // group reduce collective
    void group_reduce_nccl_kernel(
        void* send_buffer,
        void* recv_buffer,
        void* repeated_recv_buffer,
        const std::vector<int64_t>& input_split_size_list,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<int64_t>& dst_index_list,
        const std::vector<std::vector<int64_t>>& src_indices_list,
        size_t stride0,
        size_t element_size,
        c10::ScalarType type,
        ncclComm_t comm,
        at::cuda::CUDAStream& stream,
        GroupReducePostProcessArgs& args
    ) {
        auto num_input_splits = input_split_size_list.size();
        auto num_output_splits = output_split_size_list.size();

        auto nccl_data_type = to_nccl_data_type(type);
        auto nccl_comm = to_nccl_comm(comm);

        // run group-reduce kernel implemented by nccl group p2p
        int64_t input_offset = 0, output_offset = 0;
        CHECK_NCCL(ncclGroupStart());
        for (size_t input_split_idx = 0; input_split_idx < num_input_splits; ++input_split_idx) {
            auto dst_rank = dst_index_list[input_split_idx];
            auto input_size = input_split_size_list[input_split_idx] * stride0;
            CHECK_NCCL(ncclSend(
                (const void*) (send_buffer + input_offset * element_size),
                input_size,
                nccl_data_type,
                dst_rank,
                nccl_comm,
                stream
            ));
            input_offset += input_size;
        }
        for (size_t output_split_idx = 0; output_split_idx < num_output_splits; ++output_split_idx) {
            auto output_size = output_split_size_list[output_split_idx] * stride0;
            for (auto src_rank : src_indices_list[output_split_idx]) {
                CHECK_NCCL(ncclRecv(
                    (void *) (repeated_recv_buffer + output_offset * element_size),
                    output_size,
                    nccl_data_type,
                    src_rank,
                    nccl_comm,
                    stream
                ));
                /** NOTE: since nccl recv can not handle atomic add,
                 * we have to interleavedly repeat the recv buffer
                 * for each src rank for the same split
                 * and apply a post-process reduce to sum them up into the original recv buffer
                 * and for convenience and safety, we allocate this repeated_recv_buffer outside
                 * thus we can utilize the torch's caching allocator to avoid reallocation cuda memory
                 * as well as ensure meory false-reuse using workNccl's stashed_for_allocator_safety_
                 */
                output_offset += output_size;
            }
        }
        CHECK_NCCL(ncclGroupEnd());

        // run post-process reduce kernel
        run_group_reduce_post_process(args);
    }

} // namespace torch::cuda::nccl
