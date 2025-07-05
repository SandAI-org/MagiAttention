#include "group_collective.cuh"
#include "group_reduce_post_process.cuh"


// Default value: 30 minutes
static int ncclNonblockingTimeout() {
  static int timeout = -2; // -2 means not initialized
  if (timeout == -2) {
    const auto val = c10::utils::get_env("TORCH_NCCL_NONBLOCKING_TIMEOUT");
    if (val.has_value() && !val.value().empty()) {
      timeout = stoi(val.value());
    } else {
      // Default value consistent with kBackendDefaultTimeout
      timeout = 30 * 60;
    }
  }
  return timeout;
}

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

ncclResult_t to_nccl_result(torch::cuda::nccl::ncclResult var) {
  switch (var) {
    case torch::cuda::nccl::ncclResult::Success:
      return ncclResult_t::ncclSuccess;
    case torch::cuda::nccl::ncclResult::UnhandledCudaError:
      return ncclResult_t::ncclUnhandledCudaError;
    case torch::cuda::nccl::ncclResult::SystemError:
      return ncclResult_t::ncclSystemError;
    case torch::cuda::nccl::ncclResult::InternalError:
      return ncclResult_t::ncclInternalError;
    case torch::cuda::nccl::ncclResult::InvalidArgument:
      return ncclResult_t::ncclInvalidArgument;
    case torch::cuda::nccl::ncclResult::InvalidUsage:
      return ncclResult_t::ncclInvalidUsage;
#ifdef NCCL_HAS_REMOTE_ERROR
    case torch::cuda::nccl::ncclResult::RemoteError:
      return ncclResult_t::ncclRemoteError;
#endif
#ifdef NCCL_HAS_COMM_NONBLOCKING
    case torch::cuda::nccl::ncclResult::InProgress:
      return ncclResult_t::ncclInProgress;
#endif
    case torch::cuda::nccl::ncclResult::NumResults:
      return ncclResult_t::ncclNumResults;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}

torch::cuda::nccl::ncclResult from_nccl_result(ncclResult_t var) {
  switch (var) {
    case ncclSuccess:
      return torch::cuda::nccl::ncclResult::Success;
    case ncclUnhandledCudaError:
      return torch::cuda::nccl::ncclResult::UnhandledCudaError;
    case ncclSystemError:
      return torch::cuda::nccl::ncclResult::SystemError;
    case ncclInternalError:
      return torch::cuda::nccl::ncclResult::InternalError;
    case ncclInvalidArgument:
      return torch::cuda::nccl::ncclResult::InvalidArgument;
    case ncclInvalidUsage:
      return torch::cuda::nccl::ncclResult::InvalidUsage;
#ifdef NCCL_HAS_REMOTE_ERROR
    case ncclRemoteError:
      return torch::cuda::nccl::ncclResult::RemoteError;
#endif
#ifdef NCCL_HAS_COMM_NONBLOCKING
    case ncclInProgress:
      return torch::cuda::nccl::ncclResult::InProgress;
#endif
    case ncclNumResults:
      return torch::cuda::nccl::ncclResult::NumResults;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}

void throw_nccl_error(torch::cuda::nccl::ncclResult status) {
  std::ostringstream err;
  err << "NCCL Error " << static_cast<int>(status) << ": "
      << ncclGetErrorString(to_nccl_result(status));
  throw std::runtime_error(err.str());
}

inline void NCCL_CHECK(torch::cuda::nccl::ncclResult status) {
  if (status != torch::cuda::nccl::ncclResult::Success) {
    throw_nccl_error(status);
  }
}

static void NCCL_CHECK(ncclResult_t result) {
  NCCL_CHECK(from_nccl_result(result));
}

static void NCCL_CHECK_TIMEOUT(torch::cuda::nccl::ncclResult status, torch::cuda::nccl::ncclComm_t comm) {
#ifdef NCCL_HAS_COMM_NONBLOCKING
  ncclResult_t result = to_nccl_result(status);
  auto startTimepoint = std::chrono::steady_clock::now();
  while (result == ncclInProgress) {
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           currentTimepoint - startTimepoint)
                           .count();
    if (timeElapsed > ncclNonblockingTimeout()) {
      throw std::runtime_error(
          "NCCL timeout when waiting for nonblocking call to become successful.");
    }
    sched_yield(); // yield to other threads
    ncclCommGetAsyncError(to_nccl_comm(comm), &result);
  }
  if (result != ncclSuccess) {
    throw_nccl_error(from_nccl_result(result));
  }
#else
  TORCH_INTERNAL_ASSERT(
      false, "NCCL COMM NONBLOCKING USED WITH UNSUPPORTED NCCL VERSION.");
#endif
}

static void NCCL_CHECK_TIMEOUT(ncclResult_t result, torch::cuda::nccl::ncclComm_t comm) {
  NCCL_CHECK_TIMEOUT(from_nccl_result(result), comm);
}

static void NCCL_CHECK_TIMEOUT(torch::cuda::nccl::ncclResult status, std::vector<torch::cuda::nccl::ncclComm_t>& comms) {
#ifdef NCCL_HAS_COMM_NONBLOCKING
  ncclResult_t result = to_nccl_result(status);
  auto startTimepoint = std::chrono::steady_clock::now();
  if (result == ncclInProgress) {
    for (const auto i : c10::irange(comms.size())) {
      do {
        auto currentTimepoint = std::chrono::steady_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               currentTimepoint - startTimepoint)
                               .count();
        if (timeElapsed > ncclNonblockingTimeout()) {
          throw std::runtime_error(
              "NCCL timeout when waiting for nonblocking call to become successful.");
        }
        sched_yield(); // yield to other threads
        ncclCommGetAsyncError(to_nccl_comm(comms[i]), &result);
      } while (result == ncclInProgress);
      if (result != ncclSuccess) {
        break; /* fall through to failed case */
      }
    }
  }
  if (result != ncclSuccess) {
    throw_nccl_error(from_nccl_result(result));
  }
#else
  TORCH_INTERNAL_ASSERT(
      false, "NCCL COMM NONBLOCKING USED WITH UNSUPPORTED NCCL VERSION.");
#endif
}

static void NCCL_CHECK_TIMEOUT(
    ncclResult_t result,
    std::vector<torch::cuda::nccl::ncclComm_t>& comms) {
  NCCL_CHECK_TIMEOUT(from_nccl_result(result), comms);
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

        NCCL_CHECK(ncclGroupStart());
        size_t input_offset = 0, output_offset = 0;
        for (size_t input_split_idx = 0; input_split_idx < num_input_splits; ++input_split_idx) {
            auto input_size = input_split_size_list[input_split_idx] * stride0;
            for (const auto dst_rank : dst_indices_list[input_split_idx]) {
                NCCL_CHECK(ncclSend(
                  (const void*) (((char*)send_buffer) + input_offset * element_size),
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
            size_t output_size = output_split_size_list[output_split_idx] * stride0;
            NCCL_CHECK(ncclRecv(
              (void *) (((char*)recv_buffer) + output_offset * element_size),
              output_size,
              nccl_data_type,
              src_rank,
              nccl_comm,
              stream
            ));
            output_offset += output_size;
        }
        #ifndef NCCL_HAS_COMM_NONBLOCKING
          NCCL_CHECK(ncclGroupEnd());
        #else
          NCCL_CHECK_TIMEOUT(ncclGroupEnd(), comm);
        #endif
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
        size_t input_offset = 0, output_offset = 0;
        NCCL_CHECK(ncclGroupStart());
        for (size_t input_split_idx = 0; input_split_idx < num_input_splits; ++input_split_idx) {
            auto dst_rank = dst_index_list[input_split_idx];
            auto input_size = input_split_size_list[input_split_idx] * stride0;
            NCCL_CHECK(ncclSend(
              (const void*) (((char*)send_buffer) + input_offset * element_size),
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
                NCCL_CHECK(ncclRecv(
                  (void *) (((char*)repeated_recv_buffer) + output_offset * element_size),
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
        #ifndef NCCL_HAS_COMM_NONBLOCKING
          NCCL_CHECK(ncclGroupEnd());
        #else
          NCCL_CHECK_TIMEOUT(ncclGroupEnd(), comm);
        #endif

        // run post-process reduce kernel
        run_group_reduce_post_process(args);
    }

} // namespace torch::cuda::nccl
