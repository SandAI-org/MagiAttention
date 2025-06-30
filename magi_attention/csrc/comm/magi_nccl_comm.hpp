
#pragma once

#ifndef USE_C10D_NCCL
#define USE_C10D_NCCL
#endif

#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>


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


// Macro to throw on a non-successful NCCL return value, non-blocking.
#ifdef C10D_NCCL_CHECK_TIMEOUT_BASE
#undef C10D_NCCL_CHECK_TIMEOUT_BASE
#endif
#define C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, yield_fn)      \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                   \
    auto timeout = ncclNonblockingTimeout();                                \
    while (result == ncclInProgress) {                                        \
      C10D_CHECK_TIMEOUT(startTimepoint, timeout);                            \
      yield_fn;                                                               \
      ncclCommGetAsyncError(comm, &result);                                   \
    }                                                                         \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)


#ifdef C10D_NCCL_CHECK_TIMEOUT_GROUPEND
#undef C10D_NCCL_CHECK_TIMEOUT_GROUPEND
#endif
#define C10D_NCCL_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)           \
  do {                                                                       \
    ncclResult_t state = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                  \
    auto timeout = ncclNonblockingTimeout();                               \
    if (state == ncclInProgress) {                                           \
      do {                                                                   \
        C10D_CHECK_TIMEOUT(startTimepoint, timeout);                         \
        sched_yield();                                                       \
        ncclCommGetAsyncError(comm->getNcclComm(), &state);                  \
      } while (state == ncclInProgress);                                     \
    }                                                                        \
    if (state != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +    \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(state) + \
          "\n" + getNcclErrorDetailStr(state, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                        \
    }                                                                        \
  } while (0)


namespace c10d {

class TORCH_API MagiNCCLComm {
    using MutexType = std::recursive_mutex;
    using LockType = std::unique_lock<MutexType>;

public:
    explicit MagiNCCLComm(ncclComm_t ncclComm);

    MagiNCCLComm() = default;

    ~MagiNCCLComm() noexcept;

    static std::shared_ptr<MagiNCCLComm> create(
        int numRanks,
        int rank,
        ncclUniqueId commId,
        at::DeviceIndex deviceIndex);

    #ifdef NCCL_HAS_COMM_NONBLOCKING
    static std::shared_ptr<MagiNCCLComm> create(
        int numRanks,
        int rank,
        ncclUniqueId commId,
        at::DeviceIndex deviceIndex,
        ncclConfig_t& config);

    static std::shared_ptr<MagiNCCLComm> split(
        MagiNCCLComm* source,
        int color_id,
        int rank,
        ncclConfig_t& config,
        std::vector<uint64_t>& ranks_ull);
    #endif

    #if (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
    std::unordered_map<std::string, std::string> ncclCommDump();
    #endif

    ncclUniqueId getNcclId();

    // Must not be copyable
    MagiNCCLComm(const MagiNCCLComm&) = delete;
    MagiNCCLComm& operator=(const MagiNCCLComm&) = delete;

    // Do not support move assignment as there is no valid use case
    MagiNCCLComm& operator=(MagiNCCLComm&& other) = delete;

    // Move constructable
    // NOLINTNEXTLINE(*-noexcept-move-*)
    MagiNCCLComm(MagiNCCLComm&& other);

    ncclComm_t getNcclComm();

    // Wait for the communicator to be ready. This is a blocking function.
    // Useful in nonblocking mode: NCCL requires the communicator to be ready
    // before issuing a second command.
    // Arguments:
    //   longInterval: if true, wait with sleep of an interval; otherwise, wait
    //   with `sched_yield` which is faster (but acquires CPU more frequently).
    //   Use `longInterval=true` when waiting for initialization or finalize to
    //   complete. Use `longInterval=false` when waiting collective call to return
    //   ncclSuccess.
    void waitReady(bool longInterval);

    std::optional<std::string> getNcclCommFailureReason() const;

    void abort(std::optional<std::string> commFailureReason = std::nullopt);

    // Finalize a communicator -- asking it to flush its operations. When the
    // communicator is marked as nonblocking, this is a nonblocking function;
    // otherwise, it will block till all operations complete.
    void finalize();

    // Destroy a communicator. This is a blocking function.
    void destroy();

    bool isInitialized() const;

    bool isAborted() const;

    uint64_t getCommSplitCounter() const;

    ncclResult_t checkForNcclError();

    ncclResult_t registerSegment(void* ptr, size_t size);

    ncclResult_t deregisterSegment(void* ptr);

    std::string repr() const;

    friend class ProcessGroupNCCL;

    protected:
    // Unique nccl_id for this communicator.
    ncclUniqueId ncclId_{};
    bool aborted_{false};
    uint64_t ncclCommSplitCounter_{0};
    ncclResult_t ncclAsyncErr_{ncclSuccess};
    mutable MutexType mutex_;
    // Rank that this communicator corresponds to.
    int rank_{};
    // Optional reason for communicator failure, provided by ProcessGroupNCCL for
    // better error messaging.
    std::optional<std::string> commFailureReason_{};
    bool initialized_{false};
    // Whether this communicator is using nonblocking mode. Recorded during comm
    // creation or split. For safety, we give a default value of true (more
    // protection).
    bool nonBlocking_{true};
    // Device index for which the NCCL comm is created
    at::DeviceIndex deviceIndex_{-1};
    #ifdef NCCL_HAS_COMM_REGISTER
    // Stores handlers for tensors registered by NCCL
    std::unordered_map<void*, void*> registeredSegmentHandles_;
    #endif

private:
    ncclComm_t ncclComm_{nullptr};
};

} // namespace c10d