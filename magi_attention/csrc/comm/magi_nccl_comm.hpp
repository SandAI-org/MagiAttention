#pragma once

#ifndef USE_C10D_NCCL
#define USE_C10D_NCCL
#endif

// #include <torch/csrc/distributed/c10d/NCCLUtils.hpp>

#include <sched.h>
#include <cstdio>
#include <cstdlib>

#include <memory>
#include <mutex>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/util/Exception.h>
#include <nccl.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <optional>

constexpr int64_t kCommInitBusyWaitMillis = 2;

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 14)
#define NCCL_HAS_COMM_NONBLOCKING
#endif

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 18)
#define NCCL_HAS_COMM_SPLIT
#endif

// ncclGetLastError() is enabled only for NCCL versions 2.13+
// ncclRemoteError only exists in NCCL versions 2.13+
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 13)
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
#endif

static_assert(
    (NCCL_MAJOR == 2 && NCCL_MINOR >= 7) || (NCCL_MAJOR > 2),
    "NCCL version must be 2.7 or later");

// Error checking is enabled only for NCCL versions 2.4+ since ncclCommAbort()
// and ncclCommGetAsyncError() are not supported in earlier versions.
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 4)
#define ENABLE_NCCL_ERROR_CHECKING
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_ERROR_CHECKING
#endif

// P2P is enabled only for NCCL versions 2.7+ since ncclSend()
// and ncclRecv() are not supported in earlier versions.
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 7)
#define ENABLE_NCCL_P2P_SUPPORT
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_P2P_SUPPORT
#endif

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 11)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#endif

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 17)
#define NCCL_HAS_COMM_CTA_CGA
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define NCCL_HAS_COMM_CTA_CGA
#endif

#if defined(NCCL_REGISTRATION_SUPPORTED) ||                              \
    ((defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
      (NCCL_MINOR >= 19)))
#define NCCL_HAS_COMM_REGISTER
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define NCCL_HAS_COMM_REGISTER
#endif


inline std::string getNcclVersion() {
  static c10::once_flag ncclGetVersionFlag;
  static std::string versionString;

  c10::call_once(ncclGetVersionFlag, []() {
    int version = 0;
    ncclResult_t status = ncclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != ncclSuccess || version < 100) {
      versionString = "Unknown NCCL version";
    } else {
      // NCCL changed version coding starting 2.9
      const int majorBase = version < 2900 ? 1000 : 10000;
      const int minorBase = 100;
      auto ncclMajor = version / majorBase;
      auto ncclMinor = (version % majorBase) / minorBase;
      auto ncclPatch =
          version % (ncclMajor * majorBase + ncclMinor * minorBase);
      versionString = std::to_string(ncclMajor) + "." +
          std::to_string(ncclMinor) + "." + std::to_string(ncclPatch);
#ifdef NCCL_SUFFIX
      const auto ncclSuffix = std::string(NCCL_SUFFIX);
      if (!ncclSuffix.empty()) {
        versionString += "." + ncclSuffix;
      }
#endif
    }
  });

  return versionString;
}


inline std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
      getNcclVersion();
}


// Default value: 30 minutes
static int nccl_nonblocking_timeout() {
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


// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
inline std::string getNcclErrorDetailStr(
    ncclResult_t error,
    std::optional<std::string> processGroupFailureReason = std::nullopt
) {
  // Prioritize failure reason provided by PG NCCL first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (processGroupFailureReason != std::nullopt) {
    return *processGroupFailureReason;
  }
  std::string interpret;
  std::string err;
#ifdef ENABLE_NCCL_GET_LAST_ERROR
  auto ret = ncclGetLastError(nullptr);
  if (ret) {
    err = "\nLast error:\n" + std::string(ret);
  } else {
    err = "\nLast error: Unknown NCCL Error\n";
  }
#endif
  switch (error) {
    case ncclUnhandledCudaError:
      interpret = "ncclUnhandledCudaError: Call to CUDA function failed.";
      break;
    case ncclSystemError:
      interpret =
          "ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. ";
#ifndef NCCL_REMOTE_ERROR
      // Before ncclRemoteError was created, unexpected remote disconnect was
      // categorized as ncclSystemError
      interpret += "It can be also caused by unexpected exit of a remote peer.";
#endif
      break;
    case ncclInternalError:
      interpret = "ncclInternalError: Internal check failed.";
      break;
    case ncclInvalidArgument:
      interpret = "ncclInvalidArgument: Invalid value for an argument.";
      break;
    case ncclInvalidUsage:
      interpret =
          "ncclInvalidUsage: This usually reflects invalid usage of NCCL library.";
      break;
#ifdef NCCL_REMOTE_ERROR
    case ncclRemoteError:
      interpret =
          "ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.";
      break;
#endif
    default:
      interpret = "Unknown NCCL error!";
  }
  return interpret + err;
}


inline size_t hashTensors(const std::vector<at::Tensor>& tensors) {
  size_t hash = 0;
  for (auto& tensor : tensors) {
    if (tensor.numel() > 0 && tensor.storage()) {
      size_t data_size = tensor.storage().nbytes();
      if (data_size > 0 && tensor.storage().data_ptr()) {
        auto src = static_cast<const char*>(tensor.storage().data_ptr().get());
        std::vector<char> dst(data_size);
        // This is needed so that we trigger a device synchronization so we can
        // get the collective finished if launched on GPU and hash its output.
        cudaMemcpy(dst.data(), src, data_size, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < data_size; ++i) {
          // Update the hash for each byte in the tensor
          hash = c10::hash_combine(hash, c10::get_hash(dst[i], data_size));
        }
      }
    }
  }
  return hash;
}

// Macro to throw on a non-successful NCCL return value.
#define C10D_NCCL_CHECK(cmd, failureReason)                                   \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)


// Macro to throw on a non-successful NCCL return value for NONBLOCKING calls.
#define C10D_NCCL_CHECK_NONBLOCKING(cmd, failureReason)                       \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess && result != ncclInProgress) {                  \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)

// Error out if (current time - startTime) is greater than timeout (sec).
#define C10D_CHECK_TIMEOUT(startTime, timeout)                              \
  do {                                                                      \
    auto currentTime = std::chrono::steady_clock::now();                    \
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(    \
                           currentTime - startTime)                         \
                           .count();                                        \
    if (timeElapsed > timeout) {                                            \
      std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__);                                         \
      TORCH_CHECK_WITH(DistBackendError, false, err);                       \
    }                                                                       \
  } while (0)


// Macro to throw on a non-successful NCCL return value, non-blocking.
#define C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, yield_fn)      \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                   \
    auto timeout = nccl_nonblocking_timeout();                                \
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

// Sleep for kCommInitBusyWaitMillis milliseconds.
#define C10D_SCHED_SLEEP()     \
  std::this_thread::sleep_for( \
      std::chrono::milliseconds(kCommInitBusyWaitMillis))

// Macro to throw exception on a non-successful NCCL return value or timeout.
// This macro uses sched_yield() to yield the CPU.
// Thus suitable for NCCL calls that would quickly turn ncclSuccess, e.g.
// collectives.
#define C10D_NCCL_CHECK_TIMEOUT(cmd, comm, failureReason) \
  C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, sched_yield())

// Macro to throw exception on a non-successful NCCL return value or timeout.
// This macro uses sleep to yield the CPU.
// Thus suitable for NCCL calls that would take longer to turn ncclSuccess, e.g.
// ncclCommInitRankConfig, ncclCommFinalize, etc.
#define C10D_NCCL_CHECK_TIMEOUT_SLEEP(cmd, comm, failureReason) \
  C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, C10D_SCHED_SLEEP())

#define C10D_NCCL_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)           \
  do {                                                                       \
    ncclResult_t state = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                  \
    auto timeout = nccl_nonblocking_timeout();                               \
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

// Macro to print and abort on a non-successful NCCL return value.
#define C10D_NCCL_ASSERT(cmd)                            \
  do {                                                   \
    ncclResult_t result = cmd;                           \
    if (result != ncclSuccess) {                         \
      std::string err = ncclGetErrorWithVersion(result); \
      fprintf(                                           \
          stderr,                                        \
          "NCCL error in: %s:%d, %s\n",                  \
          __FILE__,                                      \
          __LINE__,                                      \
          err.c_str());                                  \
      abort();                                           \
    }                                                    \
  } while (0)

namespace c10d {

// RAII wrapper for NCCL communicator
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


// Helper that automatically cleans up premul sums.
struct ncclRedOpRAII {
  ncclRedOpRAII() = default;
  ncclRedOpRAII(ncclRedOp_t op) : op_(op) {}
  ncclRedOpRAII(ncclRedOp_t op, ncclComm_t comm)
      : op_(op), comm_(comm), premul_sum_(true) {}
  ncclRedOpRAII(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII& operator=(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII(ncclRedOpRAII&& tmp) noexcept : ncclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }
#if defined(ENABLE_NCCL_PREMUL_SUM_SUPPORT)
  ~ncclRedOpRAII() {
    if (premul_sum_) {
      ncclRedOpDestroy(op_, comm_);
    }
  }
#endif
  operator ncclRedOp_t() const {
    return op_;
  }
  ncclRedOp_t op_{};
  ncclComm_t comm_{};
  bool premul_sum_ = false;
};

} // namespace c10d
