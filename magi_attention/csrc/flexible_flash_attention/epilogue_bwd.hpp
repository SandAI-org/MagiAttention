/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/barrier.h>
#include <cutlass/cutlass.h>

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <
    class TileShape_MNK_,
    class Element_,
    class ElementAccum_,
    class ArchTag_,
    int NumEpilogueThreads_,
    bool dKV_swapAB_,
    int AtomLayoutKdKV = 1,
    bool DisableBwdDkvAtomicReduction_ = false,
    bool Deterministic_=false>
struct CollectiveEpilogueBwd {
  using TileShape_MNK = TileShape_MNK_;
  using Element = Element_;
  using ElementAccum = ElementAccum_;

  static constexpr bool IsSameType = cute::is_same_v<Element, ElementAccum>;
  static constexpr bool DisableBwdDkvAtomicReduction = DisableBwdDkvAtomicReduction_;

  using ArchTag = ArchTag_;
  static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool Use_TMA = ArchTag::kMinComputeCapability >= 90;
  static constexpr bool Deterministic = Deterministic_;

  static_assert(ArchTag::kMinComputeCapability >= 80);

  // Select the appropriate TMA copy type based on DisableBwdDkvAtomicReduction
  using GmemTiledCopydKVTMA = std::conditional_t<DisableBwdDkvAtomicReduction, cute::SM90_TMA_STORE, cute::SM90_TMA_REDUCE_ADD>;

  // These are for storing the output tensor without TMA (e.g., for setting output to zero)
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});
  static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
  static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<Shape<Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopydKV = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{})); // Val layout, 8 or 16 vals per store

  using SmemLayoutAtomdKVTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                        GMMA::Major::K,
                                        Element,
                                        // TODO: do we have to change this if dKV_swapAB is true?
                                        decltype(cute::get<1>(TileShape_MNK{})),
                                        Int<CUTE_STATIC_V(cute::get<2>(TileShape_MNK{})) / AtomLayoutKdKV>>());
  using SmemLayoutdKVTMA = decltype(tile_to_shape(SmemLayoutAtomdKVTMA{}, select<1, 2>(TileShape_MNK{})));
  using SmemLayoutdKVtTMA = decltype(cute::composition(
      SmemLayoutdKVTMA{},
      make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})), make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

  using SmemLayoutAtomdKV = SmemLayoutAtomdKVTMA;
  using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, select<1, 2>(TileShape_MNK{})));
  using SmemLayoutdKVt = decltype(cute::composition(
      SmemLayoutdKV{},
      make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})), make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

  using SmemCopyAtomdKV = Copy_Atom<cute::DefaultCopy, Element>;

  static constexpr size_t SmemAlignmentdKV = ArchTag::kMinComputeCapability >= 90 ? cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{}) : 128;
  static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

  struct TensorStorage : cute::aligned_struct<SmemAlignmentdKV> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dk;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dv;
  };

  using ShapedKV = cute::Shape<int32_t, int32_t, int32_t>; // (seqlen_k, d, head)
  using StridedKV = cute::Stride<int64_t, _1, int64_t>;

  using TMA_dKV = std::conditional_t<
      Use_TMA,
      decltype(make_tma_copy(
          GmemTiledCopydKVTMA{},
          make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
          SmemLayoutdKVTMA{},
          select<1, 2>(TileShape_MNK{}),
          _1{})), // no mcast for dKV
      std::nullptr_t>;
    
  using BlockCoordType = std::conditional_t<
    Deterministic,
    cute::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>,
    cute::tuple<int32_t, int32_t, int32_t>
  >;

  // Host side kernel arguments
  struct Arguments {
    Element* ptr_dK;
    ShapedKV const shape_dK;
    StridedKV const stride_dK;
    Element* ptr_dV;
    StridedKV const stride_dV;
    int const num_heads_q;
    int const num_heads_kv;
    int const* q_ranges;
    int const* k_ranges;
    int* determin_range_locks = nullptr;
  };

  // Device side kernel params
  struct Params {
    Element* ptr_dK;
    ShapedKV const shape_dK;
    StridedKV const stride_dK;
    Element* ptr_dV;
    StridedKV const stride_dV;
    TMA_dKV tma_store_dK, tma_store_dV;
    int const* q_ranges;
    int const* k_ranges;
    cutlass::FastDivmod qhead_per_khead_divmod;
    int const nheads;
    int* determin_range_locks = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
    Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dK, args.stride_dV);
    TMA_dKV tma_store_dK = [&] {
      if constexpr (Use_TMA) {
        return make_tma_copy(GmemTiledCopydKVTMA{}, mdK, SmemLayoutdKVTMA{}, select<1, 2>(TileShape_MNK{}), _1{}); // no mcast for dKV
      } else {
        return nullptr;
      }
    }();
    TMA_dKV tma_store_dV = [&] {
      if constexpr (Use_TMA) {
        return make_tma_copy(GmemTiledCopydKVTMA{}, mdV, SmemLayoutdKVTMA{}, select<1, 2>(TileShape_MNK{}), _1{}); // no mcast for dKV
      } else {
        return nullptr;
      }
    }();
    return {
        args.ptr_dK,
        args.shape_dK,
        args.stride_dK,
        args.ptr_dV,
        args.stride_dV,
        tma_store_dK,
        tma_store_dV,
        args.q_ranges,
        args.k_ranges,
        cutlass::FastDivmod(cute::ceil_div(args.num_heads_q, get<2>(args.shape_dK))),
        args.num_heads_kv,
        args.determin_range_locks};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (Use_TMA) {
      cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
    }
  }

    CUTLASS_DEVICE
    void deterministic_sync(int* range_lock, int bidh, int offset, int q_block_size, int num_heads, int left_range_sync_num, int right_range_sync_num) {
        if (left_range_sync_num == 0 && right_range_sync_num == 0)
            return ;

        // Calculate lock index
        int left_range_block_idx = offset / q_block_size;
        int left_range_index = left_range_block_idx * num_heads + bidh;
        int right_range_block_idx = (offset + q_block_size - 1) / q_block_size;

        // Acquire the first lock
        #pragma unroll 1
        while (atomicCAS(&range_lock[left_range_index * 2], left_range_sync_num, left_range_sync_num) != left_range_sync_num) {
        }

        // If we need a second lock
        if (left_range_block_idx != right_range_block_idx) {
            int right_range_index = right_range_block_idx * num_heads + bidh;

            // Try to acquire the second lock
            #pragma unroll 1
            while (atomicCAS(&range_lock[right_range_index * 2], right_range_sync_num, right_range_sync_num) != right_range_sync_num) {
            }
        }
    }
    
    CUTLASS_DEVICE
    void deterministic_arrive(int* range_lock, int bidh, int offset, int q_block_size, int num_heads, int arrive_num, bool left_range_arrive_twice, bool right_range_arrive_twice) {
        // Calculate lock indices
        int left_range_block_idx = offset / q_block_size;
        int left_range_index = left_range_block_idx * num_heads + bidh;
        int right_range_block_idx = (offset + q_block_size - 1) / q_block_size;
        int right_range_index = right_range_block_idx * num_heads + bidh;

        // Release the second lock
        int add_cnt = right_range_arrive_twice ? 2 : 1;
        int tmp = atomicAdd(&range_lock[right_range_index * 2 + 1], add_cnt);
        if (tmp + add_cnt == 2) {
            atomicExch(&range_lock[right_range_index * 2 + 1], 0);
            atomicExch(&range_lock[right_range_index * 2], arrive_num);
        }

        // Release the first lock
        add_cnt = left_range_arrive_twice ? 2 : 1;
        tmp = atomicAdd(&range_lock[left_range_index * 2 + 1], add_cnt);
        if (tmp + add_cnt == 2) {
            atomicExch(&range_lock[left_range_index * 2 + 1], 0);
            atomicExch(&range_lock[left_range_index * 2], arrive_num);
        }
    }

  template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
  CUTLASS_DEVICE void store(
      Params const& params,
      FrgTensorO const& tdKrdK,
      FrgTensorO const& tdVrdV,
      SharedStorage& shared_storage,
      TiledMma tiled_mma,
      int thread_idx,
      BlockCoordType const& block_coord) {
    // Get block coordinates for current job(tile)
    int n_block = get<0>(block_coord);
    int bidh = get<1>(block_coord);
    int bidb = get<2>(block_coord);
    int left_range_conflict_msg = 0, right_range_conflict_msg = 0;
    if constexpr (Deterministic) {
      left_range_conflict_msg = get<3>(block_coord);
      right_range_conflict_msg = get<4>(block_coord);
    }
    int bidh_idx_in_group;
    int bidh_kv = params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh);
    Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKV{}));
    Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKV{}));
    Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKVt{}));
    Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKVt{}));
    auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);

    // Convert the type of tdVrdV and tdKrdK to Element if they are not the same as ElementAccum
    Tensor tdVrdV_out = [&] {
      if constexpr (IsSameType) {
        return tdVrdV;
      } else {
        auto out = make_tensor_like<Element>(tdVrdV);
        flash::convert_type_out(tdVrdV, out);
        return out;
      }
    }();

    Tensor tdKrdK_out = [&] {
      if constexpr (IsSameType) {
        return tdKrdK;
      } else {
        auto out = make_tensor_like<Element>(tdKrdK);
        flash::convert_type_out(tdKrdK, out);
        return out;
      }
    }();

    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out); // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out); // ((Atom,AtomNum), MMA_M, MMA_N)
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_dKV); print(sdK); printf("\n"); print(sdKt); printf("\n"); }
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Make sure all WGs have finished reading K and V
    flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);

    cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
    cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    flash::DistributedSeqlenInfo seqlen_info{bidb, params.q_ranges, params.k_ranges};
    Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK)(_, _, bidh_kv);
    Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dK)(_, _, bidh_kv);
    Tensor gdK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (M, K)
    Tensor gdV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (M, K)
    auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
    auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
    Tensor tdKgdK = block_tma_dK.partition_D(gdK); // (TMA, TMA_M, TMA_K)
    Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
    Tensor tdVgdV = block_tma_dV.partition_D(gdV); // (TMA, TMA_M, TMA_K)
    Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)

    int offset_k = seqlen_info.offset_k;

    int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
    if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
      if constexpr (Deterministic) {
        if (cute::elect_one_sync()) {
          int qheads_per_kheads = params.qhead_per_khead_divmod;
          // batch i use [i * qheads_per_kheads + 1 , (i + 1) * qheads_per_kheads] for add rank of same khead
          // conflict_msg >> 1 is bidb + 1 of conflict bidb when conflict with previous batch
          // if not conflict, conflict_bidb is 0
          int sync_num1 = bidh_idx_in_group ? bidb * qheads_per_kheads + bidh_idx_in_group : (left_range_conflict_msg >> 1) * qheads_per_kheads;
          int sync_num2 = bidh_idx_in_group ? bidb * qheads_per_kheads + bidh_idx_in_group : (right_range_conflict_msg >> 1) * qheads_per_kheads;
          deterministic_sync(params.determin_range_locks, bidh_kv, offset_k + n_block * kBlockN, kBlockN, params.nheads, sync_num1, sync_num2);
        }
      }
      cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      if (cute::elect_one_sync()) {
        cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
        cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
        tma_store_arrive();
      }
    }
    tma_store_wait<0>();
    if constexpr(Deterministic) {
      if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1 && cute::elect_one_sync()) {
        int qheads_per_kheads = params.qhead_per_khead_divmod;
        int arrive_num = bidb * qheads_per_kheads + bidh_idx_in_group + 1;
        deterministic_arrive(params.determin_range_locks, bidh_kv, offset_k + n_block * kBlockN, kBlockN, params.nheads, arrive_num, left_range_conflict_msg & 1, right_range_conflict_msg & 1);
      }
    }
    // // Tell warp 0 that smem_k and smem_v are ready
    // cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
  }

  CUTLASS_DEVICE void store_tail() {
    // if constexpr (Use_TMA) { tma_store_wait<0>(); }
  }

  // Write 0 to dK and dV
  CUTLASS_DEVICE void store_zero(Params const& params, int thread_idx, cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {}
};

} // namespace flash