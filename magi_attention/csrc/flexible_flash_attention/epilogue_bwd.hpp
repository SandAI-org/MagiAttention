/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/barrier.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "seqlen.h"
#include "named_barrier.hpp"
#include "../common/utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, class Element_, class ArchTag_,
          int NumEpilogueThreads_, bool dKV_swapAB_, int AtomLayoutKdKV=1>
struct CollectiveEpilogueBwd {

    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool Use_TMA = ArchTag::kMinComputeCapability >= 90;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    using GmemTiledCopydKVTMA = cute::SM90_TMA_REDUCE_ADD;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomdKVTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                          // TODO: do we have to change this if dKV_swapAB is true?
                                          decltype(cute::get<1>(TileShape_MNK{})), Int<CUTE_STATIC_V(cute::get<2>(TileShape_MNK{})) / AtomLayoutKdKV>>());
    using SmemLayoutdKVTMA = decltype(tile_to_shape(SmemLayoutAtomdKVTMA{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdKVtTMA =
        decltype(cute::composition(SmemLayoutdKVTMA{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

    using SmemLayoutAtomdKV = SmemLayoutAtomdKVTMA;
    using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdKVt =
        decltype(cute::composition(SmemLayoutdKV{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

    using SmemCopyAtomdKV = Copy_Atom<cute::DefaultCopy, Element>;

    static constexpr size_t SmemAlignmentdKV = ArchTag::kMinComputeCapability >= 90 ? cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{}) : 128;
    static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

    struct TensorStorage : cute::aligned_struct<SmemAlignmentdKV> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dk;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_k, d, head)
    using StridedKV = cute::Stride<int64_t, _1, int64_t>;

    using TMA_dKV = std::conditional_t<
        Use_TMA,
        decltype(make_tma_copy(
            GmemTiledCopydKVTMA{},
            make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapedKV{}, StridedKV{}),
            SmemLayoutdKVTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{})),  // no mcast for dKV
        std::nullptr_t
        >;

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        Element* ptr_dV;
        StridedKV const stride_dV;
        int const num_heads_q;
        int const* q_ranges;
        int const* k_ranges;
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
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
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
        return {args.ptr_dK, args.shape_dK, args.stride_dK, args.ptr_dV, args.stride_dV,
                tma_store_dK, tma_store_dV, args.q_ranges, args.k_ranges,
                cutlass::FastDivmod(cute::ceil_div(args.num_heads_q, get<2>(args.shape_dK)))};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA) {
            cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO const& tdKrdK,
          FrgTensorO const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb] = block_coord;
        int bidh_idx_in_group;
        int bidh_kv = params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh);
        Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKV{}));
        Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKV{}));
        Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKVt{}));
        Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKVt{}));
        auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma);
        auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);


        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV);        // ((Atom,AtomNum), MMA_M, MMA_N)
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_dKV); print(sdK); printf("\n"); print(sdKt); printf("\n"); }
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading K and V
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);

        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);


        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        flash::DistributedSeqlenInfo seqlen_info{bidb, params.q_ranges, params.k_ranges};
        Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK)(_, _, bidh_kv);
        Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dK)(_, _, bidh_kv);
        Tensor gdK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor gdV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
        auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
        Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
        Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
        Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
        Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)


        int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
        if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            if (cute::elect_one_sync()) {
                cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
                cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
                tma_store_arrive();
            }
        }
        tma_store_wait<0>();
        // // Tell warp 0 that smem_k and smem_v are ready
        // cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
    }

    CUTLASS_DEVICE void
    store_tail() {
        // if constexpr (Use_TMA) { tma_store_wait<0>(); }
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
        Params const& params,
        int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t> const& block_coord
        ) 
    {
    }

};

} // namespace flash