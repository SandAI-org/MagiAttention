#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/grid_dependency_control.h"


using namespace cute;

template<typename T_out_, uint32_t kBlockM_, uint32_t kBlockN_, class ArchTag_>
class RepeatReduceKernel {

public:
    using ArchTag = ArchTag_;
    using T_out = T_out_;
    static constexpr uint32_t kBlockM = kBlockM_;
    static constexpr uint32_t kBlockN = kBlockN_;

    using TileShapeMN = cute::Shape<Int<kBlockM>, Int<kBlockN>>;

    // (seqlen_o, hidden_size)
    using ShapeO = cute::Shape<int32_t, int32_t>;
    using StrideO = cute::Stride<int64_t, _1>;
    // (seqlen_r, hidden_size)
    using ShapeR = cute::Shape<int32_t, int32_t>; 
    using StrideR = cute::Stride<int64_t, _1>;  

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(T_out);
    static_assert(kBlockN % kGmemElemsPerStore == 0, "kBlockN must be a multiple of kGmemElemsPerStore");

    // The "Row" below refers to a Head.
    // Bytes per head
    static constexpr int kBytePerRow = kBlockN * sizeof(T_out);
    // Number of (128-byte, 64-byte, or 32-byte) blocks per head
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(T_out);
    // Number of threads required to collaboratively read/write one (128-byte, 64-byte, or 32-byte) block
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;

    // If PackGQA, we split the work of compute O_ptr among threads in the same row, so we need this to within a warp
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);

    // Number of epilogue threads must be a multiple of kGmemThreadsPerRow
    static_assert(kBlockM % kGmemThreadsPerRow == 0, "kBlockM must be a multiple of kGmemThreadsPerRow");

    // Layout of Epilogue threads, named GmemLayoutAtom
    using GmemLayoutAtom = Layout<Shape <Int<kBlockM / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T_out>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));  // Val layout, 8 or 16 vals per store

    static constexpr int SharedStorageSize = 0;
    static constexpr uint32_t MaxThreadsPerBlock = kBlockM;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

    struct Arguments {
        T_out* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;

        T_out* ptr_R;
        ShapeR const shape_R;
        StrideR const stride_R;

        int64_t const * cu_split_size_o;
        int64_t const * split_size_list;
        int64_t const * cu_split_size_r;
        int64_t const * num_repeats_list;

        int64_t const num_splits;
        int64_t const max_split_size;
    };

    struct Params {
        T_out* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;

        T_out* ptr_R;
        ShapeR const shape_R;
        StrideR const stride_R;
        
        int64_t const * cu_split_size_o;
        int64_t const * split_size_list;
        int64_t const * cu_split_size_r;
        int64_t const * num_repeats_list;

        int64_t const num_splits;
        int64_t const max_split_size;
    };

    static
    Params
    to_underlying_arguments(
        Arguments const &args
    ){
        return {
            args.ptr_O,
            args.shape_O,
            args.stride_O,

            args.ptr_R,
            args.shape_R,
            args.stride_R,
            
            args.cu_split_size_o,
            args.split_size_list,
            args.cu_split_size_r,
            args.num_repeats_list,

            args.num_splits,
            args.max_split_size
        };
    }

    static dim3
    get_grid_shape(Params const& params) {
        int64_t hidden_size = get<1>(params.shape_O);
        int64_t num_BlockN = cute::ceil_div(hidden_size, kBlockN);
        int64_t num_blocks = cute::ceil_div(params.max_split_size, kBlockM);
        return dim3(num_blocks, num_BlockN, params.num_splits);
    }

    static dim3
    get_block_shape() {
        return dim3(kBlockM, 1, 1);
    }

    CUTLASS_DEVICE
    void
    operator()(
        Params const &params,
        char *smem_buf
    ) { 
        // Get block coordinates
        int32_t const block = blockIdx.x;
        int32_t const bidh = blockIdx.y;
        int32_t const bidb = blockIdx.z;

        // Get thread coordinates
        int32_t const thread_idx = threadIdx.x;

        // // Get offset and shape of the output tensor
        int32_t const hidden_size = cute::get<1>(params.shape_O);
        int32_t const offset_o = params.cu_split_size_o[bidb];

        int64_t const split_size = params.split_size_list[bidb];
        int64_t const offset_r = params.cu_split_size_r[bidb];
        int64_t const num_repeats = params.num_repeats_list[bidb];

        static constexpr uint32_t kBlockM = get<0>(TileShapeMN{});
        int64_t const max_block = cute::ceil_div(split_size, kBlockM);
        if (block >= max_block) {
            return;
        }

        // Initialize gmem_tiled_copy_O and gmem_thr_copy_O
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

        // Initialize global tensors
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O);
        Tensor gO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mO) , TileShapeMN{}, make_coord(block, bidh));  // (M, K)

        // Initialize tOrO and copy from tOgO
        Tensor tOrO = make_fragment_like(gmem_thr_copy_O.partition_D(make_tensor<T_out>(TileShapeMN{})));
        Tensor tOgO = gmem_thr_copy_O.partition_S(gO);
        cute::copy(gmem_tiled_copy_O, tOgO, tOrO);
        
        // Initialize tOrR
        Tensor tOrR = make_fragment_like(gmem_thr_copy_O.partition_D(make_tensor<T_out>(TileShapeMN{})));

        // For-loop each partial split
        for (int i = 0; i < num_repeats; ++i) {
            int offset_r_i = offset_r + i * split_size;

            Tensor mR = make_tensor(make_gmem_ptr(params.ptr_R), params.shape_R, params.stride_R);
            Tensor gR = local_tile(cute::domain_offset(make_coord(offset_r_i, _0{}), mR) , TileShapeMN{}, make_coord(block, bidh));  // (M, K)
            Tensor tOgR = gmem_thr_copy_O.partition_S(gR);

            cute::copy(gmem_tiled_copy_O, tOgR, tOrR);

            // Add tOrR to tOrO
            #pragma unroll
            for (int32_t i = 0; i < size(tOrO); ++i) {
                tOrO(i) += tOrR(i);
            }
        }

        cute::copy(gmem_tiled_copy_O, tOrO, tOgO);
    }
};