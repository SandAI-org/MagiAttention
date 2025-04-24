# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from magi_attention.common import AttnRange, AttnRanges
from magi_attention.common.enum import AttnMaskType, AttnRole, AttnType
from magi_attention.meta.collection import DispatchMeta
from magi_attention.meta.container import AttnBucket, AttnChunk, AttnSlice
from magi_attention.meta.solver.dispatch_solver import (
    DispatchConfig,
    DispatchJob,
    DispatchSolution,
    DispatchSolver,
    IOUAffinity,
    ToppHeapDispatchAlg,
)
from magi_attention.utils import (
    flatten_nested_list,
    nvtx,
    perm_idxs2unperm_idxs,
    wrap_to_list,
)

__all__ = [
    "calc_dispatch_meta_from_qk_ranges",
    "seqlens2cu_seqlens",
    "cu_seqlens2seqlens",
]


@nvtx.instrument_nvtx
def calc_dispatch_meta_from_qk_ranges(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: AttnMaskType | list[AttnMaskType],
    total_seqlen_q: int,
    total_seqlen_k: int,
    chunk_size: int,
    cp_size: int,
    cp_rank: int,
    dispatch_config: DispatchConfig,
    is_same_source: bool,
    is_q_permutable: bool,
    is_k_permutable: bool,
    high_bandwith_domain_size: int,
) -> tuple[DispatchMeta, DispatchMeta, list[AttnBucket]]:
    """Calculate dispatch meta from query and key ranges

    Args:
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (AttnMaskType | list[AttnMaskType]): attn mask type (list)

        total_seqlen_q (int): the total seqlen of query (i.e. number of rows in the ref attn mask)
        total_seqlen_k (int): the total seqlen of key (i.e. number of columns in the ref attn mask)

        chunk_size (int): chunk size to chunk the permutable tensor

        cp_size (int): context-parallel world size
        cp_rank (int): context-parallel local rank, ranging in [0,  cp_size)

        dispatch_config (DispatchConfig): dispatch config

        is_same_source (bool): is query tensor and key tensor share the same source
        is_q_permutable (bool): is query tensor permutable
        is_k_permutable (bool): is key tensor permutable
        NOTE: e.g.
                1. for decoder-only transformer like gpt, it applies 'self-attn' as follows:
                    a) is_same_source is True
                    b) both q and k are permutable, as long as they are permuted in the same way.
                2. for encoder-decoder transformer like t5, it applies 'cross-attn' as follows:
                    a) is_same_source is False
                    b) q is permutable but k is not
                3. for multi-modal transformer with external encoders, it applies 'cross-attn' as follows:
                    a) is_same_source is False
                    b) q is unpermutable cuz of self-attn, but k is permutable even in a different way

        high_bandwith_domain_size (int): The high bandwith domain size

    Returns:
        tuple[DispatchMeta, DispatchMeta]: dispatch_meta_q and dispatch_meta_k
        NOTE: When is_same_source is True, dispatch_meta_k should contain attributes
                that are mostly the same as those in dispatch_meta_q.
    """

    # --------------      pre-check args       -------------- #

    assert (
        total_seqlen_q % chunk_size == 0 and total_seqlen_k % chunk_size == 0
    ), f"Both {total_seqlen_q=} and {total_seqlen_k=} should be divisible by {chunk_size=}."

    num_chunks_q = total_seqlen_q // chunk_size
    num_chunks_k = total_seqlen_k // chunk_size
    assert (
        num_chunks_q % cp_size == 0 and num_chunks_k % cp_size == 0
    ), f"Both {num_chunks_q=} and {num_chunks_k=} should be divisible by {cp_size=}."

    shard_seqlen_q = total_seqlen_q // cp_size
    shard_seqlen_k = total_seqlen_k // cp_size

    assert len(q_ranges) == len(k_ranges), (
        f"The length of q_ranges and k_ranges (i.e. batch_size) should be the same, "
        f"but got {len(q_ranges)=}, {len(k_ranges)=}."
    )
    batch_size = len(q_ranges)

    attn_mask_type = wrap_to_list(attn_mask_type, broadcast_to_length=batch_size)
    assert (
        len(attn_mask_type) == batch_size
    ), f"If attn_mask_type is a list, its length ({len(attn_mask_type)}) should be equal to batch_size ({batch_size})."

    assert (
        dispatch_config.alg.is_partitions_returned
        and dispatch_config.alg.is_equal_num_workloads
    ), (
        "For now, only support dispatch config with "
        "the algorithm that returns the partitions, each of which shares the equal number of workloads, "
        f"bot got {dispatch_config.alg=}."
    )

    # --------------      calculate dispatch meta   -------------- #

    # TODO: for now, we seperate different settings in different functions
    # they had better be merged in the future
    match is_same_source, is_q_permutable, is_k_permutable:
        case True, True, True:
            return _calc_self_attn_dispatch_meta_from_qk_ranges(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                batch_size=batch_size,
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
                shard_seqlen_q=shard_seqlen_q,
                shard_seqlen_k=shard_seqlen_k,
                num_chunks_q=num_chunks_q,
                num_chunks_k=num_chunks_k,
                chunk_size=chunk_size,
                cp_size=cp_size,
                cp_rank=cp_rank,
                high_bandwith_domain_size=high_bandwith_domain_size,
                dispatch_config=dispatch_config,
            )
        case True, False, True | True, True, False:
            raise ValueError(
                "When is_same_source is True, "
                "is_q_permutable and is_k_permutable should be either both True or both False."
            )
        case True, False, False:
            raise NotImplementedError("A trivial case with no need to dispatch.")
        case False, True, True:
            raise NotImplementedError("An unknown case as a pure cross-attn setting.")
        case False, True, False:
            raise NotImplementedError(
                "A cross-attn setting for encoder-decoder transformer like T5."
            )
        case False, False, True:
            raise NotImplementedError(
                "A cross-attn setting for multi-modal transformer with external encoders."
            )
        case False, False, False:
            raise NotImplementedError("A trivial case with no need to dispatch.")
        case _:
            raise ValueError(
                f"Unknown case with {is_same_source=}, {is_q_permutable=}, {is_k_permutable=}."
            )


@nvtx.instrument_nvtx
def _calc_self_attn_dispatch_meta_from_qk_ranges(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    batch_size: int,
    total_seqlen_q: int,
    total_seqlen_k: int,
    shard_seqlen_q: int,
    shard_seqlen_k: int,
    num_chunks_q: int,
    num_chunks_k: int,
    chunk_size: int,
    cp_size: int,
    cp_rank: int,
    high_bandwith_domain_size: int,
    dispatch_config: DispatchConfig,
) -> tuple[DispatchMeta, DispatchMeta, list[AttnBucket]]:
    """Calculate dispatch meta from query and key ranges for self-attn settings

    Args:
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (list[AttnMaskType]): attn mask type list

        batch_size (int): batch size
        total_seqlen_q (int): total sequence length of query
        total_seqlen_k (int): total sequence length of key
        shard_seqlen_q (int): sequence length of query per cp rank
        shard_seqlen_k (int): sequence length of key per cp rank

        num_chunks_q (int): number of chunks for query
        num_chunks_k (int): number of chunks for key
        chunk_size (int): chunk size to chunk the permutable tensor

        cp_size (int): context-parallel world size
        cp_rank (int): context-parallel local rank, ranging in [0,  cp_size)

        high_bandwith_domain_size (int): The high bandwith domain size

        dispatch_config (DispatchConfig): dispatch config

    Returns:
        tuple[DispatchMeta, DispatchMeta]: dispatch_meta_q and dispatch_meta_k
        NOTE: When is_same_source is True, dispatch_meta_k should contain attributes
                that are mostly the same as those in dispatch_meta_q.
    """

    # --------------      pre-check args       -------------- #

    assert total_seqlen_q == total_seqlen_k and num_chunks_q == num_chunks_k, (
        f"For self-attn, {total_seqlen_q=} should be the same as {total_seqlen_k=}, "
        f"as well as {num_chunks_q=} and {num_chunks_k=}"
    )

    # --------------    extract some trivial meta info   -------------- #

    total_seqlen = total_seqlen_q
    num_chunks = num_chunks_q
    shard_seqlen = shard_seqlen_q
    max_valid_ids = max(q_range.end for q_range in q_ranges)

    # q_ranges can be transferred to cu_seqlens_q for sure
    assert q_ranges.is_cu_seqlens(
        total_seqlen
    ), "q_ranges should be cu_seqlens for now."
    cu_seqlens = q_ranges.to_cu_seqlens(total_seqlen)
    seqlens = cu_seqlens2seqlens(cu_seqlens)

    # NOTE: for now, we don't permute seqlens
    # but we keep this general functionality
    seqlens_perm_idxs = list(range(len(seqlens)))
    seqlens_unperm_idxs = perm_idxs2unperm_idxs(seqlens_perm_idxs)
    seqlens_permed = [seqlens[i] for i in seqlens_perm_idxs]
    cu_seqlens_permed = seqlens2cu_seqlens(seqlens_permed)

    # -------    calculate attn areas to construct an undispatch bucket   ------- #

    global_bucket: AttnBucket = _calc_self_attn_areas(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        attn_mask_type=attn_mask_type,
    )
    attn_areas = global_bucket.areas

    # -------    solve dispatch load balancing and get chunk partitions   ------- #

    dispatch_solver = DispatchSolver(alg=dispatch_config.alg)
    affinities = None
    if isinstance(dispatch_config.alg, ToppHeapDispatchAlg):
        affinities = [
            IOUAffinity.from_ranges(chunk.k_ranges) for chunk in global_bucket.q_chunks
        ]
    dispatch_jobs = DispatchJob.from_job_list(
        workloads=attn_areas,  # type: ignore[arg-type]
        affinities=affinities,  # type: ignore[arg-type]
    )
    dispatch_solution: DispatchSolution = dispatch_solver.solve(
        jobs=dispatch_jobs,
        num_buckets=cp_size,
    )
    partitions = dispatch_solution.bucket_partitions

    # since the order for any partition of chunk ids doesn't matter,
    # here we just keep it sorted ascendingly, like (0,5,4) -> (0,4,5)
    partitions = [sorted(p) for p in partitions]
    partitions_perm_idxs = flatten_nested_list(partitions)
    partitions_unperm_idxs = perm_idxs2unperm_idxs(partitions_perm_idxs)

    # --------------      construct buckets per rank       -------------- #

    buckets_per_rank: list[AttnBucket] = [
        AttnBucket(
            cp_rank=rank,
            q_chunks=[global_bucket.q_chunks[chunk_id] for chunk_id in partition],
        )
        for rank, partition in enumerate(partitions)
    ]

    # --------------      construct meta q and meta k       -------------- #

    common_meta_kwargs = dict(
        attn_type=AttnType.SELF_ATTN,
        attn_mask_type=attn_mask_type,
        batch_size=batch_size,
        total_seqlen=total_seqlen,
        shard_seqlen=shard_seqlen,
        max_valid_ids=max_valid_ids,
        cp_rank=cp_rank,
        cp_size=cp_size,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        seqlens=seqlens,
        seqlens_permed=seqlens_permed,
        seqlens_perm_idxs=seqlens_perm_idxs,
        seqlens_unperm_idxs=seqlens_unperm_idxs,
        cu_seqlens=cu_seqlens,
        cu_seqlens_permed=cu_seqlens_permed,
        partitions=partitions,
        partitions_perm_idxs=partitions_perm_idxs,
        partitions_unperm_idxs=partitions_unperm_idxs,
        global_bucket=global_bucket,
        buckets_per_rank=buckets_per_rank,
        high_bandwith_domain_size=high_bandwith_domain_size,
    )

    meta_q = DispatchMeta(
        attn_role=AttnRole.QUERY,
        ranges=q_ranges,
        **common_meta_kwargs,  # type: ignore
    )
    meta_k = DispatchMeta(
        attn_role=AttnRole.KEY,
        ranges=k_ranges,
        **common_meta_kwargs,  # type: ignore
    )

    return meta_q, meta_k, buckets_per_rank


@nvtx.instrument_nvtx
def _calc_self_attn_areas(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    num_chunks: int,
    chunk_size: int,
) -> AttnBucket:
    """Compute the self-attn areas, with constructing the global bucket,
    which is mainly consists of a list of all the chunks in ascending order, with a length of `cp_size`

    Args:
        q_ranges (AttnRanges): the query ranges
        k_ranges (AttnRanges): the key ranges
        attn_mask_type (list[AttnMaskType]): the attn mask type list
        chunk_size (int | None): the chunk size, which should be divisible by `cp_size`

    Returns:
        global_bucket(AttnBucket): the global bucket
    """

    # -----------    init meta info and global bucket    ----------- #

    global_bucket = AttnBucket()
    range_idx, seqi_mid = 0, 0

    # -----------    compute attn areas for self-attn settings    ----------- #

    for chunk_id in range(num_chunks):  # for each chunk
        chunk: AttnChunk = AttnChunk(chunk_id=chunk_id)
        cur_chunk_size = 0

        slice_id = 0
        while cur_chunk_size < chunk_size:  # for each slice
            mask_type = attn_mask_type[range_idx]
            is_causal = mask_type == AttnMaskType.CAUSAL

            slice: AttnSlice = AttnSlice(slice_id=slice_id, mask_type=mask_type)

            seqi_end = q_ranges[range_idx].end
            seqi_len_bottom = seqi_end - seqi_mid

            attn_len = k_ranges[range_idx].seqlen
            attn_start, attn_end = (
                k_ranges[range_idx].start,
                k_ranges[range_idx].end,
            )

            exceed_size = seqi_len_bottom + cur_chunk_size - chunk_size

            q_range_start, q_range_end, k_range_start, k_range_end = (
                None,
                None,
                None,
                None,
            )

            # analyze this slice
            # HACK: The logic for calculating the area will be encapsulated in AttnSlice later, and the area will be read-only.
            # This feature of directly setting the area is retained here for now.
            if exceed_size <= 0:  # this bottom half of seqi should be all in this chunk
                # set start and end for q_range of this slice
                q_range_start, q_range_end = seqi_mid, seqi_end

                # set start and end for k_range of this slice
                k_range_start, k_range_end = attn_start, attn_end

                # compuate areas
                if is_causal:
                    if attn_len > seqi_len_bottom:  # the area of a trapezoid
                        slice.area = (
                            (2 * attn_len - seqi_len_bottom + 1) * seqi_len_bottom // 2
                        )
                    else:  # the area of a triangle
                        slice.area = (1 + attn_len) * attn_len // 2
                else:  # the area of a rectangle
                    slice.area = seqi_len_bottom * attn_len

                # iterate to the next sample within the same chunk
                range_idx += 1
                seqi_mid = seqi_end
                cur_chunk_size += seqi_len_bottom
            else:  # only the prefix of this bottom half of seqi should be in this chunk
                # truncate the seqlen to the edge of chunk line
                seqi_end_truncate = seqi_end - exceed_size
                seqi_len_bottom_truncate = seqi_end_truncate - seqi_mid
                attn_len_truncate = attn_len - exceed_size

                # set start and end for q_range of this slice
                q_range_start, q_range_end = seqi_mid, seqi_end_truncate

                # compuate areas
                if is_causal:
                    if attn_len > seqi_len_bottom:  # the area of a trapezoid
                        slice.area = (
                            (
                                2 * (attn_len - seqi_len_bottom)
                                + seqi_len_bottom_truncate
                                + 1
                            )
                            * seqi_len_bottom_truncate
                            // 2
                        )
                        # set start and end for k_range of this slice
                        k_range_start, k_range_end = (
                            attn_start,
                            attn_start + attn_len_truncate,
                        )
                    elif attn_len > exceed_size:  # the area of a triangle
                        slice.area = (1 + attn_len_truncate) * attn_len_truncate // 2
                        # set start and end for k_range of this slice
                        k_range_start, k_range_end = (
                            attn_start,
                            attn_start + attn_len_truncate,
                        )
                    else:  # no area to compute
                        slice.area = 0
                        # set start and end for k_range of this slice
                        k_range_start, k_range_end = attn_start, attn_start
                else:  # the area of a rectangle
                    slice.area = seqi_len_bottom_truncate * attn_len
                    # set start and end for k_range of this slice
                    k_range_start, k_range_end = attn_start, attn_end

                # iterate to next chunk within the same sample
                seqi_mid = seqi_end_truncate
                cur_chunk_size = chunk_size

            # set q_range, k_range for this slice
            slice.q_range = AttnRange(start=q_range_start, end=q_range_end)
            slice.k_range = AttnRange(start=k_range_start, end=k_range_end)

            if slice.k_range.seqlen > 0 and slice.area > 0:
                # append this q slice to the current chunk except invalid slice
                chunk.q_slices.append(slice)
                slice_id += 1

        global_bucket.q_chunks.append(chunk)

    return global_bucket


def seqlens2cu_seqlens(seqlens: list[int]) -> list[int]:
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return cu_seqlens


def cu_seqlens2seqlens(cu_seqlens: list[int]) -> list[int]:
    seqlens = []
    for i in range(1, len(cu_seqlens)):
        seqlens.append(cu_seqlens[i] - cu_seqlens[i - 1])
    return seqlens
