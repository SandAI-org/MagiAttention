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

from magi_attention.common.enum import AttnMaskType


def ceil_div(
    x: int,
    y: int,
) -> int:
    return (x + y - 1) // y


def get_n_block_min_max(
    seqlen_q: int,
    seqlen_k: int,
    attn_mask_type: AttnMaskType,
    m_block: int,
    kBlockM: int,
    kBlockN: int,
):
    n_block_max = ceil_div(seqlen_k, kBlockN)
    if attn_mask_type == AttnMaskType.FULL or attn_mask_type == AttnMaskType.INVCAUSAL:
        pass
    elif (
        attn_mask_type == AttnMaskType.CAUSAL or attn_mask_type == AttnMaskType.BICAUSAL
    ):
        # if row == column, m_idx_max is the number of 1 in the last row of the tile
        m_idx_max = min(seqlen_q, (m_block + 1) * kBlockM)
        # if row != column, need add seqlen_k - seqlen_q.
        # Note: can be negative, which means the last row is all zero
        n_block_max = min(
            n_block_max, ceil_div(max(0, m_idx_max + seqlen_k - seqlen_q), kBlockN)
        )
    n_block_min = 0
    if attn_mask_type == AttnMaskType.FULL or attn_mask_type == AttnMaskType.CAUSAL:
        n_block_min = 0
    elif attn_mask_type == AttnMaskType.INVCAUSAL or AttnMaskType.BICAUSAL:
        # if row == column, m_id_min is the number of 0 in the first row of the tile
        m_id_min = m_block * kBlockM
        n_block_min = n_block_max if m_id_min >= seqlen_k else m_id_min // kBlockN
    return n_block_min, n_block_max


def get_m_block_min_max(
    seqlen_q: int,
    seqlen_k: int,
    attn_mask_type: AttnMaskType,
    n_block: int,
    kBlockM: int,
    kBlockN: int,
):
    m_block_max = ceil_div(seqlen_q, kBlockM)
    if attn_mask_type == AttnMaskType.FULL or attn_mask_type == AttnMaskType.CAUSAL:
        pass
    elif (
        attn_mask_type == AttnMaskType.INVCAUSAL
        or attn_mask_type == AttnMaskType.BICAUSAL
    ):
        # if row == column, m_idx_max is the number of 1 in the last column of the tile
        m_idx_max = min(seqlen_k, (n_block + 1) * kBlockN)
        m_block_max = min(m_block_max, ceil_div(m_idx_max, kBlockM))
    m_block_min = 0
    if attn_mask_type == AttnMaskType.CAUSAL or attn_mask_type == AttnMaskType.BICAUSAL:
        # if row == column, m_id_min is the number of 0 in the first column of the tile
        m_id_min = n_block * kBlockN
        # if row != column, need add seqlen_q - seqlen_k.
        # Note: can be negative, which means the first column is all 1
        m_block_min = max(m_block_min, (m_id_min + seqlen_q - seqlen_k) // kBlockM)
    elif attn_mask_type == AttnMaskType.INVCAUSAL or AttnMaskType.FULL:
        pass
    return m_block_min, m_block_max


def calc_area_qkrange(
    seqlen_q: int,
    seqlen_k: int,
    attn_mask_type: AttnMaskType,
    kBlockM: int,
    kBlockN: int,
    tile_q: bool,
) -> int:
    area = 0
    if tile_q:
        m_block_num = ceil_div(seqlen_q, kBlockM)
        for i in range(m_block_num):
            n_block_min, n_block_max = get_n_block_min_max(
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                attn_mask_type=attn_mask_type,
                m_block=i,
                kBlockM=kBlockM,
                kBlockN=kBlockN,
            )
            if n_block_max >= n_block_min:
                area += n_block_max - n_block_min
    else:  # tile k range
        n_block_num = ceil_div(seqlen_k, kBlockN)
        for i in range(n_block_num):
            m_block_min, m_block_max = get_m_block_min_max(
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                attn_mask_type=attn_mask_type,
                n_block=i,
                kBlockM=kBlockM,
                kBlockN=kBlockN,
            )
            if m_block_max > m_block_min:
                area += m_block_max - m_block_min
    return area


def calc_area_qkranges(
    q_ranges: list,
    k_ranges: list,
    attn_mask_type: list,
    kBlockM: int,
    kBlockN: int,
    tile_q: bool,
) -> int:
    area = 0
    batch_size = len(q_ranges)
    for i in range(batch_size):
        seqlen_q = q_ranges[i][1] - q_ranges[i][0]
        seqlen_k = k_ranges[i][1] - k_ranges[i][0]
        area += calc_area_qkrange(
            seqlen_q, seqlen_k, attn_mask_type[i], kBlockM, kBlockN, tile_q
        )
    return area


def calc_uncover_ranges_gaps(
    ranges: list,
) -> list:
    if not ranges:
        return []
    ranges.sort(key=lambda x: x[0])
    uncovered = []
    last_end = ranges[0][0]
    for start, end in ranges:
        if start >= last_end:
            uncovered.append([last_end, start])
        last_end = max(last_end, end)
    # if need last gaps
    # uncovered.append([last_end, float('inf')])
    return uncovered


def calc_ranges_sum_lens(
    ranges: list,
) -> int:
    sum_lens = 0
    for start, end in ranges:
        sum_lens += end - start
    return sum_lens


def calc_merged_ranges_set(
    ranges: list,
) -> list:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0])

    merged = [sorted_ranges[0]]

    for current in sorted_ranges[1:]:
        last = merged[-1]

        if current[0] <= last[1]:
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)

    return merged


def calc_overlap_ranges_set(
    ranges: list,
    local_ranges: list,
) -> list:
    rangesA = calc_merged_ranges_set(ranges)
    rangesB = calc_merged_ranges_set(local_ranges)
    i = j = 0
    intersections = []
    while i < len(rangesA) and j < len(rangesB):
        a_start, a_end = rangesA[i]
        b_start, b_end = rangesB[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersections.append([start, end])
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return calc_merged_ranges_set(intersections)


def local_calc_remote_hold_ranges_comm_lens(
    local_ranges: list,
    bucket: list,  # required range of all rank
    cp_rank: int,  # rank id
) -> int:
    merge_ranges = calc_merged_ranges_set(bucket[cp_rank] + local_ranges)
    merge_ranges_sum_lens = calc_ranges_sum_lens(merge_ranges)
    local_ranges_sum_lens = calc_ranges_sum_lens(local_ranges)
    return merge_ranges_sum_lens - local_ranges_sum_lens


def local_hold_remote_calc_ranges_comm_lens(
    local_ranges: list,
    bucket: list,  # required range of all rank
    cp_rank: int,  # rank id
) -> int:
    send_lens = 0
    for i in range(len(bucket)):
        if i != cp_rank:
            send_ranges = calc_overlap_ranges_set(bucket[i], local_ranges)
            send_lens += calc_ranges_sum_lens(send_ranges)
    return send_lens


def calc_full_comm_meta_from_qk_ranges(
    q_ranges: list,
    k_ranges: list,
    attn_mask_type: list,
    # total_seqlen_q: int,
    # total_seqlen_k: int,
    # cp_size: int,
    # cp_rank: int,
    kBlockM: int,
    kBlockN: int,
):
    if len(q_ranges) != len(k_ranges):
        raise ValueError("q_ranges must equal k_ranges")

    rectangles = list(zip(q_ranges, k_ranges, attn_mask_type))

    rectangles.sort(key=lambda x: (x[1][0], x[1][1]))

    sorted_q, sorted_k, sorted_attn_mask_type = zip(*rectangles)

    for rec in rectangles:
        area = calc_area_qkrange(rec[0], rec[1], rec[2], kBlockM, kBlockN, tile_q=True)
        print(area)

    print(sorted_q)
    print(sorted_k)
    print(sorted_attn_mask_type)
    print(calc_uncover_ranges_gaps(list(sorted_k)))


def calc_load_area_message(
    bucket: list,  # q k range mask type of all rank
    kBlockM: int,
    kBlockN: int,
    tile_q: bool,
) -> list:
    cp_size = len(bucket)
    areas = []
    for i in range(cp_size):
        q_ranges = bucket[i][0]
        k_ranges = bucket[i][1]
        attn_mask_type = bucket[i][2]
        area = calc_area_qkranges(
            q_ranges, k_ranges, attn_mask_type, kBlockM, kBlockN, tile_q
        )
        areas.append(area)
    return areas


def calc_comm_message(
    bucket: list,  # q k range mask type of all rank
    local_ranges: list,  # local q k range of all rank
) -> list:
    cp_size = len(bucket)
    comm_list = []
    q_ranges_send_bucket = []
    k_ranges_send_bucket = []
    for i in range(cp_size):
        q_ranges_send_bucket.append(bucket[i][0])
        k_ranges_send_bucket.append(bucket[i][1])
    for i in range(cp_size):
        qo_lcrh_len = local_calc_remote_hold_ranges_comm_lens(
            local_ranges[i], q_ranges_send_bucket, i
        )
        kv_lcrh_len = local_calc_remote_hold_ranges_comm_lens(
            local_ranges[i], k_ranges_send_bucket, i
        )
        qo_lhrc_len = local_hold_remote_calc_ranges_comm_lens(
            local_ranges[i], q_ranges_send_bucket, i
        )
        kv_lhrc_len = local_hold_remote_calc_ranges_comm_lens(
            local_ranges[i], k_ranges_send_bucket, i
        )
        # fwd input q k v, output o
        # fwd send local-hold-remote-calc q k v, send local-calc-remote-hold o
        fwd_send_len = qo_lhrc_len + kv_lhrc_len * 2 + qo_lcrh_len
        # fwd recv local-calc-remote-hold q k v, recv local-hold-remote-calc o
        fwd_recv_len = qo_lcrh_len + kv_lcrh_len * 2 + qo_lhrc_len
        # bwd input q k v do, output dq dk dv
        # bwd send local-hold-remote-calc q k v do, send local-calc-remote-hold dq dk dv
        bwd_send_len = qo_lhrc_len * 2 + kv_lhrc_len * 2 + qo_lcrh_len + kv_lcrh_len * 2
        # bwd recv local-calc-remote-hold q k v do, recv local-hold-remote-calc dq dk dv
        bwd_recv_len = qo_lcrh_len * 2 + kv_lcrh_len * 2 + qo_lhrc_len + kv_lhrc_len * 2
        comm_list.append([fwd_send_len, fwd_recv_len, bwd_send_len, bwd_recv_len])
    return comm_list


def eval_solver_result(
    bucket: list,  # q k range mask type of all rank
    local_ranges: list,  # local q k range of all rank
    kBlockM: int,
    kBlockN: int,
):
    fwd_load_areas = calc_load_area_message(bucket, kBlockM, kBlockN, True)
    bwd_load_areas = calc_load_area_message(bucket, kBlockM, kBlockN, False)
    comm_msg = calc_comm_message(bucket, local_ranges)
    print(f"fwd_load_areas:{fwd_load_areas}")
    print(f"bwd_load_areas:{bwd_load_areas}")
    print(f"comm_msg:{comm_msg}")
    cp_size = len(bucket)
    total_fwd_area = 0
    total_bwd_area = 0
    max_fwd_area = 0
    max_bwd_area = 0
    max_fwd_comm_len = 0
    max_bwd_comm_len = 0
    for i in range(cp_size):
        total_fwd_area += fwd_load_areas[i]
        total_bwd_area += bwd_load_areas[i]
        max_fwd_area = max(max_fwd_area, fwd_load_areas[i])
        max_bwd_area = max(max_bwd_area, bwd_load_areas[i])
        max_fwd_comm_len = max(max_fwd_comm_len, max(comm_msg[i][0], comm_msg[i][1]))
        max_bwd_comm_len = max(max_bwd_comm_len, max(comm_msg[i][2], comm_msg[i][3]))
    print(
        f"max_fwd_area:{max_fwd_area}\t\t load_balance_rate:{max_fwd_area * cp_size / total_fwd_area}"
    )
    print(f"max_fwd_comm_len:{max_fwd_comm_len}")
    print(
        f"max_bwd_area:{max_bwd_area}\t\t load_balance_rate:{max_bwd_area * cp_size / total_bwd_area}"
    )
    print(f"max_bwd_comm_len:{max_bwd_comm_len}")


if __name__ == "__main__":
    bucket = [
        [
            [[0, 1024]],
            [[1024, 2048]],
            [AttnMaskType.FULL],
        ],
        [
            [[1024, 2048]],
            [[1024, 2048]],
            [AttnMaskType.FULL],
        ],
        [
            [[2048, 3072]],
            [[2048, 3072]],
            [AttnMaskType.FULL],
        ],
        [
            [[3072, 4096]],
            [[3072, 4096]],
            [AttnMaskType.FULL],
        ],
    ]
    local_ranges = [[[0, 1024]], [[1024, 2048]], [[2048, 3072]], [[3072, 4096]]]

    kBlockM = 128
    kBlockN = 128
    eval_solver_result(bucket, local_ranges, kBlockM, kBlockN)
