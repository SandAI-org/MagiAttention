---
blogpost: true
date: Jan 21, 2026
author: Jin Li, Zewei Tao, Qiangang Wang, Yunpeng Huang
location: China
category: MagiAttention
tags: Dynamic Load Balance, Sparse Attention, Hybrid Attention, Distributed Attention, Context Parallelism
---

# Dynamic Attention Solver

## Introduction

Context Parallelism (CP) shards sequence activations across distributed devices to overcome memory constraints in long-context training. In MagiAttention, the `static attn solver` has largely solved CP scheduling for standard long-context workloads where masks are static or deterministic prior to the iteration (e.g. causal, causal document, sliding window). By leveraging initial token dispatch before the iteration starts, the static solver ensures compute balance across ranks while restricting data movement to **{math}`\mathrm{KV}`-communication only**, keeping Query/Output ({math}`\mathrm{QO}`) tokens fixed on their host devices.

### Limitations of the Static Solver

To achieve computation load balancing across distributed ranks, the static solver relies on **pre-iteration token dispatch**: it splits the sequence into chunks, estimates the FLOP workload for each chunk, and heuristically dispatches these chunks to CP ranks prior to execution.

Crucially, this approach requires the global attention mask to be **static and fully deterministic before the iteration begins**, so that each chunk's attention FLOP workload can be evaluated in advance to drive the heuristic dispatch algorithm.

However, the static solver faces two fundamental limitations that prevent it from supporting dynamic sparse attention mechanisms:
- **Runtime Dynamic Sparse Attention**: Architectures such as DeepSeek Sparse Attention (DSA) {cite}`deepseekai2025deepseekv32pushingfrontieropen` and Native Sparse Attention (NSA) {cite}`yuan2025nativesparseattentionhardwarealigned` compute sparse {math}`\mathrm{KV}` selections at runtime via lightweight indexers during each layer's forward pass. Because the resulting attention masks depend directly on dynamic activation states and learned parameters, mask structure cannot be known prior to iteration execution, making pre-iteration token dispatch completely infeasible.
- **CPU-Only Mask Representation**: The static solver only accepts **CPU-side mask descriptors** (e.g., mask type enums, document boundary indices). It cannot directly consume attention masks represented as **GPU-resident tensors**. Many modern attention variants naturally produce or manipulate masks as on-device tensors, making them incompatible with the static solver's CPU-only mask interface.

These limitations leave the static solver incapable of supporting modern dynamic and device-native attention mechanisms.

### Core Problem & Design Approach

The core systems challenge in dynamic Context Parallelism is: **under strict activation memory balance constraints (where each rank maintains a uniform sequence shard), how to perform online workload partitioning and compute-communication co-scheduling when token dispatching/reordering is unavailable and attention masks are resolved dynamically at runtime?**

To address this, MagiAttention introduces the **`Dynamic Attention Solver`**. The core idea rests on two key design choices:

1. **Fixed Token Sharding with {math}`\mathrm{QO}`-Comm Enabled**: Without pre-iteration token dispatch, each rank holds a fixed sequence shard. In this setup, communicating KV only strictly fixes each rank's computation workload. This leaves no room for load rebalancing. Therefore, the solver enables communication for both {math}`\mathrm{QO}` and {math}`\mathrm{KV}` (**{math}`\mathrm{QO}`-comm enabled**). Communicating both {math}`\mathrm{QO}` and {math}`\mathrm{KV}` allows any computation workload to be scheduled onto any rank.
2. **Per-Layer Online Solving**: During each attention layer's forward pass, the dynamic solver perceives the current layer's mask structure, calculates the FLOP workload for each attention tile, and dynamically assigns computation tasks to CP ranks. It balances compute FLOPs across devices while minimizing the maximum communication volume among all ranks (i.e., bottleneck minimization), generating execution metadata (`CalcMeta` and `CommMeta`) **in real time**.

```
            ┌─────────────────────────────────────────────────────────┐
            │               Context Parallelism Design                │
            └────────────────────────────┬────────────────────────────┘
                                         │
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
     MagiAttention Static Solver                     MagiAttention Dynamic Solver
┌───────────────────────────────────┐             ┌────────────────────────────────────┐
│ • Mask: Static / Pre-iter known   │             │ • Mask: Dynamic / Per-layer runtime│
│ • Token Sharding: Heuristic       │             │ • Token Sharding: Fixed sequential │
│   reordering (Pre-iteration)      │             │   sharding                         │
│ • Comm Space: KV-only comm        │             │ • Comm Space: QO & KV dual comm    │
│ • Schedule: Local QO, pull KV     │             │ • Schedule: Dynamic tile-to-rank   │
└───────────────────────────────────┘             └────────────────────────────────────┘
```

In the following sections, we describe the detailed formulation and scheduling algorithms of the `dynamic attn solver` ([Overview](#overview)), present its user interface ([User Interface](#user-interface)), and outline current limitations and future roadmap ([Current Limitations & Future Roadmap](#current-limitations--future-roadmap)).


## System Abstraction & Cost Modeling

The **Dynamic Attention Solver** formulates Context Parallelism (CP) scheduling as a **mask-aware online task-to-rank mapping problem without token reordering**. By maintaining sequence tokens in their original, contiguous physical layout {math}`(0..P-1)`, it preserves strict activation memory balance. For each distinct dynamic mask, the solver inspects its structure to perceive true FLOP workloads across {math}`\mathrm{Q} \times \mathrm{KV} \times \text{Head}` dimensions. Its core objective is to **strictly equalize non-zero compute FLOPs across ranks while minimizing bottleneck {math}`\mathrm{QO}`/{math}`\mathrm{KV}` communication volume**.

### Fixed Sequential Sharding

Without pre-iteration token dispatch, the dynamic solver adopts **fixed sequential sharding** as its default data layout. A sequence of length {math}`S` is evenly partitioned into {math}`P` contiguous chunks {math}`(0..P-1)`, where rank {math}`p` stores the {math}`p`-th chunk. This ensures every rank holds exactly {math}`S/P` tokens, guaranteeing perfectly balanced activation memory across all CP ranks by construction.

Without prior knowledge of the attention mask, sequential sharding serves as an effective mask-agnostic layout that preserves token locality. By storing adjacent tokens on the same rank, it aligns with common attention patterns where neighboring tokens frequently attend to each other. Maintaining this locality increases the proportion of local computation and reduces cross-rank data transfers, making sequential sharding friendly to the dynamic solver.

Furthermore, this sharding induces a natural grid structure over the global attention space. With {math}`P` query chunks and {math}`P` key-value chunks, the full {math}`\mathrm{Q} \times \mathrm{KV}` attention matrix is partitioned into a {math}`P \times P` grid. Block {math}`(i, j)` represents the attention between query chunk {math}`i` and key-value chunk {math}`j`. Blocks on the principal diagonal {math}`(i, i)` are **local** to rank {math}`i`, whereas off-diagonal blocks require communication to compute.

```{figure} ../../../assets/dynamic_solver/sequential_sharding.png
:name: sequential_sharding
:align: center
:width: 600px
:alt: Fixed Sequential Sharding

Illustration of fixed sequential sharding with {math}`P = 4` ranks on a variable-length (varlen) causal mask containing two sequence samples of lengths 21 and 11. The vertical axis represents Query/Output ({math}`\mathrm{QO}`) and the horizontal axis represents Key/Value ({math}`\mathrm{KV}`), both ordered by token ID. Colored entries indicate valid non-masked positions requiring computation. Blue tiles represent {math}`\mathrm{KV}` stored on rank `r1`, yellow tiles represent {math}`\mathrm{QO}` stored on rank `r2`, and green tiles represent the cross-rank attention block where {math}`\mathrm{QO}` resides on `r2` while {math}`\mathrm{KV}` resides on `r1`. Because {math}`\mathrm{QO}` and {math}`\mathrm{KV}` belong to different ranks, the green block cannot be computed locally and strictly requires inter-rank communication.
```

### Computation Cost Estimation

The computational workload of attention is directly estimated by the **sum of effective mask areas**. Since attention FLOPs scale linearly with the number of valid (non-masked) entries, the global attention space ({math}`\mathrm{Q} \times \mathrm{KV} \times \text{Head}`) is partitioned into basic computation blocks {math}`B_{i,j,h}`, where each block represents the sub-matrix computation between query chunk {math}`i`, key-value chunk {math}`j`, and attention head {math}`h`.

The workload of each block {math}`B_{i,j,h}` is quantified by its **effective mask area** {math}`A(B_{i,j,h})` — defined as the total count of valid (non-masked) entries within that sub-matrix. In practice (e.g., with packed variable-length sequences, multi-document boundaries, or dynamic sparse masks), a single block may not possess a uniform structure; instead, it can contain a combination of multiple sub-regions, arbitrary shapes, or scattered valid entries. Regardless of the internal layout, {math}`A(B_{i,j,h})` is determined by directly counting the valid non-zero entries in the corresponding mask sub-matrix.

For instance, typical patterns within a block include:
- **Full Mask**: Every entry is valid, yielding {math}`A = S_q \times S_k` (where {math}`S_q` and {math}`S_k` are the query and key-value chunk sizes).
- **Causal Mask**: Entries are restricted by causality, yielding {math}`A = \frac{S_q \times (S_q + 1)}{2}` for diagonal blocks (assuming {math}`S_q = S_k`).
- **Mixed or irregular composition**: The block intersects multiple sequence boundaries or dynamic sparse selections, forming a combination of multiple shapes or sparse patterns.

```{figure} ../../../assets/dynamic_solver/computation_cost_estimation.png
:name: computation_cost_estimation
:align: center
:width: 800px
:alt: Computation Cost Estimation

Illustration of computation workload estimation based on effective mask area across different block composition cases.
```

### Communication Cost Estimation

When a rank is assigned to compute an attention block whose required data does not fully reside locally, **inter-rank communication** is triggered to transfer the missing data chunks. The communication cost of a data chunk is estimated by its **volume**, defined as the number of tokens in the chunk multiplied by the number of attention heads:

```{math}
\text{Vol}(\text{chunk}) = S_{\text{chunk}} \times H
```

Taking the **forward pass** as an example: computing an attention block requires {math}`\mathrm{Q}`, {math}`\mathrm{K}`, {math}`\mathrm{V}` as inputs and produces {math}`\mathrm{O}` and {math}`\text{lse}` (log-sum-exp) as outputs. Since the size of {math}`\text{lse}` is negligible compared to activation tensors, it is omitted from cost estimation. Therefore, a {math}`\mathrm{QO}` chunk communication transfers {math}`\mathrm{Q}` to the computing rank and {math}`\mathrm{O}` back, while a {math}`\mathrm{KV}` chunk communication transfers {math}`\mathrm{K}` and {math}`\mathrm{V}` together.

Each communication event involves a **sender** and a **receiver**: when a data chunk is transferred from rank {math}`s` to rank {math}`d`, the transfer simultaneously increases rank {math}`s`'s **send volume** and rank {math}`d`'s **receive volume**. The solver tracks per-rank send and receive volumes separately, as both directions contribute to the overall communication bottleneck. Its secondary objective is to **minimize the maximum communication volume across all ranks** (bottleneck minimization), preventing any single rank from becoming a communication straggler.

```{figure} ../../../assets/dynamic_solver/communication_cost_estimation.png
:name: communication_cost_estimation
:align: center
:width: 800px
:alt: Communication Cost Estimation

Illustration of communication cost estimation. Under {math}`P = 4` sequential sharding, the red attention tile (bottom right) is scheduled on rank 3 (r3). While r3 exhibits high data locality for this task, a partial dependency triggers a remote fetch from r2. This cross-rank transfer costs {math}`S_{\text{chunk}} \times H`, incrementing r2's and r3's communication volumes. Remote fetch overhead is amortized via data reuse when multiple local tiles share the same chunk.
```

### Optimization Model

The online solver searches for an optimal block-to-rank mapping {math}`\mathcal{M}: \{B_{i,j,h}\} \to \{0, \dots, P-1\}` that optimizes communication and computation overlap under strict memory and dependency constraints:

1. **Compute Load Balance Constraint**: Rather than a strict minimization, compute balance is formulated as an inequality constraint using an imbalance ratio {math}`\mu`. For each rank {math}`r`, its assigned workload must be strictly less than {math}`\mu` times the total global workload:
```{math}
\sum_{B \in \mathcal{M}^{-1}(r)} A(B) < \mu \sum_{p=0}^{P-1} \sum_{B \in \mathcal{M}^{-1}(p)} A(B)
```
2. **Bottleneck Communication Minimization**: Minimize the maximum unidirectional communication volume (either send or receive) across all ranks, preventing network stragglers:
```{math}
\min_{\mathcal{M}} \max_{r \in [0, P-1]} \max \left( \text{SendVol}(r), \text{RecvVol}(r) \right)
```
3. **Local Priority & Overlap Maximization**: Prioritize assigning local tasks to their host ranks. This allows the system to immediately begin local computation, effectively overlapping it with the communication required for remote tasks.

## Dynamic Solver Algorithm

Given the optimization model defined above, the dynamic solver employs a **binary-search-driven greedy heuristic** to find a near-optimal block-to-rank mapping in real time. The algorithm is designed for parallelizable execution on multi-core CPUs with millisecond-level latency.

### Algorithm Pipeline

The solver runs three stages inside a binary search loop over the communication threshold {math}`K`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Binary Search over Threshold K                       │
│                                                                         │
│  for each candidate K in [0, K_max]:                                    │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │ Stage 1: Candidate Edge Scoring & Greedy Selection              │  │
│    │   • Score each candidate comm edge by benefit/cost ratio        │  │
│    │   • Greedily select edges under per-rank budget K               │  │
│    ├─────────────────────────────────────────────────────────────────┤  │
│    │ Stage 2: Greedy Task Assignment                                 │  │
│    │   • Build per-task executable rank set from selected edges      │  │
│    │   • Assign tasks sorted by (degree↑, area↓)                     │  │
│    │   • Local-first → USP-heuristic → min-load fallback             │  │
│    ├─────────────────────────────────────────────────────────────────┤  │
│    │ Stage 3: Local Refinement                                       │  │
│    │   • Iteratively migrate tasks from overloaded ranks             │  │
│    │   • Check feasibility: max_load ≤ μ × avg_load                  │  │
│    └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  if feasible: shrink K_max (tighter comm budget)                        │
│  else:        raise K_min (relax comm budget)                           │
│  stop when: K_max - K_min < ε × K_max                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Stage 1: Candidate Edge Scoring & Greedy Selection.**
Each candidate communication edge represents transferring a {math}`\mathrm{QO}` or {math}`\mathrm{KV}` chunk from its host rank to a computing rank. The solver scores each edge by a benefit-to-cost ratio:

```{math}
\text{score}(e) = \frac{W(e)}{C(e)}
```

where {math}`W(e)` estimates the compute workload that edge {math}`e` would unlock for the target rank (combining already-assigned tasks, a fraction of unassigned tasks, and a smoothing term), and {math}`C(e) = S_{\text{chunk}} \times H` is the communication cost. Edges are sorted by score in descending order and greedily accepted as long as neither the sender's nor receiver's accumulated cost exceeds threshold {math}`K`.

**Stage 2: Greedy Task Assignment.**
After edge selection determines data accessibility, the solver constructs a **bitmask of executable ranks** for each grid task {math}`B_{i,j}` by intersecting the {math}`\mathrm{QO}`-reachable ranks for chunk {math}`i` and {math}`\mathrm{KV}`-reachable ranks for chunk {math}`j`. Tasks are sorted by ascending degree (number of candidate ranks) and descending area, then assigned with the following priority:

1. **Local priority**: if both {math}`\mathrm{QO}` and {math}`\mathrm{KV}` reside on the same rank, assign locally (zero communication cost).
2. **USP heuristic**: prefer the rank suggested by head-group affinity mapping.
3. **Min-load fallback**: choose the candidate rank with the smallest current load.

**Stage 3: Local Refinement.**
After initial assignment, the solver performs a bounded number of refinement passes. For each task currently on an overloaded rank (load exceeding {math}`\mu \times \bar{L}`), it attempts migration to a lower-load candidate rank. This step reduces the maximum compute imbalance without changing the overall assignment structure.

### Benchmark

**Solving Latency.** Under GQA (64:8 heads) with per-device sequence length 8192, the solver completes within 30 ms for all mask types (full, causal, full document, causal document) at CP ≤ 64. For document-sparse masks specifically, solving stays below 45 ms up to CP = 128. Under MHA (64:64 heads), the task count inflates by ~8× due to head-dimension flattening; document-sparse masks remain within 32 ms at CP ≤ 32.

**Communication Volume.** Under GQA (64:8) with variable-length document packing on 8–64 GPUs (H100), the dynamic solver reduces communication volume relative to USP:
- **Full document mask**: forward −48%~55%, backward −6%~42%.
- **Causal document mask**: forward −62%~70%, backward −33%~64%.


## Overlapped Execution

Once the solver determines the block-to-rank mapping, the remaining problem is how to overlap communication with computation during execution. As the simplest possible design, the current implementation adopts a **two-stage** execution model — just a **local stage** and a **remote stage** — and generates `CalcMeta` and `CommMeta` to drive each stage. More sophisticated multi-stage pipelining is left for future work.

### Two-Stage Execution Model

**Stage 0 (Local):** Each rank immediately begins computing attention tasks whose {math}`\mathrm{QO}` and {math}`\mathrm{KV}` data both reside locally — no communication is needed. In parallel, the communication stream starts fetching remote {math}`\mathrm{Q}`, {math}`\mathrm{K}`, {math}`\mathrm{V}` chunks required by the next stage.

**Stage 1 (Remote):** After remote data arrives, each rank computes attention tasks that depend on non-local chunks. Once computation finishes, the output {math}`\mathrm{O}` is sent back (reduced) to the {math}`\mathrm{QO}` host rank.

```{figure} ../../../assets/dynamic_solver/two_stage_streaming.png
:name: two_stage_streaming
:align: center
:width: 800px
:alt: Two-Stage Overlapped Execution

Illustration of the two-stage overlapped execution model. Local attention computation overlaps with remote data fetching on separate CUDA streams.
```

The solver's local-priority assignment directly maximizes Stage 0 workload. The more local computation available, the more communication latency can be hidden behind useful work.

### CalcMeta & CommMeta

The solver output is encoded into two metadata objects that fully specify the execution plan:

- **`CalcMeta`**: records the attention tasks for each stage.
  - `local_attn_arg`: list of `(q_range, k_range, mask_type)` for Stage 0.
  - `remote_attn_args_list`: list of `(q_range, k_range, mask_type)` for Stage 1, with ranges expressed in local buffer coordinates.

- **`CommMeta`**: records the data movement plan per stage.
  - `num_remote_kv_tokens_per_stage`: number of remote {math}`\mathrm{KV}` tokens to receive at each stage.
  - `kv_group_collective_args_list`: per-stage `GroupCollectiveArg` specifying how to group-cast {math}`\mathrm{K}`, {math}`\mathrm{V}` (forward) or group-reduce {math}`\mathrm{dK}`, {math}`\mathrm{dV}` (backward).
  - `num_remote_qo_tokens_per_stage`: number of remote {math}`\mathrm{QO}` tokens to receive at each stage.
  - `qo_group_collective_args_list`: per-stage `GroupCollectiveArg` specifying how to group-cast {math}`\mathrm{Q}` (forward) or group-reduce {math}`\mathrm{dQ}` (backward), and how to reduce {math}`\mathrm{O}` back to host ranks.

  Each `GroupCollectiveArg` contains: `input_split_size_list` (how to split local data for sending), `output_split_size_list` (expected receive sizes), `dst_indices_list` (target ranks for each send chunk), and `src_index_list` (source rank for each receive chunk).


## User Interface

The dynamic solver is activated by enabling **QO-communication mode** in MagiAttention's environment configuration. Once enabled, the system automatically switches from the static solver to the dynamic solver path.

### Environment Configuration

```bash
# Enable QO-comm to activate dynamic solver
export MAGI_ATTENTION_QO_COMM=1

# Optional: enable head-group flattening for better load balance granularity
export MAGI_ATTENTION_FLATTEN_HEAD_GROUPS=1
```

The API usage is identical to the static solver — users call `magi_attn_flex_key` → `dispatch` → `calc_attn` → `undispatch` as usual. When `MAGI_ATTENTION_QO_COMM=1` is set, the system automatically switches to the dynamic solver path internally, invoking `DynamicAttnSolver.solve()` → `make_calc_meta()` → `make_comm_meta()` → `dist_attn_func()` and caching results for repeated mask patterns. No code changes are needed beyond setting the environment variable.

### Direct Solver API

For advanced use cases (e.g., simulation, benchmarking, or custom execution pipelines), the solver can be invoked directly:

```python
from magi_attention.meta.solver.dynamic_attn_solver import DynamicAttnSolver
from magi_attention.meta.algorithms import BinaryGreedyParallelDynamicAttnAlgorithm

# Create solver instance
solver = DynamicAttnSolver(
    algorithm=BinaryGreedyParallelDynamicAttnAlgorithm(),
    num_heads_q=64,
    num_heads_kv=8,
    head_dim=128,
    cp_group=cp_group,
    dispatch_meta_q=dispatch_meta_q,
    dispatch_meta_k=dispatch_meta_k,
)

# Solve for current layer's mask
solver.solve(q_ranges, k_ranges, attn_mask_type)

# Extract execution metadata
calc_meta = solver.make_calc_meta()
comm_meta = solver.make_comm_meta()

# Optionally visualize the partition result
solver.output_solve_result(visualize=True, save_path="solver_output.png")
```


## Current Limitations & Future Roadmap

### Current Limitations

1. **Host-side solving**: The solver currently runs on CPU. For dynamic sparse attention where masks are generated by on-device indexers, transferring mask metadata from device to host introduces a synchronization barrier.

2. **Two-stage overlap only**: The current execution engine supports a single local + single remote stage. Multi-stage pipelining is not yet implemented, limiting overlap efficiency.

3. **CPU overhead at large CP scales**: The solver time increases as the CP count grows, resulting in higher CPU overhead.

### Future Roadmap

1. **Device-side solver**: Design a parallel-friendly solver algorithm that runs directly on device, eliminating host-device synchronization and CPU overhead. The solver would take mask metadata as on-device tensors and output `CalcMeta`/`CommMeta` entirely on device side, achieving better scalability as CP count grows.

2. **Multi-stage overlapped execution**: Extend the execution engine to support N-stage pipelining, where remote tasks are further decomposed into sub-stages with interleaved communication and computation, maximizing overlap for high-communication workloads.


## Citation

If you find MagiAttention useful in your research, please cite:

```bibtex
@misc{magiattention2025,
  title={MagiAttention: A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training},
  author={Zewei, Tao and Yunpeng, Huang},
  year={2025},
  howpublished={\url{https://github.com/SandAI-org/MagiAttention/}},
}
```

## References

```{bibliography} refs/dynamic_solver.bib
```
