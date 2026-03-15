---
blogpost: true
date: Jan 24, 2026
author: Yunpeng Huang, Zewei Tao
location: China
category: MagiAttention
tags: Group Collective, DeepEP, Collective Communication, Distributed Attention, Context Parallelism
language: English
---

# Support Native Group Collective

## Introduction

With the release of [MagiAttention-v1.1.0](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.1.0), we are excited to announce the support for native group collective CUDA kernels for both intranode and internode communication, based upon the amazing work of DeepEP {cite}`deepep2025_native_grpcoll`.

Compared to the original [`AlltoAll-v` implementation](./magi_attn.md#alltoall-v-implementation), this new approach not only **mitigates the extra pre-/post-processing D2D copies due to tensor layout transfer by kernel fusion**, but also significantly improves efficiency via the optimization of **RDMA transfer de-duplication**, particularly for hierarchical CP groups spanning internode and intranode peers.


## User Interface

### Installation

### Enabling


## Implementation


## Experiments


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

```{bibliography} refs/native_grpcoll.bib
```
