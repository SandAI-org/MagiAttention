---
blogpost: true
date: Feb 7, 2026
author: Jerry Chen, Yujia Liu, Yufeng Yang, Yunpeng Huang, Zewei Tao, Qiangang Wang, Kunlun Li
location: China
category: MagiAttention
tags: Blackwell, Flex-Flash-Attention, Flash-Attention, HSTU Function Representation
language: English
---

# Support Blackwell with FFA_FA4 Backend

## Introduction

Before the release of [MagiAttention-v1.1.0](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.1.0), `MagiAttention` had supported only the Hopper GPUs, since the attention kernel backend [`Flex-Flash-Attention` (`FFA`)](./magi_attn.md#flex-flash-attention) is built upon open-sourced `Flash-Attention 3` (`FA3`) {cite}`shah2024flashattention3fastaccurateattention_blackwell_ffa_fa4`, tailored for SM90 compute capability.

To early support the latest `Blackwell` GPUs, instead of natively extending the `FFA` kernels, which is the future plan to deliver utmost flexibility and performance potential, we have been actively collaborating with MINIMAX peers and NVIDIA team and implemented a temporary attention kernel backend named `FFA_FA4`, built upon a forked [`Flash-Attention 4` (`FA4`)](https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support) and equipped with flexible mask support via an `HSTU Function` representation.

This allows us to quickly integrate `Blackwell` support into `MagiAttention` and provide users with the opportunity to leverage the enhanced SM100+ capabilities of `Blackwell` for their attention computations, while we continue to work on the native `FFA` extension for `Blackwell` in the background.


## User Interface

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

```{bibliography} refs/blackwell_ffa_fa4.bib
```
