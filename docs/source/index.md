% MagiAttention documentation master file, created by
%  sphinx-quickstart on Fri Dec 23 13:31:47 2016.
%  You can adapt this file completely to your liking, but it should at least
%  contain the root `toctree` directive.

% :github_url: https://github.com/SandAI-org/MagiAttention

MagiAttention documentation
===================================

**Overview :**

Training large-scale models for video generation presents two major challenges: (1) The extremely long context length of video tokens, which reaching up to 4 million during training, results in prohibitive computational and memory overhead. (2) The combination of block-causal attention and Packing-and-Padding (PnP) introduces highly complex attention mask patterns.

To address these challenges, we propose MagiAttention, which aims to support a wide variety of attention mask types with kernel-level flexibility, while achieving linear scalability with respect to context-parallel (CP) size across a broad range of scenarios, particularly suitable for training tasks involving ultra-long, heterogeneous mask training like video-generation for Magi-1.


```{toctree}
:glob:
:maxdepth: 2
:caption: Contents

guide
```
