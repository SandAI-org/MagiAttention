% MagiAttention documentation master file, created by
%  sphinx-quickstart on Fri Dec 23 13:31:47 2016.
%  You can adapt this file completely to your liking, but it should at least
%  contain the root `toctree` directive.

% :github_url: https://github.com/SandAI-org/MagiAttention

MagiAttention documentation
===================================

This is a test for Overview of MagiAttention.

Features described in this documentation are classified by release status:

**Overview :**
(1) FFA, an efficient kernel based on Flash-Attention 3, supports flexible mask patterns; (2) The dispatch solver shards and dispatches packed data with ultra-long contexts and heterogeneous masks, ensuring load-balanced computation; (3) Group-Cast and Group-Reduce primitives eliminate redundant communication; (4) The overlap solver adaptively partitions communication for optimal overlap; (5) Forward and backward timelines of MagiAttention. With all techniques together, MagiAttention reach linear scalability under diverse scenarios.

Training large-scale models for video generation presents two major challenges: (1) The extremely long context length of video tokens, which reaching up to 4 million during training, results in prohibitive computational and memory overhead. (2) The combination of block-causal attention and Packing-and-Padding (PnP) introduces highly complex attention mask patterns.

To address these challenges, we propose MagiAttention, which aims to support a wide variety of attention mask types with kernel-level flexibility, while achieving linear scalability with respect to context-parallel (CP) size across a broad range of scenarios, particularly suitable for training tasks involving ultra-long, heterogeneous mask training like video-generation for Magi-1.

```{toctree}
:glob:
:maxdepth: 2
:hidden
:caption

api
```
