---
layout: distill
permalink: /ffa_with_sink
title: FFA with Attention Sink
description: Integrating Flex-Flash-Attention with Attention Sink
date: 2025-11-17
featured: true
pretty_table: true
tabs: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

external-links:
  github: https://github.com/SandAI-org/MagiAttention
  arxiv: https://arxiv.org/pdf/2505.13211
  docs: https://sandai-org.github.io/MagiAttention/docs

authors:
  - name: Yunpeng Huang
    email: yunpenghuang@sand.ai
    affiliations:
      name: SandAI

bibliography: ffa_with_sink.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: Overview
  - name: User Interface
    subsections:
      - name: FFA API
      - name: MagiAttn API
      - name: Flash-Attention Extension
  - name: Math Derivation
    subsections:
      - name: FFA Forward
      - name: FFA Backward
  - name: Implementations
    subsections:
      - name: Torch Reference
      - name: FFA Impl
      - name: MagiAttn Impl
  - name: Citation
  - name: References

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

Large-Scaled Models (LMs) assign significant attention to few tokens (<em>such as the intial tokens in the sequence</em>), even if they are not semantically important, which is known as <b>attention sink</b><d-cite key="xiao2024efficientstreaminglanguagemodels,kang2025toldvisualattentionsink"></d-cite>. Researchers attribute this interesting phenomenon to the nature of $softmax$, which requires attention scores of each query token to always sum up to $1$ for all key tokens in the context, even when some query token does not strongly attend to any key token at all<d-cite key="gu2025attentionsinkemergeslanguage"></d-cite>. Therefore, during the training, we can deliberately add some <u><em>learnable sink tokens</em></u> to the key sequence for each query token to collect those unneeded attention scores to relax the <em>"sum-up-to-one"</em> constraint, which can be seen as a learnable version of $\textit{off-by-one}\space softmax$<d-cite key="miller2025attentionmisc"></d-cite>.

However, since sink tokens only affect the $softmax$ operation during the attention forward/backward passes w.r.t. the GPT-OSS implementation<d-cite key="openaiGPT-OSScode-misc"></d-cite>, <b>it is non-trivial to apply learnable attention sink with the (distributed) attention implementations in the style of <u>Flash Attention</u></b><d-cite key="dao2022flashattention,dao2023flashattention,shah2024flashattention3fastaccurateattention"></d-cite>, particularly our own kernel implemenation of <u>Flex-Flash-Attention</u>, as well as the distributed implementation of <u>MagiAttention</u><d-cite key="magiattention2025"></d-cite>.


## Overview

With the release of [MagiAttention-v1.0.5](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.5), we have not only <b>supported the learnable attention sink mechanism</b> for our own kernel / distributed implementations of <u>Flex-Flash-Attention</u> / <u>MagiAttention</u> respectively, but also <b>provided the <em>plug-and-play</em> implementations</b> to integrate the original <u>Flash Attention</u> 2/3 interface<d-cite key="daoFlashAttnInterfaceMisc,daoFlashAttnInterfaceHopperMisc"></d-cite> with attention sink, as one of the [MagiAttention Extensions](https://github.com/SandAI-org/MagiAttention/tree/main/extensions#flashattention-with-attention-sink).

In this blog, we will share our own methods about how to integrate the attention implementations in the Flash-Attention style with the learnable attention sink mechanism, including:

- the [User Interface](#user-interface) update for [Flex-Flash-Attention](#ffa-api), [MagiAttention](#magiattn-api) and [Flash-Attention Extension](#flash-attention-extension).
- the [Math Derivation](#math-derivation) of applying the attention sink in both [forward](#ffa-forward) and [backward](#ffa-backward) passes of Flex-Flash-Attention.
- the [Implementations](#implementations) of the (distributed) learnable attention sink mechanism for [Flex-Flash-Attention](#ffa-impl) and [MagiAttention](#magiattn-impl), as well as the naive [Torch Reference](#torch-reference).



## User Interface

Below, we show the minor update of the user interfaces to support learnable attention sink mechanism for original Flex-Flash-Attention, MagiAttention, as well as the Flash-Attention 2/3 as one of the [MagiAttention Extensions](https://github.com/SandAI-org/MagiAttention/tree/main/extensions#flashattention-with-attention-sink).


### FFA API

- Just add an optional tensor `sink` to the argument list of `flex_flash_attn_func`.
- And when and only when `sink` tensor is given, `flex_flash_attn_func` will apply attention sink during the forward pass, and compute `dsink`  (<em>the gradient of `sink`</em>) during the backward pass. 
- Otherwise, attention sink is skipped and `dsink` is also returned as `None`.
- dtype: `float32` only.
- shape: `[seqlen_sink, num_heads_q]`, where `seqlen_sink` in `[1, 8]`.
- interface difference with the original `flex_flash_attn_func`:

```diff
def flex_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None = None,
+   sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    disable_fwd_atomic_reduction: bool = False,
    auto_range_merge: bool = False,
    ref_block_size: tuple[int, int] | None = None,
    profile_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```


### MagiAttn API

- Just add an optional **replicated** tensor `sink` to the argument list of `calc_attn`.
- And when and only when **replicated** `sink` tensor is given, `calc_attn` will apply attention sink during the forward pass for each **local** query token, and compute **partial** `dsink` during the backward pass.
- And an `all-reduce` communication might be applied across cp ranks to return the **reduced** `dsink` if required (<em>see the environment variable `MAGI_ATTENTION_DSINK_ALL_REDUCE_OP` in our [docs](https://sandai-org.github.io/MagiAttention/docs/main/env_variables.html#for-correctness)</em>).
- Otherwise, attention sink is skipped and `dsink` is also returned as `None`.
- dtype: `float32` only.
- shape: `[seqlen_sink, num_heads_q]`, where `seqlen_sink` in `[1, 8]`.
- parallel style: `Replicate`.
- interface difference with the original `calc_attn`:

```diff
def calc_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key: DistAttnRuntimeKey,
+   sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```


### Flash Attention Extension

- Just add an optional tensor `sink` to the argument list of `flash_attn_func`, `flash_attn_varlen_func`, etc.
- And when and only when `sink` tensor is given, flash attention will apply attention sink during the forward pass, and compute `dsink` during the backward pass.
- Otherwise, attention sink is skipped and `dsink` is also returned as `None`.
- dtype: `float32` only.
- shape: `[seqlen_sink, num_heads_q]`, where `seqlen_sink` has no limit.
- interface difference with the original flash attention:

```diff
- def flash_attn_func(
+ def flash_attn_func_with_sink(
    q,
    k,
    v,
+   sink=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    ...

- def flash_attn_varlen_func(
+ def flash_attn_varlen_func_with_sink(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
+   sink=None,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    ...
```


## Math Derivation

Below, we provide the step-by-step math derivation of the original forward / backward passes for Flex-Flash-Attention (<em>the same as Flash-Attention</em>) w/o sink tokens, and then the differences when involving the learnable attention sink mechanism, serving as the guidence for our implementations in the next section.

<b>NOTE: </b>

<b>1. To simplify the derivation, we drop the `batch` dimension and only keep the `num_heads` dimension to the leftmost acting as the implicit `batch` dimension.</b>

<b>2. To focus on the attention sink mechanism, we assume you're already familiar with Flash Attention and will skip over its finer details, like the double-loop tiling strategy and the derivation of online softmax correction based on `log-sum-exp` operations.</b>

<b>3. If you are new to Flash Attention or well-interested in the full original math derivation, we highly recommend this blog post: [Flash Attention 2 Math Derivation](https://github.com/Strivin0311/llms-learning/blob/main/dev/modeling/lm/transformer/attn/fa2_deri.md).</b>

<b>Symbol Notation:</b>

| symbol              | notation                                                                           |
|-----------------------|----------------------------------------------------------------------------------|
| $\times$           | matrix multiplication                                                               |
| $\cdot$            | scalar multiplication                                                               |
| $\odot$            | element-wise multiplication (Hadamard product)                                      |
| $sq, sk, s\\_sink$ | the sequence length of query tokens, key tokens, and attention sink tokens          |
| $nhq, nhk$         | the number of heads of query tokens and key tokens                                  |
| $hd$               | the head dimension of query, key and value tokens                                   |
| $X_i$              | the column vector made by the $i$-th row of matrix $X$ along the sequence dimension |

### FFA Forward

#### FFA forward w/o sink tokens

- step1:

$$
\begin{aligned}

&S = Q \times K^{\mathrm T} \cdot scale\; + \; bias \notag \\ 

&where\; Q \in \mathbb{R}^{nhq \times sq\times  hd}, K \in \mathbb{R}^{nhk\times sk \times hd}, \notag \\ 

& scale \in \mathbb{R}^{}, \; bias \in \mathbb{R}^{nhq\times sq\times sk}, \; S \in \mathbb{R}^{nhq\times sq\times sk} \notag
\end{aligned}
$$

- step2:

$$
\begin{aligned}
& softmax_{row}(X_i) = \cfrac{\mathrm{exp}(X_i - M_i)}{L_i}, \; i \in [1,sq]\notag \\
& where\; M_i = \mathrm{rowmax}(X_i), \; L_i = \mathrm{rowsum}(\mathrm{exp}(X_i - M_i))\notag
\end{aligned}
$$

$$
\begin{aligned}
&P = \mathrm{softmax}_{row}(S) \notag \\ 
&where\; S, P \in \mathbb{R}^{nhq\times sq\times sk} \notag
\end{aligned}
$$

- step3:

$$
\begin{aligned}
&O = P \times V, \;\mathrm{LSE}_i = \log(L_i) + M_i, \; i \in [1, sq]\notag \\ 

&where\; P \in \mathbb{R}^{nhq\times sq\times sk}, \; V \in \mathbb{R}^{nhk\times sk\times hd}, \notag \\

& O \in \mathbb{R}^{nhq\times sq\times hd}, \;\mathrm{LSE} \in \mathbb{R}^{nhq\times sq}\notag

\end{aligned}
$$

#### FFA forward with sink tokens

- step1: (<em>the same</em>)

- step2:

$$
\begin{aligned}
&\tilde{P} = \mathrm{softmax}_{row}(\tilde{S}),\;\tilde{S}_i = [S_i, sink], \; i \in [1, sq] \notag \\ 
&where\; \tilde{S},\tilde{P} \in \mathbb{R}^{nhq\times sq\times (sk+s\_sink)} ,\;sink \in \mathbb{R}^{nhq\times s\_sink}\notag \\
\end{aligned}
$$

$$
\begin{aligned}
& \tilde{P}_i = [\tilde{P}^{qk}_{i}, P^{sink}_{i}],\; i \in [1, sq] \notag \\

&where\; \tilde{P}^{qk} \in \mathbb{R}^{nhq\times sq\times sk} ,\notag\\
&P^{sink} \in \mathbb{R}^{nhq\times sq\times s\_sink} \notag \\
\end{aligned}
$$

- step3:

$$
\begin{aligned}
&\tilde{O} = \tilde{P}^{qk} \times V, \;\tilde{\mathrm{LSE}}_i = \log(\tilde{L}_i) + M_i, \; i \in [1, sq]\notag \\ 

& \tilde{L}_i = L_i + \sum\limits_{j=1}^{s\_sink}\mathrm{exp}(sink_j - M_i), \; i \in [1, sq]\notag \\

& \tilde{P}^{qk}_i = P^{qk}_i \times \cfrac{L_i}{\tilde{L}_i}, \; i \in [1, sq]\notag \\

&where\; P^{qk},\tilde{P}^{qk} \in \mathbb{R}^{nhq\times sq\times sk}, \; V \in \mathbb{R}^{nhk\times sk\times hd}, \notag \\

& \tilde{O} \in \mathbb{R}^{nhq\times sq\times hd}, \;\tilde{\mathrm{LSE}} \in \mathbb{R}^{nhq\times sq}\notag

\end{aligned}
$$

- <b>sink correction</b>: (<em>as a post-processing of original ffa forward w/o sink tokens</em>)

$$
\begin{aligned}
& \mathrm{LSE}^{sink} = \log\big(  \sum\limits_{j=1}^{s\_sink}\mathrm{exp}(sink_j)\big)\notag \\

& \tilde{\mathrm{LSE}}_i = \log\big(\exp(\mathrm{LSE}_i) + \exp(\mathrm{LSE}^{sink})\big), \; i \in [1, sq]\notag \\ 

&\tilde{O} = O \cdot \exp\big(\mathrm{LSE} - \tilde{\mathrm{LSE}}  \big)\notag \\ 


&where\; sink \in \mathbb{R}^{nhq\times s\_sink},\;\mathrm{LSE}^{sink} \in \mathbb{R}^{nhq}\notag\\

&\mathrm{LSE},\tilde{\mathrm{LSE}} \in \mathbb{R}^{nhq\times sq}, \;O,\tilde{O}\in \mathbb{R}^{nhq\times sq\times hd}\;\notag

\end{aligned}
$$

### FFA Backward

#### FFA backward w/o sink tokens

- step1: (<em>as a pre-processing</em>)

$$
\begin{aligned}
&\Delta_i = P^{\mathrm T}_i \times dP_i = O^{\mathrm T}_i \times dO_i, \; i \in [1, sq] \notag\\

&\Delta = \mathrm{sum}_{hd}(O \;\odot\; dO) \notag\\ 

&where\; O,dO \in \mathbb{R}^{nhq\times sq\times hd}, \; \Delta \in \mathbb{R}^{nhq\times sq} \notag
\end{aligned}
$$

- step2:(<em>recomputation</em>)

$$
\begin{aligned}
&S = Q \times K^{\mathrm T} \cdot scale\; + \; bias \notag \\ 

&P_i = \exp\big(S_i - \mathrm{LSE}_i), \; i \in [1, sq] \notag \\

&where\; Q \in \mathbb{R}^{nhq \times sq\times  hd}, K \in \mathbb{R}^{nhk\times sk \times hd}, \;scale \in \mathbb{R}^{}\notag \\ 

&bias \in \mathbb{R}^{nhq\times sq\times sk}, \; S,P \in \mathbb{R}^{nhq\times sq\times sk}, \;\mathrm{LSE} \in \mathbb{R}^{nhq\times sq} \notag
\end{aligned}
$$

- step3:

$$
\begin{aligned}
&dV = P^{\mathrm T} \times dO \notag \\ 

&dP = dO \times V^{\mathrm T} \notag \\

&where\; P,dP \in \mathbb{R}^{nhq\times sq\times sk}\notag\\

&V,dV \in \mathbb{R}^{nhk\times sk\times hd} \notag \\

&dO \in \mathbb{R}^{nhq\times sq\times hd} \notag
\end{aligned}

$$

- step4:

$$
\begin{aligned}
&dS_i = P_i \odot (dP_i - \Delta_i), \; i \in [1, sq] \notag \\ 

&where\; P,dP \in \mathbb{R}^{nhq\times sq\times sk}\notag\\

&dS \in \mathbb{R}^{nhq\times sq\times sk},\;\Delta \in \mathbb{R}^{nhq\times sq} \notag \\
\end{aligned}

$$

- step5:

$$
\begin{aligned}
&\hat{dS} = dS \cdot scale \notag \\ 

&dQ = \hat{dS} \times K \notag \\ 

&dK = \hat{dS}^{\mathrm T} \times Q \notag \\ 

&where\; dS,\hat{dS},bias \in \mathbb{R}^{nhq\times sq\times sk}, \;scale \in \mathbb{R}^{}\notag \\

&Q,dQ \in \mathbb{R}^{nhq\times sq\times hd}, \; K,dK \in \mathbb{R}^{nhk\times sk\times hd} \notag
\end{aligned}
$$

#### FFA backward with sink tokens

- step1: (<em>as a pre-processing as well</em>)

$$
\begin{aligned}
&\tilde{\Delta}_i = \tilde{P}^{\mathrm T}_i \times dP_i = [\tilde{P}^{qk}_i, P^{sink}_i]^{\mathrm T} \times [dP^{qk}_i, \underbrace{dP^{sink}_i}_{zeros}]  \notag\\

&= {\tilde{P}^{qk}_i}^{\mathrm T} \times dP^{qk}_i \;+\; {P^{sink}_i}^{\mathrm T} \times \underbrace{dP^{sink}_i}_{zeros}\notag\\

&= {\tilde{P}^{qk}_i}^{\mathrm T} \times dP^{qk}_i = \tilde{O}^{\mathrm T}_i \times dO_i, \; i \in [1, sq] \notag\\

&\tilde{\Delta} = \mathrm{sum}_{hd}(\tilde{O} \;\odot\; dO) \notag\\ 

&where\; \tilde{O},dO \in \mathbb{R}^{nhq\times sq\times hd}, \; \tilde{\Delta} \in \mathbb{R}^{nhq\times sq} \notag \\

&\tilde{P},dP \in \mathbb{R}^{nhq\times sq\times (sk+s\_sink)} \notag

\end{aligned}
$$

- step2:(<em>recomputation</em>)

$$
\begin{aligned}
&S = Q \times K^{\mathrm T} \cdot scale\; + \; bias \notag \\ 

&\tilde{S}_i = [S_i, sink], \; i \in [1, sq] \notag \\

&\tilde{P}_i = \exp\big(\tilde{S}_i - \tilde{\mathrm{LSE}}_i), \; i \in [1, sq] \notag \\

& \tilde{P}_i = [\tilde{P}^{qk}_{i}, P^{sink}_{i}],\; i \in [1, sq] \notag \\

&where\; Q \in \mathbb{R}^{nhq \times sq\times  hd}, K \in \mathbb{R}^{nhk\times sk \times hd}, \;scale \in \mathbb{R}^{}\notag \\ 

&bias \in \mathbb{R}^{nhq\times sq\times sk}, \; \tilde{S},\tilde{P} \in \mathbb{R}^{nhq\times sq\times (sk+s\_sink)}, \;\tilde{\mathrm{LSE}} \in \mathbb{R}^{nhq\times sq} \notag \\

&\tilde{P}^{qk} \in \mathbb{R}^{nhq\times sq\times sk} ,\; P^{sink} \in \mathbb{R}^{nhq\times sq\times s\_sink} \notag \\

\end{aligned}
$$

- step3:

$$
\begin{aligned}
&dV = \tilde{P}^{\mathrm T} \times dO \notag \\ 

&dP = dO \times V^{\mathrm T} \notag \\

&where\; \tilde{P},dP \in \mathbb{R}^{nhq\times sq\times sk}\notag\\

&V,dV \in \mathbb{R}^{nhk\times sk\times hd} \notag \\

&dO \in \mathbb{R}^{nhq\times sq\times hd} \notag
\end{aligned}
$$

- step4:

$$
\begin{aligned}
&\tilde{dS}_i = [dS_{i}, dsink_{i}] = \tilde{P}_i \odot (dP_i - \tilde{\Delta}_i) \notag\\

&= [\tilde{P}^{qk}_{i}, P^{sink}_{i}] \odot ([dP^{qk}_i, \underbrace{dP^{sink}_i}_{zeros}]  - \tilde{\Delta}_i)\notag\\

&= [\tilde{P}^{qk}_{i} \odot (dP^{qk}_i- \tilde{\Delta}_i), P^{sink}_{i}\odot (\underbrace{dP^{sink}_i}_{zeros}- \tilde{\Delta}_i)] \notag \\ 

&= [\underbrace{\tilde{P}^{qk}_{i} \odot (dP^{qk}_i- \tilde{\Delta}_i)}_{dS_{i}}, \underbrace{P^{sink}_{i}\cdot - \tilde{\Delta}_i]}_{dsink_{i}}, \; i \in [1, sq] \notag \\ 

&dsink = \sum\limits_{i=1}^{sq} dsink_i = \sum\limits_{i=1}^{sq} \big(P^{sink}_{i}\cdot - \tilde{\Delta}_i\big) = {P^{sink}}^{\mathrm T} \times -\tilde{\Delta}\notag\\

&where\; \tilde{P},dP,\tilde{dS} \in \mathbb{R}^{nhq\times sq\times (sk+s\_sink)}\notag\\

&dS \in \mathbb{R}^{nhq\times sq\times sk},\;\tilde{\Delta} \in \mathbb{R}^{nhq\times sq}, \; dsink \in \mathbb{R}^{nhq\times s\_sink} \notag \\

& P^{sink} \in \mathbb{R}^{nhq\times sq\times s\_sink}\notag

\end{aligned}
$$

- step5: (<em>the same</em>)

- <b>dsink computation</b>: (<em>as another pre-processing of original ffa backward w/o sink tokens</em>)

$$
\begin{aligned}
&dsink = {P^{sink}}^{\mathrm T} \times -\tilde{\Delta} = \sum\limits_{i=1}^{sq} \big(P^{sink}_{i}\cdot - \tilde{\Delta}_i\big) \notag\\

&= -\sum\limits_{i=1}^{sq} \big(\exp(sink - \tilde{\mathrm{LSE}}_i)\cdot \tilde{\Delta}_i\big)\notag\\

&where\; sink,dsink \in \mathbb{R}^{nhq\times s\_sink},\;\tilde{\mathrm{LSE}}, \tilde{\Delta} \in \mathbb{R}^{nhq\times sq}\notag

\end{aligned}
$$


## Implementations

Based on the math derivation in the previous section, folding a learnable attention sink into the attention implementations in the Flash Attention style boils down to just two edits:

- For forward pass, we have nothing to change about the original implementation, but should apply an additional post-processing to correct the returned `out` and `lse` with `sink` tokens (<em>see the <b>sink correction</b> of the [FFA forward with sink tokens](#ffa-forward-with-sink-tokens)</em>).
- For backward pass, we have nothing to change about the original implementation, but should apply an additional pre-processing to compute the `dsink`, i.e. the gradient of `sink` (<em>see the <b>dsink computation</b> of the [FFA backward with sink tokens](#ffa-backward-with-sink-tokens)</em>).

Therefore, we share the following code snippets to present our implementations of the learnable attention sink mechanism: a naive PyTorch reference, Flex-Flash-Attention (<em>both internal and external to the kernels, which fit Flash Attention as well</em>), and the distributed implementation of MagiAttention.


### Torch Reference

- reference implementation w/o sink tokens:

```python
# apply `S = Q x K.T * scale + bias`
# where S.shape = [nhq, sq, sk]
s = q @ k.transpose(-2, -1) * softmax_scale + bias

# apply row-wise lse `LSE = logsumexp(S, dim=-1)`
# where LSE.shape = [nhq, sq, 1]
lse = s.logsumexp(dim=-1, keepdim=True)

# apply row-wise softmax `P = softmax(S, dim=-1)`
# where P.shape = [nhq, sq, sk]
p = softmax(s).to(q.dtype)

# apply `O = P x V`
# where O.shape = [nhq, sq, d]
out = p @ v

return out, lse
```

- reference implementation difference with sink tokens:

```diff
# apply `S = Q x K.T * scale + bias`
# where S.shape = [nhq, sq, sk]
s = q @ k.T * softmax_scale + bias

+ # apply `S = S.concat(sink, dim=-1)`
+ # where S.shape = [nhq, sq, sk + s_sink]
+ s = torch.concat([s, sink], dim=-1)

# apply row-wise lse `LSE = logsumexp(S, dim=-1)`
# where LSE.shape = [nhq, sq, 1]
lse = s.logsumexp(dim=-1, keepdim=True)

# apply row-wise softmax `P = softmax(S, dim=-1)`
- # where P.shape = [nhq, sq, sk]
+ # where P.shape = [nhq, sq, sk + s_sink]
p = softmax(s).to(q.dtype)

+ # apply `P = P.drop(sink, dim=-1)`
+ # where P.shape = [nhq, sq, sk]
+ p = p[..., : -sink.size(dim=-1)]

# apply `O = P x V`
# where O.shape = [nhq, sq, d]
out = p @ v

return out, lse
```

### FFA Impl

#### FFA Forward Impl

##### External Impl

- Use <b>sink correction</b> to correct `out`, `lse` after the ffa forward kernel returns, as an external post-processing kernel (<em>which is the way we extend the Flash Attention 2/3 forward with sink tokens, and see the [source code](https://github.com/SandAI-org/MagiAttention/blob/main/extensions/fa3_interface_with_sink.py) for more detals</em>):

```python
# given sink with shape: [s_sink, nhq]
# calculate and repeat to lse_sink with shape: [sq, nhq]
lse_sink = sink.logsumexp(dim=0, keepdim=True).repeat(sq, 1)

# given ffa returned lse with shape: [sq, nhq]
# correct lse with lse_sink
corrected_lse = log(exp(lse) + exp(lse_sink))

# given ffa returned out with shape: [sq, nhq, hd]
# correct out with corrected_lse and original lse
out *= exp(lse - corrected_lse)

return out, lse
```

##### Internal Impl

- Since FFA forward already has a post-processing kernel `FlashAttnFwdPostprocess` to zero-fill up the never-stored rows of `O`, indicated by "whether the corr. row of `lse` is still `-inf`", ...

- Then we can fuse the <b>sink correction</b> process into the `FlashAttnFwdPostprocess` kernel as follows (<em>see the [source code](https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/csrc/flexible_flash_attention/flash_fwd_postprocess_kernel.h) for more details</em>):
  
  - As for lse correction:
    - If the current row of `lse` is not `-inf`, then we update this row of `lse` with `lse_sink`.
    - Otherwise, the `lse` should also be filled up with `lse_sink`, instead of `-inf`.
  
  - As for out correction:
    - If the current row of `lse` is not `-inf`, then load the corr. row of `O`, rescale it and write it back.
    - Otherwise, the corr. row of `O` still needs to be filled up with `0`, so the same as before.


#### FFA Backward Impl

##### External Impl

- Use <b>dsink computation</b> to compute dsink before the ffa backward kernel launchs, as an external pre-processing kernel (<em>which is the way we extend the Flash Attention 2/3 backward with sink tokens, and see the [source code](https://github.com/SandAI-org/MagiAttention/blob/main/extensions/fa3_interface_with_sink.py) for more detals</em>):

```python
# calculate delta = (o * do).sum(dim=-1)
# where o.shape = [sq, nhq, d]
#       do.shape = [sq, nhq, d]
#       delta.shape = [nhq, sq, 1]
delta = reduce((o * do).to(lse.dtype), "sq hq d -> hq sq 1", "sum")

# calculate p_sink = exp(sink - lse)
# where sink.shape = [nhq, sq, s_sink]
#       lse.shape = [nhq, sq, 1]
#       p_sink.shape = [nhq, sq, s_sink]
p_sink = torch.exp(sink - lse)

# calculate dsink = p_sink.T x -delta
# where p_sink.shape = [nhq, sq, s_sink]
#       delta.shape = [nhq, sq, 1]
#       dsink.shape = [s_sink, nhq]
dsink = reduce(p_sink * -delta, "nhq sq s_sink -> s_sink nhq", "sum")

return dsink
```

##### Internal Impl

- Since FFA backward already has a pre-processing kernel `FlashAttnBwdPreprocess` to compute $\Delta$ (<em>in FA / FFA, we name it `dPsum`</em>), w.r.t. the step1 in the [FFA backward w/o sink tokens](#ffa-backward-wo-sink-tokens), ...

- The we can fuse the <b>dsink computation</b> process into the `FlashAttnBwdPreprocess` kernel as follows (<em>see the [source code](https://github.com/SandAI-org/MagiAttention/blob/main/magi_attention/csrc/flexible_flash_attention/flash_bwd_preprocess_kernel.h) for more details</em>):
  
  - As for `lse`, the same as before, each thread in one block loads one unique row of `lse`.
  
  - As for `p_sink`, the first `seqlen_sink` of threads in one block load the `sink` to shared memory, and each thread computes `p_sink = exp(sink - lse)` with its own unique row of `lse`, storing to shared memory as well.
  
  - As for `dPsum`, the same as before, each block loads a unique `kBlockM` rows of `O` and `dO`, applies `O * dO`, reduces across the head dimension to get the local block of `dPsum` in register files, and stores it to global memory.
  
  - As for `d_sink`, since it requires to be reduced across the whole `seqlen_q` dimension, the following steps are performed:
    - step1: each thread loads a unique row of `dPsum` from register files and the corr. row of `p_sink` from shared memory, and computes thread-partial `dsink = p_sink * -dPsum` for this row, and stores to  shared memory first (<em>since `p_sink` is not used afterwards, we can reuse its shared memory buffer to store `dsink`</em>).
    - step2: each block loads all the thread-partial `dsink` from shared memory, applies a `block-reduction` to get the block-reduced `dsink` for these `kBlockM` rows, and stores it to a temporary buffer in global memory.
    - step3: after a device-level memory fence, the last block who stores its block-reduced `dsink` loads all the block-reduced `dsink` back from the temporary buffer, applies another `block-reduction` to get the reduced `dsink` across the whole `seqlen_q` dimension, and finally stores it to global memory.


### MagiAttn Impl

#### MagiAttn Forward

- Since `sink` is replicated across cp ranks, we can easily apply attention sink by just passing `sink` into `_flex_flash_attn_forward`.
- However, the attention sink is supposed to be applied <u>once and only once</u> for the same query token, thus we can apply it at the host stage, i.e. each cp rank only applies to their own local `q`.
- Then, If the host stage is not skipped, just apply attention sink by passing `sink` into `_flex_flash_attn_forward`:

```diff
partial_out, partial_lse = _flex_flash_attn_forward(
    q=q,
    k=k,
    v=v,
+   # NOTE: sink token needs to be applied only once
+   # thus we only apply it at the host stage if not skipped
+   sink=sink if is_host_stage else None,
    out=out_acc,
    lse=lse_acc,
    **attn_arg.to_ffa_args(is_bwd=False),
    ...
)
```

- Otherwise, we should zero-initialize `local_out` as before, but initialize `local_lse` with `lse_sink`, instead of `-inf`

```diff
out = torch.zeros_like(
    q,
    dtype=torch.float32,
    device=q.device,
)

+ if sink is not None:
+   # in skipped host stage if sink is given,
+   # we directly use lse_sink to initialize lse
+   lse = calc_lse_sink(
+       sink=sink,
+       seqlen_lse=q.size(0),
+   )
+ else:
    lse = torch.full(
        (q.size(0), q.size(1)),
        fill_value=float("-inf"),
        dtype=float32,
        device=q.device,
    )
    
return out, lse
```

#### MagiAttn Backward

- The same to the forward, to form a complete, non-overlapping breakdown of `dsink` computation, we can compute partial `dsink` by just passing `sink` into `_flex_flash_attn_backward` only at the host stage, if not skipped.

```diff
(
    partial_dq,
    partial_dk,
    partial_dv,
+   partial_dsink,
) = _flex_flash_attn_backward(
    dout=do,
    q=q,
    k=k,
    v=v,
+   # NOTE: dsink should be computed only once
+   # thus we only compute it at the host stage if not skipped
+   sink=sink if is_host_stage else None,
    out=o,
    lse=lse,
    dq=dq_acc,
    dk=partial_dk,
    dv=partial_dv,
+   dsink=None,  # let kernel initialize dsink if required
    **attn_arg.to_ffa_args(is_bwd=True),
    ...
)
```

- And according to the formula of <b>dsink computation</b>, `dsink` is required to be sum-reduced along the `seqlen_q` dim, therefore, to get the reduced `dsink` for each cp rank, we have to additionally launch an all-reduce communication with `ReduceOp.Sum`, and wait it to complete before returning from the backward.
- However, the tricky thing is that during the acutal training scenario, the learnable `sink` tensor will be considered as a regular parameter in the model similar to `bias` in `nn.Linear` layer. So under some popular training frameworks, such as `Megatron-LM`, `FSDP`, the sum-reduction across cp ranks of the partial gradients of `sink` might be automatically applied within the whole `dp x cp` mesh.
- To avoid repeated reduction, we provide the environment variable `MAGI_ATTENTION_DSINK_ALL_REDUCE_OP` to let the user specify the all-reduce op for `dsink` within MagiAttention (<em>see the [docs](https://sandai-org.github.io/MagiAttention/docs/main/env_variables.html#for-correctness) for more details</em>). Defaults to `none` to <b>NOT</b> apply any reduction to `dsink` and let the framework handle it. Other options include `sum` and `avg` if needed.

```diff
+ # after the host stage when the partial dsink is ready
+ work = dist.all_reduce(
+    dsink,
+    op=dsink_reduce_op, # specified by `MAGI_ATTENTION_DSINK_ALL_REDUCE_OP`
+    group=self.cp_group_gc,
+    async_op=True,
+ )

...

+ # before returning from the backward
+ work.wait()

...

- return dq, dk, dv, ...
+ return dq, dk, dv, dsink, ...
```


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