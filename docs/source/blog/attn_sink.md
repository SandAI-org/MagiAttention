---
blogpost: true
date: Nov 17, 2025
author: Yunpeng Huang
location: China
category: MagiAttention
tags: Attention Sink, Flex-Flash-Attention, Flash-Attention
language: English
---

# Support Learnable Attention Sink

## Introduction

Large-Scaled Models assign significant attention to few tokens (<em>such as the intial tokens in the sequence</em>), even if they are not semantically important, which is known as <b>attention sink</b> {cite}`xiao2024efficientstreaminglanguagemodels`. Researchers attribute this interesting phenomenon to the nature of {math}`softmax`, which requires attention scores of each query token to always sum up to {math}`1` for all key tokens in the context, even when some query token does not strongly attend to any key token at all {cite}`gu2025attentionsinkemergeslanguage`. Therefore, during the training, we can deliberately add some <u><em>learnable sink tokens</em></u> to the key sequence for each query token to collect those unneeded attention scores to relax the <em>"sum-up-to-one"</em> constraint, as a learnable version of {math}`\textit{off-by-one}\space softmax` {cite}`miller2025attentionmisc`.

However, since sink tokens only affect the {math}`softmax` operation during the attention forward/backward passes w.r.t. the GPT-OSS implementation {cite}`openaiGPT-OSScode-misc`, <b>it is non-trivial to apply learnable attention sink with the (distributed) attention implementations in the style of <u>Flash Attention</u></b> {cite}`dao2022flashattention_attn_sink,dao2023flashattention_attn_sink,shah2024flashattention3fastaccurateattention_attn_sink`, particularly our own kernel implemenation of <u>Flex-Flash-Attention</u>, as well as the distributed implementation of <u>MagiAttention</u> {cite}`magiattention2025_attn_sink`.


## Overview

With the release of [MagiAttention-v1.0.5](https://github.com/SandAI-org/MagiAttention/releases/tag/v1.0.5), we have not only <b>supported the learnable attention sink mechanism</b> for our own kernel / distributed implementations of <u>Flex-Flash-Attention</u> / <u>MagiAttention</u> respectively, but also <b>provided the <em>plug-and-play</em> implementations</b> to integrate the original <u>Flash Attention</u> 2/3 interface {cite}`daoFlashAttnInterfaceMisc,daoFlashAttnInterfaceHopperMisc` with attention sink, as one of the [MagiAttention Extensions](https://github.com/SandAI-org/MagiAttention/tree/main/extensions#flashattention-with-attention-sink-).

In this blog, we will share our own methods about how to integrate the attention implementations in the Flash-Attention style with the learnable attention sink mechanism, including:

- the [User Interface](#user-interface) update for [Flex-Flash-Attention](#ffa-api), [MagiAttention](#magiattn-api) and [Flash-Attention Extension](#flash-attention-extension).
- the [Math Derivation](#math-derivation) of applying the attention sink in both [forward](#ffa-forward) and [backward](#ffa-backward) passes of Flex-Flash-Attention.
- the [Implementations](#implementations) of the (distributed) learnable attention sink mechanism for [Flex-Flash-Attention](#ffa-impl) and [MagiAttention](#magiattn-impl), as well as the naive [Torch Reference](#torch-reference).



## User Interface

Below, we show the minor update of the user interfaces to support learnable attention sink mechanism for original Flex-Flash-Attention, MagiAttention, as well as the Flash-Attention 2/3 as one of the [MagiAttention Extensions](https://github.com/SandAI-org/MagiAttention/tree/main/extensions#flashattention-with-attention-sink-).


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
        ... # other optional arguments
    ) -> tuple[torch.Tensor, AttnForwardMeta]:
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
        ... # other optional arguments
    ) -> tuple[torch.Tensor, AttnForwardMeta]:
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
        ... # other optional arguments
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
        ... # other optional arguments
    ):
        ...
    ```


## Math Derivation

Below, we provide the step-by-step math derivation of the original forward / backward passes for Flex-Flash-Attention (<em>the same as Flash-Attention</em>) w/o sink tokens, and then the differences when involving the learnable attention sink mechanism, serving as the guidence for our implementations in the next section.

:::{note}
1. To simplify the derivation, we drop the `batch` dimension and only keep the `num_heads` dimension to the leftmost acting as the implicit `batch` dimension.

2. To focus on the attention sink mechanism, we assume you're already familiar with Flash Attention and will skip over its finer details, like the <em>double-loop tiling</em> strategy and the derivation of <em>online softmax correction</em> based on `log-sum-exp` operations.

3. If you are new to Flash Attention or well-interested in the full original math derivation, <b>we highly recommend our another blog post: [Flash Attention 2 Math Derivation](./fa2_math_derivation.md)</b>.
:::


<b>Symbol Notation:</b>

| symbol                   | notation                                                                                        |
|--------------------------|-------------------------------------------------------------------------------------------------|
| {math}`\times`           | matrix multiplication                                                                           |
| {math}`\cdot`            | scalar multiplication                                                                           |
| {math}`\odot`            | element-wise multiplication (Hadamard product)                                                  |
| {math}`sq, sk, s\_sink`  | the sequence length of query tokens, key tokens, and attention sink tokens                      |
| {math}`nhq, nhk`         | the number of heads of query tokens and key tokens                                              |
| {math}`hd`               | the head dimension of query, key and value tokens                                               |
| {math}`X_i`              | the column vector made by the {math}`i`-th row of matrix {math}`X` along the sequence dimension |

### FFA Forward

#### FFA forward w/o sink tokens

- step1:

```{math}
:label: ffa_forward_wo_sink_step1

\begin{aligned}
S &= Q K^{\mathrm{T}} \cdot \mathrm{scale} + \mathrm{bias} \\
\text{where } & Q \in \mathbb{R}^{n_{hq} \times s_q \times h_d},\; K \in \mathbb{R}^{n_{hk} \times s_k \times h_d}, \\
& \mathrm{scale} \in \mathbb{R},\; \mathrm{bias} \in \mathbb{R}^{n_{hq} \times s_q \times s_k},\; S \in \mathbb{R}^{n_{hq} \times s_q \times s_k}
\end{aligned}
```

- step2:

```{math}
:label: ffa_forward_wo_sink_step2

\begin{aligned}
\mathrm{softmax}_{\mathrm{row}}(X_i) &= \frac{\exp(X_i - M_i)}{L_i}, \quad i \in [1, s_q] \\
\text{where } M_i &= \mathrm{rowmax}(X_i), \quad L_i = \mathrm{rowsum}(\exp(X_i - M_i))
\end{aligned}
```
```{math}
\begin{aligned}
&P = \mathrm{softmax}_{row}(S) \notag \\
&where\; S, P \in \mathbb{R}^{nhq\times sq\times sk} \notag
\end{aligned}
```

- step3:

```{math}
:label: ffa_forward_wo_sink_step3

\begin{aligned}
O &= P \times V, \quad \mathrm{LSE}_i = \log(L_i) + M_i, \quad i \in [1, s_q] \\
\text{where } & P \in \mathbb{R}^{n_{hq} \times s_q \times s_k}, \quad V \in \mathbb{R}^{n_{hk} \times s_k \times h_d}, \\
& O \in \mathbb{R}^{n_{hq} \times s_q \times h_d}, \quad \mathrm{LSE} \in \mathbb{R}^{n_{hq} \times s_q}
\end{aligned}
```

#### FFA forward with sink tokens

- step1: <em>the same with {eq}`ffa_forward_wo_sink_step1`</em>

- step2:

```{math}
:label: ffa_forward_w_sink_step2

\begin{aligned}
\tilde{P} &= \mathrm{softmax}_{\mathrm{row}}(\tilde{S}), \quad \tilde{S}_i = [S_i, \mathrm{sink}], \quad i \in [1, s_q] \\
\text{where } & \tilde{S}, \tilde{P} \in \mathbb{R}^{n_{hq} \times s_q \times (s_k + s_{\mathrm{sink}})}, \quad \mathrm{sink} \in \mathbb{R}^{n_{hq} \times s_{\mathrm{sink}}}
\end{aligned}
```

```{math}
\begin{aligned}
\tilde{P}_i &= [\tilde{P}^{\mathrm{qk}}_{i}, P^{\mathrm{sink}}_{i}], \quad i \in [1, s_q] \\
\text{where } & \tilde{P}^{\mathrm{qk}} \in \mathbb{R}^{n_{hq} \times s_q \times s_k}, \\
& P^{\mathrm{sink}} \in \mathbb{R}^{n_{hq} \times s_q \times s_{\mathrm{sink}}}
\end{aligned}
```

- step3:

```{math}
:label: ffa_forward_w_sink_step3
\begin{aligned}
&\tilde{O} = \tilde{P}^{qk} \times V, \;\tilde{\mathrm{LSE}}_i = \log(\tilde{L}_i) + M_i, \; i \in [1, sq] \\
&\tilde{L}_i = L_i + \sum\limits_{j=1}^{s_{\mathrm{sink}}}\mathrm{exp}(sink_j - M_i), \; i \in [1, sq] \\
&\tilde{P}^{qk}_i = P^{qk}_i \times \cfrac{L_i}{\tilde{L}_i}, \; i \in [1, sq] \\
&\text{where } P^{qk},\tilde{P}^{qk} \in \mathbb{R}^{nhq\times sq\times sk}, \; V \in \mathbb{R}^{nhk\times sk\times hd}, \\
&\tilde{O} \in \mathbb{R}^{nhq\times sq\times hd}, \;\tilde{\mathrm{LSE}} \in \mathbb{R}^{nhq\times sq}
\end{aligned}
```

- <b>sink correction</b>: <em>as a post-processing of original ffa forward w/o sink tokens</em>

```{math}
:label: ffa_forward_w_sink_sink_correction
\begin{aligned}
&\mathrm{LSE}^{sink} = \log\big(\sum\limits_{j=1}^{s_{\mathrm{sink}}}\mathrm{exp}(sink_j)\big) \\
&\tilde{\mathrm{LSE}}_i = \log\big(\exp(\mathrm{LSE}_i) + \exp(\mathrm{LSE}^{sink})\big), \; i \in [1, sq] \\
&\tilde{O} = O \cdot \exp\big(\mathrm{LSE} - \tilde{\mathrm{LSE}}\big) \\
&\text{where } sink \in \mathbb{R}^{nhq\times s_{\mathrm{sink}}},\;\mathrm{LSE}^{sink} \in \mathbb{R}^{nhq} \\
&\mathrm{LSE},\tilde{\mathrm{LSE}} \in \mathbb{R}^{nhq\times sq}, \;O,\tilde{O}\in \mathbb{R}^{nhq\times sq\times hd}
\end{aligned}
```

### FFA Backward

#### FFA backward w/o sink tokens

- step1: <em>as a pre-processing</em>

```{math}
:label: ffa_backward_wo_sink_step1

\begin{aligned}
&\Delta_i = P^{\mathrm T}_i \times dP_i = O^{\mathrm T}_i \times dO_i,\quad i \in [1, s_q] \\[4pt]
&\Delta = \mathrm{sum}_{hd}(O \;\odot\; dO)
\\[4pt]
&\text{where } O,dO \in \mathbb{R}^{n_{hq}\times s_q\times h_d},\; \Delta \in \mathbb{R}^{n_{hq}\times s_q}
\end{aligned}
```

- step2: <em>recomputation</em>

```{math}
:label: ffa_backward_wo_sink_step2

\begin{aligned}
&S = Q \times K^{\mathrm T} \cdot \mathrm{scale} \; + \; \mathrm{bias} \\[4pt]
&P_i = \exp\big(S_i - \mathrm{LSE}_i\big), \quad i \in [1, s_q] \\[4pt]
&\text{where } Q \in \mathbb{R}^{n_{hq}\times s_q\times h_d},\; K \in \mathbb{R}^{n_{hk}\times s_k\times h_d}, \\[2pt]
&\mathrm{scale} \in \mathbb{R},\; \mathrm{bias} \in \mathbb{R}^{n_{hq}\times s_q\times s_k}, \\[2pt]
&S,P \in \mathbb{R}^{n_{hq}\times s_q\times s_k},\; \mathrm{LSE} \in \mathbb{R}^{n_{hq}\times s_q}
\end{aligned}
```

- step3:

```{math}
:label: ffa_backward_wo_sink_step3

\begin{aligned}
&dV = P^{\mathrm T} \times dO \\[4pt]
&dP = dO \times V^{\mathrm T} \\[4pt]
&\text{where } P,dP \in \mathbb{R}^{n_{hq}\times s_q\times s_k},\; V,dV \in \mathbb{R}^{n_{hk}\times s_k\times h_d},\\[2pt]
&dO \in \mathbb{R}^{n_{hq}\times s_q\times h_d}
\end{aligned}
```

- step4:

```{math}
:label: ffa_backward_wo_sink_step4

\begin{aligned}
&dS_i = P_i \odot (dP_i - \Delta_i), \quad i \in [1, s_q] \\[4pt]
&\text{where } P,dP \in \mathbb{R}^{n_{hq}\times s_q\times s_k},\; dS \in \mathbb{R}^{n_{hq}\times s_q\times s_k},\\[2pt]
&\Delta \in \mathbb{R}^{n_{hq}\times s_q}
\end{aligned}
```

- step5:

```{math}
:label: ffa_backward_wo_sink_step5

\begin{aligned}
&\hat{dS} = dS \cdot \mathrm{scale} \\[4pt]
&dQ = \hat{dS} \times K \\[4pt]
&dK = \hat{dS}^{\mathrm T} \times Q \\[4pt]
&\text{where } dS,\hat{dS} \in \mathbb{R}^{n_{hq}\times s_q\times s_k},\; \mathrm{scale}\in\mathbb{R},\\[2pt]
&Q,dQ \in \mathbb{R}^{n_{hq}\times s_q\times h_d},\; K,dK \in \mathbb{R}^{n_{hk}\times s_k\times h_d}
\end{aligned}
```


#### FFA backward with sink tokens

- step1: <em>as a pre-processing as well</em>

```{math}
:label: ffa_backward_w_sink_step1

\begin{aligned}
&\tilde{\Delta}_i = \tilde{P}_i^{\mathrm T} \times dP_i = [\tilde{P}^{qk}_i, P^{sink}_i]^{\mathrm T} \times [dP^{qk}_i, dP^{sink}_i] \\
&\quad\;=\; {\tilde{P}^{qk}_i}^{\mathrm T} \times dP^{qk}_i \;+\; {P^{sink}_i}^{\mathrm T} \times dP^{sink}_i \\
&\quad\;=\; {\tilde{P}^{qk}_i}^{\mathrm T} \times dP^{qk}_i \;=\; \tilde{O}_i^{\mathrm T} \times dO_i,\quad i\in[1,s_q] \\
&\tilde{\Delta} = \mathrm{sum}_{hd}(\tilde{O}\odot dO) \\
&\text{where }\tilde{O},dO\in\mathbb{R}^{n_{hq}\times s_q\times h_d},\; \tilde{\Delta}\in\mathbb{R}^{n_{hq}\times s_q}
\end{aligned}
```

- step2: <em>recomputation</em>

```{math}
:label: ffa_backward_w_sink_step2

\begin{aligned}
&S = QK^{\mathrm T}\cdot\mathrm{scale} + \mathrm{bias} \\
&\tilde{S}_i = [S_i,\,\mathrm{sink}],\quad i\in[1,s_q] \\
&\tilde{P}_i = \exp\big(\tilde{S}_i - \tilde{\mathrm{LSE}}_i\big),\quad i\in[1,s_q] \\
&\tilde{P}_i = [\tilde{P}^{qk}_i,\,P^{sink}_i] \\
&\text{where } \tilde{S},\tilde{P}\in\mathbb{R}^{n_{hq}\times s_q\times (s_k+s_{\mathrm{sink}})},\; \tilde{\mathrm{LSE}}\in\mathbb{R}^{n_{hq}\times s_q}
\end{aligned}
```

- step3:

```{math}
:label: ffa_backward_w_sink_step3

\begin{aligned}
&dV = {\tilde{P}}^{\mathrm T}\times dO \\
&dP = dO\times V^{\mathrm T} \\
&\text{where } \tilde{P},dP\in\mathbb{R}^{n_{hq}\times s_q\times (s_k+s_{\mathrm{sink}})},\; dV\in\mathbb{R}^{n_{hk}\times s_k\times h_d}
\end{aligned}
```

- step4:

```{math}
:label: ffa_backward_w_sink_step4

\begin{aligned}
&\tilde{dS}_i = \tilde{P}_i \odot (dP_i - \tilde{\Delta}_i) = [dS_i,\,dsink_i],\quad i\in[1,s_q] \\
&dS_i = \tilde{P}^{qk}_i \odot (dP^{qk}_i - \tilde{\Delta}_i) \\
&dsink_i = P^{sink}_i \odot (dP^{sink}_i - \tilde{\Delta}_i) = -\,P^{sink}_i\odot\tilde{\Delta}_i \quad(\text{since } dP^{sink}_i=0)\\
&dsink = \sum_{i=1}^{s_q} dsink_i = {P^{sink}}^{\mathrm T}\times(-\tilde{\Delta}) \\
&\text{where } dS\in\mathbb{R}^{n_{hq}\times s_q\times s_k},\; dsink\in\mathbb{R}^{n_{hq}\times s_{\mathrm{sink}}}
\end{aligned}
```

- step5: <em>the same with {eq}`ffa_backward_wo_sink_step5`</em>

- dsink computation: <em>as another pre-processing of original ffa backward w/o sink tokens</em>

```{math}
:label: ffa_backward_w_sink_dsink_comp

\begin{aligned}
&dsink = {P^{sink}}^{\mathrm T}\times(-\tilde{\Delta})
= -\sum_{i=1}^{s_q}\big(\exp(\mathrm{sink}-\tilde{\mathrm{LSE}}_i)\cdot\tilde{\Delta}_i\big) \\
&\text{where } \mathrm{sink},dsink\in\mathbb{R}^{n_{hq}\times s_{\mathrm{sink}}},\; \tilde{\mathrm{LSE}},\tilde{\Delta}\in\mathbb{R}^{n_{hq}\times s_q}
\end{aligned}
```


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

- Use <b>sink correction</b> to correct `out`, `lse` after the ffa forward kernel returns, as an external post-processing kernel (<em>which is the way we extend the Flash Attention 2/3 forward with sink tokens, and see the [source code](https://github.com/SandAI-org/MagiAttention/blob/main/extensions/magi_attn_extensions/fa3_interface_with_sink.py) for more detals</em>):

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

- Use <b>dsink computation</b> to compute dsink before the ffa backward kernel launchs, as an external pre-processing kernel (<em>which is the way we extend the Flash Attention 2/3 backward with sink tokens, and see the [source code](https://github.com/SandAI-org/MagiAttention/blob/main/extensions/magi_attn_extensions/fa3_interface_with_sink.py) for more detals</em>):

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

- Since FFA backward already has a pre-processing kernel `FlashAttnBwdPreprocess` to compute {math}`\Delta` (<em>in FA / FFA, we name it `dPsum`</em>), w.r.t. the step1 in the [FFA backward w/o sink tokens](#ffa-backward-wo-sink-tokens), ...

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

## References

```{bibliography} refs/attn_sink.bib
```
