---
layout: distill
permalink: /ffa_with_sink
title: FFA with Attention Sink
description: Integrating Flex-Flash-Attention with Attention Sink
date: 2025-11-15
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

bibliography: magiattn.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: User Interface
    subsections:
      - name: Flex-Flash-Attention
      - name: MagiAttention
      - name: Flash-Attention Extension
  - name: Math Derivation
    subsections:
      - name: FFA Forward
      - name: FFA Backward
  - name: Implementations
    subsections:
      - name: Torch Reference
      - name: FFA
      - name: MagiAttention
  - name: References
  - name: Citation

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


## User Interface

### FFA

- Just add an optional tensor sink to the argument list of flex_flash_attn_func
- And when and only when sink tensor is given, FFA will apply attention sink for forward and compute dsink for backward, otherwise, attention sink is skipped and dsink is also returned as None
- dtype: float32
- shape: [seqlen_sink, num_heads_q], where seqlen_sink in [1, 8]

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


### MagiAttn

- Just add an optional global replicated tensor sink to the argument list of calc_attn, dist_attn_func
- And when and only when sink tensor is given, dist_attn_func will apply attention sink for forward once and only once for the same q row and compute local partial dsink and all-reduce across cp ranks, otherwise, attention sink is skipped and dsink is also returned as None
- dtype: float32
- shape: [seqlen_sink, num_heads_q], where seqlen_sink in [1, 8]
- parallel style: Replicate

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


### Flash Attention

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
&O = P \times V, \;\mathrm{LSE}_i = L_i + M_i, \; i \in [1, sq]\notag \\ 

&where\; P \in \mathbb{R}^{nhq\times sq\times sk}, \; V \in \mathbb{R}^{nhk\times sk\times hd}, \notag \\

& O \in \mathbb{R}^{nhq\times sq\times hd}, \;\mathrm{LSE} \in \mathbb{R}^{nhq\times sq}\notag

\end{aligned}
$$

#### FFA forward with sink tokens

- step1: (the same)

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
&\tilde{O} = \tilde{P}^{qk} \times V, \;\tilde{\mathrm{LSE}}_i = \tilde{L}_i + M_i, \; i \in [1, sq]\notag \\ 

& \tilde{L}_i = L_i + \sum\limits_{j=1}^{s\_sink}\mathrm{exp}(sink_j), \; i \in [1, sq]\notag \\ 

& \tilde{P}^{qk}_i = P^{qk}_i \times \cfrac{L_i}{\tilde{L}_i}, \; i \in [1, sq]\notag \\

&where\; P^{qk},\tilde{P}^{qk} \in \mathbb{R}^{nhq\times sq\times sk}, \; V \in \mathbb{R}^{nhk\times sk\times hd}, \notag \\

& \tilde{O} \in \mathbb{R}^{nhq\times sq\times hd}, \;\tilde{\mathrm{LSE}} \in \mathbb{R}^{nhq\times sq}\notag

\end{aligned}
$$

- sink correction: (as a post-processing of original ffa forward w/o sink tokens)

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

- step1: (as a pre-processing)

$$
\begin{aligned}
&\Delta_i = P^{\mathrm T}_i \times dP_i = O^{\mathrm T}_i \times dO_i, \; i \in [1, sq] \notag\\

&\Delta = \mathrm{sum}_{hd}(O \;\odot\; dO) \notag\\ 

&where\; O,dO \in \mathbb{R}^{nhq\times sq\times hd}, \; \Delta \in \mathbb{R}^{nhq\times sq} \notag
\end{aligned}
$$

- step2:(recomputation)

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
&\hat{dS} = dS \cdot scale + bias \notag \\ 

&dQ = \hat{dS} \times K \notag \\ 

&dK = \hat{dS}^{\mathrm T} \times Q \notag \\ 

&where\; dS,\hat{dS},bias \in \mathbb{R}^{nhq\times sq\times sk}, \;scale \in \mathbb{R}^{}\notag \\

&Q,dQ \in \mathbb{R}^{nhq\times sq\times hd}, \; K,dK \in \mathbb{R}^{nhk\times sk\times hd} \notag
\end{aligned}
$$

#### FFA backward with sink tokens

- step1: (as a pre-processing)

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

- step2:(recomputation)

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

- step5: (the same)

- dsink computation: (as a pre-processing of original ffa backward w/o sink tokens)

$$
\begin{aligned}
&dsink = {P^{sink}}^{\mathrm T} \times -\tilde{\Delta} = \sum\limits_{i=1}^{sq} \big(P^{sink}_{i}\cdot - \tilde{\Delta}_i\big) \notag\\

&= -\sum\limits_{i=1}^{sq} \big(\exp(sink - \tilde{\mathrm{LSE}}_i)\cdot \tilde{\Delta}_i\big)\notag\\

&where\; sink,dsink \in \mathbb{R}^{nhq\times s\_sink},\;\tilde{\mathrm{LSE}}, \tilde{\Delta} \in \mathbb{R}^{nhq\times sq}\notag

\end{aligned}
$$


## Implementations

### Torch Reference

- reference implementation w/o sink tokens

```python
# apply `S = Q x K.T * scale + bias`
# where S.shape = [nhq, sq, sk]
s = q @ k.T * softmax_scale + bias

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

- reference implementation difference with sink tokens

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

### FFA Forward

#### Outside

- use sink correction to correct out, lse when ffa forward kernel returns, as a post-processing kernel outside

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

#### Inside
- Since FFA forward already has a post-processing kernel FastZeroFillKernel to zero-fill up the never-stored rows of O, indicated by "whether the corr. row of lse is still -inf", ...

- We can integrate the sink correction process into FastZeroFillKernel as follows:
  - As for lse correction, if the current row of lse is not -inf, then we update this lse with lse_sink, otherwise, the lse should also be filled up with lse_sink, instead of -inf

  - As for out correction, if the current row of lse is not -inf, then load the corr. row of O, rescale it and write it back, otherwise, the corr. row of O still needs to be filled up with 0, so nothing to do


### FFA Backward

#### Outside

- use dsink computation to compute dsink before ffa backward kernel launchs, as a pre-processing kernel outside

```python
# calculate delta = (o * do).sum(dim=-1)
# where o.shape = [sq, nhq, d]
#       do.shape = [sq, nhq, d]
#       delta.shape = [nhq, sq, 1]
delta = reduce((o * do).to(lse.dtype), "sq hq d -> hq sq 1", "sum")

# calculat p_sink = exp(sink - lse)
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

#### Inside

- Since FFA backward already has a pre-processing kernel FlashAttnBwdPreprocess to compute Delta w.r.t. step1 (in ffa, we name it dPsum), and rescale lse from base e to 2(in ffa, we name it LSE_log2), as follows:
  - The kernel is launched with grid size: (seqlen_q//kBlockM, nhq), and block size: (MaxThreadsPerBlock,), where kBlockM <= MaxThreadsPerBlock

  - And each block is theoretically responsible for one q head, and a kBlockM chunk of seqlen_q

  - So for lse, each thread in one block can load one unique lse, rescale it and write it to LSE_log2


  - As for dPsum, as O and dO have an extra head dim to reduce, it uses tiled copy to load tiled O and dO of shape (c3, kBlockM / c1, kHeadDim / c2), apply O * dO, and reduce it across the head dim to get tiled and reduced dPsum


  - For now, each thread has a tile of reduced  dPsum, so we should let only several of them write the corr. row of dPsum back to global memory

- We can integrate the dsink computation process into FlashAttnBwdPreprocess in several methods as follows:
  - Since each thread in one block loads a unique row of lse for a certain q head, it's supposed to load all the sink tokens of size s_sink for the same q head, to compute p_sink = exp(sink - lse), where the repeated load in the same block can be optimized by loading sink tokens to shared memory first

  - Then, since the tiled rows of dPsum might not be aligned with the same row of p_sink, lse, for now we just wait dPsum to write back first as shown above, block-level thread fence, and then reload the corr. row of dPsum to compute partial dsink = p_sink * -dPsum, and reduce it back to global memory

  - Notably, the reduction of partial dsink is a cross-block store since each block holds only a chunk of seqlen_q, so there're several ways to reduce for non-deterministic mode:
    - way1: just use atomicAdd to apply the reduction, but introducing severe thread-level data race
    - way2: first apply block-reduce to get block-partial dsink in thread0 for each block, and then atomicAdd, which introduces block-level data race
  - As for deterministic mode, there're several ways to reduce as well:
    - way1: first apply block-reduce to get block-partial dsink in thread0 for each block, and then store it to a workspace, device-level thread fence, and then block0 loads all the block-partial dsink, reduces, and writes the all-reduced dsink back
    - way2: first apply block-reduce to get block-partial dsink in thread0 for each block, and then each thread0 in each block waits for the block-id counter in self-rotated style, to decide the timing to reduce for each block-reduced dsink
  - To avoid the block-level thread fence for dPsum, each thread can first load its p_sink to shared memory, and then let those threads who write back dPsum to compute partial d_sink and store to the same shared memory as p_sink


### MagiAttn Forward

- Since sink is replicated across cp ranks, we can easily apply attention sink by just passing sink into _flex_flash_attn_forward
- However, the attention sink can be applied once and only once for the same q row, thus we apply it at the host stage, i.e. only for local attention.
- Then, there are two cases:
  - case1: if the host stage is not skipped, just apply attention sink by passing sink into _flex_flash_attn_forward

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

  - case2: otherwise, we should zero-initialize local_out as usual, but use lse_sink to initialize local_lse, instead of using -inf

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

### MagiAttn Backward

- Since sink is replicated while q is sharded across cp ranks, and dsink needs to be reduced along the seqlen_q dim, ...

- The same as the forward, we can compute partial dsink by just passing sink into _flex_flash_attn_backward once and only once at the host stage

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

- And then, we need to launch an async all-reduce kernel with ReduceOp.Sum to get the identical reduced dsink for each cp rank, and wait for it to be finished before returning from backward.

```diff
+ work = dist.all_reduce(
+    dsink,
+    group=self.cp_group_gc,
+    async_op=True,
+ )

...

+ work.wait()

+ return dsink
```


## References

### Research
- StreamingLLM:
  - paper: Efficient Streaming Language Models with Attention Sinks
  - github: https://github.com/mit-han-lab/streaming-llm
- Empirical View:
  - paper: When Attention Sink Emerges in Language Models: An Empirical View
  - github: https://github.com/sail-sg/Attention-Sink
- Attention Is Off By One:
  - blog: https://www.evanmiller.org/attention-is-off-by-one.html
  - github: https://github.com/kyegomez/AttentionIsOFFByOne
- Others:
  - See What You Are Told: Visual Attention Sink in Large Multimodal Models
  - Softpick: No Attention Sink, No Massive Activations with Rectified Softmax

### Implementation
- GPT-OSS:
  - github: https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L153
- Flash-Attention 4:
  - github: https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute


## Citation

If you use MagiAttention in your research, please cite:

```bibtex
@misc{magiattention2025,
  title={MagiAttention: A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training},
  author={Zewei, Tao and Yunpeng, Huang},
  year={2025},
  howpublished={\url{https://github.com/SandAI-org/MagiAttention/}},
}
```