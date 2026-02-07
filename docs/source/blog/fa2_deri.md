---
blogpost: true
date: Dec 22, 2025
author: Yunpeng Huang
location: China
category: MagiAttention
tags: Flash-Attention, Flex-Flash-Attention
language: English
---

# Flash Attention 2 Math Derivation

This blog post is a detailed math derivation of well-known **Flash Attention 2 (FA2)**, a memory-efficient, highly optimized and <em>de facto</em> kernel implementation {cite}`dao2022flashattention_fa2_deri, dao2023flashattention_fa2_deri, shah2024flashattention3fastaccurateattention_fa2_deri` of <em>scaled dot-product attention</em> operation introduced by Transformer {cite}`vaswani2023attentionneed_fa2_deri`, which is re-implemented and further extended in **Flex-Flash-Attention** kernels of MagiAttention {cite}`magiattention2025_fa2_deri`.

:::{note}
1. We omit specific softmax strategies, e.g. `softmax_scale`, `softcap`, `attention_sink`, for simplicity.
2. We omit any batch dimensions, e.g. `batch_size`, `num_heads`, but keep only the `seqlen` dimension and the `head` dimension for simplicity.
:::


## Forward

### Standard Attention Forward

```{math}
:label: std_attn_forward

\begin{cases}
\begin{align} 
&S = \mathrm{mask}(QK^{\mathrm{T}} + bias)  \in \mathbb{R}^{N\times N} \\
&P = \mathrm{softmax}_{row\text{-}wise}(S) = \mathrm{diag}(l)^{-1}A  \in \mathbb{R}^{N\times N},\\
&\quad \text{where}\; l = \mathrm{rowsum}(A) \in \mathbb{R}^{N}, \space A = \exp{(S  - \mathrm{rowmax}(S))} \in \mathbb{R}^{N\times N} \\ 
&O = PV \in \mathbb{R}^{N\times d}
\end{align}
\end{cases}
```

```{math}
given\quad Q,K,V \in \mathbb{R}^{N\times d}, \space bias \in \mathbb{R}^{N\times N}
```

### Flash Attention Forward

#### Step1. Basic Row Decomposition

```{math}
:label: fa2_forward_step1_basic_row_decomp

\begin{cases}
\begin{aligned}
&S = \left[ S_1\quad S_2 \right] \in \mathbb{R}^{B_q\times 2B_k},\\
&\quad\text{where}\; S_i = \mathrm{mask}(QK_i^{\mathrm{T}} + \text{bias}_{i}) \in \mathbb{R}^{B_q\times B_k},\\
&\quad Q \in \mathbb{R}^{B_q\times d},\ K_i \in \mathbb{R}^{B_k\times d},\ i \in \{1,2\} \\
&m = \max\left( \mathrm{rowmax}(S_1), \mathrm{rowmax}(S_2) \right) \in \mathbb{R}^{B_q} \\
&A = \left[ A_1\quad A_2 \right] \in \mathbb{R}^{B_q\times 2B_k},\\
&\quad\text{where}\; A_i = \exp(S_i - m) \in \mathbb{R}^{B_q\times B_k},\ i \in \{1,2\} \\
&l = \mathrm{rowsum}(A_1) + \mathrm{rowsum}(A_2) \in \mathbb{R}^{B_q} \\
&P = \left[ P_1\quad P_2 \right] = \mathrm{diag}(l)^{-1} \left[ A_1\quad A_2 \right] \in \mathbb{R}^{B_q\times 2B_k} \\
&O = \left[ P_1\quad P_2 \right] \left[
\begin{matrix}
V_1 \\ 
V_2 
\end{matrix}
\right] = \mathrm{diag}(l)^{-1} \left( A_1V_1 + A_2V_2 \right) \in \mathbb{R}^{B_q\times d}
\end{aligned}
\end{cases}
```

#### Step2. Online Softmax Correction

```{math}
:label: fa2_forward_step2_online_softmax_correction_base

\text{base}: 
\begin{cases}
\begin{align} 
&m_1 = \mathrm{rowmax}(S_1) \in \mathbb{R}^{B_q}\notag\\
&A_1 = \exp(S_1 - m_1) \in \mathbb{R}^{B_q\times B_k}\notag\\
&l_1 = \mathrm{rowsum}(A_1)\in \mathbb{R}^{B_q}\notag\\
&P_1 = \mathrm{diag}(l_1)^{-1}A_1\in \mathbb{R}^{B_q\times B_k}\notag\\
&O_1 = P_1V_1\in \mathbb{R}^{B_q\times d}\notag
\end{align}\\
\end{cases}
```

```{math}
:label: fa2_forward_step2_online_softmax_correction_update

\text{update}: 
\begin{cases}
\begin{align} 
&m_2 = \max(m_1, \mathrm{rowmax}(S_2)) \in \mathbb{R}^{B_q}\\
&A_2 = \exp(S_2 - m_2) \in \mathbb{R}^{B_q\times B_k}\notag\\
&l_2 = \delta_m l_1 + \mathrm{rowsum}(A_2)\in \mathbb{R}^{B_q}\\
&P_2 = \mathrm{diag}(l_2)^{-1}A_2\in \mathbb{R}^{B_q\times B_k}\notag\\
&O_2 = \mathrm{diag}(l_1/l_2)^{-1}\delta_m O_1 + P_2V_2 \in \mathbb{R}^{B_q\times d} \notag
\end{align}
\end{cases}
```
    
```{math}
\begin{align}
&\text{where}\; \delta_m := \exp(m_1 -m_2)
\end{align}
```

#### Step3. Double-Loop Tiling

* the outer loop runs through {math}`i := 1 \rightarrow N_q` for each block of {math}`Q_i` to compute {math}`O_i`,  where {math}`N_q = \lceil\frac{N}{B_q}\rceil`, and for each {math}`i`-th outer iteration:

```{math}
:label: fa2_forward_step3_double_loop_tiling_outer

\begin{cases}
\begin{align} 
&\text{load}\space  Q_i \in \mathbb{R}^{B_q\times d}\space  \text{from HBM to SRAM}\notag\\
&\text{initialize}\space \tilde{O_{i}}^{(0)} = 0_{ B_q\times d },\space  l_i^{(0)} = 0_{B_q} \in \mathbb{R}^{B_q},\space  m_i^{(0)} = -\infty_{B_q} \in \mathbb{R}^{B_q}  \notag\\
\\
&\text{loop over}\space  j := 1 \rightarrow N_k\space \text{, and for each}\space j \text{-th inner iteration:} \notag\\
&\quad\text{compute}\space  O_i = \mathrm{diag}(l_{i}^{(N_k)})^{-1} \tilde{O_i}^{(N_k)}\in \mathbb{R}^{B_q\times d}\\
&\quad\quad\text{and write it to HBM to return as output} \notag\\
&\quad\text{compute}\space  \mathrm{LSE_i} = m_i^{(N_k)} + \log(l_i^{(N_k)})\in \mathbb{R}^{B_q}\\
&\quad\quad\text{and write it to HBM to save for backward} \notag
\end{align}
\end{cases}
```

```{math}
\begin{align}
&\text{where}\; \text{LSE}( \mathbf{x}) := \log\left(\sum\limits_{i=1}^n \exp(x_i)\right) = \max( \mathbf x) + \text{LSE}( \mathbf{x}-\max( \mathbf x)),\space   \mathbf x \in \mathbb{R}^{n},\\
&\quad\text{and}\space \tilde{O_i} \space\text{is the un-normalized} \space O_i, \space\text{i.e.}\space O_i = \mathrm{diag}(l_{i})^{-1}\tilde{O_i}
\end{align}
```

* in which each inner loop goes across {math}`j := 1 \rightarrow N_k` for each block of {math}`K_j,V_j` to update {math}`\tilde{O_i}^{(j)}, l_i^{(j)}, m_i^{(j)}`, where {math}`N_k = \lceil\frac{N}{B_k}\rceil`, and for each {math}`j`-th inner iteration:

```{math}
:label: fa2_forward_step3_double_loop_tiling_inner

\begin{cases}
\begin{align} 
&\text{load}\space  K_j, V_j \in \mathbb{R}^{B_k\times d}\space  \text{from HBM to SRAM} \notag\\
&\text{compute}\space  S_{i}^{(j)} = \text{mask}(Q_iK_j^{\mathrm T} + bias_{(i,j)}) \in \mathbb{R}^{B_q\times B_k} \notag\\
&\text{update}\space  m_i^{(j)} = \max\big(m_i^{(j-1)}, \mathrm{rowmax}(S_{i}^{(j)})\big) \in \mathbb{R}^{B_q} \notag\\
&\text{compute}\space A_i^{(j)} = \exp(S_i^{(j)} - m_i^{(j)}) \in \mathbb{R}^{B_q\times B_k} \notag\\
&\text{update}\space  l_i^{(j)} = \delta_{m_i^{(j)}}l_i^{(j-1)} + \mathrm{rowsum}(A_i^{(j)})\in \mathbb{R}^{B_q}  \notag\\
&\text{update}\space  \tilde{O_i}^{(j)} = \mathrm{diag}(\delta_{m_i^{(j)}})^{-1}\tilde{O_i}^{(j-1)} + A_i^{(j)}V_j\in \mathbb{R}^{B_q\times d} \notag
\end{align}
\end{cases}
```
    
```{math}
\begin{align}
&\text{where}\; \delta_{m_i^{(j)}} := \exp(m_i^{(j-1)} -m_i^{(j)})
\end{align}
```

## Backward

### Standard Attention Backward

```{math}
:label: std_attn_backward

\begin{cases}
\begin{align}
&\mathrm{d}{V} = P^{\mathrm T} \mathrm{d}{O} \in \mathbb{R}^{N\times d}, \quad \mathrm{d}{P} = \mathrm{d}{O}V^{\mathrm T} \in \mathbb{R}^{N\times N} \notag \\
&\mathrm{d}{S_{i:}} = \cfrac{\partial P_{i:}}{\partial S_{i:}}\cdot\mathrm{d}{P_{i:}}\in \mathbb{R}^{N}, \\
&\quad where\space  \cfrac{\partial P_{i:}}{\partial S_{i:}} = J_{softmax} = \mathrm{diag}(P_{i:}) - P_{i:}P_{i:}^{\mathrm T} \in \mathbb{R}^{N\times N} \notag \\
&\mathrm{d}{Q} = \mathrm{d}{S}K \in \mathbb{R}^{N\times d}, \quad \mathrm{d}{K} = \mathrm{d}{S}^{\mathrm T}Q \in \mathbb{R}^{N\times d} \notag
\end{align}
\end{cases}
```

```{math}
\begin{align}
&\text{where}\space\space \mathrm{d}X \space\space\text{denotes}\space \cfrac{\partial{\mathbb{loss}}}{\partial{X}}, \space\text{and}\space X_{i:} \space\text{denotes the column vector}\\
&\text{made of the $i$-th row of}\space X, \space\text{for any matrix}\space X
\end{align}
```

```{math}
given\quad \mathrm{d}{O} \in \mathbb{R}^{N\times d}
```


### Flash Attention Backward

#### Step0. Save LSE during forward

for each {math}`i`-th row: 

```{math}
:label: fa2_backward_step0_save_lse

\begin{cases}
\begin{align} 
&\text{since}\space P_{i:} = \cfrac{A_{i:}}{l_{i:}} \in \mathbb{R}^{B_k}, \; l_{i} = \mathrm{sum}(A_{i:}) \in \mathbb{R}, \\
&\quad\quad A_{i:} = \exp(S_{i:} - m_{i}) \in \mathbb{R}^{B_k}, \; m_{i} = \max(S_{i:})\in \mathbb{R} \notag\\
&\text{therefore}\space  P_{i:} = \cfrac{\exp(S_{i:} - m_{i})}{\mathrm{sum}(\exp(S_{i:} - m_{i}))} = \cfrac{\exp(S_{i:} - m_{i})}{\exp(\mathrm{LSE}(S_{i:} - m_{i}))}\\
&\quad\quad\quad\quad = \exp(S_{i:} - (m_{i} + \mathrm{LSE}(S_{i:} - m_i))) \notag\\
\\
&\text{and according to}\space  \text{LSE}( \mathbf{x}) = \max( \mathbf x) + \text{LSE}( \mathbf{x}-\max( \mathbf x)), \notag\\
&\text{therefore}\space  P_{i:} = \exp(S_{i:} - (m_{i} + \mathrm{LSE}(S_{i:} - m_i)))\\
&\quad\quad\quad\quad = \exp(S_{i:} - \mathrm{LSE}(S_{i:})) = \exp(S_{i:} - \mathrm{LSE_i})\notag
\end{align}
\end{cases}
```

so we can jump storing {math}`m_i, l_i` to compute {math}`A_{i:}`, but computing {math}`P_{i:}` from {math}`S_{i:}` directly with only {math}`\mathrm{LSE_i}`


#### Step1. Compute Delta as a Pre-Processing

for each {math}`i`-th row: 

```{math}
:label: fa2_backward_step1_compute_delta

\begin{cases}
\begin{align} 
&\text{since}\space \mathrm{d}{S_{i:}} = \cfrac{\partial P_{i:}}{\partial S_{i:}}\cdot\mathrm{d}{P_{i:}} = (\mathrm{diag}(P_{i:}) - P_{i:}P_{i:}^{\mathrm T} )\cdot\mathrm{d}{P_{i:}}\\
&\quad\quad = P_{i:}\odot\mathrm{d}{P_{i:}} - (P_{i:}P_{i:}^{\mathrm T})\mathrm{d}{P_{i:}}  \in \mathbb{R}^{B_k}\notag\\
&\text{then}\space \mathrm{d}{S_{i:}} = P_{i:}\odot\mathrm{d}{P_{i:}} - P_{i:}(P_{i:}^{\mathrm T}\mathrm{d}{P_{i:}}) = P_{i:}\odot\mathrm{d}{P_{i:}} - (P_{i:}^{\mathrm T}\mathrm{d}{P_{i:}})P_{i:}\notag\\
\\
&\text{define}\space  \Delta_{i} = P_{i:}^{\mathrm T}\mathrm{d}{P_{i:}} \in \mathbb{R},\\
&\text{and because}\space  \mathrm{d}{P_{i:}} = (\mathrm{d}{O_{i:}}^{\mathrm T}V^{\mathrm T})^{\mathrm T} = VdO_{i:}  \in \mathbb{R}^{B_k}\notag\\
&\text{therefore}\space \Delta_{i} = P_{i:}^{\mathrm T}\mathrm{d}{P_{i:}} = P_{i:}^{\mathrm T}(VdO_{i:}) = (P_{i:}^{\mathrm T}V)dO_{i:} = O_{i:}^{\mathrm T}dO_{i:}\notag\\
\end{align}
\end{cases}
```

then for all rows, we compute {math}`\Delta = \mathrm{rowsum}(O\odot dO)\in \mathbb{R}^{B_q}` during preprocessing, so we can avoid massive matrix computing like {math}`P_{i:}P_{i:}^{\mathrm T} \in \mathbb{R}^{B_k\times B_k}`


#### Step2. Swapped Double-Loop Tiling with Recomputation

* the outer loop runs through {math}`j := 1 \rightarrow N_k` for each block of {math}`K_j, V_j` to compute {math}`dK_j, dV_j`,  where {math}`N_k = \lceil\frac{N}{B_k}\rceil`, and for each {math}`j`-th outer iteration:

```{math}
:label: fa2_backward_step2_swapped_double_loop_tiling_outer

\begin{cases}
\begin{align} 
&\text{load}\space  K_j, V_j \in \mathbb{R}^{B_k\times d}\space  \text{from HBM to SRAM, }\\
&\text{and initialize}\space  dK_j^{(0)}, dV_j^{(0)} = (0)_{B_c\times d} \in \mathbb{R}^{B_k\times d} \notag \\
\\
&\text{loop over}\space  i := 1 \rightarrow N_q\space \text{, and for each }\space i \text{-th inner iteration: } \notag \\
&\quad\text{write}\space  dK_j = dK_j^{(N_q)}, dV_j = dV_j^{(N_q)} \space \text{back to HBM to return as output} \notag
\end{align}
\end{cases}
```

* in which each inner loop goes across {math}`i := 1 \rightarrow N_q` for each block of {math}`Q_i, dO_i` to update {math}`dQ_i, dK_j^{(i)}, dV_j^{(i)}`, where {math}`N_q = \lceil\frac{N}{B_q}\rceil`, and for each {math}`i`-th inner iteration:

```{math}
:label: fa2_backward_step2_swapped_double_loop_tiling_inner

\begin{cases} 
\begin{align} 
&\text{load}\space  Q_i, dO_i, \mathrm{LSE_i}, \Delta_i\space  \text{from HBM to SRAM} \notag \\
&\text{recompute}\space  S_j^{(i)} = \mathrm{mask}(Q_iK_j^{\mathrm{T}} + bias_{(i,j)}) \in \mathbb{R}^{B_q\times B_k} \notag \\
&\text{recompute}\space  P_j^{(i)} = \exp(S_j^{(i)} - \mathrm{LSE_i}) \in \mathbb{R}^{B_q\times B_k} \notag \\
&\text{update}\space  dV_j^{(i)} = dV_j^{(i-1)} + (P_j^{(i)})^{\mathrm T} dO_i \in \mathbb{R}^{B_k\times d} \notag \\
&\text{compute}\space  dP_j^{(i)} = dO_iV_j^{\mathrm T} \in \mathbb{R}^{B_q\times B_k} \notag \\
&\text{compute}\space  dS_j^{(i)} = P_j^{(i)}\odot (dP_j^{(i)} - \Delta_i) \in \mathbb{R}^{B_q\times B_k} \notag \\
&\text{update}\space  dK_j^{(i)} = dK_j^{(i-1)} + (dS_j^{(i)})^{\mathrm T} Q_i \in \mathbb{R}^{B_k\times d} \notag \\
&\text{update}\space dQ_i \stackrel{atomic\space add}\longleftarrow dS_j^{(i)}K_j \in \mathbb{R}^{B_q\times d} \notag
\end{align}
\end{cases}
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

```{bibliography} refs/fa2_deri.bib
```