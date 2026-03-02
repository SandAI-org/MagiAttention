% MagiAttention documentation master file
% :github_url: https://github.com/SandAI-org/MagiAttention

MagiAttention documentation
===================================

**Overview :**

MagiAttention is a distributed attention (Context Parallelism) solution tailored for the demanding requirements of ultra-long sequences and heterogeneous masking patterns. It combines `Flex-Flash-Attention` (`FFA`)—a kernel supporting distributable and **flexible mask** representations—with a `dispatch solver` for **load-balanced computation** and new `Group Collective` primitives for communication to achieve **zero-redundant communication**. By coordinating these components through an **adaptive multi-stage overlap** strategy, MagiAttention delivers **linear scalability** across a broad range of training scenarios, such as large-scale video generation in [Magi-1](https://github.com/SandAI-org/MAGI-1).

We are committed to continually improving the performance and generality of MagiAttention for the broader research community. Stay tuned for exciting enhancements and new features on the horizon!

```{toctree}
:glob:
:maxdepth: 2
:caption: Contents

user_guide/toc
blog/toc

```
