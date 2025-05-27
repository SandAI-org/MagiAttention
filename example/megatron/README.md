# Integrating Megatron with MagiAttention

This repository provides an example of training the LLaMA-3B model using MagiAttention. For more details on data preparation, checkpoint setup, integration, and experimentation, please refer to the [MagiAttention experiments](https://github.com/SandAI-org/Megatron-LM-MagiAttention/pull/1).

## Convergence Experiments

We compared the loss convergence curves of TE Ring Attention and MagiAttention by training the LLaMA-1B model from scratch.

### Training Settings

| **Configuration**                 | **Value**                                                                                |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| **Dataset**                   | [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)                        |
| **Model Size**                | LLaMA-1B                                                                                     |
| **Number of Layers**          | 16                                                                                           |
| **Hidden Size**               | 2048                                                                                         |
| **Number of Attention Heads** | 32                                                                                           |
| **Group Query Attention**     | Enabled                                                                                      |
| **Number of Query Groups**    | 8                                                                                            |
| **Sequence Length**           | 8192                                                                                         |
| **Context Parallel Size**     | CP1/2/4/8 (MagiAttention vs. TE Ring Attention) with a global batch size of 16               |
| **Training Iterations**       | 100,000                                                                                      |


### Results

MagiAttention aligns well with TE Ring Attention.

![Results](./results.png)

Feel free to open an issue in the [Megatron-LM-MagiAttention](https://github.com/SandAI-org/Megatron-LM-MagiAttention) repository if you have any questions.
