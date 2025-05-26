## integrate megatron with magiattention
We fork a repo of megatron and provide an example for you to train llama-3-1b with magiattention with different cp size. You can refer to https://github.com/SandAI-org/Megatron-LM-MagiAttention/pull/1 for more details about how to prepare data, checkpoint, integrate magiattention and do experiments.


### Convergence Experiments
We compare the loss convergence curve of te context_parallel(ring_attention) and magiattention by training llama-1b model from scratch.

Training_settings:
- dataset:  openwebtext(https://huggingface.co/datasets/Skylion007/openwebtext).
- model-size: llama-1b
    - num-layers: 16
    - hidden-size: 2048
    - num-attention-heads: 32
    - group-query-attention
    - num-query-groups: 8
- seqlen: 8192
- context_parallel_size: cp1/2/4/8(magiattention vs te ring attention) with global batch size 16.
- train_iters: 100000

Results:
**MagiAttention aligns well with te ring attention.**
 ![alt text](./results.png)

Feel free to open any issue in [Megatron-LM-MagiAttention](https://github.com/SandAI-org/Megatron-LM-MagiAttention) repo if you have any question!
