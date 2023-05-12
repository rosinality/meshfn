import os

import torch
from torch import nn, optim
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from meshfn.distributed import ParallelContext, ParallelMode
from meshfn.nn.parallel import tensor1d, parallelize
from meshfn.transformers import AutoParallelModelForCausalLM
from meshfn.transformers.models.bloom import BloomPolicy
from meshfn.nn.parallel.tensor1d.ops import gather
from meshfn.nn.load import init_empty_weights, load_checkpoint
from meshfn.logging import get_logger

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    pc = ParallelContext(
        rank,
        world_size,
        host,
        port,
        local_rank,
        tensor_parallel_size=4,
        tensor_parallel_mode=ParallelMode.TENSOR_1D,
    )
    logger = get_logger(pc)

    dtype = torch.float32

    path = "/root/workspace/bloom-560m"

    model_parallel = AutoParallelModelForCausalLM.from_pretrained(path, pc)

    tied = (
        model_parallel.lm_head.weight
        is model_parallel.transformer.word_embeddings.weight
    )
    sanity = (
        model_parallel.lm_head.weight
        is model_parallel.transformer.h[23].mlp.dense_4h_to_h.weight
    )
    logger.info(f"weight tied {tied}, {sanity}", ranks=[0, 1, 2, 3])

    model_parallel.to("cuda").eval()

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = (
        AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m").to("cuda").eval()
    )

    token = tokenizer(["Hello, how are you? I am fine thank you!"], return_tensors="pt")
    token_cuda = {}
    for k, v in token.items():
        token_cuda[k] = v.to("cuda")

    out = model(**token_cuda).logits
    out_parallel = model_parallel(**token_cuda).logits

    diff = (out - out_parallel).abs().view(-1)
    diff_max = diff.argmax()
    diff_max_val = diff[diff_max]
    max_val = out.view(-1)[diff_max]
    max_val_parallel = out_parallel.view(-1)[diff_max]

    logger.info(f"out diff: {diff_max_val}, {max_val}, {max_val_parallel}", ranks=[0])
