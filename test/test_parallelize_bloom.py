import os

import torch
from torch import nn, optim
from transformers import BloomForCausalLM, AutoTokenizer

from meshfn.distributed import ParallelContext, ParallelMode
from meshfn.nn.parallel import tensor1d, parallelize
from meshfn.transformers.models.bloom import BloomPolicy
from meshfn.nn.parallel.tensor1d.ops import gather
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

    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m").to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model_parallel = parallelize(model, BloomPolicy, pc)
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m").to("cuda").eval()

    logger.info(str(model_parallel), ranks=[0])
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
