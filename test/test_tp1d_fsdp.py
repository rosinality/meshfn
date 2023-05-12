import os

import torch
from torch import nn, optim
from torch.distributed import fsdp

from meshfn.distributed import ParallelContext, ParallelMode
from meshfn.nn.parallel import tensor1d
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
        tensor_parallel_size=2,
        tensor_parallel_mode=ParallelMode.TENSOR_1D,
    )
    psc = pc[ParallelMode.TENSOR]
    logger = get_logger(pc)

    dtype = torch.float32

    def get_model():
        model = nn.Sequential(
            tensor1d.VocabParallelEmbedding(
                16, 512, parallel_context=pc, dtype=dtype, device="cuda"
            ),
            nn.LeakyReLU(0.2),
            tensor1d.LinearColumn1D(
                512, 512 * 4, parallel_context=pc, dtype=dtype, device="cuda"
            ),
            nn.LeakyReLU(0.2),
            tensor1d.LinearRow1D(
                512 * 4, 512, parallel_context=pc, dtype=dtype, device="cuda"
            ),
        )

        return model

    model_raw = get_model()
    state_dict = model_raw.state_dict()
    model_raw = model_raw.to("cuda")

    model_fsdp = fsdp.FullyShardedDataParallel(
        model_raw,
        process_group=pc.group(ParallelMode.DATA),
        mixed_precision=fsdp.MixedPrecision(param_dtype=torch.float32),
        device_id=local_rank,
    )

    model = get_model()
    model.load_state_dict(state_dict)
    # model[0].weight.detach().add_(1e-4)

    logger.info(str(model), ranks=[0])

    input = torch.arange(16).view(1, -1).to("cuda")

    optimizer = optim.Adam(model.parameters())
    optimizer_single = optim.Adam(model_fsdp.parameters())

    for _ in range(5):
        out = model(input)
        out_single = model_fsdp(input)
        logger.info(f"out diff: {(out_single - out).abs().max()}", ranks=[0])
        loss = out.pow(2).sum()
        loss2 = out_single.pow(2).sum()

        loss.backward()
        loss2.backward()

        optimizer.step()
        optimizer_single.step()
        model.zero_grad()
        model_fsdp.zero_grad()
