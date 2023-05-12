import os

import torch
from torch import nn, optim

from meshfn.distributed import ParallelContext, ParallelMode
from meshfn.nn.parallel import tensor1d
from meshfn.nn import init
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
    psc = pc[ParallelMode.TENSOR]
    logger = get_logger(pc)

    dtype = torch.float32

    model = nn.Sequential(
        tensor1d.VocabParallelEmbedding(
            16,
            512,
            parallel_context=pc,
            dtype=dtype,
            device="cuda",
            weight_init=init.normal(),
        ),
        nn.LeakyReLU(0.2),
        tensor1d.LinearColumn1D(
            512,
            512 * 4,
            parallel_context=pc,
            dtype=dtype,
            device="cuda",
            weight_init=init.normal(),
            bias_init=init.zeros(),
        ),
        nn.LeakyReLU(0.2),
        tensor1d.LinearRow1D(
            512 * 4,
            512,
            parallel_context=pc,
            dtype=dtype,
            device="cuda",
            weight_init=init.normal(),
            bias_init=init.zeros(),
        ),
    )

    model_single = nn.Sequential(
        nn.Embedding(16, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 512 * 4),
        nn.LeakyReLU(0.2),
        nn.Linear(512 * 4, 512),
    )

    def gather_params():
        return [
            gather(model[0].weight.detach(), psc, 0),
            gather(model[2].weight.detach(), psc, 0),
            gather(model[2].bias.detach(), psc, 0),
            gather(model[4].weight.detach(), psc, 1),
            model[4].bias.detach(),
        ], [
            model_single[0].weight,
            model_single[2].weight,
            model_single[2].bias,
            model_single[4].weight,
            model_single[4].bias,
        ]

    @torch.no_grad()
    def weight_diff():
        multi_params, single_params = gather_params()

        for m, s in zip(multi_params, single_params):
            logger.info(f"weight diff: {(m - s).abs().max()} {m.shape}", ranks=[0])

    def gather_grads():
        return [
            gather(model[0].weight.grad, psc, 0),
            gather(model[2].weight.grad, psc, 0),
            gather(model[2].bias.grad, psc, 0),
            gather(model[4].weight.grad, psc, 1),
            model[4].bias.grad,
        ], [
            model_single[0].weight.grad,
            model_single[2].weight.grad,
            model_single[2].bias.grad,
            model_single[4].weight.grad,
            model_single[4].bias.grad,
        ]

    with torch.no_grad():
        params, _ = gather_params()
        model_single[0].weight.copy_(params[0])
        model_single[2].weight.copy_(params[1])
        model_single[2].bias.copy_(params[2])
        model_single[4].weight.copy_(params[3])
        model_single[4].bias.copy_(params[4])

    model_single = model_single.to("cuda").to(dtype)

    logger.info(str(model), ranks=[0])

    input = torch.arange(16).view(1, -1).to("cuda")

    optimizer = optim.Adam(model.parameters())
    optimizer_single = optim.Adam(model_single.parameters())

    for _ in range(5):
        weight_diff()

        out = model(input)
        out_single = model_single(input)
        logger.info(f"out diff: {(out_single - out).abs().max()}", ranks=[0])
        loss = out.pow(2).sum()
        loss2 = out_single.pow(2).sum()

        loss.backward()
        loss2.backward()

        multi_grad, single_grad = gather_grads()
        for m, s in zip(multi_grad, single_grad):
            logger.info(f"grad diff: {(m - s).abs().max()}", ranks=[0])

        optimizer.step()
        optimizer_single.step()
        model.zero_grad()
        model_single.zero_grad()
