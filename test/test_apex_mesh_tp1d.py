import os

import torch
from torch import distributed as dist
from torch import nn, optim

from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from apex.transformer.tensor_parallel.mappings import (
    _gather_along_last_dim,
    _gather_along_first_dim,
)
from apex.transformer.tensor_parallel.random import get_cuda_rng_tracker
from meshfn.distributed import ParallelContext, ParallelMode
from meshfn.nn.parallel import tensor1d
from meshfn.nn import init
from meshfn.nn.parallel.tensor1d.ops import gather
from meshfn.logging import get_logger


class GetFirst(nn.Module):
    def forward(self, input):
        return input[0]


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

    parallel_state.initialize_model_parallel(
        4,
        1,
        None,
    )

    get_cuda_rng_tracker().add("model-parallel-rng", 1024)

    dtype = torch.float32

    model_apex = nn.Sequential(
        VocabParallelEmbedding(
            16, 512, params_dtype=dtype, init_method=nn.init.normal_
        ),
        nn.LeakyReLU(0.2),
        ColumnParallelLinear(
            512, 512 * 4, params_dtype=dtype, init_method=nn.init.normal_
        ),
        GetFirst(),
        nn.LeakyReLU(0.2),
        RowParallelLinear(
            512 * 4, 512, params_dtype=dtype, init_method=nn.init.normal_
        ),
        GetFirst(),
    )

    model_mesh = nn.Sequential(
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

    def gather_params():
        return [
            model_apex[0].weight,
            model_apex[2].weight,
            model_apex[2].bias,
            model_apex[5].weight,
            model_apex[5].bias,
        ], [
            model_mesh[0].weight,
            model_mesh[2].weight,
            model_mesh[2].bias,
            model_mesh[4].weight,
            model_mesh[4].bias,
        ]

    @torch.no_grad()
    def weight_diff():
        multi_params, single_params = gather_params()

        for m, s in zip(multi_params, single_params):
            logger.info(f"weight diff: {(m - s).abs().max()} {m.shape}", ranks=[0])

    def gather_grads():
        return [
            model_apex[0].weight.grad,
            model_apex[2].weight.grad,
            model_apex[2].bias.grad,
            model_apex[5].weight.grad,
            model_apex[5].bias.grad,
        ], [
            model_mesh[0].weight.grad,
            model_mesh[2].weight.grad,
            model_mesh[2].bias.grad,
            model_mesh[4].weight.grad,
            model_mesh[4].bias.grad,
        ]

    with torch.no_grad():
        params, _ = gather_params()
        model_mesh[0].weight.copy_(params[0])
        model_mesh[2].weight.copy_(params[1])
        model_mesh[2].bias.copy_(params[2])
        model_mesh[4].weight.copy_(params[3])
        model_mesh[4].bias.copy_(params[4])

    logger.info(str(model_mesh), ranks=[0])

    input = torch.arange(16).view(1, -1).to("cuda")

    optimizer = optim.Adam(model_apex.parameters())
    optimizer_single = optim.Adam(model_mesh.parameters())

    for _ in range(5):
        weight_diff()

        out = model_apex(input)
        out_single = model_mesh(input)
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
        model_apex.zero_grad()
        model_mesh.zero_grad()
