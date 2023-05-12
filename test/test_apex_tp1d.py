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


class GetFirst(nn.Module):
    def forward(self, input):
        return input[0]


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        init_method=f"tcp://{host}:{port}",
    )

    torch.cuda.set_device(local_rank)

    parallel_state.initialize_model_parallel(
        4,
        1,
        None,
    )

    get_cuda_rng_tracker().add("model-parallel-rng", 1024)

    dtype = torch.float32

    model = nn.Sequential(
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

    model_single = nn.Sequential(
        nn.Embedding(16, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 512 * 4),
        nn.LeakyReLU(0.2),
        nn.Linear(512 * 4, 512),
    )

    def logger(msg):
        if local_rank == 0:
            print(msg)

    def gather_params():
        return [
            _gather_along_first_dim(model[0].weight.detach()),
            _gather_along_first_dim(model[2].weight.detach()),
            _gather_along_first_dim(model[2].bias.detach()),
            _gather_along_last_dim(model[5].weight.detach()),
            model[5].bias.detach(),
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
            logger(f"weight diff: {(m - s).abs().max()} {m.shape}")

    def gather_grads():
        return [
            _gather_along_first_dim(model[0].weight.grad),
            _gather_along_first_dim(model[2].weight.grad),
            _gather_along_first_dim(model[2].bias.grad),
            _gather_along_last_dim(model[5].weight.grad),
            model[5].bias.grad,
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

    logger(str(model))

    input = torch.arange(16).view(1, -1).to("cuda")

    optimizer = optim.Adam(model.parameters())
    optimizer_single = optim.Adam(model_single.parameters())

    for _ in range(5):
        weight_diff()

        out = model(input)
        out_single = model_single(input)
        logger(f"out diff: {(out_single - out).abs().max()}")
        loss = out.pow(2).sum()
        loss2 = out_single.pow(2).sum()

        loss.backward()
        loss2.backward()

        multi_grad, single_grad = gather_grads()
        for m, s in zip(multi_grad, single_grad):
            logger(f"grad diff: {(m - s).abs().max()}")

        optimizer.step()
        optimizer_single.step()
        model.zero_grad()
        model_single.zero_grad()
