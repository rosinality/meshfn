import torch
from torch import distributed as dist

from meshfn.distributed.parallel_mode import ParallelMode
from meshfn.distributed.parallel_context import ParallelContext


def broadcast(
    tensor: torch.Tensor,
    src: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode,
    async_op: bool = False,
):
    world_size = parallel_context.world_size(parallel_mode)

    if world_size == 1:
        out = tensor
        work = None

    else:
        out = tensor.contiguous()
        group = (
            parallel_context.cpu_group(parallel_mode)
            if tensor.device.type == "cpu"
            else parallel_context.group(parallel_mode)
        )
        work = dist.broadcast(out, src=src, group=group, async_op=async_op)

    if async_op:
        return out, work

    else:
        return out


def broadcast_rank0(
    tensor, parallel_context: ParallelContext, parallel_mode: ParallelMode
):
    return broadcast(
        tensor.detach(),
        parallel_context.ranks_in_group(parallel_mode)[0],
        parallel_context,
        parallel_mode,
    )
