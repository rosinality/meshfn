import torch
from torch import distributed as dist
from torch import autograd


class LinearWithAsyncGrad(autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias, parallel_context, parallel_mode, async_grad_allreduce
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.parallel_context = parallel_context
        ctx.parallel_mode = parallel_mode
        ctx.async_grad_allreduce = async_grad_allreduce

        output = torch.matmul(input, weight.t())

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        grad_input = grad_output.matmul(weight)

        grad_output = grad_output.view(-1, grad_output.shape[-1])
        input = input.view(-1, input.shape[-1])

        if ctx.async_grad_allreduce:
            handle = dist.all_reduce(
                grad_input,
                group=ctx.parallel_context.group(ctx.parallel_mode),
                async_op=True,
            )
            # from colossalai:
            # delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def linear_with_async_grad(
    input, weight, bias, parallel_context, parallel_mode, async_grad_allreduce
):
    return LinearWithAsyncGrad.apply(
        input, weight, bias, parallel_context, parallel_mode, async_grad_allreduce
    )


def all_reduce(input, parallel_context, parallel_mode):
    if parallel_context.world_size(parallel_mode) == 1:
        return input

    group = (
        parallel_context.cpu_group(parallel_mode)
        if input.device.type == "cpu"
        else parallel_context.group(parallel_mode)
    )
    dist.all_reduce(input, group=group)

    return input


def split(input, parallel_context, parallel_mode, dim=-1):
    world_size = parallel_context.world_size(parallel_mode)

    if world_size == 1:
        return input

    size = input.size(dim)

    tensor_list = torch.split(input, size // world_size, dim=dim)
    rank = parallel_context.local_rank(parallel_mode)
    out = tensor_list[rank].contiguous()

    return out


def gather(input, parallel_context, parallel_mode, dim=-1):
    world_size = parallel_context.world_size(parallel_mode)

    if world_size == 1:
        return input

    rank = parallel_context.local_rank(parallel_mode)
    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    tensor_list[rank] = input
    group = (
        parallel_context.cpu_group(parallel_mode)
        if input.device.type == "cpu"
        else parallel_context.group(parallel_mode)
    )
    dist.all_gather(tensor_list, input, group=group)

    out = torch.cat(tensor_list, dim=dim).contiguous()

    return out


class SplitForwardGatherBackward(autograd.Function):
    @staticmethod
    def forward(ctx, input, parallel_context, parallel_mode, dim):
        ctx.parallel_context = parallel_context
        ctx.parallel_mode = parallel_mode
        ctx.dim = dim

        return split(input, parallel_context, parallel_mode, dim)

    @staticmethod
    def backwrad(ctx, grad_output):
        return (
            gather(grad_output, ctx.parallel_context, ctx.parallel_mode, ctx.dim),
            None,
            None,
            None,
        )


def split_forward_gather_backward(input, parallel_context, parallel_mode, dim):
    return SplitForwardGatherBackward.apply(input, parallel_context, parallel_mode, dim)


class GatherForwardSplitBackward(autograd.Function):
    @staticmethod
    def forward(ctx, input, parallel_context, parallel_mode, dim):
        ctx.parallel_context = parallel_context
        ctx.parallel_mode = parallel_mode
        ctx.dim = dim

        return gather(input, parallel_context, parallel_mode, dim)

    @staticmethod
    def backwrad(ctx, grad_output):
        return (
            split(grad_output, ctx.parallel_context, ctx.parallel_mode, ctx.dim),
            None,
            None,
            None,
        )


def gather_forward_split_backward(input, parallel_context, parallel_mode, dim):
    return GatherForwardSplitBackward.apply(input, parallel_context, parallel_mode, dim)


class ReduceInput(autograd.Function):
    @staticmethod
    def forward(ctx, input, parallel_context, parallel_mode):
        return all_reduce(input, parallel_context, parallel_mode)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def reduce_input(input, parallel_context, parallel_mode):
    return ReduceInput.apply(input, parallel_context, parallel_mode)
