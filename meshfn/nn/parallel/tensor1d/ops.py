import torch
from torch import distributed as dist
from torch import autograd

from meshfn.distributed import ParallelMode


class LinearWithAsyncGrad(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        parallel_context,
        async_grad_allreduce,
        sequence_parallel,
    ):
        ctx.use_bias = bias is not None
        ctx.parallel_context = parallel_context
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.weight_grad = weight.requires_grad

        if ctx.weight_grad:
            ctx.save_for_backward(input, weight)

        else:
            ctx.save_for_backward(weight)

        if sequence_parallel:
            if async_grad_allreduce:
                raise ValueError(
                    "async_grad_allreduce and sequence_parallel cannot be used together"
                )

            world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)
            shape = list(input.shape)
            shape[0] = shape[0] * world_size

            buffer = parallel_context.get_buffer(shape, input.dtype, "seq")
            dist._all_gather_base(
                buffer, input, group=parallel_context.group(ParallelMode.TENSOR_1D)
            )
            input = buffer

        output = torch.matmul(input, weight.t())

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weight_grad:
            input, weight = ctx.saved_tensors

        else:
            (weight,) = ctx.saved_tensors
            input = None

        use_bias = ctx.use_bias

        handle = None

        if ctx.weight_grad:
            if ctx.sequence_parallel:
                world_size = ctx.parallel_context.world_size(ParallelMode.TENSOR_1D)
                shape = list(input.shape)
                shape[0] = shape[0] * world_size

                buffer = ctx.parallel_context.get_buffer(shape, input.dtype, "seq")
                handle = dist._all_gather_base(
                    buffer,
                    input,
                    group=ctx.parallel_context.group(ParallelMode.TENSOR_1D),
                    async_op=True,
                )

                input = buffer

        grad_input = grad_output.matmul(weight)

        if handle is not None:
            handle.wait()

        if ctx.async_grad_allreduce:
            handle = dist.all_reduce(
                grad_input,
                group=ctx.parallel_context.group(ParallelMode.TENSOR_1D),
                async_op=True,
            )

        if not ctx.weight_grad:
            if ctx.sequence_parallel:
                shape = list(grad_input.shape)
                shape[0] //= ctx.parallel_context.world_size(ParallelMode.TENSOR_1D)

                sub_grad_input = torch.empty(
                    shape,
                    dtype=input.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                handle = torch.distributed._reduce_scatter_base(
                    sub_grad_input,
                    grad_input,
                    group=ctx.parallel_context.group(ParallelMode.TENSOR_1D),
                    async_op=True,
                )
                handle.wait()

                return sub_grad_input, None, None, None, None, None

            if ctx.async_grad_allreduce:
                handle.wait()

            return grad_input, None, None, None, None, None

        grad_output = grad_output.contiguous().view(-1, grad_output.shape[-1])
        input = input.view(-1, input.shape[-1])

        if ctx.sequence_parallel:
            sub_grad_input = torch.empty(
                shape,
                dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input,
                grad_input,
                group=ctx.parallel_context.group(ParallelMode.TENSOR_1D),
                async_op=True,
            )

        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()

            return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def linear_with_async_grad(
    input,
    weight,
    bias,
    parallel_context,
    async_grad_allreduce,
    sequence_parallel,
):
    return LinearWithAsyncGrad.apply(
        input,
        weight,
        bias,
        parallel_context,
        async_grad_allreduce,
        sequence_parallel,
    )


def all_reduce(input, parallel_context):
    if parallel_context.world_size(ParallelMode.TENSOR_1D) == 1:
        return input

    group = parallel_context.device_group(ParallelMode.TENSOR_1D, input.device)
    dist.all_reduce(input, group=group)

    return input


def split(input, parallel_context, dim=-1):
    world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

    if world_size == 1:
        return input

    size = input.size(dim)

    tensor_list = torch.split(input, size // world_size, dim=dim)
    rank = parallel_context.local_rank(ParallelMode.TENSOR_1D)
    out = tensor_list[rank].contiguous()

    return out


def gather(input, parallel_context, dim=-1):
    world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

    if world_size == 1:
        return input

    rank = parallel_context.local_rank(ParallelMode.TENSOR_1D)
    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    tensor_list[rank] = input
    group = parallel_context.device_group(ParallelMode.TENSOR_1D, input.device)
    dist.all_gather(tensor_list, input, group=group)

    out = torch.cat(tensor_list, dim=dim).contiguous()

    return out


def gather_first_dim(input, parallel_context):
    world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

    if world_size == 1:
        return input

    shape = list(input.shape)
    shape[0] *= world_size

    out = torch.empty(shape, dtype=input.dtype, device="cuda")
    dist.all_gather_into_tensor(
        out, input.contiguous(), group=parallel_context.group(ParallelMode.TENSOR_1D)
    )

    return out


def reduce_scatter(input, parallel_context, dim=0):
    world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

    if world_size == 1:
        return input

    shape = list(input.shape)
    shape[0] //= world_size
    out = torch.empty(shape, dtype=input.dtype, device="cuda")
    dist.reduce_scatter_tensor(
        out, input.contiguous(), parallel_context.group(ParallelMode.TENSOR_1D)
    )

    return out


class SplitForwardGatherBackward(autograd.Function):
    @staticmethod
    def forward(ctx, input, parallel_context, dim):
        ctx.parallel_context = parallel_context
        ctx.dim = dim

        return split(input, parallel_context, dim)

    @staticmethod
    def backwrad(ctx, grad_output):
        return (
            gather(grad_output, ctx.parallel_context, ctx.dim),
            None,
            None,
        )


def split_forward_gather_backward(input, parallel_context, dim):
    return SplitForwardGatherBackward.apply(input, parallel_context, dim)


class GatherForwardSplitBackward(autograd.Function):
    @staticmethod
    def forward(ctx, input, parallel_context, dim):
        ctx.parallel_context = parallel_context
        ctx.dim = dim

        return gather(input, parallel_context, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            split(grad_output, ctx.parallel_context, ctx.dim),
            None,
            None,
        )


def gather_forward_split_backward(input, parallel_context, dim):
    return GatherForwardSplitBackward.apply(input, parallel_context, dim)


class ReduceInput(autograd.Function):
    @staticmethod
    def forward(ctx, input, parallel_context):
        return all_reduce(input, parallel_context)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def reduce_input(input, parallel_context):
    return ReduceInput.apply(input, parallel_context)


class ReduceScatter(autograd.Function):
    @staticmethod
    def forward(ctx, input, parallel_context):
        ctx.parallel_context = parallel_context

        return reduce_scatter(input, parallel_context, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return gather_first_dim(grad_output, ctx.parallel_context), None


def reduce_scatter(input, parallel_context):
    return ReduceScatter.apply(input, parallel_context)
