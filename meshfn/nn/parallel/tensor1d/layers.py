import math
from typing import Optional, Callable

import torch
from torch import nn
from torch.nn import functional as F

from meshfn.distributed import ParallelContext, ParallelMode, SEEDS, broadcast_rank0
from meshfn.nn import init
from meshfn.nn.parallel.tensor1d.ops import (
    linear_with_async_grad,
    split_forward_gather_backward,
    gather_forward_split_backward,
    reduce_input,
    gather,
)


class LinearColumn1D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        gather_output: bool = False,
        parallel_context: Optional[ParallelContext] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        weight_init: Callable = init.kaiming_uniform(a=math.sqrt(5)),
        bias_init: Callable = init.linear_bias(),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.gather_output = gather_output

        self.parallel_context = parallel_context

        self.weight = nn.UninitializedParameter(
            dtype=dtype,
            device=device,
        )

        self.bias = None

        if bias:
            self.bias = nn.UninitializedParameter(dtype=dtype, device=device)

        self.weight_init = weight_init
        self.bias_init = bias_init

    def materialize(self, device=None, dtype=None):
        world_size = 1
        if self.parallel_context is not None:
            world_size = self.parallel_context.world_size(ParallelMode.TENSOR_1D)

        self.weight.materialize(
            (self.out_features // world_size, self.in_features),
            device=device,
            dtype=dtype,
        )

        if self.bias is not None:
            self.bias.materialize(
                (self.out_features // world_size,), device=device, dtype=dtype
            )

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = linear_with_async_grad(
            input,
            self.weight,
            self.bias,
            self.parallel_context,
            ParallelMode.TENSOR_1D,
            async_grad_allreduce=True,
        )

        if self.gather_output:
            out = gather_forward_split_backward(
                out, self.parallel_context, ParallelMode.TENSOR_1D, dim=-1
            )

        return out

    def reset_parameters(self):
        if self.parallel_context is not None:
            with SEEDS.seed(ParallelMode.TENSOR):
                self._reset_parameters()

        else:
            self._reset_parameters()

    def _reset_parameters(self):
        self.weight_init(self.weight)

        if self.bias is not None:
            fan_in = self.in_features

            self.bias_init(self.bias, fan_in)

    def parallelize(
        self,
        parallel_context: ParallelContext,
    ):
        world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

        weight = self.weight
        bias = self.bias

        if hasattr(self.weight, "materialize"):
            self.parallel_context = parallel_context

            return

        if self.parallel_context is not None:
            weight = gather(
                weight.detach(), self.parallel_context, ParallelMode.TENSOR_1D, 0
            )

            if bias is not None:
                bias = gather(
                    bias.detach(), self.parallel_context, ParallelMode.TENSOR_1D, 0
                )

        else:
            broadcast_rank0(
                weight.detach(),
                parallel_context,
                ParallelMode.TENSOR_1D,
            )

            if bias is not None:
                broadcast_rank0(
                    bias.detach(),
                    parallel_context,
                    ParallelMode.TENSOR_1D,
                )

        local_rank = parallel_context.local_rank(ParallelMode.TENSOR_1D)

        self.weight.data = weight.chunk(world_size, 0)[local_rank].data

        if bias is not None:
            self.bias.data = bias.chunk(world_size, 0)[local_rank].data

        self.parallel_context = parallel_context

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LinearRow1D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        parallel_input: bool = True,
        parallel_context: Optional[ParallelContext] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        weight_init: Callable = init.kaiming_uniform(a=math.sqrt(5)),
        bias_init: Callable = init.linear_bias(),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.parallel_context = parallel_context
        self.parallel_input = parallel_input

        self.weight = nn.UninitializedParameter(dtype=dtype, device=device)

        self.bias = None

        if bias:
            self.bias = nn.UninitializedParameter(dtype=dtype, device=device)

        self.weight_init = weight_init
        self.bias_init = bias_init

    def materialize(self, device=None, dtype=None):
        world_size = 1

        if self.parallel_context is not None:
            world_size = self.parallel_context.world_size(ParallelMode.TENSOR_1D)

        self.weight.materialize(
            (self.out_features, self.in_features // world_size),
            device=device,
            dtype=dtype,
        )

        if self.bias is not None:
            self.bias.materialize((self.out_features,), device=device, dtype=dtype)

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.parallel_input:
            input = split_forward_gather_backward(
                input, self.parallel_context, ParallelMode.TENSOR_1D, dim=-1
            )

        out = F.linear(input, self.weight)
        out = reduce_input(out, self.parallel_context, ParallelMode.TENSOR_1D)

        if self.bias is not None:
            out = out + self.bias

        return out

    def reset_parameters(self):
        if self.parallel_context is not None:
            with SEEDS.seed(ParallelMode.TENSOR):
                self._reset_parameters()

        else:
            self._reset_parameters()

    def _reset_parameters(self):
        self.weight_init(self.weight)

        if self.bias is not None:
            fan_in = self.in_features

            self.bias_init(self.bias, fan_in)

            if self.parallel_context is not None:
                broadcast_rank0(
                    self.bias,
                    self.parallel_context,
                    ParallelMode.TENSOR_1D,
                )

    def parallelize(
        self,
        parallel_context: ParallelContext,
    ):
        world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

        weight = self.weight
        bias = self.bias

        if hasattr(self.weight, "materialize"):
            self.parallel_context = parallel_context

            return

        if self.parallel_context is not None:
            weight = gather(
                weight.detach(), self.parallel_context, ParallelMode.TENSOR_1D, 1
            )

        else:
            broadcast_rank0(
                weight.detach(),
                parallel_context,
                ParallelMode.TENSOR_1D,
            )

        local_rank = parallel_context.local_rank(ParallelMode.TENSOR_1D)

        self.weight.data = weight.chunk(world_size, 1)[local_rank].data

        if bias is not None:
            broadcast_rank0(
                bias.detach(),
                parallel_context,
                ParallelMode.TENSOR_1D,
            )
            self.bias.data = bias.data

        self.parallel_context = parallel_context

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Embedding1D(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[int] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        parallel_context: Optional[ParallelContext] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        weight_init: Callable = init.normal(),
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.parallel_context = parallel_context

        self.weight = nn.UninitializedParameter(dtype=dtype, device=device)

        self.weight_init = weight_init

    def materialize(self, device=None, dtype=None):
        world_size = 1

        if self.parallel_context is not None:
            world_size = self.parallel_context.world_size(ParallelMode.TENSOR_1D)

        self.weight.materialize(
            (self.num_embeddings, self.embed_dim // world_size),
            device=device,
            dtype=dtype,
        )

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        out = gather_forward_split_backward(
            out, self.parallel_context, ParallelMode.TENSOR_1D, dim=-1
        )

        return out

    def reset_parameters(self):
        if self.parallel_context is not None:
            with SEEDS.seed(ParallelMode.TENSOR):
                self._reset_parameters()

        else:
            self._reset_parameters()

    def _reset_parameters(self):
        self.weight_init(self.weight)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def parallelize(
        self,
        parallel_context: ParallelContext,
    ):
        world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

        weight = self.weight

        if hasattr(self.weight, "materialize"):
            self.parallel_context = parallel_context

            return

        if self.parallel_context is not None:
            weight = gather(
                weight.detach(), self.parallel_context, ParallelMode.TENSOR_1D, 1
            )

        else:
            broadcast_rank0(
                weight.detach(),
                parallel_context,
                ParallelMode.TENSOR_1D,
            )

        local_rank = parallel_context.local_rank(ParallelMode.TENSOR_1D)

        self.weight.data = weight.chunk(world_size, 1)[local_rank].data

        self.parallel_context = parallel_context
