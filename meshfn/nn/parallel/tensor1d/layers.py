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
    reduce_scatter,
    gather,
)


def divide(numerator, denominator):
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")

    return numerator // denominator


@torch.no_grad()
def init_weight(
    weight,
    shape,
    dtype,
    init_fn,
    shard_dim,
    parallel_context,
    stride=1,
    device="cuda",
):
    if device is None:
        device = "cpu"

    device = torch.device(device)

    fan_in, fan_out = None, None

    if len(shape) > 1:
        fan_in, fan_out = init.calc_fan_in_and_out(shape)

    if device.type == "cuda":
        with SEEDS.seed(ParallelMode.TENSOR):
            init_fn(weight, fan_in, fan_out)

    else:
        master_weight = torch.empty(*shape, dtype=dtype, device=device)
        init_fn(master_weight, fan_in, fan_out)
        shard_size = divide(
            shape[shard_dim], parallel_context.world_size(ParallelMode.TENSOR_1D)
        )
        shard_size = divide(shard_size, stride)
        weight_shard = master_weight.split(shard_size, dim=shard_dim)
        local_weight = weight_shard[
            parallel_context.local_rank(
                ParallelMode.TENSOR_1D
            ) :: parallel_context.world_size(ParallelMode.TENSOR_1D)
        ]
        torch.cat(local_weight, shard_dim, out=weight)

    return weight


def vocab_range_from_shard(shard, rank):
    start = rank * shard
    end = start + shard

    return start, end


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int
):
    shard = divide(global_vocab_size, world_size)

    return vocab_range_from_shard(shard, rank)


class LinearColumn1D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        gather_output: bool = False,
        sequence_parallel: bool = False,
        parallel_context: ParallelContext = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        weight_init: Callable = init.kaiming_uniform(a=math.sqrt(5)),
        bias_init: Callable = init.zeros(),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.out_shard = divide(
            out_features, parallel_context.world_size(ParallelMode.TENSOR_1D)
        )

        self.gather_output = gather_output

        self.parallel_context = parallel_context
        self.dtype = dtype
        self.device = device

        self.weight = nn.Parameter(
            torch.empty(
                self.out_shard, self.in_features, dtype=self.dtype, device=self.device
            )
        )

        self.bias = None

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_shard, dtype=self.dtype, device=self.device)
            )

        self.weight_init = weight_init
        self.bias_init = bias_init

        self.sequence_parallel = sequence_parallel

        self.reset_parameters()

    @classmethod
    def from_module(
        cls, module: nn.Linear, parallel_context: ParallelContext, **kwargs
    ):
        is_meta = module.weight.is_meta

        linear_args = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
            "dtype": module.weight.dtype,
            "device": module.weight.device if not is_meta else None,
            "parallel_context": parallel_context,
        }
        linear_args.update(kwargs)
        linear = cls(**linear_args)

        if not is_meta:
            shard_start = (
                parallel_context.local_rank(ParallelMode.TENSOR_1D) * linear.out_shard
            )
            shard_end = (
                parallel_context.local_rank(ParallelMode.TENSOR_1D) + 1
            ) * linear.out_shard

            with torch.no_grad():
                linear.weight.copy_(module.weight[shard_start:shard_end])

                if linear.bias is not None:
                    linear.bias.copy_(module.bias[shard_start:shard_end])

        return linear

    def reset_parameters(self):
        self.weight = init_weight(
            self.weight,
            (self.out_features, self.in_features),
            self.dtype,
            self.weight_init,
            0,
            self.parallel_context,
            device=self.device,
        )

        if self.bias is not None:
            init_weight(
                self.bias,
                (self.out_features,),
                self.dtype,
                self.bias_init,
                0,
                self.parallel_context,
                device=self.device,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = linear_with_async_grad(
            input,
            self.weight,
            self.bias,
            self.parallel_context,
            async_grad_allreduce=True,
            sequence_parallel=self.sequence_parallel,
        )

        if self.gather_output:
            out = gather_forward_split_backward(out, self.parallel_context, dim=-1)

        return out

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
        sequence_parallel: bool = False,
        parallel_context: ParallelContext = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        weight_init: Callable = init.kaiming_uniform(a=math.sqrt(5)),
        bias_init: Callable = init.zeros(),
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.in_shard = divide(
            in_features, parallel_context.world_size(ParallelMode.TENSOR_1D)
        )

        self.parallel_context = parallel_context
        self.parallel_input = parallel_input

        self.dtype = dtype
        self.device = device

        self.weight = nn.Parameter(
            torch.empty(
                self.out_features, self.in_shard, dtype=self.dtype, device=self.device
            )
        )

        self.bias = None

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=self.dtype, device=self.device)
            )

        self.weight_init = weight_init
        self.bias_init = bias_init

        self.sequence_parallel = sequence_parallel

        self.reset_parameters()

    @classmethod
    def from_module(
        cls, module: nn.Linear, parallel_context: ParallelContext, **kwargs
    ):
        is_meta = module.weight.is_meta

        linear_args = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
            "dtype": module.weight.dtype,
            "device": module.weight.device if not is_meta else None,
            "parallel_context": parallel_context,
        }
        linear_args.update(kwargs)
        linear = cls(**linear_args)

        if not is_meta:
            shard_start = (
                parallel_context.local_rank(ParallelMode.TENSOR_1D) * linear.in_shard
            )
            shard_end = (
                parallel_context.local_rank(ParallelMode.TENSOR_1D) + 1
            ) * linear.in_shard

            with torch.no_grad():
                linear.weight.copy_(module.weight[:, shard_start:shard_end])

                if linear.bias is not None:
                    linear.bias.copy_(module.bias)

        return linear

    def reset_parameters(self):
        self.weight = init_weight(
            self.weight,
            (self.out_features, self.in_features),
            self.dtype,
            self.weight_init,
            1,
            self.parallel_context,
            device=self.device,
        )

        if self.bias is not None:
            self.bias_init(self.bias)

            if self.parallel_context is not None:
                broadcast_rank0(
                    self.bias, self.parallel_context, ParallelMode.TENSOR_1D
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.parallel_input:
            input = split_forward_gather_backward(input, self.parallel_context, dim=-1)

        out = linear_with_async_grad(
            input,
            self.weight,
            None,
            self.parallel_context,
            async_grad_allreduce=False,
            sequence_parallel=False,
        )

        if self.sequence_parallel:
            out = reduce_scatter(out, self.parallel_context)

        else:
            out = reduce_input(out, self.parallel_context)

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[int] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        parallel_context: ParallelContext = None,
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

        self.vocab_start, self.vocab_end = vocab_range_from_global_vocab_size(
            self.num_embeddings,
            self.parallel_context.local_rank(ParallelMode.TENSOR_1D),
            self.parallel_context.world_size(ParallelMode.TENSOR_1D),
        )
        self.embed_shard = self.vocab_end - self.vocab_start
        self.dtype = dtype
        self.device = device

        self.weight = nn.Parameter(
            torch.empty(
                self.embed_shard, self.embed_dim, dtype=self.dtype, device=self.device
            )
        )

        self.weight_init = weight_init

        self.reset_parameters()

    @classmethod
    def from_module(
        cls, module: nn.Embedding, parallel_context: ParallelContext, **kwargs
    ):
        is_meta = module.weight.is_meta

        embed_args = {
            "num_embeddings": module.num_embeddings,
            "embedding_dim": module.embedding_dim,
            "padding_idx": module.padding_idx,
            "max_norm": module.max_norm,
            "norm_type": module.norm_type,
            "scale_grad_by_freq": module.scale_grad_by_freq,
            "sparse": module.sparse,
            "dtype": module.weight.dtype,
            "device": module.weight.device if not is_meta else None,
            "parallel_context": parallel_context,
        }
        embed_args.update(kwargs)
        embed = cls(**embed_args)

        if not is_meta:
            with torch.no_grad():
                weight = module.weight
                n_embed = weight.shape[0]
                world_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

                if n_embed % world_size != 0:
                    pad = math.ceil(n_embed / world_size) * world_size - n_embed
                    weight = F.pad(weight, (0, 0, 0, pad))

                embed.weight.copy_(weight[embed.vocab_start : embed.vocab_end])

        return embed

    def reset_parameters(self):
        init_weight(
            self.weight,
            (self.num_embeddings, self.embed_dim),
            self.dtype,
            self.weight_init,
            0,
            self.parallel_context,
            device=self.device,
        )

        if (
            self.padding_idx is not None
            and self.parallel_context.local_rank(ParallelMode.TENSOR_1D) == 0
        ):
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.parallel_context.world_size(ParallelMode.TENSOR_1D) > 1:
            input_mask = (input < self.vocab_start) | (input >= self.vocab_end)
            input = input.clone() - self.vocab_start
            input[input_mask] = 0

        out = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        if self.parallel_context.world_size(ParallelMode.TENSOR_1D) > 1:
            out[input_mask, :] = 0

        out = reduce_input(out, self.parallel_context)

        return out
