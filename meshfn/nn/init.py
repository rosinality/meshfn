import math
import warnings

import torch
from torch import nn


def calc_fan_in_and_out(shape):
    dimensions = len(shape)

    if dimensions < 2:
        raise ValueError(
            "fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1

    if len(shape) > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def empty():
    def initializer(tensor: torch.Tensor, fan_in=None, fan_out=None):
        return

    return initializer


def normal(mean: float = 0, std: float = 1.0):
    def initializer(tensor: torch.Tensor, fan_in=None, fan_out=None):
        nn.init.normal_(tensor, mean, std)

    return initializer


def trunc_normal(mean: float = 0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    def initializer(tensor: torch.Tensor, fan_in=None, fan_out=None):
        nn.init.trunc_normal_(tensor, mean, std, a, b)

    return initializer


def uniform(a: float = 0, b: float = 1):
    def initializer(tensor: torch.Tensor, fan_in=None, fan_out=None):
        nn.init.uniform(tensor, a, b)

    return initializer


def zeros():
    def initializer(tensor: torch.Tensor, fan_in=None, fan_out=None):
        nn.init.zeros_(tensor)

    return initializer


def constant(val: float):
    def initializer(tensor: torch.Tensor, fan_in=None, fan_out=None):
        nn.init.constant_(tensor, val)

    return initializer


def kaiming_uniform(
    a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
    def initializer(tensor: torch.Tensor, fan_in=None, fan_out=None):
        if 0 in tensor.shape:
            warnings.warn("Initializing zero-element tensors is no-op")

            return tensor

        fan = nn.init._calculate_correct_fan(tensor, mode)

        if mode == "fan_in" and fan_in is not None:
            fan = fan_in

        if mode == "fan_out" and fan_out is not None:
            fan = fan_out

        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std

        nn.init.uniform_(tensor, -bound, bound)

    return initializer


def linear_bias():
    def initializer(tensor: torch.Tensor, fan_in, fan_out=None):
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(tensor, -bound, bound)

    return initializer
