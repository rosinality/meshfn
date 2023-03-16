from contextlib import contextmanager
import math
import warnings

import torch
from torch import nn


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    orig_register_parameter = nn.Module.register_parameter

    if include_buffers:
        orig_register_buffer = nn.Module.register_buffer

    def register_parameter(module, name, param):
        orig_register_parameter(module, name, param)

        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    def register_buffer(module, name, buffer):
        orig_register_buffer(module, name, buffer)

        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    tensor_constructors_to_patch = {}

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device

            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_parameter

        if include_buffers:
            nn.Module.register_buffer = register_buffer

        for (
            torch_function_name,
            orig_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(orig_torch_function),
            )

        yield

    finally:
        nn.Module.register_parameter = orig_register_parameter

        if include_buffers:
            nn.Module.register_buffer = orig_register_buffer

        for (
            torch_function_name,
            orig_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, orig_torch_function)


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
