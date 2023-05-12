from contextlib import contextmanager
from typing import Dict, List, Optional, Union
import os
import json
import gc
import importlib

import torch
from torch import nn


def is_safetensors_available():
    return importlib.util.find_spec("safetensors") is not None


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file


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


class FindTiedParametersResult(list):
    """
    This is a subclass of a list to handle backward compatibility for Transformers. Do not rely on the fact this is not
    a list or on the `values` method as in the future this will be removed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def values(self):
        # TODO: at the next Transformers release (4.28.0) issue a deprecation warning here.
        return sum([x[1:] for x in self], [])


def find_tied_parameters(model: nn.Module, **kwargs):
    """
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`torch.nn.Module`): The model to inspect.

    Returns:
        List[List[str]]: A list of lists of parameter names being all tied together.

    Example:

    ```py
    >>> from collections import OrderedDict
    >>> import torch.nn as nn

    >>> model = nn.Sequential(OrderedDict([("linear1", nn.Linear(4, 4)), ("linear2", nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    [['linear1.weight', 'linear2.weight']]
    ```
    """
    # Initialize result and named_parameters before recursing.
    named_parameters = kwargs.get("named_parameters", None)
    prefix = kwargs.get("prefix", "")
    result = kwargs.get("result", {})

    if named_parameters is None:
        named_parameters = {n: p for n, p in model.named_parameters()}
    else:
        # A tied parameter will not be in the full `named_parameters` seen above but will be in the `named_parameters`
        # of the submodule it belongs to. So while recursing we track the names that are not in the initial
        # `named_parameters`.
        for name, parameter in model.named_parameters():
            full_name = name if prefix == "" else f"{prefix}.{name}"
            if full_name not in named_parameters:
                # When we find one, it has to be one of the existing parameters.
                for new_name, new_param in named_parameters.items():
                    if new_param is parameter:
                        if new_name not in result:
                            result[new_name] = []
                        result[new_name].append(full_name)

    # Once we have treated direct parameters, we move to the child modules.
    for name, child in model.named_children():
        child_name = name if prefix == "" else f"{prefix}.{name}"
        find_tied_parameters(
            child, named_parameters=named_parameters, prefix=child_name, result=result
        )

    return FindTiedParametersResult(
        [sorted([weight] + list(set(tied))) for weight, tied in result.items()]
    )


def retie_parameters(model, tied_params):
    """
    Reties tied parameters in a given model if the link was broken (for instance when adding hooks).

    Args:
        model (`torch.nn.Module`):
            The model in which to retie parameters.
        tied_params (`List[List[str]]`):
            A mapping parameter name to tied parameter name as obtained by `find_tied_parameters`.
    """
    for tied_group in tied_params:
        param_to_tie = None
        # First iteration of the loop will set param_to_tie, next ones will tie it to the others
        for param_name in tied_group:
            module = model
            splits = param_name.split(".")
            for split in splits[:-1]:
                module = getattr(module, split)
            if param_to_tie is None:
                param_to_tie = getattr(module, splits[-1])
            else:
                setattr(module, splits[-1], param_to_tie)


def load_state_dict(checkpoint_file):
    """
    Load a checkpoint from a given file. If the checkpoint is in the safetensors format and a device map is passed, the
    weights can be fast-loaded directly on the GPU.

    Args:
        checkpoint_file (`str`): The path to the checkpoint to load.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
    """
    if checkpoint_file.endswith(".safetensors"):
        if not is_safetensors_available():
            raise ImportError(
                f"To load {checkpoint_file}, the `safetensors` library is necessary `pip install safetensors`."
            )
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
            weight_names = f.keys()
        if metadata.get("format") not in ["pt", "tf", "flax"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        elif metadata["format"] != "pt":
            raise ValueError(
                f"The checkpoint passed was saved with {metadata['format']}, we need a the pt format."
            )

        return safe_load_file(checkpoint_file)

    else:
        return torch.load(checkpoint_file)


def load_checkpoint(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    dtype: Optional[Union[str, torch.dtype]] = None,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded.

    <Tip warning={true}>

    Once loaded across devices, you still need to call [`dispatch_model`] on your model to make it able to run. To
    group the checkpoint loading and dispatch in one single call, use [`load_checkpoint_and_dispatch`].

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit.
        offload_buffers (`bool`, *optional*, defaults to `False):
            Whether or not to include the buffers in the weights offloaded to disk.
    """
    tied_params = find_tied_parameters(model)

    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("torch.", "")
        dtype = getattr(torch, dtype)

    checkpoint_files = None
    index_filename = None
    if os.path.isfile(checkpoint):
        if str(checkpoint).endswith(".json"):
            index_filename = checkpoint
        else:
            checkpoint_files = [checkpoint]

    elif os.path.isdir(checkpoint):
        potential_index = [
            f for f in os.listdir(checkpoint) if f.endswith(".index.json")
        ]

        if len(potential_index) == 0:
            raise ValueError(
                f"{checkpoint} is not a folder containing a `.index.json` file."
            )

        elif len(potential_index) == 1:
            index_filename = os.path.join(checkpoint, potential_index[0])

        else:
            raise ValueError(
                f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
            )

    else:
        raise ValueError(
            "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
            f"checkpoint, or a folder containing a sharded checkpoint, but got {checkpoint}."
        )

    if index_filename is not None:
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename, "r") as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [
            os.path.join(checkpoint_folder, f) for f in checkpoint_files
        ]

    for checkpoint_file in checkpoint_files:
        checkpoint = load_state_dict(checkpoint_file)
        model.load_state_dict(checkpoint, strict=False)

        # Force Python to clean up.
        del checkpoint
        gc.collect()

    retie_parameters(model, tied_params)

    return model
