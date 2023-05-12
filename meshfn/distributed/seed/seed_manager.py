from contextlib import contextmanager
from functools import wraps
import random
from typing import Tuple

import numpy as np
import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager

from meshfn.distributed.parallel_mode import ParallelMode


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


class SeedManager:
    def __init__(self):
        self.reset()

    def get_states(self, copy=False):
        states = self.seed_states

        if copy:
            new_states = {}

            for k, v in states.items():
                new_states[k] = v.clone()

            return new_states

        return self.seed_states

    def set_state(self, parallel_mode: ParallelMode, state: Tuple):
        if parallel_mode not in self.seed_states:
            raise KeyError(
                f"parallel mode {parallel_mode} not found in the seed manager"
            )

        self.seed_states[parallel_mode] = state

    def set_mode(self, parallel_mode: ParallelMode, cuda_only: bool = True):
        if self.current_mode is not None:
            self.seed_states[self.current_mode] = self.get_rng_state(cuda_only)

        self.current_mode = parallel_mode
        self.set_rng_state(self.seed_states[parallel_mode])

    def get_rng_state(self, cuda_only: bool = True):
        if cuda_only:
            return (torch.cuda.get_rng_state(),)

        return (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state(),
        )

    def set_rng_state(self, states):
        if len(states) == 4:
            rand_state, np_state, torch_state, cuda_state = states

            random.setstate(rand_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            _set_cuda_rng_state(cuda_state)

        else:
            _set_cuda_rng_state(states[0])

    def sync_states(self, cuda_only: bool = True):
        self.set_state(self.current_mode, self.get_rng_state(cuda_only))

    def add_seed(
        self,
        parallel_mode: ParallelMode,
        seed: int,
        overwrite: bool = False,
        cuda_only: bool = True,
    ):
        if not isinstance(parallel_mode, ParallelMode):
            raise TypeError(f"expect ParallelMode, got {type(parallel_mode)}")

        if not overwrite and parallel_mode in self.seed_states:
            raise ValueError(f"seed for {parallel_mode} already exists")

        current_state = self.get_rng_state(cuda_only)

        if not cuda_only:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        torch.cuda.manual_seed(seed)

        self.seed_states[parallel_mode] = self.get_rng_state(cuda_only)
        self.seeds[parallel_mode] = seed

        self.set_rng_state(current_state)

    def reset(self):
        self.current_mode = None
        self.seeds = {}
        self.seed_states = {}

    @contextmanager
    def seed(self, parallel_mode: ParallelMode, cuda_only: bool = True):
        try:
            current_mode = self.current_mode
            yield self.set_mode(parallel_mode, cuda_only)

        finally:
            self.set_mode(current_mode, cuda_only)

    def with_seed(self, parallel_mode: ParallelMode, cuda_only: bool = True):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                current_mode = self.current_mode
                self.set_mode(parallel_mode, cuda_only)

                out = fn(*args, **kwargs)

                self.set_mode(current_mode, cuda_only)

                return out

            return wrapper

        return decorator


SEEDS = SeedManager()


def seed_manager():
    return SEEDS
