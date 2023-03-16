from contextlib import contextmanager
from functools import wraps
import random
from typing import Tuple

import numpy as np
import torch

from meshfn.distributed.parallel_mode import ParallelMode


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

    def set_mode(self, parallel_mode: ParallelMode):
        if self.current_mode is not None:
            self.seed_states[self.current_mode] = self.get_rng_state()

        self.current_mode = parallel_mode
        self.set_rng_state(self.seed_states[parallel_mode])

    def get_rng_state(self):
        return (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state(),
        )

    def set_rng_state(self, states):
        rand_state, np_state, torch_state, cuda_state = states

        random.setstate(rand_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state(cuda_state)

    def sync_states(self):
        self.set_state(self.current_mode, self.get_rng_state())

    def add_seed(self, parallel_mode: ParallelMode, seed: int, overwrite: bool = False):
        if not isinstance(parallel_mode, ParallelMode):
            raise TypeError(f"expect ParallelMode, got {type(parallel_mode)}")

        if not overwrite and parallel_mode in self.seed_states:
            raise ValueError(f"seed for {parallel_mode} already exists")

        current_state = self.get_rng_state()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.seed_states[parallel_mode] = self.get_rng_state()
        self.seeds[parallel_mode] = seed

        self.set_rng_state(current_state)

    def reset(self):
        self.current_mode = None
        self.seeds = {}
        self.seed_states = {}

    @contextmanager
    def seed(self, parallel_mode: ParallelMode):
        try:
            current_mode = self.current_mode
            yield self.set_mode(parallel_mode)

        finally:
            self.set_mode(current_mode)

    def with_seed(self, parallel_mode: ParallelMode):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                current_mode = self.current_mode
                self.set_mode(parallel_mode)

                out = fn(*args, **kwargs)

                self.set_mode(current_mode)

                return out

            return wrapper

        return decorator


SEEDS = SeedManager()
