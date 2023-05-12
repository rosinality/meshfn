import random
from functools import reduce
from typing import List, Optional, Union
import operator

import numpy as np
import torch
from torch import distributed as dist

from meshfn.distributed.seed import SEEDS
from meshfn.distributed.initializer.data_parallel import DataParallelInitializer
from meshfn.distributed.initializer.model_parallel import ModelParallelInitializer
from meshfn.distributed.initializer.tensor_parallel import TensorParallelInitializer
from meshfn.distributed.initializer.tensor_parallel_1d import (
    TensorParallel1DInitializer,
)
from meshfn.distributed.initializer.pipeline_parallel import PipelineParallelInitializer

from meshfn.distributed.parallel_mode import ParallelMode
from meshfn.logging import logger


class ParallelContext:
    def __init__(
        self,
        rank: int,
        world_size: int,
        host: str,
        port: int,
        local_rank: Optional[int] = None,
        backend: str = "nccl",
        tensor_parallel_size: int = 1,
        tensor_parallel_mode: Optional[Union[str, ParallelMode]] = None,
        pipeline_parallel_size: int = 1,
        seed: int = 1024,
        verbose: bool = False,
    ):
        self.verbose = verbose

        self._global_ranks = {}
        self._local_ranks = {}
        self._world_sizes = {}
        self._groups = {}
        self._cpu_groups = {}
        self._ranks = {}

        self.host = host
        self.port = port
        self.backend = backend

        self.init_global_dist(rank, world_size, host, port, backend)

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = world_size // (
            self.pipeline_parallel_size * self.tensor_parallel_size
        )

        if isinstance(tensor_parallel_mode, str):
            tensor_parallel_mode = ParallelMode[tensor_parallel_mode]

        self.tensor_parallel_mode = tensor_parallel_mode

        self.init_parallel_contexts()

        if torch.cuda.is_available():
            self.set_device(local_rank)

        self.set_seed(seed)

        self._memory_buffer = {}

    def get_buffer(self, shape, dtype, name):
        required_size = reduce(operator.mul, shape, 1)

        if (
            self._memory_buffer.get((name, dtype), None) is None
            or self._memory_buffer[(name, dtype)].numel() < required_size
        ):
            self._memory_buffer[(name, dtype)] = torch.empty(
                required_size,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self._memory_buffer[(name, dtype)][0:required_size].view(*shape)

    def __getitem__(self, parallel_mode: ParallelMode):
        return ParallelGroup(
            parallel_mode,
            self.local_rank(parallel_mode),
            self.global_rank,
            self.world_size(parallel_mode),
            self.group(parallel_mode),
            self.cpu_group(parallel_mode),
            self.ranks_in_group(parallel_mode),
            self._memory_buffer,
        )

    def is_parallel_mode(self, parallel_mode: ParallelMode):
        if not isinstance(parallel_mode, ParallelMode):
            raise TypeError(f"expect ParallelMode, got {type(parallel_mode)}")

    def get_world_size(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._world_sizes[parallel_mode]

    def init_global_dist(
        self, rank: int, world_size: int, host: str, port: int, backend: str
    ):
        init_method = f"tcp://{host}:{port}"

        dist.init_process_group(
            rank=rank, world_size=world_size, backend=backend, init_method=init_method
        )

        ranks = list(range(world_size))
        cpu_group = (
            dist.new_group(ranks, backend="gloo")
            if dist.get_backend() != "gloo"
            else None
        )
        self.register_dist(
            rank,
            world_size,
            dist.GroupMember.WORLD,
            cpu_group,
            ranks,
            ParallelMode.GLOBAL,
        )
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    @property
    def global_rank(self):
        return self._global_ranks[ParallelMode.GLOBAL]

    def next_global_rank(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        local_rank = self.local_rank(parallel_mode)
        world_size = self.world_size(parallel_mode)
        ranks_in_group = self.ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def init_parallel_contexts(self):
        rank = self.global_rank

        init_params = {
            "rank": rank,
            "world_size": self.world_size(ParallelMode.GLOBAL),
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }

        tensor_parallel_initializer = {}

        process_groups = []

        process_groups.append(
            DataParallelInitializer(**init_params).init_process_group()
        )

        process_groups.append(
            ModelParallelInitializer(**init_params).init_process_group()
        )

        tp_group = TensorParallelInitializer(**init_params).init_process_group()

        process_groups.append(tp_group)

        if self.tensor_parallel_mode is not None:
            if self.tensor_parallel_mode == ParallelMode.TENSOR_1D:
                process_groups.append({**tp_group, "mode": ParallelMode.TENSOR_1D})

            else:
                process_groups.append(
                    tensor_parallel_initializer[self.tensor_parallel_mode](
                        **init_params
                    ).init_process_group()
                )

        if self.pipeline_parallel_size > 1:
            process_groups.append(
                PipelineParallelInitializer(**init_params).init_process_group()
            )

        for groups in process_groups:
            if isinstance(groups, (list, tuple)):
                for group in groups:
                    self.register_dist(**group)

            else:
                self.register_dist(**groups)

    def set_device(self, device: Optional[int] = None):
        if device is None:
            n_device = torch.cuda.device_count()
            device = self.global_rank % n_device

        torch.cuda.set_device(device)

        if self.verbose:
            logger.info(f"Set device of process rank {self.global_rank} to {device}")

    def set_seed(self, seed: int, pipeline_mutiplier: int = 1024):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            parallel_seed = seed
            SEEDS.add_seed(ParallelMode.DATA, parallel_seed)

            pipeline_offset = self._local_ranks.get(ParallelMode.PIPELINE, 0)

            if self.is_initialized(ParallelMode.TENSOR):
                tp_rank = self.local_rank(ParallelMode.TENSOR)
                tp_rank_with_offset = tp_rank + pipeline_offset * pipeline_mutiplier
                tp_seed = seed + tp_rank_with_offset
                SEEDS.add_seed(ParallelMode.TENSOR, tp_seed)

            SEEDS.set_mode(ParallelMode.DATA)

            if self.verbose:
                seeds = SEEDS.seeds
                seed_str = ", ".join([f"{k}: {v}" for k, v in seeds.items()])

                logger.info(
                    f"INIT Initialized seed on rank {self.global_rank};"
                    f" python: {seed}, numpy: {seed},"
                    f" {seed_str}; default parallel seed is {ParallelMode.DATA}."
                )

        else:
            if self.verbose:
                logger.info(
                    f"Initialized seed on rank {self.global_rank};"
                    f"python: {seed}, numpy: {seed}, pytorch: {seed}"
                    f" {seed_str}, default parallel seed is {ParallelMode.DATA}."
                )
                logger.warning(
                    "CUDA is not available,"
                    " thus CUDA RNG cannot be used to track CUDA random number states"
                )

    def register_dist(
        self,
        local_rank: int,
        world_size: int,
        process_group: dist.ProcessGroup,
        cpu_group: dist.ProcessGroup,
        ranks: List[int],
        mode: ParallelMode,
    ):
        self.add_local_rank(mode, local_rank)
        self.add_world_size(mode, world_size)
        self.add_group(mode, process_group)
        self.add_cpu_group(mode, cpu_group)
        self.add_ranks(mode, ranks)

    def is_initialized(self, parallel_mode: ParallelMode):
        return parallel_mode in self._groups

    def add_global_rank(self, parallel_mode: ParallelMode, rank: int):
        self.is_parallel_mode(parallel_mode)

        self._global_ranks[parallel_mode] = rank

    def add_local_rank(self, parallel_mode: ParallelMode, local_rank: int):
        self.is_parallel_mode(parallel_mode)

        self._local_ranks[parallel_mode] = local_rank

    def local_rank(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._local_ranks[parallel_mode]

    def add_world_size(self, parallel_mode: ParallelMode, world_size: int):
        self.is_parallel_mode(parallel_mode)

        self._world_sizes[parallel_mode] = world_size

    def world_size(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._world_sizes[parallel_mode]

    def add_group(self, parallel_mode: ParallelMode, process_group: dist.ProcessGroup):
        self.is_parallel_mode(parallel_mode)

        self._groups[parallel_mode] = process_group

    def group(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._groups[parallel_mode]

    def add_cpu_group(self, parallel_mode: ParallelMode, cpu_group: dist.ProcessGroup):
        self.is_parallel_mode(parallel_mode)

        self._cpu_groups[parallel_mode] = cpu_group

    def cpu_group(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._cpu_groups[parallel_mode]

    def device_group(self, parallel_mode: ParallelMode, device: torch.device):
        return (
            self.cpu_group(parallel_mode)
            if device == torch.device("cpu")
            else self.group(parallel_mode)
        )

    def add_ranks(self, parallel_mode: ParallelMode, ranks: List[int]):
        self.is_parallel_mode(parallel_mode)

        self._ranks[parallel_mode] = ranks

    def ranks_in_group(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._ranks[parallel_mode]

    def state_dict(self):
        return {
            "world_size": self.world_size(ParallelMode.GLOBAL),
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "tensor_parallel_mode": str(self.tensor_parallel_mode),
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }

    def __str__(self):
        world_size = self.world_size(ParallelMode.GLOBAL)

        local_ranks = "{\n"

        for k, v in self._local_ranks.items():
            local_ranks += f"        {k}: {v},\n"

        local_ranks += "    }"

        ranks = "{\n"

        for k, v in self._ranks.items():
            ranks += f"        {k}: {v},\n"

        ranks += "    }"

        return f"""{self.__class__.__name__}(
    rank={self.global_rank},
    world_size={world_size},
    host="{self.host}",
    port={self.port}
    backend="{self.backend}",
    data_parallel_size={self.data_parallel_size},
    tensor_parallel_size={self.tensor_parallel_size},
    tensor_parallel_mode={self.tensor_parallel_mode},
    pipeline_parallel_size={self.pipeline_parallel_size},
    local_ranks={local_ranks},
    ranks={ranks},
)"""


class ParallelGroup:
    def __init__(
        self,
        parallel_mode,
        local_rank,
        global_rank,
        world_size,
        group,
        cpu_group,
        ranks_in_group,
        memory_buffer,
    ):
        self.parallel_mode = parallel_mode
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.group = group
        self.cpu_group = cpu_group
        self.ranks_in_group = ranks_in_group
        self._memory_buffer = memory_buffer

    def get_buffer(self, shape, dtype, name):
        required_size = reduce(operator.mul, shape, 1)

        if (
            self._memory_buffer.get((name, dtype), None) is None
            or self._memory_buffer[(name, dtype)].numel() < required_size
        ):
            self._memory_buffer[(name, dtype)] = torch.empty(
                required_size,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self._memory_buffer[(name, dtype)][0:required_size].view(*shape)

    def device_group(self, device: torch.device):
        return self.cpu_group if device == torch.device("cpu") else self.group


class VirtualParallelContext:
    def __init__(
        self,
        rank: int,
        world_size: int,
        tensor_parallel_size: int = 1,
        tensor_parallel_mode: Optional[Union[str, ParallelMode]] = None,
        pipeline_parallel_size: int = 1,
        verbose: bool = False,
    ):
        self.verbose = verbose

        self._global_ranks = {}
        self._local_ranks = {}
        self._world_sizes = {}
        self._groups = {}
        self._cpu_groups = {}
        self._ranks = {}

        self.init_global_dist(rank, world_size)

        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = world_size // (
            self.pipeline_parallel_size * self.tensor_parallel_size
        )

        if isinstance(tensor_parallel_mode, str):
            tensor_parallel_mode = ParallelMode[tensor_parallel_mode]

        self.tensor_parallel_mode = tensor_parallel_mode

        self.init_parallel_contexts()

        self._memory_buffer = {}

    def __getitem__(self, parallel_mode: ParallelMode):
        return ParallelGroup(
            parallel_mode,
            self.local_rank(parallel_mode),
            self.global_rank,
            self.world_size(parallel_mode),
            self.group(parallel_mode),
            self.cpu_group(parallel_mode),
            self.ranks_in_group(parallel_mode),
            self._memory_buffer,
        )

    def is_parallel_mode(self, parallel_mode: ParallelMode):
        if not isinstance(parallel_mode, ParallelMode):
            raise TypeError(f"expect ParallelMode, got {type(parallel_mode)}")

    def get_world_size(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._world_sizes[parallel_mode]

    def init_global_dist(self, rank: int, world_size: int):
        ranks = list(range(world_size))
        self.register_dist(
            rank,
            world_size,
            dist.GroupMember.WORLD,
            None,
            ranks,
            ParallelMode.GLOBAL,
        )
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    @property
    def global_rank(self):
        return self._global_ranks[ParallelMode.GLOBAL]

    def next_global_rank(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        local_rank = self.local_rank(parallel_mode)
        world_size = self.world_size(parallel_mode)
        ranks_in_group = self.ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def init_parallel_contexts(self):
        rank = self.global_rank

        init_params = {
            "rank": rank,
            "world_size": self.world_size(ParallelMode.GLOBAL),
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }

        tensor_parallel_initializer = {}

        process_groups = []

        process_groups.append(
            DataParallelInitializer(**init_params).init_process_group(init_group=False)
        )

        process_groups.append(
            ModelParallelInitializer(**init_params).init_process_group(init_group=False)
        )

        tp_group = TensorParallelInitializer(**init_params).init_process_group(
            init_group=False
        )

        process_groups.append(tp_group)

        if self.tensor_parallel_mode is not None:
            if self.tensor_parallel_mode == ParallelMode.TENSOR_1D:
                process_groups.append({**tp_group, "mode": ParallelMode.TENSOR_1D})

            else:
                process_groups.append(
                    tensor_parallel_initializer[self.tensor_parallel_mode](
                        **init_params
                    ).init_process_group(init_group=False)
                )

        if self.pipeline_parallel_size > 1:
            process_groups.append(
                PipelineParallelInitializer(**init_params).init_process_group(
                    init_group=False
                )
            )

        for groups in process_groups:
            if isinstance(groups, (list, tuple)):
                for group in groups:
                    self.register_dist(**group)

            else:
                self.register_dist(**groups)

    def register_dist(
        self,
        local_rank: int,
        world_size: int,
        process_group: dist.ProcessGroup,
        cpu_group: dist.ProcessGroup,
        ranks: List[int],
        mode: ParallelMode,
    ):
        self.add_local_rank(mode, local_rank)
        self.add_world_size(mode, world_size)
        self.add_group(mode, process_group)
        self.add_cpu_group(mode, cpu_group)
        self.add_ranks(mode, ranks)

    def is_initialized(self, parallel_mode: ParallelMode):
        return parallel_mode in self._groups

    def add_global_rank(self, parallel_mode: ParallelMode, rank: int):
        self.is_parallel_mode(parallel_mode)

        self._global_ranks[parallel_mode] = rank

    def add_local_rank(self, parallel_mode: ParallelMode, local_rank: int):
        self.is_parallel_mode(parallel_mode)

        self._local_ranks[parallel_mode] = local_rank

    def local_rank(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._local_ranks[parallel_mode]

    def add_world_size(self, parallel_mode: ParallelMode, world_size: int):
        self.is_parallel_mode(parallel_mode)

        self._world_sizes[parallel_mode] = world_size

    def world_size(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._world_sizes[parallel_mode]

    def add_group(self, parallel_mode: ParallelMode, process_group: dist.ProcessGroup):
        self.is_parallel_mode(parallel_mode)

        self._groups[parallel_mode] = process_group

    def group(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._groups[parallel_mode]

    def add_cpu_group(self, parallel_mode: ParallelMode, cpu_group: dist.ProcessGroup):
        self.is_parallel_mode(parallel_mode)

        self._cpu_groups[parallel_mode] = cpu_group

    def cpu_group(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._cpu_groups[parallel_mode]

    def device_group(self, parallel_mode: ParallelMode, device: torch.device):
        return (
            self.cpu_group(parallel_mode)
            if device == torch.device("cpu")
            else self.group(parallel_mode)
        )

    def add_ranks(self, parallel_mode: ParallelMode, ranks: List[int]):
        self.is_parallel_mode(parallel_mode)

        self._ranks[parallel_mode] = ranks

    def ranks_in_group(self, parallel_mode: ParallelMode):
        self.is_parallel_mode(parallel_mode)

        return self._ranks[parallel_mode]

    def state_dict(self):
        return {
            "world_size": self.world_size(ParallelMode.GLOBAL),
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "tensor_parallel_mode": str(self.tensor_parallel_mode),
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }

    def __str__(self):
        world_size = self.world_size(ParallelMode.GLOBAL)

        local_ranks = "{\n"

        for k, v in self._local_ranks.items():
            local_ranks += f"        {k}: {v},\n"

        local_ranks += "    }"

        ranks = "{\n"

        for k, v in self._ranks.items():
            ranks += f"        {k}: {v},\n"

        ranks += "    }"

        return f"""{self.__class__.__name__}(
    rank={self.global_rank},
    world_size={world_size},
    data_parallel_size={self.data_parallel_size},
    tensor_parallel_size={self.tensor_parallel_size},
    tensor_parallel_mode={self.tensor_parallel_mode},
    pipeline_parallel_size={self.pipeline_parallel_size},
    local_ranks={local_ranks},
    ranks={ranks},
)"""
