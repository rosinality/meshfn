from contextlib import contextmanager
import time
from typing import Any, Callable, Optional, Union

import torch
from torch import nn
from torch.distributed import rpc
from torch import multiprocessing as mp

from meshfn.distributed.launch import launch
from meshfn.distributed import set_context
from meshfn.distributed.parallel_mode import ParallelMode
from meshfn.inference.device import DeviceType, build_device_maps
from meshfn.inference.pipe import RPCPipe
from meshfn.inference.task import TaskEntry, StreamingTaskEntry
from meshfn.inference.threading import Terminator
from meshfn.logging import get_logger


class StreamingWorker:
    def __init__(
        self,
        rank: int,
        tensor_parallel_size: int,
        tensor_parallel_mode: ParallelMode,
        pipeline_parallel_size: int,
        host: str,
        port: int,
        rpc_port: int,
        n_proc_per_node: int,
        model_fn: Callable[[Any], nn.Module],
        pipe_size: int = 1,
        rpc_disable_shm: bool = True,
        device: DeviceType = "cuda",
        **model_kwargs: Any,
    ):
        self.global_rank = rank
        self.world_size = tensor_parallel_size * pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size

        self.context = launch(
            rank,
            self.world_size,
            host,
            port,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_mode=tensor_parallel_mode,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        set_context(self.context)
        self.tensor_parallel_rank = self.context.local_rank(ParallelMode.TENSOR_1D)
        self.pipeline_parallel_rank = (
            self.context.local_rank(ParallelMode.PIPELINE)
            if self.context.is_initialized(ParallelMode.PIPELINE)
            else 0
        )

        self.device = device

        model_kwargs["parallel_context"] = self.context
        self.model: nn.Module = model_fn(**model_kwargs).to(self.device)

        self.rpc_name = f"worker-{self.global_rank}"
        rpc_options = {}

        if rpc_disable_shm:
            rpc_options["_transports"] = ["uv"]

        rpc.init_rpc(
            self.rpc_name,
            rank=self.global_rank + 1,
            world_size=self.world_size + 1,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{host}:{rpc_port}",
                device_maps=build_device_maps(
                    self.world_size, n_proc_per_node, rank=self.global_rank
                ),
            ),
        )

        self.to_master_pipe = RPCPipe(
            f"{self.global_rank}_to_m", self.rpc_name, "master"
        )
        self.to_master_pipe.send(self.pipeline_parallel_rank)

        if self.pipeline_parallel_rank == 0:
            self.input_pipe = RPCPipe(
                f"m_to_{self.global_rank}", "master", self.rpc_name, max_size=pipe_size
            )

        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            self.output_pipe = self.to_master_pipe

        else:
            next_rank = self.context.next_global_rank(ParallelMode.PIPELINE)
            self.output_pipe = RPCPipe(
                f"{self.global_rank}_to_{next_rank}",
                self.rpc_name,
                f"worker-{next_rank}",
                max_size=pipe_size,
            )

        self.logger = get_logger("meshfn-inference")
        self.logger.info(f"INIT starting {self.rpc_name}")
        self.start()

    @contextmanager
    def lifespan(self):
        try:
            yield

        finally:
            self.shutdown()

    @torch.inference_mode()
    def start(self) -> None:
        with self.lifespan():
            while True:
                try:
                    task_entry: TaskEntry = self.input_pipe.recv_nowait()

                    for i, output in enumerate(self.forward(task_entry.batch)):
                        entry = StreamingTaskEntry(
                            task_entry.uids,
                            tuple([i] * len(task_entry.uids)),
                            output,
                            False,
                        )
                        self.output_pipe.send(entry)

                    self.output_pipe.send(
                        StreamingTaskEntry(
                            task_entry.uids,
                            tuple([i + 1] * len(task_entry.uids)),
                            tuple([[None]] * len(task_entry.uids)),
                            True,
                        )
                    )

                except RuntimeError:
                    time.sleep(0.01)

    def shutdown(self) -> None:
        Terminator.shield()
        rpc.rpc_sync("master", Terminator.terminate)
        rpc.shutdown()

    def forward(self, inputs: Any) -> Any:
        if isinstance(inputs, (tuple, list)):
            return self.model(*inputs)

        elif isinstance(inputs, dict):
            return self.model(**inputs)

        return self.model(inputs)


def launch_workers(
    worker,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    host: str,
    port: int,
    rpc_port: int,
    model_fn: Callable[[Any], nn.Module],
    n_proc_per_node: int = 1,
    node_rank: int = 0,
    pipe_size: int = 1,
    rpc_disable_shm: bool = True,
    device: DeviceType = "cuda",
    **model_kwargs: Any,
) -> None:
    ctx = mp.get_context("spawn")
    procs = []

    for i in range(n_proc_per_node):
        rank = n_proc_per_node * node_rank + i
        p = ctx.Process(
            target=worker,
            args=(
                rank,
                tensor_parallel_size,
                ParallelMode.TENSOR_1D,
                pipeline_parallel_size,
                host,
                port,
                rpc_port,
                n_proc_per_node,
                model_fn,
                pipe_size,
                rpc_disable_shm,
                device,
            ),
            kwargs=model_kwargs,
        )
        procs.append(p)
        p.start()
