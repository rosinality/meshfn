import asyncio
from collections import deque, defaultdict
import signal
from threading import Lock, Thread, ThreadError
import time
from typing import Any, Deque, Dict, Hashable, Optional, List, Tuple, Callable

from torch import nn
from torch.distributed import rpc

from meshfn.distributed import get_context
from meshfn.inference.batch_manager import BatchManager, SubmitEntry
from meshfn.inference.device import build_device_maps, DeviceType
from meshfn.inference.pipe import RPCPipe
from meshfn.inference.threading import use_lock, Terminator
from meshfn.inference.lm_worker import launch_workers, StreamingWorker
from meshfn.inference.task import StreamingTaskEntry
from meshfn.logging import get_logger


class QueueFullError(Exception):
    pass


class AsyncInferenceEngine:
    def __init__(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        host: str,
        rpc_port: int,
        n_proc_per_node: int,
        batch_manager: Optional[BatchManager] = None,
        pipe_size: int = 1,
        queue_size: int = 0,
        rpc_disable_shm: bool = True,
    ) -> None:
        self.lock = Lock()
        self.logger = get_logger()

        if batch_manager is None:
            batch_manager = BatchManager()

        self.batch_manager = batch_manager

        self.world_size = tensor_parallel_size * pipeline_parallel_size

        rpc_options = {}
        if rpc_disable_shm:
            rpc_options["_transport"] = ["uv"]

        rpc.init_rpc(
            "master",
            rank=0,
            world_size=self.world_size + 1,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{host}:{rpc_port}",
                device_maps=build_device_maps(self.world_size, n_proc_per_node),
            ),
        )

        self.from_worker_pipes: List[RPCPipe] = []
        for i in range(self.world_size):
            pipe = RPCPipe(f"{i}_to_m", f"worker-{i}", "master")
            self.from_worker_pipes.append(pipe)

        self.submit_pipes: List[RPCPipe] = []
        self.completion_pipes: List[RPCPipe] = []
        for i, pipe in enumerate(self.from_worker_pipes):
            worker_pipe_rank = pipe.recv()

            if worker_pipe_rank == 0:
                self.submit_pipes.append(
                    RPCPipe(f"m_to_{i}", "master", f"worker-{i}", max_size=pipe_size)
                )

            if worker_pipe_rank == pipeline_parallel_size - 1:
                self.completion_pipes.append(pipe)

        self.running: bool = False
        self.submit_thread = None
        self.completion_thread = None
        self.queue_size = queue_size
        self.submit_queue: Deque[SubmitEntry] = deque()
        self.batch_info: Dict[Hashable, Any] = {}
        self.timer_info: Dict[Hashable, Tuple[int, float]] = {}
        self.completion_map: Dict[Hashable, Dict[int, Any]] = defaultdict(dict)
        self.uid_index_track: Dict[Hashable, int] = defaultdict(lambda: -1)

        self.logger.info("INIT starting engine")
        self.start()
        self.register_sigint()

    def submit_loop(self) -> None:
        while self.running:
            if len(self.submit_queue) > 0:
                task_entry, batch_info = self.batch_manager.make_batch(
                    self.submit_queue
                )
                self.batch_info[task_entry.uids] = batch_info
                self.timer_info[task_entry.uids] = (
                    len(task_entry.uids),
                    time.perf_counter(),
                )

                for pipe in self.submit_pipes:
                    pipe.send(task_entry)

            else:
                time.sleep(0.01)

    def completion_loop(self) -> None:
        received_data: Dict[int, Any] = {}

        while self.running:
            for i, pipe in enumerate(self.completion_pipes):
                if i not in received_data:
                    try:
                        received_data[i] = pipe.recv_nowait()

                    except RuntimeError:
                        pass

            if len(received_data) == len(self.completion_pipes):
                task_entries: List[StreamingTaskEntry] = list(
                    map(lambda k: received_data[k], sorted(received_data.keys()))
                )
                received_data.clear()

                batch_info = self.batch_info[task_entries[0].uids]

                if task_entries[0].finished:
                    self.batch_info.pop(task_entries[0].uids)

                for uid, index, output in self.batch_manager.split_batch(
                    task_entries[0], **batch_info
                ):
                    self.completion_map[uid][index] = output

                if task_entries[0].finished:
                    batch_size, start_time = self.timer_info[task_entries[0].uids]
                    self.logger.info(
                        f"batch size: {batch_size}; time: {time.perf_counter() - start_time:.3f}"
                    )

            else:
                time.sleep(0.01)

    def submit(self, uid: Hashable, data: Any) -> None:
        if not self.submit_thread.is_alive():
            raise ThreadError(f"submit thread is not alive")

        if uid in self.completion_map:
            raise ValueError(f"uid {uid} found in completion map")

        if self.queue_size > 0 and len(self.submit_queue) >= self.queue_size:
            raise QueueFullError(f"submit queue is full; size: {self.queue_size}")

        self.submit_queue.append(SubmitEntry(uid, data))

    async def wait(self, uid: Hashable, interval: float = 0.01) -> Any:
        if not self.completion_thread.is_alive():
            raise ThreadError(f"completion thread is not alive")

        while True:
            if uid in self.completion_map:
                outputs = self.completion_map[uid]

                if len(outputs) < 1:
                    await asyncio.sleep(interval)

                    continue

                min_id = min(outputs.keys())

                if self.uid_index_track[uid] + 1 == min_id:
                    output = self.completion_map[uid][min_id]
                    self.uid_index_track[uid] += 1

                    del self.completion_map[uid][min_id]

                    if output is None:
                        del self.completion_map[uid]
                        del self.uid_index_track[uid]

                    return output

            await asyncio.sleep(interval)

    def get(self, uid: Hashable, interval: float = 0.01) -> Any:
        if not self.completion_thread.is_alive():
            raise ThreadError(f"completion thread is not alive")

        while True:
            if uid in self.completion_map:
                outputs = self.completion_map[uid]
                min_id = min(outputs.keys())

                if self.uid_index_track[uid] + 1 == min_id:
                    output = self.completion_map[uid][min_id]

                    if output is None:
                        del self.completion_map[uid]
                        del self.uid_index_track[uid]

                return output

            time.sleep(interval)

    def start(self) -> None:
        self.running = True
        self.submit_thread = Thread(target=self.submit_loop)
        self.submit_thread.start()
        self.completion_thread = Thread(target=self.completion_loop)
        self.completion_thread.start()

    def shutdown(self) -> None:
        with use_lock(self.lock):
            if not self.running:
                return

            self.running = False

        Terminator.shield()

        for i in range(self.world_size):
            rpc.rpc_sync(f"worker-{i}", Terminator.terminate)

        rpc.shutdown()

        self.submit_thread.join()
        self.completion_thread.join()

    def sigint_handler(self, *_):
        self.shutdown()

        raise KeyboardInterrupt

    def register_sigint(self):
        signal.signal(signal.SIGINT, self.sigint_handler)


def launch_engine(
    tp_world_size: int,
    pp_world_size: int,
    master_host: str,
    master_port: int,
    rpc_port: int,
    model_fn: Callable[[Any], nn.Module],
    n_nodes: int = 1,
    node_rank: int = 0,
    batch_manager: Optional[BatchManager] = None,
    pipe_size: int = 1,
    queue_size: int = 0,
    rpc_disable_shm: bool = True,
    device: DeviceType = "cuda",
    **model_kwargs: Any,
) -> Optional[AsyncInferenceEngine]:
    world_size = tp_world_size * pp_world_size
    assert world_size % n_nodes == 0
    n_proc_per_node = world_size // n_nodes

    launch_workers(
        StreamingWorker,
        tp_world_size,
        pp_world_size,
        master_host,
        master_port,
        rpc_port,
        model_fn,
        n_proc_per_node=n_proc_per_node,
        node_rank=node_rank,
        pipe_size=pipe_size,
        rpc_disable_shm=rpc_disable_shm,
        device=device,
        **model_kwargs,
    )

    if node_rank == 0:
        engine = AsyncInferenceEngine(
            tp_world_size,
            pp_world_size,
            master_host,
            rpc_port,
            n_proc_per_node,
            batch_manager=batch_manager,
            pipe_size=pipe_size,
            queue_size=queue_size,
            rpc_disable_shm=rpc_disable_shm,
        )

        return engine
